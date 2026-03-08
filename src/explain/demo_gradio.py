import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import SimpleDRCNN
from transforms import get_transforms


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

METRICS_MD = """
### Model snapshot
- **Accuracy:** 62.1%
- **Macro-F1:** 41.6%
- **Weighted-F1:** 63.3%

Good at common classes, weaker on rare severe cases.
"""


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255).astype(np.uint8)


def resize01(x01: np.ndarray, h: int, w: int, mode=Image.BILINEAR) -> np.ndarray:
    img = Image.fromarray(to_uint8(x01))
    img = img.resize((w, h), resample=mode)
    return np.asarray(img).astype(np.float32) / 255.0


def make_overlay(gray01: np.ndarray, heat01: np.ndarray, alpha=0.45) -> np.ndarray:
    h, w = gray01.shape
    heat_up = resize01(heat01, h, w, mode=Image.BILINEAR)
    heat_rgb = plt.get_cmap("magma")(heat_up)[..., :3]
    base = np.stack([gray01, gray01, gray01], axis=-1)
    out = np.clip((1 - alpha) * base + alpha * heat_rgb, 0.0, 1.0)
    return to_uint8(out)


def plot_strengths(strength: np.ndarray, selected: int, layer_name: str):
    fig, ax = plt.subplots(figsize=(10.5, 3.0), dpi=120)
    colors = ["#4C78A8"] * len(strength)
    colors[selected] = "#54A24B"
    ax.bar(np.arange(len(strength)), strength, color=colors)
    ax.set_title(f"{layer_name.upper()} channel strengths", fontsize=10)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean |activation|")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kernel_grid(kernels: np.ndarray, channel_ids, title: str):
    n = len(channel_ids)
    cols = 2
    rows = int(np.ceil(n / cols))

    # Square figure for square-looking panel
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 7.0), dpi=120)
    axes = np.array(axes).reshape(rows, cols)

    vmax = float(np.max(np.abs(kernels))) + 1e-8

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue

        k = kernels[i]
        ch = channel_ids[i]
        ax.imshow(k, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Channel {ch}", fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

        for r in range(3):
            for c in range(3):
                v = float(k[r, c])
                txt_color = "white" if abs(v) > 0.45 * vmax else "black"
                ax.text(
                    c,
                    r,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                    fontweight="bold",
                )

    fig.suptitle(title, fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


def plot_class_probs(probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=120)
    labels = [f"C{i}\n{name}" for i, name in enumerate(CLASS_NAMES)]
    pred_idx = int(np.argmax(probs))
    colors = ["#4C78A8"] * len(probs)
    colors[pred_idx] = "#F58518"

    ax.bar(np.arange(len(probs)), probs * 100.0, color=colors)
    ax.set_xticks(np.arange(len(probs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Class probabilities")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, p in enumerate(probs):
        ax.text(i, p * 100.0 + 1.2, f"{p * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


class Explainer:
    def __init__(self, ckpt_path: str):
        ckpt_path = str(Path(ckpt_path).expanduser())
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found:\n  {ckpt_path}\n\n"
                "If Colab runtime reset, /content is wiped.\n"
                "Use your Drive checkpoint path, e.g.:\n"
                "  --ckpt /content/drive/MyDrive/dr-demo-cnn-checkpoints/best.pt"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.image_size = int(ckpt.get("image_size", 224))
        self.model = SimpleDRCNN(num_classes=ckpt.get("num_classes", 5)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tf = get_transforms(image_size=self.image_size, train=False)
        self.activations = {}

        self.available_layers = []
        for i in [1, 2, 3]:
            c, r, p = f"conv{i}", f"relu{i}", f"pool{i}"

            if hasattr(self.model, c):
                getattr(self.model, c).register_forward_hook(self._hook(c))
                self.available_layers.append(c)
            if hasattr(self.model, r):
                getattr(self.model, r).register_forward_hook(self._hook(r))
            if hasattr(self.model, p):
                getattr(self.model, p).register_forward_hook(self._hook(p))

        if not self.available_layers:
            raise RuntimeError("No supported conv layers found (expected conv1/conv2/conv3).")

    def _hook(self, name):
        def fn(_, __, out):
            self.activations[name] = out.detach().cpu()
        return fn

    def get_layer_max_channel(self, layer_name: str) -> int:
        mod = getattr(self.model, layer_name, None)
        return int(mod.out_channels - 1) if mod is not None else 0

    def _require_activation(self, key: str):
        if key not in self.activations:
            raise RuntimeError(f"Activation '{key}' was not captured during forward pass.")

    def run(self, pil_img, layer_name="conv1", auto_select=True, manual_idx=0):
        if pil_img is None:
            return (None, None, None, None, None, None, None, None, gr.update())

        if layer_name not in self.available_layers:
            layer_name = self.available_layers[0]

        relu_name = layer_name.replace("conv", "relu")
        pool_name = layer_name.replace("conv", "pool")

        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        self._require_activation(layer_name)
        self._require_activation(relu_name)
        self._require_activation(pool_name)

        inp = x[0, 0].detach().cpu().numpy()
        inp01 = np.clip(inp * 0.5 + 0.5, 0.0, 1.0)

        conv = self.activations[layer_name][0].numpy()
        relu = self.activations[relu_name][0].numpy()
        pool = self.activations[pool_name][0].numpy()

        strength = np.mean(np.abs(conv), axis=(1, 2))
        strongest_idx = int(np.argmax(strength))
        fmap_idx = strongest_idx if auto_select else int(np.clip(int(manual_idx), 0, conv.shape[0] - 1))

        conv01 = norm01(conv[fmap_idx])
        relu01 = norm01(relu[fmap_idx])
        pool01 = norm01(pool[fmap_idx])

        conv_up = resize01(conv01, 224, 224, mode=Image.NEAREST)
        relu_up = resize01(relu01, 224, 224, mode=Image.NEAREST)
        pool_up = resize01(pool01, 224, 224, mode=Image.NEAREST)
        overlay = make_overlay(inp01, conv01, alpha=0.45)

        sorted_idx = np.argsort(-strength).tolist()
        top_channels = [fmap_idx] + [c for c in sorted_idx if c != fmap_idx][:3]

        conv_module = getattr(self.model, layer_name)
        w = conv_module.weight.detach().cpu().numpy()
        kernels = [w[c].mean(axis=0) for c in top_channels]

        kernel_fig = plot_kernel_grid(
            np.array(kernels),
            top_channels,
            title=f"Kernel view ({layer_name.upper()}): selected + top channels",
        )
        strength_fig = plot_strengths(strength, fmap_idx, layer_name)
        class_fig = plot_class_probs(probs)

        slider_update = gr.update(value=int(fmap_idx))

        return (
            to_uint8(inp01),
            to_uint8(conv_up),
            to_uint8(relu_up),
            to_uint8(pool_up),
            overlay,
            kernel_fig,
            strength_fig,
            class_fig,
            slider_update,
        )


def build_app(ckpt_path: str):
    explainer = Explainer(ckpt_path)

    def infer(img, layer_name, auto_select, manual_idx):
        return explainer.run(img, layer_name, auto_select, manual_idx)

    def on_auto_change(auto_select):
        return gr.update(interactive=not auto_select)

    def on_layer_change(layer_name, current_idx):
        max_idx = explainer.get_layer_max_channel(layer_name)
        safe_idx = int(np.clip(int(current_idx), 0, max_idx))
        return gr.update(minimum=0, maximum=max_idx, value=safe_idx)

    default_layer = explainer.available_layers[0]
    default_max = explainer.get_layer_max_channel(default_layer)

    with gr.Blocks(title="DR CNN Layer Explorer") as demo:
        # Cross-version-safe CSS injection
        gr.HTML(
            """
            <style>
            #run-explain-btn {
                background: #f97316 !important;
                border-color: #f97316 !important;
                color: white !important;
            }
            #run-explain-btn:hover {
                background: #ea580c !important;
                border-color: #ea580c !important;
            }
            </style>
            """
        )

        gr.Markdown("## DR CNN Layer Explorer (Classroom Demo)")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=360):
                inp = gr.Image(type="pil", label="Fundus Image", height=320)
                layer_select = gr.Dropdown(
                    choices=explainer.available_layers,
                    value=default_layer,
                    label="Layer",
                )
                auto_select = gr.Checkbox(value=True, label="Auto-select strongest channel")
                manual_idx = gr.Slider(
                    minimum=0,
                    maximum=default_max,
                    step=1,
                    value=0,
                    label="Manual channel index",
                    interactive=False,
                )
                run_btn = gr.Button("Run Explain", elem_id="run-explain-btn")

            with gr.Column(scale=5, min_width=360):
                gr.Markdown(METRICS_MD)
                out_class_plot = gr.Plot(label="Class probabilities (C0-C4)")

        # Display Row 1
        with gr.Row(equal_height=True):
            out_input = gr.Image(label="Input to CNN (single channel)", height=210)
            out_conv = gr.Image(label="Conv output", height=210)
            out_relu = gr.Image(label="ReLU output", height=210)
            out_pool = gr.Image(label="Pool output", height=210)

        # Display Row 2: overlay + kernel
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=420):
                out_overlay = gr.Image(
                    label="Overlay: input + conv heat",
                    height=620,
                    width=620,
                )
            with gr.Column(scale=1, min_width=420):
                out_kernel = gr.Plot(label="Kernel view")

        # Display Row 3
        with gr.Row():
            out_strength = gr.Plot(label="Channel strengths")

        gr.ClearButton(
            components=[
                inp,
                out_input,
                out_conv,
                out_relu,
                out_pool,
                out_overlay,
                out_kernel,
                out_strength,
                out_class_plot,
            ],
            value="Clear",
        )

        auto_select.change(on_auto_change, inputs=[auto_select], outputs=[manual_idx])
        layer_select.change(on_layer_change, inputs=[layer_select, manual_idx], outputs=[manual_idx])

        run_btn.click(
            infer,
            inputs=[inp, layer_select, auto_select, manual_idx],
            outputs=[
                out_input,
                out_conv,
                out_relu,
                out_pool,
                out_overlay,
                out_kernel,
                out_strength,
                out_class_plot,
                manual_idx,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_app(args.ckpt)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()