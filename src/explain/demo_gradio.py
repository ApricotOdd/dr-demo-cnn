import argparse
import inspect
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
**Validation Summary (Best Checkpoint)**  
- Accuracy: **0.6207**  
- Macro-F1: **0.4161**  
- Weighted-F1: **0.6327**  
- Samples: **733**

**Takeaway:** strong on class 0/2, weaker on severe minority classes (3/4).
"""

RETRO_CSS = r"""
:root {
  --bg: #c9c9c9;
  --panel: #dcdcdc;
  --panel-2: #efefef;
  --border-dark: #7a7a7a;
  --border-mid: #a2a2a2;
  --border-light: #ffffff;
  --title-blue: #123f7a;
  --title-blue-dark: #0f3363;
  --title-text: #ffffff;
  --text: #111111;
  --muted: #333333;
  --btn: #dcdcdc;
  --btn-dark: #808080;
  --btn-light: #ffffff;
}

html, body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: Tahoma, Verdana, Arial, sans-serif !important;
  color-scheme: light !important;

  --body-background-fill: var(--bg) !important;
  --body-background-fill-dark: var(--bg) !important;
  --background-fill-primary: var(--panel) !important;
  --background-fill-primary-dark: var(--panel) !important;
  --background-fill-secondary: var(--panel-2) !important;
  --background-fill-secondary-dark: var(--panel-2) !important;

  --block-background-fill: var(--panel) !important;
  --block-background-fill-dark: var(--panel) !important;
  --block-border-color: var(--border-dark) !important;
  --block-border-color-dark: var(--border-dark) !important;

  --input-background-fill: #ffffff !important;
  --input-background-fill-dark: #ffffff !important;
  --input-border-color: #8b8b8b !important;
  --input-border-color-dark: #8b8b8b !important;

  --button-primary-background-fill: var(--btn) !important;
  --button-primary-background-fill-dark: var(--btn) !important;
  --button-primary-text-color: #111 !important;
  --button-primary-text-color-dark: #111 !important;
  --button-primary-border-color: var(--btn-dark) !important;
  --button-primary-border-color-dark: var(--btn-dark) !important;
}

.dark, .dark * {
  color-scheme: light !important;
}

#app-root {
  max-width: 1680px;
  margin: 0 auto;
  padding: 8px 6px;
}

.window {
  border: 1px solid var(--border-dark) !important;
  background: var(--panel) !important;
  box-shadow:
    inset 1px 1px 0 var(--border-light),
    inset -1px -1px 0 var(--border-mid) !important;
  padding: 0 !important;
  margin-bottom: 10px !important;
}

.titlebar {
  width: 100%;
  box-sizing: border-box;
  background: linear-gradient(to bottom, var(--title-blue), var(--title-blue-dark));
  color: var(--title-text);
  font-weight: bold;
  font-size: 13px;
  letter-spacing: 0.1px;
  padding: 7px 9px;
  border-bottom: 1px solid #0b2a50;
}

.window-body {
  padding: 10px;
}

.small-note {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}

.gradio-container .block,
.gradio-container [data-testid="block"],
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-panel {
  background: var(--panel) !important;
  color: #111 !important;
  border-color: var(--border-dark) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  border-radius: 0 !important;
  border: 1px solid #8b8b8b !important;
  background: #ffffff !important;
  color: #111 !important;
  box-shadow: inset 1px 1px 0 #ffffff !important;
}

.gradio-container [data-testid="dropdown"] *,
.gradio-container [data-testid="textbox"] *,
.gradio-container [data-testid="number"] * {
  color: #111 !important;
}

.gradio-container input[type="checkbox"] {
  accent-color: #123f7a !important;
  width: 16px !important;
  height: 16px !important;
  cursor: pointer !important;
}

.gradio-container button {
  border: 1px solid var(--btn-dark) !important;
  background: var(--btn) !important;
  color: #111 !important;
  border-radius: 0 !important;
  box-shadow:
    inset 1px 1px 0 var(--btn-light),
    inset -1px -1px 0 #9c9c9c !important;
  font-weight: 600 !important;
}

.gradio-container button:hover { filter: brightness(0.98); }

.gradio-container button:active {
  box-shadow:
    inset -1px -1px 0 var(--btn-light),
    inset 1px 1px 0 #9c9c9c !important;
  transform: translateY(1px);
}

.gradio-container [data-testid="image"],
.gradio-container .image-container,
.gradio-container .gr-image {
  background: #f3f3f3 !important;
  border-radius: 0 !important;
  color: #111 !important;
}

.gradio-container .prose,
.gradio-container .gr-markdown,
.gradio-container label,
.gradio-container .gr-form,
.gradio-container p,
.gradio-container span,
.gradio-container div {
  color: #111 !important;
}
"""


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
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
    heat_rgb = plt.get_cmap("jet")(heat_up)[..., :3]
    base = np.stack([gray01, gray01, gray01], axis=-1)
    out = np.clip((1 - alpha) * base + alpha * heat_rgb, 0.0, 1.0)
    return to_uint8(out)


def plot_strengths(strength: np.ndarray, selected: int, layer_name: str):
    fig, ax = plt.subplots(figsize=(7.2, 2.7), dpi=120)
    colors = ["#1f73be"] * len(strength)
    colors[selected] = "#2ca24f"
    ax.bar(np.arange(len(strength)), strength, color=colors)
    ax.set_title(f"{layer_name.upper()} Channel Activation Strength (mean |activation|)", fontsize=10)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Strength")
    ax.set_xticks(np.arange(len(strength)))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_kernel_grid(kernels: np.ndarray, channel_ids, title):
    n = len(channel_ids)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4, 3.1 * rows), dpi=120)
    axes = np.array(axes).reshape(rows, cols)

    vmax = float(np.max(np.abs(kernels))) + 1e-8
    im = None
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue

        k = kernels[i]
        ch = channel_ids[i]
        im = ax.imshow(k, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Channel {ch}", fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

        for r in range(3):
            for c in range(3):
                v = float(k[r, c])
                txt_color = "white" if abs(v) > 0.45 * vmax else "black"
                ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color=txt_color, fontweight="bold")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
        cbar.ax.set_ylabel("Weight", rotation=90)

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout()
    return fig


def probs_to_markdown(probs: np.ndarray) -> str:
    order = np.argsort(-probs)
    lines = ["| Class | Probability |", "|---|---:|"]
    for i in order:
        lines.append(f"| {CLASS_NAMES[i]} | {probs[i]*100:.2f}% |")
    return "\n".join(lines)


class Explainer:
    def __init__(self, ckpt_path: str):
        ckpt_path = str(Path(ckpt_path).expanduser())
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found:\n  {ckpt_path}\n\n"
                "Tip: in Colab runtime reset, /content files disappear.\n"
                "Use your Drive path, e.g.:\n"
                "  --ckpt /content/drive/MyDrive/dr-demo-cnn-checkpoints/best.pt"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.image_size = ckpt.get("image_size", 224)
        self.model = SimpleDRCNN(num_classes=ckpt.get("num_classes", 5)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tf = get_transforms(image_size=self.image_size, train=False)
        self.activations = {}

        self.available_layers = []
        for i in [1, 2, 3]:
            conv_name = f"conv{i}"
            relu_name = f"relu{i}"
            pool_name = f"pool{i}"

            if hasattr(self.model, conv_name):
                getattr(self.model, conv_name).register_forward_hook(self._hook(conv_name))
            if hasattr(self.model, relu_name):
                getattr(self.model, relu_name).register_forward_hook(self._hook(relu_name))
            if hasattr(self.model, pool_name):
                getattr(self.model, pool_name).register_forward_hook(self._hook(pool_name))

            if hasattr(self.model, conv_name) and hasattr(self.model, relu_name) and hasattr(self.model, pool_name):
                self.available_layers.append(conv_name)

        if not self.available_layers:
            self.available_layers = ["conv1"]

    def _hook(self, name):
        def fn(_, __, out):
            self.activations[name] = out.detach().cpu()
        return fn

    def get_layer_max_channel(self, layer_name: str) -> int:
        module = getattr(self.model, layer_name, None)
        if module is None:
            return 0
        return int(module.out_channels - 1)

    def run(self, pil_img, layer_name="conv1", auto_select=True, manual_idx=0):
        if pil_img is None:
            return (
                None, None, None, None, None, None, None, "",
                "No image loaded.",
                "Upload a fundus image, then click Run Explain."
            )

        if layer_name not in self.available_layers:
            layer_name = self.available_layers[0]

        relu_name = layer_name.replace("conv", "relu")
        pool_name = layer_name.replace("conv", "pool")

        x = self.tf(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        inp = x[0, 0].detach().cpu().numpy()
        inp01 = np.clip(inp * 0.5 + 0.5, 0.0, 1.0)

        conv = self.activations[layer_name][0].numpy()
        relu = self.activations[relu_name][0].numpy()
        pool = self.activations[pool_name][0].numpy()

        strength = np.mean(np.abs(conv), axis=(1, 2))
        best_idx = int(np.argmax(strength))
        fmap_idx = best_idx if auto_select else (int(manual_idx) % conv.shape[0])

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
            title=f"Kernel Inspector ({layer_name.upper()}): selected + top-3 channels"
        )
        strength_fig = plot_strengths(strength, fmap_idx, layer_name)

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        relu_zero_frac = float(np.mean(relu[fmap_idx] <= 0))
        score = float(strength[fmap_idx])

        summary = (
            f"**Layer:** `{layer_name}`  \n"
            f"**Selected channel/kernel:** `{fmap_idx}` ({'auto strongest' if auto_select else 'manual override'})  \n"
            f"**Selection score (mean |conv activation|):** `{score:.4f}`  \n"
            f"**Prediction:** `{pred_name}` with confidence `{pred_conf:.2%}`"
        )

        explain = (
            "### Explain Panel\n"
            f"- Selected feature map from **{layer_name}**, channel **{fmap_idx}**.\n"
            f"- ReLU keeps positive evidence (zeroed fraction: **{relu_zero_frac:.1%}**).\n"
            "- MaxPool keeps strongest local evidence and downsamples.\n"
            "- Overlay aligns selected conv activation with retinal structure.\n"
            "- Kernel inspector shows learned \(3 \\times 3\) values "
            "(for deeper convs: averaged across input channels)."
        )

        probs_md = probs_to_markdown(probs)

        return (
            to_uint8(inp01),
            to_uint8(conv_up),
            to_uint8(relu_up),
            to_uint8(pool_up),
            overlay,
            kernel_fig,
            strength_fig,
            probs_md,
            summary,
            explain,
        )


def build_app(ckpt_path: str, inject_css_fallback: bool = False):
    explainer = Explainer(ckpt_path)

    def infer(img, layer_name, auto_select, manual_idx):
        return explainer.run(img, layer_name, auto_select, manual_idx)

    def on_auto_change(auto_select):
        return gr.update(interactive=not auto_select)

    def on_layer_change(layer_name, current_idx):
        max_idx = explainer.get_layer_max_channel(layer_name)
        safe_idx = int(min(max(0, int(current_idx)), max_idx))
        return gr.update(minimum=0, maximum=max_idx, value=safe_idx)

    default_layer = explainer.available_layers[0]
    default_max = explainer.get_layer_max_channel(default_layer)

    with gr.Blocks(title="DR CNN Retro Explain Console") as demo:
        # Fallback for very old Gradio versions that don't accept css in launch()
        if inject_css_fallback:
            gr.HTML(f"<style>{RETRO_CSS}</style>")

        with gr.Column(elem_id="app-root"):
            with gr.Group(elem_classes=["window"]):
                gr.HTML('<div class="titlebar">Diabetic Retinopathy CNN — Retro Explain Console</div>')
                with gr.Column(elem_classes=["window-body"]):
                    gr.Markdown(
                        "<p class='small-note'>Classic panel mode: compact controls, clear borders, "
                        "and teaching-first explainability.</p>"
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=380):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Controls</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            inp = gr.Image(type="pil", label="Fundus Image", height=320)

                            layer_select = gr.Dropdown(
                                choices=explainer.available_layers,
                                value=default_layer,
                                label="Convolution Block to Inspect",
                            )

                            auto_select = gr.Checkbox(
                                value=True,
                                label="Auto-select strongest channel/kernel",
                            )

                            manual_idx = gr.Slider(
                                minimum=0,
                                maximum=default_max,
                                step=1,
                                value=0,
                                label="Manual channel index (used only when auto-select is OFF)",
                                interactive=False,
                            )

                            run_btn = gr.Button("Run Explain")

                with gr.Column(scale=5, min_width=380):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Performance + Prediction</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            gr.Markdown(METRICS_MD)
                            out_probs = gr.Markdown(label="Dense + Softmax Output")
                            out_summary = gr.Markdown()
                            out_explain = gr.Markdown()

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Input (CNN Gray)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_input = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Conv Map (Selected)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_conv = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">ReLU Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_relu = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=200):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">MaxPool Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_pool = gr.Image(height=170, show_label=False)

            with gr.Row(equal_height=True):
                with gr.Column(scale=4, min_width=340):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Overlay (Input + Selected Conv Heat)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_overlay = gr.Image(height=260, show_label=False)

                with gr.Column(scale=6, min_width=460):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Kernel Inspector + Channel Strengths</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_kernel = gr.Plot()
                            out_strength = gr.Plot()

            gr.ClearButton(
                components=[
                    inp, out_input, out_conv, out_relu, out_pool,
                    out_overlay, out_kernel, out_strength,
                    out_probs, out_summary, out_explain,
                ],
                value="Clear",
            )

        auto_select.change(on_auto_change, inputs=[auto_select], outputs=[manual_idx])
        layer_select.change(on_layer_change, inputs=[layer_select, manual_idx], outputs=[manual_idx])

        run_btn.click(
            infer,
            inputs=[inp, layer_select, auto_select, manual_idx],
            outputs=[
                out_input, out_conv, out_relu, out_pool, out_overlay,
                out_kernel, out_strength, out_probs, out_summary, out_explain,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    launch_sig = inspect.signature(gr.Blocks.launch).parameters
    supports_theme_in_launch = "theme" in launch_sig
    supports_css_in_launch = "css" in launch_sig

    demo = build_app(args.ckpt, inject_css_fallback=not supports_css_in_launch)

    launch_kwargs = {
        "server_port": args.port,
        "share": args.share,
    }
    if supports_theme_in_launch:
        launch_kwargs["theme"] = gr.themes.Base()
    if supports_css_in_launch:
        launch_kwargs["css"] = RETRO_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()