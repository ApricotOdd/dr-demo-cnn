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

APP_CSS = r"""
:root {
  --bg: #0b1220;
  --panel: #111a2b;
  --panel-2: #17233a;
  --muted: #8ea2c8;
  --text: #e6edf8;
  --line: #223454;
  --accent: #4f7cff;
  --accent-2: #36c2a1;
}

html, body, .gradio-container {
  background: radial-gradient(circle at top left, #101a2e 0%, #0b1220 45%, #090f1b 100%) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

#app-root {
  max-width: 1640px;
  margin: 0 auto;
  padding: 12px 10px 18px 10px;
}

.card {
  background: linear-gradient(180deg, var(--panel) 0%, #0f1828 100%) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
  overflow: hidden;
  margin-bottom: 10px !important;
}

.card-header {
  padding: 10px 12px;
  background: linear-gradient(90deg, #162642 0%, #11203a 100%);
  border-bottom: 1px solid var(--line);
  color: #dfe8fb;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.2px;
}

.card-body {
  padding: 12px;
}

.subtle {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.gradio-container .block,
.gradio-container [data-testid="block"],
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-panel {
  background: transparent !important;
  border: none !important;
  color: var(--text) !important;
}

.gradio-container label,
.gradio-container .gr-markdown,
.gradio-container .prose,
.gradio-container p,
.gradio-container span,
.gradio-container div {
  color: var(--text) !important;
}

.gradio-container .prose table {
  color: var(--text) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border: 1px solid #2a3f67 !important;
  border-radius: 10px !important;
}

.gradio-container input[type="checkbox"] {
  accent-color: var(--accent) !important;
}

.gradio-container button {
  background: linear-gradient(180deg, #2c4d9b 0%, #243f82 100%) !important;
  border: 1px solid #4063b3 !important;
  color: #f1f6ff !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
}
.gradio-container button:hover {
  filter: brightness(1.06);
}

.gradio-container [data-testid="image"],
.gradio-container .image-container,
.gradio-container .gr-image {
  background: #0e1728 !important;
  border: 1px solid #24395f !important;
  border-radius: 10px !important;
}

hr {
  border-color: #25395d !important;
}

.kpi {
  display: inline-block;
  padding: 4px 8px;
  border: 1px solid #2b4168;
  border-radius: 999px;
  color: #cfe0ff;
  background: #12213a;
  font-size: 11px;
  margin-right: 6px;
}
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
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=120)
    colors = ["#5f85ff"] * len(strength)
    colors[selected] = "#35c7a5"
    ax.bar(np.arange(len(strength)), strength, color=colors)
    ax.set_title(f"{layer_name.upper()} channel activation strength", fontsize=10)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean abs activation")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_kernel_grid(kernels: np.ndarray, channel_ids, title: str):
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
        lines.append(f"| {CLASS_NAMES[i]} | {probs[i] * 100:.2f}% |")
    return "\n".join(lines)


class Explainer:
    def __init__(self, ckpt_path: str):
        ckpt_path = str(Path(ckpt_path).expanduser())
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found:\n  {ckpt_path}\n\n"
                "If Colab runtime reset, /content gets wiped.\n"
                "Use Drive path, for example:\n"
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

        # Hook expected blocks conv1/relu1/pool1 ... conv3/relu3/pool3 if present
        self.available_layers = []
        for i in [1, 2, 3]:
            c, r, p = f"conv{i}", f"relu{i}", f"pool{i}"
            if hasattr(self.model, c):
                getattr(self.model, c).register_forward_hook(self._hook(c))
            if hasattr(self.model, r):
                getattr(self.model, r).register_forward_hook(self._hook(r))
            if hasattr(self.model, p):
                getattr(self.model, p).register_forward_hook(self._hook(p))
            if hasattr(self.model, c):
                self.available_layers.append(c)

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
            raise RuntimeError(
                f"Activation '{key}' not captured.\n"
                "This usually means the model forward does not use that module directly."
            )

    def run(self, pil_img, layer_name="conv1", auto_select=True, manual_idx=0):
        if pil_img is None:
            return (
                None, None, None, None, None, None, None, "",
                "No image loaded.",
                "Upload a fundus image, then click Run Explain.",
                "No debug info.",
                gr.update()
            )

        if layer_name not in self.available_layers:
            layer_name = self.available_layers[0]

        relu_name = layer_name.replace("conv", "relu")
        pool_name = layer_name.replace("conv", "pool")

        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Ensure hooks fired
        self._require_activation(layer_name)
        self._require_activation(relu_name)
        self._require_activation(pool_name)

        inp = x[0, 0].detach().cpu().numpy()
        inp01 = np.clip(inp * 0.5 + 0.5, 0.0, 1.0)

        conv = self.activations[layer_name][0].numpy()  # [C,H,W]
        relu = self.activations[relu_name][0].numpy()
        pool = self.activations[pool_name][0].numpy()

        # Strongest channel by mean abs activation
        strength = np.mean(np.abs(conv), axis=(1, 2))
        strongest_idx = int(np.argmax(strength))

        if auto_select:
            fmap_idx = strongest_idx
        else:
            fmap_idx = int(np.clip(int(manual_idx), 0, conv.shape[0] - 1))

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
            title=f"Kernel inspector ({layer_name.upper()}): selected + top-3 channels"
        )
        strength_fig = plot_strengths(strength, fmap_idx, layer_name)

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        relu_zero_frac = float(np.mean(relu[fmap_idx] <= 0))
        score = float(strength[fmap_idx])

        summary = (
            f"**Layer:** `{layer_name}`  \n"
            f"**Selected channel:** `{fmap_idx}` ({'auto strongest' if auto_select else 'manual'})  \n"
            f"**Selection score (mean |activation|):** `{score:.4f}`  \n"
            f"**Prediction:** `{pred_name}` with confidence `{pred_conf:.2%}`"
        )

        explain = (
            "### Explain Panel\n"
            f"- Feature map from **{layer_name}**, channel **{fmap_idx}**.\n"
            f"- ReLU zeroed fraction for selected map: **{relu_zero_frac:.1%}**.\n"
            "- MaxPool keeps strongest local evidence while downsampling.\n"
            "- Overlay aligns selected convolution evidence with retinal structure.\n"
            "- Kernel inspector shows learned \(3 \\times 3\) weights "
            "(deeper layers shown as input-channel mean)."
        )

        top5 = sorted_idx[:5]
        debug_text = (
            f"Strongest channel in {layer_name}: **{strongest_idx}**  \n"
            f"Top-5 by strength: `{top5}`  \n"
            f"Mode: `{'AUTO' if auto_select else 'MANUAL'}`"
        )

        probs_md = probs_to_markdown(probs)

        # IMPORTANT: update slider value so auto-select is visible in UI
        slider_update = gr.update(value=int(fmap_idx))

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
            debug_text,
            slider_update,
        )


def build_app(ckpt_path: str, inject_css_fallback: bool = False):
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

    with gr.Blocks(title="DR CNN Explain Console") as demo:
        if inject_css_fallback:
            gr.HTML(f"<style>{APP_CSS}</style>")

        with gr.Column(elem_id="app-root"):
            with gr.Group(elem_classes=["card"]):
                gr.HTML('<div class="card-header">Diabetic Retinopathy CNN · Explain Console</div>')
                with gr.Column(elem_classes=["card-body"]):
                    gr.Markdown(
                        "<p class='subtle'>Clean diagnostics view for channel-level interpretability. "
                        "No gimmicks, just model internals and evidence maps.</p>"
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=380):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">Controls</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            inp = gr.Image(type="pil", label="Fundus Image", height=320)

                            layer_select = gr.Dropdown(
                                choices=explainer.available_layers,
                                value=default_layer,
                                label="Convolution Block",
                            )

                            auto_select = gr.Checkbox(
                                value=True,
                                label="Auto-select strongest channel",
                            )

                            manual_idx = gr.Slider(
                                minimum=0,
                                maximum=default_max,
                                step=1,
                                value=0,
                                label="Channel Index (used only when auto is OFF)",
                                interactive=False,
                            )

                            run_btn = gr.Button("Run Explain")

                with gr.Column(scale=5, min_width=380):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">Prediction + Notes</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            gr.Markdown(METRICS_MD)
                            out_probs = gr.Markdown(label="Class Probabilities")
                            out_summary = gr.Markdown()
                            out_explain = gr.Markdown()
                            out_debug = gr.Markdown()

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">Input (Gray)</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            out_input = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">Conv Map</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            out_conv = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">ReLU Map</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            out_relu = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=200):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">MaxPool Map</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            out_pool = gr.Image(height=170, show_label=False)

            with gr.Row(equal_height=True):
                with gr.Column(scale=4, min_width=340):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">Overlay (Input + Conv Evidence)</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            out_overlay = gr.Image(height=260, show_label=False)

                with gr.Column(scale=6, min_width=460):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML('<div class="card-header">Kernel Inspector + Strength Plot</div>')
                        with gr.Column(elem_classes=["card-body"]):
                            out_kernel = gr.Plot()
                            out_strength = gr.Plot()

            gr.ClearButton(
                components=[
                    inp, out_input, out_conv, out_relu, out_pool,
                    out_overlay, out_kernel, out_strength,
                    out_probs, out_summary, out_explain, out_debug
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
                out_debug, manual_idx,  # <- slider updates to selected channel
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
        launch_kwargs["css"] = APP_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()