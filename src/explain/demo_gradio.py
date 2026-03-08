import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import SimpleDRCNN
from transforms import get_transforms


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


RETRO_CSS = """
:root {
  --bg: #dcdcdc;
  --panel: #efefef;
  --border-dark: #6b6b6b;
  --border-light: #ffffff;
  --title: #003c74;
  --title-text: #ffffff;
  --text: #111111;
  --btn: #d9d9d9;
  --btn-active: #c8c8c8;
}

.gradio-container {
  font-family: Tahoma, Verdana, Arial, sans-serif !important;
  color: var(--text) !important;
  background:
    repeating-linear-gradient(
      0deg,
      #d8d8d8 0px,
      #d8d8d8 2px,
      #dcdcdc 2px,
      #dcdcdc 4px
    ) !important;
}

#app-root {
  max-width: 1400px;
  margin: 0 auto;
}

.window {
  background: var(--panel) !important;
  border: 1px solid var(--border-dark) !important;
  box-shadow:
    inset 1px 1px 0 var(--border-light),
    inset -1px -1px 0 #b5b5b5 !important;
  padding: 0 !important;
  margin-bottom: 10px !important;
}

.titlebar {
  background: linear-gradient(to right, #003c74, #0a4f90);
  color: var(--title-text);
  font-weight: 700;
  padding: 6px 8px;
  border-bottom: 1px solid #002b54;
  font-size: 13px;
  letter-spacing: 0.2px;
}

.window-body {
  padding: 10px;
}

.compact-note {
  font-size: 12px;
  line-height: 1.35;
  margin: 0;
}

.gr-button {
  border-radius: 0 !important;
  border: 1px solid var(--border-dark) !important;
  background: var(--btn) !important;
  color: #111 !important;
  box-shadow:
    inset 1px 1px 0 var(--border-light),
    inset -1px -1px 0 #9f9f9f !important;
}
.gr-button:active {
  background: var(--btn-active) !important;
  box-shadow:
    inset -1px -1px 0 var(--border-light),
    inset 1px 1px 0 #9f9f9f !important;
  transform: translate(1px, 1px);
}

input, textarea, select {
  border-radius: 0 !important;
  border: 1px solid var(--border-dark) !important;
  box-shadow: inset 1px 1px 0 var(--border-light) !important;
}

label, .gr-form, .gr-markdown, .gr-checkbox {
  color: #111 !important;
}

hr {
  border: 0;
  border-top: 1px solid #9a9a9a;
  border-bottom: 1px solid #fff;
  margin: 8px 0;
}
"""


def norm_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255).astype(np.uint8)


def resize_heatmap_to(heat01: np.ndarray, target_hw):
    h, w = target_hw
    img = Image.fromarray(to_uint8(heat01))
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def make_overlay(gray01: np.ndarray, heat01: np.ndarray, alpha=0.45) -> np.ndarray:
    heat_up = resize_heatmap_to(heat01, gray01.shape)
    cmap = plt.get_cmap("jet")(heat_up)[..., :3]
    base = np.stack([gray01, gray01, gray01], axis=-1)
    out = np.clip((1 - alpha) * base + alpha * cmap, 0.0, 1.0)
    return to_uint8(out)


def plot_kernel_3x3(kernel: np.ndarray):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    vmax = float(np.max(np.abs(kernel))) + 1e-8
    im = ax.imshow(kernel, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Conv1 Kernel (3x3) with Values", fontsize=10)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(3):
            val = kernel[i, j]
            txt_color = "white" if abs(val) > (0.45 * vmax) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Weight", rotation=90)
    fig.tight_layout()
    return fig


def plot_activation_strengths(strength: np.ndarray, selected_idx: int):
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=120)
    colors = ["#4f81bd"] * len(strength)
    colors[selected_idx] = "#c0504d"
    ax.bar(np.arange(len(strength)), strength, color=colors)
    ax.set_title("Conv1 Channel Strength (mean |activation|)", fontsize=10)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Strength")
    ax.set_xticks(np.arange(len(strength)))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


class Explainer:
    def __init__(self, ckpt_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.image_size = ckpt.get("image_size", 224)

        self.model = SimpleDRCNN(num_classes=ckpt.get("num_classes", 5)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tf = get_transforms(image_size=self.image_size, train=False)
        self.activations = {}

        self.model.conv1.register_forward_hook(self._hook("conv1"))
        self.model.relu1.register_forward_hook(self._hook("relu1"))
        self.model.pool1.register_forward_hook(self._hook("pool1"))

    def _hook(self, name):
        def fn(_, __, output):
            self.activations[name] = output.detach().cpu()
        return fn

    def run(self, pil_img, auto_select=True, manual_fmap_idx=0):
        if pil_img is None:
            return (
                None, None, None, None, None,
                None, None, {},
                "No image loaded.",
                "Upload an image, then click **Run Explain**."
            )

        x = self.tf(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        inp = x[0, 0].detach().cpu().numpy()          # normalized [-1, 1]
        inp01 = np.clip(inp * 0.5 + 0.5, 0.0, 1.0)    # denormalize for display

        conv = self.activations["conv1"][0].numpy()   # (C, H, W)
        relu = self.activations["relu1"][0].numpy()
        pool = self.activations["pool1"][0].numpy()

        strength = np.mean(np.abs(conv), axis=(1, 2))
        best_idx = int(np.argmax(strength))

        if auto_select:
            fmap_idx = best_idx
            reason = "auto-selected strongest channel by mean absolute Conv1 activation"
        else:
            fmap_idx = int(manual_fmap_idx) % conv.shape[0]
            reason = "manual channel override"

        kernel_idx = fmap_idx  # Conv1 output channel corresponds to kernel index
        kernel = self.model.conv1.weight[kernel_idx, 0].detach().cpu().numpy()  # (3,3)

        conv_map01 = norm_map(conv[fmap_idx])
        relu_map01 = norm_map(relu[fmap_idx])
        pool_map01 = norm_map(pool[fmap_idx])

        overlay = make_overlay(inp01, conv_map01, alpha=0.45)

        kernel_fig = plot_kernel_3x3(kernel)
        strength_fig = plot_activation_strengths(strength, fmap_idx)

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        relu_zero_frac = float(np.mean(relu[fmap_idx] <= 0.0))
        conv_mean_abs = float(strength[fmap_idx])

        summary = (
            f"**Selected channel/kernel:** `{fmap_idx}` ({reason})  \n"
            f"**Selection score (mean |conv activation|):** `{conv_mean_abs:.4f}`  \n"
            f"**Model top prediction:** `{pred_name}` with confidence `{pred_conf:.2%}`"
        )

        explanation = (
            "### Explain Panel\n"
            f"- The app inspected all Conv1 channels and chose channel **{fmap_idx}** "
            "because it responded most strongly to this image.\n"
            f"- **Conv map** shows signed filter response intensity (normalized for viewing).\n"
            f"- **ReLU map** keeps only positive evidence; here about **{relu_zero_frac:.1%}** of "
            "values are zeroed in this channel.\n"
            "- **MaxPool map** is spatially downsampled evidence passed deeper into the network.\n"
            "- **Overlay map** highlights where this selected channel is most active on the input.\n"
            "- **Kernel 3x3 plot** shows the actual learned Conv1 weights (with exact values), "
            "so you can inspect whether it behaves like an edge/blob detector."
        )

        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return (
            to_uint8(inp01),
            to_uint8(conv_map01),
            to_uint8(relu_map01),
            to_uint8(pool_map01),
            overlay,
            kernel_fig,
            strength_fig,
            prob_dict,
            summary,
            explanation
        )


def build_app(ckpt_path):
    explainer = Explainer(ckpt_path)

    def infer(img, auto_select, manual_idx):
        return explainer.run(img, auto_select, manual_idx)

    def on_auto_change(auto_select):
        return gr.update(interactive=not auto_select)

    with gr.Blocks(title="DR CNN Retro Explain Console", css=RETRO_CSS, theme=gr.themes.Default()) as demo:
        with gr.Column(elem_id="app-root"):
            with gr.Group(elem_classes=["window"]):
                gr.HTML('<div class="titlebar">Diabetic Retinopathy CNN — Retro Explain Console</div>')
                with gr.Column(elem_classes=["window-body"]):
                    gr.Markdown(
                        "<p class='compact-note'><b>Purpose:</b> Inspect Conv1 behavior, "
                        "auto-pick strongest channel, and review kernel values with a 3×3 heatmap.</p>"
                    )

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Control Panel</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            inp = gr.Image(type="pil", label="Fundus Image", height=360)
                            auto_select = gr.Checkbox(
                                value=True,
                                label="Auto-select strongest Conv1 channel/kernel"
                            )
                            fmap_idx = gr.Slider(
                                0, 15, value=0, step=1,
                                label="Manual channel (used only when auto-select is OFF)",
                                interactive=False
                            )
                            with gr.Row():
                                btn = gr.Button("Run Explain", variant="primary")
                                # Clear everything compactly:
                                # (clear button placed after output declarations below)

                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Prediction + Explanation</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_probs = gr.Label(label="Dense + Softmax Output")
                            out_summary = gr.Markdown()
                            out_explain = gr.Markdown()

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Input (CNN Gray)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_input = gr.Image(height=240)
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Conv1 Selected Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_conv = gr.Image(height=240)
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">ReLU Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_relu = gr.Image(height=240)
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">MaxPool Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_pool = gr.Image(height=240)

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Overlay (Input + Conv1 Heat)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_overlay = gr.Image(height=280)
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Kernel Inspector (3×3 + Values)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_kernel = gr.Plot()
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Activation Strengths</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_bar = gr.Plot()

            clear_btn = gr.ClearButton(
                value="Clear",
                components=[
                    inp, out_input, out_conv, out_relu, out_pool,
                    out_overlay, out_kernel, out_bar, out_probs,
                    out_summary, out_explain
                ]
            )

        auto_select.change(on_auto_change, inputs=[auto_select], outputs=[fmap_idx])

        btn.click(
            infer,
            inputs=[inp, auto_select, fmap_idx],
            outputs=[
                out_input, out_conv, out_relu, out_pool, out_overlay,
                out_kernel, out_bar, out_probs, out_summary, out_explain
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