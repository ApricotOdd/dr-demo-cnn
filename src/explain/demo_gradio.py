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

# You can update this text with fresh metrics anytime
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
  --panel: #efefef;
  --panel-border-dark: #6b6b6b;
  --panel-border-light: #ffffff;
  --title: #003c74;
  --title-text: #ffffff;
  --text: #111111;
  --muted: #333333;
  --btn: #dcdcdc;
  --btn-dark: #808080;
  --btn-light: #ffffff;
}

/* App background + typography */
.gradio-container {
  font-family: Tahoma, Verdana, Arial, sans-serif !important;
  color: var(--text) !important;
  background:
    repeating-linear-gradient(
      0deg,
      #c6c6c6 0px,
      #c6c6c6 2px,
      #c0c0c0 2px,
      #c0c0c0 4px
    ) !important;
}

#app-root {
  max-width: 1450px;
  margin: 0 auto;
}

/* Window look */
.window {
  border: 1px solid var(--panel-border-dark) !important;
  background: var(--panel) !important;
  box-shadow:
    inset 1px 1px 0 var(--panel-border-light),
    inset -1px -1px 0 #a0a0a0 !important;
  padding: 0 !important;
  margin-bottom: 10px !important;
}

.titlebar {
  background: var(--title);
  color: var(--title-text);
  font-weight: bold;
  font-size: 13px;
  padding: 6px 8px;
  border-bottom: 1px solid #00284e;
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

/* Buttons + controls */
.gr-button {
  border: 1px solid var(--btn-dark) !important;
  background: var(--btn) !important;
  color: #111 !important;
  border-radius: 0 !important;
  box-shadow:
    inset 1px 1px 0 var(--btn-light),
    inset -1px -1px 0 #9c9c9c !important;
}
.gr-button:active {
  box-shadow:
    inset -1px -1px 0 var(--btn-light),
    inset 1px 1px 0 #9c9c9c !important;
  transform: translateY(1px);
}

input, textarea, select {
  border-radius: 0 !important;
  border: 1px solid #888 !important;
  background: #fff !important;
  box-shadow: inset 1px 1px 0 #fff !important;
}

label, .gr-markdown, .gr-form, .gr-checkbox {
  color: #111 !important;
}

/* Make panel images feel "framed" */
.image-container, .gr-image {
  border-radius: 0 !important;
}

/* Tight spacing */
.block, .gr-group {
  gap: 6px !important;
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


def plot_strengths(strength: np.ndarray, selected: int):
    fig, ax = plt.subplots(figsize=(7.2, 2.7), dpi=120)
    colors = ["#1f73be"] * len(strength)
    colors[selected] = "#2ca24f"
    ax.bar(np.arange(len(strength)), strength, color=colors)
    ax.set_title("Conv1 Channel Activation Strength (mean |activation|)", fontsize=10)
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


class Explainer:
    def __init__(self, ckpt_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.image_size = ckpt.get("image_size", 224)

        self.model = SimpleDRCNN(num_classes=ckpt.get("num_classes", 5)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tf = get_transforms(image_size=self.image_size, train=False)
        self.activations = {}

        # Current teaching panel focuses Conv1 block
        self.model.conv1.register_forward_hook(self._hook("conv1"))
        self.model.relu1.register_forward_hook(self._hook("relu1"))
        self.model.pool1.register_forward_hook(self._hook("pool1"))

    def _hook(self, name):
        def fn(_, __, out):
            self.activations[name] = out.detach().cpu()
        return fn

    def run(self, pil_img, auto_select=True, manual_idx=0):
        if pil_img is None:
            return (
                None, None, None, None, None, None, None, {},
                "No image loaded.",
                "Upload a fundus image, then click **Run Explain**."
            )

        x = self.tf(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        inp = x[0, 0].detach().cpu().numpy()               # normalized [-1,1]
        inp01 = np.clip(inp * 0.5 + 0.5, 0.0, 1.0)         # display [0,1]

        conv = self.activations["conv1"][0].numpy()        # (16,H,W)
        relu = self.activations["relu1"][0].numpy()
        pool = self.activations["pool1"][0].numpy()

        strength = np.mean(np.abs(conv), axis=(1, 2))
        best_idx = int(np.argmax(strength))
        fmap_idx = best_idx if auto_select else (int(manual_idx) % conv.shape[0])

        conv01 = norm01(conv[fmap_idx])
        relu01 = norm01(relu[fmap_idx])
        pool01 = norm01(pool[fmap_idx])

        # MaxPool is lower-res; upscale for visibility but keep smaller panel in UI
        pool_up = resize01(pool01, 224, 224, mode=Image.NEAREST)
        overlay = make_overlay(inp01, conv01, alpha=0.45)

        # selected + top3 strongest channel kernels
        sorted_idx = np.argsort(-strength).tolist()
        top_channels = [fmap_idx] + [c for c in sorted_idx if c != fmap_idx][:3]
        kernels = [self.model.conv1.weight[c, 0].detach().cpu().numpy() for c in top_channels]

        kernel_fig = plot_kernel_grid(
            np.array(kernels),
            top_channels,
            title="Kernel Inspector (3x3 values): selected + top-3 channels"
        )
        strength_fig = plot_strengths(strength, fmap_idx)

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        relu_zero_frac = float(np.mean(relu[fmap_idx] <= 0))
        score = float(strength[fmap_idx])

        summary = (
            f"**Selected channel/kernel:** `{fmap_idx}` "
            f"({'auto strongest' if auto_select else 'manual override'})  \n"
            f"**Selection score (mean |conv activation|):** `{score:.4f}`  \n"
            f"**Prediction:** `{pred_name}` with confidence `{pred_conf:.2%}`"
        )

        explain = (
            "### Explain Panel\n"
            f"- Conv1 selected map shows where channel **{fmap_idx}** responds strongly.\n"
            f"- ReLU keeps positive evidence (zeroed fraction: **{relu_zero_frac:.1%}**).\n"
            "- MaxPool keeps strongest local evidence and downsamples before deeper layers.\n"
            "- Overlay aligns activation with the retinal structure.\n"
            "- Kernel inspector shows actual learned \(3 \times 3\) weights with numeric values."
        )

        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return (
            to_uint8(inp01),
            to_uint8(conv01),
            to_uint8(relu01),
            to_uint8(pool_up),
            overlay,
            kernel_fig,
            strength_fig,
            prob_dict,
            summary,
            explain,
        )


def build_app(ckpt_path: str):
    explainer = Explainer(ckpt_path)

    def infer(img, auto_select, manual_idx):
        return explainer.run(img, auto_select, manual_idx)

    def on_auto_change(auto_select):
        return gr.update(interactive=not auto_select)

    with gr.Blocks(
        title="DR CNN Retro Explain Console",
        css=RETRO_CSS,
        theme=gr.themes.Default(),
    ) as demo:
        with gr.Column(elem_id="app-root"):
            # Header window
            with gr.Group(elem_classes=["window"]):
                gr.HTML('<div class="titlebar">Diabetic Retinopathy CNN — Retro Explain Console</div>')
                with gr.Column(elem_classes=["window-body"]):
                    gr.Markdown(
                        "<p class='small-note'>Classic panel mode: compact controls, clear borders, "
                        "and teaching-first explainability.</p>"
                    )

            # Top split: controls + performance/prediction
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=360):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Controls</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            inp = gr.Image(type="pil", label="Fundus Image", height=320)
                            auto_select = gr.Checkbox(value=True, label="Auto-select strongest Conv1 channel/kernel")
                            manual_idx = gr.Slider(
                                minimum=0, maximum=15, step=1, value=0,
                                label="Manual channel index (used only when auto-select is OFF)",
                                interactive=False
                            )
                            with gr.Row():
                                run_btn = gr.Button("Run Explain", variant="primary")

                with gr.Column(scale=5, min_width=360):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Performance + Prediction</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            gr.Markdown(METRICS_MD)
                            out_probs = gr.Label(label="Dense + Softmax Output")
                            out_summary = gr.Markdown()
                            out_explain = gr.Markdown()

            # 4 maps in one row (maxpool deliberately smaller)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Input (CNN Gray)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_input = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Conv1 Selected Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_conv = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=220):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">ReLU Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_relu = gr.Image(height=220, show_label=False)

                with gr.Column(scale=1, min_width=180):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">MaxPool Map</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_pool = gr.Image(height=170, show_label=False)

            # Bottom diagnostics
            with gr.Row(equal_height=True):
                with gr.Column(scale=4, min_width=320):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Overlay (Input + Conv1 Heat)</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_overlay = gr.Image(height=260, show_label=False)

                with gr.Column(scale=6, min_width=440):
                    with gr.Group(elem_classes=["window"]):
                        gr.HTML('<div class="titlebar">Kernel Inspector + Strengths</div>')
                        with gr.Column(elem_classes=["window-body"]):
                            out_kernel = gr.Plot()
                            out_strength = gr.Plot()

            clear_btn = gr.ClearButton(
                components=[
                    inp, out_input, out_conv, out_relu, out_pool,
                    out_overlay, out_kernel, out_strength,
                    out_probs, out_summary, out_explain
                ],
                value="Clear",
            )

        auto_select.change(on_auto_change, inputs=[auto_select], outputs=[manual_idx])

        run_btn.click(
            infer,
            inputs=[inp, auto_select, manual_idx],
            outputs=[
                out_input, out_conv, out_relu, out_pool, out_overlay,
                out_kernel, out_strength, out_probs, out_summary, out_explain
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
    # Your current gradio build expects plain launch signature
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()