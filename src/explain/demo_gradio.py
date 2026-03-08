import argparse
import os
import sys

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import SimpleDRCNN
from transforms import get_transforms


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

RETRO_CSS = r"""
:root{
  --panel:#efefef;
  --dark:#6b6b6b;
  --mid:#9a9a9a;
  --light:#ffffff;
  --title:#003c74;
  --text:#111111;
}

/* App background + font */
.gradio-container{
  font-family: Tahoma, Verdana, Arial, sans-serif !important;
  color: var(--text) !important;
  background:
    repeating-linear-gradient(
      0deg,
      #d6d6d6,
      #d6d6d6 2px,
      #cecece 2px,
      #cecece 4px
    ) !important;
}

/* Window look */
.retro-window{
  border: 1px solid var(--dark) !important;
  background: var(--panel) !important;
  box-shadow:
    inset 1px 1px 0 var(--light),
    inset -1px -1px 0 var(--mid) !important;
  padding: 8px !important;
}

/* Title bars */
.retro-title{
  background: linear-gradient(#0b5ca3, var(--title)) !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  border: 1px solid #00284d !important;
  padding: 6px 8px !important;
  margin-bottom: 8px !important;
}

/* Labels */
.retro-subtitle{
  font-weight: 700 !important;
  margin-bottom: 4px !important;
}

/* Buttons */
.retro-btn button{
  border-radius: 0 !important;
  border: 1px solid #222 !important;
  border-top-color: #fff !important;
  border-left-color: #fff !important;
  border-right-color: #6b6b6b !important;
  border-bottom-color: #6b6b6b !important;
  background: #dcdcdc !important;
  color: #111 !important;
}
.retro-btn button:active{
  transform: translate(1px, 1px) !important;
  border-top-color: #6b6b6b !important;
  border-left-color: #6b6b6b !important;
  border-right-color: #fff !important;
  border-bottom-color: #fff !important;
}

/* Inputs */
.retro-input input,
.retro-input textarea,
.retro-input select,
.retro-input .wrap,
.retro-input .block{
  border-radius: 0 !important;
}

.retro-note{
  border: 1px solid #5a0000 !important;
  background: #ffecec !important;
  padding: 6px 8px !important;
  font-size: 12px !important;
}
"""


def norm_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


class Explainer:
    def __init__(self, ckpt_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.image_size = int(ckpt.get("image_size", 224))
        self.num_classes = int(ckpt.get("num_classes", 5))

        self.model = SimpleDRCNN(num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tf = get_transforms(image_size=self.image_size, train=False)
        self.activations = {}

        # hooks for early-layer teaching views
        self.model.conv1.register_forward_hook(self._hook("conv1"))
        self.model.relu1.register_forward_hook(self._hook("relu1"))
        self.model.pool1.register_forward_hook(self._hook("pool1"))

        # inferred channel counts
        self.conv1_out_channels = self.model.conv1.out_channels

    def _hook(self, name: str):
        def fn(_, __, output):
            self.activations[name] = output.detach().cpu()
        return fn

    def run(
        self,
        pil_img,
        channel_mode="Auto strongest",
        fmap_idx_manual=0,
        link_kernel=True,
        kernel_idx_manual=0,
    ):
        if pil_img is None:
            return (None, None, None, None, None, None, None, "No image provided.")

        try:
            x = self.tf(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            inp = x[0, 0].detach().cpu().numpy()
            conv = self.activations["conv1"][0].numpy()  # [C, H, W]
            relu = self.activations["relu1"][0].numpy()
            pool = self.activations["pool1"][0].numpy()

            strength = np.mean(np.abs(conv), axis=(1, 2))
            n_channels = conv.shape[0]

            # Channel selection
            if channel_mode == "Auto strongest":
                fmap_idx = int(np.argmax(strength))
            elif channel_mode == "Auto 2nd strongest":
                fmap_idx = int(np.argsort(strength)[-2]) if n_channels > 1 else 0
            elif channel_mode == "Auto weakest":
                fmap_idx = int(np.argmin(strength))
            else:  # Manual
                fmap_idx = int(fmap_idx_manual) % n_channels

            # Kernel selection
            if link_kernel:
                kernel_idx = fmap_idx
            else:
                kernel_idx = int(kernel_idx_manual) % self.model.conv1.weight.shape[0]

            conv_map = norm_map(conv[fmap_idx])
            relu_map = norm_map(relu[fmap_idx])
            pool_map = norm_map(pool[fmap_idx])
            input_map = norm_map(inp)

            kernel = self.model.conv1.weight[kernel_idx, 0].detach().cpu().numpy()
            kernel_map = norm_map(kernel)

            # activation strength chart
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(np.arange(len(strength)), strength)
            ax.set_title("Conv1 channel activation strength (mean abs activation)")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Strength")
            fig.tight_layout()

            # Dense + Softmax dict
            prob_dict = {
                CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}": float(probs[i])
                for i in range(len(probs))
            }

            info = (
                f"Selected channel: {fmap_idx} ({channel_mode}) | "
                f"Selected kernel: {kernel_idx} ({'linked' if link_kernel else 'manual'}) | "
                f"Conv1 channels: {n_channels}"
            )

            return (input_map, conv_map, relu_map, pool_map, kernel_map, fig, prob_dict, info)

        except Exception as e:
            return (None, None, None, None, None, None, None, f"Error: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    explainer = Explainer(args.ckpt)

    with gr.Blocks(
        title="DR CNN Layer Explorer",
        css=RETRO_CSS,
        theme=gr.themes.Base(),
    ) as demo:
        with gr.Column(elem_classes=["retro-window"]):
            gr.Markdown("### DR Layer Explorer — Retro Teaching Panel", elem_classes=["retro-title"])
            gr.Markdown(
                "**Purpose:** Inspect what each stage passes forward in a CNN-like pipeline.\n\n"
                "**Reminder:** This is an educational demo, not a clinical diagnostic system."
            )
            gr.Markdown(
                "⚠️ Research/education only. Not a medical device. Do not use for diagnosis or treatment decisions.",
                elem_classes=["retro-note"],
            )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["retro-window"]):
                gr.Markdown("#### Controls Panel", elem_classes=["retro-title"])

                inp = gr.Image(type="pil", label="Input Fundus Image", elem_classes=["retro-input"])

                channel_mode = gr.Dropdown(
                    choices=["Auto strongest", "Auto 2nd strongest", "Auto weakest", "Manual"],
                    value="Auto strongest",
                    label="Channel selection mode",
                    elem_classes=["retro-input"],
                )

                fmap_idx = gr.Slider(
                    0,
                    max(15, explainer.conv1_out_channels - 1),
                    value=0,
                    step=1,
                    label="Manual feature map channel",
                    visible=False,
                    elem_classes=["retro-input"],
                )

                link_kernel = gr.Checkbox(
                    value=True,
                    label="Link kernel to selected channel (recommended)",
                    elem_classes=["retro-input"],
                )

                kernel_idx = gr.Slider(
                    0,
                    max(15, explainer.conv1_out_channels - 1),
                    value=0,
                    step=1,
                    label="Manual kernel index",
                    visible=False,
                    elem_classes=["retro-input"],
                )

                btn = gr.Button("Run Explain", elem_classes=["retro-btn"])

            with gr.Column(scale=3, elem_classes=["retro-window"]):
                gr.Markdown("#### Feature View Panel", elem_classes=["retro-title"])

                with gr.Row():
                    out_input = gr.Image(label="Input (single-channel used by CNN)")
                    out_conv = gr.Image(label="Convolution output (normalized)")
                with gr.Row():
                    out_relu = gr.Image(label="ReLU output")
                    out_pool = gr.Image(label="MaxPool output")

                with gr.Row():
                    out_kernel = gr.Image(label="Selected Conv1 Kernel Values")
                    out_bar = gr.Plot(label="Activation Strengths")

                out_probs = gr.Label(label="Dense + Softmax output (demo classes)")
                out_info = gr.Textbox(label="Selection info", interactive=False)

        def infer(img, c_mode, fmap_manual, link_k, kernel_manual):
            return explainer.run(img, c_mode, fmap_manual, link_k, kernel_manual)

        def toggle_manual_channel(mode):
            return gr.update(visible=(mode == "Manual"))

        def toggle_manual_kernel(linked):
            return gr.update(visible=(not linked))

        channel_mode.change(toggle_manual_channel, inputs=channel_mode, outputs=fmap_idx)
        link_kernel.change(toggle_manual_kernel, inputs=link_kernel, outputs=kernel_idx)

        btn.click(
            infer,
            inputs=[inp, channel_mode, fmap_idx, link_kernel, kernel_idx],
            outputs=[out_input, out_conv, out_relu, out_pool, out_kernel, out_bar, out_probs, out_info],
        )

    demo.queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
    