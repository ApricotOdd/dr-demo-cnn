# src/explain/demo_gradio.py
import argparse
import inspect
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


# ----------------------------
# Path / import setup
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]   # repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _import_model_class():
    """
    Try to import your project model class from src/model.py.
    Falls back to the first nn.Module class found.
    """
    try:
        import model as model_module
    except Exception as e:
        raise RuntimeError(f"Could not import src/model.py: {e}")

    preferred_names = [
        "SimpleCNN",
        "DRCNN",
        "Net",
        "Model",
        "BaselineCNN",
    ]
    for name in preferred_names:
        if hasattr(model_module, name):
            cls = getattr(model_module, name)
            if inspect.isclass(cls) and issubclass(cls, nn.Module):
                return cls

    # Fallback: first nn.Module class in module
    for _, obj in model_module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            return obj

    raise RuntimeError("No nn.Module class found in src/model.py")


def _infer_num_classes_from_state_dict(state_dict, default=5):
    candidates = [k for k in state_dict.keys() if k.endswith(".weight")]
    # Try typical classifier layers first
    priority = ["fc.weight", "classifier.weight", "head.weight", "out.weight"]
    for p in priority:
        for k in candidates:
            if k.endswith(p):
                w = state_dict[k]
                if hasattr(w, "shape") and len(w.shape) >= 2:
                    return int(w.shape[0])

    # fallback: last 2D weight
    for k in reversed(candidates):
        w = state_dict[k]
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[0])

    return default


def _build_model(model_cls, num_classes):
    # Try common constructor styles
    for kwargs in [
        {"num_classes": num_classes},
        {"n_classes": num_classes},
        {},
    ]:
        try:
            return model_cls(**kwargs)
        except TypeError:
            continue
    # last resort
    return model_cls()


def _to_uint8_img(arr2d):
    arr = np.asarray(arr2d, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _fig_from_array(img, title="", cmap="gray"):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


def _activation_bar_figure(values):
    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=120)
    x = np.arange(len(values))
    ax.bar(x, values, color="#2f6db2")
    ax.set_title("Conv1 channel activation strength (mean abs activation)", fontsize=10)
    ax.set_xlabel("Channel", fontsize=9)
    ax.set_ylabel("Strength", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


class Explainer:
    def __init__(self, ckpt_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            meta = ckpt
        elif isinstance(ckpt, dict):
            state_dict = ckpt
            meta = {}
        else:
            raise RuntimeError("Checkpoint format not recognized.")

        self.class_names = (
            meta.get("class_names")
            or ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        )

        model_cls = _import_model_class()
        num_classes = _infer_num_classes_from_state_dict(state_dict, default=len(self.class_names))
        self.model = _build_model(model_cls, num_classes=num_classes).to(self.device)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict strict=False | missing={len(missing)}, unexpected={len(unexpected)}")

        self.model.eval()

        # first conv layer
        self.first_conv = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.first_conv = m
                break
        if self.first_conv is None:
            raise RuntimeError("No Conv2d layer found in model.")

        # preprocessing
        self.tf_rgb = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 0..1
        ])

        self.tf_gray = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def run(
        self,
        pil_img,
        channel_mode="Manual",
        fmap_idx_manual=0,
        link_kernel=True,
        kernel_idx_manual=0,
    ):
        if pil_img is None:
            blank = Image.new("RGB", (224, 224), color=(0, 0, 0))
            pil_img = blank

        pil_img = pil_img.convert("RGB")
        x_rgb = self.tf_rgb(pil_img).unsqueeze(0).to(self.device)   # [1,3,H,W]
        x_gray = self.tf_gray(pil_img).unsqueeze(0).to(self.device) # [1,1,H,W]

        # Choose model input based on conv in-channels
        if self.first_conv.in_channels == 1:
            x_in = x_gray
        else:
            x_in = x_rgb[:, : self.first_conv.in_channels, :, :]

        # Manual conv/relu/pool from first layer (for visualization)
        conv_out = self.first_conv(x_in)              # [1,C,H,W]
        relu_out = F.relu(conv_out)
        pool_out = F.max_pool2d(relu_out, kernel_size=2, stride=2)

        # activations
        acts = conv_out.abs().mean(dim=(0, 2, 3)).detach().cpu().numpy()
        c = int(conv_out.shape[1])

        mode = (channel_mode or "Manual").strip().lower()
        if mode == "auto strongest":
            fmap_idx = int(np.argmax(acts))
        elif mode == "auto weakest":
            fmap_idx = int(np.argmin(acts))
        elif mode == "auto median":
            fmap_idx = int(np.argsort(acts)[len(acts) // 2])
        else:
            fmap_idx = int(np.clip(int(fmap_idx_manual), 0, c - 1))

        # kernel choice (for display only; kernel does not affect conv result after training)
        if bool(link_kernel):
            kernel_idx = fmap_idx
            kernel_source = "linked to channel"
        else:
            kernel_idx = int(np.clip(int(kernel_idx_manual), 0, c - 1))
            kernel_source = "manual"

        # selected maps
        conv_map = conv_out[0, fmap_idx].detach().cpu().numpy()
        relu_map = relu_out[0, fmap_idx].detach().cpu().numpy()
        pool_map = pool_out[0, fmap_idx].detach().cpu().numpy()

        conv_img = _to_uint8_img(conv_map)
        relu_img = _to_uint8_img(relu_map)
        pool_img = _to_uint8_img(pool_map)

        # input shown as single-channel used by CNN
        if x_in.shape[1] == 1:
            in_show = (x_in[0, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        else:
            in_show = (x_in[0].mean(dim=0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # kernel visualization
        w = self.first_conv.weight.detach().cpu().numpy()  # [C_out, C_in, k, k]
        k_vis = w[kernel_idx]
        if k_vis.ndim == 3:
            k_vis = k_vis.mean(axis=0)  # mean over input channels
        k_img = _to_uint8_img(k_vis)

        # full model prediction
        logits = self.model(x_in)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class {pred_idx}"

        # figures
        fig_act = _activation_bar_figure(acts)
        fig_kernel = _fig_from_array(k_img, title="Selected kernel values", cmap="viridis")

        # probabilities HTML
        lines = [f"<div style='font-size:28px;font-weight:700;margin-bottom:8px'>{pred_name}</div>"]
        order = np.argsort(-probs)
        for i in order:
            name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            pct = float(probs[i] * 100.0)
            lines.append(
                f"""
                <div style="margin:6px 0;">
                  <div style="display:flex;justify-content:space-between;font-size:13px;">
                    <span>{name}</span><span>{pct:.1f}%</span>
                  </div>
                  <div style="height:8px;background:#20242b;border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{pct:.1f}%;background:#4b84d6;"></div>
                  </div>
                </div>
                """
            )
        probs_html = "\n".join(lines)

        selection_info = (
            f"Selected channel: {fmap_idx} ({channel_mode}) | "
            f"Selected kernel: {kernel_idx} ({kernel_source}) | "
            f"Conv1 channels: {c}"
        )

        return (
            Image.fromarray(in_show),
            Image.fromarray(conv_img),
            Image.fromarray(relu_img),
            Image.fromarray(pool_img),
            fig_kernel,
            fig_act,
            probs_html,
            selection_info,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    explainer = Explainer(args.ckpt)

    css = """
    .gradio-container {
      max-width: 1420px !important;
      margin: 0 auto !important;
      padding: 10px !important;
      background: repeating-linear-gradient(
        0deg,
        #d7d7d7 0px,
        #d7d7d7 2px,
        #d4d4d4 2px,
        #d4d4d4 4px
      ) !important;
    }

    .section-card {
      border: 1px solid #8e8e8e !important;
      border-radius: 8px !important;
      background: #e8e8e8 !important;
      padding: 10px !important;
    }

    .retro-title {
      background: linear-gradient(#245aa3, #103f80) !important;
      border: 1px solid #133a70 !important;
      border-radius: 6px !important;
      color: #f4f8ff !important;
      font-weight: 700 !important;
      padding: 10px 12px !important;
      margin-bottom: 10px !important;
    }

    /* Teaching panel readability fix */
    .teaching-panel {
      background: #ececec !important;
      border: 1px solid #8d8d8d !important;
      border-radius: 8px !important;
      padding: 12px !important;
      color: #1f2937 !important;
    }
    .teaching-panel * {
      color: #1f2937 !important;
    }
    .teaching-panel h2 {
      color: #0f3d7a !important;
      margin: 0 0 8px 0 !important;
    }
    .teaching-panel .warning {
      margin-top: 10px !important;
      border: 1px solid #b88a2a !important;
      background: #fff3cd !important;
      border-radius: 6px !important;
      padding: 8px 10px !important;
      color: #5b4a00 !important;
      font-weight: 600 !important;
    }

    .dark-block {
      background: #1f2229 !important;
      border-radius: 6px !important;
      padding: 8px !important;
      color: #e5e7eb !important;
    }

    img, canvas, .plot-container {
      max-width: 100% !important;
      height: auto !important;
    }

    @media (max-width: 1100px) {
      .gradio-container { padding: 8px !important; }
    }
    """

    with gr.Blocks(title="DR Layer Explorer — Retro Teaching Panel", css=css) as demo:
        gr.Markdown(
            """
<div class="teaching-panel">
  <h2>DR Layer Explorer — Retro Teaching Panel</h2>
  <p><strong>Purpose:</strong> Inspect what each stage “sees” in a CNN-like pipeline.</p>
  <p><strong>Reminder:</strong> This is an educational prototype, not a clinical diagnostic system.</p>
  <div class="warning">⚠️ Research/education use only. Not intended for clinical diagnosis.</div>
</div>
"""
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown('<div class="retro-title">Controls Panel</div>')

                input_img = gr.Image(
                    type="pil",
                    label="Input Fundus Image",
                    height=320
                )

                channel_mode = gr.Dropdown(
                    choices=["Manual", "Auto strongest", "Auto weakest", "Auto median"],
                    value="Manual",
                    label="Channel selection mode"
                )
                fmap_idx = gr.Slider(0, 63, value=0, step=1, label="Manual feature map channel")
                link_kernel = gr.Checkbox(value=True, label="Link kernel to selected channel (recommended)")
                kernel_idx = gr.Slider(0, 63, value=0, step=1, label="Manual kernel index")
                run_btn = gr.Button("Run Explain", variant="primary")

            with gr.Column(scale=3, min_width=760):
                gr.Markdown('<div class="retro-title">Feature View Panel</div>')

                with gr.Row():
                    out_in = gr.Image(label="Input (single-channel used by CNN)", type="pil")
                    out_conv = gr.Image(label="Convolution output (normalized)", type="pil")
                with gr.Row():
                    out_relu = gr.Image(label="ReLU output", type="pil")
                    out_pool = gr.Image(label=r"MaxPool \(2 \times 2\) output", type="pil")

                with gr.Row():
                    fig_kernel = gr.Plot(label="Kernel values")
                    fig_act = gr.Plot(label="Activation strengths")

                probs_html = gr.HTML(label="Dense + Softmax output (demo classes)")
                selection_info = gr.Textbox(label="Selection info", interactive=False)

        def _sync_slider_ranges(img, _mode):
            # dynamic slider max by actual conv1 channels
            c = int(explainer.first_conv.out_channels)
            max_idx = max(c - 1, 0)
            return (
                gr.update(maximum=max_idx, value=min(0, max_idx)),
                gr.update(maximum=max_idx, value=min(0, max_idx)),
            )

        channel_mode.change(
            fn=_sync_slider_ranges,
            inputs=[input_img, channel_mode],
            outputs=[fmap_idx, kernel_idx]
        )

        run_btn.click(
            fn=explainer.run,
            inputs=[input_img, channel_mode, fmap_idx, link_kernel, kernel_idx],
            outputs=[out_in, out_conv, out_relu, out_pool, fig_kernel, fig_act, probs_html, selection_info],
        )

    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )


if __name__ == "__main__":
    main()