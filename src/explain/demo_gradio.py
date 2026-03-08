import argparse
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import SimpleDRCNN
from transforms import get_transforms


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def norm_map(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


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

    def run(self, pil_img, fmap_idx=0, kernel_idx=0):
        if pil_img is None:
            return None, None, None, None, None, None, None

        x = self.tf(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        inp = x[0, 0].detach().cpu().numpy()
        conv = self.activations["conv1"][0].numpy()
        relu = self.activations["relu1"][0].numpy()
        pool = self.activations["pool1"][0].numpy()

        fmap_idx = int(fmap_idx) % conv.shape[0]
        kernel_idx = int(kernel_idx) % self.model.conv1.weight.shape[0]

        conv_map = norm_map(conv[fmap_idx])
        relu_map = norm_map(relu[fmap_idx])
        pool_map = norm_map(pool[fmap_idx])
        input_map = norm_map(inp)

        kernel = self.model.conv1.weight[kernel_idx, 0].detach().cpu().numpy()
        kernel_map = norm_map(kernel)

        strength = np.mean(np.abs(conv), axis=(1, 2))
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(np.arange(len(strength)), strength)
        ax.set_title("Conv1 channel activation strength (mean abs activation)")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Strength")
        fig.tight_layout()

        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        return input_map, conv_map, relu_map, pool_map, kernel_map, fig, prob_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    explainer = Explainer(args.ckpt)

    def infer(img, fmap_idx, kernel_idx):
        return explainer.run(img, fmap_idx, kernel_idx)

    with gr.Blocks(title="DR CNN Layer Explorer") as demo:
        gr.Markdown("## Diabetic Retinopathy CNN Layer Explorer")
        gr.Markdown("Upload a fundus image and inspect what early CNN layers are passing forward.")

        with gr.Row():
            inp = gr.Image(type="pil", label="Fundus Image")
            with gr.Column():
                fmap_idx = gr.Slider(0, 15, value=0, step=1, label="Feature map channel (Conv/ReLU/Pool)")
                kernel_idx = gr.Slider(0, 15, value=0, step=1, label="Kernel index (Conv1)")
                btn = gr.Button("Run Explain")

        with gr.Row():
            out_input = gr.Image(label="Input (single-channel used by CNN)")
            out_conv = gr.Image(label="Convolution output (normalized)")
            out_relu = gr.Image(label="ReLU output")
            out_pool = gr.Image(label="MaxPool output")

        with gr.Row():
            out_kernel = gr.Image(label="Selected Conv1 Kernel Values")
            out_bar = gr.Plot(label="Activation Strengths")

        out_probs = gr.Label(label="Dense + Softmax output (demo classes)")

        btn.click(
            infer,
            inputs=[inp, fmap_idx, kernel_idx],
            outputs=[out_input, out_conv, out_relu, out_pool, out_kernel, out_bar, out_probs],
        )

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()