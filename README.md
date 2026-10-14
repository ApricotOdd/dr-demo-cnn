# dr-demo-cnn

A small, learning-focused deep learning project for diabetic retinopathy image classification.

## Project Introduction

This is a small AI-assisted project created for the **Orbis Future Vision Leader Campaign 2025/26: Heal the Future** campaign.

It is intended as a practical, educational prototype:
- simple CNN baseline
- train/evaluate scripts
- lightweight explainability demo (Gradio)

## Build note

This project was developed using **AI-assisted (vibe-coded) workflow**:
- rapid prototyping with AI help
- human review, editing, and testing by the author

The repository reflects a learning-focused prototype, not a production medical system.

## Purpose

The goal is to explore how computer vision can support early screening workflows for diabetic retinopathy, while keeping the pipeline understandable and reproducible.

> This repository is for educational/prototyping purposes and is **not** a clinical diagnostic tool.

## What’s included

- `src/model.py` – baseline CNN model
- `src/transforms.py` – image transforms
- `src/dataset.py` – dataset loader
- `src/make_splits.py` – train/val split creation
- `src/train.py` – training loop
- `src/evaluate.py` – evaluation script
- `src/explain/demo_gradio.py` – local/web demo UI

## Quick start

### 1) Clone
```bash
git clone https://github.com/ApricotOdd/dr-demo-cnn.git
cd dr-demo-cnn
```

2) Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision pandas scikit-learn pillow tqdm gradio matplotlib
```

3) Prepare data
Place dataset files under:

data/raw/train.csv
data/raw/train_images/*.png

4) Run
```bash
python src/make_splits.py
python src/train.py --epochs 12 --batch_size 32
python src/evaluate.py
```

5) Launch demo
```bash
python src/explain/demo_gradio.py --ckpt outputs/best.pt --share
```

⚠️ Notes
This is a baseline prototype, not optimized for production.
Performance depends heavily on data quality, preprocessing, and class balance.
For serious deployment, use stronger models, robust validation, and clinical governance.

Acknowledgment
Built in the spirit of the Orbis Future Vision Leader Campaign 2025/26 — Heal the Future.

