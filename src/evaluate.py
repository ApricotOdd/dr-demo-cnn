import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

from dataset import AptosDataset
from transforms import get_transforms
from model import SimpleDRCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--val_csv", default="data/processed/splits/val.csv")
    parser.add_argument("--image_dir", default="data/raw/train_images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)

    image_size = ckpt.get("image_size", 224)
    model = SimpleDRCNN(num_classes=ckpt.get("num_classes", 5)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = AptosDataset(
        csv_path=args.val_csv,
        image_dir=args.image_dir,
        transform=get_transforms(image_size=image_size, train=False),
        has_labels=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    preds, labels = [], []
    with torch.no_grad():
        for imgs, y, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(y.numpy().tolist())

    print("Accuracy:", round(accuracy_score(labels, preds), 4))
    print("Macro F1:", round(f1_score(labels, preds, average="macro"), 4))
    print("\nClassification report:\n")
    print(classification_report(labels, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()