import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AptosDataset
from transforms import get_transforms
from model import SimpleDRCNN


def make_class_weights(train_csv, num_classes=5):
    df = pd.read_csv(train_csv)
    counts = df["diagnosis"].value_counts().sort_index()
    freq = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    inv = 1.0 / freq
    w = inv / inv.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    all_preds, all_labels = [], []
    running_loss = 0.0

    for imgs, labels, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/splits/train.csv")
    parser.add_argument("--val_csv", default="data/processed/splits/val.csv")
    parser.add_argument("--image_dir", default="data/raw/train_images")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = AptosDataset(
        csv_path=args.train_csv,
        image_dir=args.image_dir,
        transform=get_transforms(args.image_size, train=True),
        has_labels=True,
    )
    val_ds = AptosDataset(
        csv_path=args.val_csv,
        image_dir=args.image_dir,
        transform=get_transforms(args.image_size, train=False),
        has_labels=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SimpleDRCNN(num_classes=5).to(device)

    class_weights = make_class_weights(args.train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                    "image_size": args.image_size,
                    "num_classes": 5,
                },
                best_path,
            )
            print(f"Saved best model to {best_path} (val macro F1={best_f1:.4f})")

    print("Training complete. Best val macro F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()