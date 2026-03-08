import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AptosDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, has_labels=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.has_labels = has_labels and ("diagnosis" in self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id_code"]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.has_labels:
            label = int(row["diagnosis"])
            return img, label, image_id
        return img, image_id