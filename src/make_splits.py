import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/raw/train.csv")
    parser.add_argument("--out_dir", default="data/processed/splits")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["diagnosis"],
    )

    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("Saved:", train_path, len(train_df))
    print("Saved:", val_path, len(val_df))
    print("Train class counts:\n", train_df["diagnosis"].value_counts().sort_index())
    print("Val class counts:\n", val_df["diagnosis"].value_counts().sort_index())


if __name__ == "__main__":
    main()