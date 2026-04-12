import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_task_id(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split the dataset by task_id to avoid task leakage across splits.
    Each task_id will appear in exactly one of train / val / test.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    if "task_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'task_id' column.")

    task_ids = df["task_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(task_ids)

    n_tasks = len(task_ids)
    n_train = int(n_tasks * train_ratio)
    n_val = int(n_tasks * val_ratio)
    n_test = n_tasks - n_train - n_val

    train_tasks = set(task_ids[:n_train])
    val_tasks = set(task_ids[n_train : n_train + n_val])
    test_tasks = set(task_ids[n_train + n_val :])

    train_df = df[df["task_id"].isin(train_tasks)].copy()
    val_df = df[df["task_id"].isin(val_tasks)].copy()
    test_df = df[df["task_id"].isin(test_tasks)].copy()

    return train_df, val_df, test_df


def print_split_summary(name: str, split_df: pd.DataFrame):
    """
    Print basic statistics for one split.
    """
    n_rows = len(split_df)
    n_tasks = split_df["task_id"].nunique() if "task_id" in split_df.columns else 0

    if "label" in split_df.columns and n_rows > 0:
        pass_rate = split_df["label"].mean()
        print(
            f"{name:<6} | rows = {n_rows:>7,} | tasks = {n_tasks:>5,} | pass rate = {pass_rate:.2%}"
        )
    else:
        print(f"{name:<6} | rows = {n_rows:>7,} | tasks = {n_tasks:>5,}")


def main():
    parser = argparse.ArgumentParser(
        description="Split samples.csv into train/val/test by task_id."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--outdir", type=str, default=".", help="Directory to save split CSV files"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test split ratio"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows.")

    train_df, val_df, test_df = split_by_task_id(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved splits:")
    print(f"  Train -> {train_path}")
    print(f"  Val   -> {val_path}")
    print(f"  Test  -> {test_path}")

    print("\nSplit summary:")
    print_split_summary("train", train_df)
    print_split_summary("val", val_df)
    print_split_summary("test", test_df)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    print(f"\nTotal rows check: {total_rows:,}")

    overlap_train_val = set(train_df["task_id"]).intersection(set(val_df["task_id"]))
    overlap_train_test = set(train_df["task_id"]).intersection(set(test_df["task_id"]))
    overlap_val_test = set(val_df["task_id"]).intersection(set(test_df["task_id"]))

    print("\nTask overlap check:")
    print(f"  train ∩ val  = {len(overlap_train_val)}")
    print(f"  train ∩ test = {len(overlap_train_test)}")
    print(f"  val ∩ test   = {len(overlap_val_test)}")


if __name__ == "__main__":
    main()
