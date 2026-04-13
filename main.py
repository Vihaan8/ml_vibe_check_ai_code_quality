"""
main.py — Vibe Check pipeline orchestrator

Runs the full pipeline end-to-end or individual stages:
  1. Data collection and preprocessing
  2. Feature engineering
  3. Model training and evaluation

By default, only model training runs (stage 3). Preprocessing and feature
engineering are skipped unless explicitly requested, since they only need
to run once and the outputs are cached as CSVs.

Usage
-----
  python main.py                        # train all models (default)
  python main.py --all                  # run everything from scratch
  python main.py --preprocess           # run only data collection + splitting
  python main.py --features             # run only feature extraction
  python main.py --models v1            # train only v1 models
  python main.py --models v2            # train only v2 models
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paths used by the pipeline
SAMPLES_CSV   = ROOT / "data" / "clean" / "samples.csv"
SPLITS_DIR    = ROOT / "data" / "clean" / "splits"
TRAIN_FEAT    = SPLITS_DIR / "train_features.csv"


def run(cmd, description):
    """Run a command, print what's happening, and exit on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFailed: {description}")
        sys.exit(result.returncode)


def stage_preprocess():
    """Stage 1: Download raw data, build samples.csv, split into train/val/test."""
    run(
        [sys.executable, "data/preprocessing/collect_data.py"],
        "Collecting and processing raw data",
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [sys.executable, "data/preprocessing/split_data.py",
         "--input", str(SAMPLES_CSV),
         "--outdir", str(SPLITS_DIR)],
        "Splitting data into train / val / test by task_id",
    )


def stage_features():
    """Stage 2: Run feature extraction on each split."""
    for split in ["train", "val", "test"]:
        input_csv  = SPLITS_DIR / f"{split}.csv"
        output_csv = SPLITS_DIR / f"{split}_features.csv"
        run(
            [sys.executable, "feature_engineering/run_feature_extraction.py",
             "--input", str(input_csv),
             "--out", str(output_csv)],
            f"Extracting features for {split} split",
        )


def stage_models(version="all"):
    """Stage 3: Train and evaluate models."""
    if version in ("all", "v1"):
        run(
            [sys.executable, "models/train_models_v1.py"],
            "Training v1 models (static features only: LogReg + LightGBM)",
        )
    if version in ("all", "v2"):
        run(
            [sys.executable, "models/train_models_v2.py"],
            "Training v2 models (static + TF-IDF: LogReg + LightGBM + Random Forest)",
        )


def check_prerequisites(stage):
    """Verify that upstream outputs exist before running a stage."""
    if stage == "features" and not SAMPLES_CSV.exists():
        print(f"Error: {SAMPLES_CSV} not found.")
        print("Run with --preprocess first (or --all).")
        sys.exit(1)
    if stage == "models" and not TRAIN_FEAT.exists():
        print(f"Error: {TRAIN_FEAT} not found.")
        print("Run with --features first (or --all).")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Vibe Check: run the defect prediction pipeline",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run the full pipeline: preprocess -> features -> models",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Run data collection, processing, and splitting",
    )
    parser.add_argument(
        "--features", action="store_true",
        help="Run feature extraction on the split CSVs",
    )
    parser.add_argument(
        "--models", nargs="?", const="all", default=None,
        choices=["all", "v1", "v2"],
        help="Train models. Options: all (default), v1, v2",
    )
    args = parser.parse_args()

    # Default behavior: if no flags given, just train all models
    nothing_selected = not (args.all or args.preprocess or args.features or args.models)
    if nothing_selected:
        args.models = "all"

    if args.all:
        stage_preprocess()
        stage_features()
        stage_models("all")
        return

    if args.preprocess:
        stage_preprocess()

    if args.features:
        check_prerequisites("features")
        stage_features()

    if args.models:
        check_prerequisites("models")
        stage_models(args.models)

    print("\nDone.")


if __name__ == "__main__":
    main()
