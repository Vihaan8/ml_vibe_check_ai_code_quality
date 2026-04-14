"""
main.py

Vibe Check pipeline orchestrator. Runs the full pipeline end-to-end or
individual stages:
  1. Data collection and preprocessing
  2. Feature engineering
  3. Model training and evaluation
  4. Threshold tuning

By default, only model training runs (stage 3). Verbose subprocess output
is suppressed unless --verbose is passed.

Usage:
  python main.py                             # train all models (quiet)
  python main.py --verbose                   # train all models (full output)
  python main.py --all                       # run everything from scratch
  python main.py --features                  # run only feature extraction
  python main.py --models baseline           # train only baseline models
  python main.py --models crossval threshold # train crossval then tune thresholds
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paths used by the pipeline
SAMPLES_CSV = ROOT / "data" / "clean" / "samples.csv"
SPLITS_DIR  = ROOT / "data" / "clean" / "splits"
TRAIN_FEAT  = SPLITS_DIR / "train_features.csv"

VERBOSE = False


def run(cmd, description):
    """Run a command, print status, suppress output unless verbose."""
    print(f"  [{description}] ... ", end="", flush=True)
    if VERBOSE:
        print()
        result = subprocess.run(cmd, cwd=ROOT)
    else:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        if not VERBOSE and result.stderr:
            print(result.stderr)
        if not VERBOSE and result.stdout:
            lines = result.stdout.strip().split("\n")
            print("\n".join(lines[-20:]))
        sys.exit(result.returncode)
    else:
        if not VERBOSE:
            print("done")


def stage_preprocess():
    run(
        [sys.executable, "data/preprocessing/collect_data.py"],
        "Collecting and processing raw data",
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "data/preprocessing/split_data.py",
         "--input", str(SAMPLES_CSV),
         "--outdir", str(SPLITS_DIR)],
        "Splitting data by task_id",
    )


def stage_features():
    train_csv = SPLITS_DIR / "train.csv"
    if not train_csv.exists():
        if not SAMPLES_CSV.exists():
            print(f"Error: {SAMPLES_CSV} not found. Run with --preprocess first.")
            sys.exit(1)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        run(
            [sys.executable, "data/preprocessing/split_data.py",
             "--input", str(SAMPLES_CSV),
             "--outdir", str(SPLITS_DIR)],
            "Splitting data by task_id",
        )

    for split in ["train", "val", "test"]:
        input_csv  = SPLITS_DIR / f"{split}.csv"
        output_csv = SPLITS_DIR / f"{split}_features.csv"
        run(
            [sys.executable, "feature_engineering/run_feature_extraction.py",
             "--input", str(input_csv),
             "--out", str(output_csv)],
            f"Feature extraction ({split})",
        )


def stage_models(versions):
    if isinstance(versions, str):
        versions = [versions]

    run_all = "all" in versions

    if run_all or "baselines" in versions:
        run([sys.executable, "models/train_baselines.py"], "Baselines")
    if run_all or "baseline" in versions:
        run([sys.executable, "models/train_baseline.py"], "Baseline (static features)")
    if run_all or "tfidf" in versions:
        run([sys.executable, "models/train_tfidf.py"], "TF-IDF (static + code text)")
    if run_all or "crossval" in versions:
        run([sys.executable, "models/train_crossval.py"], "Cross-validation (GroupKFold)")
    if run_all or "threshold" in versions:
        run([sys.executable, "models/tune_threshold.py"], "Threshold tuning")


def print_summary():
    import re

    print("\nResults summary")
    print(f"  {'Approach':<40} {'AUC-ROC':>8}  {'F1':>6}")

    metrics_files = [
        ("outputs_baselines", "Baselines"),
        ("outputs_baseline",  "Baseline (static)"),
        ("outputs_tfidf",     "TF-IDF (static + code text)"),
        ("outputs_crossval",  "Cross-validation (GroupKFold)"),
    ]

    for folder, label in metrics_files:
        path = ROOT / "models" / folder / "metrics.txt"
        if not path.exists():
            continue
        text = path.read_text()
        aucs = re.findall(r"AUC-ROC\s*[:\|]\s*([\d.]+)", text)
        f1s  = re.findall(r"F1\s*[:\|]\s*([\d.]+)", text)
        if aucs and f1s:
            print(f"  {label:<40} {max(float(a) for a in aucs):>8.4f}  {max(float(f) for f in f1s):>6.4f}")

    # Threshold-tuned results
    for folder, label in [("outputs_tfidf", "TF-IDF + threshold tuning"),
                          ("outputs_crossval", "Crossval + threshold tuning")]:
        path = ROOT / "models" / folder / "threshold_metrics.txt"
        if not path.exists():
            continue
        text = path.read_text()
        aucs = re.findall(r"AUC-ROC\s*[:\|]\s*([\d.]+)", text)
        f1s  = re.findall(r"F1\s*[:\|]\s*([\d.]+)", text)
        if aucs and f1s:
            print(f"  {label:<40} {max(float(a) for a in aucs):>8.4f}  {max(float(f) for f in f1s):>6.4f}")

    print()


def check_prerequisites(stage):
    if stage == "features" and not SAMPLES_CSV.exists():
        print(f"Error: {SAMPLES_CSV} not found. Run with --preprocess first.")
        sys.exit(1)
    if stage == "models" and not TRAIN_FEAT.exists():
        print(f"Error: {TRAIN_FEAT} not found. Run with --features first.")
        sys.exit(1)


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="Vibe Check: run the defect prediction pipeline",
    )
    parser.add_argument("--all", action="store_true",
        help="Run the full pipeline: preprocess -> features -> models -> threshold")
    parser.add_argument("--preprocess", action="store_true",
        help="Run data collection, processing, and splitting")
    parser.add_argument("--features", action="store_true",
        help="Run feature extraction on the split CSVs")
    parser.add_argument("--models", nargs="+", default=None,
        choices=["all", "baselines", "baseline", "tfidf", "crossval", "threshold"],
        help="Train models: all, baselines, baseline, tfidf, crossval, threshold")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Show full subprocess output")
    args = parser.parse_args()

    VERBOSE = args.verbose

    nothing_selected = not (args.all or args.preprocess or args.features or args.models)
    if nothing_selected:
        args.models = ["all"]

    if args.all:
        stage_preprocess()
        stage_features()
        stage_models(["all", "threshold"])
        print_summary()
        return

    if args.preprocess:
        stage_preprocess()

    if args.features:
        check_prerequisites("features")
        stage_features()

    if args.models:
        check_prerequisites("models")
        stage_models(args.models)
        print_summary()

    print("Done.")


if __name__ == "__main__":
    main()
