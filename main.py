"""
main.py
 
Vibe Check pipeline orchestrator. Runs the full pipeline end-to-end or
individual stages:
  1. Data collection and preprocessing
  2. Feature engineering
  3. Model training and evaluation
 
By default, only model training runs (stage 3). Verbose subprocess output
is suppressed unless --verbose is passed.
 
Usage:
  python main.py                             # train all models (quiet)
  python main.py --verbose                   # train all models (full output)
  python main.py --all                       # run everything from scratch
  python main.py --features                  # run only feature extraction
  python main.py --models baseline           # train only baseline models
  python main.py --models baselines          # compute baselines only
  python main.py --models crossval           # train crossval models
  python main.py --models threshold          # tune thresholds on crossval models
  python main.py --models crossval threshold # train crossval then tune thresholds
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
CROSSVAL_PKL  = ROOT / "models" / "outputs_crossval" / "xgb_model.pkl"
 
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
            # Print last 20 lines of stdout for context
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
    """
    versions is a list of model stages to run, e.g. ["crossval", "threshold"].
    Handles ordering so threshold always runs after crossval if both are requested.
    """
    # Normalize to list
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
        # Check that crossval models exist before trying to tune thresholds
        if not CROSSVAL_PKL.exists():
            print(
                "\n  Warning: crossval models not found. "
                "Run --models crossval first, or use --models crossval threshold "
                "to train and tune in one command."
            )
            sys.exit(1)
        run([sys.executable, "models/tune_threshold.py"], "Threshold tuning (crossval models)")
 
 
def print_summary():
    """Print a compact summary from metrics files."""
    print("\nResults summary")
    print(f"  {'Approach':<40} {'AUC-ROC':>8}  {'F1':>6}")
 
    metrics_files = [
        ("outputs_baselines",  "Baselines"),
        ("outputs_baseline",   "Baseline (static)"),
        ("outputs_tfidf",      "TF-IDF (static + code text)"),
        ("outputs_crossval",   "Cross-validation (GroupKFold)"),
    ]
 
    for folder, label in metrics_files:
        path = ROOT / "models" / folder / "metrics.txt"
        if not path.exists():
            continue
        text = path.read_text()
 
        import re
        aucs = re.findall(r"AUC-ROC\s*[:\|]\s*([\d.]+)", text)
        f1s  = re.findall(r"F1\s*[:\|]\s*([\d.]+)", text)
 
        if aucs and f1s:
            best_auc = max(float(a) for a in aucs)
            best_f1  = max(float(f) for f in f1s)
            print(f"  {label:<40} {best_auc:>8.4f}  {best_f1:>6.4f}")
 
    # Also show threshold-tuned results if they exist
    threshold_path = ROOT / "models" / "outputs_crossval" / "threshold_metrics.txt"
    if threshold_path.exists():
        text = threshold_path.read_text()
        import re
        aucs = re.findall(r"AUC-ROC\s*[:\|]\s*([\d.]+)", text)
        f1s  = re.findall(r"F1\s*[:\|]\s*([\d.]+)", text)
        if aucs and f1s:
            best_auc = max(float(a) for a in aucs)
            best_f1  = max(float(f) for f in f1s)
            print(f"  {'Crossval + threshold tuning':<40} {best_auc:>8.4f}  {best_f1:>6.4f}")
 
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
        help="Run the full pipeline: preprocess -> features -> models -> threshold tuning")
    parser.add_argument("--preprocess", action="store_true",
        help="Run data collection, processing, and splitting")
    parser.add_argument("--features", action="store_true",
        help="Run feature extraction on the split CSVs")
    parser.add_argument("--models", nargs="+", default=None,
        choices=["all", "baselines", "baseline", "tfidf", "crossval", "threshold"],
        help=(
            "Train models. Pass one or more options:\n"
            "  all        run every model stage (default)\n"
            "  baselines  majority class / random / LOC threshold\n"
            "  baseline   static features, val-set tuning\n"
            "  tfidf      static + TF-IDF, val-set tuning\n"
            "  crossval   static features, GroupKFold CV\n"
            "  threshold  tune decision thresholds on crossval models\n"
            "\nExamples:\n"
            "  python main.py --models crossval\n"
            "  python main.py --models crossval threshold\n"
            "  python main.py --models threshold\n"
        ),
    )
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
