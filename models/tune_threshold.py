"""
tune_threshold.py

Tunes the decision threshold for the XGBoost model from train_crossval.py.
The default threshold of 0.5 gives collapsed F1 (0.356) despite decent AUC (0.629).
This script finds the threshold that maximizes F1 on the val set, then
re-evaluates all three crossval models on the test set using their optimal thresholds.

Run from the project root:
    python3 models/tune_threshold.py

Output goes into models/outputs_crossval/
  threshold_metrics.txt   <- updated metrics with tuned thresholds
  threshold_curves.png    <- F1 vs threshold plot for each model
  pr_curves_tuned.png     <- updated PR curve
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Paths
VAL  = Path("data/clean/splits/val_features.csv")
TEST = Path("data/clean/splits/test_features.csv")
OUT  = Path("models/outputs_crossval")

FEATURE_COLS = [
    "classical_loc",
    "classical_cyclomatic_complexity",
    "classical_max_nesting_depth",
    "ast_if_count",
    "ast_for_count",
    "ast_while_count",
    "ast_try_count",
    "ast_except_count",
    "ast_return_count",
    "ast_import_count",
    "ast_has_error_handling",
    "align_lib_coverage",
    "align_missing_libs",
    "align_length_ratio",
    "smell_hardcoded_return_funcs",
    "smell_placeholder_hits",
    "smell_is_very_short",
    "smell_relative_length",
]
LABEL = "label"


# ── Load data ──────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    X  = df[FEATURE_COLS].copy()
    y  = df[LABEL].astype(int).values
    return X, y, df


# ── Find best threshold on val set ────────────────────────

def find_best_threshold(y_val, val_prob, metric="f1"):
    """
    Sweep thresholds from 0.1 to 0.9 and return the one
    that maximizes F1 on the validation set.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores     = []
    for t in thresholds:
        pred  = (val_prob >= t).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx], thresholds, scores


# ── Report ─────────────────────────────────────────────────

def report(name, y_true, y_pred, y_prob, threshold, fout):
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    txt = (
        f"\n{'='*55}\n"
        f"  {name}\n"
        f"  Threshold : {threshold:.2f}\n"
        f"{'='*55}\n"
        f"  AUC-ROC  : {auc:.4f}\n"
        f"  F1       : {f1:.4f}\n"
        f"  Accuracy : {acc:.4f}\n\n"
        + classification_report(y_true, y_pred, digits=4)
    )
    print(txt)
    fout.write(txt + "\n")
    return auc, f1


# ── Plots ──────────────────────────────────────────────────

def plot_threshold_curves(results: dict):
    """Plot F1 vs threshold for each model, marking the best threshold."""
    colors = {
        "Logistic Regression": "#534AB7",
        "XGBoost":             "#0F6E56",
        "Random Forest":       "#D85A30",
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, data in results.items():
        thresholds = data["thresholds"]
        scores     = data["val_f1_scores"]
        best_t     = data["best_threshold"]
        best_f1    = data["best_val_f1"]
        color      = colors.get(name, "#888780")

        ax.plot(thresholds, scores, label=name, color=color, lw=1.8)
        ax.axvline(best_t, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.scatter([best_t], [best_f1], color=color, zorder=5, s=50)

    ax.axvline(0.5, color="#888780", linestyle=":", linewidth=1, label="default (0.5)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score (val set)")
    ax.set_title("F1 vs decision threshold", fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved -> models/outputs_crossval/threshold_curves.png")


def plot_pr_curves(probs_dict: dict, y_test):
    colors = {
        "Logistic Regression": "#534AB7",
        "XGBoost":             "#0F6E56",
        "Random Forest":       "#D85A30",
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob in probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = float((getattr(np, "trapz", None) or np.trapezoid)(prec[::-1], rec[::-1]))
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=colors.get(name, "#888780"), lw=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve — tuned thresholds (test set)", fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves_tuned.png", dpi=150)
    plt.close(fig)
    print("  Saved -> models/outputs_crossval/pr_curves_tuned.png")


# ── Main ───────────────────────────────────────────────────

def main():
    print("="*55)
    print("  Threshold tuning for crossval models")
    print("="*55)

    # Load data
    X_val,  y_val,  _       = load(VAL)
    X_test, y_test, df_test = load(TEST)
    print(f"\n  Val={len(y_val):,}  Test={len(y_test):,}")

    # Load saved models
    models = {}
    for key, fname in [
        ("Logistic Regression", "logreg_model.pkl"),
        ("XGBoost",             "xgb_model.pkl"),
        ("Random Forest",       "rf_model.pkl"),
    ]:
        pkl_path = OUT / fname
        if not pkl_path.exists():
            print(f"  WARNING: {fname} not found, skipping {key}")
            continue
        with open(pkl_path, "rb") as f:
            models[key] = pickle.load(f)
        print(f"  Loaded {fname}")

    if not models:
        print("\nERROR: No models found. Run train_crossval.py first.")
        return

    # Get val + test probabilities
    val_probs  = {name: m.predict_proba(X_val)[:, 1]  for name, m in models.items()}
    test_probs = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}

    # Find best threshold for each model on val set
    print("\n── Threshold search (val set) ───────────────────────")
    threshold_results = {}
    for name, val_prob in val_probs.items():
        best_t, best_f1, thresholds, scores = find_best_threshold(y_val, val_prob)
        default_f1 = f1_score(y_val, (val_prob >= 0.5).astype(int), zero_division=0)
        print(f"\n  {name}")
        print(f"    Default threshold (0.50) → val F1 = {default_f1:.4f}")
        print(f"    Best threshold   ({best_t:.2f}) → val F1 = {best_f1:.4f}  (+{best_f1 - default_f1:.4f})")
        threshold_results[name] = {
            "best_threshold": best_t,
            "best_val_f1":    best_f1,
            "thresholds":     thresholds,
            "val_f1_scores":  scores,
        }

    # Test-set evaluation with tuned thresholds
    print("\n── Test-set results (tuned thresholds) ──────────────")
    comparison = []
    with open(OUT / "threshold_metrics.txt", "w") as fout:
        fout.write("Crossval models — tuned decision thresholds\n")
        fout.write("Threshold selected to maximize F1 on val set\n")

        for name, model in models.items():
            best_t    = threshold_results[name]["best_threshold"]
            test_prob = test_probs[name]
            test_pred = (test_prob >= best_t).astype(int)
            auc, f1   = report(name, y_test, test_pred, test_prob, best_t, fout)
            comparison.append((name, auc, f1, best_t))

        # Summary table
        header = f"\n{'='*55}\n  Summary\n{'='*55}\n"
        header += f"  {'Model':<25} {'Threshold':>9}  {'AUC':>6}  {'F1':>6}\n"
        header += f"  {'-'*50}\n"
        for name, auc, f1, t in sorted(comparison, key=lambda x: -x[1]):
            header += f"  {name:<25} {t:>9.2f}  {auc:>6.4f}  {f1:>6.4f}\n"
        print(header)
        fout.write(header)

    print("  Saved -> models/outputs_crossval/threshold_metrics.txt")

    # Save updated predictions
    df_out = df_test.copy()
    for name, test_prob in test_probs.items():
        best_t    = threshold_results[name]["best_threshold"]
        test_pred = (test_prob >= best_t).astype(int)
        col       = name.lower().replace(" ", "_")
        df_out[f"{col}_prob"]      = test_prob
        df_out[f"{col}_pred"]      = test_pred
        df_out[f"{col}_threshold"] = best_t
    df_out.to_csv(OUT / "results_tuned.csv", index=False)
    print("  Saved -> models/outputs_crossval/results_tuned.csv")

    # Plots
    print("\n── Generating plots ─────────────────────────────────")
    plot_threshold_curves(threshold_results)
    plot_pr_curves(test_probs, y_test)

    print("\nDone!")


if __name__ == "__main__":
    main()
