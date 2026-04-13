"""
train_baselines.py

Simple baselines to benchmark against learned models. No training needed.

  1. Majority class    : always predict "fail" (the 59% class)
  2. Random stratified : coin flip weighted by class proportions
  3. Code length       : predict pass if LOC > median LOC for that task
  4. Single feature    : threshold on classical_loc (strongest single predictor)

Run from the project root:
    python3 models/train_baselines.py

Output goes into models/outputs_baselines/
"""

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
TRAIN = Path("data/clean/splits/train_features.csv")
TEST  = Path("data/clean/splits/test_features.csv")
OUT   = Path("models/outputs_baselines")
OUT.mkdir(parents=True, exist_ok=True)

LABEL = "label"


def load(path):
    return pd.read_csv(path)


def report(name, y_true, y_pred, y_prob, fout):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
    f1  = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    txt = (
        f"\n  {name}\n"
        f"  AUC-ROC  : {auc:.4f}\n"
        f"  F1       : {f1:.4f}\n"
        f"  Accuracy : {acc:.4f}\n\n"
        + classification_report(y_true, y_pred, digits=4)
    )
    print(txt)
    fout.write(txt + "\n")
    return auc, f1, acc


def main():
    print("Computing baselines")

    train_df = load(TRAIN)
    test_df  = load(TEST)
    y_train  = train_df[LABEL].values
    y_test   = test_df[LABEL].values
    n_test   = len(y_test)

    comparison = []

    with open(OUT / "metrics.txt", "w") as fout:
        fout.write("Baseline comparisons (no learned models)\n\n")

        # 1. Majority class: always predict fail (0)
        pred = np.zeros(n_test, dtype=int)
        prob = np.full(n_test, 1 - y_train.mean())  # P(fail)
        auc, f1, acc = report("Majority class (always predict fail)", y_test, pred, prob, fout)
        comparison.append(("Majority class", auc, f1, acc))

        # 2. Random stratified: coin flip at training class proportions
        rng = np.random.default_rng(42)
        pass_rate = y_train.mean()
        prob = rng.random(n_test)
        pred = (prob < pass_rate).astype(int)
        prob_score = np.where(pred == 1, pass_rate, 1 - pass_rate)
        auc, f1, acc = report("Random stratified", y_test, pred, prob_score, fout)
        comparison.append(("Random stratified", auc, f1, acc))

        # 3. Code length: predict pass if LOC > task median (computed from train)
        task_medians = train_df.groupby("task_id")["classical_loc"].median()
        test_median = test_df["task_id"].map(task_medians)
        # For tasks not in train (shouldn't happen but just in case), use global median
        global_median = train_df["classical_loc"].median()
        test_median = test_median.fillna(global_median)
        pred = (test_df["classical_loc"].values > test_median.values).astype(int)
        prob = test_df["classical_loc"].values / test_df["classical_loc"].values.max()
        auc, f1, acc = report("Code length > task median", y_test, pred, prob, fout)
        comparison.append(("Code length threshold", auc, f1, acc))

        # 4. Single feature: best threshold on classical_loc
        loc = test_df["classical_loc"].values
        best_f1, best_thresh = 0, 0
        for pct in range(10, 91, 5):
            thresh = np.percentile(train_df["classical_loc"].values, pct)
            p = (loc > thresh).astype(int)
            score = f1_score(y_test, p)
            if score > best_f1:
                best_f1, best_thresh = score, thresh
        pred = (loc > best_thresh).astype(int)
        prob = loc / loc.max()
        auc, f1, acc = report(f"LOC threshold (>{best_thresh:.0f} lines)", y_test, pred, prob, fout)
        comparison.append((f"LOC threshold (>{best_thresh:.0f})", auc, f1, acc))

        # Comparison table
        fout.write("\n  Baseline comparison\n")
        fout.write(f"  {'Baseline':<30} {'AUC':>7}  {'F1':>6}  {'Acc':>6}\n")
        for name, auc, f1, acc in comparison:
            line = f"  {name:<30} {auc:>7.4f}  {f1:>6.4f}  {acc:>6.4f}"
            print(line)
            fout.write(line + "\n")

    print(f"\n  Saved -> models/outputs_baselines/metrics.txt")

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    loc = test_df["classical_loc"].values
    prob = loc / loc.max()
    prec, rec, _ = precision_recall_curve(y_test, prob)
    trapz = getattr(np, "trapz", None) or np.trapezoid
    ap = float(trapz(prec[::-1], rec[::-1]))
    ax.plot(rec, prec, label=f"LOC (AP={ap:.3f})", color="#534AB7", lw=1.8)
    ax.axhline(y_test.mean(), color="#888780", ls="--", lw=1, label=f"Random (AP={y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Baseline precision-recall curves", fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> models/outputs_baselines/pr_curves.png")

    print("\nDone! All outputs saved to models/outputs_baselines/")


if __name__ == "__main__":
    main()
