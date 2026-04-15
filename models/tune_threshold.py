"""
tune_threshold.py

Tunes the decision threshold for all trained models. The default threshold
of 0.5 is rarely optimal with class imbalance. This script sweeps thresholds
on the validation set to maximize F1, then evaluates on the held-out test set.

The threshold search is done entirely on the validation set. The test set is
only used once for final evaluation, preventing any data snooping.

Run from the project root:
    python3 models/tune_threshold.py

Output goes into each model's output directory:
    outputs_tfidf/threshold_metrics.txt
    outputs_crossval/threshold_metrics.txt
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Paths
SPLITS = Path("data/clean/splits")
VAL_FEAT  = SPLITS / "val_features.csv"
TEST_FEAT = SPLITS / "test_features.csv"
VAL_RAW   = SPLITS / "val.csv"
TEST_RAW  = SPLITS / "test.csv"

FEATURE_COLS = [
    "classical_loc", "classical_cyclomatic_complexity", "classical_max_nesting_depth",
    "ast_if_count", "ast_for_count", "ast_while_count", "ast_try_count",
    "ast_except_count", "ast_return_count", "ast_import_count", "ast_has_error_handling",
    "align_lib_coverage", "align_missing_libs", "align_length_ratio",
    "smell_hardcoded_return_funcs", "smell_placeholder_hits",
    "smell_is_very_short", "smell_relative_length",
]
LABEL = "label"


def find_best_threshold(y_val, val_prob):
    """Sweep thresholds on validation set only. Test set is never seen here."""
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx], thresholds, scores


def report(name, y_true, y_pred, y_prob, threshold, fout):
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    txt = (
        f"\n  {name} (threshold={threshold:.2f})\n"
        f"  AUC-ROC  : {auc:.4f}\n"
        f"  F1       : {f1:.4f}\n"
        f"  Accuracy : {acc:.4f}\n\n"
        + classification_report(y_true, y_pred, digits=4)
    )
    print(txt)
    fout.write(txt + "\n")
    return auc, f1, acc


def plot_threshold_curves(results, out_path):
    colors = ["#534AB7", "#0F6E56", "#D85A30", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax.plot(data["thresholds"], data["val_f1_scores"], label=name, color=color, lw=1.8)
        ax.axvline(data["best_threshold"], color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.scatter([data["best_threshold"]], [data["best_val_f1"]], color=color, zorder=5, s=50)
    ax.axvline(0.5, color="#888780", linestyle=":", linewidth=1, label="default (0.5)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score (validation set)")
    ax.set_title("F1 vs decision threshold (tuned on val, not test)", fontweight="normal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def tune_group(group_name, models_dict, get_probs_fn, out_dir):
    """Tune thresholds for a group of models. Returns comparison list."""
    print(f"\n  {group_name}")

    val_feat = pd.read_csv(VAL_FEAT)
    test_feat = pd.read_csv(TEST_FEAT)
    y_val = val_feat[LABEL].values
    y_test = test_feat[LABEL].values

    val_probs = {}
    test_probs = {}
    for name, model in models_dict.items():
        vp, tp = get_probs_fn(model, val_feat, test_feat)
        val_probs[name] = vp
        test_probs[name] = tp

    # Find best threshold on validation set only
    threshold_results = {}
    for name, val_prob in val_probs.items():
        best_t, best_f1, thresholds, scores = find_best_threshold(y_val, val_prob)
        default_f1 = f1_score(y_val, (val_prob >= 0.5).astype(int), zero_division=0)
        print(f"    {name}: threshold 0.50 -> {best_t:.2f} (val F1: {default_f1:.4f} -> {best_f1:.4f})")
        threshold_results[name] = {
            "best_threshold": best_t, "best_val_f1": best_f1,
            "thresholds": thresholds, "val_f1_scores": scores,
        }

    # Evaluate on test set (seen only here)
    comparison = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "threshold_metrics.txt", "w") as fout:
        fout.write(f"{group_name} -- threshold tuning\n")
        fout.write("Thresholds selected on validation set to maximize F1.\n")
        fout.write("Test set used only for final evaluation.\n")

        for name, model in models_dict.items():
            best_t = threshold_results[name]["best_threshold"]
            test_prob = test_probs[name]
            test_pred = (test_prob >= best_t).astype(int)
            auc, f1, acc = report(name, y_test, test_pred, test_prob, best_t, fout)
            comparison.append((name, auc, f1, acc, best_t))

    plot_threshold_curves(threshold_results, out_dir / "threshold_curves.png")
    print(f"    Saved -> {out_dir.name}/threshold_metrics.txt, threshold_curves.png")
    return comparison


def main():
    print("Threshold tuning (all models)")
    print("Thresholds optimized on validation set only. Test set used once for evaluation.")

    all_results = []

    # TF-IDF models
    tfidf_dir = Path("models/outputs_tfidf")
    tfidf_models = {}
    for name, fname in [("LogReg (tfidf)", "logreg_model.pkl"),
                        ("LightGBM (tfidf)", "lgbm_model.pkl")]:
        pkl = tfidf_dir / fname
        if pkl.exists():
            with open(pkl, "rb") as f:
                tfidf_models[name] = pickle.load(f)

    if tfidf_models:
        # Load TF-IDF vectorizers
        with open(tfidf_dir / "word_tfidf.pkl", "rb") as f:
            word_tfidf = pickle.load(f)
        with open(tfidf_dir / "char_tfidf.pkl", "rb") as f:
            char_tfidf = pickle.load(f)

        def tfidf_probs(model, val_feat, test_feat):
            val_raw = pd.read_csv(VAL_RAW)
            test_raw = pd.read_csv(TEST_RAW)

            def build_X(feat_df, raw_df):
                X_static = csr_matrix(feat_df[FEATURE_COLS].fillna(0).values.astype(float))
                code_col = "solution" if "solution" in raw_df.columns else "generated_code"
                texts = raw_df[code_col].fillna("").astype(str).tolist()
                return hstack([X_static, word_tfidf.transform(texts), char_tfidf.transform(texts)])

            return (model.predict_proba(build_X(val_feat, val_raw))[:, 1],
                    model.predict_proba(build_X(test_feat, test_raw))[:, 1])

        results = tune_group("TF-IDF models", tfidf_models, tfidf_probs, tfidf_dir)
        all_results.extend(results)

    # Crossval models
    crossval_dir = Path("models/outputs_crossval")
    crossval_models = {}
    for name, fname in [("LogReg (crossval)", "logreg_model.pkl"),
                        ("XGBoost (crossval)", "xgb_model.pkl")]:
        pkl = crossval_dir / fname
        if pkl.exists():
            with open(pkl, "rb") as f:
                crossval_models[name] = pickle.load(f)

    if crossval_models:
        def static_probs(model, val_feat, test_feat):
            X_va = val_feat[FEATURE_COLS].fillna(0)
            X_te = test_feat[FEATURE_COLS].fillna(0)
            return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]

        results = tune_group("Crossval models", crossval_models, static_probs, crossval_dir)
        all_results.extend(results)

    # Summary
    print(f"\n  {'Model':<25} {'AUC':>7}  {'F1':>6}  {'Acc':>6}  {'Thresh':>6}")
    for name, auc, f1, acc, t in sorted(all_results, key=lambda x: -x[1]):
        print(f"  {name:<25} {auc:>7.4f}  {f1:>6.4f}  {acc:>6.4f}  {t:>6.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
