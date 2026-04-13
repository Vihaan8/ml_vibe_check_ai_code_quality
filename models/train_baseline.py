"""
train_baseline.py

Defect prediction with static features only. Trains Logistic Regression
and LightGBM on the 17 static features from feature extraction.

Run from the project root:
    python3 models/train_baseline.py

Output goes into models/outputs_baseline/
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Paths
TRAIN = Path("data/clean/splits/train_features.csv")
VAL   = Path("data/clean/splits/val_features.csv")
TEST  = Path("data/clean/splits/test_features.csv")
OUT   = Path("models/outputs_baseline")
OUT.mkdir(parents=True, exist_ok=True)

# Feature columns (must match feature extraction output)
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
    "smell_is_very_short",
    "meta_parse_error",
]
LABEL = "label"


def load(path):
    df = pd.read_csv(path)
    X  = df[FEATURE_COLS].fillna(0).values.astype(float)
    y  = df[LABEL].values.astype(int)
    return X, y, df


def report(name, y_true, y_pred, y_prob, fout):
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred)
    txt = (
        f"\n{'='*55}\n  {name}\n{'='*55}\n"
        f"  AUC-ROC : {auc:.4f}\n"
        f"  F1      : {f1:.4f}\n\n"
        + classification_report(y_true, y_pred, digits=4)
    )
    print(txt)
    fout.write(txt + "\n")


# Logistic Regression
def train_logreg(X_tr, y_tr, X_va, y_va):
    print("\nLogistic Regression")
    best_auc, best_model = -1, None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=C, class_weight="balanced",
                max_iter=1000, solver="lbfgs", random_state=42,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
        print(f"  C={C:<8}  val AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_model = auc, pipe
    print(f"  Best val AUC={best_auc:.4f}")
    return best_model


def plot_logreg(model):
    scaler = model.named_steps["scaler"]
    clf    = model.named_steps["clf"]
    coefs  = clf.coef_[0] * scaler.scale_
    idx    = np.argsort(np.abs(coefs))
    colors = ["#1D9E75" if v > 0 else "#D85A30" for v in coefs[idx]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([FEATURE_COLS[i] for i in idx], coefs[idx], color=colors, height=0.6)
    ax.axvline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("Coefficient (scaled by feature std)")
    ax.set_title("Logistic Regression — feature weights", fontweight="normal")
    fig.tight_layout()
    fig.savefig(OUT / "logreg_coefs.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> models/outputs_baseline/logreg_coefs.png")


# LightGBM
def train_lgbm(X_tr, y_tr, X_va, y_va):
    print("\nLightGBM")
    neg, pos  = np.bincount(y_tr)
    scale_pos = neg / pos
    print(f"  Class imbalance -> scale_pos_weight={scale_pos:.2f}")

    best_auc, best_model = -1, None
    for n in [200, 500]:
        for lr in [0.05, 0.1]:
            for d in [4, 7]:
                m = lgb.LGBMClassifier(
                    n_estimators=n, learning_rate=lr, max_depth=d,
                    scale_pos_weight=scale_pos, subsample=0.8,
                    colsample_bytree=0.8, min_child_samples=20,
                    random_state=42, verbose=-1,
                )
                m.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
                auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
                print(f"  n={n} lr={lr} depth={d}  val AUC={auc:.4f}")
                if auc > best_auc:
                    best_auc, best_model = auc, m
    print(f"  Best val AUC={best_auc:.4f}")
    return best_model


def plot_shap(model, X_tr):
    print("  Computing SHAP values (may take a moment) ...")
    explainer   = shap.TreeExplainer(model)
    sv          = explainer.shap_values(X_tr[:2000])
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    idx  = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([FEATURE_COLS[i] for i in idx], mean_abs[idx], color="#1D9E75", height=0.6)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("LightGBM — SHAP feature importance", fontweight="normal")
    fig.tight_layout()
    fig.savefig(OUT / "lgbm_shap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> models/outputs_baseline/lgbm_shap.png")


def main():
    print("Loading feature files ...")
    X_tr, y_tr, _       = load(TRAIN)
    X_va, y_va, _       = load(VAL)
    X_te, y_te, df_test = load(TEST)
    print(f"  Train={len(X_tr):,}  Val={len(X_va):,}  Test={len(X_te):,}")

    # Train
    logreg = train_logreg(X_tr, y_tr, X_va, y_va)
    lgbm   = train_lgbm(X_tr, y_tr, X_va, y_va)

    # Evaluate on test set
    print("\nTest-set results")
    with open(OUT / "metrics.txt", "w") as fout:
        lr_prob  = logreg.predict_proba(X_te)[:, 1]
        lr_pred  = logreg.predict(X_te)
        report("Logistic Regression", y_te, lr_pred, lr_prob, fout)

        lgbm_prob = lgbm.predict_proba(X_te)[:, 1]
        lgbm_pred = lgbm.predict(X_te)
        report("LightGBM", y_te, lgbm_pred, lgbm_prob, fout)

    print(f"  Saved -> models/outputs_baseline/metrics.txt")

    # Save predictions
    df_test["logreg_prob"] = lr_prob
    df_test["logreg_pred"] = lr_pred
    df_test["lgbm_prob"]   = lgbm_prob
    df_test["lgbm_pred"]   = lgbm_pred
    df_test.to_csv(OUT / "results.csv", index=False)
    print(f"  Saved -> models/outputs_baseline/results.csv")

    # Save model files
    with open(OUT / "logreg_model.pkl", "wb") as f:
        pickle.dump(logreg, f)
    with open(OUT / "lgbm_model.pkl", "wb") as f:
        pickle.dump(lgbm, f)
    print(f"  Saved -> models/outputs_baseline/logreg_model.pkl")
    print(f"  Saved -> models/outputs_baseline/lgbm_model.pkl")

    # Plots
    print("\nGenerating plots")
    plot_logreg(logreg)
    plot_shap(lgbm, X_tr)

    # PR curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob, color in [
        ("Logistic Regression", lr_prob,   "#534AB7"),
        ("LightGBM",            lgbm_prob, "#0F6E56"),
    ]:
        prec, rec, _ = precision_recall_curve(y_te, prob)
        ap = float(np.trapz(prec[::-1], rec[::-1]))
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)", fontweight="normal")
    ax.legend()
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> models/outputs_baseline/pr_curves.png")

    print("\nDone! All outputs saved to models/outputs_baseline/")


if __name__ == "__main__":
    main()
