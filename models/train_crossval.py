"""
train_crossval.py

Defect prediction with static features and StratifiedGroupKFold cross-validation.
Trains Logistic Regression and XGBoost on the 17 static features,
using 5-fold CV grouped by task_id for hyperparameter tuning.

The final model (Logistic Regression) is retrained on train+val before test evaluation.

Run from the project root:
    python3 models/train_crossval.py

Output goes into models/outputs_crossval/
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Paths
TRAIN = Path("data/clean/splits/train_features.csv")
VAL   = Path("data/clean/splits/val_features.csv")
TEST  = Path("data/clean/splits/test_features.csv")
OUT   = Path("models/outputs_crossval")
OUT.mkdir(parents=True, exist_ok=True)

# Feature columns
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


def load(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].copy()
    y = df[LABEL].astype(int)
    groups = df["task_id"] if "task_id" in df.columns else None
    return X, y, groups, df


def report(name, y_true, y_pred, y_prob, fout):
    auc = roc_auc_score(y_true, y_prob)
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
    return auc, f1


# Logistic Regression with GridSearchCV
def tune_logreg(X_train, y_train, groups_train):
    print("\nLogistic Regression (GridSearchCV, 5-fold StratifiedGroupKFold)")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42,
        )),
    ])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        },
        scoring="roc_auc",
        cv=sgkf,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train, groups=groups_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


# XGBoost with RandomizedSearchCV
def tune_xgboost(X_train, y_train, groups_train):
    print("\nXGBoost (RandomizedSearchCV, 5-fold StratifiedGroupKFold, 20 iterations)")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "model__n_estimators": randint(100, 401),
            "model__max_depth": randint(2, 6),
            "model__learning_rate": uniform(0.01, 0.09),
            "model__subsample": uniform(0.6, 0.3),
            "model__colsample_bytree": uniform(0.6, 0.3),
            "model__min_child_weight": randint(1, 8),
            "model__gamma": uniform(0, 3),
            "model__reg_lambda": uniform(1, 5),
            "model__reg_alpha": uniform(0, 2),
        },
        n_iter=20,
        scoring="roc_auc",
        cv=sgkf,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train, groups=groups_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def plot_pr_curves(probs, y_test):
    colors = {
        "Logistic Regression": "#534AB7",
        "XGBoost":             "#0F6E56",
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = float((getattr(np, "trapz", None) or np.trapezoid)(prec[::-1], rec[::-1]))
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=colors.get(name, "#888780"), lw=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)", fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> models/outputs_crossval/pr_curves.png")


def main():
    print("Training crossval models (static features, StratifiedGroupKFold)")

    # Load data
    print("\nLoading data ...")
    X_train, y_train, groups_train, _ = load(TRAIN)
    X_val, y_val, _, _                = load(VAL)
    X_test, y_test, _, df_test        = load(TEST)
    print(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # Tune models using cross-validation on training set
    logreg  = tune_logreg(X_train, y_train, groups_train)
    xgboost = tune_xgboost(X_train, y_train, groups_train)

    # Validate before final test (informational)
    print("\nValidation-set check")
    for name, model in [("Logistic Regression", logreg), ("XGBoost", xgboost)]:
        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_f1 = f1_score(y_val, (val_prob >= 0.5).astype(int))
        print(f"  {name:<22} val AUC={val_auc:.4f}  val F1={val_f1:.4f}")

    # Retrain best model (LogReg) on train+val for final test evaluation
    print("\nRetraining Logistic Regression on train+val for final evaluation")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    logreg.fit(X_train_full, y_train_full)

    # Test-set evaluation
    print("\nTest-set results")
    comparison = []
    probs_dict = {}

    with open(OUT / "metrics.txt", "w") as fout:
        fout.write("Crossval models: static features + StratifiedGroupKFold\n")
        fout.write("Logistic Regression retrained on train+val before test evaluation\n\n")

        # LogReg (retrained on train+val)
        lr_prob = logreg.predict_proba(X_test)[:, 1]
        lr_pred = (lr_prob >= 0.5).astype(int)
        auc, f1 = report("Logistic Regression (train+val)", y_test, lr_pred, lr_prob, fout)
        comparison.append(("Logistic Regression", auc, f1))
        probs_dict["Logistic Regression"] = lr_prob

        # XGBoost (trained on train only)
        xgb_prob = xgboost.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        auc, f1 = report("XGBoost", y_test, xgb_pred, xgb_prob, fout)
        comparison.append(("XGBoost", auc, f1))
        probs_dict["XGBoost"] = xgb_prob

    print(f"  Saved -> models/outputs_crossval/metrics.txt")

    # Save predictions
    df_test = df_test.copy()
    df_test["logreg_prob"] = lr_prob
    df_test["logreg_pred"] = lr_pred
    df_test["xgb_prob"]    = xgb_prob
    df_test["xgb_pred"]    = xgb_pred
    df_test.to_csv(OUT / "results.csv", index=False)
    print(f"  Saved -> models/outputs_crossval/results.csv")

    # Save models
    for fname, obj in [
        ("logreg_model.pkl", logreg),
        ("xgb_model.pkl",    xgboost),
    ]:
        with open(OUT / fname, "wb") as f:
            pickle.dump(obj, f)
    print(f"  Saved -> models/outputs_crossval/*.pkl")

    # Plots
    print("\nGenerating plots")
    plot_pr_curves(probs_dict, y_test)

    print("\nDone! All outputs saved to models/outputs_crossval/")


if __name__ == "__main__":
    main()
