"""
train_models_v2.py  —  Phase 2 improved: TF-IDF + static features
==================================================================
Combines raw code text (TF-IDF tokens) with the 17 static features
from Phase 1 to give models much richer signal than numbers alone.

Run from the project root:

  python3 train_models_v2.py

Output goes into models_v2/
  models_v2/logreg_model.pkl
  models_v2/lgbm_model.pkl
  models_v2/rf_model.pkl
  models_v2/results.csv
  models_v2/metrics.txt
  models_v2/feature_importance.png
  models_v2/pr_curves.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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

# ── Paths ──────────────────────────────────────────────────
# We need both the original splits (for raw code) and the
# feature splits (for the 17 static features)
TRAIN_FEAT = Path("data/splits/train_features.csv")
VAL_FEAT   = Path("data/splits/val_features.csv")
TEST_FEAT  = Path("data/splits/test_features.csv")

TRAIN_RAW  = Path("data/splits/train.csv")
VAL_RAW    = Path("data/splits/val.csv")
TEST_RAW   = Path("data/splits/test.csv")

OUT = Path("models_v2")
OUT.mkdir(exist_ok=True)

# ── Feature columns ────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════

def load_data(feat_path, raw_path):
    """Load feature CSV + raw CSV, return X_static, y, code_texts."""
    feat_df = pd.read_csv(feat_path)
    raw_df  = pd.read_csv(raw_path)

    # Static features
    X_static = feat_df[FEATURE_COLS].fillna(0).values.astype(float)
    y        = feat_df[LABEL].values.astype(int)

    # Raw code text — column is called 'solution' in the raw splits
    code_col  = "solution" if "solution" in raw_df.columns else "generated_code"
    texts     = raw_df[code_col].fillna("").astype(str).tolist()

    return X_static, y, texts, feat_df


# ══════════════════════════════════════════════════════════
# BUILD TF-IDF + COMBINED MATRIX
# ══════════════════════════════════════════════════════════

def build_tfidf(train_texts, val_texts, test_texts):
    """
    Fit TF-IDF on training code, transform all splits.
    Uses character n-grams (2-4) to capture code tokens
    like function names, operators, and keywords.
    Also uses word n-grams (1-2) to capture higher-level patterns.
    """
    print("  Fitting TF-IDF on training code ...")

    # Word-level TF-IDF: captures keywords, function names
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",  # code identifiers
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )

    # Character-level TF-IDF: captures syntax patterns
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=10000,
        sublinear_tf=True,
    )

    # Fit on training text only
    word_tr = word_tfidf.fit_transform(train_texts)
    char_tr = char_tfidf.fit_transform(train_texts)

    word_va = word_tfidf.transform(val_texts)
    char_va = char_tfidf.transform(val_texts)

    word_te = word_tfidf.transform(test_texts)
    char_te = char_tfidf.transform(test_texts)

    print(f"  Word TF-IDF: {word_tr.shape[1]:,} features")
    print(f"  Char TF-IDF: {char_tr.shape[1]:,} features")

    return (
        (word_tr, char_tr),
        (word_va, char_va),
        (word_te, char_te),
        (word_tfidf, char_tfidf),
    )


def combine(X_static, word_tfidf, char_tfidf):
    """Stack static features + word TF-IDF + char TF-IDF into one matrix."""
    static_sparse = csr_matrix(X_static)
    return hstack([static_sparse, word_tfidf, char_tfidf])


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════

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
    return auc, f1


# ══════════════════════════════════════════════════════════
# MODEL 1 — Logistic Regression
# ══════════════════════════════════════════════════════════

def train_logreg(X_tr, y_tr, X_va, y_va):
    print("\n── Logistic Regression ──────────────────────────────")
    best_auc, best_model = -1, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, solver="saga",
            random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
        print(f"  C={C:<6}  val AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_model = auc, m
    print(f"  Best val AUC={best_auc:.4f}")
    return best_model


# ══════════════════════════════════════════════════════════
# MODEL 2 — LightGBM
# ══════════════════════════════════════════════════════════

def train_lgbm(X_tr, y_tr, X_va, y_va):
    print("\n── LightGBM ─────────────────────────────────────────")
    neg, pos  = np.bincount(y_tr)
    scale_pos = neg / pos
    print(f"  scale_pos_weight={scale_pos:.2f}")

    best_auc, best_model = -1, None
    for n in [300, 500]:
        for lr in [0.05, 0.1]:
            m = lgb.LGBMClassifier(
                n_estimators=n, learning_rate=lr, max_depth=6,
                scale_pos_weight=scale_pos, subsample=0.8,
                colsample_bytree=0.3,   # lower because we have many TF-IDF cols
                min_child_samples=20,
                random_state=42, verbose=-1, n_jobs=-1,
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
            print(f"  n={n} lr={lr}  val AUC={auc:.4f}")
            if auc > best_auc:
                best_auc, best_model = auc, m
    print(f"  Best val AUC={best_auc:.4f}")
    return best_model


# ══════════════════════════════════════════════════════════
# MODEL 3 — Random Forest
# ══════════════════════════════════════════════════════════

def train_rf(X_tr, y_tr, X_va, y_va):
    print("\n── Random Forest ────────────────────────────────────")
    # Random Forest doesn't work well on very high-dimensional sparse TF-IDF
    # so we use only the 17 static features for this one
    print("  (uses static features only — RF + sparse TF-IDF is slow)")

    # Extract just the static feature columns (first 17 columns)
    X_tr_static = X_tr[:, :len(FEATURE_COLS)] if hasattr(X_tr, '__getitem__') else X_tr.toarray()[:, :len(FEATURE_COLS)]
    X_va_static = X_va[:, :len(FEATURE_COLS)] if hasattr(X_va, '__getitem__') else X_va.toarray()[:, :len(FEATURE_COLS)]

    # Convert sparse to dense for RF
    if hasattr(X_tr_static, "toarray"):
        X_tr_static = X_tr_static.toarray()
    if hasattr(X_va_static, "toarray"):
        X_va_static = X_va_static.toarray()

    best_auc, best_model = -1, None
    for n in [200, 500]:
        for depth in [8, 15, None]:
            m = RandomForestClassifier(
                n_estimators=n, max_depth=depth,
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            m.fit(X_tr_static, y_tr)
            auc = roc_auc_score(y_va, m.predict_proba(X_va_static)[:, 1])
            print(f"  n={n} depth={depth}  val AUC={auc:.4f}")
            if auc > best_auc:
                best_auc, best_model = auc, m
    print(f"  Best val AUC={best_auc:.4f}")
    return best_model


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

def plot_pr_curves(probs: dict, y_test):
    colors = {
        "Logistic Regression": "#534AB7",
        "LightGBM":            "#0F6E56",
        "Random Forest":       "#D85A30",
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = float(np.trapz(prec[::-1], rec[::-1]))
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
    print(f"  Saved -> models_v2/pr_curves.png")


def plot_logreg_top_features(model, word_tfidf, char_tfidf):
    """Show top TF-IDF tokens by logistic regression coefficient."""
    coefs = model.coef_[0]

    # Feature names: static + word vocab + char vocab
    static_names = FEATURE_COLS
    word_names   = [f"word:{w}" for w in word_tfidf.get_feature_names_out()]
    char_names   = [f"char:{c}" for c in char_tfidf.get_feature_names_out()]
    all_names    = static_names + word_names + char_names

    # Trim to actual number of coefficients
    all_names = all_names[:len(coefs)]

    top_pos = np.argsort(coefs)[-20:]
    top_neg = np.argsort(coefs)[:20]
    idx     = np.concatenate([top_neg, top_pos])

    names = [all_names[i] for i in idx]
    vals  = coefs[idx]
    colors = ["#1D9E75" if v > 0 else "#D85A30" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(names, vals, color=colors, height=0.6)
    ax.axvline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("Logistic Regression coefficient")
    ax.set_title("Top features predicting pass (green) vs fail (red)",
                 fontweight="normal")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> models_v2/feature_importance.png")


# ══════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════

def print_comparison(results: list, fout):
    header = f"\n{'='*55}\n  Model comparison (test set)\n{'='*55}"
    rows   = f"  {'Model':<25} {'AUC-ROC':>8}  {'F1':>6}\n  {'-'*42}"
    for name, auc, f1 in sorted(results, key=lambda x: -x[1]):
        rows += f"\n  {name:<25} {auc:>8.4f}  {f1:>6.4f}"
    block = header + "\n" + rows + "\n"
    print(block)
    fout.write(block + "\n")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("="*55)
    print("  Phase 2 v2 — TF-IDF + Static Features")
    print("="*55)

    # Load data
    print("\nLoading data ...")
    X_tr_s, y_tr, tr_texts, _       = load_data(TRAIN_FEAT, TRAIN_RAW)
    X_va_s, y_va, va_texts, _       = load_data(VAL_FEAT,   VAL_RAW)
    X_te_s, y_te, te_texts, df_test = load_data(TEST_FEAT,  TEST_RAW)
    print(f"  Train={len(y_tr):,}  Val={len(y_va):,}  Test={len(y_te):,}")

    # Build TF-IDF
    print("\nBuilding TF-IDF features ...")
    (word_tr, char_tr), (word_va, char_va), (word_te, char_te), (word_tfidf, char_tfidf) = \
        build_tfidf(tr_texts, va_texts, te_texts)

    # Combined matrices (static + word + char TF-IDF)
    X_tr = combine(X_tr_s, word_tr, char_tr)
    X_va = combine(X_va_s, word_va, char_va)
    X_te = combine(X_te_s, word_te, char_te)
    print(f"  Combined feature matrix: {X_tr.shape[1]:,} total features")

    # Train models
    logreg = train_logreg(X_tr, y_tr, X_va, y_va)
    lgbm   = train_lgbm(X_tr, y_tr, X_va, y_va)
    rf     = train_rf(X_tr, y_tr, X_va, y_va)

    # Test-set evaluation
    print("\n── Test-set results ─────────────────────────────────")
    comparison = []
    probs_dict = {}

    with open(OUT / "metrics.txt", "w") as fout:
        fout.write("Phase 2 v2 — TF-IDF + static features\n")

        lr_prob  = logreg.predict_proba(X_te)[:, 1]
        lr_pred  = logreg.predict(X_te)
        auc, f1  = report("Logistic Regression", y_te, lr_pred, lr_prob, fout)
        comparison.append(("Logistic Regression", auc, f1))
        probs_dict["Logistic Regression"] = lr_prob

        lgbm_prob = lgbm.predict_proba(X_te)[:, 1]
        lgbm_pred = lgbm.predict(X_te)
        auc, f1   = report("LightGBM", y_te, lgbm_pred, lgbm_prob, fout)
        comparison.append(("LightGBM", auc, f1))
        probs_dict["LightGBM"] = lgbm_prob

        # RF uses static features only
        X_te_static = X_te_s
        X_va_static = X_va_s
        rf_prob  = rf.predict_proba(X_te_static)[:, 1]
        rf_pred  = rf.predict(X_te_static)
        auc, f1  = report("Random Forest", y_te, rf_pred, rf_prob, fout)
        comparison.append(("Random Forest", auc, f1))
        probs_dict["Random Forest"] = rf_prob

        print_comparison(comparison, fout)

    # Save predictions
    df_test = df_test.copy()
    df_test["logreg_prob"] = lr_prob
    df_test["logreg_pred"] = lr_pred
    df_test["lgbm_prob"]   = lgbm_prob
    df_test["lgbm_pred"]   = lgbm_pred
    df_test["rf_prob"]     = rf_prob
    df_test["rf_pred"]     = rf_pred
    df_test.to_csv(OUT / "results.csv", index=False)
    print(f"  Saved -> models_v2/results.csv")

    # Save models + vectorizers
    for fname, obj in [
        ("logreg_model.pkl",  logreg),
        ("lgbm_model.pkl",    lgbm),
        ("rf_model.pkl",      rf),
        ("word_tfidf.pkl",    word_tfidf),
        ("char_tfidf.pkl",    char_tfidf),
    ]:
        with open(OUT / fname, "wb") as f:
            pickle.dump(obj, f)
    print(f"  Saved -> models_v2/*.pkl")

    # Plots
    print("\n── Generating plots ─────────────────────────────────")
    plot_pr_curves(probs_dict, y_te)
    plot_logreg_top_features(logreg, word_tfidf, char_tfidf)

    print("\nDone! All outputs saved to models_v2/")


if __name__ == "__main__":
    main()
