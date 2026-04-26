"""
Cross-model generalization (leave-one-family-out).

For each LLM family F: retrain LogReg + TF-IDF without F, retune the threshold
on F-excluded validation, and evaluate on the F slice of the test set.
The drop from the deployed model's score on the same slice is the cost of
seeing a new family for the first time.

    python3 models/train_crossmodel.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

SPLITS = Path("data/clean/splits")
TFIDF_DIR = Path("models/outputs_tfidf")
OUT = Path("models/outputs_crossmodel")
OUT.mkdir(parents=True, exist_ok=True)

# Threshold of the deployed LogReg + TF-IDF (outputs_tfidf/threshold_metrics.txt)
IN_DIST_THRESHOLD = 0.39

FEATURE_COLS = [
    "classical_loc", "classical_cyclomatic_complexity", "classical_max_nesting_depth",
    "ast_if_count", "ast_for_count", "ast_while_count", "ast_try_count",
    "ast_except_count", "ast_return_count", "ast_import_count", "ast_has_error_handling",
    "align_lib_coverage", "align_missing_libs", "align_length_ratio",
    "smell_hardcoded_return_funcs", "smell_placeholder_hits",
    "smell_is_very_short", "smell_relative_length",
]
LABEL = "label"

# Models not listed stay in training but aren't evaluated as a held-out family.
FAMILIES = {
    "GPT": [
        "GPT_3.5_Turbo_0125", "GPT_4_0613",
        "GPT_4_Turbo_2024_04_09", "GPT_4o_2024_05_13",
    ],
    "Claude": [
        "Claude_3_Haiku_20240307", "Claude_3_Sonnet_20240229",
        "Claude_3_Opus_20240229", "Claude_3.5_Sonnet_20240620",
    ],
    "Llama": [
        "Llama_3_70B_Instruct", "Llama_3_8B_Instruct",
        "CodeLlama_7B_Instruct", "CodeLlama_13B_Instruct",
        "CodeLlama_34B_Instruct", "CodeLlama_70B_Instruct",
        "Hermes_2_Theta_Llama_3_70B",
        "ReflectionCoder_CL_7B", "ReflectionCoder_CL_34B",
        "WaveCoder_Ultra_6.7B",
    ],
    "DeepSeek": [
        "DeepSeek_Coder_1.3B_Instruct", "DeepSeek_Coder_6.7B_Instruct",
        "DeepSeek_Coder_33B_Instruct", "DeepSeek_Coder_V2_Instruct",
        "DeepSeek_Coder_V2_Lite_Instruct", "DeepSeek_V2_Chat",
        "Magicoder_S_DS_6.7B",
        "OpenCodeInterpreter_DS_1.3B", "OpenCodeInterpreter_DS_6.7B",
        "ReflectionCoder_DS_6.7B", "ReflectionCoder_DS_33B",
        "AutoCoder_S_6.7B",
    ],
    "Mistral": [
        "Mistral_7B_Instruct_v0.3", "Mistral_Small_2402",
        "Mistral_Large_2402", "Mixtral_8x22B_Instruct",
        "Codestral_22B_v0.1",
    ],
    "Qwen": [
        "Qwen1.5_32B_Chat", "Qwen1.5_72B_Chat", "Qwen1.5_110B_Chat",
        "Qwen2_72B_Chat", "CodeQwen1.5_7B_Chat",
        "AutoCoder_QW_7B",
    ],
}


def load_split(name):
    feat = pd.read_csv(SPLITS / f"{name}_features.csv")
    raw  = pd.read_csv(SPLITS / f"{name}.csv")
    code_col = "solution" if "solution" in raw.columns else "generated_code"
    feat = feat.copy()
    feat["_code"] = raw[code_col].fillna("").astype(str).values
    return feat


def build_tfidf(train_texts, *eval_text_lists):
    word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
        ngram_range=(1, 2), max_features=10000, sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4), max_features=10000, sublinear_tf=True,
    )
    word_tr = word.fit_transform(train_texts)
    char_tr = char.fit_transform(train_texts)
    evals = [(word.transform(t), char.transform(t)) for t in eval_text_lists]
    return (word_tr, char_tr), evals


def combine(feat_df, word_mat, char_mat):
    X_static = csr_matrix(feat_df[FEATURE_COLS].fillna(0).values.astype(float))
    return hstack([X_static, word_mat, char_mat])


def find_best_threshold(y, prob):
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def train_logreg(X_tr, y_tr, X_va, y_va):
    best_auc, best_model = -1, None
    for C in [0.1, 1.0, 10.0]:
        m = LogisticRegression(
            C=C, class_weight="balanced",
            penalty="l2", solver="liblinear",
            max_iter=200, tol=1e-3,
            random_state=42,
        )
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
        print(f"      C={C:<5} val AUC={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_model = auc, m
    return best_model


def in_distribution(test_df):
    with open(TFIDF_DIR / "logreg_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(TFIDF_DIR / "word_tfidf.pkl", "rb") as f:
        word = pickle.load(f)
    with open(TFIDF_DIR / "char_tfidf.pkl", "rb") as f:
        char = pickle.load(f)

    out = {}
    for fam, members in FAMILIES.items():
        mask = test_df["model_name"].isin(members)
        if mask.sum() == 0:
            continue
        sub = test_df.loc[mask]
        X = combine(sub, word.transform(sub["_code"].tolist()),
                         char.transform(sub["_code"].tolist()))
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= IN_DIST_THRESHOLD).astype(int)
        out[fam] = (
            roc_auc_score(sub[LABEL], prob),
            f1_score(sub[LABEL], pred, zero_division=0),
        )
    return out


def held_out(train_df, val_df, test_df):
    out = {}
    for fam, members in FAMILIES.items():
        sub_tr = train_df.loc[~train_df["model_name"].isin(members)]
        sub_va = val_df.loc[~val_df["model_name"].isin(members)]
        sub_te = test_df.loc[test_df["model_name"].isin(members)]
        if len(sub_te) == 0:
            continue
        print(f"\n[{fam}]  n_train={len(sub_tr):,}  n_val={len(sub_va):,}  n_test={len(sub_te):,}", flush=True)

        (word_tr, char_tr), evals = build_tfidf(
            sub_tr["_code"].tolist(),
            sub_va["_code"].tolist(),
            sub_te["_code"].tolist(),
        )
        word_va, char_va = evals[0]
        word_te, char_te = evals[1]

        X_tr = combine(sub_tr, word_tr, char_tr)
        X_va = combine(sub_va, word_va, char_va)
        X_te = combine(sub_te, word_te, char_te)
        y_tr = sub_tr[LABEL].values.astype(int)
        y_va = sub_va[LABEL].values.astype(int)
        y_te = sub_te[LABEL].values.astype(int)

        model = train_logreg(X_tr, y_tr, X_va, y_va)
        val_prob  = model.predict_proba(X_va)[:, 1]
        test_prob = model.predict_proba(X_te)[:, 1]
        t, _ = find_best_threshold(y_va, val_prob)
        pred = (test_prob >= t).astype(int)
        auc = roc_auc_score(y_te, test_prob)
        f1  = f1_score(y_te, pred, zero_division=0)
        print(f"  -> {fam:<10}  AUC={auc:.4f}  F1={f1:.4f}  t={t:.2f}", flush=True)
        out[fam] = (auc, f1, t, len(sub_te), len(members))
    return out


def plot_comparison(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["in_dist_auc"], w, label="In-distribution", color="#534AB7")
    ax.bar(x + w/2, df["ood_auc"],     w, label="Held-out family", color="#D85A30")
    ax.axhline(0.5, color="#888780", linestyle=":", linewidth=1, label="Chance")
    ax.set_ylim(0, max(0.75, df[["in_dist_auc", "ood_auc"]].values.max() + 0.05))
    ax.set_xticks(x)
    ax.set_xticklabels(df["family"])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Cross-model generalization: in-distribution vs held-out family",
                 fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
    for i, row in df.iterrows():
        ax.text(i - w/2, row["in_dist_auc"] + 0.01, f"{row['in_dist_auc']:.3f}",
                ha="center", fontsize=8)
        ax.text(i + w/2, row["ood_auc"] + 0.01, f"{row['ood_auc']:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    in_dist = in_distribution(test_df)
    ood     = held_out(train_df, val_df, test_df)

    rows = []
    for fam in FAMILIES:
        if fam not in in_dist or fam not in ood:
            continue
        in_auc, in_f1 = in_dist[fam]
        ood_auc, ood_f1, t, n_te, n_models = ood[fam]
        rows.append({
            "family": fam,
            "n_models": n_models,
            "n_test_rows": n_te,
            "in_dist_auc": round(in_auc, 4),
            "ood_auc": round(ood_auc, 4),
            "auc_drop": round(in_auc - ood_auc, 4),
            "in_dist_f1": round(in_f1, 4),
            "ood_f1": round(ood_f1, 4),
            "f1_drop": round(in_f1 - ood_f1, 4),
            "ood_threshold": round(t, 2),
        })
    df = pd.DataFrame(rows)

    df.to_csv(OUT / "crossmodel_results.csv", index=False)
    with open(OUT / "metrics.txt", "w") as fout:
        fout.write("Cross-model generalization (leave-one-family-out)\n\n")
        fout.write(
            f"In-distribution: deployed LogReg+TF-IDF (full train, threshold "
            f"{IN_DIST_THRESHOLD}) on this family's slice of test.\n"
            "Held-out: model retrained without this family, threshold re-tuned "
            "on family-excluded validation, evaluated on this family's slice of test.\n\n"
        )
        fout.write(df.to_string(index=False))
        fout.write("\n")
    plot_comparison(df, OUT / "crossmodel_comparison.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
