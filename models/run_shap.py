"""
SHAP analysis on the deployed LogReg + TF-IDF model.

Ranks the 18 static features by their mean absolute contribution to the model's
predictions on a 3,000-row test sample, and writes a bar chart, ranking CSV,
and text summary. TF-IDF tokens are excluded from the figure since the
human-code defect prediction literature is at the level of software metrics,
not individual tokens.

    python3 models/run_shap.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.sparse import hstack, csr_matrix

SPLITS = Path("data/clean/splits")
TFIDF_DIR = Path("models/outputs_tfidf")
OUT = Path("models/outputs_shap")
OUT.mkdir(parents=True, exist_ok=True)

N_TEST_SAMPLE = 3000
N_BG_SAMPLE = 2000
RNG = 42

FEATURE_COLS = [
    "classical_loc", "classical_cyclomatic_complexity", "classical_max_nesting_depth",
    "ast_if_count", "ast_for_count", "ast_while_count", "ast_try_count",
    "ast_except_count", "ast_return_count", "ast_import_count", "ast_has_error_handling",
    "align_lib_coverage", "align_missing_libs", "align_length_ratio",
    "smell_hardcoded_return_funcs", "smell_placeholder_hits",
    "smell_is_very_short", "smell_relative_length",
]

LABELS = {
    "classical_loc":                  "Lines of code",
    "classical_cyclomatic_complexity":"Cyclomatic complexity",
    "classical_max_nesting_depth":    "Max nesting depth",
    "ast_if_count":                   "If-statement count",
    "ast_for_count":                  "For-loop count",
    "ast_while_count":                "While-loop count",
    "ast_try_count":                  "Try-block count",
    "ast_except_count":               "Except-handler count",
    "ast_return_count":               "Return count",
    "ast_import_count":               "Import count",
    "ast_has_error_handling":         "Has error handling",
    "align_lib_coverage":             "Library coverage",
    "align_missing_libs":             "Missing libraries",
    "align_length_ratio":             "Code/prompt length ratio",
    "smell_hardcoded_return_funcs":   "Hardcoded returns",
    "smell_placeholder_hits":         "Placeholder patterns",
    "smell_is_very_short":            "Very short solution",
    "smell_relative_length":          "Relative length",
}


def load_split(name):
    feat = pd.read_csv(SPLITS / f"{name}_features.csv")
    raw  = pd.read_csv(SPLITS / f"{name}.csv")
    code_col = "solution" if "solution" in raw.columns else "generated_code"
    feat = feat.copy()
    feat["_code"] = raw[code_col].fillna("").astype(str).values
    return feat


def build_X(df, word, char):
    static = csr_matrix(df[FEATURE_COLS].fillna(0).values.astype(float))
    return hstack([static, word.transform(df["_code"].tolist()),
                           char.transform(df["_code"].tolist())]).tocsr()


def to_dense_static(shap_vals, n_static):
    arr = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
    arr = arr[:, :n_static]
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def plot(df, path):
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#2E8C5A" if v > 0 else "#D85A30" for v in df["mean_signed_shap"]]
    ax.barh(df["label"], df["mean_signed_shap"], color=colors)
    ax.axvline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("Average contribution to predicted pass  "
                  "(← pushes toward failure | pushes toward passing →)")
    ax.set_title("What patterns the model relies on")
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(df, path):
    with open(path, "w") as fout:
        fout.write("SHAP analysis — deployed LogReg + TF-IDF, test sample of "
                   f"{N_TEST_SAMPLE} rows.\n")
        fout.write("Static features ranked by mean |SHAP|.\n\n")
        fout.write(df.to_string(index=False))
        fout.write("\n\nTop patterns by absolute contribution:\n")
        for _, r in df.head(8).iterrows():
            direction = "→ failure" if r.mean_signed_shap < 0 else "→ passing"
            fout.write(f"  {r.label:<28s}  |SHAP|={r.mean_abs_shap:.4f}  {direction}\n")


def main():
    with open(TFIDF_DIR / "logreg_model.pkl", "rb") as f: model = pickle.load(f)
    with open(TFIDF_DIR / "word_tfidf.pkl",   "rb") as f: word  = pickle.load(f)
    with open(TFIDF_DIR / "char_tfidf.pkl",   "rb") as f: char  = pickle.load(f)

    train = load_split("train")
    test  = load_split("test")
    bg    = train.sample(N_BG_SAMPLE, random_state=RNG)
    te    = test.sample(N_TEST_SAMPLE, random_state=RNG)

    X_bg = build_X(bg, word, char)
    X_te = build_X(te, word, char)

    print(f"  bg shape={X_bg.shape}  test shape={X_te.shape}", flush=True)
    print("  building explainer ...", flush=True)
    explainer = shap.LinearExplainer(model, X_bg, feature_perturbation="interventional")
    print("  computing SHAP values ...", flush=True)
    shap_te = explainer.shap_values(X_te)

    static = to_dense_static(shap_te, len(FEATURE_COLS))
    mean_abs    = np.abs(static).mean(axis=0)
    mean_signed = static.mean(axis=0)

    df = pd.DataFrame({
        "feature":          FEATURE_COLS,
        "label":            [LABELS[c] for c in FEATURE_COLS],
        "mean_abs_shap":    mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df.to_csv(OUT / "shap_static_ranking.csv", index=False)
    plot(df, OUT / "shap_static_logreg.png")
    summary(df, OUT / "summary.txt")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
