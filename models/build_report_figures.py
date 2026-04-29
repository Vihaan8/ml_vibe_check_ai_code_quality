"""
Builds the report figures from saved test predictions and analysis CSVs.

Produces:
  report_figures/fig2_model_comparison_v2.png   AUC across models, y-axis from 0,
                                                95% bootstrap CIs on learned models.
  report_figures/fig5_crossmodel_drops.png      Per-family AUC drop with the human-
                                                code cross-project range as context.
  report_figures/fig6_shap_importance.png       Static features by mean |SHAP|,
                                                colored by direction.
  report_figures/table1_model_results.png       Table 1 rendered as a PNG with the
                                                same spelled-out labels as Figure 2.
  report_figures/model_metrics_with_ci.csv      AUC and F1 with 95% bootstrap CIs
                                                for the Table 1 update.

    python3 models/build_report_figures.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT     = Path(__file__).resolve().parent.parent
OUT_FIG  = ROOT / "report_figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED   = 42

GRAY    = "#A9A8A0"
PURPLE  = "#534AB7"
GREEN   = "#2E8C5A"
RED     = "#D85A30"
DARKGRAY= "#5A5A55"


# ----- bootstrap CI -----------------------------------------------------------

def bootstrap_metric(y_true, y_score, fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals[i] = fn(y_true[idx], y_score[idx])
        except ValueError:
            vals[i] = np.nan
    point = float(fn(y_true, y_score))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def auc_ci(y_true, y_prob):
    return bootstrap_metric(y_true, y_prob, roc_auc_score)


def f1_ci(y_true, y_pred):
    return bootstrap_metric(y_true, y_pred, lambda a, b: f1_score(a, b, zero_division=0))


# ----- threshold parsing ------------------------------------------------------

def read_threshold(metrics_path, label):
    txt = Path(metrics_path).read_text()
    m = re.search(rf"{re.escape(label)}.*?threshold=([0-9.]+)", txt)
    return float(m.group(1)) if m else 0.5


# ----- model registry ---------------------------------------------------------

def load_models():
    """Return list of (display_name, group, y_true, y_prob, y_pred_at_tuned_t)."""
    base = pd.read_csv(ROOT / "models/outputs_baseline/results.csv")
    tfidf = pd.read_csv(ROOT / "models/outputs_tfidf/results.csv")
    cv    = pd.read_csv(ROOT / "models/outputs_crossval/results_tuned.csv")

    t_lr_b   = read_threshold(ROOT / "models/outputs_baseline/threshold_metrics.txt", "LogReg (baseline)")
    t_lgb_b  = read_threshold(ROOT / "models/outputs_baseline/threshold_metrics.txt", "LightGBM (baseline)")
    t_lr_t   = read_threshold(ROOT / "models/outputs_tfidf/threshold_metrics.txt",   "LogReg (tfidf)")
    t_lgb_t  = read_threshold(ROOT / "models/outputs_tfidf/threshold_metrics.txt",   "LightGBM (tfidf)")
    t_lr_cv  = read_threshold(ROOT / "models/outputs_crossval/threshold_metrics.txt", "LogReg (crossval)")
    t_xgb_cv = read_threshold(ROOT / "models/outputs_crossval/threshold_metrics.txt", "XGBoost (crossval)")

    rows = [
        ("Logistic Regression\n(static features)",      "learned", base["label"].values,  base["logreg_prob"].values, (base["logreg_prob"].values >= t_lr_b).astype(int)),
        ("LightGBM\n(static features)",                 "learned", base["label"].values,  base["lgbm_prob"].values,   (base["lgbm_prob"].values   >= t_lgb_b).astype(int)),
        ("Logistic Regression\n(cross-validated)",      "learned", cv["label"].values,    cv["logistic_regression_prob"].values, (cv["logistic_regression_prob"].values >= t_lr_cv).astype(int)),
        ("XGBoost\n(cross-validated)",                  "learned", cv["label"].values,    cv["xgboost_prob"].values,             (cv["xgboost_prob"].values             >= t_xgb_cv).astype(int)),
        ("LightGBM\n(with text features)",              "learned", tfidf["label"].values, tfidf["lgbm_prob"].values,  (tfidf["lgbm_prob"].values  >= t_lgb_t).astype(int)),
        ("Logistic Regression\n(with text features)",   "best",    tfidf["label"].values, tfidf["logreg_prob"].values, (tfidf["logreg_prob"].values >= t_lr_t).astype(int)),
    ]
    return rows


# ----- figures ----------------------------------------------------------------

def fig_model_comparison(rows, out_path):
    # baselines from metrics.txt are deterministic — point estimates only
    baseline_rows = [
        ("Always predict\nfail",                   "baseline", 0.500),
        ("Random guess at\nclass proportions",     "baseline", 0.503),
        ("Lines-of-code\nthreshold",               "baseline", 0.385),
    ]

    learned = []
    for name, grp, y, p, _ in rows:
        a, lo, hi = auc_ci(y, p)
        learned.append((name, grp, a, lo, hi))

    labels  = [b[0] for b in baseline_rows] + [r[0] for r in learned]
    means   = [b[2] for b in baseline_rows] + [r[2] for r in learned]
    los     = [None]*3 + [r[3] for r in learned]
    his     = [None]*3 + [r[4] for r in learned]
    colors  = [GRAY]*3 + [GREEN if r[1] == "best" else PURPLE for r in learned]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)

    # error bars only on learned models
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        if lo is None: continue
        ax.errorbar(i, m, yerr=[[m-lo], [hi-m]], color=DARKGRAY,
                    capsize=4, capthick=1.0, elinewidth=1.0, fmt="none")

    # value annotations
    for i, m in enumerate(means):
        ax.text(i, m + 0.018, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="#888780", linestyle=":", linewidth=1)
    ax.text(len(labels) - 0.4, 0.51, "chance", color="#888780", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Ranking quality (AUC-ROC, higher is better)")
    ax.set_title("How well each model ranks AI-generated code by failure risk\n"
                 "(95% bootstrap confidence interval on the learned models)",
                 fontweight="normal", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=GRAY, label="Baselines"),
              Patch(facecolor=PURPLE, label="Learned models"),
              Patch(facecolor=GREEN, label="Best model")]
    ax.legend(handles=legend, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_crossmodel_drops(out_path):
    df = pd.read_csv(ROOT / "models/outputs_crossmodel/crossmodel_results.csv")
    df = df.sort_values("auc_drop", ascending=False).reset_index(drop=True)
    df["full_pct"] = (df["in_dist_auc"] * 100).round(1)
    df["loo_pct"]  = (df["ood_auc"]    * 100).round(1)

    n = len(df)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    bar_h = 0.36
    y = np.arange(n)

    ax.barh(y - bar_h/2, df["full_pct"], height=bar_h, color=DARKGRAY,
            label="Trained with this AI family")
    ax.barh(y + bar_h/2, df["loo_pct"],  height=bar_h, color=GREEN,
            label="Trained without this AI family")

    for i, row in df.iterrows():
        ax.text(row["full_pct"] - 0.4, i - bar_h/2, f"{row['full_pct']:.0f}",
                va="center", ha="right", fontsize=10, color="white", fontweight="bold")
        ax.text(row["loo_pct"]  - 0.4, i + bar_h/2, f"{row['loo_pct']:.0f}",
                va="center", ha="right", fontsize=10, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["family"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(50, 70)
    ax.set_xlabel("Pairs of (passing, failing) code ranked correctly, out of 100", fontsize=10)
    ax.set_title("Removing an AI family from training barely changes accuracy",
                 fontweight="normal", fontsize=12, loc="left")
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10,
              bbox_to_anchor=(1.0, 1.02), ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_shap_importance(out_path):
    df = pd.read_csv(ROOT / "models/outputs_shap/shap_static_ranking.csv")
    df = df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)

    colors = [GREEN if v > 0 else RED for v in df["mean_signed_shap"]]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(df["label"], df["mean_abs_shap"], color=colors, height=0.7)
    for i, v in enumerate(df["mean_abs_shap"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.15)
    ax.set_xlabel("Average contribution to a prediction (longer bar = pattern matters more)")
    ax.set_title("Which patterns the model uses most when predicting failure risk",
                 fontweight="normal", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=RED, label="More of this → predicted failure"),
              Patch(facecolor=GREEN, label="More of this → predicted passing")]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----- table 1 png ------------------------------------------------------------

def fig_table1(rows, out_path):
    baselines = [
        ("Always predict fail",                 "—",                     "0.500", "0.000", "—"),
        ("Random guess at class proportions",   "—",                     "0.503", "0.413", "—"),
        ("Lines-of-code threshold",             "Lines of code only",    "0.385", "0.526", "—"),
    ]

    thresholds = {
        "Logistic Regression\n(static features)":      "0.36",
        "LightGBM\n(static features)":                  "0.37",
        "Logistic Regression\n(cross-validated)":       "0.37",
        "XGBoost\n(cross-validated)":                   "0.29",
        "LightGBM\n(with text features)":               "0.35",
        "Logistic Regression\n(with text features)":    "0.39",
    }
    feature_text = {
        "Logistic Regression\n(static features)":      "18 static",
        "LightGBM\n(static features)":                  "18 static",
        "Logistic Regression\n(cross-validated)":       "18 static",
        "XGBoost\n(cross-validated)":                   "18 static",
        "LightGBM\n(with text features)":               "18 static + 20K text",
        "Logistic Regression\n(with text features)":    "18 static + 20K text",
    }

    learned = []
    for name, grp, y, p, pred in rows:
        a, alo, ahi = auc_ci(y, p)
        f, flo, fhi = f1_ci(y, pred)
        flat = name.replace("\n", " ")
        learned.append((
            flat,
            feature_text[name],
            f"{a:.3f} [{alo:.3f}, {ahi:.3f}]",
            f"{f:.3f} [{flo:.3f}, {fhi:.3f}]",
            thresholds[name],
            grp == "best",
        ))

    headers = ["Model", "Features", "AUC-ROC", "F1 (tuned)", "Threshold"]
    cells = []
    row_colors = []
    for b in baselines:
        cells.append(list(b))
        row_colors.append("#F2F1ED")
    for r in learned:
        cells.append(list(r[:5]))
        row_colors.append("#E8F2EC" if r[5] else "white")

    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.34, 0.20, 0.20, 0.18, 0.10],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    n_cols = len(headers)
    for j in range(n_cols):
        c = tbl[(0, j)]
        c.set_facecolor(PURPLE)
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("white")

    for i, color in enumerate(row_colors, start=1):
        is_best = (i - 1) >= len(baselines) and learned[i - 1 - len(baselines)][5]
        for j in range(n_cols):
            c = tbl[(i, j)]
            c.set_facecolor(color)
            c.set_edgecolor("#D6D5CF")
            if is_best:
                c.set_text_props(fontweight="bold")
            if j == 0:
                c.set_text_props(ha="left", fontweight=c.get_text().get_fontweight())
                c.PAD = 0.04

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----- table 1 metrics --------------------------------------------------------

def write_metrics_csv(rows, out_path):
    out = []
    for name, grp, y, p, pred in rows:
        a, alo, ahi = auc_ci(y, p)
        f, flo, fhi = f1_ci(y, pred)
        out.append({
            "model":      name,
            "auc":        round(a, 4),
            "auc_ci_low": round(alo, 4),
            "auc_ci_hi":  round(ahi, 4),
            "f1":         round(f, 4),
            "f1_ci_low":  round(flo, 4),
            "f1_ci_hi":   round(fhi, 4),
        })
    df = pd.DataFrame(out)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))


# ----- main -------------------------------------------------------------------

def main():
    print("loading test predictions ...", flush=True)
    rows = load_models()

    print("computing metrics with bootstrap CIs ...", flush=True)
    write_metrics_csv(rows, OUT_FIG / "model_metrics_with_ci.csv")

    print("drawing figures ...", flush=True)
    fig_model_comparison(rows, OUT_FIG / "fig2_model_comparison_v2.png")
    fig_crossmodel_drops(OUT_FIG / "fig5_crossmodel_drops.png")
    fig_shap_importance(OUT_FIG / "fig6_shap_importance.png")
    fig_table1(rows, OUT_FIG / "table1_model_results.png")
    print("written:", OUT_FIG)


if __name__ == "__main__":
    main()
