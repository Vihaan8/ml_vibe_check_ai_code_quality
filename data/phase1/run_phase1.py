"""
run_phase1.py
Phase 1 pipeline: load BigCodeBench CSV, run feature extraction, save output.

Confirmed column schema (from actual data):
    task_id, model_name, split, solution, label,
    complete_prompt, instruct_prompt, libs, entry_point

Usage
-----
    python run_phase1.py --input bigcodebench.csv
    python run_phase1.py --input bigcodebench.csv --max-rows 5000   # quick test
    python run_phase1.py --skip-download                            # offline demo

Output
------
    features_bigcodebench.csv  —  metadata + 62 features + label, ready for Phase 2
"""

import pandas as pd
import numpy as np
from feature_extraction import extract_features_batch


# ---- 1. Load ----

def load_bigcodebench(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load BigCodeBench from a local CSV export.

    Confirmed columns:
        task_id, model_name, split, solution, label,
        complete_prompt, instruct_prompt, libs, entry_point

    Returns a DataFrame with:
        - generated_code  (renamed from 'solution')
        - prompt          (renamed from 'instruct_prompt')
        - all other columns kept as-is
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Validate expected columns are present
    required = {"task_id", "model_name", "solution", "label", "instruct_prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}\n"
                         f"Actual columns: {list(df.columns)}")

    df = df.rename(columns={
        "solution":        "generated_code",
        "instruct_prompt": "prompt",
    })
    df["label"] = df["label"].astype(int)

    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=42).reset_index(drop=True)
        print(f"Capped to {len(df):,} rows for testing.")

    print(f"Loaded: {len(df):,} rows | "
          f"{df['model_name'].nunique()} models | "
          f"pass rate = {df['label'].mean():.2%}")
    if "split" in df.columns:
        print(f"Splits: {df['split'].value_counts().to_dict()}")
    return df


# ---- 2. Extract features ----

def run_extraction(df: pd.DataFrame, out_path: str = "features_bigcodebench.csv") -> pd.DataFrame:
    print("\nRunning Phase 1 feature extraction ...")
    feat_df = extract_features_batch(df, code_col="generated_code", prompt_col="prompt")

    # Keep metadata columns alongside features
    meta_cols = ["task_id", "model_name", "split", "label", "entry_point", "libs"]
    available_meta = [c for c in meta_cols if c in df.columns]
    out_df = pd.concat(
        [df[available_meta].reset_index(drop=True), feat_df.reset_index(drop=True)],
        axis=1
    )

    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"  Shape:            {out_df.shape[0]:,} rows x {out_df.shape[1]} columns")
    print(f"  Feature columns:  {len(feat_df.columns)}")
    print(f"  Parse error rate: {out_df['meta_parse_error'].mean():.2%}")
    print(f"  Pass rate:        {out_df['label'].mean():.2%}")
    return out_df


# ---- 3. Feature summary ----

def feature_summary(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if any(
        c.startswith(pfx) for pfx in
        ["meta_", "loc_", "classical_", "halstead_", "ast_", "align_", "smell_"]
    )]
    print(f"\n{'─'*60}")
    print(f"Feature summary ({len(feat_cols)} features)")
    print(f"{'─'*60}")
    grps = {
        "Meta":      [c for c in feat_cols if c.startswith("meta_")],
        "LOC":       [c for c in feat_cols if c.startswith("loc_")],
        "Classical": [c for c in feat_cols if c.startswith("classical_")],
        "Halstead":  [c for c in feat_cols if c.startswith("halstead_")],
        "AST":       [c for c in feat_cols if c.startswith("ast_")],
        "Alignment": [c for c in feat_cols if c.startswith("align_")],
        "LLM Smell": [c for c in feat_cols if c.startswith("smell_")],
    }
    for grp, cols in grps.items():
        print(f"  {grp:<12}: {len(cols):2d} features  {', '.join(cols[:4])}{'...' if len(cols) > 4 else ''}")

    print(f"\nTop 10 features by |correlation| with label:")
    corr = df[feat_cols].corrwith(df["label"]).abs().sort_values(ascending=False)
    for feat, val in corr.head(10).items():
        print(f"  {feat:<45} {val:.4f}")


# ---- Main ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default=None,
                        help="Path to BigCodeBench CSV file")
    parser.add_argument("--out",       default="features_sample.csv",
                        help="Output CSV path (default: features_sample.csv)")
    parser.add_argument("--max-rows",  type=int, default=None,
                        help="Cap rows for quick testing, e.g. --max-rows 5000")
    parser.add_argument("--skip-download", action="store_true",
                        help="Run offline demo with synthetic data (no CSV needed)")
    args = parser.parse_args()

    if args.skip_download:
        print("Running offline demo with 100 synthetic samples ...")

        SAMPLE_CODES = [
            'import pandas as pd\nimport requests\ndef task_func(url):\n    try:\n        r = requests.get(url)\n        return pd.json_normalize(r.json())\n    except Exception as e:\n        raise ValueError(e)',
            'def task_func(url):\n    return None',
            'def task_func(url):\n    # TODO\n    pass',
            'import pandas as pd\ndef task_func(url):\n    data = url\n    return pd.DataFrame()',
        ]
        SAMPLE_PROMPTS = [
            "Write task_func(url) to fetch JSON with requests and return a pandas DataFrame.",
        ] * 4

        n = 100
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 4, size=n)
        df = pd.DataFrame({
            "task_id":        [f"BigCodeBench/{i % 10}" for i in range(n)],
            "model_name":     rng.choice(["gpt-4o", "llama3", "deepseek-coder", "mistral-7b"], size=n).tolist(),
            "split":          rng.choice(["complete", "instruct"], size=n).tolist(),
            "generated_code": [SAMPLE_CODES[i] for i in indices],
            "prompt":         [SAMPLE_PROMPTS[i] for i in indices],
            "label":          rng.integers(0, 2, size=n).tolist(),
            "entry_point":    ["task_func"] * n,
            "libs":           ["['pandas', 'requests']"] * n,
        })
    elif args.input:
        df = load_bigcodebench(args.input, max_rows=args.max_rows)
    else:
        parser.error("Provide --input <csv_path> or use --skip-download for the offline demo.")

    out_df = run_extraction(df, out_path=args.out)
    feature_summary(out_df)