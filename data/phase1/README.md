# Phase 1: Feature Extraction

This folder contains the Phase 1 feature engineering pipeline for the **Vibe Check** project.

The goal of Phase 1 is to convert each `(prompt, generated code)` sample into a compact set of static features that can later be used for defect prediction / pass-fail prediction.

## Files

- `feature_extraction.py`  
  Contains all feature extraction functions.

- `run_phase1.py`  
  Loads the BigCodeBench CSV, runs feature extraction on all rows, saves the output CSV, and prints a quick feature summary.

- `split_data.py`  
  Splits the original dataset into **train / validation / test** sets before feature extraction.

---

## Feature groups

This implementation extracts four feature groups, matching the proposal.

### 1. Classical software metrics
These are traditional static metrics used in software defect prediction.

Extracted features:
- `classical_loc`
- `classical_cyclomatic_complexity`
- `classical_max_nesting_depth`

### 2. AST structural features
These count important Python AST node types to capture code structure.

Extracted features:
- `ast_if_count`
- `ast_for_count`
- `ast_while_count`
- `ast_try_count`
- `ast_except_count`
- `ast_return_count`
- `ast_import_count`
- `ast_has_error_handling`

### 3. Prompt-code alignment features
These measure whether the generated code appears to align with what the prompt asks for.

Extracted features:
- `align_lib_coverage`
- `align_missing_libs`
- `align_length_ratio`

### 4. LLM smell features
These target common LLM code-generation failure modes.

Extracted features:
- `smell_hardcoded_return_funcs`
- `smell_is_very_short`

### Meta feature
- `meta_parse_error` : 1 if the code has a syntax error and cannot be parsed, else 0.

In total, this version outputs **16 features + 1 meta field** per sample.

> Note: if your current `feature_extraction.py` still includes `smell_placeholder_hits`, then update this count accordingly.

---

## Input format

`run_phase1.py` expects a CSV with the following columns:

- `task_id`
- `model_name`
- `split`
- `solution`
- `label`
- `complete_prompt`
- `instruct_prompt`
- `libs`
- `entry_point`

The script renames:
- `solution` → `generated_code`
- `instruct_prompt` → `prompt`

So in the current implementation, the alignment features are computed using **`instruct_prompt`**.

---

## Installation

Install the required packages before running:

```bash
pip install pandas numpy radon
```

## Recommended workflow

Step 1: put samples.csv in this folder

Place the original dataset file here:

data/phase1/samples.csv

Step 2: split the dataset into train / validation / test

Run the split script first:

python split_data.py --input samples.csv --outdir splits

This will create:

splits/train.csv
splits/val.csv
splits/test.csv

The split is done before feature extraction, so that feature engineering can be run separately on each subset.

Step 3: run feature extraction on each split

Run Phase 1 separately for train / validation / test:

python run_phase1.py --input splits/train.csv --out splits/train_features.csv
python run_phase1.py --input splits/val.csv --out splits/val_features.csv
python run_phase1.py --input splits/test.csv --out splits/test_features.csv

This will create:

splits/train_features.csv
splits/val_features.csv
splits/test_features.csv
Quick test

If you want to test the pipeline on a smaller subset first:

python run_phase1.py --input samples.csv --max-rows 5000 --out features_sample.csv
Offline demo

If you do not want to use the real CSV, you can run the built-in synthetic demo:

python run_phase1.py --skip-download
Output

Each output CSV contains:

metadata columns:
task_id
model_name
split
label
entry_point
libs
all extracted Phase 1 features

After extraction, the script also prints:

dataset shape
parse error rate
pass rate
feature group counts
top features by absolute correlation with label
Programmatic usage
Single sample
from feature_extraction import extract_features

code = '''
def task_func(x):
    return x + 1
'''

prompt = "Write a function task_func that increments x by 1."

features = extract_features(code, prompt)
print(features)
Batch mode
import pandas as pd
from feature_extraction import extract_features_batch

df = pd.DataFrame({
    "generated_code": ["def f(x): return x"],
    "prompt": ["Write a function f that returns x."]
})

feat_df = extract_features_batch(df, code_col="generated_code", prompt_col="prompt")
print(feat_df.head())
