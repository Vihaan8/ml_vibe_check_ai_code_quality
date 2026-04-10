# Phase 1: Feature Extraction

This folder contains the Phase 1 feature engineering pipeline for the **Vibe Check** project.

The goal of Phase 1 is to convert each `(prompt, generated code)` sample into a compact set of static features that can later be used for defect prediction / pass-fail prediction.

## Files

- `feature_extraction.py`  
  Contains all feature extraction functions.

- `run_phase1.py`  
  Loads the BigCodeBench CSV, runs feature extraction on all rows, saves the output CSV, and prints a quick feature summary.

---

## Feature groups

This implementation extracts four feature groups, matching the proposal:

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
- `smell_placeholder_hits`
- `smell_is_very_short`

### Meta feature
- `meta_parse_error`

In total, this version outputs **17 features + 1 meta field** per sample.

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