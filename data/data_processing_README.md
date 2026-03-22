# Data

Run `collect_data.py` from the project root to reproduce everything from scratch. It downloads all raw data and builds the final CSV.

```bash
pip install datasets pandas requests tqdm
python data/collect_data.py
```

## Output

One file: `processed/samples.csv` — 123,416 rows, one per (model, task, prompt format).

| Column | Description |
|---|---|
| task_id | e.g. BigCodeBench/0 |
| model_name | the LLM that generated the code, e.g. GPT_4o_2024_05_13 |
| split | complete or instruct (see below) |
| solution | the generated Python code |
| label | 1 = passed all tests, 0 = failed |
| complete_prompt | long docstring-style prompt for this task |
| instruct_prompt | short natural language version of the same task |
| libs | required libraries |
| entry_point | function name being tested |

## How it was built

### Step 1 — Task descriptions

1,140 Python programming tasks downloaded from `bigcode/bigcodebench` on HuggingFace (split `v0.1.4`). Each task has two prompt variants: a long docstring-style prompt (`complete_prompt`) and a short natural language instruction (`instruct_prompt`). Saved to `raw/bigcodebench_tasks.jsonl`.

### Step 2 — LLM-generated code samples

The BigCodeBench team ran 100+ LLMs on all 1,140 tasks and published the outputs as a zip on their GitHub releases page (v0.2.5, `sanitized_calibrated_samples.zip`, 158 MB). Each model gets its own JSONL file where every line is one task solution. Extracted to `raw/sanitized_calibrated_samples/`.

The zip has 6 subdirectories: `complete/`, `instruct/`, `full/complete/`, `full/instruct/`, `hard/complete/`, `hard/instruct/`. We only use `complete/` and `instruct/`. The `full/` versions add newer models (DeepSeek-R1, Qwen2.5, Llama-4) that don't have corresponding eval results yet, so we'd have code with no labels. The `hard/` subdirectory only covers 148 of 1,140 tasks.

### Step 3 — Pass/fail labels

The team also published per-task pass/fail results for each model on HuggingFace: `bigcode/bigcodebench-complete-perf` (85 models) and `bigcode/bigcodebench-instruct-perf` (52 models). Each HuggingFace split corresponds to one model and has rows of `{task_id, status}` where status is 0 or 1. Downloaded and saved as individual JSON files in `raw/eval_results/`.

### Step 4 — Merging

Matching samples to labels was the tricky part. The zip files use full HuggingFace model IDs (e.g. `codellama--CodeLlama-7b-Instruct-hf`) while the eval datasets use display names (e.g. `CodeLlama_7B_Instruct`). We match them in two ways: fuzzy normalization first (strip org prefix, strip `-hf` suffix, remove separators), then a manual mapping dict in `collect_data.py` for the cases normalization misses (version suffixes like `-v0.1`, API date stamps, etc.).

Models we can't match are dropped. This mainly affects base models (non-instruction-tuned, like `CodeLlama_7B_Base`) which appear in the eval datasets but have no corresponding sample files in the zip — the zip only has instruct/chat variants. After matching we end up with 57 unique models.

The `complete` and `instruct` splits are kept as separate rows because the same model produces different code under each prompt format, and pass rates differ (45% for complete, 37% for instruct). Task metadata (prompts, libs, entry point) is merged directly into the CSV so everything is in one place.

## Raw data layout

```
raw/
  bigcodebench_tasks.jsonl           # 1,140 task descriptions
  sanitized_calibrated_samples.zip   # original download (158 MB)
  sanitized_calibrated_samples/
    complete/                        # 87 model files x 1,140 tasks
    instruct/                        # 134 model files x 1,140 tasks
    full/                            # newer models, no labels available
    hard/                            # 148-task subset, not used
  eval_results/                      # 137 JSON files, one per model x split
```
