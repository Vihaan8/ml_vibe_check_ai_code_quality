"""
Downloads BigCodeBench task descriptions, LLM-generated code samples, and
pass/fail eval results, then merges them into two CSVs in data/processed/.

Run from the project root: python data/collect_data.py
"""

import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
SAMPLES_ZIP = RAW / "sanitized_calibrated_samples.zip"
SAMPLES_DIR = RAW / "sanitized_calibrated_samples"
EVAL_DIR = RAW / "eval_results"
TASKS_PATH = RAW / "bigcodebench_tasks.jsonl"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# The GitHub zip uses full HuggingFace model IDs (e.g. codellama--CodeLlama-7b-Instruct-hf)
# but the HuggingFace eval datasets use shorter display names (e.g. CodeLlama_7B_Instruct).
# Fuzzy normalization handles most cases. This dict covers the ones it misses,
# mainly due to -hf suffixes, version tags, API date stamps, and org-prefix quirks.
MANUAL_MAP = {
    "codellama--CodeLlama-7b-Instruct-hf":        "CodeLlama_7B_Instruct",
    "codellama--CodeLlama-13b-Instruct-hf":       "CodeLlama_13B_Instruct",
    "codellama--CodeLlama-34b-Instruct-hf":       "CodeLlama_34B_Instruct",
    "codellama--CodeLlama-70b-Instruct-hf":       "CodeLlama_70B_Instruct",
    "mistralai--Mixtral-8x22B-Instruct-v0.1":     "Mixtral_8x22B_Instruct",
    "mistralai--Mistral-7B-Instruct-v0.3":        "Mistral_7B_Instruct_v0.3",
    "gemini-1.5-flash":                           "Gemini_1.5_Flash_API_0514",
    "gemini-1.5-pro":                             "Gemini_1.5_Pro_API_0514",
    "google--codegemma-7b-it":                    "CodeGemma_7B_Instruct",
    "google--gemma-2-9b-it":                      "Gemma_2_9B_Instruct",
    "meta-llama--Meta-Llama-3-8B-Instruct":       "Llama_3_8B_Instruct",
    "meta-llama--Meta-Llama-3-70B-Instruct":      "Llama_3_70B_Instruct",
    "meta-llama--Meta-Llama-3.1-8B-Instruct":     "Llama_3_8B_Instruct",
    "meta-llama--Meta-Llama-3.1-70B-Instruct":    "Llama_3_70B_Instruct",
    "bigcode--starcoder2-15b-instruct-v0.1":      "StarCoder2_15B_Instruct_v0.1",
    "CohereForAI--c4ai-command-r-plus":           "Command_R_plus",
    "ibm-granite--granite-3b-code-instruct":      "Granite_Code_3B_Instruct",
    "ibm-granite--granite-8b-code-instruct":      "Granite_Code_8B_Instruct",
    "ibm-granite--granite-20b-code-instruct":     "Granite_Code_20B_Instruct",
    "ibm-granite--granite-34b-code-instruct":     "Granite_Code_34B_Instruct",
}


def download_tasks():
    if TASKS_PATH.exists():
        print(f"tasks already downloaded, skipping")
        return
    print("Downloading BigCodeBench tasks from HuggingFace...")
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    ds.to_json(str(TASKS_PATH))
    print(f"saved {len(ds)} tasks")


def download_samples():
    if SAMPLES_DIR.exists() and any(SAMPLES_DIR.glob("**/*.jsonl")):
        print("samples already extracted, skipping")
        return

    if not SAMPLES_ZIP.exists():
        url = "https://github.com/bigcode-project/bigcodebench/releases/download/v0.2.5/sanitized_calibrated_samples.zip"
        print(f"downloading sample zip ({url})...")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(SAMPLES_ZIP, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    print("extracting zip...")
    with zipfile.ZipFile(SAMPLES_ZIP) as zf:
        zf.extractall(RAW)


def download_eval_results():
    if len(list(EVAL_DIR.glob("*.json"))) >= 137:
        print("eval results already downloaded, skipping")
        return

    print("Downloading eval results from HuggingFace...")
    for ds_name, split_tag in [
        ("bigcode/bigcodebench-complete-perf", "complete"),
        ("bigcode/bigcodebench-instruct-perf", "instruct"),
    ]:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(ds_name)
        model_splits = list(builder.info.splits.keys())
        for model_split in tqdm(model_splits, desc=split_tag):
            out = EVAL_DIR / f"{model_split}--{split_tag}_eval_results.json"
            if out.exists():
                continue
            ds = load_dataset(ds_name, split=model_split)
            labels = {row["task_id"]: int(row["status"]) for row in ds}
            with open(out, "w") as fh:
                json.dump(labels, fh)


def _normalize(s: str) -> str:
    if "--" in s:
        s = s.split("--", 1)[1]
    s = re.sub(r"-hf$", "", s)
    s = re.sub(r"[-_v. ]", "", s)
    s = re.sub(r"\d{8}$", "", s)
    return s.lower()


def _load_eval_lookup(split: str) -> dict:
    lookup = {}
    for f in EVAL_DIR.glob(f"*--{split}_eval_results.json"):
        model = f.stem.replace(f"--{split}_eval_results", "")
        with open(f) as fh:
            lookup[model] = json.load(fh)
    return lookup


def build_csvs():
    tasks = {}
    with open(TASKS_PATH) as fh:
        for line in fh:
            t = json.loads(line)
            tasks[t["task_id"]] = t

    records = []

    for split in ["complete", "instruct"]:
        sample_subdir = SAMPLES_DIR / split
        if not sample_subdir.exists():
            continue

        eval_lookup = _load_eval_lookup(split)
        eval_norm = {_normalize(k): (k, v) for k, v in eval_lookup.items()}

        matched, skipped = 0, 0
        for sample_file in tqdm(list(sample_subdir.glob("*.jsonl")), desc=split):
            sample_model = sample_file.stem.split("--bigcodebench")[0]

            if sample_model in MANUAL_MAP:
                eval_model_name = MANUAL_MAP[sample_model]
                labels = eval_lookup.get(eval_model_name)
            else:
                match = eval_norm.get(_normalize(sample_model))
                eval_model_name, labels = match if match else (None, None)

            if labels is None:
                skipped += 1
                continue

            matched += 1
            with open(sample_file) as fh:
                for line in fh:
                    row = json.loads(line)
                    task_id = row["task_id"]
                    label = labels.get(task_id)
                    if label is None:
                        continue
                    records.append({
                        "task_id":    task_id,
                        "model_name": eval_model_name,
                        "split":      split,
                        "solution":   row.get("solution", ""),
                        "label":      label,
                    })

        print(f"{split}: matched {matched} models, skipped {skipped} (no eval results found)")

    samples_df = pd.DataFrame(records)

    # merge task metadata directly into samples so we have one flat file
    tasks_df = pd.DataFrame([
        {
            "task_id":         t["task_id"],
            "complete_prompt": t.get("complete_prompt", ""),
            "instruct_prompt": t.get("instruct_prompt", ""),
            "libs":            t.get("libs", ""),
            "entry_point":     t.get("entry_point", ""),
        }
        for t in tasks.values()
    ])
    samples_df = samples_df.merge(tasks_df, on="task_id", how="left")
    samples_df.to_csv(PROCESSED / "samples.csv", index=False)
    print(f"\nsamples.csv: {len(samples_df):,} rows, {samples_df['model_name'].nunique()} models, {samples_df['label'].mean():.2f} pass rate")
    print(f"columns: {list(samples_df.columns)}")


if __name__ == "__main__":
    print("step 1: task descriptions")
    download_tasks()

    print("\nstep 2: code samples")
    download_samples()

    print("\nstep 3: eval results")
    download_eval_results()

    print("\nstep 4: building CSVs")
    build_csvs()
