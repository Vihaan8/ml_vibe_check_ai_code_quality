# Software Defect Prediction for LLM-Generated Code

Can we predict whether AI-generated code will pass its test suite — without running it? We train classical and deep learning models on 123k labeled code samples across 57 LLMs to find out.

## Data

The dataset is hosted privately on HuggingFace at `Vihaan8/bigcodebench-sdp`. You'll need to be added as a collaborator to access it.

The main file you need is `processed/samples.csv` — 123k rows, one per (model, task, prompt format), with the generated code and a pass/fail label. To load it:

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="Vihaan8/bigcodebench-sdp",
    filename="processed/samples.csv",
    repo_type="dataset"
)
df = pd.read_csv(path)
```

The repo structure mirrors the local `data/` folder:

```
Vihaan8/bigcodebench-sdp
├── processed/
│   └── samples.csv                  # 123k rows, the ML dataset
└── raw/
    ├── bigcodebench_tasks.jsonl      # 1,140 task descriptions
    ├── sanitized_calibrated_samples.zip
    └── eval_results/                 # 137 per-model pass/fail JSON files
```

See `data/README.md` for full details on how the data was collected and processed.
