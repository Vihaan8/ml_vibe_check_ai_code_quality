"""
Uploads samples.csv to a private HuggingFace dataset repository.
Run once after collect_data.py has been run.

    python data/upload_to_hf.py
"""

from huggingface_hub import HfApi

REPO_ID = "Vihaan8/bigcodebench-sdp"
FILE = "data/processed/samples.csv"

api = HfApi()

api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

api.upload_file(
    path_or_fileobj=FILE,
    path_in_repo="samples.csv",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")
