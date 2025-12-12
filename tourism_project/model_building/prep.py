"""
Data Preparation Script for Tourism Package Prediction

Steps performed:
1. Download the raw CSV from the Hugging Face dataset repo (robust hf_hub_download usage)
2. Perform basic data cleaning and label fixes
3. Split the data into train / test sets
4. Save the splits locally
5. Upload the processed split files back to the Hugging Face dataset repo

Notes:
- This script uses hf_hub_download to avoid intermittent issues when reading
  directly from "hf://..." paths (explicit token passing is more reliable).
- Make sure HF_TOKEN is set in the environment before running this script.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# ------------------------
# Configuration
# ------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set. Set it and re-run.")

HF_OWNER = "JefferyMendis"
DATASET_REPO = f"{HF_OWNER}/tourism-package-prediction"
REMOTE_FILENAME = "tourism.csv"  # filename inside the HF dataset repo

# local folders for storing intermediate/processed files
OUT_DIR = Path("tourism_project/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# filenames for splits (will be saved to OUT_DIR)
X_TRAIN_FNAME = OUT_DIR / "X_train.csv"
X_TEST_FNAME = OUT_DIR / "X_test.csv"
Y_TRAIN_FNAME = OUT_DIR / "y_train.csv"
Y_TEST_FNAME = OUT_DIR / "y_test.csv"

# ------------------------
# Helper: download dataset reliably
# ------------------------
def download_dataset(repo_id: str, filename: str, token: str) -> Path:
    """
    Download a file from a Hugging Face dataset repo and return the local path.
    Uses hf_hub_download with an explicit token to avoid hf:// auth issues.
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        return Path(local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename} from {repo_id}: {e}") from e


# ------------------------
# 1) Verify auth & list files (optional but helpful)
# ------------------------
api = HfApi(token=HF_TOKEN)
try:
    who = api.whoami()
    print("Authenticated as:", who.get("name") or who.get("user", {}).get("name"))
except Exception as e:
    print("Warning: could not verify whoami:", e)

try:
    repo_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Files in {DATASET_REPO}:", repo_files)
except RepositoryNotFoundError:
    raise RuntimeError(f"Dataset repository {DATASET_REPO} not found. Create it or fix repo name.")
except Exception as e:
    print("Warning listing repo files (non-fatal):", e)

# ------------------------
# 2) Download the raw CSV to a local path and read it
# ------------------------
print(f"Downloading {REMOTE_FILENAME} from {DATASET_REPO}...")
local_csv = download_dataset(DATASET_REPO, REMOTE_FILENAME, HF_TOKEN)
print("Downloaded to:", local_csv)

df = pd.read_csv(local_csv)
print("Loaded dataset, shape:", df.shape)

# ------------------------
# 3) Basic data cleaning
# ------------------------
# Standardize typos / inconsistent labels
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female", "Female ": "Female"}).astype(str)

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Single": "Unmarried"}).astype(str)

# Drop identifier columns that should not be used as features
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])
    print("Dropped CustomerID column.")

# Basic sanity checks
if "ProdTaken" not in df.columns:
    raise RuntimeError("Target column 'ProdTaken' not found in the dataset.")

# ------------------------
# 4) Prepare features and target, split
# ------------------------
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

print("Split sizes ->",
      "X_train:", X_train.shape,
      "X_test:", X_test.shape,
      "y_train:", y_train.shape,
      "y_test:", y_test.shape)

# ------------------------
# 5) Save processed splits locally
# ------------------------
X_train.to_csv(X_TRAIN_FNAME, index=False)
X_test.to_csv(X_TEST_FNAME, index=False)
y_train.to_csv(Y_TRAIN_FNAME, index=False)
y_test.to_csv(Y_TEST_FNAME, index=False)
print("Saved processed split files to:", OUT_DIR)

# ------------------------
# 6) Upload processed files back to Hugging Face dataset repo
# ------------------------
files_to_upload = [
    (X_TRAIN_FNAME, "X_train.csv"),
    (X_TEST_FNAME,  "X_test.csv"),
    (Y_TRAIN_FNAME, "y_train.csv"),
    (Y_TEST_FNAME,  "y_test.csv"),
]

for local_path, repo_path in files_to_upload:
    try:
        print(f"Uploading {local_path} -> {DATASET_REPO}/{repo_path} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print("Uploaded", repo_path)
    except Exception as e:
        print(f"Failed to upload {local_path} to HF: {e}")

print("Data preparation completed successfully.")
