
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# -------------------------------------------------------------------
# Define the Hugging Face dataset repository details
# repo_id must follow format: <username>/<repository-name>
# repo_type = "dataset" tells Hugging Face this repo stores dataset files
# -------------------------------------------------------------------
repo_id = "JefferyMendis/tourism-package-prediction"
repo_type = "dataset"

os.environ["HF_TOKEN"] = "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"

# -------------------------------------------------------------------
# Initialize the Hugging Face API client
# The HF token must be stored as an environment variable: HF_TOKEN
# Never hardcode tokens inside source code.
# -------------------------------------------------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -------------------------------------------------------------------
# Step 1: Check whether the dataset repository already exists.
# - If it exists → proceed with upload
# - If not → create a new dataset repository
# -------------------------------------------------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating a new one...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print(f"Dataset repository '{repo_id}' created successfully.")

# -------------------------------------------------------------------
# Step 2: Upload the contents of the local data folder
# This will push all files inside tourism_project/data/ to Hugging Face
# -------------------------------------------------------------------
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded to Hugging Face successfully.")
