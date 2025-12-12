import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER", "JefferyMendis")
SPACE_NAME = os.getenv("SPACE_NAME", "tourism-app-space")   # use hyphens
LOCAL_FOLDER = os.getenv("LOCAL_FOLDER", "tourism_project/deployment")
REPO_ID = f"{HF_USER}/{SPACE_NAME}"
REPO_TYPE = "space"

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN environment variable (write scope).")

api = HfApi(token=HF_TOKEN)

# Use docker SDK here — requires a Dockerfile in LOCAL_FOLDER
SPACE_SDK = "docker"

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Space '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    try:
        print(f"Creating Space '{REPO_ID}' with SDK='{SPACE_SDK}' ...")
        create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, space_sdk=SPACE_SDK)
        print("Space created.")
    except Exception as e:
        raise RuntimeError(f"Failed to create Space: {e}") from e

# Upload files to the Space
try:
    api.upload_folder(folder_path=LOCAL_FOLDER, repo_id=REPO_ID, repo_type="space", path_in_repo="")
    print("Upload complete.")
except Exception as e:
    raise RuntimeError(f"Failed to upload folder to Space: {e}") from e
