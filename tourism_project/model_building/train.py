"""
Production training script for Tourism Package Prediction (robust & pickle-safe).

This script is defensive across multiple xgboost versions and will try:
  1) estimator.fit(..., callbacks=[EarlyStopping(...)])
  2) estimator.fit(..., early_stopping_rounds=...)
  3) fallback to xgboost.train(...) with DMatrix and evals_result capture

It also:
- Downloads processed splits from the Hugging Face dataset repo (hf_hub_download)
- Builds and fits a preprocessing transformer
- Trains XGBoost (with early stopping)
- Evaluates and logs metrics to MLflow
- Saves the final pipeline and uploads it to the Hugging Face Model Hub

Environment variables required:
- HF_TOKEN: Hugging Face token with write permissions
- MLFLOW_TRACKING_URI: URL of the MLflow tracking server (required in production)

Optional environment variables to tune XGBoost behavior:
- XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_N_JOBS
- XGB_EARLY_STOPPING_ROUNDS, XGB_VERBOSE, PROD_VAL_SIZE, PROD_MODEL_REPO
"""

import os
import json
import types
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow

from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb

# -------------------------
# Top-level helper (picklable) to attach to estimator if needed
# -------------------------
def _get_booster(self):
    """Return attached Booster if present, else None."""
    return getattr(self, "_Booster", None)

# -------------------------
# Configuration & env checks
# -------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set. Set it and re-run.")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI must be set in the production environment.")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("tourism-prediction-prod")

HF_OWNER = "JefferyMendis"
DATASET_REPO = f"{HF_OWNER}/tourism-package-prediction"
REMOTE_FILES = {
    "X_train": "X_train.csv",
    "X_test":  "X_test.csv",
    "y_train": "y_train.csv",
    "y_test":  "y_test.csv",
}

OUT_DIR = Path("tourism_project/model_building")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = os.getenv("PROD_MODEL_FILENAME", "model_prod.joblib")
LOCAL_MODEL_PATH = OUT_DIR / MODEL_FILENAME

# XGBoost / training hyperparams (can be overridden via env)
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", 200))
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", 5))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", 0.05))
XGB_N_JOBS = int(os.getenv("XGB_N_JOBS", -1))
EARLY_STOPPING_ROUNDS = int(os.getenv("XGB_EARLY_STOPPING_ROUNDS", 20))
VAL_SIZE = float(os.getenv("PROD_VAL_SIZE", 0.1))
XGB_VERBOSE = int(os.getenv("XGB_VERBOSE", 10))

# -------------------------
# Helper: download splits reliably
# -------------------------
def download_and_read_csv(repo_id: str, filename: str, token: str) -> pd.DataFrame:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    return pd.read_csv(local_path)

# -------------------------
# 1) Load full splits
# -------------------------
print("Downloading full train/test splits from Hugging Face dataset repo...")
X_train = download_and_read_csv(DATASET_REPO, REMOTE_FILES["X_train"], HF_TOKEN)
y_train = download_and_read_csv(DATASET_REPO, REMOTE_FILES["y_train"], HF_TOKEN).squeeze()

X_test = download_and_read_csv(DATASET_REPO, REMOTE_FILES["X_test"], HF_TOKEN)
y_test = download_and_read_csv(DATASET_REPO, REMOTE_FILES["y_test"], HF_TOKEN).squeeze()

print("Shapes ->", "X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)

# Ensure target arrays are 1-D numpy arrays for xgboost
y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()

# -------------------------
# 2) Feature lists (adjust if schema differs)
# -------------------------
numeric_features = [c for c in [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome'
] if c in X_train.columns]

categorical_features = [c for c in [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation', 'CityTier', 'PreferredPropertyStar'
] if c in X_train.columns]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# -------------------------
# 3) Build preprocessor (ColumnTransformer)
# -------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features) if numeric_features else (),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) if categorical_features else (),
    remainder='passthrough'
)

# -------------------------
# 4) Create XGBoost estimator (production hyperparams from tuning)
# -------------------------
xgb_params = {
    'n_estimators': XGB_N_ESTIMATORS,
    'max_depth': XGB_MAX_DEPTH,
    'learning_rate': XGB_LEARNING_RATE,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': XGB_N_JOBS,
}

xgb_model = xgb.XGBClassifier(**xgb_params)

# -------------------------
# 5) Prepare validation split for early stopping
# -------------------------
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VAL_SIZE,
    random_state=42,
    stratify=y_train if np.unique(y_train).size > 1 else None
)

print("Train/val shapes:", X_train_full.shape, X_val.shape)

# -------------------------
# 6) Fit preprocessor and transform sets
# -------------------------
print("Fitting preprocessor on the full training fold...")
preprocessor.fit(X_train_full)

X_train_t = preprocessor.transform(X_train_full)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

# -------------------------
# Helper: robust training with early stopping across xgboost versions
# -------------------------
def train_xgb_with_early_stopping(estimator, X_tr, y_tr, X_val_, y_val_, early_rounds, verbose):
    """
    Try multiple strategies for early stopping:
    1) estimator.fit(..., callbacks=[xgb.callback.EarlyStopping(...)])  # modern API
    2) estimator.fit(..., early_stopping_rounds=..., eval_set=[...])    # older sklearn wrapper
    3) fallback to xgboost.train with DMatrix and explicit evals_result dict

    Returns (estimator, evals_result_dict_or_None)
    """
    # Strategy 1: callbacks-based training
    try:
        es_cb = xgb.callback.EarlyStopping(rounds=early_rounds, save_best=True)
        estimator.fit(X_tr, y_tr, eval_set=[(X_val_, y_val_)], callbacks=[es_cb], verbose=verbose)
        evals_result = None
        try:
            evals_result = estimator.evals_result() if hasattr(estimator, "evals_result") else None
        except Exception:
            evals_result = None
        return estimator, evals_result
    except TypeError as e1:
        print("Callback-based fit not supported or failed:", e1)
    except Exception as e:
        print("Callback-based fit raised:", e)

    # Strategy 2: older sklearn wrapper signature
    try:
        estimator.fit(X_tr, y_tr, eval_set=[(X_val_, y_val_)], early_stopping_rounds=early_rounds, verbose=verbose)
        evals_result = None
        try:
            evals_result = estimator.evals_result() if hasattr(estimator, "evals_result") else None
        except Exception:
            evals_result = None
        return estimator, evals_result
    except TypeError as e2:
        print("early_stopping_rounds fit not supported or failed:", e2)
    except Exception as e:
        print("early_stopping_rounds fit raised:", e)

    # Strategy 3: fallback to xgboost.train with DMatrix and explicit evals_result dict
    try:
        print("Falling back to xgboost.train API (DMatrix).")
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val_, label=y_val_)

        params = estimator.get_xgb_params()
        # sklearn wrapper may store some params separately; ensure n_estimators available
        num_boost_round = int(estimator.get_params().get("n_estimators", params.pop("n_estimators", 100)))

        # Ensure objective present
        if "objective" not in params:
            params["objective"] = "binary:logistic" if np.unique(y_tr).size == 2 else "multi:softprob"

        evals_result = {}
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=early_rounds,
            evals_result=evals_result,
            verbose_eval=verbose
        )

        # Attach booster to sklearn wrapper so predict()/predict_proba() still work
        try:
            estimator._Booster = booster
        except Exception:
            pass

        # Bind top-level _get_booster method so it's picklable
        try:
            estimator.get_booster = types.MethodType(_get_booster, estimator)
        except Exception:
            pass

        return estimator, evals_result
    except Exception as e:
        raise RuntimeError(f"All training strategies failed: {e}") from e

# -------------------------
# 7) Train with robust method
# -------------------------
print("Training XGBoost estimator with robust early stopping...")
xgb_model, evals_result = train_xgb_with_early_stopping(
    xgb_model, X_train_t, y_train_full, X_val_t, y_val, EARLY_STOPPING_ROUNDS, XGB_VERBOSE
)

# After training, assemble final pipeline with fitted preprocessor + (possibly updated) fitted estimator
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgbclassifier", xgb_model)
])

# -------------------------
# 8) Evaluate on test set
# -------------------------
print("Evaluating on the test set...")
y_pred = final_pipeline.predict(X_test)
y_proba = None
try:
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
except Exception:
    y_proba = None

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba) if (y_proba is not None and np.unique(y_test).size == 2) else None

print(f"Test metrics -> accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, roc_auc: {roc_auc}")

# Optionally get best iteration / evals_result for logging
best_iteration = None
try:
    booster = None
    if hasattr(xgb_model, "get_booster"):
        booster = xgb_model.get_booster()
    elif hasattr(xgb_model, "_Booster"):
        booster = getattr(xgb_model, "_Booster", None)

    if booster is not None:
        best_iteration = getattr(booster, "best_iteration", None)
except Exception:
    pass

if best_iteration is not None:
    print("XGBoost best_iteration:", best_iteration)

# -------------------------
# 9) Log to MLflow and save artifact
# -------------------------
with mlflow.start_run():
    mlflow.log_param("mode", "prod")
    mlflow.log_params(xgb_params)
    mlflow.log_param("dataset_repo", DATASET_REPO)
    mlflow.log_param("val_size", VAL_SIZE)
    mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
    if best_iteration is not None:
        mlflow.log_param("best_iteration", int(best_iteration))

    mlflow.log_metric("accuracy", float(acc))
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("f1", float(f1))
    if roc_auc is not None:
        mlflow.log_metric("roc_auc", float(roc_auc))

    # Log evals_result (training history) if available
    if evals_result:
        try:
            hist_path = OUT_DIR / "evals_result.json"
            with open(hist_path, "w") as fh:
                json.dump(evals_result, fh)
            mlflow.log_artifact(str(hist_path))
        except Exception:
            pass

    # Save pipeline artifact (pickle-safe now)
    joblib.dump(final_pipeline, LOCAL_MODEL_PATH)
    mlflow.log_artifact(str(LOCAL_MODEL_PATH))
    print("Saved and logged model artifact to MLflow:", LOCAL_MODEL_PATH)

# -------------------------
# 10) Upload model to Hugging Face Model Hub
# -------------------------
HF_MODEL_REPO = os.getenv("PROD_MODEL_REPO", f"{HF_OWNER}/tourism-package-model")
print(f"Uploading model to Hugging Face model repo: {HF_MODEL_REPO}")

api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print("Model repo already exists.")
except RepositoryNotFoundError:
    print("Model repo not found. Creating a new repository...")
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)
    print("Model repo created.")

# Upload serialized final pipeline
try:
    api.upload_file(
        path_or_fileobj=str(LOCAL_MODEL_PATH),
        path_in_repo=LOCAL_MODEL_PATH.name,
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )
    print("Model uploaded successfully to Hugging Face.")
except Exception as e:
    print("Failed to upload model to Hugging Face:", str(e))

print("Production training finished successfully.")
