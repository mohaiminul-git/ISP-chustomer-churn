from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"





config = {
    "file_path": "features.parquet",
    "numeric_cols": ['subscription_age','bill_avg', 'remaining_contract','service_failure_count','download_avg', 'upload_avg', 'download_over_limit','cost_per_usage','num_services_subscribed'],
    "binary_cols": ['is_tv_subscriber', 'is_movie_package_subscriber'],
    "target_col": "churn",
    "test_size": 0.2,
    "random_state": 42,
    "scaler": "standard",
    "models": {
    "logistic_regression": {
        "estimator": LogisticRegression(
            solver="liblinear"
        ),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10]
        }
    },

    "random_forest": {
        "estimator": RandomForestClassifier(),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5]
        }
    },
    # "xgboost" removed: the xgboost package isn't installed in this venv yet.
    # Add it back once you `uv pip install xgboost` and add it to pyproject.toml.
    },
    "cv_folds": 3,
    "mlflow_experiment": "training_pipeline",
    "mlflow_model_name": "fault_classifier",
}









# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
