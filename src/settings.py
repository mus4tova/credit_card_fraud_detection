import os.path
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)

PREDICTIONS_DIR = BASE_DIR / "predictions"
if not os.path.isdir(PREDICTIONS_DIR):
    os.mkdir(PREDICTIONS_DIR)

# MLflow
# TRACKING_URI = settings.MLFLOW_TRACKING_URI

# columns
# drop features "V14", "V16", "V21", "V23", "V24", "V26", "Time"
FEATURES = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V15",
    "V17",
    "V18",
    "V19",
    "V20",
    "V22",
    "V25",
    "V27",
    "V28",
    "Amount",
]
TARGET = ["Class"]
MODELS_LIST = ["Boosting", "RandomForest", "Encoder"]
SAMPLING_STRATEGY = 0.5
