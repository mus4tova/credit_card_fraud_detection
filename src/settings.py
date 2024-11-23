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
DROP_FEATURES = ["V14", "V16", "V21", "V23", "V24", "V26", "Time"]
TARGET = ["Class"]

MODELS_LIST = ["Encoder", "Boosting", "RandomForest"]

SAMPLING_STRATEGY = 0.5
