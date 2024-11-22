import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn_pmml_model.auto_detect import auto_detect_estimator

from src.data_adloox import DataLoaderAdloox
from src.data_sizmek import DataLoaderSizmek
from src.settings import MODELS_DIR, PREDICTIONS_DIR, FEATURES_SIZMEK, FEATURES_ADLOOX


class ModelPredictor:
    predictions_folder: str = PREDICTIONS_DIR

    def __init__(self, data_type: str):
        self.data_type = data_type
        with open(f"{MODELS_DIR}/thresholds.pkl", "rb") as f:
            self.thresholds_dict = pickle.load(f)
        self.encoder = joblib.load(f"{MODELS_DIR}/encoder.joblib")
        if data_type == "sizmek":
            self.features = FEATURES_SIZMEK
        elif data_type == "adloox":
            self.features = FEATURES_ADLOOX
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        if self.data_type == "adloox":
            data = DataLoaderAdloox("predict").load_data()
        elif self.data_type == "sizmek":
            data = DataLoaderSizmek("predict").load_data()
        data = data[self.features]
        data = self.encode_cols(data)
        return data

    def encode_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df["visitor_ip"] = df["visitor_ip"].astype(str)
        obj_cols = df.select_dtypes(include=["object"]).columns
        df[obj_cols] = df[obj_cols].fillna("empty")
        df[obj_cols] = self.encoder.transform(df[obj_cols])
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[num_cols] = df[num_cols].fillna(0)
        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
        return df

    def predict_one_model(self, model_type: str):
        if model_type == "Catboost":
            model = CatBoostClassifier()
            model.load_model(f"{MODELS_DIR}/{model_type}")
        else:
            model = auto_detect_estimator(pmml=f"{MODELS_DIR}/{model_type}.pmml")
        y_pred_prob = model.predict_proba(self.data)[:, 1]
        thrsh = self.thresholds_dict[model_type]
        y_pred = (y_pred_prob >= thrsh) * 1
        logger.info(f"Threshold: {thrsh}")
        logger.info(f"Predicted probability: {y_pred_prob}")
        logger.info(f"Predicted binary: {y_pred}")
        self.save_predictions(y_pred, model_type)

    def predict_all_models(self):
        logger.info("Start predicting")
        models_list = ["RandomForest", "Catboost", "AutoEncoder"]
        for model_type in models_list:
            logger.info(f"model_type: {model_type}")
            self.predict_one_model(model_type)

    def save_predictions(self, y_pred: np.array, model_type: str):
        y_pred = y_pred.tolist()
        result = json.dumps(y_pred)
        pred_file = os.path.join(self.predictions_folder, f"{model_type}.json")
        with open(pred_file, "w") as f:
            json.dump(result, f)
        logger.info(f"Predictions are saved")
