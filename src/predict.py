import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn_pmml_model.auto_detect import auto_detect_estimator

from src.dataset import DataLoader
from src.settings import MODELS_DIR, PREDICTIONS_DIR, FEATURES, MODELS_LIST


class ModelPredictor:
    predictions_folder: str = PREDICTIONS_DIR

    def __init__(self):
        with open(f"{MODELS_DIR}/thresholds.pkl", "rb") as f:
            self.thresholds_dict = pickle.load(f)
        self.features = FEATURES
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        data = DataLoader("predict").dataset
        data = data[self.features]
        data = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        logger.info(f"Data prepared for prediction: {data}")
        return data

    def predict_one_model(self, model_type: str):
        if model_type == "Catboost":
            model_ctb = CatBoostClassifier()
            model_ctb.load_model(f"{MODELS_DIR}/{model_type}")
            y_pred_prob = model_ctb.predict_proba(self.data)[:, 1]
        elif model_type == "Encoder":
            model_enc = tf.keras.models.load_model("AutoEncoder.keras")
            model_lr = auto_detect_estimator(
                pmml=f"{MODELS_DIR}/LogisticRegression.pmml"
            )
            hid_rep = model_enc.predict(self.data)
            y_pred_prob = model_lr.predict(hid_rep)
        else:
            model_rf = auto_detect_estimator(pmml=f"{MODELS_DIR}/RandomForest.pmml")
            y_pred_prob = model_rf.predict_proba(self.data)[:, 1]

        thrsh = self.thresholds_dict[model_type]
        y_pred = (y_pred_prob >= thrsh) * 1
        logger.info(f"Threshold: {thrsh}")
        logger.info(f"Predicted probability: {y_pred_prob}")
        logger.info(f"Predicted binary: {y_pred}")
        self.save_predictions(y_pred, model_type)

    def predict_all_models(self):
        logger.info("Start predicting")
        for model_type in MODELS_LIST:
            logger.info(f"model_type: {model_type}")
            self.predict_one_model(model_type)
            logger.info("------------")

    def save_predictions(self, y_pred: np.array, model_type: str):
        y_pred = y_pred.tolist()
        result = json.dumps(y_pred)
        pred_file = os.path.join(self.predictions_folder, f"{model_type}.json")
        with open(pred_file, "w") as f:
            json.dump(result, f)
        logger.info(f"Predictions are saved")
