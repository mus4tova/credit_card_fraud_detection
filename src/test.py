import pickle
import numpy as np
import pandas as pd
from time import time
from loguru import logger
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_metric, log_figure, log_param
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import tensorflow as tf
from src.settings import MODELS_DIR, MODELS_LIST


class ModelTester:
    def __init__(
        self,
        models: dict[str, Pipeline | CatBoostClassifier],
        dataset: tuple[pd.DataFrame, pd.DataFrame],
    ):
        self.test_dataset = dataset
        self.models = models

    def function_time(func):
        def wrapper(*args):
            start_time = time()
            value = func(*args)
            end_time = time()
            print(f"Function execution time {end_time - start_time} sec.")
            return value

        return wrapper

    def test_all_models(self):
        logger.info("==========================================")
        logger.info("Start testing")

        thresholds_dict = {}
        X_test, y_test = self.test_dataset
        logger.info(f"X_test.shape:{X_test.shape}")
        for model_name in MODELS_LIST:
            if model_name == "Encoder":
                model_enc = self.models["AutoEncoder"]
                model_lr = self.models["LogisticRegression"]
                hid_rep = model_enc.predict(X_test)
                y_pred_prob = model_lr(
                    hid_rep, train=False
                )  # model_lr.predict(hid_rep)

            else:
                logger.info(f"Start testing {model_name}")
                model = self.models[model_name]
                y_pred_prob = model.predict(X_test)
            logger.info(f"y_pred_prob: {y_pred_prob}")

            threshold = self.threshold_tuning(y_pred_prob, y_test, model_name)
            thresholds_dict[model_name] = threshold
            y_pred = tf.cast(
                y_pred_prob > threshold, tf.float32
            )  # (y_pred_prob >= threshold) * 1
            logger.info(f"y_pred: {y_pred}")
            self.cm, self.rec, self.pr, self.acc = self.model_evaluate(y_test, y_pred)
            self.log_metrics(model_name)
            self.log_graphics(y_test, y_pred_prob, model_name)
            logger.info("------------------------------------------")
        logger.info("Finish testing")
        self.save_threshold_dict(thresholds_dict)
        logger.info(f"all")

    def save_threshold_dict(self, thresholds_dict: dict[str, float]):
        logger.info(f"saving threshold")
        with open(f"{MODELS_DIR}/thresholds.pkl", "wb") as f:
            pickle.dump(thresholds_dict, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"threshold is saved")

    @function_time
    def predict(
        self,
        model: LogisticRegression | RandomForestClassifier | CatBoostClassifier,
        X_test: np.ndarray[np.ndarray],
    ) -> np.ndarray:
        y_pred = model.predict(X_test)
        return y_pred

    def threshold_tuning(
        self, y_pred_prob: np.ndarray, y_test: np.ndarray, model_name: str
    ):
        logger.info(f"model_name: {model_name}")
        thrsh_ds = pd.DataFrame()
        times = 1000
        thrsh = 0
        for i in range(times):
            gap = max(np.median(y_pred_prob), np.mean(y_pred_prob)) / times
            preds = tf.cast(
                y_pred_prob > thrsh, tf.float32
            )  # (y_pred_prob >= thrsh) * 1
            cm, rec, pr, acc = self.model_evaluate(y_test, preds)
            thrsh_ds.loc[i, "threshold"] = thrsh
            thrsh_ds.loc[i, "recall"] = rec
            thrsh += gap
        logger.info(f"thrsh_ds: {thrsh_ds}")
        threshold = self.threshold_choosing(thrsh_ds, model_name)
        return threshold

    def threshold_choosing(self, thrsh_ds: pd.DataFrame, model_name: str):
        slice = thrsh_ds[thrsh_ds.recall >= 0.75]
        logger.info(f"slice: {slice}")
        thrsh = slice.tail(1).threshold.values[0]
        logger.info(f"Threshold: {thrsh}")
        log_param(f"Threshold for {model_name}", thrsh)
        return thrsh

    def model_evaluate(self, y_test: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        pr = precision_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return cm, rec, pr, acc

    def log_metrics(self, model_name: str):
        log_metric(f"{model_name}_accuracy", self.acc)
        log_metric(f"{model_name}_precision", self.pr)
        log_metric(f"{model_name}_recall", self.rec)

    def log_graphics(
        self, y_test: np.ndarray, y_pred_prob: np.ndarray, model_name: str
    ):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        pr_curve = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc_curve = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot()
        log_figure(pr_curve.figure_, f"{model_name}_PR_curve.png")
        log_figure(roc_auc_curve.figure_, f"{model_name}_ROC_curve.png")
        log_figure(disp.figure_, f"{model_name}_confusion_matrix.png")
