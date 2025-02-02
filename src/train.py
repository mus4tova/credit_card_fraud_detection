import time
import mlflow
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from multiprocessing.pool import ThreadPool

from src.settings import MODELS_LIST
from src.dataset import DataPreprocessor
from src.model import Boosting, RandomForest, Encoder


class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models_list = MODELS_LIST

    def train_all_models(
        self,
    ) -> tuple[
        dict[str, Pipeline or CatBoostClassifier],
        tuple[pd.DataFrame, pd.DataFrame],
    ]:
        # dataset for boosting and random forest
        train, test = self.preprocessor.get_train_test()

        # create test dataset
        X_test, y_test = self.preprocessor.xy_split(test)
        test = X_test, y_test

        # save features
        self.features = list(X_test.columns)
        logger.info(f"Features: {self.features}")

        # rf, catboost train sample
        self.train_base = self.preprocessor.train_base(train)
        # autoencoder train sample
        self.X_enc, self.train_lr = self.preprocessor.train_encoder(train)

        self.models = {}
        logger.info("Start multiprocessing training")
        start = time.time()
        with ThreadPool() as p:
            p.map(self.train_one_model, self.models_list)
        end = time.time() - start
        logger.info(f"Running time (sec) using multiprocessing: {end}")
        logger.info("Finish multiprocessing training")

        logger.info("==========================================")
        return self.models, test

    def train_one_model(self, model_name: str):
        if model_name == "RandomForest":
            logger.info(f"RandomForest")
            random_forest = RandomForest(self.train_base, self.features).train()
            self.models["RandomForest"] = random_forest
        elif model_name == "Boosting":
            logger.info(f"Boosting")
            catboost = Boosting(self.train_base, self.features).train()
            self.models["Boosting"] = catboost
        elif model_name == "Encoder":
            logger.info(f"Encoder")
            autoencoder, lr_model = Encoder(
                self.X_enc, self.train_lr, self.features
            ).train()
            self.models["AutoEncoder"] = autoencoder
            self.models["LogisticRegression"] = lr_model
