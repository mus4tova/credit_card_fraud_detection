import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from collections import Counter
from imblearn.over_sampling import SMOTE
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

from src.settings import FEATURES, TARGET, BASE_DIR


class DataLoader:
    def __init__(self, mode: str):
        # download file from kaggle
        self.download_file()
        # read file with dataset
        self.dataset = self.load_data_from_file()
        # split dataset into train and test
        self.length = int(len(self.dataset) * 0.95)
        if mode == "train":
            self.dataset = self.dataset[: self.length]
        elif mode == "predict":
            self.dataset = self.dataset[self.length :]

    def is_folder_empty(self, folder_path: str) -> bool:
        return not any(Path(folder_path).iterdir())

    def download_file(self):
        # initialize API
        api = KaggleApi()
        api.authenticate()

        # set path and name
        dataset_name = "mlg-ulb/creditcardfraud"
        save_path = f"{BASE_DIR}/datasets/"

        # download data
        if self.is_folder_empty(save_path):
            api.dataset_download_files(dataset_name, path=save_path, unzip=True)
            logger.info(f"Dataset downloaded and saved at {save_path}")
        else:
            logger.info(f"Dataset is already downloaded")

    def load_data_from_file(self) -> pd.DataFrame:
        df = pd.read_csv(f"{BASE_DIR}/datasets/creditcard.csv")
        logger.info(f"Dataframe: \n{df}")
        return df


class DataPreprocessor:
    def __init__(self):
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.loader = DataLoader("train")
        self.target = TARGET[0]

    def get_base_dataset(self) -> pd.DataFrame:
        self.data = self.loader.dataset

        # drop columns where fraud class almost have no differences from non-fraud
        self.data = self.data[FEATURES + TARGET]

        # clean from duplicated lines if any
        self.data.drop_duplicates(inplace=True)

        # fill empty values for numerical columns
        num_cols = self.data.select_dtypes(include=["float64", "int64"]).columns
        self.data[num_cols] = self.data[num_cols].fillna(0)

        # scaling data
        self.data = pd.DataFrame(
            MinMaxScaler().fit_transform(self.data), columns=self.data.columns
        )

        # make sure that target column is integer
        self.data[self.target] = self.data[self.target].values.astype(np.int64)
        return self.data

    def get_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """common method for all 3 models, return two dataframes"""
        data = self.get_base_dataset()
        data_len = int(len(data) * 0.8)
        train, test = data[:data_len], data[data_len:]

        logger.info(f"Shape of test dataset: {test.shape}")
        mlflow.log_param("Shape of test dataset", test.shape)
        logger.info(f"Shape of train dataset: {train.shape}")
        mlflow.log_param("Shape of train dataset", train.shape)
        logger.info(f"Number of values in test: {Counter(test[self.target])}")
        mlflow.log_param("Number of values in test", Counter(test[self.target]))
        logger.info(f"Number of values in train: {Counter(train[self.target])}")
        return train, test

    def train_base(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """method for catboost and random forest, is used after self.get_train_test(), return two arrays"""
        X_train, y_train = self.xy_split(train)
        logger.info(f"Shape of X_train - base before balancing: {X_train.shape}")
        logger.info(
            f"Number of values in X_train - base before balancing: {Counter(y_train)}"
        )
        mlflow.log_param(
            "Number of values in X_train - base before balancing", Counter(y_train)
        )

        # balancing data
        X_train, y_train = self.over_sampling(X_train, y_train)
        logger.info(f"Shape of X_train - base after balancing: {X_train.shape}")
        logger.info(
            f"Number of values in X_train - base after balancing: {Counter(y_train)}"
        )
        mlflow.log_param(
            "Number of values in X_train - base after balancing", Counter(y_train)
        )

        train = X_train, y_train
        return train

    def train_encoder(self, df: pd.DataFrame) -> tuple[
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ]:
        """method for encoder+lr model, is used after self.get_train_test(),
        return two dataframes - for autoencoder and for logistic regression"""
        # choose small sample to train unsupervised model - autoencoder
        sample_len = int(len(df) * 0.3)
        ds_enc = df[df[self.target] == 0][:sample_len]

        # the rest of the dataset is trained using supervised learning - logistic regression
        non_fraud = df[df[self.target] == 0][sample_len:]
        fraud = df[df[self.target] == 1]
        ds_lr = pd.concat([non_fraud, fraud]).reset_index(drop=True)

        # sample for autoencoders
        X_enc, y_enc = self.xy_split(ds_enc)
        logger.info(f"Shape of X_enc - autoencoder: {X_enc.shape}")
        logger.info(f"Number of values in X_enc - autoencoder: {Counter(y_enc)}")
        mlflow.log_param("Number of values in X_enc - autoencoder", Counter(y_enc))

        # sample for logistic regression
        X_lr, y_lr = self.xy_split(ds_lr)
        # consider two classes separately so they do not mix
        X_norm_lr, X_fraud_lr = (
            X_lr[y_lr == 0],
            X_lr[y_lr == 1],
        )

        train_enc = X_enc
        train_lr = X_norm_lr, X_fraud_lr
        return train_enc, train_lr

    def xy_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """method for split data into features and target"""
        X = df.drop(columns=self.target, axis=1)
        y = df[self.target]
        return X, y

    def over_sampling(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """method for balancing data"""
        logger.info(f"Shape before balancing: {Counter(y)}")
        X_bal, y_bal = SMOTE(sampling_strategy=0.5).fit_resample(X, y)
        logger.info(f"Shape after balancing: {Counter(y_bal)}")
        return X_bal, y_bal
