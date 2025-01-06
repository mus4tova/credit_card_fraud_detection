import keras
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from nyoka import skl_to_pmml
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.settings import MODELS_DIR


class Boosting:
    model: CatBoostClassifier

    def __init__(self, train: tuple[pd.DataFrame, pd.DataFrame], features: list[str]):
        self.features = features
        self.X_train, self.y_train = train

    def train(self) -> CatBoostClassifier:
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.02,
            depth=12,
            eval_metric="AUC",
            random_seed=2,
            bagging_temperature=0.2,
            od_type="Iter",
            metric_period=50,
            od_wait=100,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.05
        )
        val_data = X_val, y_val
        model.fit(
            X_train,
            y_train,
            eval_set=val_data,
            use_best_model=True,
        )
        ModelSaver().save_model(model, self.features, "Boosting")
        return model


class RandomForest:
    model: Pipeline

    def __init__(self, train: tuple[pd.DataFrame, pd.DataFrame], features: list[str]):
        self.features = features
        self.X_train, self.y_train = train

    def train(self) -> Pipeline:
        model = Pipeline([("rfc", RandomForestClassifier())])
        model.fit(self.X_train, self.y_train)
        ModelSaver().save_model(model, self.features, "RandomForest")
        return model


class Encoder:
    def __init__(
        self,
        train_enc: pd.DataFrame,
        train_lr: tuple[pd.DataFrame, pd.DataFrame],
        features: list[str],
    ):
        self.features = features
        self.X_enc = train_enc
        self.X_norm_lr, self.X_fraud_lr = train_lr

    def train(self) -> tuple[keras.Model, Pipeline]:
        # train 1st model - autoencoder
        enc_model = self.autoenc_model()
        ModelSaver().save_model(enc_model, self.features, "AutoEncoder")
        # get hidden representation using autoencoder
        hidden_representation = self.hid_representaton(enc_model)
        train = self.use_hid_rep(hidden_representation)
        # train 2nd model - logistic regression - by hidden representation
        lr_model = self.logistic_regr_model(train)
        ModelSaver().save_model(lr_model, self.features, "LogisticRegression")
        return enc_model, lr_model

    def use_hid_rep(
        self, hidden_representation: keras.Model
    ) -> tuple[np.ndarray[np.ndarray], np.ndarray]:
        # combine fraud and non-fraud into one array X
        hid_rep_norm = hidden_representation.predict(self.X_norm_lr)
        hid_rep_fraud = hidden_representation.predict(self.X_fraud_lr)
        X_rep = np.append(hid_rep_norm, hid_rep_fraud, axis=0)

        # same with y
        y_norm = np.zeros(hid_rep_norm.shape[0])
        y_fraud = np.ones(hid_rep_fraud.shape[0])
        y_rep = np.append(y_norm, y_fraud)
        logger.info(f"Shape of X_rep - logistic regression: {X_rep.shape}")
        logger.info(
            f"Number of values in X_rep - logistic regression: {Counter(y_rep)}"
        )
        mlflow.log_param(
            "Number of values in X_rep - logistic regression", Counter(y_rep)
        )
        return X_rep, y_rep

    def logistic_regr_model(
        self, train: tuple[np.ndarray[np.ndarray], np.ndarray]
    ) -> Pipeline:
        X_train, y_train = train
        model = Pipeline([("lr", LogisticRegression(solver="lbfgs"))])
        model.fit(X_train, y_train)
        return model

    def autoenc_model(self) -> keras.Model:
        input_dim = self.X_enc.shape[1]
        epochs = 50
        batch_size = 128
        val_split = 0.2

        input_layer = Input(shape=(input_dim,))

        # encoder
        encoded = Dense(128, activation="tanh")(input_layer)
        encoded = Dense(64, activation="relu")(encoded)
        encoded = Dense(16, activation="relu")(encoded)
        encoded = Dense(input_dim, activation="relu")(encoded)

        # decoder
        decoded = Dense(16, activation="relu")(encoded)
        decoded = Dense(64, activation="relu")(decoded)
        decoded = Dense(128, activation="tanh")(decoded)

        # output
        output_layer = Dense(input_dim, activation="relu")(decoded)

        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(
            metrics=["accuracy", "precision", "recall"],
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        )
        autoencoder.summary()
        history = autoencoder.fit(
            self.X_enc,
            self.X_enc,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=val_split,
            verbose=1,
        )
        # self.log_history(history)
        for metric in ["loss", "accuracy", "precision", "recall"]:
            self.plot_training_history(history, metric)
        return autoencoder

    def plot_training_history(self, history, metric: str):
        import matplotlib

        matplotlib.use("Agg")

        plt.figure(figsize=(8, 6))
        plt.plot(history.history[metric], label=f"Train {metric.capitalize()}")
        plt.plot(
            history.history[f"val_{metric}"], label=f"Validation {metric.capitalize()}"
        )
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()
        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.savefig(f"training_{metric}_enc.png")
        mlflow.log_artifact(f"training_{metric}_enc.png")
        plt.savefig(f"val_{metric}_enc.png")
        mlflow.log_artifact(f"val_{metric}_enc.png")

    def log_history(self, history):

        for epoch, metrics in enumerate(
            zip(
                history.history["loss"],
                history.history["val_loss"],
                history.history["accuracy"],
                history.history["val_accuracy"],
                history.history["precision"],
                history.history["val_precision"],
                history.history["recall"],
                history.history["val_recall"],
            )
        ):
            (
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                train_pr,
                val_pr,
                train_rec,
                val_rec,
            ) = metrics
            mlflow.log_metric("train_loss_enc", train_loss, step=epoch)
            mlflow.log_metric("val_loss_enc", val_loss, step=epoch)
            mlflow.log_metric("accuracy_enc", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy_enc", val_acc, step=epoch)
            mlflow.log_metric("precision_enc", train_pr, step=epoch)
            mlflow.log_metric("val_precision_enc", val_pr, step=epoch)
            mlflow.log_metric("recall_enc", train_rec, step=epoch)
            mlflow.log_metric("val_recall_enc", val_rec, step=epoch)

    def hid_representaton(self, autoencoder: keras.Model) -> keras.Model:
        # add into hidden representation only encoding part
        hidden_representation = Sequential()
        hidden_representation.add(autoencoder.layers[0])
        hidden_representation.add(autoencoder.layers[1])
        hidden_representation.add(autoencoder.layers[2])
        hidden_representation.add(autoencoder.layers[3])
        hidden_representation.add(autoencoder.layers[4])
        return hidden_representation


class ModelSaver:
    def __init__(self):
        pass

    def save_model(
        self,
        model: keras.Model | Pipeline | CatBoostClassifier,
        features: list[str],
        model_type: str,
    ):
        folder = f"{MODELS_DIR}/{model_type}"
        if model_type == "Boosting":
            model.save_model(f"{folder}")
        elif model_type == "AutoEncoder":
            model.save(f"{folder}.keras")
        else:
            skl_to_pmml(
                model,
                features,
                "traffic_filtration",
                folder + ".pmml",
            )
