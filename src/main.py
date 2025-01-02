import mlflow
import argparse
from datetime import datetime

from src import settings
from src.test import ModelTester
from src.train import ModelTrainer
from src.predict import ModelPredictor

mlflow.set_tracking_uri(settings.TRACKING_URI)
mlflow.set_experiment("Credit Card Fraud Detection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if args.train:
        with mlflow.start_run(
            run_name="Train" + "/" + datetime.now().strftime("%Y-%m-%d %H:%M")
        ):
            models, test = ModelTrainer().train_all_models()
            ModelTester(models, test).test_all_models()

    elif args.predict:
        with mlflow.start_run(
            run_name="Predict" + "/" + datetime.now().strftime("%Y-%m-%d %H:%M")
        ):
            ModelPredictor().predict_all_models()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
