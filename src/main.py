import mlflow
import argparse
from datetime import datetime

from src.train import ModelTrainer
from src.test import ModelTester

# from src.predict import ModelPredictor
from src import settings

# mlflow.set_tracking_uri(settings.TRACKING_URI)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    # if args.train:
    # experiment_name = "Antifraud test"
    # mlflow.set_experiment(experiment_name)
    # with mlflow.start_run(run_name=datetime.now().strftime("%Y-%m-%d %H:%M")):
    models, test = ModelTrainer().train_all_models()
    ModelTester(models, test).test_all_models()

    # elif args.predict:
    # experiment_name = "Antifraud predict"
    # mlflow.set_experiment(experiment_name)
    # ModelPredictor().predict_all_models()

    # else:
    # parser.print_help()


if __name__ == "__main__":
    main()
