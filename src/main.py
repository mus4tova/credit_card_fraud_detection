import argparse

from src.test import ModelTester
from src.train import ModelTrainer
from src.predict import ModelPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if args.train:
        models, test = ModelTrainer().train_all_models()
        ModelTester(models, test).test_all_models()

    elif args.predict:
        ModelPredictor().predict_all_models()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
