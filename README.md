# Credit card fraud detection

The predictor is designed to detect fraudulent credit card activity. To do this, it uses three model architectures: Random Forest, CatBoost, and Autoencoder + Logistic Regression. In the documentation, the model using the combination of Autoencoder + Logistic Regression will often be referred to simply as Encoder. The predictor is launched via the terminal. All information about the training process and predictions is saved in MLflow.

## Models traning

### Launching the predictor for training

```bash
podman-compose -f docker-compose.yml build
podman-compose -f docker-compose.yml up train
```
All data for training and prediction is taken from a file **creditcard.scv**.
Information about the progress of model training is collected in MLFlow local host http://127.0.0.1:8080.

Default experiment name: **Antifraud test / current date and time**.


### List of features that the predictor uses to train models

* V1
* V2
* V3
* V4
* V5
* V6
* V7
* V8
* V9
* V10
* V11
* V12
* V13
* V15
* V17
* V18
* V19
* V20
* V22
* V25
* V27
* V28
* Amount

Target column - **Class**.
All features have a float numeric data type.


### Models saving

Model files and the file with thresholds are saved in the folder `models`. 


## Predictions

### Launch predictor for predictions

```bash
podman-compose -f docker-compose.yml build
podman-compose -f docker-compose.yml up predict
```

### Getting Predictions
In the predict.py file, the data for predictions is loaded from the **creditcard.scv** file as a pandas DataFrame. And the model predictions are returned as a numpy array.

We have 3 types of models to predict:  RandomForest (sklearn library), Catboost, Encoder model (consist of two parts - first one is keras Autoencoder for getting hidden representation, second one -sklearn Logistic Regression to get final prediction).

Predictions are saved in the `predictions` folder with the name `{model_type}.json`, where model_type takes one of three options - **'RandomForest'**, **'Boosting'**, **'Encoder'**.

In the prediction file, 1 means the operation is fraudulent, 0 means the operation is clean.

## Detailed description
### src/main.py
File for running the project. It has two arguments `--train` for training and `--predict` for obtaining predictions. In the first case (in case of models training), an experiment called "Antifraud test" is created in MlFlow, if it has not yet been created. There the main parameters and metrics of the run will be recorded. The run name is formed from the current date and time.

Then the models training is started - the `train_all_models()` method of the `ModelTrainer()` class from the `train.py` file. The results of the method are a dictionary with 4 models and a test dataset.

After training, the models are tested. The `test_all_models()` method of the `ModelTester(models, test)` class from the `test.py` file is called. The `ModelTester(models, test)` class receives as input a dictionary of models and a test dataset.

When running a project to obtain predictions, i.e. when the argument is `--predict`, the "Antifraud predict" experiment is created in MlFlow. The `predict_all_models()` method of the `ModelPredictor()` class of the `predict.py` file is launched.


### src/train.py
The file where the training of all 3 models takes place. The key method of the file is `train_all_models()`. The main actions for collecting datasets and starting training take place here.

First of all, the data is split into training and testing by calling the `get_train_test()` method of the `DataPreprocessor()` class from the `dataset.py` file.

Then the test dataset is split into features (`X_test`) and target (`y_test`).

For the training dataset, splitting into features and target occurs differently, since the Encoder model has a more complex training structure than RandomForest and Boosting, and therefore we cannot use the same datasets.
Therefore, the training datasets are further split into two:
1) for RandomForest and Boosting (they come with the `_base` prefix)
The `train_base()` method of the `DataPreprocessor()` class from the `dataset.py` file is called to split into features and targets.
2) for Encoder (they come with the `_enc` prefix)
The `train_encoder()` method of the `DataPreprocessor()` class is called. It returns `X_enc` (the part of the data consisting only of non-fraud operations for training the encoder); as well as `train_lr` - the remaining part of the training data for training logistic regression based on the autoencoder (consists of `X_norm_lr` and `X_fraud_lr` - data without fraud and with fraud, respectively).

Then the `train_one_model` method is called to train all models simultaneously using multiprocessing.

`train_all_models()` method return a test dataset `test` and a dictionary with trained models `models`, which are needed for further testing calls in the `test.py` file.


### src/dataset.py

The file loads data (the `DataLoader` class) and creates training and test datasets (the `DataPreprocessor` class)

- The `get_base_dataset()` method pre-processes and cleans the data: sets the target column to a numeric type, removes duplicates, fills empty cells, and scales the data.
- The `get_train_test()` method divides data into two datasets - training and testing.
- The `train_base()` method splits the training dataset into features and target, and also causes dataset balancing. The method is used only for the RandomForest and Boosting models.
- The `train_encoder()` method splits the training dataset into a part for the autoencoder and a part for logistic regression. The method is used only for the Encoder model.
- The `xy_split()` method divides the data into features and target.
- The `over_sampling()` method balances the data using **imblearn.over_sampling.SMOTE()**.

### src/model.py

The file defines the architecture of three models. Contains 4 classes.

`Boosting` , `RandomForest` , `Encoder` - classes with a given model architecture and their training, the `ModelSaver` class - for saving models to files.

### src/test.py

A file for testing and evaluating, obtaining metrics. The only class `ModelTester` receives a dictionary with models and a test dataset as input.

Contains 1 class and 7 methods:

- `test_all_models()` - collects a test run for all three models.
- `save_threshold_dict()` saves a dictionary with thresholds.
- `threshold_tuning()` and `threshold_choosing()` select a threshold for models and save it in pickle format.
- `model_evaluate()` calculates metrics.
- `log_metrics()` logs metrics in MlFlow.
- `plot_metrics()` plots graphs.

### src/predict.py

File for obtaining predictions by models. Called separately from the main.py file.


### src/settings.py

Файл, где хранятся важные параметры для быстрого доступа. Такие как директории для сохранения моделей и предсказаний, MlFlow TRACKING_URI, список фич FEATURES, класс таргета TARGET, список моделей MODELS_LIST.


### test_notebook.ipynb
Notebook for preliminary research, exploratory data analysis, data cleaning, features selection, hypothesis testing, building test models, choosing suitable architectures etc.