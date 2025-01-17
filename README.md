# Credit card fraud detection

This predictor is designed to detect fraudulent credit card activity. It uses three model architectures: Random Forest, CatBoost, and Autoencoder + Logistic Regression. In the documentation, the model that combines Autoencoder + Logistic Regression is often referred to simply as "Encoder." The predictor is launched using Docker.

## Models traning
Before launching, you can either manually save the dataset (link to dataset https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), or click the `Create New Token` button in the `API` section using the link https://www.kaggle.com/settings/account and place the downloaded file in the folder `secrets`
If you decide to save dataset manually, then remove next string from `docker-compose` file:

`- ./secrets/kaggle.json:/root/.config/kaggle/kaggle.json`

### Launching the predictor for training

```bash
docker-compose.yml build
docker-compose.yml up train
```
Training progress is monitored in MLFlow on the local host: http://13.60.52.168:5000. 
Default experiment name: **Credit Card Fraud Detection**.


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
All features are of type float.


### Models saving
Model files and threshold data are saved in the `models` folder.


## Predictions
### Launch predictor for predictions

```bash
docker-compose.yml up predict
```

### Getting Predictions
In the `predict.py` file, data for predictions is loaded from the **creditcard.scv** file as a Pandas DataFrame. Model predictions are returned as a NumPy array.

Three model types are available for predictions:

* RandomForest (using the sklearn library).
* CatBoost.
* Encoder Model (consists of two parts: a Keras Autoencoder for hidden representation and an sklearn Logistic Regression for final predictions).

Predictions are saved in the `predictions` folder with filenames in the format `{model_type}.json`, where model_type is one of the following:  - **'RandomForest'**, **'Boosting'**, **'Encoder'**.

In the prediction output, 1 indicates a fraudulent operation, and 0 indicates a non-fraudulent operation.


## Detailed description
### src/main.py
This file is the entry point for running the project. It accepts two arguments:
`--train` for training the models.
`--predict` for obtaining predictions.

Training:
* When the argument is `--train`, an experiment named "Antifraud Test" is created in MLFlow (if not already created). Model parameters and metrics are logged. The run name is based on the current date and time.
* The `train_all_models()` method of the `ModelTrainer` class in the `train.py` file is called to train the models. This method returns a dictionary of 4 trained models and a test dataset.
* After training, the `test_all_models()` method of the `ModelTester` class in the `test.py` file is invoked to test the models.

Prediction:
* When the argument is `--predict`, an experiment named "Antifraud Predict" is created in MLFlow.
* The `predict_all_models()` method of the `ModelPredictor` class in the `predict.py` file is called to generate predictions.


### src/train.py
This file handles the training of all three models.

Key Method:
* `train_all_models()` coordinates the entire training process. It first splits the data into training and testing sets using the `get_train_test()` method from the `DataPreprocessor` class in the `dataset.py` file. Then the test dataset is split into features (`X_test`) and target (`y_test`).

Training Data Splits:

For the training dataset, splitting into features and target occurs differently, since the Encoder model has a more complex training structure than RandomForest and Boosting, and therefore we cannot use the same datasets.

RandomForest and CatBoost:
* The `train_base()` method of the `DataPreprocessor` class is used to split the data.
Encoder Model:
* The `train_encoder()` method splits the data into:
X_enc: Non-fraud data for training the autoencoder.
train_lr: Data for training logistic regression (split into X_norm_lr and X_fraud_lr for non-fraud and fraud data, respectively).

Training Execution:
* Models are trained in parallel using multiprocessing with the `train_one_model()` method. The method returns a dictionary of trained models and a test dataset for further evaluation.


### src/dataset.py
This file handles data loading (`DataLoader` class) and preprocessing(`DataPreprocessor` class).
 
- The `download_file()` retrieves data from Kaggle and saves it as creditcard.csv.
- The `load_data_from_file()` reads the dataset from the creditcard.csv.
- The `get_base_dataset()` preprocesses the data, removes duplicates, fills missing values, and scales features.
- The `get_train_test()` splits the data into training and testing sets.
- The `train_base()` prepares the dataset for RandomForest and CatBoost.
- The `train_encoder()` prepares data specifically for the Encoder model.
- The `xy_split()` separates features and the target column.
- The `over_sampling()` balances the dataset using SMOTE (from the imblearn library).


### src/model.py
Defines the architecture and training logic for the models.

Classes:
* `Boosting` 
* `RandomForest` 
* `Encoder` 
* `ModelSaver` - for saving model files.


### src/test.py
Handles testing and evaluation of models.
Class: `ModelTester`

Key Methods:
- `test_all_models()` - tests all three models.
- `save_threshold_dict()` saves threshold values.
- `threshold_tuning()` and `threshold_choosing()` determine and save optimal thresholds.
- `model_evaluate()` calculates evaluation metrics.
- `log_metrics()` logs metrics in MlFlow.
- `plot_metrics()` generates performance plots.


### src/predict.py
File for obtaining predictions by models. Called separately from the `main.py` file.


### src/settings.py
Stores important parameters, such as:

* Directories for saving models and predictions.
* MLFlow tracking URI.
* Feature list.
* Target class.
* Model list.


### test_notebook.ipynb
Jupyter notebook for exploratory data analysis, data cleaning, feature selection, hypothesis testing, building prototype models, and selecting suitable architectures.
