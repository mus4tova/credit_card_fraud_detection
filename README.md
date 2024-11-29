# Credit card fraud detection

Based on the transaction data it determines the presence of fraud.

## Models traning

### Launching the predictor for training

```bash
podman-compose -f docker-compose.yml build
podman-compose -f docker-compose.yml up train
```

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

Большинство фичей имеют строковый тип данных, 8 последних - численный.


### Models saving

Model files and the file with thresholds are saved in the folder `models`. 


## Predictions

### Launch predictor for predictions

```bash
podman-compose -f docker-compose.yml build
podman-compose -f docker-compose.yml up predict
```
