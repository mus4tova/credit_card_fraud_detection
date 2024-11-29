FROM python:3.10

WORKDIR /credit_card_fraud_detection

RUN pip install poetry==1.5.0

COPY .env ./pyproject.toml ./poetry.lock ./

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev
RUN mlflow server --host 127.0.0.1 --port 8080

COPY . .

ENV PYTHONPATH /credit_card_fraud_detection
