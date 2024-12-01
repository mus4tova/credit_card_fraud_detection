FROM python:3.10

WORKDIR /credit_card_fraud_detection

RUN pip install poetry==1.5.0

COPY ./pyproject.toml ./poetry.lock ./

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . .

ENV PYTHONPATH /credit_card_fraud_detection
