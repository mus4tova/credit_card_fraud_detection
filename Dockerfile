FROM python:3.10

WORKDIR /fraud-detection

RUN pip install poetry==1.5.0

COPY .env ./pyproject.toml ./poetry.lock ./

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . .

ENV PYTHONPATH /fraud-detection