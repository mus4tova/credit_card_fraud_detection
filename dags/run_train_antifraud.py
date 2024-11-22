import pendulum
from docker.types import Mount
from airflow.decorators import dag
from airflow.providers.docker.operators.docker import DockerOperator


@dag(
    dag_id="run_train_antifraud",
    start_date=pendulum.now("UTC").subtract(minutes=10),
    schedule="00 10 * * *",
    catchup=False,
    default_args={
        "owner": "dsp",
        "retries": 0,
    },
    description="Anrifraud models predict and train",
)
def TrainPredictors():
    train_antifraud = DockerOperator(
        task_id="train_antifraud",
        image="fraud-detection_train:latest",
        command="python src/main.py --train",
        api_version="auto",
        network_mode="host",
        docker_url="http://docker-socket-proxy:2375",
        mounts=[
            Mount(
                source="/opt/rec-mlflow-updated",
                target="/opt/fraud-detection",
                type="bind",
            ),
        ],
    )

    predict_antifraud = DockerOperator(
        task_id="predict_antifraud",
        image="fraud-detection_predict:latest",
        command="python src/main.py --predict",
        api_version="auto",
        network_mode="host",
        docker_url="http://docker-socket-proxy:2375",
        mounts=[
            Mount(
                source="/opt/rec-mlflow-updated",
                target="/opt/fraud-detection",
                type="bind",
            ),
        ],
    )
    train_antifraud >> predict_antifraud


dag = TrainPredictors()
