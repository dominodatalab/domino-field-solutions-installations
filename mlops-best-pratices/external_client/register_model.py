"""
An example script making MLflow API calls as an external client to a Domino deployment.
"""

import os

import mlflow.sklearn
from domino import Domino
from mlflow import MlflowClient
from mlflow.tracking.request_header.registry import _request_header_provider_registry
from sklearn.ensemble import RandomForestRegressor

from domino_api_key_header_provider import DominoApiKeyRequestHeaderProvider
from domino_execution_header_provider import DominoExecutionRequestHeaderProvider


def set_mlflow_tracking_uri():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        print("MLFLOW_TRACKING_URI environment variable must be set.")
        exit(1)
    mlflow.set_tracking_uri(mlflow_tracking_uri)


def register_domino_api_key_request_header_provider():
    if not os.getenv("DOMINO_USER_API_KEY"):
        print("DOMINO_USER_API_KEY environment variable must be set.")
        exit(1)
    # Set X-Domino-Api-Key request header
    _request_header_provider_registry.register(DominoApiKeyRequestHeaderProvider)


def register_domino_execution_request_header_provider():
    if not os.getenv("DOMINO_RUN_ID"):
        print("DOMINO_RUN_ID environment variable must be set.")
        exit(1)
    # Set X-Domino-Execution request header
    _request_header_provider_registry.register(DominoExecutionRequestHeaderProvider)


def start_run_and_register_model():
    model_name = "RandomForestRegressionModel"
    artifact_path = "sklearn-model"

    run = start_run(artifact_path)
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    print(client.get_registered_model(model_name))


def start_run(artifact_path):
    with mlflow.start_run() as run:
        params = {"n_estimators": 3, "random_state": 42}
        model = RandomForestRegressor(**params).fit([[0, 1]], [1])
        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, artifact_path)
    return run

def start_stub_job(domino_host,project_owner,project_name,api_key,job_name):
    domino_2 = Domino(host=domino_host,project=f"{project_owner}/{project_name}", api_key=api_key)
    result = domino_2.runs_start(["test.py"], title="Api key based execution",command=f"ls")
    result

def main():
    set_mlflow_tracking_uri()
    register_domino_api_key_request_header_provider()
    register_domino_execution_request_header_provider()

    start_run_and_register_model()


if __name__ == "__main__":
    #Ex. DOMINO_EXTERNAL_HOST = mydomino.cs.domino.tech
    domino_host= os.environ['DOMINO_EXTERNAL_HOST']

    domino_tracking_uri = f'https://{domino_host}'
    job_name='external-job-1'
    project_owner='integration-test'
    project_name='quick-start'
    api_key= os.environ['DOMINO_USER_API_KEY']

    start_stub_job(project_owner,project_name,api_key,job_name)
    main()
