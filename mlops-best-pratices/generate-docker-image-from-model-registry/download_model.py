"""
An example script making MLflow API calls as an external client to a Domino deployment.
"""
import sys
import os
import mlflow
import mlflow.sklearn
import requests
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

def main():
    set_mlflow_tracking_uri()
    register_domino_api_key_request_header_provider()
    register_domino_execution_request_header_provider()

    start_run_and_register_model()


if __name__ == "__main__":

    ## Use Service Account Key

    ## Create a API Key
    #API_KEY = ''
    #RUN_ID= '66aa4d79a6faa6673b4b6da6'

    ## Download artifacts
    #os.environ['MLFLOW_TRACKING_URI'] = 'https://secureds53799.cs.domino.tech/'
    #os.environ['DOMINO_USER_API_KEY'] = API_KEY
    #os.environ['DOMINO_RUN_ID'] = RUN_ID

    token = os.environ['SVC_TOKEN']
    url = os.environ['MLFLOW_TRACKING_URI']
    generate_token_endpoint = f'{url}/account/66aa68b7a6faa6673b4b6db8/regenerateApiKey'
    out = requests.post(generate_token_endpoint,headers={'Authorization':f"Bearer {token}"})
    api_key = out.json()['raw']
    os.environ['DOMINO_USER_API_KEY'] = api_key

    set_mlflow_tracking_uri()
    register_domino_api_key_request_header_provider()
    register_domino_execution_request_header_provider()
    artifact_path = "sklearn-model"
    model_name = "RFRegression"


    client = MlflowClient()

    model_name = sys.argv[1]
    model_version = sys.argv[2]
    print(f'Download Model Name {model_name} and Model Version {model_version}')
    mv = client.get_model_version(model_name, model_version)

    dst_path = "./artifacts"

    downloaded_artifacts_path = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=dst_path)

    print(f'Downloaded artifacts for this model version to {downloaded_artifacts_path}')
    #print(mv.run_id)
    print(mlflow.pyfunc.get_model_dependencies(mv.source))

    #!ls - la $downloaded_artifacts_path
    ## Run Artifacts in conda env

    '''
    import mlflow
    logged_model = f'runs:/{mv.run_id}/sklearn-model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    import pandas as pd
    import numpy as np

    df = pd.DataFrame([['5', '4.2']])
    loaded_model.predict(df)
    '''