"""
An example script making MLflow API calls as an external client to a Domino deployment.
"""

import os

import mlflow.sklearn
from domino import Domino
from mlflow import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
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





def start_run_and_register_model(artifact_path,model_name):

    client = MlflowClient()
    mlflow.set_experiment('EXTERNAL_EXPERIMENT')
    params = {"n_estimators": 3, "random_state": 42}
    rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
    try:
        client.create_registered_model(model_name)
    except:
        print('Model Already exists.')
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        model_info = mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")
        runs_uri = model_info.model_uri
        model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
        # Create a new model version of the RandomForestRegression model from this run
        desc = "A testing version of the model"
        model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
        print(f'Model Src {model_src}')
        mv = client.create_model_version(model_name,model_uri)
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Description: {}".format(mv.description))
        print("Status: {}".format(mv.status))
        print("Stage: {}".format(mv.current_stage))
    return run


def main():
    set_mlflow_tracking_uri()
    register_domino_api_key_request_header_provider()
    register_domino_execution_request_header_provider()
    artifact_path = "sklearn-model"
    model_name = "RFRegression"

    client = MlflowClient()
    start_run_and_register_model(artifact_path, model_name)
    print(client.get_registered_model(artifact_path))

def start_stub_job(domino_host,project_owner,project_name,api_key,job_name):
    domino_2 = Domino(host=domino_host,project=f"{project_owner}/{project_name}", api_key=api_key)
    result = domino_2.runs_start([f"echo `{job_name}`"], title="Api key based execution")
    return result


if __name__ == "__main__":
    #Ex. DOMINO_EXTERNAL_HOST = mydomino.cs.domino.tech
    domino_host= os.environ['DOMINO_EXTERNAL_HOST']

    domino_tracking_uri = f'https://{domino_host}'
    job_name = 'external-job-1'
    project_owner = os.environ['DOMINO_PROJECT_OWNER']
    project_name = 'quick-start'
    api_key = os.environ['DOMINO_USER_API_KEY']
    result = start_stub_job(domino_tracking_uri,project_owner,project_name,api_key,job_name)
    print(result)
    run_id=result['runId']

    #You can re-use and existing run-id for the project/user combination
    #run_id = '6595f16d12b6f32d73f22bd6'
    os.environ['DOMINO_USER_API_KEY'] = api_key
    os.environ['MLFLOW_TRACKING_URI'] = domino_tracking_uri
    os.environ['DOMINO_RUN_ID'] = run_id
    main()



