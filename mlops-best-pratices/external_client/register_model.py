"""
An example script making MLflow API calls as an external client to a Domino deployment.
"""

import os
import domino_utils
import mlflow.sklearn
import requests
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
    _request_header_provider_registry.register(DominoApiKeyRequestHeaderProvider)


def register_domino_execution_request_header_provider():

    if not os.getenv("DOMINO_RUN_ID"):
        print("DOMINO_RUN_ID environment variable must be set.")
        exit(1)
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
    #This is needed if you run this as a long running process like a REST API endpoint
    #where multiple calls can be made to the main() function
    _request_header_provider_registry._registry.clear()
    register_domino_api_key_request_header_provider()
    register_domino_execution_request_header_provider()

    start_run_and_register_model()


if __name__ == "__main__":
    #This will be the mlflow tracking uri
    domino_host= os.environ['DOMINO_EXTERNAL_HOST']
    mlflow_tracking_uri = f'https://{domino_host}'
    '''
    Starting a stub job is an alternative but leads to several ghost jobs
    Instead run job in the desired project. Get its run id 
    '''
    ''''
    
    job_name='external-job-1'
    project_owner='integration-test'
    project_name='quick-start'
    api_key= os.environ['DOMINO_USER_API_KEY']
    start_stub_job(project_owner,project_name,api_key,job_name)
    '''

    '''
    Three environment variables must be set
    1. MLFLOW_TRACKING_URI
    2. DOMINO_USER_API_KEY
    3. DOMINO_RUN_ID - This is the run_id from the project where the MLFLOW Run is
       being created. It should also belong to the user who is represented by 
       DOMINO_USER_API_KEY
    '''

    '''
    An alternative method to bringing a long lived DOMINO_USER_API_KEY (You do not
    have one for a service account) is to use the OAuth JWT Token to invoke
    the Domino `regenerateApiKey`. This call is made as follows:
    
    1. Construct a url
            id_of_the_user='6283a3966d4fd0362f8ba2a8'
            url = https://{domino_host}/account/{idoftheuser}/regenerateApiKey
    2. Use the service account token as a Authorization Bearer Token to make this call
    3. It should return a result (if status_code=200) which looks something like this
    {"raw":"67534xxxxxx1fa5fcccf055axxxxaa297975xxxxx76a841128d6ad3f1b2028be"}
    4. The value in the "raw" attribute can be used as the DOMINO_API_KEY
    5. Discard it each time you use it. Do not store it anywhere unless you plan to use
       it like a long lived api key
       
       
    Lastly the id_of_the_user can be obtained using the call to the endpoint 
    https://{domino_host}/v4/users/self using the service account token as the 
    Authorization Bearer Token
    '''

    #This was to illustrate that the MLFLOW_TRACKING_URI from outside DOMINO
    #is the same as the Domino URL being used to access Domino
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    oauth_token="eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJzdjJuOGdVTy11eEQ3V3cxOURPSGZxZW82TnowS1dubVMtdHFLakxpZzI0In0.eyJleHAiOjE3Mjg1NjQ0NDAsImlhdCI6MTcyODU2NDE0MCwiYXV0aF90aW1lIjoxNzI4MzA5MTYwLCJqdGkiOiI0YTdlMjI0OS03OTBjLTRjMDAtOGJmNi00ZDU5OGUzYTdhY2MiLCJpc3MiOiJodHRwczovL3NlY3VyZWRzNTM3OTkuY3MuZG9taW5vLnRlY2gvYXV0aC9yZWFsbXMvRG9taW5vUmVhbG0iLCJhdWQiOlsiZmx5dGVhZG1pbiIsInJlYWxtLW1hbmFnZW1lbnQiLCJ0b29sa2l0LWNsaWVudCIsImdyYWZhbmEtY2xpZW50IiwiYnJva2VyIiwiYWNjb3VudCJdLCJzdWIiOiI1ODVlMWIwYy0xNjdiLTQxMTUtOTljNi1jNTcyZmE0YmRiYzgiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJkb21pbm8tcGxheSIsInNlc3Npb25fc3RhdGUiOiIzNTI1NmNkYS05MGEzLTQyOTItYWM4My00OTQ2ZmJlZTBjZWIiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtZG9taW5vcmVhbG0iXX0sInJlc291cmNlX2FjY2VzcyI6eyJyZWFsbS1tYW5hZ2VtZW50Ijp7InJvbGVzIjpbIm1hbmFnZS11c2VycyJdfSwidG9vbGtpdC1jbGllbnQiOnsicm9sZXMiOlsidG9vbGtpdF9hZG1pbiJdfSwiZ3JhZmFuYS1jbGllbnQiOnsicm9sZXMiOlsiZ3JhZmFuYV9hZG1pbiJdfSwiYnJva2VyIjp7InJvbGVzIjpbInJlYWQtdG9rZW4iXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIGVtYWlsIG9mZmxpbmVfYWNjZXNzIHByb2ZpbGUiLCJzaWQiOiIzNTI1NmNkYS05MGEzLTQyOTItYWM4My00OTQ2ZmJlZTBjZWIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImF3c3JvbGVzZXNzaW9uIjoiaW50ZWdyYXRpb24tdGVzdEBkb21pbm9kYXRhbGFiLmNvbSIsInJvbGVzIjpbIlN5c0FkbWluIiwiUHJhY3RpdGlvbmVyIl0sImlkcGJyb2tlciI6IktleWNsb2FrLUlEUC1zZWN1cmVkczUzNzk5LVhOWE0iLCJuYW1lIjoiSW50ZWdyYXRpb24gVGVzdCIsInByZWZlcnJlZF91c2VybmFtZSI6ImludGVncmF0aW9uLXRlc3QiLCJnaXZlbl9uYW1lIjoiSW50ZWdyYXRpb24iLCJmYW1pbHlfbmFtZSI6IlRlc3QiLCJlbWFpbCI6ImludGVncmF0aW9uLXRlc3RAZG9taW5vZGF0YWxhYi5jb20iLCJhd3Nyb2xlcyI6WyJhcm46YXdzOmlhbTo6OTQ2NDI5OTQ0NzY1OnJvbGUvc2VjdXJlZHM1Mzc5OS1TYW1sUmVhbG0tc2FtbC1yb2xlQGEuYixhcm46YXdzOmlhbTo6OTQ2NDI5OTQ0NzY1OnNhbWwtcHJvdmlkZXIvc2VjdXJlZHM1Mzc5OS1TYW1sUmVhbG0tc2FtbC1wcm92aWRlciJdLCJ1c2VyX2dyb3VwcyI6WyIvcm9sZXMvUHJhY3RpdGlvbmVyIiwiL3JvbGVzL1N5c0FkbWluIl19.aHwrgxBHQx7BGc9x2F8TGESkUm_wAmEgqDWv-kUpR0F5sw8FuB9qk-FdXmkpenkQ2f3vYG-cvOUjvoif5tejIrgOaweqfhcNE2NxvyxlZztw2FbCkFqoX7HXWu5PFX_fY7Z768bgX7svo9bFJlbb-v2lv27pnzcbuGYwaDpZJleLfdJI_UT-pJnVgQIqdzpljbP5rNvfE-yNeHf4wJGm9Tqo07LCoXsyZixLn9waVPnF2mYRNnIN4-C9-R_fCDWubBxHxMpHObkZO_f-iI9TXnAcG71AJQ9lOZFl4rcZDZ_inVPQIOSkb64f8d46LsEuq_TILS55iizrNtUANtawRw"

    #For any user you can get a token valid for 5 mins making the following call in a workspace
    #curl http://localhost:8899/access-token

    '''
    api_key = domino_utils.get_domino_user_api_key(domino_host,oauth_token)
    os.environ['DOMINO_USER_API_KEY'] = api_key
    '''
    main()


