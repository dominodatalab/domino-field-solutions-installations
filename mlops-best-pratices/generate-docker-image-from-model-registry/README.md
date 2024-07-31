## Create a Dockerfile based on Model Registry

## How is the model registered

Imagine running the following inside a Domino workspace or a job

```python
# License for code in this notebook: BSD 3 clause
# import everything we'll need for the rest of the notebook
import mlflow.sklearn
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor

client = MlflowClient()
name = "RandomForestRegression_2" 
registered_model = client.create_registered_model(name)

# create an experiment run in MLflow
mlflow.set_experiment('export-model')
params = {"n_estimators": 3, "random_state": 42}
rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
# Log MLflow entities
with mlflow.start_run() as run:
    mlflow.log_params(params)
    model_info = mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")
    runs_uri = model_info.model_uri
    
    # Create a new model version of the RandomForestRegression model from this run
    desc = "A testing version of the model"
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
    mv = client.create_model_version(name, model_src, run.info.run_id, description=desc)
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))
    print("Description: {}".format(mv.description))
    print("Status: {}".format(mv.status))
    print("Stage: {}".format(mv.current_stage))
''' Assume this is the output
Name: RandomForestRegression_2
Version: 1
Description: A testing version of the model
Status: READY
Stage: None
'''

```
The model is now registered. Note the model name and model version. You will need it later


## Preparation

1. Create a Domino Service Account, generate a token and make it a practitioner
2. Ask your Domino Admin what the `id` of this user is
3. Create the following environment variables
    - MLFLOW_TRACKING_URI = `https://mydomino.cs.domino.tech/`
    - SVC_TOKEN = `TOKEN_FROM_STEP_1`
    - DOMINO_RUN_ID = `<YOU NEED A VALID RUN ID FOR THE SERVICE ACCOUNT`

### How do I generate a valid run_id for the service account

For that run a job as that service account using the following snippet

```python
domino_host = 'https://mydomino.cs.domino.tech/'
project_owner = 'john_doe'
project_name = 'quickstart'

from domino import Domino

domino_client = Domino(host=domino_host,project=f"{project_owner}/{project_name}", auth_token=svc_token)
result = domino_client.runs_start(command="ls",title="Dummy Job")
result
```
You can do this everytime or once for a project and store the run_id

## Download Artifacts

```shell
cd ./generate-docker-image-from-model-registry 

export  MLFLOW_TRACKING_URI = `https://mydomino.cs.domino.tech/`
export  DOMINO_RUN_ID = `...`
export SVC_TOKEN = `...`
export MODEL_NAME=RandomForestRegression_2
export MODEL_VERSION=1

python download_model.py $MODEL_NAME $MODEL_VERSION
```

This will store the model artifacts in the folder `artifacts`. Note the sub-folder `sklearn-model`.
Each model will have their own based on how it is registered. The above is only intended as an example

Now build the image as you like it

```shell
docker build . -t mydominomodel
```
Run the image
```shell
docker run mydominomodel:latest 
```


