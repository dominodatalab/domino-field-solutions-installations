## External Client

### Example Script

This section describes how to run an example script that makes MLflow API calls as an external client to a Domino deployment.

#### Pre-requisites
The following environment variables must be defined:
- `MLFLOW_TRACKING_URI` - URI of the Domino deployment
- `DOMINO_USER_API_KEY` - Domino user API key to identify the user making MLflow API calls
- `DOMINO_RUN_ID` - Domino execution id to identify the project for MLflow API calls

To retrieve the `DOMINO_USER_API_KEY` and `DOMINO_RUN_ID`:
- Launch a workspace
- Open a workspace terminal (via `New > Terminal`)
- `env | grep DOMINO_USER_API_KEY`
- `env | grep DOMINO_RUN_ID`

#### Running the script
```shell
export MLFLOW_TRACKING_URI=https://pocheung13386.dmo-team-sandbox.domino.tech
export DOMINO_USER_API_KEY=<Domino user API key>
export DOMINO_RUN_ID=<Domino run id>

python register_model.py
```

#### Sample output (reformatted)
```
Successfully registered model 'RandomForestRegressionModel'.
2023/06/21 13:20:02 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: RandomForestRegressionModel, version 1
Created version '1' of model 'RandomForestRegressionModel'.
<RegisteredModel:
  creation_timestamp=1687378801932, description='',
  last_updated_timestamp=1687378802143,
  latest_versions=[
    <ModelVersion:
      creation_timestamp=1687378802143,
      current_stage='None',
      description='',
      last_updated_timestamp=1687378802143,
      name='RandomForestRegressionModel',
      run_id='cb7510c5b2c84d09881a533195c99683',
      run_link='',
      source='mlflow-artifacts:/mlflow/cb7510c5b2c84d09881a533195c99683/artifacts/sklearn-model',
      status='READY',
      status_message='',
      tags={
        'domino.project': 'quick-start',
        'mlflow.domino.dataset_info': '64635e00bf61321f46003634-64635e00bf61321f46003633,64635e1ebf61321f46003638-64635e1ebf61321f46003637',
        'mlflow.domino.environment_id': '64633d1c70175a297c36b20a',
        'mlflow.domino.environment_revision_id': '64633d1c70175a297c36b20c',
        'mlflow.domino.hardware_tier': 'small-k8s',
        'mlflow.domino.project': 'quick-start',
        'mlflow.domino.project_id': '64635dfabf61321f4600362f',
        'mlflow.domino.project_name': 'quick-start',
        'mlflow.domino.run_id': '64935a02cca641326ecb2ca6',
        'mlflow.domino.run_number': '3',
        'mlflow.domino.user': 'integration-test',
        'mlflow.domino.user_id': '64635df8bf61321f4600362d',
        'mlflow.source.type': 'NOTEBOOK',
        'mlflow.user': 'integration-test'
      },
      user_id='',
      version='1'
    >
  ],
  name='RandomForestRegressionModel',
  tags={
    'domino.project': 'quick-start',
    'mlflow.domino.dataset_info': '64635e00bf61321f46003634-64635e00bf61321f46003633,64635e1ebf61321f46003638-64635e1ebf61321f46003637',
    'mlflow.domino.environment_id': '64633d1c70175a297c36b20a',
    'mlflow.domino.environment_revision_id': '64633d1c70175a297c36b20c',
    'mlflow.domino.hardware_tier': 'small-k8s',
    'mlflow.domino.project': 'quick-start',
    'mlflow.domino.project_id': '64635dfabf61321f4600362f',
    'mlflow.domino.project_name': 'quick-start',
    'mlflow.domino.run_id': '64935a02cca641326ecb2ca6',
    'mlflow.domino.run_number': '3',
    'mlflow.domino.user': 'integration-test',
    'mlflow.domino.user_id': '64635df8bf61321f4600362d',
    'mlflow.source.type': 'NOTEBOOK',
    'mlflow.user': 'integration-test'
  }
>
```

### Curl

This section describes how to make MLflow API calls using `curl` as an external client to a Domino deployment.

#### Pre-requisites
Define the following environment variables to simplify curl commands:
- `MLFLOW_TRACKING_URI` - URI of the Domino deployment
- `DOMINO_USER_API_KEY` - Domino user API key to identify the user making MLflow API calls
- `DOMINO_RUN_ID` - Domino execution id to identify the project for MLflow API calls
- `DOMINO_EXECUTION_ID_JWT` - JWT containing a Domino execution id to identify the project for MLflow API calls

Certain MLflow API calls require the `X-Domino-Execution` request header whose value is a JWT containing a Domino execution id.  Use the `jwt_encode_execution_id.py` helper script to generate the JWT, for example:
```shell
export DOMINO_EXECUTION_ID_JWT=$(python jwt_encode_execution_id.py ${DOMINO_RUN_ID})
```

#### Curl commands

Searching for experiments
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search -d '{"max_results": 10}' | jq
```

Getting an experiment
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/get?experiment_id=1 | jq
```

Searching for registered models
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/search | jq
```

Searching for registered models with filter: name='my_model'
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/search?filter=name%3D%27my_model%27 | jq
```

Creating a registered model
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" -H "X-Domino-Execution: ${DOMINO_EXECUTION_ID_JWT}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/create -d '{"name": "my_model", "tags": []}' | jq
```

Getting a registered model
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get?name=my_model | jq
```

Deleting a registered model
```shell
curl -s -H "Content-Type: application/json" -H "X-Domino-Api-Key: ${DOMINO_USER_API_KEY}" -H "X-Domino-Execution: ${DOMINO_EXECUTION_ID_JWT}" ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/delete -d '{"name": "my_model"}' -X DELETE | jq
```
