# Domino Extensions API (Field Extensions)

This library enables adding new API endpoints to support customer requirements 

## Installation

From the root folder of this project run the following commands:

1. First publish the image
```shell
tag="${tag:-latest}"
operator_image="${operator_image:-quay.io/domino/domino-extensions-api}"
docker build -f ./Dockerfile -t ${operator_image}:${tag} .
docker push ${operator_image}:${tag}
```

or run

```shell
./create_and_push_docker_image.sh ${tag}
```
2. ## Use Helm to Install
Check if you have domino-field namespace in the current cluster 
```shell
kubectl get ns
``` 
if domino-field namspace is not present create using below command
```shell
kubectl create namespace domino-field
kubectl label namespace domino-field  domino-compute=true
```
Install using Helm

```shell
export field_namespace=domino-field
helm install -f helm/domino-extensions-api/values.yaml domino-extensions-api helm/domino-extensions-api -n ${field_namespace}
```
3. To upgrade use helm
```shell
export field_namespace=domino-field
helm upgrade -f helm/domino-extensions-api/values.yaml domino-extensions-api helm/domino-extensions-api -n ${field_namespace}
```

4. To delete use helm 

```shell
export field_namespace=domino-field
helm delete  domino-extensions-api -n ${field_namespace}
```

## Using the API

This API Service supports endpoints which are broadly classified into two major categories:

1. **Extending the existing API** - If the objects returned by the existing endpoints are 
   missing certain attributes, use this section of the service to invoke the existing API endpoints and
   decorate the returned objects with additional details.
2. **Central Management of Workspace Autoshutdown Rules** - These are endpoints provided to enable administrator actions
   not currently supported via endpoints.
3. **Domsed WebClient** - Endpoints to manage (list/create/update/delete) Domsed mutations.
   

### Extending the existing API

#### Refresh Cache - `/api-extended/refresh_cache`

Invoke this endpoint if you want to refresh all caches. To avoid having to read Mongo repeatedly,
the EnvironmentRevision and Project information from the Mongo collections are cached.
If you want to refresh the cache invoke this method.

#### Enhanced Projects -  `/api-extended/projects/beta/projects`

This is an extension of the endpoint `/api/projects/beta/projects`

The returned json contains an attribute `projects` which is a list of the projects the user is member of.

Each project json in the list contains two additional attributes as compared to the orignal from the `projects` 
Mongo collection

- `environment_id`
- `default_environment_revision_spec`

#### Enhanced Environments - `/api-extended/environments/beta/environments`

This is an extension of the endpoint ` /api/environments/beta/environments`

The returned json contains an attribute `environments` which is a list of the environments the user has access to

Each environment instance in the list has two attributes , `latestRevision` and `selectedRevision`. Each of them
is enhanced by adding attributes

- `basedOnDockerImage` (This is the root docker image based on the env hierarchy that the environment revsion is based on)
- `basedOnDockerImageStatusMessage` (This will contain an error message or `Success` )

For brevity the attribute `availableTools` is replaced with `None` 

### Central Management of Workspace Autoshutdown Rules

Currently there are two levers to manage the workspace auto-shutdown intervals: 
   - Central config parameter - `com.cerebro.domino.workspaceAutoShutdown.globalMaximumLifetimeInSeconds` which defines both,
      the default value and the maximum value for the workspace auto-shutdown interval.
   - User can choose a lower value from the `User Settings` page
   
   These endpoints support a Domino Administrator to centrally manage an individual users workspace auto-shutdown interval.   

#### Workspace auto-shutdown interval management `/workspaceautoshutdown/interval`

The full endpoint inside the Domino workspace is (assuming `domino-platform` as the platform namespace)
```shell
http://domino-extensions-api-svc.domino-platform/v4-extended/autoshutdownwksrules
```

Type : POST

Headers and Body:
```
--header 'X-Domino-Api-Key: ADD YOUR API KEY HERE ' \
--header 'Content-Type: application/json' \
--data-raw '{
    "users": {
        "wadkars": 3600,
        "integration-test":  21600
    },
    "override_to_default" : false
}'
```
Or if using the bearer token obtained from `http://localhost:8899/access-token`

```shell
--header 'Authorization: Bearer <ADD YOUR TOKEN HERE>' \
--header 'Content-Type: application/json' \
--data-raw '{
    "users": {
        "wadkars": 3600,
        "integration-test":  21600
    },
    "override_to_default" : false
}'
```

For each user you want to override the default value update the `users`
attribute above as:
`{domino-user-name}` : {auto_shutdown_duration_in_seconds}

The default auto-shutdown-duration is obtained from the central config parameter:
```shell
com.cerebro.domino.workspaceAutoShutdown.globalDefaultLifetimeInSeconds
```
The `override_to_default` attribute is used to determine if all users (not specificied)
in the `users` attribute tag as also update to have their default autoshutdown duration
set to default.

if `override_to_default` is set to `true` every user except the users mentioned in the 
`users` attribute will be configured for the default value of autoshutdown

The value of `com.cerebro.domino.workspaceAutoShutdown.globalDefaultLifetimeInSeconds`
is expected to be lower than `com.cerebro.domino.workspaceAutoShutdown.globalMaximumLifetimeInSeconds`

Likewise for the values provided for each user in the `users` attribute

If not, the auto shutdown duration is capped at the value of `com.cerebro.domino.workspaceAutoShutdown.globalMaximumLifetimeInSeconds`


### Domsed Webclient

These endpoints allow managing Domsed mutations from a Domino Admin workspace

#### List Mutations `/mutation/list`(GET)

List all mutations. 

Invoke using python client code `client.domsed_webclient.list`

#### Get Mutation `/mutation/<name>`(GET)

Get the definition of a mutation with name = `<name>`

Invoke using python client code `client.domsed_webclient.get(name)`



#### Delete Mutation `/mutation/<name>`(DELETE)

Delete mutation with name = `<name>`

Invoke using python client code `client.domsed_webclient.delete(name)`

#### Apply Mutation `/mutation/apply`(POST)

Apply mutation. It takes the mutation yaml file in JSON format

Invoke using python client code `client.domsed_webclient.apply_file(yaml_file_name)`

Invoke using python client code `client.domsed_webclient.apply(mutation_json)`


### User Management

These endpoints allow a Domino Administrator to do the following:
1. Find which accounts have been inactive for `x` number of days. An inactive account is an account that has not started a workload for x number of days
2. Deactivate a list of users
3. ReActivate a list of users

#### Find Inactive Users for `x` days `/user-management/inactivefor/<days>` (GET)
Returns a list of inactive accounts. Expect a json which looks like this
```json
{
    "inactive_accounts": [
        {
            "is_svc_account": true,
            "user_name": "svc-user-ds-2"
        },
        {
            "is_svc_account": true,
            "user_name": "svc-user-ds-3"
        },
        {
            "is_svc_account": true,
            "user_name": "svc-test-app-1"
        },
        {
            "is_svc_account": true,
            "user_name": "svc-prod-app-1"
        },
        {
            "is_svc_account": true,
            "user_name": "svc-admin"
        },
        {
            "is_svc_account": false,
            "user_name": "user-ds-1"
        },
        {
            "is_svc_account": false,
            "user_name": "user-ds-2"
        }
    ]
}
```

Client code
```python
import os
import json
import requests
def get_token():
    # Fetch the mounted service account token
    uri = os.environ['DOMINO_API_PROXY']
    return requests.get(f"{uri}/access-token").text

service_name="domino-extensions-api-svc"
dns_name = f"{service_name}.domino-field.svc.cluster.local"

endpoint = "user-management/inactivefor/5"
url = f"http://{dns_name}/{endpoint}"

token = get_token()
my_headers = {"Authorization": f"Bearer {token}"}
domino_api_host = os.environ['DOMINO_API_HOST']
r = requests.get(url,headers=my_headers)
# Pretty-print the JSON object
pretty_json = json.dumps(r.json(), indent=4)
print(pretty_json)
```


#### Deactivate users `/user-management/deactivate` (POST)

Example payload
```json
{"users":["user-ds-1","user-ds-2"]}
```
Client code
```python
import os
import json
import requests
def get_token():
    # Fetch the mounted service account token
    uri = os.environ['DOMINO_API_PROXY']
    return requests.get(f"{uri}/access-token").text

service_name="domino-extensions-api-svc"
dns_name = f"{service_name}.domino-field.svc.cluster.local"
endpoint = "user-management/deactivate"
url = f"http://{dns_name}/{endpoint}"
payload = {"users":['user-ds-1','user-ds-2']}
token = get_token()
my_headers = {"Authorization": f"Bearer {token}",'Content-Type': 'application/json'}
domino_api_host = os.environ['DOMINO_API_HOST']

r = requests.post(url,headers=my_headers,json=payload)
print(r)
r.text
```
Output should look like this when successful
```json
{"updated_users":["user-ds-1","user-ds-2"]}
```


#### Activate users `/user-management/activate` (POST)

Example payload
```json
{"users":["user-ds-1","user-ds-2"]}
```
Client code
```python
import os
import json
import requests
def get_token():
    # Fetch the mounted service account token
    uri = os.environ['DOMINO_API_PROXY']
    return requests.get(f"{uri}/access-token").text

service_name="domino-extensions-api-svc"
dns_name = f"{service_name}.domino-field.svc.cluster.local"
endpoint = "user-management/activate"
url = f"http://{dns_name}/{endpoint}"
payload = {"users":['user-ds-1','user-ds-2']}
token = get_token()
my_headers = {"Authorization": f"Bearer {token}",'Content-Type': 'application/json'}
domino_api_host = os.environ['DOMINO_API_HOST']

r = requests.post(url,headers=my_headers,json=payload)
print(r)
r.text
```
Output should look like this when successful
```json
{"updated_users":["user-ds-1","user-ds-2"]}
```
> When a user is deactivated their roles are removed. When reactivated, there is no memory of what
> roles were held by the user prior to being deactivated. A newly reactivated
> user is a `Lite` user or `ResultsConsumer`


## Motivating Use-cases and Client Code

The following lists the use-cases which motivated the endpoints in this service

### Prepare Environment for Archival

A customer wants to retire Environments periodically (say every 3 months). However, these environments are 
actively used in a large number of projects. You cannot archive an environment (or its derivatives) without removing the environment from a 
projects default setting. The `/api-extended/environments/beta/environments` returns
all the environments along with their base environments. Iterating over this list will help you determine
all the environments that are derived from the `to-be-archived` environment.
   
The `/api-extended/projects/beta/projects` supports identifying all the projects which use the to-be-archived environment and its
derivatives (obtained from the previous call). You can use this information to update the default environment of these projects

```python
import requests
import json
import os

api_host = os.environ.get("EXTENDED_API_HOST", "extendedapi-svc.domino-platform")
api_port = os.environ.get("EXTENDED_API_PORT", "80")
auth_token = requests.get(os.environ.get('DOMINO_API_PROXY') + '/access-token').text
api_key =  os.environ.get("DOMINO_USER_API_KEY")

url = f"http://{api_host}:{api_port}/v4-extended/autoshutdownwksrules"
payload = json.dumps(
    {
        "users": {"wadkars": 3600, "integration-test": 21600},
        "override_to_default": False,
    }
)
headers = {
    "X-Domino-Api-Key": api_key,
    "Content-Type": "application/json",
}
# Or the newer, more auth standards compliant version 
headers = {
    "Authorization": f"Bearer {auth_token}",
    "Content-Type": "application/json",
}
      
## Update Project Settings
project_id = "ADD HERE"
url = f"http://nucleus-frontend.domino-platform.svc.cluster.local:80/v4/projects/{project_id}/settings"

payload = json.dumps({"defaultEnvironmentId": "ADD THE ENV ID TO REPLACE WITH"})

response = requests.request("PUT", url, headers=headers, data=payload)
print(response.text)
 ```

### Manage a user's workspace auto-shutdown interval centrally via Admin endpoints

The default value for `com.cerebro.domino.workspaceAutoShutdown.globalMaximumLifetimeInSeconds`  is 72 hours. Customers rarely 
change this. Users can proactively choose a lower value by updating their user-settings but rarely have a good motivation
to do this. Consequently workspaces keep running forever, even if idle. This causes a significant increase cloud costs. This functionality is
provided to allow administrators to manage the auto-shutdown interval to manageable level.

In the central config, configure an additional parameter

`com.cerebro.domino.workspaceAutoShutdown.globalDefaultLifetimeInSeconds` (Set default to a typical working day say 10 hours)

The value for this parameter should be lower than the value for the central config parameter

`com.cerebro.domino.workspaceAutoShutdown.globalMaximumLifetimeInSeconds` (Default 72 hours)


As an Domino Administrator user, run the following
```python
import requests
import json
import os

api_host = "domino-extensions-api-svc.domino-platform"
api_port = "80"
auth_token = requests.get(os.environ.get('DOMINO_API_PROXY') + '/access-token').text
api_key =  os.environ.get("DOMINO_USER_API_KEY")

url = f"http://{api_host}:{api_port}/autoshutdown/interval"
payload = json.dumps(
    {
        "users": {"wadkars": 3600, "integration-test": 21600},
        "override_to_default": False,
    }
)
headers = {
    "X-Domino-Api-Key": api_key,
    "Content-Type": "application/json",
}
# Or the newer, more auth standards compliant version 
headers = {
    "Authorization": f"Bearer {auth_token}",
    "Content-Type": "application/json",
}

payload = json.dumps(
    {
        "users": {"wadkars": 3600, "integration-test": 21600},
        "override_to_default": False,
    }
)

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```


   




