import requests
import json
import os

api_host = os.environ.get("EXTENDED_API_HOST", "domino-extendedapi-svc.domino-platform")
api_port = os.environ.get("EXTENDED_API_PORT", "80")
url = f"http://{api_host}:{api_port}/v4-extended/autoshutdownwksrules"

payload = json.dumps({
  "users": {
    "wadkars": 3600,
    "integration-test": 21600
  },
  "override_to_default": False
})
headers = {
  'X-Domino-Api-Key': os.environ.get('DOMINO_USER_API_KEY'),
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

## Alternatively from inside the workspace you could run. This is a safer approach to using the API Key

access_token_endpoint='http://localhost:8899/access-token'
resp = requests.get(access_token_endpoint)


token = resp.text
headers = {
             "Content-Type": "application/json",
             "Authorization": "Bearer " + token,
        }
response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

## Get All Environments including the base docker image they are based on

import requests
access_token_endpoint='http://localhost:8899/access-token'
resp = requests.get(access_token_endpoint)


token = resp.text
headers = {
             "Content-Type": "application/json",
             "Authorization": "Bearer " + token,
        }
params={'offset':0,'limit':10000}
url = f"http://extendedapi-svc.domino-platform.svc.cluster.local/api-extended/environments/beta/environments"
response = requests.request("GET", url, headers=headers,params=params)
print(response.status_code)
environments = response.json()['environments']
for e in environments:
    print(e)

## Get All Projects enhanced with the environment id in the settings

url = f"http://extendedapi-svc.domino-platform.svc.cluster.local/api-extended/projects/beta/projects"
response = requests.request("GET", url, headers=headers,params=params)
print(response.status_code)
projects = response.json()['projects']
for p in projects:
    print(p)


## Update Project Settings

project_id='ADD HERE'
url  = f"http://nucleus-frontend.domino-platform.svc.cluster.local:80/v4/projects/{project_id}/settings"

payload = json.dumps({
  "defaultEnvironmentId": "ADD THE ENV ID TO REPLACE WITH"
})

response = requests.request("PUT", url, headers=headers, data=payload)
print(response.text)
