## Pre-requisites

> **WARNING** -  You need Boto3 [SDK](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts-minimum-sdk.html) above a minimum version for IRSA to work
> You will experience strange errors if your SDK version is lower despite having the right configurations
> And example is `botocore.exceptions.PartialCredentialsError: Partial credentials found in assume-role, missing: source_profile or credential_source`


Create a namespace domino-field

```shell
kubectl create ns domino-field
kubectl label namespace domino-field  domino-compute=true
kubectl label namespace domino-field  domino-platform=true
```

## Delete IRSA
If you have already installed IRSA and want to delete it for reinstallation

```shell
helm delete irsa -n domino-field
export compute_namespace=domino-compute
kubectl delete secret irsa-certs -n ${compute_namespace}
```
> ***Attention***:  **After reinstalling IRSA you will need to recreate the mappings**

## IRSA Installation

a. First create organizations in domino matching the AWS role names

b. Update the `values.yaml` with the proper values

```shell

export platform_namespace=domino-platform
export compute_namespace=domino-compute
export field_namespace=domino-field

#Update the values.yaml with the above values
cd irsa
helm install -f ./values.yaml -n ${field_namespace} irsa helm/irsa
```

d. Copy the `irsa-certs` secret from the `domino-field` namespace to the `domino-compute` namespace
```shell
kubectl get secret irsa-certs --namespace=${field_namespace} -o yaml | sed 's/namespace: .*/namespace: domino-compute/' | kubectl apply -f -
```
This allows the IRSA service to become SSL enabled and invokable from the workloads in the `domino-compute` namespace

## IRSA Update

For helm updates run 
```shell
# Use the default ./helm/irsa/values.yaml
helm upgrade irsa helm/irsa -n ${field_namespace}
#or
helm upgrade -f ./values.yaml irsa helm/irsa -n ${field_namespace}
```

## Create Mappings

Open the notebook [enablement.ipynb](./enablement.ipynb). There is a section called `## Add/Update Role Mapping (Only Domino Administrators can make this call)`

This section is used to map Domino Organizations to AWS Roles (and AWS proxy roles)
```python
EKS_ACCOUNT_NO="<ADD>"
ASSETS_ACCOUNT_NO="<ADD"
#Fetch my mappings (Any user can do this)
import requests
import os
access_token_endpoint='http://localhost:8899/access-token'
resp = requests.get(access_token_endpoint)


token = resp.text
headers = {
             "Content-Type": "application/json",
             "Authorization": "Bearer " + token,
        }


endpoint='https://irsa-svc.domino-field/update_role_mapping'
body={
    "domino_org":"irsa-iamrole-list-bucket",
    "iam_role_arn":f"arn:aws:iam::{ASSETS_ACCOUNT_NO}:role/acme-list-bucket-role",
    "proxy_iam_role_arn":f"arn:aws:iam::{EKS_ACCOUNT_NO}:role/acme-list-bucket-role"
}
resp = requests.post(endpoint,headers=headers,json=body,verify=False)
body={
    "domino_org":"irsa-iamrole-read-bucket",
    "iam_role_arn":f"arn:aws:iam::{ASSETS_ACCOUNT_NO}:role/acme-read-bucket-role",
    "proxy_iam_role_arn":f"arn:aws:iam::{EKS_ACCOUNT_NO}:role/acme-read-bucket-role"
}
resp = requests.post(endpoint,headers=headers,json=body,verify=False)
body={
    "domino_org":"irsa-iamrole-update-bucket",
    "iam_role_arn":f"arn:aws:iam::{ASSETS_ACCOUNT_NO}:role/acme-update-bucket-role",
    "proxy_iam_role_arn":f"arn:aws:iam::{EKS_ACCOUNT_NO}:role/acme-update-bucket-role"
}
resp = requests.post(endpoint,headers=headers,json=body,verify=False)

```

Walk through this notebook to get an end to end walkthrough on how to use IRSA for Domino.

An example mapping in the configmap `domino-org-iamrole-mapping` in `domino-field` namespace is shown below:

The AWS Account number is `111111111111` and the actual roles `list-bucket-role`, `read-bucket-role` and 
`update-bucket-role` are mapped via proxy roles `acme-list-bucket-role`, `acme-read-bucket-role` and 
`acme-update-bucket-role` in the same account. Note that the actual roles and proxy roles can be in separate accounts


```yaml
apiVersion: v1
data:
  irsa-iamrole-list-bucket: '{"iam_role_arn": "arn:aws:iam::111111111111:role/list-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::111111111111:role/acme-list-bucket-role"}'
  irsa-iamrole-read-bucket: '{"iam_role_arn": "arn:aws:iam::111111111111:role/read-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::111111111111:role/acme-read-bucket-role"}'
  irsa-iamrole-update-bucket: '{"iam_role_arn": "arn:aws:iam::111111111111:role/update-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::946429944765:role/acme-update-bucket-role"}'
kind: ConfigMap
metadata:
  annotations:
    meta.helm.sh/release-name: irsa
    meta.helm.sh/release-namespace: domino-field  
  name: domino-org-iamrole-mapping
  namespace: domino-field
```

## Scaling

Each time a workload starts up the proxy role trust relationship is updated with the service account for the workload.
The maximum size of a trust policy document is 4096 characters (after requesting AWS to increase it. Default is 2048).

If you need more you will need to create additional domino-orgs and map the same role to a new proxy role. For example 
the above configmap would like the following:

```yaml
apiVersion: v1
data:
  irsa-iamrole-list-bucket: '{"iam_role_arn": "arn:aws:iam::111111111111:role/list-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::111111111111:role/acme-list-bucket-role"}'
  irsa-iamrole-list-bucket-2: '{"iam_role_arn": "arn:aws:iam::111111111111:role/list-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::111111111111:role/acme-list-bucket-role-2"}'
  irsa-iamrole-read-bucket: '{"iam_role_arn": "arn:aws:iam::111111111111:role/read-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::111111111111:role/acme-read-bucket-role"}'
  irsa-iamrole-update-bucket: '{"iam_role_arn": "arn:aws:iam::111111111111:role/update-bucket-role",
    "proxy_iam_role_arn": "arn:aws:iam::946429944765:role/acme-update-bucket-role"}'
kind: ConfigMap
metadata:
  annotations:
    meta.helm.sh/release-name: irsa
    meta.helm.sh/release-namespace: domino-field  
  name: domino-org-iamrole-mapping
  namespace: domino-field
```

We have added a new domino org `irsa-iamrole-list-bucket-2` and created a new proxy role `arn:aws:iam::111111111111:role/acme-list-bucket-role-2`
for the aws role `iam_role_arn": "arn:aws:iam::111111111111:role/list-bucket-role`.

Next redistribute the users added to domino org `irsa-iamrole-list-bucket` between the two orgs-
- `irsa-iamrole-list-bucket`
- `irsa-iamrole-list-bucket-2`

You scale with multiple mappings for the same role in this fashion. This allows the Domino-IRSA solution to scale to
a large number of simultaneous domino workloads despite each workload having a unique k8s service account.

The mappings are deleted when the workload ends.
