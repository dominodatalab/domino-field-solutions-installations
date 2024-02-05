## Using IRSA with Domino Workloads (using Static User based Service Accounts)

This document outlines how to use IRSA with Domino with statically defined Service Accounts for users.

This approach assumes joint responsibility for propagating cloud identity, between Domino and the Customer.

Domino assumes the responsibility for providing tooling to configure the K8s side and workload definitions 
which allow the Domino workload to consistently identify itself as the workload owning user (A K8s Service Account
in the `domino-compute` namespace) . 

The customer will assume responsibility for:

1. Providing the right permissions to this user (via the K8s SA)
   by updating the trust policy for the IAM Roles. This action happens outside Domino.
   
2. A Domino user will bring awareness of which roles they are allowed to assume. This can be done by
   either setting the environment variable `AWS_ROLE_ARN` in the workspace or even by self configuring the
   `AWS_CONFIG_FILE` env variable location to a valid `AWS_CONFIG_FILE`. 
   
3. Domino may provide automation for (2) if requested but (1) will be the responsibility of the customer

This approach has two benefits for the customer:

1. Domino service pods no longer have elevated privileges to update IAM Trust Policies. This reduces the
   blast radius if the service pod is compromised
   
2. IAM Trust Policy updates can sometimes take several minutes to propagate across the AWS Ecosystem. This
   can result in job failures even if the trust policy has been updated. Allowing trust policies to be 
   updated statically on a separate schedule using static Service Accounts provides more reliability for jobs
   which need AWS Access.
   


## Installation

Assume our root foolder as the repo folder

```shell
export PROJECT_ROOT_DIR=<PATH_TO_THE_REPO_FOLDER>
#Ex. export PROJECT_ROOT_DIR=$HOME/repos/domino-field-solutions-installations
```

### Pre-requisites

If you have the previous `irsa` solution installed, uninstal it

```shell
cd $PROJECT_ROOT_DIR/irsa
```

```shell
export field_namespace=domino-field
export compute_namespace=domino-compute

helm delete irsa -n ${field_namespace}
kubectl delete secret irsa-certs -n ${compute_namespace}
```

### Install Domsed


1. Delete existing Domsed
```shell
cd $PROJECT_ROOT_DIR/domsed/
```

```shell
export platform_namespace=domino-platform
export compute_namespace=domino-compute
helm delete domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} "operator-enabled"-
```

2. Install Domsed

```shell
cd $PROJECT_ROOT_DIR/domsed/
```

```shell
export platform_namespace=domino-platform
export compute_namespace=domino-compute
helm install -f helm/domsed/values.yaml domsed helm/domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} operator-enabled=true
```

3. Upgrade Domsed (If necessary)

```shell
cd $PROJECT_ROOT_DIR/domsed/
```

```shell
export platform_namespace=domino-platform
export compute_namespace=domino-compute
helm upgrade -f helm/domsed/values.yaml domsed helm/domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} operator-enabled=true
```

### Install Domino Extensions API With Cloud Identity Mutations

1. Delete helm chart

```shell
cd $PROJECT_ROOT_DIR/domino-extensions-api/
```

```shell
export field_namespace=domino-field
helm delete  domino-extensions-api -n ${field_namespace}
```

2. Install helm chart

```shell
cd $PROJECT_ROOT_DIR/domino-extensions-api/
```

```shell
export field_namespace=domino-field
helm install -f helm/domino-extensions-api/values.yaml domino-extensions-api helm/domino-extensions-api -n ${field_namespace}
```

3. Upgrade helm chart

```shell
cd $PROJECT_ROOT_DIR/domino-extensions-api/
```

```shell
export field_namespace=domino-field
helm upgrade -f helm/domino-extensions-api/values.yaml extendedapi helm/domino-extensions-api -n ${field_namespace}
```

### Admin Perspective

#### Generating K8s Service Account per User

If your domino user names only contain alpha-numeric characters execute the following in an Admin 
workspace to generate service accounts
```python
import os
import requests
from urllib.parse import urljoin
domino_user_api_key = os.environ.get("DOMINO_USER_API_KEY")
headers = {
        "Content-Type": "application/json",
        "X-Domino-Api-Key": domino_user_api_key,
    }
EXTENDED_API_URI = 'https://domino-extensions-api-svc.domino-field.svc.cluster.local'
manage_sa_url = urljoin(EXTENDED_API_URI, 'domino-sa-management-api/usersa')
get_all_sa_url = urljoin(EXTENDED_API_URI, 'domino-sa-management-api/allusersa')

url: str = urljoin(os.environ.get('DOMINO_API_HOST'), '/api/users/v1/users')
users = requests.get(url, headers=headers).json()
for u in users['users']:
    id = u['id']
    name = u['userName']
    body = {'user_name': name, 'user_id': id, 'domino_sa': None}
    print(f'Generating K8s SA for user {body} ')
    response = requests.post(manage_sa_url, headers=headers, json=body, verify=False)
    if response.status_code==200:
        print('Success')
        print(response.status_code)
        print(response.text)
    else:
        print('Failure')
        print(response.status_code)
        print(response.text)        
    print('\n')
```

All your service accounts will be generated. If your domino user namesdo not contain any special 
characters outside of `.` or `-` and do not have an email address as the user name there is nothing more to do.

If not refer to the next section

#### Mapping K8s Service Account to a Domino User for Cloud Identity

WORK IN PROGRESS. WATCH THIS SPACE FOR DETAILS

### User Perspective

#### Static Configuration on Domino Side

1. Assume you have two usesr called `john-doe` and `jane-doe` in Domino
2. Next assume you have two corresponding Service Account in `domino-compute` namespace as `john-doe` and `jane-doe` in Domino
   
### Static Configuration on EKS Side

1. Assume you have a role `arn:aws:eks:us-west-2:111111111111:cluster/myeksrole`
2. Next assume that the OIDC Connect Provide associated with your EKS account is 
`https://oidc.eks.us-west-2.amazonaws.com/id/4BFF6850498B7D7ED319EFXXXXXXXXX`
3. Update the Trust Policy of this role as follows to allow both users the permissions to assume this role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::111111111111:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/4BFF6850498B7D7ED319EFXXXXXXXXX"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.us-west-2.amazonaws.com/id/4BFF6850498B7D7ED319EFXXXXXXXXX:aud": "sts.amazonaws.com",
                    "oidc.eks.us-west-2.amazonaws.com/id/4BFF6850498B7D7ED319EFXXXXXXXXX:sub": ["*:domino-compute:john-doe","*:domino-compute:jane-doe"]
                }
            }
        }
    ]
}
```


3. `john-doe` launches a workspace. Once logged in notice that the following environment variables are set  
    ```shell
      AWS_DEFAULT_REGION=us-west-2
      AWS_REGION=us-west-2
      AWS_CONFIG_FILE=/var/run/.aws/config
      AWS_WEB_IDENTITY_TOKEN_FILE=/var/run/secrets/eks.amazonaws.com/serviceaccount/token
      AWS_STS_REGIONAL_ENDPOINTS=regional
    ```    
    The workspace is now IRSA enabled. To assume the role all `john-doe` has to do is run the following

```python
import boto3
import os
os.environ['AWS_ROLE_ARN']='arn:aws:iam::111111111111:role/myeksrole'
sts = boto3.client('sts')
sts.get_caller_identity()
```
This will produce and output like

```shell
{'UserId': 'AROA5YW464O6VNYC7IYHD:botocore-session-1707137859',
 'Account': '946429944765',
 'Arn': 'arn:aws:sts::111111111111:assumed-role/myeksrole/botocore-session-1707137859',
 'ResponseMetadata': {'RequestId': 'f263cb13-d050-4f14-b834-3f1bae3071c7',
  'HTTPStatusCode': 200,
  'HTTPHeaders': {'x-amzn-requestid': 'f263cb13-d050-4f14-b834-3f1bae3071c7',
   'content-type': 'text/xml',
   'content-length': '482',
   'date': 'Mon, 05 Feb 2024 12:59:03 GMT'},
  'RetryAttempts': 0}}

```

If `jane-doe` launches a workspace and runs the same code as above, the output will be the same.

However if you head over to AWS CLOUD TRAIL, you will clearly see that the role was assumed once by `john-doe`
and one by `jane-doe`. This helps with traceability

#### Assuming Roles in Cross Account Scenario

Now let us configure these users to assume a role `myassetrole` in another AWS Account `222222222222`
`arn:aws:eks:us-west-2:222222222222:cluster/myassetrole`

First attach a permission policy to the role `arn:aws:eks:us-west-2:111111111111:cluster/myeksrole`
as follows:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": "arn:aws:iam::222222222222:role/myassetrole"
        }
    ]
}
```

Head over to AWS Account `222222222222` IAM and open the trust policy of role `arn:aws:eks:us-west-2:222222222222:cluster/myassetrole`

Update it as follows to allow sessions of role `arn:aws:eks:us-west-2:111111111111:cluster/myeksrole` to
assume this role

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::111111111111:role/myeksrole"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

And we are all set. In your workspace add the following set of lines 

```python
import boto3
import os
os.environ['AWS_ROLE_ARN']='arn:aws:iam::111111111111:role/myeksrole'
sts = boto3.client('sts')
sts.get_caller_identity()
```
This assumes the role `myeksrole` in the `111111111111` account

Next run 
```python

response = sts.assume_role(
    RoleArn="arn:aws:iam::222222222222:role/myassetrole",
    RoleSessionName="myassetrole"
)
```

The response object will return temporary credentials for role `arn:aws:iam::222222222222:role/myassetrole`