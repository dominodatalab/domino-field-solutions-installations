## Pre-requisites

> Check with your Domino CSM before using this capability. It is a significant departure from how Domino manages
> pod identities and may not be suitable for your requirements. It is currently only tested upto Domino version 5.8

1. Install [Domsed](../domsed/README.md)
2. For users needing to assume AWS role identities create service account per user in the `domino-compute` namespace. 
   For example,

```shell

export compute_namespace=domino-compute

kubectl -n ${compute_namespace} create sa vaibhavd
kubectl -n ${compute_namespace} create sa sameerw
kubectl -n ${compute_namespace} create sa marcd
```
## Installation

1. Update the [mutation](user-identity-based-irsa.yaml) as follows:

Update the environment variable mutation as appropriate to your environment

```yaml
  modifyEnv:
    env:
    - name: AWS_WEB_IDENTITY_TOKEN_FILE
      value: /var/run/secrets/eks.amazonaws.com/serviceaccount/token
    - name: AWS_CONFIG_FILE
      value: /var/run/.aws/config
    - name: AWS_DEFAULT_REGION
      value: us-west-2
    - name: AWS_REGION
      value: us-west-2
    - name: AWS_STS_REGIONAL_ENDPOINTS
      value: regional
```
Next update the user to K8s service account mappings (see Pre-requisites)

```yaml
- cloudWorkloadIdentity:
    cloud_type: aws
    default_sa: ""
    user_mappings:
      domino-sameerw: sameerw
      domino-vaibhavd: vaibhavd
      domino-marcd: marcd
```

2. Apply the mutations

```shell
export platform_namespace=domino-platform
kubectl -n ${platform_namespace} apply -f ./user-identity-based-irsa.yaml
```

3. Update your AWS trust policy for the role the user wants to assume (Ex. AWS Role `sw-irsa-test-role`)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::111111111111:oidc-provider/oidc.eks.<AWS_REGION>.amazonaws.com/id/<OIDC_ID>"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.<AWS_REGION>.amazonaws.com/id/<OIDC_ID>:sub": [
                       "system:serviceaccount:domino-compute:vaibhavd",
                        "system:serviceaccount:domino-compute:sameerw",
                        "system:serviceaccount:domino-compute:marcd"
                    ]
                }
            }
        }
    ]
}
```

4. Next as one of the mapped users start a workspace and run the following Python code

```python
import os
## You can change this to any role you (based on the k8s service account) have permission to assume
os.environ['AWS_ROLE_ARN']='arn:aws:iam::111111111111:role/sw-irsa-test-role'

## Now verify you have assumed it successfully
import boto3.session
session = boto3.session.Session()
sts_client = session.client('sts')
sts_client.get_caller_identity()
```

This should produce the output below which indicates that you have successfully assumed the role

```shell
{'UserId': 'AROA5YW464O6XT4444V43:botocore-session-1701963056',
 'Account': '111111111111',
 'Arn': 'arn:aws:sts::111111111111:assumed-role/sw-irsa-test-role/botocore-session-1701963056',
 'ResponseMetadata': {'RequestId': '77f078d6-237f-4351-9527-c959d7b409a8',
  'HTTPStatusCode': 200,
  'HTTPHeaders': {'x-amzn-requestid': '77f078d6-237f-4351-9527-c959d7b409a8',
   'content-type': 'text/xml',
   'content-length': '478',
   'date': 'Thu, 07 Dec 2023 15:30:56 GMT'},
  'RetryAttempts': 0}}

```

