## IRSA ROLE For MLFLOW Deployment

Configure the AWS Role the mlflow deployment in the domino-platform namespace will assume

Define S3 Bucket for MLFLOW Artifact Storage - `arn:aws:s3:::sw-domino-project-based-mlflow`

Create MLFLOW Role which will be assumed by mlflow Deployment in the domino-platform namespace - `arn:aws:iam::<ACCOUNT_NO>:role/sw-mlflow-full-access`

Update its trust policy to allow the mlflow service account to assume this role
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::<ACCOUNT_NO>:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/<OIDC_ID>"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "oidc.eks.us-west-2.amazonaws.com/id/<OIDC_ID>:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
```


Attach the following policy to the role - `sw-mlflow-full-access`
```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "VisualEditor1",
			"Effect": "Allow",
			"Action": "s3:*",
			"Resource": "arn:aws:s3:::sw-domino-project-based-mlflow"
		}
	]
}
```

## Update the mlflow Service Account in the `domino-platform` namespace to assume the Role

Add the annotation to the service account
```shell
eks.amazonaws.com/role-arn: arn:aws:iam::<ACCOUNT_NO>:role/<AWS_ROLE>
```


## Update the mlflow deployment in the domino-platform namespace

Add the following volume to the Pod
```yaml
volumes:
      - name: aws-user-token
        projected:
          defaultMode: 422
          sources:
          - serviceAccountToken:
              audience: sts.amazonaws.com
              expirationSeconds: 86400
              path: token
```


Update the environment variable `MLFLOW_ARTIFACTS_DESTINATION` to point to the s3 bucket defined earlier. 
In my example it is `sw-domino-project-based-mlflow`

Add the following volume and volume mount (to the `mlflow` container)
```yaml
volumeMounts:
      - mountPath: /var/run/secrets/eks.amazonaws.com/serviceaccount/
        name: aws-user-token
```
      
      

Add the following environment variables to the `mlflow` container
```yaml
      - env:
        - name: AWS_WEB_IDENTITY_TOKEN_FILE
          value: /var/run/secrets/eks.amazonaws.com/serviceaccount/token
        - name: AWS_ROLE_ARN
          value: arn:aws:iam::<ACCOUNT_NO>:role/<AWS_ROLE>
```
Remove the environment variable for `AWS_CONFIG_FILE`

## Restart the mlflow deployment in the `domino-platform` namespace

```shell
kubectl -n domino-platform rollout restart deployment mlflow
```


## Add a Trigger to the `mlflow` tracking database

First get the `mlflow` user password for the tracking database

```shell
export ps=domino-platform
kubectl get secret mlflow-postgresql  -n $ps -o jsonpath='{.data.postgresql-password}' | base64 --decode && echo

```

Next port-forward into the database
```shell
kubectl port-forward -n domino-platform svc/mlflow-postgresql 5432:5432
```


Connect to the database via localhost:5432 using a postgres client and add the following trigger

```shell
CREATE OR REPLACE FUNCTION public.update_artifact_location()  
returns trigger 
language plpgsql
as $$
begin
	if new.key='mlflow.domino.project_id' then 
		UPDATE mlflow.public.experiments 
		set artifact_location = concat('s3://sw-domino-project-based-mlflow/mlflow/',new.value)
		where mlflow.public.experiments.experiment_id = new.experiment_id ;
		return new;
	else
		return null;
	end if;
end;
$$; 


create trigger update_artifact_location_trg
   after insert 
      on public.experiment_tags 
      for each row 
      execute function update_artifact_location(); 
```


This creates a artifact as follows:
```shell
s3://<BUCKET_NAME>/mlflow/<DOMINO_PROJECT_ID>
```



Instead if you wanted to actually set an artifact path that was not relative but purely custom use the following trigger

```shell
CREATE OR REPLACE FUNCTION public.update_artifact_location()  
returns trigger 
language plpgsql
as $$
begin
	if new.key='domino.artifacts_location' then 
		UPDATE mlflow.public.experiments 
		set artifact_location = new.value
		where mlflow.public.experiments.experiment_id = new.experiment_id ;
		return new;
	else
		return null;
	end if;
end;
$$; 
```


Now if you create an experiment as follows you can have an entirely custom location. This is done via the experiment tag  domino.artifacts_location whose value can be chosen by the user. Note that the two approaches as defined as mutually exclusive. You can update the trigger with more complex implementations based on your unique requirements


```python

import pandas as pd
import os
import mlflow
user_name = os.environ['DOMINO_USER_NAME']
project_id = os.environ['DOMINO_PROJECT_ID'] 
exp_name = f'Exp-test-{user_name}'
exp = None
exp = mlflow.get_experiment_by_name(exp_name)    
if not exp:
    my_tags={'domino.artifacts_location':'s3://bw-scale-test/mlflow/my-test-project/'}
    print(my_tags.items())
    print('Experiment Not Found Create it')
    mlflow.create_experiment(exp_name,tags=my_tags)    
exp = mlflow.get_experiment_by_name(exp_name)         
print(exp)

```

It is not your responsibility to ensure that the user service accounts have the right permissions to write to the 
appropriate S3 folders where the `mlflow run id` is created

Writing to S3 only requires setting the environment variabls `AWS_ROLE_ARN` and the domsed mutation has ensured that
all the web identity tokens are properly configured.

However to write to S3 directly from the workspace using the MLFLOW Artifact Store API you also need to configure
the AWS Access Key Environment Variables as follows

```python
import os
import boto3
import boto3.session

AWS_ACCOUNT_NO=os.environ['AWS_ACCOUNT_NO']
AWS_ROLE_NAME=os.environ['AWS_ROLE_NAME']
AWS_ROLE_ARN=f'arn:aws:iam::{AWS_ACCOUNT_NO}:role/{AWS_ROLE_NAME}'
## The above is only helper. Note from here
os.environ['AWS_ROLE_ARN'] = AWS_ROLE_ARN
session = boto3.session.Session()
sts_client = session.client('sts')
sts_client.get_caller_identity()

session = boto3.Session()
region_name=session.region_name    
creds = session.get_credentials()
os.environ['AWS_ACCESS_KEY_ID'] = session.get_credentials().access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = session.get_credentials().secret_key
os.environ['AWS_SESSION_TOKEN'] = session.get_credentials().token
```
