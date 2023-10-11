Domsed/IRSA Installation - AWS Setup

## Two AWS Accounts

We need one AWS Role for the IRSA service pod to assume.

And we need two ***logical*** AWS Accounts for this installation. This is a general case. They can be the 
same ***physical*** AWS Acccount.

`{EKS_AWS_ACCOUNT}` - This is the AWS Account where the EKS cluster hosting Domino is installed
`{ASSETS_AWS_ACCOUNT}` - This is the AWS Account where your corporate assets like `s3` buckets are hosted



## IRSA Service Role

This is a role that the IRSA Service pod assumes. This role must be able to update the policy files of the proxy aws roles.

For the roles assumed by Domino workloads we follow the terminology below:

1. `AWS ROLE` - This is the role which accesses AWS roles on behalf of the user
2. `AWS PROXY ROLE` - This is the role that only has the permissions to assume the `AWS ROLE`

The reason we need this pair is because the IRSA requires that the trust policy of role be 
modified to allow the workload service account to assume it. The default size of this trust policy
is 2048 characters. The maximum size is 4096 characters.

Domino creates a new service account for each workloads. It is possible for a busy Domino
installation with a large number of workloads to exceed this limit. The paired design
allows for horizontal scaling. You can add x/2 number of users to two organizations mapping to a different 
`AWS PROXY ROLE` each of which maps to the same `AWS ROLE`. You can make it x/n for 
any number of users which allows this design to scale as needed indefinitely.


### IRSA Service Role

The following role must be created:

Role Name : `irsa-svc-role`

Trust Relationship:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::{EKS_AWS_ACCOUNT}:oidc-provider/oidc.eks.{EKS_CLUSTER_REGION}.amazonaws.com/id/{EKS_OIDC_ID}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "oidc.eks.{EKS_CLUSTER_REGION}.amazonaws.com/id/{EKS_OIDC_ID}:sub": "system:serviceaccount:domino-field:irsa"
                }
            }
        }
    ]
}
```

Permission Policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "acmeirsaadmin",
            "Effect": "Allow",
            "Action": [
                "iam:ListPolicies",
                "iam:ListPolicyVersions",
                "iam:ListRolePolicies",
                "iam:ListRoles",
                "iam:GetRole",
                "iam:PutRolePolicy",
                "iam:UpdateAssumeRolePolicy"
            ],
            "Resource": "arn:aws:iam::{EKS_AWS_ACCOUNT}:role/*"
        }
    ]
}
```

Note that the `"Resource": "arn:aws:iam::{EKS_AWS_ACCOUNT}:role/*"` allows this policy to update
the policies of any role. You should restrict this to only allow updating the 
proxy policies of the roles, your domino workloads can assume.

#### IRSA Workload Proxy Role

For each role the workload assumes there will be a proxy role.

Role Name : Provide a proxy role name. Use the format `proxy-{aws-role-name}`

Trust Relationship:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::{EKS_AWS_ACCOUNT}:oidc-provider/oidc.eks.{EKS_CLUSTER_REGION}.amazonaws.com/id/{EKS_OIDC_ID}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.{EKS_CLUSTER_REGION}.amazonaws.com/id/{EKS_OIDC_ID}:sub": ""
                }
            }
        }
    ]
}
```

Permission Policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": "arn:aws:iam::{ASSETS_AWS_ACCOUNT}:role/{aws-role-name}"
        }
    ]
}
```

#### Workload Role

This is the actual role in the `ASSETS_AWS_ACCOUNT` corresponding to one or many proxy roles.

Note that this role can be have many proxy roles. But a proxy role can point to only one such role

Role Name : Provide a proxy role name. Use the format `{aws-role-name}`

Trust Relationship:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::{EKS_AWS_ACCOUNT}:root"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```
This trust policy allows any role in the `EKS_AWS_ACCOUNT` to assume itself. You might want to
restrict to only a list of corresponding proxy roles permitted to assume itself.

Example Permission Policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ListObjectsInBucket",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::{bucket_name}"
            ]
        }
    ]
}
```

