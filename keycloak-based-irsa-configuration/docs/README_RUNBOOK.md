# Helm installation


## External Prerequisites

### Move the following images into your private registry

1. `quay.io/domino/domino-kc-groups-based-irsa-mapper:prod-v1`
2. `quay.io/domino/cloud-identity-management-utils:prod-v1`

### Create a IAM Role for the service account executing the domino-irsa-lite-admin service

The service which manages your IRSA mappings also needs to have an IAM role with permissions to update trust policies of
the roles. Add the following policy to the role:

**Note the Resource ARN pattern. Update the account id and roles as needed**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:UpdateAssumeRolePolicy",
                "iam:GetRole"
            ],
            "Resource": "arn:aws:iam::11111111111:role/acme_domino_*"
        }
    ]
}
```

Add the following trust policy to allow the service account to assume this role:

```json

    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::11111111111:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/AAAAAAAAAAAAAAAAAAAAAAAA"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.us-west-2.amazonaws.com/id/AAAAAAAAAAAAAAAAAAAAAAAA:sub": "system:*:domino-field:domino-irsa-lite-admin-sa",
                    "oidc.eks.us-west-2.amazonaws.com/id/AAAAAAAAAAAAAAAAAAAAAAAA:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
```

## Installation Prerequisites

### Create Namespace


```bash
export ns=domino-field
kubectl create namespace ${ns}

# Label Namespace for Domino Compute
kubectl label namespace ${ns} domino-compute=true
```

### Create a keycloak client

For this you will need your Keycloak admin username and password. The username is usually `keycloak`.
```shell
export ps=domino-platform
kubectl get secret -n $ps keycloak-http -o jsonpath='{.data.password}' | base64 --decode && echo
```

Create a Keycloak  client with minimum privileges 

```shell
export KC_URL="https://<DOMINO_URL>/auth"
export REALM="DominoRealm"
export KC_ADMIN_USER="keycloak"
export CLIENT_ID="irsa-mapping-manager"
export KC_ADMIN_PASS="<ADD_ADMIN_PASS>"
python ./scripts/bootstrap_keycloak_client.py | jq
```

This will generate an output like this:

```
>> Minting admin token (master realm, admin-cli)…
>> Checking if client 'irsa-mapping-manager already exists…
>> Creating client 'irsa-mapping-manager'…
>> Fetching service-account user…
>> Resolving realm-management client…
>> Assigning roles to service-account user…
>> Fetching client secret…
{
  "kc_url": "https://<DOMINO_URL>.cs.domino.tech/auth",
  "realm": "DominoRealm",
  "client_id": "irsa-mapping-manager-3",
  "client_uuid": "5f...",
  "client_secret": "j5...",
  "service_account_user_id": "bf...",
  "granted_roles_client": "realm-management",
  "granted_roles": [
    "query-groups",
    "query-users",
    "manage-users"
  ]
}
```

Note the `client_secret`. You will need to add this to your `value.yaml` for helm install/upgrade

## Helm

### Install

Update the value of `client_secret` in `helm/cloud-identity-management-utils/values.yaml` with the value obtained from previous step.

```bash
export ns=domino-field 
helm install  cloud-identity-management-utils  helm/cloud-identity-management-utils/ -n $ns -f helm/cloud-identity-management-utils/myvalues.yaml
```


### Upgrade

```bash
export ns=domino-field 
helm upgrade  cloud-identity-management-utils  helm/cloud-identity-management-utils/ -n $ns -f helm/cloud-identity-management-utils/myvalues.yaml

```

### Delete

```bash
export ns=domino-field 
helm delete cloud-identity-management-utils -n $ns 

```

## Configure the user group

```shell
export KC_URL="http://keycloak-http.domino-platform.svc.cluster.local:80/auth"
export REALM="DominoRealm"
export KC_ADMIN_USER="keycloak"
export CLIENT_ID="irsa-mapping-manager"
export KC_ADMIN_PASS="<ADD_PASS>"
export KC_GROUPS_CONFIG="./scripts/example_kc_groups.json"
# Dry run to see the changes
python ./scripts/kc_apply_group_spec.py --spec ${KC_GROUPS_CONFIG} --dry-run

# Apply the changes if you like them
python ./scripts/kc_apply_group_spec.py --spec ${KC_GROUPS_CONFIG}
```

## IRSA configuration

1. Create a default group for all users with a trust policy that looks like this-
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::11111111111:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/<ID>"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.us-west-2.amazonaws.com/id/<ID>:aud": "sts.amazonaws.com",
                    "oidc.eks.us-west-2.amazonaws.com/id/<ID>:sub": "system:serviceaccount:domino-compute:*"
                }
            }
        }
    ]
} 
```
All service accounts in Domino Compute namespace can assume this role by default.
***Make sure you do not add any policies to this role. This is only to test irsa is configured by AWS webhooks***

2. Update the trust policies of all the IAM roles managed by Keycloak groups to include the following condition-

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::11111111111:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/<ID>"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.us-west-2.amazonaws.com/id/<ID>:aud": "sts.amazonaws.com",
                    "oidc.eks.us-west-2.amazonaws.com/id/<ID>:sub": ""
                }
            }
        }
    ]
} 
```
Domino endpoints will not update anything but the `sub` claim. The others will need to match what is configured in the service
used to update this trust policy. This includes the OIDC provider URL and the audience.

## Sync Domino users to K8s SA

```shell
#export DOMINO_API_PROXY="http://localhost:8899"
export DEFAULT_AWS_ROLE_ARN="arn:aws:iam::11111111111:role/acme_domino_default_role"

# Dry run first
python ./scripts/sync.py users --default-aws-role-arn $DEFAULT_AWS_ROLE_ARN --dry-run

# Apply
python ./scripts/sync.py users --default-aws-role-arn $DEFAULT_AWS_ROLE_ARN
```

## Sync Keycloak groups to IAM IRSA

```shell
export DOMINO_API_PROXY="http://localhost:8899"

# Dry run first
python ./scripts/sync.py irsa --dry-run

# Apply
python ./scripts/sync.py irsa
```

## Summary

This is a five step process:

1. Create Keycloak client with minimum privileges
2. Install the helm chart for cloud-identity-management-utils
3. Configure Keycloak groups and map them to IAM roles with IRSA trust policies
4. Sync Domino users to K8s Service Accounts with proper annotations
5. Sync Keycloak groups to IAM roles with IRSA trust policies

Step (1) and (2) need to be done only once. 

Steps (3), (4) and (5) need to be repeated as needed in the following situations:
1. New user gets added to Domino. You will need to update your JSON spec used in step (3) to include this user in a Keycloak group.
   Run step (4) to create a user based K8s Service Account and then run step (5) to update the IAM role mappings.

2. New IAM role needs to be created and mapped to a Keycloak group. Update the JSON spec used in step (3) to include this mapping
   and run step 5 again.

3. User changes groups in Domino. Update the JSON spec used in step (3) to reflect this change and run step (5) again.


