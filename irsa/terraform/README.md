# Example implementation of AWS resources required for Domino IRSA

This directory contains an example of the roles and policies involved in [allowing Domino workloads to assume AWS IAM roles](https://github.com/dominodatalab/irsa_installation)
across (or within) AWS accounts.

## Pre-requisites

Create an IAM OIDC provider for your cluster using [eksctl or AWS Management Console](https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html)

## Optional (but recommended) Pre-requisites

Create the IAM policies that will be associated with the workload roles. Update the [irsa_workload_files.tf](irsa_workload_files.tf) file with the name of the policy to be associated
with each workload role.

## Requirements

This code assumes that you have access to two separate AWS accounts, and that you have an awscli configuration with two profiles:

1. Account: `domino-eks`. This account contains the EKS cluster that runs your Domino workloads. This account will be used to set up the [IRSA service account for the Domino IRSA app](domino_irsa_svc.tf), as well an [example proxy role](proxy_irsa.tf) that the app will modify dynamically to allow users to assume this proxy role.

2. Account: `asset-acct`. This account contains whatever AWS resources you want your Domino users to be able to access. For the purpose of this example,
we've set up [an IAM role that can list the S3 buckets within this account](asset_roles.tf).

NOTE: this _can_ run within one AWS account, but two awscli profiles would still be required.

## Setup

Run `terraform init` to initialize the AWS provider.

Run `terraform plan` to show the changes that will be made.

*NOTE*: If you wish to use the demo policy (with `s3:ListBucket` permissions), run `terraform apply -target aws_iam_policy.irsa-workload-example-policy` before continuing. You will also 
need to run this after a `terraform destroy`.

## Deploy

Run `terraform apply` to apply the planned changes.

## Destroy

Run `terraform destroy` to remove all of the applied changes.


## Selective Deploy: IRSA Service Account roles/policies only

To plan: `terraform plan -target aws_iam_role_policy_attachment.domino-irsa-svc`

To apply: `terraform apply -target aws_iam_role_policy_attachment.domino-irsa-svc`

## Selective Deploy: Asset roles/policies

*NOTE*: Due to object dependencies in the Terraform code, this deploy will also roll out the associated _proxy role_ on the EKS account in question.

To plan: `terraform plan -target aws_iam_role_policy_attachment.irsa-workload-example`

To apply: `terraform apply -target aws_iam_role_policy_attachment.irsa-workload-example`

## Selective Deploy: Proxy roles/policies only

*NOTE*: Due to object dependencies in the Terraform code, this deploy will also roll out the associated _workload role_ on the asset account in question.

To plan: `terraform plan -target aws_iam_role_policy_attachment.domino-irsa-proxy`

To apply: `terraform apply -target aws_iam_role_policy_attachment.domino-irsa-proxy`


