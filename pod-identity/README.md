# Use Pod Identity with EKS for Domino Workloads based on Domino Service Accounts

[EKS Pod Identities](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html) allows a K8s pod assume an AWS role based on the ServiceAccount associated with the Pod

Domino Workloads have a dynamically generated Service Accounts. To use Pod Identities we need to use Static Service Accounts

Follow these steps:

1. Domino Service Accounts can be created based on the following [docs](https://docs.dominodatalab.com/en/latest/admin_guide/6921e5/domino-service-accounts/). 
   Assume the following service accounts:
    ```commandline
    svc-user-1
    svc-user-2
    svc-user-3
    ```
2. Create a unique K8s account for each of the Domino Service Accounts created above which need to assume an IAM role using Pod Identities for EKS
```shell
kubectl -n create sa svc-user-1 -n domino-compute
kubectl -n create sa svc-user-2 -n domino-compute
```
3. Apply the following mutation 

```yaml
apiVersion: apps.dominodatalab.com/v1alpha1
kind: Mutation
metadata:
  name: map-domino-user-k8s-sa
  namespace: domino-platform
rules:
- labelSelectors:
  - dominodatalab.com/workload-type in (Workspace,Batch,Scheduled)
  cloudWorkloadIdentity:
    cloud_type: aws
    default_sa: ""
    assume_sa_mapping: false
    user_mappings:
      svc-user-1: svc-user-1 
      svc-user-2: svc-user-2
```
The workloads started by Domino service accounts `svc-user-1` and `svc-user-2` will have 
the dynamically generated K8s Service Account's replaced with the statically created K8s service accounts.

For workloads started by `svc-user-3` the K8s service account will be a dynamic one.

4. Configure [EKS Pod Identities](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html) by following the AWS docs
5. Map the Domino Service Accounts `svc-user-1 ` and `svc-user-2` to custom IAM Roles 


