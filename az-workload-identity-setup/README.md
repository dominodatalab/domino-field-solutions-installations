## Installation

1. First install [Domsed](../domsed/README.md) v1.3.3

2. Create K8s service accounts for domino users who need to 
   assume a AZ Workload Identity
   ```yaml
        apiVersion: v1
        kind: ServiceAccount
        metadata:
        annotations:
            azure.workload.identity/client-id: fb2489bd-f790-41f2-bd9a-274b1a9e456b  
        name: user-test-1-sa
        namespace: domino-compute
    ```
    Note above that the client-id comes from Azure Application Id in Azure ENTRA ID
   
   ![AZ Application(assets/AZApplication.png)
   
   > For the record it does not have to be a valid client id. The annotation 
   > needs to exist if it points to non-existent client id. More on that later

   Create a unique K8s service account for every user who needs to assume AZ Workload Identity

   Note : In order for workload identity to work we need understand two attributes
   
   1. TenantId - You will get this under the section `Microsoft Entra ID`
   2. Client Id- This is the Application ID of the service principal or managed identity

   In our example we have used managed identity. So we go to the managed identity and add 
   Federated Credentials. This is where you attach the K8s SA account to a Managed Identity
   (or Service Principal)
   
   Make [sure](https://learn.microsoft.com/en-us/azure/aks/use-oidc-issuer) your AKS cluster has a OIDC provider associated with it.
   You will need this issuer url
   
   Set the following values:
   1. `Federated Credential Scenario` - `Configure a Kubernetes service account to get tokens as this application and access Azure resources
   2. Cluster Issuer URL - The OIDC provider associated with the Cluster
   3. Namespace - domino-plaform
   4. Service Account - The K8s Service Account you created above
   
   You will need one entry for every K8s service account that needs to assume
   this managed identity
   
3. Next add the following mutation

```yaml
apiVersion: apps.dominodatalab.com/v1alpha1
kind: Mutation
metadata:
  name: azure-wi
  namespace: domino-platform
rules:
- cloudWorkloadIdentity:
    assume_sa_mapping: false
    cloud_type: azure
    default_sa: ''
    user_mappings:
      user-test-1: user-test-1-sa
      user-test-2: user-test-2-sa
  matchBuilds: false
```

## Using the Workload Identity

As one the mapped users, start a workspace. When the workspace starts up
start a terminal and type the following:

```shell
env | grep AZURE
```

This should come back with the following 4 identifiers
```shell
AZURE_TENANT_ID=<MICROSOFT_ENTRA_ID_TENANT_ID>
AZURE_FEDERATED_TOKEN_FILE=/var/run/secrets/azure/tokens/azure-identity-token
AZURE_AUTHORITY_HOST=https://login.microsoftonline.com/
AZURE_CLIENT_ID=<THE_CLIENT_ID_YOU_ADDED_TO_THE_ANNOTATION>
```
Next run the following:

```python


from azure.identity import WorkloadIdentityCredential
from azure.storage.blob import BlobServiceClient
import os

account_url = f'https://blobtoazurepoc.blob.core.windows.net'
os.environ['AZURE_CLIENT_ID']='<FIX IT HERE>'
creds = WorkloadIdentityCredential(tenant_id=os.environ.get('AZURE_TENANT_ID'),
                                   client_id=os.environ.get('AZURE_CLIENT_ID'),
                                   token_file=os.environ.get('AZURE_FEDERATED_TOKEN_FILE')
                                  )

blob_service_client = BlobServiceClient(account_url, credential=creds)

[print(container) for container in blob_service_client.list_containers(include_metadata=True)]
```