# Extended API Installation


1. Install the Domino Extended API client
```shell
export platform_namespace=domino-platform
helm install -f helm/extendedapi/values.yaml extendedapi helm/extendedapi -n ${platform_namespace}
```
2. To upgrade
```shell
export platform_namespace=domino-platform
helm upgrade -f helm/extendedapi/values.yaml extendedapi helm/extendedapi -n ${platform_namespace}
```
3. To delete

```shell
export platform_namespace=domino-platform
helm delete  extendedapi -n ${platform_namespace}
```

The details about the endpoints can be found in this [README.md](./domino-extended-api/README.md)

## Notebooks

1. [Domsed Client Notebook](./notebooks/domsed_client.ipynb) - An admin can use this notebook to manage mutations
2. [Control HW Tier Access Notebook](./notebooks/manage_hwtier_rbac.ipynb) - This is a helper notebooks to create the complex mutations to manage the hw tiers