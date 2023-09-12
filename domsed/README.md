## Domsed

Let us begin with assuming that Domsed is already installed. First set the root directory for this project

```shell
export PROJECT_ROOT_DIR=<PATH_TO_THE_REPO_FOLDER>
#Ex. export PROJECT_ROOT_DIR=$HOME/repos/domino_extended_api_installation
```



1. First we take a backup of all mutations

```shell

cd $PROJECT_ROOT_DIR/domsed
export platform_namespace=domino-platform
export compute_namespace=domino-compute
export backup_mutations_path=/tmp/mutations.yaml

kubectl -n $platform_namespace get mutations -oyaml > $backup_mutations_path
##Verify you got all the mutations in yaml format
cat $backup_mutations_path 
```

2. First lets delete the current `domsed` installation

a. If you have installed the older version of `domsed` which did not use helm charts do the following or skip to step b
```shell
cd $PROJECT_ROOT_DIR/domsed/scripts
export platform_namespace=domino-platform
export compute_namespace=domino-compute
./destroy.sh
```

b. If your Domsed instance was installed using Helm then do the following

```yaml
cd $PROJECT_ROOT_DIR/domsed/
helm delete domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} "operator-enabled"-
```

3. Reinstall Domsed

### Install Domsed
```shell
cd domsed
export platform_namespace=domino-platform
export compute_namespace=domino-compute
helm install -f values.yaml domsed helm/domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} operator-enabled=true
```


### Delete Domsed
```shell
cd $PROJECT_ROOT_DIR/domsed/
export platform_namespace=domino-platform
export compute_namespace=domino-compute
helm install -f values.yaml domsed helm/domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} operator-enabled=true
```

## Test Domsed

### Tail the logs

```shell
export platform_namespace=domino-platform
kubectl -n ${platform_namespace} get pods | grep operator
## Example output
operator-webhook-767cfcfddc-rh685                            1/1     Running    
kubectl -n ${platform_namespace} logs operator-webhook-767cfcfddc-rh685 -f
```
### Smoke Test

- Create this mutation object 
```shell
cat <<EOF | kubectl apply -f -
apiVersion: apps.dominodatalab.com/v1alpha1
kind: Mutation
metadata:
  name: label
  namespace: domino-platform
rules:
- # Insert label
  modifyLabel:
    key: "foo.com/bar"
    value: "out"
EOF
```

- Create this pod in the compute namespace
```shell
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  namespace: domino-compute
  labels:
    dominodatalab.com/hardware-tier-id: medium-k8s  
    dominodatalab.com/project-name: quick-start
    dominodatalab.com/project-owner-username: sameer-wadkar
    dominodatalab.com/starting-user-username: sameer-wadkar
spec:
  containers:
    - name: foo
      image: busybox:unstable
EOF
```

- Verify the pod has a new label
```shell
kubectl -n ${compute_namespace} describe pod busybox
```

- Delete the test pod and mutation
```shell
kubectl -n ${compute_namespace} delete pod busybox
kubectl -n ${platform_namespace} delete mutation label
```

### Finally Re-Install the Mutations you backedup in step 1
```shell
export platform_namespace=domino-platform
export compute_namespace=domino-compute
export backup_mutations_path=/tmp/mutations.yaml
kubectl apply -f $backup_mutations_path
##Verify they are applied
kubectl -n $platform_namespace get mutations
```