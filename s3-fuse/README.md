## Installation Instructions for s3-fuse

This document provides step-by-step instructions to install and configure s3-fuse, a FUSE-based file system that allows 
you to mount an S3 bucket as a local file system via the Domino External Data Volume feature.

## Prerequisites on AWS IAM Side 

We will create two IAM roles:

1. An IAM role with S3 read/write access to the target S3 bucket(s) that will be mounted via s3-fuse. 
   This role will be assumed by the Kubernetes service account that will be used by the s3-fuse CSI driver.
2. An IAM role for the Kubernetes service account that will be associated with the user workload pods that need access 
   to the mounted S3 bucket(s). 

The two strategies to access S3 buckets via s3-fuse are:
1. We allow the access to the S3 fuse drive (Using the IAM role created in step 1) from the user workload pods.
   The access is delegated to the S3 fuse driver via the Kubernetes service account IAM role created in step 1.
   
2. We allow access to the S3 bucket(s) via the S3 fuse mount via the role attached to the use r workload pods (IAM role created in step 2).
   In this case, the s3-fuse driver will use the user workload pod's IAM role to access the S3 bucket(s).

### Role 1: IAM Role for s3-fuse CSI Driver

Create an IAM role with the following trust relationship policy to allow the Kubernetes service account used by the

1. s3-fuse CSI driver to assume this role:

**Name** - `acme-s3-fuse-role`

**Trust Relationship Policy**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/<OIDC_PROVIDER>"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "<OIDC_PROVIDER>:sub": "system:serviceaccount:domino-platform:s3-csi-driver-sa"
        }
      }
    }
  ]
}
```

**Permissions Policy**:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::bucket-name-1",
                "arn:aws:s3:::bucket-name-2",
                "arn:aws:s3:::bucket-name-3"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::bucket-name-1/*",
                "arn:aws:s3:::bucket-name-2/*",
                "arn:aws:s3:::bucket-name-3/*"
            ]
        }
    ]
}
```


### Role 2: IAM Role for User Workload Pods

**Name** - `acme-read-bucket-role`

**Trust Relationship Policy**:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::946429944765:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/FF87726BFB448D3079A24017ECA6B6E5"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.us-west-2.amazonaws.com/id/FF87726BFB448D3079A24017ECA6B6E5:aud": "sts.amazonaws.com",
                    "oidc.eks.us-west-2.amazonaws.com/id/FF87726BFB448D3079A24017ECA6B6E5:sub": "system:serviceaccount:domino-compute:aws-test"
                }
            }
        }
    ]
}
```

**Permissions Policy**:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::bucket-name-1"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::bucket-name-1/*"
            ]
        }
    ]
}
```

## Installation Steps

The capability uses the AWS IRSA (IAM Roles for Service Accounts) feature to secure access to S3 buckets. 
The feature is enabled by The Mountpoint for Amazon S3 Container Storage Interface (CSI) Driver allows your Kubernetes 
applications to access Amazon S3 objects through a file system interface. Built on Mountpoint for Amazon S3, the 
Mountpoint CSI driver presents an Amazon S3 bucket as a storage volume accessible by containers in your Kubernetes 
cluster. The Mountpoint CSI driver implements the CSI specification for container orchestrators (CO) to manage storage 
volumes.

For Amazon EKS clusters, the Mountpoint for Amazon S3 CSI driver is also available as an EKS add-on to provide automatic
installation and management. This is the capability user in this guide.


### Create an K8s service account for user with the s3 CSI driver controller

```shell
kubectl -n domino-platform create serviceaccount s3-csi-driver-sa
```

### Attach the IAM role to the service account

```shell
export AWS_ACCOUNT_ID=<account-id>
export AWS_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/acme-s3-fuse-role
export platform_namespace=domino-platform
export s3_csi_driver_sa=s3-csi-driver-sa

kubectl -n ${platform_namespace} annotate serviceaccount ${s3_csi_driver_sa}\ 
         eks.amazonaws.com/role-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:role/${AWS_ROLE_ARN}
```

### Install the Mountpoint for Amazon S3 CSI Driver as an EKS add-on

```shell

export AWS_ACCOUNT_ID=<account-id>
export AWS_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/acme-s3-fuse-role
export platform_namespace=domino-platform
export s3_csi_driver_sa=s3-csi-driver-sa

helm repo add aws-mountpoint-s3-csi-driver https://awslabs.github.io/mountpoint-s3-csi-driver
helm repo update


helm upgrade --install aws-mountpoint-s3-csi-driver \
   --namespace ${platform_namespace} \
   --set node.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=${AWS_ROLE_ARN} \
   --set node.serviceAccount.name=${s3_csi_driver_sa} \
   aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver
```

### Create a PV/PVC for the S3 bucket mount to be used by the S3 fuse driver


You will need one for each S3 bucket you want to mount.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mountpoints3-pv
spec:
  capacity:
    storage: 10Gi # Ignored, required
  accessModes:
    - ReadWriteMany # Supported options: ReadWriteMany / ReadOnlyMany
  storageClassName: "" # Required for static provisioning
  claimRef: # To ensure no other PVCs can claim this PV
    namespace: domino-compute # Namespace is required even though it's in "default" namespace.
    name: mountpoints3-pvc # Name of your PVC
  mountOptions:
    - uid=12574
    - gid=12574
    - allow-other
    - allow-overwrite
    - allow-delete
    - region=us-west-2
  csi:
    driver: s3.csi.aws.com # Required
    volumeHandle: s3-csi-driver-volume
    volumeAttributes:
      bucketName: bucket-name-1
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mountpoints3-pvc
  namespace: domino-compute
  labels:
      dominodatalab.com/external-data-volume: Generic
spec:
  accessModes:
    - ReadWriteMany # Supported options: ReadWriteMany / ReadOnlyMany
  storageClassName: "" # Required for static provisioning
  resources:
    requests:
      storage: 10Gi # Ignored, required
  volumeName: mountpoints3-pv # Name of your PV

```

### Create a PV/PVC for the S3 bucket mount to be used by the user workload pods


You will need one for each S3 bucket you want to mount. Note the volume attribute `authenticationSource: pod` which 
tells the s3-fuse driver to use the IAM role attached to the user workload pod to access the S3 bucket.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mountpoints3-user-pv
spec:
  capacity:
    storage: 10Gi # Ignored, required
  accessModes:
    - ReadWriteMany # Supported options: ReadWriteMany / ReadOnlyMany
  storageClassName: "" # Required for static provisioning
  claimRef: # To ensure no other PVCs can claim this PV
    namespace: domino-compute # Namespace is required even though it's in "default" namespace.
    name: mountpoints3-user-pvc # Name of your PVC
  mountOptions:
    - uid=12574
    - gid=12574
    - allow-other
    - allow-overwrite
    - allow-delete
    - region=us-west-2
  csi:
    driver: s3.csi.aws.com # Required
    volumeHandle: s3-csi-driver-volume
    volumeAttributes:
      bucketName: bucket-name-1
      authenticationSource: pod
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mountpoints3-user-pvc
  namespace: domino-compute
  labels:
      dominodatalab.com/external-data-volume: Generic
spec:
  accessModes:
    - ReadWriteMany # Supported options: ReadWriteMany / ReadOnlyMany
  storageClassName: "" # Required for static provisioning
  resources:
    requests:
      storage: 10Gi # Ignored, required
  volumeName: mountpoints3-user-pv # Name of your PV

```

## Configure the Domino External Data Volume

Finally, configure the Domino External Data Volume to use the created PVCs.

## Using IRSA Domino Feature

If accessing the S3 bucket using IRSA Domino feature, install Domsed and the mutation included
in the file name [irsa-domino-mutation.yaml](./irsa-domino-mutation.yaml) to your cluster.

The other core thing to remember here is this approach needs the user specific SA to be annotated. We
will need to use the AWS provided IRSA configuration. The Domino approach to replicate IRSA via the
Domsed mutation will not work here. Ask the professional services team for help if needed.



## Things to remember

- In a single workload pod you cannot have two different S3-Fuse mounts using two different IAM roles or
mix driver based role and pod based role access. This is a limitation of the S3-Fuse CSI driver.

- Use S3-Fuse for browsing only. It is not recommended for high performance workloads.

- When processing data stored in S3 buckets for high performance workloads, use IRSA based access.

- When using pod based role access make sure you have the correct region for the S3 bucket specified in the PV mount options.

