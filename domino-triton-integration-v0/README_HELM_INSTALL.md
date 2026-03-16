# Helm installation

The Triton inference server and the proxy are installed in their own namespace. This allows us to manage
multiple installations independently:

1. Dev  
2. Test
3. Production


## Prerequisites

This installation relies on S3 bucket exposed as a mount inside the Triton server pod.

The same location is mounted as an EDV (External Data Volume) inside Domino Workspaces to allow
deployment of models from Domino to Triton.

As a pre-requistite, please make sure you have the following:
1. An S3 bucket created to store the models
2. A AWS role and policy with full access to the S3 bucket


### Installation for the S3 Fuse CSI Driver

The capability uses the AWS IRSA (IAM Roles for Service Accounts) feature to secure access to S3 buckets. 
The feature is enabled by The Mountpoint for Amazon S3 Container Storage Interface (CSI) Driver allows your Kubernetes 
applications to access Amazon S3 objects through a file system interface. Built on Mountpoint for Amazon S3, the 
Mountpoint CSI driver presents an Amazon S3 bucket as a storage volume accessible by containers in your Kubernetes 
cluster. The Mountpoint CSI driver implements the CSI specification for container orchestrators (CO) to manage storage 
volumes.

For Amazon EKS clusters, the Mountpoint for Amazon S3 CSI driver is also available as an EKS add-on to provide automatic
installation and management. This is the capability used in this guide.


### Create the IAM role and policy and attach the policy to the role

IAM Role - `acme-s3-fuse-role`

IAM Policy - `acme-s3-fuse-policy`
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
                "arn:aws:s3:::<BUCKET_NAME>"
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
                "arn:aws:s3:::<BUCKET_NAME>/*"
            ]
        }
    ]
}
```

IAM Role Trust Relationship 
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/<OIDC_PROVIDER>"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "oidc.eks.us-west-2.amazonaws.com/id/<OIDC_PROVIDER>:aud": "sts.amazonaws.com",
                    "oidc.eks.us-west-2.amazonaws.com/id/<OIDC_PROVIDER>:sub": "system:serviceaccount:domino-platform:s3-csi-driver-sa"
                }
            }
        }
    ]
}
```

### Install the S3 Fuse CSI Driver using Helm

```bash

export AWS_ACCOUNT_ID=<account-id>
export AWS_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/acme-s3-fuse-role
export platform_namespace=domino-platform
export s3_csi_driver_sa=s3-csi-driver-sa

helm repo add aws-mountpoint-s3-csi-driver https://awslabs.github.io/mountpoint-s3-csi-driver
helm repo update

helm upgrade --install aws-mountpoint-s3-csi-driver \
  --namespace ${platform_namespace} \
  --set node.serviceAccount.create=true \
  --set node.serviceAccount.name=${s3_csi_driver_sa} \
  --set node.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=${AWS_ROLE_ARN} \
  aws-mountpoint-s3-csi-driver/aws-mountpoint-s3-csi-driver
```

## Images needed for the installation

This installation needs the following images to be available in your container registry:

1. Domino Admin image - `quay.io/domino/domino-triton-admin:v1.0.0"`
2. Domino Triton Rest Proxy image - `quay.io/domino/domino-triton-rest-proxy:v1.0.0`
3. Domino Triton gRPC Proxy image - `quay.io/domino/domino-triton-grpc-proxy:v1.0.0`
4. NVIDIA Triton Inference Server image - `nvcr.io/nvidia/tritonserver:23.03-py3`


## Create Namespace

Create a namespace for the Domino Inference installation. In this example, we create a `domino-inference-dev` namespace

Do not forget to label the namespace for Domino Compute usage. This is required to make calls to Nucleus for
authz purposes. Nucleus frontend will reject requests from namespace without this label.

The other label `domino-triton=true` marks this namespace as hosting a Domino Triton installation.

```bash
export ns=domino-inference-dev
kubectl create namespace ${ns}

## Label Namespace for Domino Compute
kubectl label namespace ${ns} domino-compute=true
kubectl label namespace ${ns} domino-triton=true
```
## Helm Delete

```bash
export ns=domino-inference-dev
helm delete  -n $ns domino-triton

```

## Helm Install

```bash
export ns=domino-inference-dev
helm install  domino-triton  helm/domino-triton/ -n $ns -f helm/domino-triton/values.yaml
```


## Helm Upgrade

```bash
export ns=domino-inference-dev
helm upgrade  domino-triton  helm/domino-triton/ -n $ns -f helm/domino-triton/values.yaml
```

## Testing the installation

### First add the following claims for the admin user in keycloak
```json
[{"namespace":"domino-inference-dev","service":"triton-inference-server","role":{"admin":true}} ]
```


From inside a Domino workload run the following after copying the following files into the `src/` folder:
1. `mm_client.py`
2. `mm_client_api.py`
3. `multimodal_pb2.py`
4. `multimodal_pb2_grpc.py`

Also copy the `samples` folder into the working directory. And create and empty `results/` folder.

Then run the following commands:

```bash
pip install --no-cache-dir \
    "grpcio==1.67.1" \
    "grpcio-tools==1.67.1" \
    "protobuf==5.28.3" \
    "tritonclient[grpc]==2.61.0" \
    "numpy==2.2.6" \
    "requests==2.32.4" \
    "opencv-python-headless==4.10.0.84"

export MM_ADDR=triton-inference-server-proxy.domino-inference-dev.svc.cluster.local:50051  # if using from inside the cluster
export VIDEO_PATH=./samples/video.avi
export MODEL_NAME=yolov8n
export MODEL_VERSION=1
export INPUT_NAME=images
export OUTPUT_NAMES=output0
export IMG_SIZE=640

export PARSE_NUMPY=1
python3 src/mm_client.py
```

This will produce two files inside the `results/` folder:
1. `results/annotated.mp4` : the input video with bounding boxes drawn around detected objects
2. `results/frame_counts.json1` : a json file with the count of detected objects per frame

The original video is located the `samples/video.avi` file. You can download the mp4 version by clicking the image below:
[![Video Demo](results/original_screenshot.png)](https://github.com/domino-field/grpc-based-triton-integration/releases/download/do-not-use/original_video.mp4)

Download the annotated video by clicking the image below:
[![Video Demo](results/screenshot.png)](https://github.com/domino-field/grpc-based-triton-integration/releases/download/do-not-use/annotated.mp4)

The annotated json file can be downloaded file is located [here](./results/frame_counts.jsonl):

The full notebooks used to create the client and test the installation are located in the `notebooks/` folder.