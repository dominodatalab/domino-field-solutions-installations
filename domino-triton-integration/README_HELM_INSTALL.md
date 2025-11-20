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
3. [Install](https://github.com/dominodatalab/domino-field-solutions-installations/blob/main/s3-fuse/README.md) S3-fuse on the Kubernetes cluster nodes


## Create Namespace

Create a namespace for the Domino Inference installation. In this example, we create a `domino-inference-dev` namespace

Do not forget to label the namespace for Domino Compute usage. This is required to make calls to Nucleus for
authz purposes. Nucleus frontend will reject requests from namespace without this label.

```bash

kubectl create namespace domino-inference-dev

## Label Namespace for Domino Compute
kubectl label namespace domino-inference-dev domino-compute=true
```
## Helm Delete

```bash
export ns=domino-inference-dev
helm delete  -n $ns grpc-triton-domino

```

## Helm Install

```bash
export ns=domino-inference-dev
helm install  grpc-triton-domino  helm/grpc-triton-domino/ -n $ns -f helm/grpc-triton-domino/values.yaml
```


## Helm Upgrade

```bash
export ns=domino-inference-dev
helm upgrade  grpc-triton-domino  helm/grpc-triton-domino/ -n $ns -f helm/grpc-triton-domino/values.yaml
```

## Testing the installation

This assumes that your have created and mounted an EDV (created in this installation) into a workpspace and copied
the  folder structure `yolov8n` from the `./models` folder into the EDV:
The EDV should have the following structure:

```bash
-yolov8n
  -1    
    -model.onnx
  -config.pbtxt
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

export MM_ADDR=grpc-domino-triton-proxy.domino-inference-dev.svc.cluster.local:50051  # if using from inside the cluster
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