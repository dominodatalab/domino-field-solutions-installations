## Notebooks (User Enablement)

This document defines how the user can use a gRPC-based Triton Inference Server deployment inside Domino.

It covers:

1. Download a model from an external source and conver to ONNX format. [download_yolo.ipynb](notebooks/download_yolo.ipynb) demonstrates how to:
   - Download a YOLOv8 model from Ultralytics Hub
   - Convert the model to ONNX format
   - Validate the ONNX model using ONNX Runtime
2. Model registration in Domino. Use the notebook [model_registration_deploy.ipynb](notebooks/model_registration_deploy.ipynb) to:
   - Create an experiment and register the model (with the converted ONNX file)
   - Download the model from the model registry and write to an EDV (mounted into the workspace). EDV creation is outside the scope of this document.
   - EDV points to an S3 bucket location which is also mounted into the Triton Inference Server deployment.
3. Model testing using Triton client libraries from a Domino workspace - Use a sample video file, extract frames, and send them to the Triton Inference Server for inference using gRPC.
   Use the notebook [model_testing.ipynb](notebooks/model_testing.ipynb) to:
4. Administration of the Triton Inference Server deployment from a Domino workspace . Use the notebook [admin.ipynb](notebooks/admin.ipynb) to perform the following admin tasks:
   - Scale up/down the number of replicas for Triton Inference Server (Scale to zero nightly or when not in use to save costs)
   - List models deployed in Triton Inference Server
   - Download logs from the Triton Inference Server pods for debugging purposes