# Quickstart Guide

This guide walks you through setting up the Domino Triton Inference Server from scratch.

## Prerequisites

- Access to a Domino Data Lab environment
- AWS credentials configured for S3 access
- kubectl access to the target Kubernetes cluster

---

## Step 1: Create External Data Volume (EDV)

Create an EDV to store the Triton model repository:

1. Log into your Domino environment as an admin
2. Navigate to **Admin** > **Data** > **External Data Volumes**
3. Click **Create External Data Volume**
4. Configure the EDV:
   - **Name**: `{triton-namespace}-triton-repo-pvc` (e.g., `domino-inference-dev-triton-repo-pvc`)
   - **Volume Type**: Select your S3-backed volume type
   - **Path**: Point to your S3 bucket path for the triton-repo
5. Under **Access**, select **Everyone** to give all users access
6. Click **Create**

> **Note**: The EDV name must match the PVC name expected by the Helm chart. The naming convention is `{namespace}-triton-repo-pvc`.

---

## Step 2: Add EDV to Your Workspace

Anyone who wants to deploy models to the Triton Inference Server must include the EDV in their workspace:

1. Open your Domino project
2. Navigate to **Workspaces** > **New Workspace**
3. In the workspace configuration, go to **Data**
4. Under **External Data Volumes**, select the EDV created in Step 1 (e.g., `domino-inference-dev-triton-repo-pvc`)
5. The EDV will be mounted at `/domino/edv/{edv-name}` in your workspace

> **Tip**: You can also add the EDV to an existing workspace by editing its configuration.

---

## Step 3: Create Domino File System Project

1. Log into your Domino environment
2. Navigate to **Projects** > **New Project**
3. Select **File System** as the project type
4. Name your project (e.g., `triton-inference-server`)
5. Click **Create Project**

---

## Step 4: Clone the Repository and Set Up Project Structure

Open a terminal in your Domino workspace and run the following commands:

```bash
# Clone the repository in the home folder
cd ~
git clone https://github.com/dominodatalab/domino-field-solutions-installations.git

# Copy required subfolders to the project root (/mnt)
cp -r domino-field-solutions-installations/domino-triton/app-src /mnt/
cp -r domino-field-solutions-installations/domino-triton/docker /mnt/
cp -r domino-field-solutions-installations/domino-triton/docs /mnt/
cp -r domino-field-solutions-installations/domino-triton/samples /mnt/
cp -r domino-field-solutions-installations/domino-triton/scripts /mnt/
cp -r domino-field-solutions-installations/domino-triton/notebooks /mnt/
cp -r domino-field-solutions-installations/domino-triton/triton-repo-reference /mnt/

# Clean up the cloned repository
rm -rf ~/domino-field-solutions-installations
```

Your project structure should now look like:

```
/mnt/
├── app-src/           # Application source code
├── docs/              # Documentation
├── scripts/           # Utility scripts
└── notebooks/         # Jupyter notebooks
```

---

## Step 5: Configure the Environment

Install the required Python client dependencies:

```bash
cd /mnt/
pip install -r ./docker/requirements-client.txt
```

This installs the Triton client libraries and other dependencies needed to interact with the inference server.

---

## Step 6: Download and Deploy Test Models

Run the download notebook to download sample models and copy them to the EDV:

1. Open Jupyter in your workspace
2. Navigate to `notebooks/download_models.ipynb`
3. Run all cells in order:
   - **Setup**: Configures paths and verifies scripts exist
   - **Download All Models**: Downloads YOLOv8n, BERT, Whisper, and SmolLM models
   - **Install Packages**: Installs model-specific dependencies (e.g., humanize for Whisper)
   - **Copy to EDV**: Update the `NAMESPACE` variable to match your target namespace, then run to copy models to the EDV

> **Note**: The copy-to-EDV cell uses S3-safe copying that handles S3 fuse limitations. It will overwrite existing files.

---

## Step 7: Run Inference Demo

Test the deployed models using the inference demo notebook:

1. Open `notebooks/triton_inference_demo.ipynb`
2. Update the connection settings at the top:
   - `TRITON_REST_URL`: HTTP proxy URL (e.g., `http://triton-proxy.domino-inference-dev.svc.cluster.local:8080`)
   - `TRITON_GRPC_URL`: gRPC proxy URL (e.g., `triton-proxy.domino-inference-dev.svc.cluster.local:50051`)
3. Run the cells to:
   - Check server health and list available models
   - Load models on-demand
   - Run inference on each model type
   - View results and performance metrics

---

## Step 8: Model-Specific Notebooks (Optional)

For deeper exploration of individual models, run the model-specific notebooks:

| Notebook | Model | Description |
|----------|-------|-------------|
| `notebooks/yolov8n_inference.ipynb` | YOLOv8n | Object detection on images and video |
| `notebooks/bert_inference.ipynb` | BERT | Text classification and sentiment analysis |
| `notebooks/whisper_inference.ipynb` | Whisper | Audio transcription with timestamps |
| `notebooks/llm_inference.ipynb` | SmolLM-135M | Text generation and chat |

Each notebook includes:
- Model loading and health checks
- Sample inference with visualization
- Performance benchmarking
- Error handling examples

---

## Step 9: Run Benchmarks (Optional)

Run comprehensive benchmarks on all models using the sequential test script:

```bash
export ENV=dev
export NAMESPACE=domino-inference-${ENV}
export TRITON_GRPC_URL="triton-inference-server-proxy.${NAMESPACE}.svc.cluster.local:50051"
export TRITON_REST_URL="http://triton-inference-server-proxy.${NAMESPACE}.svc.cluster.local:8080"
# Run all models with benchmarks
python scripts/testing/test_models_sequential.py

# If models are already on EDV (skip copy step)
python scripts/testing/test_models_sequential.py --skip-copy

# Test specific models only
python scripts/testing/test_models_sequential.py --models bert-base-uncased yolov8n

# List available models
python scripts/testing/test_models_sequential.py --list
```

The script will:
- Load each model sequentially (to avoid GPU memory issues)
- Run inference tests
- Execute benchmarks and record results
- Unload models after testing

Results are saved to the `results/` directory.

---

## Step 10: Save Benchmarks to Domino Dataset (Optional)

Copy benchmark results to a Domino dataset for persistence and sharing:

```bash
# Create benchmarks folder in Domino dataset
mkdir -p /domino/datasets/local/${DOMINO_PROJECT_NAME}/benchmarks

# Copy benchmark results for all four models
cp -r results/yolov8/benchmark/*.md /domino/datasets/local/${DOMINO_PROJECT_NAME}/benchmarks/
cp -r results/bert/benchmark/*.md /domino/datasets/local/${DOMINO_PROJECT_NAME}/benchmarks/
cp -r results/whisper/benchmark/*.md /domino/datasets/local/${DOMINO_PROJECT_NAME}/benchmarks/
cp -r results/llm/benchmark/*.md /domino/datasets/local/${DOMINO_PROJECT_NAME}/benchmarks/
```

> **Note**: `DOMINO_PROJECT_NAME` is automatically set by Domino in your workspace environment.

---

## Step 11: Start the Triton Dashboard

Launch the Triton Admin Dashboard as a Domino App to monitor and manage your inference server.

### Pre-requisites

1. #### Install mutation
```shell
kubectl -n domino-platform apply -f app-src/mutation.yaml
```
2. #### Turn on Deep Linking

- Set the feature flag `SecureIdentityPropagationToAppsEnabled` to `true`
- Set the central config `ShortLived.iFrameRequired` to `false` (Restart nucleus services)


### Create App Environment

Create a Environment to start the app below. This is the Dockerfile for it (Based it on the Domino Standard Env)

```Dockerfile
RUN pip install --no-cache-dir \                                                                                                                                                                                                 
      fastapi>=0.104.0 \                                                                                                                                                                                                           
      "uvicorn[standard]>=0.24.0" \                                                                                                                                                                                                
      httpx>=0.25.0 \                                                                                                                                                                                                              
      jinja2>=3.1.0 \                                                                                                                                                                                                              
      pydantic>=2.5.0 \                                                                                                                                                                                                            
      python-multipart>=0.0.6 \                                                                                                                                                                                                    
      "numpy>=1.24.0,<2" \                                                                                                                                                                                                         
      "tritonclient[all]>=2.40.0" \                                                                                                                                                                                                
      transformers>=4.30.0 \                                                                                                                                                                                                       
      opencv-python-headless>=4.8.0 \                                                                                                                                                                                              
      librosa>=0.10.0 \                                                                                                                                                                                                            
      soundfile>=0.12.0 \                                                                                                                                                                                                          
      grpcio==1.67.1 \                                                                                                                                                                                                             
      grpcio-tools==1.67.1 \                                                                                                                                                                                                       
      protobuf==5.28.3 \                                                                                                                                                                                                           
      requests>=2.28.0 \                                                                                                                                                                                                           
      torch>=2.0.0 \                                                                                                                                                                                                               
      onnx>=1.14.0 \                                                                                                                                                                                                               
      onnxruntime>=1.15.0 \                                                                                                                                                                                                        
      onnxscript>=0.1.0 \                                                                                                                                                                                                          
      ultralytics>=8.0.0 \                                                                                                                                                                                                         
      "optimum[onnxruntime]>=1.12.0" \                                                                                                                                                                                             
      huggingface_hub>=0.16.0         
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \                                                                                                                                                               
      ffmpeg \                                                                                                                                                                                                                     
      && rm -rf /var/lib/apt/lists/* 
```


<!-- TODO: Add environment setup instructions -->

### Deploy App

1. Navigate to your Domino project
2. Go to **App** > **New App** (or **Publish** > **App**)
3. Configure the app:
   - **Title**: `Triton Dashboard`
   - **Custom URL**: `triton-dashboard`
   - **Script**: `app-src/start_dfs_based.sh`
4. Click **Publish**

The dashboard provides:
- Model status and health monitoring
- Load/unload model controls
- Inference testing interface
- Benchmark results viewer
- Real-time metrics

Once published, access the dashboard at your custom URL (e.g., `https://<domino-host>/triton-dashboard`).

---

## Other Docs

- [Helm Installation Guide](helm-install.md) - Deploy Triton to Kubernetes
- [Model Configuration Guide](model-configuration.md) - Configure models for deployment
