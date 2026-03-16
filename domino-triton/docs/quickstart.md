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

## Other Docs

- [Helm Installation Guide](helm-install.md) - Deploy Triton to Kubernetes
- [Model Configuration Guide](model-configuration.md) - Configure models for deployment
