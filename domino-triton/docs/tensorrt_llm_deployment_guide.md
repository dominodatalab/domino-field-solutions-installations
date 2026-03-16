# TensorRT-LLM Kubernetes Deployment Guide

A guide to deploying TensorRT-LLM models on Kubernetes, addressing common challenges with GPU architecture compatibility, node scheduling, and storage.

---

## GPU Architecture Compatibility

### The Problem

**TensorRT engines are NOT portable between GPU architectures.** A TensorRT engine built on one GPU type will fail on a different GPU type with an error like:

```
[TensorRT-LLM][ERROR] The engine was built with different GPU architecture.
[TensorRT-LLM][ERROR] Engine was built for SM 8.9, but running on SM 7.0
```

### GPU Compute Capabilities

| GPU | Compute Capability | AWS Instance Types |
|-----|-------------------|-------------------|
| V100 | 7.0 | p3.2xlarge, p3.8xlarge, p3.16xlarge |
| T4 | 7.5 | g4dn.xlarge, g4dn.2xlarge, g4dn.4xlarge |
| A10G | 8.6 | g5.xlarge, g5.2xlarge, g5.4xlarge |
| L4 | 8.9 | g6.xlarge, g6.2xlarge, g6.4xlarge, g6f.xlarge, g6e.xlarge |
| A100 | 8.0 | p4d.24xlarge, p4de.24xlarge |
| H100 | 9.0 | p5.48xlarge |

### Solution

1. **Build engines on the target GPU**: Use a workspace with the same GPU type as your deployment target
2. **Configure deployment to match**: Ensure Kubernetes schedules pods only on matching GPU nodes

---

## Physical GPU vs vGPU Compatibility

### The Problem

Even when compute capability matches, TensorRT engines may fail with:

```
[TensorRT-LLM][ERROR] Assertion failed: Failed to deserialize cuda engine.
```

This occurs when the engine was built on a **physical GPU** but runs on a **vGPU** (virtual GPU partition), or vice versa.

### Understanding vGPUs

Cloud providers offer virtualized GPU partitions for cost efficiency:

| Physical GPU | vGPU Partitions | Identifier |
|--------------|-----------------|------------|
| L4 (24GB) | L4-1Q (6GB), L4-2Q (12GB), L4-3Q (18GB), L4-4Q (24GB) | Suffix `-nQ` |
| A10G (24GB) | A10G-1Q, A10G-2Q, A10G-4Q | Suffix `-nQ` |
| A100 (40/80GB) | Various MIG partitions | MIG profiles |

**Example:** `NVIDIA L4-3Q` is a vGPU with 3/4 of an L4's memory (~18GB usable).

### Why Engines Are Incompatible

TensorRT engines encode:
- GPU architecture (SM version)
- Memory layout optimizations
- Kernel configurations specific to the GPU variant

A physical L4 and L4-3Q vGPU have the same compute capability (8.9) but different:
- Memory bandwidth characteristics
- Available memory
- Potentially different driver behaviors

### Compatibility Requirements

For TensorRT-LLM engines to load successfully:

| Requirement | Must Match |
|-------------|------------|
| **Compute Capability** | Exact (e.g., 8.9 = 8.9) |
| **GPU Type** | Physical ↔ Physical, vGPU ↔ vGPU |
| **vGPU Partition** | Same partition size (L4-3Q ↔ L4-3Q) |
| **TensorRT Version** | Major.minor (e.g., 10.3.x) |
| **CUDA Driver** | Minimum version met |

### How to Verify GPU Type

```bash
# On workspace (where you build engines)
nvidia-smi --query-gpu=name,compute_cap --format=csv

# On Kubernetes node (where engines run)
kubectl exec <pod> -- nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**Physical GPU output:**
```
name, compute_cap
NVIDIA L4, 8.9
```

**vGPU output:**
```
name, compute_cap
NVIDIA L4-3Q, 8.9
```

### Solutions

**Option 1: Match Build and Runtime Environments (Recommended)**

Build TensorRT engines in an environment identical to the deployment target:
- If deploying to vGPU nodes, use a workspace with the same vGPU partition
- If deploying to physical GPU nodes, use a workspace with physical GPU

**Option 2: Use Python Backend Instead**

Python backend models (HuggingFace Transformers) work on any GPU:

```
tinyllama-python      # Works on any GPU (slower)
tinyllama-trtllm      # Only works on matching GPU (10-20x faster)
```

Trade-off: Python backend is 10-20x slower but fully portable.

**Option 3: Ensure AMI/Driver Consistency**

The workspace AMI and EKS node AMI should have:
- Same CUDA driver major version
- Same TensorRT version
- Same vGPU driver configuration

### Domino-Specific Considerations

In Domino environments:

1. **Workspace GPU**: Determined by hardware tier selection
2. **EKS Node GPU**: Determined by node group instance types and AMI
3. **vGPU vs Physical**: Check with platform admin - some clusters use vGPU for cost efficiency

**To ensure compatibility:**
```bash
# 1. Check workspace GPU
nvidia-smi

# 2. Check what GPU the Triton pod gets
kubectl exec <triton-pod> -n <namespace> -- nvidia-smi

# 3. Compare - they must show identical GPU names
```

---

## Node Selection Strategies

### Mixed GPU Node Pools

Many Kubernetes clusters have node pools with multiple instance types. A "default-gpu" pool might include both V100 and L4 nodes:

```yaml
# This is too broad - may schedule on wrong GPU type
nodeSelector:
  dominodatalab.com/node-pool: default-gpu
```

### Strategy 1: Single Instance Type

Use when you need one specific instance type:

```yaml
# values.yaml
env:
  instance_type: g6f.xlarge  # Single L4 instance type
```

```yaml
# deployment.yaml
nodeSelector:
  node.kubernetes.io/instance-type: {{ .Values.env.instance_type }}
```

### Strategy 2: Multiple Instance Types (Recommended)

Use when any of several instance types with the same GPU is acceptable:

```yaml
# values.yaml
env:
  instance_types:
    - g6.xlarge
    - g6.2xlarge
    - g6.4xlarge
    - g6f.xlarge
    - g6f.2xlarge
    - g6f.4xlarge
    - g6e.xlarge
    - g6e.2xlarge
```

```yaml
# deployment.yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
          {{- range .Values.env.instance_types }}
          - {{ . }}
          {{- end }}
```

### Strategy 3: Specific Node (Testing/Debugging)

Pin to a specific node by name:

```yaml
# values.yaml
env:
  node_name: ip-10-0-1-123.ec2.internal
```

```yaml
# deployment.yaml
nodeName: {{ .Values.env.node_name }}
```

### Verifying Node GPU Type

```bash
# List nodes with GPU info
kubectl get nodes -o custom-columns=\
NAME:.metadata.name,\
INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type,\
GPU:.metadata.labels.nvidia\\.com/gpu\\.product

# Check specific node
kubectl describe node <node-name> | grep -E "(instance-type|gpu)"
```

---

## PersistentVolume Zone Affinity

### The Problem

EBS volumes are zone-specific. If your PersistentVolume is in `us-west-2a` but the only available L4 nodes are in `us-west-2b`, scheduling fails:

```
0/10 nodes are available: 3 node(s) didn't match Pod's node affinity/selector,
7 node(s) had volume node affinity conflict.
```

### Diagnosing the Issue

```bash
# Check PV zone
kubectl get pv <pv-name> -o jsonpath='{.spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[?(@.key=="topology.kubernetes.io/zone")].values[0]}'

# Check node zones
kubectl get nodes -l node.kubernetes.io/instance-type=g6f.xlarge \
  -o custom-columns=NAME:.metadata.name,ZONE:.metadata.labels.topology\\.kubernetes\\.io/zone
```

### Solution Options

**Option A: Delete and Recreate PVC (Data Loss)**

If data can be recreated:
```bash
kubectl delete pvc <pvc-name> -n <namespace>
# Redeploy - new PVC will be created in correct zone
helm upgrade ...
```

**Option B: Cross-Zone Volume Access (Performance Impact)**

Use a storage class that supports multi-zone access (e.g., EFS instead of EBS).

**Option C: Zone-Aware Scheduling**

Add zone affinity to match the existing volume:
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: topology.kubernetes.io/zone
          operator: In
          values:
          - us-west-2a  # Match PV zone
```

---

## Node Taints and Tolerations

### The Problem

GPU nodes often have taints that prevent non-GPU workloads from scheduling. Additionally, newly provisioned nodes have initialization taints:

```
0/10 nodes are available: 1 node(s) had untolerated taint {node.kubernetes.io/unschedulable: },
1 node(s) had untolerated taint {node.cloudprovider.kubernetes.io/uninitialized: true}
```

### Required Tolerations

Add these tolerations to your deployment:

```yaml
tolerations:
  # GPU node taint
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

  # Node being cordoned/drained
  - key: "node.kubernetes.io/unschedulable"
    operator: Exists
    effect: NoSchedule

  # Node initializing (cloud provider)
  - key: "node.cloudprovider.kubernetes.io/uninitialized"
    operator: Exists
    effect: NoSchedule

  # Node not ready (brief tolerance)
  - effect: NoExecute
    key: node.kubernetes.io/not-ready
    operator: Exists
    tolerationSeconds: 300

  # Node unreachable (brief tolerance)
  - effect: NoExecute
    key: node.kubernetes.io/unreachable
    operator: Exists
    tolerationSeconds: 300
```

### Checking Node Taints

```bash
kubectl describe node <node-name> | grep -A5 Taints
```

---

## Cluster Autoscaler Considerations

### The Problem

When no nodes match your scheduling requirements, the cluster autoscaler provisions new nodes. This causes delays:

1. Autoscaler detects unschedulable pod (~30s)
2. New node provisioned (~2-5 min for GPU instances)
3. Node initialized and joins cluster (~1-2 min)
4. Node passes health checks (~30s)
5. Pod scheduled and starts (~1-2 min for large images)

**Total: 5-10 minutes for first deployment**

### Optimization Strategies

1. **Keep minimum nodes running**: Configure node pool with `minSize: 1` for critical GPU types

2. **Use spot instances with fallback**: Faster provisioning, lower cost
   ```yaml
   # values.yaml
   env:
     instance_types:
       - g6.xlarge      # Preferred (spot)
       - g6f.xlarge     # Fallback
       - g6e.xlarge     # Fallback
   ```

3. **Pre-pull images**: Use DaemonSets to pre-pull large Triton images on GPU nodes

4. **Monitor autoscaler**:
   ```bash
   kubectl logs -n kube-system deployment/cluster-autoscaler | grep -E "(scale|unschedulable)"
   ```

---

## Deployment Checklist

Before deploying TensorRT-LLM:

- [ ] **Engine built on target GPU**: Check compute capability matches
- [ ] **Physical/vGPU matches**: Engine built on same GPU type (physical L4 ↔ physical L4, L4-3Q ↔ L4-3Q)
- [ ] **TensorRT versions match**: Build environment and runtime have same TensorRT version
- [ ] **Node selection configured**: Use `instance_types` list for specific GPU architecture
- [ ] **Tolerations added**: Include GPU and initialization taints
- [ ] **Storage zone verified**: PVC zone matches available node zones
- [ ] **Image accessible**: Verify image pull secrets if using private registry
- [ ] **Sufficient GPU memory**: vGPU partitions have less memory than physical GPUs

### Quick Verification Commands

```bash
# Check pod scheduling status
kubectl get pods -n <namespace> -o wide

# Check events for scheduling issues
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | tail -20

# Describe pod for detailed errors
kubectl describe pod <pod-name> -n <namespace> | tail -50
```

---

## Troubleshooting Matrix

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Engine was built for SM X.X, but running on SM Y.Y` | GPU architecture mismatch | Rebuild engine on target GPU or use instance_type selector |
| `Failed to deserialize cuda engine` | Physical/vGPU mismatch or TensorRT version mismatch | Rebuild engine on matching GPU type (physical↔physical, vGPU↔vGPU) |
| `didn't match Pod's node affinity/selector` | No nodes match selector | Check instance type exists, verify node labels |
| `had volume node affinity conflict` | PV in different zone | Delete/recreate PVC or add zone affinity |
| `had untolerated taint` | Missing tolerations | Add required tolerations to deployment |
| `Insufficient nvidia.com/gpu` | No GPU capacity | Wait for autoscaler or add nodes |
| `CUDA out of memory` | vGPU has less memory than expected | Use smaller model or larger vGPU partition |

---

## Example: Complete Helm Values

```yaml
env:
  namespace: domino-inference-dev
  name: triton-inference-server
  type: domino-triton-inference

  # GPU node selection - all L4 instance types
  instance_types:
    - g6.xlarge
    - g6.2xlarge
    - g6.4xlarge
    - g6f.xlarge
    - g6f.2xlarge
    - g6f.4xlarge
    - g6e.xlarge
    - g6e.2xlarge

  # Resources
  memory: 8Gi
  cores: 1
  gpu: 1

triton_inference_server:
  image: "nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3"
  replicas: 1

persistence:
  s3:
    bucket: my-triton-s3-bucket
    region: "us-west-2"
  cache:
    enabled: true
    storageClass: "dominodisk"
    size: "50Gi"
```

This configuration ensures the TensorRT-LLM engine (built on L4 GPU) runs only on L4 nodes.

---

## Future Improvements

### Build-on-Startup with Caching (Planned)

To eliminate GPU compatibility issues entirely, implement automatic engine building at pod startup.

#### Concept

```
Pod Startup
    │
    ▼
┌─────────────────────────────────────┐
│  Init Container: Engine Builder     │
│  1. Detect GPU (L4, L4-3Q, V100...) │
│  2. Check cache for compatible engine│
│  3. If missing, build from checkpoint│
│  4. Store in cache with GPU fingerprint│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Triton Container                   │
│  Load pre-built engine from cache   │
└─────────────────────────────────────┘
```

#### Storage Strategy

| Location | Contents | Size |
|----------|----------|------|
| S3 (model repo) | Checkpoints + tokenizer | ~2GB per model |
| EBS Cache | Built TensorRT engines | ~3GB per model per GPU type |

#### Cache Key (GPU Fingerprint)

Engines are cached using a fingerprint that captures all compatibility factors:

```
{model_name}-{compute_capability}-{trt_version}-{gpu_name_hash}
```

Example: `tinyllama-1.1b-sm89-trt10.3-l4q3`

This ensures engines are only reused when all compatibility requirements match.

#### Trade-offs

| Aspect | Pro | Con |
|--------|-----|-----|
| **Compatibility** | Always works - built on exact runtime GPU | - |
| **Startup time** | Cached builds are instant | First build: 5-15 min |
| **Storage** | Checkpoints smaller than engines | Need cache storage per GPU type |
| **Complexity** | Simpler deployment (no pre-building) | Build logic in container |

#### Implementation Options

**Option A: Init Container (Simpler)**
- Runs before Triton starts
- Blocks until engine is ready
- Pod not ready until all engines built

**Option B: Sidecar + Lazy Loading (Better UX)**
- Builds engines in background sidecar
- Triton starts immediately (Python models work)
- TRT-LLM models become available when build completes
- Better for mixed deployments (Python + TensorRT models)

#### Manifest-Driven Builds (Recommended)

Instead of hardcoding models in Helm, use a **manifest file in S3** that specifies what to build. This supports both base models and user fine-tuned models.

**S3 Structure:**

```
s3://triton-bucket/
├── models/                         # Triton model configs (config.pbtxt)
├── checkpoints/                    # HuggingFace checkpoints
│   ├── tinyllama-1.1b/            # Base model
│   ├── my-finetuned-llama/        # User fine-tuned model
│   └── customer-sentiment-bert/    # Another fine-tuned model
├── engines/                        # Built engines (cache)
│   ├── tinyllama-trtllm-sm89-trt10.3/
│   └── my-finetuned-llama-sm89-trt10.3/
└── manifests/                      # Per-model build manifests
    ├── tinyllama-trtllm.yaml
    ├── my-finetuned-llama.yaml
    └── customer-sentiment.yaml
```

**Per-Model Manifest Files:**

Each model has its own manifest file. The filename becomes the model name in Triton.

`manifests/tinyllama-trtllm.yaml`:
```yaml
# TensorRT-LLM Build Manifest for tinyllama-trtllm
checkpoint: checkpoints/tinyllama-1.1b
architecture: llama
dtype: float16
max_batch_size: 4
max_input_len: 512
max_seq_len: 1024
```

`manifests/my-finetuned-llama.yaml`:
```yaml
# Fine-tuned Llama model
checkpoint: checkpoints/my-finetuned-llama
architecture: llama
dtype: float16
max_batch_size: 2
max_input_len: 1024
max_seq_len: 2048
# Optional: custom tokenizer if different from checkpoint
# tokenizer: checkpoints/my-finetuned-llama
```

`manifests/customer-sentiment.yaml`:
```yaml
# BERT-based sentiment model
checkpoint: checkpoints/customer-sentiment-bert
architecture: bert
dtype: float16
max_batch_size: 8
```

**Benefits of Per-Model Manifests:**

| Benefit | Description |
|---------|-------------|
| **No conflicts** | Multiple users can add models simultaneously |
| **Self-contained** | Each model's config is independent |
| **Easy cleanup** | Delete model by removing its manifest file |
| **Clear naming** | Filename = model name in Triton |
| **Simpler parsing** | No need to merge/parse large YAML arrays |

**Triton Model Directory Structure:**

TensorRT-LLM models require this structure:

```
/triton-repo/models/
└── tinyllama-trtllm/
    ├── config.pbtxt              # Points to tensorrt_llm/ subdirectory
    ├── tokenizer.json            # Tokenizer files
    ├── tokenizer_config.json
    └── tensorrt_llm/             # Engine directory (referenced in config.pbtxt)
        ├── config.json           # TRT-LLM config
        └── rank0.engine          # Actual TensorRT engine (~2-3GB)
```

The `config.pbtxt` references the engine location:

```protobuf
parameters {
  key: "gpt_model_path"
  value: { string_value: "/triton-repo/models/tinyllama-trtllm/tensorrt_llm" }
}
```

**Init Container Responsibilities:**

The init container must:
1. Build the engine from checkpoint
2. Copy/symlink the engine to the correct model directory
3. Generate or update the `config.pbtxt` if needed

```
Init Container Flow:
    │
    ├── Read manifest from S3
    │
    ├── For each model in manifest:
    │   │
    │   ├── Check GPU fingerprint
    │   │
    │   ├── Check cache: s3://bucket/engines/{model}-{fingerprint}/
    │   │   │
    │   │   ├── Cache HIT: Download cached engine
    │   │   │
    │   │   └── Cache MISS: Build engine from checkpoint
    │   │       └── Upload to cache for future use
    │   │
    │   ├── Copy engine to: /triton-repo/models/{model}/tensorrt_llm/
    │   │
    │   ├── Copy tokenizer from checkpoint to model directory
    │   │
    │   └── Generate config.pbtxt from template (if not exists)
    │
    └── Signal ready for Triton to start
```

**Workflow:**

1. **User prepares fine-tuned model:**
   ```bash
   # In Domino workspace, upload checkpoint to S3
   aws s3 cp ./my-finetuned-model s3://triton-bucket/checkpoints/my-finetuned-llama --recursive
   ```

2. **User creates manifest file:**
   ```bash
   # Create manifest for the new model (filename = model name)
   cat > my-finetuned-llama.yaml << 'EOF'
   checkpoint: checkpoints/my-finetuned-llama
   architecture: llama
   dtype: float16
   max_batch_size: 2
   max_input_len: 1024
   max_seq_len: 2048
   EOF

   # Upload to manifests folder
   aws s3 cp my-finetuned-llama.yaml s3://triton-bucket/manifests/
   ```

3. **Trigger rebuild:**
   ```bash
   # Restart Triton pod to pick up new manifest
   kubectl rollout restart deployment/triton-inference-server -n <namespace>
   ```

4. **Init container:**
   - Scans `manifests/` folder for all `.yaml` files
   - For each manifest, uses filename as model name
   - Checks if cached engine exists for current GPU
   - Builds missing engines, uploads to cache
   - **Copies engines to correct model directories**
   - **Generates config.pbtxt from template with correct paths**
   - Triton starts with all engines ready

5. **Remove a model:**
   ```bash
   # Simply delete the manifest file
   aws s3 rm s3://triton-bucket/manifests/my-finetuned-llama.yaml

   # Restart to clean up
   kubectl rollout restart deployment/triton-inference-server -n <namespace>
   ```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Self-service** | Users add fine-tuned models without code changes |
| **Centralized config** | Single manifest controls all TRT-LLM builds |
| **Cached builds** | Engines cached by GPU fingerprint, rebuilt only when needed |
| **Flexible** | Supports any HuggingFace-compatible checkpoint |

**Limitations:**

- Supported architectures only (llama, gpt2, bloom, falcon, etc.)
- Checkpoint must be in HuggingFace format
- First build after adding model takes 5-15 minutes

#### Implementation Tasks

- [ ] Define per-model manifest schema (YAML format)
- [ ] Create `scripts/build_trtllm_engine.py` - Engine builder with GPU detection and caching
- [ ] Create `scripts/scan_manifests.py` - Scan `manifests/` folder and validate each file
- [ ] Add GPU fingerprint generation logic
- [ ] Create init container Dockerfile with TensorRT-LLM build tools
- [ ] Add Helm chart configuration for init container
- [ ] Add cache volume mount for built engines (S3 `engines/` folder)
- [ ] Implement cache invalidation (checkpoint modified time vs engine time)
- [ ] Add health check that waits for engine availability
- [ ] Create user documentation for adding fine-tuned models
- [ ] Add example manifests for common architectures (llama, mistral, phi, qwen, etc.)
- [ ] Add cleanup logic to remove models when manifest is deleted

#### Example Helm Configuration (Future)

```yaml
triton_inference_server:
  image: "nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3"

  # Enable manifest-driven build-on-startup
  trtllm:
    build_on_startup: true
    # Folder containing per-model manifest files
    manifests_folder: "manifests/"
    # Where to cache built engines
    engine_cache_path: "engines/"
    # Rebuild if checkpoint is newer than cached engine
    auto_invalidate: true
    # Timeout for building a single model
    build_timeout_minutes: 30

persistence:
  s3:
    bucket: my-triton-s3-bucket
    region: "us-west-2"
```

**User workflow to add a fine-tuned model:**

```bash
# 1. Upload checkpoint to S3
aws s3 cp ./my-model s3://my-triton-s3-bucket/checkpoints/my-model --recursive

# 2. Create manifest file (filename = model name)
cat > my-model.yaml << 'EOF'
checkpoint: checkpoints/my-model
architecture: llama
dtype: float16
max_batch_size: 2
max_input_len: 1024
max_seq_len: 2048
EOF
aws s3 cp my-model.yaml s3://my-triton-s3-bucket/manifests/

# 3. Restart pod to trigger build
kubectl rollout restart deployment/triton-inference-server -n <namespace>

# 4. Wait for build to complete, then load model
python scripts/model_management/load_model.py my-model
```

**User workflow to remove a model:**

```bash
# Just delete the manifest file
aws s3 rm s3://my-triton-s3-bucket/manifests/my-model.yaml

# Restart to clean up
kubectl rollout restart deployment/triton-inference-server -n <namespace>
```

This approach makes TensorRT-LLM deployment self-service - users add checkpoints and per-model manifest files, the system handles GPU-specific engine building automatically. No conflicts when multiple users add models simultaneously.
