# Python Model Dependencies

This guide explains how to manage Python dependencies for Triton Python backend models.

## Architecture

Dependencies are split into two layers:

| Layer | Location | Contents | Rebuild Triton? |
|-------|----------|----------|-----------------|
| **Base** | `docker/requirements-triton.txt` | Shared libs (transformers, torch, numpy, pandas, scikit-learn) | Yes |
| **Model-specific** | `models/<model>/requirements-model.txt` | Per-model extras (humanize, embeddings, etc.) | No |

## Directory Structure

```
models/                              # Source (checked into git)
└── whisper-tiny-python/
    ├── config.pbtxt
    ├── requirements-model.txt       # Model-specific dependencies
    └── 1/model.py                   # Adds packages/ to sys.path

triton-repo/models/                  # Deployment folder
└── whisper-tiny-python/
    ├── config.pbtxt                 # Copied from models/
    ├── requirements-model.txt       # Copied from models/
    ├── packages/                    # Built packages (auto-generated)
    │   └── humanize/
    └── 1/model.py                   # Copied from models/
```

**Key points:**
- Source files (`config.pbtxt`, `model.py`, `requirements-model.txt`) are in `models/` and checked into git
- Download scripts copy these to `triton-repo/models/`
- Packages are built into `triton-repo/models/<model>/packages/` (not in source folder)
- The `packages/` directory is gitignored (build artifact)

## Adding Shared Dependencies

For libraries used by multiple models (torch, transformers, numpy, etc.):

1. Edit `docker/requirements-triton.txt`
2. Rebuild Triton:
   ```bash
   docker compose up --build -d backend
   ```

## Adding Model-Specific Dependencies

For libraries used by only one model:

1. Edit the source requirements file:
   ```bash
   echo "humanize>=4.0.0" >> models/whisper-tiny-python/requirements-model.txt
   ```

2. Copy to deployment folder (or re-run download script):
   ```bash
   cp models/whisper-tiny-python/requirements-model.txt \
      triton-repo/models/whisper-tiny-python/
   ```

3. Build the packages:
   ```bash
   ./scripts/build_model_packages.sh whisper
   ```

4. Restart Triton (no rebuild needed):
   ```bash
   docker compose restart backend
   ```

## How It Works

The `model.py` adds the `packages/` directory to `sys.path` at startup:

```python
import sys
import os
_model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_packages_dir = os.path.join(_model_dir, "packages")
if os.path.isdir(_packages_dir) and _packages_dir not in sys.path:
    sys.path.insert(0, _packages_dir)

# Now you can import model-specific packages
import humanize  # Loaded from packages/ directory
```

This allows model-specific imports without affecting other models or requiring a Triton rebuild.

## Example: Whisper with humanize

The Whisper model uses `humanize` to format output nicely:

**models/whisper-tiny-python/requirements-model.txt:**
```
humanize>=4.0.0
```

**models/whisper-tiny-python/1/model.py:**
```python
# Add packages to path
_packages_dir = os.path.join(_model_dir, "packages")
if os.path.isdir(_packages_dir):
    sys.path.insert(0, _packages_dir)

# Import model-specific package
try:
    import humanize
    _HAS_HUMANIZE = True
except ImportError:
    _HAS_HUMANIZE = False

# Use in code
if _HAS_HUMANIZE:
    print(f"Transcribed in {humanize.precisedelta(elapsed)}")
```

## Build Script Usage

```bash
# Build packages for all models
./scripts/build_model_packages.sh

# Build for specific model
./scripts/build_model_packages.sh whisper
./scripts/build_model_packages.sh smollm

# The script:
# 1. Reads requirements-model.txt from triton-repo/models/<model>/
# 2. Runs pip install --target to install packages directly
# 3. Installs to triton-repo/models/<model>/packages/
```

## Current Configuration

### Base Image (`docker/requirements-triton.txt`)
- transformers, tokenizers, sentencepiece
- optimum[onnxruntime]
- numpy, pandas, scikit-learn

### Whisper (`models/whisper-tiny-python/requirements-model.txt`)
- humanize (for human-readable output formatting)

### SmolLM (`models/smollm-135m-python/requirements-model.txt`)
- (none currently)

## Notes

- The `packages/` directory is gitignored (build artifact)
- Packages are installed directly with `pip install --target`
- For pure Python packages (like `humanize`), this works without Docker
- For packages with native extensions, ensure compatibility with the Triton runtime
- Core libs (numpy, torch) must stay in base image to avoid serialization issues
- Source files go in `models/`, packages are built in `triton-repo/models/`
