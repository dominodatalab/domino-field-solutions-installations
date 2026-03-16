#!/bin/bash
# ==============================================================================
# Build Model-Specific Packages for Triton Python Backend
# ==============================================================================
# This script installs model-specific dependencies into a packages/ directory
# within each model folder. The model.py adds this to sys.path at runtime.
#
# This approach:
# - Uses base Triton's numpy/torch/transformers (no compatibility issues)
# - Adds only model-specific libraries (librosa, embeddings, etc.)
# - No Triton image rebuild needed for model-specific deps
#
# Usage:
#   ./scripts/build_model_packages.sh              # Build all
#   ./scripts/build_model_packages.sh whisper      # Build whisper only
#   ./scripts/build_model_packages.sh smollm       # Build smollm only
#
# Output:
#   triton-repo/models/<model>/packages/
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/triton-repo/models"

# Models with potential model-specific requirements
WHISPER_MODEL="whisper-tiny-python"
SMOLLM_MODEL="smollm-135m-python"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

build_packages() {
    local model_name=$1
    local model_dir="$MODELS_DIR/$model_name"
    local requirements_file="$model_dir/requirements-model.txt"
    local packages_dir="$model_dir/packages"

    if [ ! -f "$requirements_file" ]; then
        log_warn "No requirements-model.txt found for $model_name, skipping"
        return 0
    fi

    # Check if requirements file has actual packages (not just comments)
    if ! grep -q "^[^#]" "$requirements_file" 2>/dev/null; then
        log_info "$model_name has no model-specific requirements, skipping"
        return 0
    fi

    log_info "Building packages for: $model_name"
    log_info "Requirements: $requirements_file"
    log_info "Output: $packages_dir/"

    # Remove existing packages directory
    rm -rf "$packages_dir"
    mkdir -p "$packages_dir"

    # Install packages directly with pip
    log_info "Installing packages with pip..."
    python -m pip install --target="$packages_dir" -r "$requirements_file"

    log_info "Installed packages:"
    ls -la "$packages_dir" | head -20

    if [ -d "$packages_dir" ]; then
        local count=$(ls -1 "$packages_dir" 2>/dev/null | wc -l)
        log_info "Successfully created $packages_dir/ with $count items"
        return 0
    else
        log_error "Failed to create packages for $model_name"
        return 1
    fi
}

# Parse arguments
BUILD_WHISPER=false
BUILD_SMOLLM=false

if [ $# -eq 0 ]; then
    BUILD_WHISPER=true
    BUILD_SMOLLM=true
else
    for arg in "$@"; do
        case $arg in
            whisper|whisper-tiny-python) BUILD_WHISPER=true ;;
            smollm|smollm-135m-python|llm) BUILD_SMOLLM=true ;;
            all) BUILD_WHISPER=true; BUILD_SMOLLM=true ;;
            *) log_error "Unknown model: $arg"; exit 1 ;;
        esac
    done
fi

echo "============================================================"
echo "Building Model-Specific Packages"
echo "============================================================"
echo ""

SUCCESS=0
FAILED=0

if [ "$BUILD_WHISPER" = true ]; then
    if build_packages "$WHISPER_MODEL"; then ((SUCCESS++)); else ((FAILED++)); fi
    echo ""
fi

if [ "$BUILD_SMOLLM" = true ]; then
    if build_packages "$SMOLLM_MODEL"; then ((SUCCESS++)); else ((FAILED++)); fi
    echo ""
fi

echo "============================================================"
echo "Build Summary: $SUCCESS succeeded, $FAILED failed"
echo "============================================================"

[ $FAILED -gt 0 ] && exit 1

echo ""
log_info "Next steps:"
echo "  1. Rebuild Triton (if base deps changed): docker compose up --build -d backend"
echo "  2. Or just restart (if only model packages changed): docker compose restart backend"
echo "  3. Test: python scripts/whisper_audio_grpc_client.py --audio samples/audio_sample.wav"
