#!/bin/bash
#
# benchmark_all.sh
#
# Runs all benchmark scripts against a Triton deployment in Kubernetes.
# Sets up environment variables based on the target namespace.
#
# Usage:
#   ./scripts/benchmark_all.sh <namespace>
#
# Examples:
#   ./scripts/benchmark_all.sh domino-inference-dev
#   ./scripts/benchmark_all.sh domino-inference-test
#   ./scripts/benchmark_all.sh domino-inference-prod
#
# Requirements:
#   - DOMINO_USER_API_KEY must be set
#   - pip install -r docker/requirements-client.txt
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Service name (matches Helm chart)
SERVICE_NAME="triton-inference-server-proxy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_usage() {
    echo ""
    echo "Usage: $0 <namespace>"
    echo ""
    echo "Runs all benchmarks against the Triton deployment in the specified namespace."
    echo ""
    echo "Examples:"
    echo ""
    echo -e "  ${GREEN}# Development environment${NC}"
    echo "  $0 domino-inference-dev"
    echo ""
    echo -e "  ${YELLOW}# Test environment${NC}"
    echo "  $0 domino-inference-test"
    echo ""
    echo -e "  ${RED}# Production environment${NC}"
    echo "  $0 domino-inference-prod"
    echo ""
    echo "This will set the following environment variables:"
    echo ""
    echo "  TRITON_GRPC_URL=${SERVICE_NAME}.<namespace>.svc.cluster.local:50051"
    echo "  TRITON_REST_URL=http://${SERVICE_NAME}.<namespace>.svc.cluster.local:8080"
    echo ""
    echo "Prerequisites:"
    echo "  - DOMINO_USER_API_KEY must be set"
    echo "  - Python dependencies: pip install -r docker/requirements-client.txt"
    echo ""
    echo "For local testing (Docker Compose), run without this script:"
    echo "  python scripts/benchmark_all.py"
    echo ""
}

# Check if namespace argument is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}No namespace provided.${NC}"
    show_usage
    exit 1
fi

NAMESPACE="$1"

# Check if API key is set
if [ -z "$DOMINO_USER_API_KEY" ]; then
    echo -e "${RED}Error: DOMINO_USER_API_KEY is not set.${NC}"
    echo ""
    echo "Please set your API key:"
    echo "  export DOMINO_USER_API_KEY=your-api-key"
    echo ""
    exit 1
fi

# Set proxy URLs based on namespace
export TRITON_GRPC_URL="${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:50051"
export TRITON_REST_URL="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:8080"

echo ""
echo "============================================================"
echo "Running All Benchmarks"
echo "============================================================"
echo ""
echo "Namespace:     ${NAMESPACE}"
echo "gRPC URL:      ${TRITON_GRPC_URL}"
echo "REST URL:      ${TRITON_REST_URL}"
echo "API Key:       ****${DOMINO_USER_API_KEY: -4}"
echo ""
echo "============================================================"

# Run the Python benchmark script
cd "$PROJECT_ROOT"
python scripts/benchmark_all.py

echo ""
echo -e "${GREEN}Done!${NC}"
