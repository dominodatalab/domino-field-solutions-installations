#!/bin/bash
# Startup script for Triton Admin Dashboard
#
# Usage (from project root):
#   bash app-src/start.sh
#
# This script is designed to run in Domino behind a reverse proxy.
# The app will be accessible at: /ap/

set -e

# Configuration
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8888}"
# ROOT_PATH is used for generating URLs in templates (for browser navigation).
# Domino's gateway strips this prefix before forwarding to the app.
# The app handles requests at / but browsers need the full path in URLs.
APP_ROOT_PATH_RAW="${APP_ROOT_PATH:-${DOMINO_RUN_HOST_PATH:-}}"
export APP_ROOT_PATH="${APP_ROOT_PATH_RAW%/}"
export ROOT_PATH="${APP_ROOT_PATH}"

# Paths (relative to project root in Domino)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export SAMPLES_PATH="${SAMPLES_PATH:-$PROJECT_ROOT/samples}"
export SCRIPTS_PATH="${SCRIPTS_PATH:-$PROJECT_ROOT/scripts}"
export NAMESPACES_FILE="${NAMESPACES_FILE:-$SCRIPT_DIR/namespaces.json}"

# Benchmark results path - where benchmark markdown files are published
# This points to the shared dataset folder for persistent benchmark results
export BENCHMARK_RESULTS_PATH="${BENCHMARK_RESULTS_PATH:-/mnt/data/grpc-git/benchmarks}"
export RESULTS_PATH="$BENCHMARK_RESULTS_PATH"

# Default proxy URLs (can be overridden via namespaces.json or environment)
export DEFAULT_PROXY_URL="${DEFAULT_PROXY_URL:-http://localhost:8080}"

echo "=========================================="
echo "Triton Admin Dashboard"
echo "=========================================="
echo "Host: $APP_HOST"
echo "Port: $APP_PORT"
echo "Root Path: $APP_ROOT_PATH"
echo "Samples: $SAMPLES_PATH"
echo "Scripts: $SCRIPTS_PATH"
echo "Benchmark Results: $RESULTS_PATH"
echo "Namespaces: $NAMESPACES_FILE"
echo "=========================================="

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_PATH"

# Install dependencies if needed
if [ ! -f "$SCRIPT_DIR/.deps_installed" ]; then
    echo "Installing dependencies..."
    # Install app dependencies
    pip install -r "$SCRIPT_DIR/requirements.txt"
    # Install client dependencies with pinned versions from docker folder
    pip install -r "$PROJECT_ROOT/docker/requirements-client.txt"
    touch "$SCRIPT_DIR/.deps_installed"
fi

# Change to app directory and start server
cd "$SCRIPT_DIR"

echo "Starting server behind reverse proxy..."
# Note: Don't use --root-path when Domino's gateway already strips the path prefix
# The gateway forwards requests to / directly, so no path rewriting is needed
uvicorn server:app \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --proxy-headers \
    --forwarded-allow-ips='*'
