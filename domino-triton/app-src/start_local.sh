#!/bin/bash
# Local development startup script for Triton Admin Dashboard
#
# Usage (from any directory):
#   bash app-src/start_local.sh
#
# This script runs the app directly without reverse proxy.
# The app will be accessible at: http://localhost:8888/

set -e

# Configuration for local development
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8888}"

# Paths (relative to project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export SAMPLES_PATH="${SAMPLES_PATH:-$PROJECT_ROOT/samples}"
export SCRIPTS_PATH="${SCRIPTS_PATH:-$PROJECT_ROOT/scripts}"
export RESULTS_PATH="${RESULTS_PATH:-$PROJECT_ROOT/results}"
# Use local namespaces config (localhost URLs instead of Docker container names)
export NAMESPACES_FILE="${NAMESPACES_FILE:-$SCRIPT_DIR/namespaces.local.json}"

# Local proxy URLs (assuming docker-compose is running)
export DEFAULT_PROXY_URL="${DEFAULT_PROXY_URL:-http://localhost:8080}"

echo "=========================================="
echo "Triton Admin Dashboard (Local Development)"
echo "=========================================="
echo "Host: $APP_HOST"
echo "Port: $APP_PORT"
echo "URL: http://localhost:$APP_PORT/"
echo "Samples: $SAMPLES_PATH"
echo "Scripts: $SCRIPTS_PATH"
echo "Results: $RESULTS_PATH"
echo "=========================================="

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_PATH"

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Install dependencies
echo "Installing/updating dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Change to app directory and start server with hot reload
cd "$SCRIPT_DIR"

echo "Starting development server with hot reload..."
uvicorn server:app \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --reload \
    --reload-dir "$SCRIPT_DIR"
