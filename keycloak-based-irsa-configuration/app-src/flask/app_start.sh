#!/usr/bin/env bash
# Startup script for IRSA Admin Dashboard
#
# Usage:
#   LOCAL=1 bash app-src/flask/app_start.sh    # Local development (no reverse proxy)
#   bash app-src/flask/app_start.sh            # Production (behind reverse proxy)

set -e

export APP_PORT="${APP_PORT:-8888}"
export DEFAULT_IAM_ROLE_ARN="${DEFAULT_IAM_ROLE_ARN:-arn:aws:iam::123456789012:role/YourDefaultRole}"

if [ "$LOCAL" = "1" ]; then
    echo "Running in LOCAL mode..."
    # No APP_ROOT_PATH - direct access without reverse proxy
    unset APP_ROOT_PATH

    pip install --user flask requests
    python app-src/flask/app.py
else
    echo "Running in PRODUCTION mode (behind reverse proxy)..."
    export APP_ROOT_PATH="${APP_ROOT_PATH:-/apps/irsa-admin-v2/}"

    pip install --user flask requests
    python app-src/flask/app.py
fi