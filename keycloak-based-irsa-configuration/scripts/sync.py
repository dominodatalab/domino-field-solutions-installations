#!/usr/bin/env python3
"""
Sync utilities for cloud identity management.

Usage:
    # Sync Domino users to K8s service accounts
    python scripts/sync.py users --default-aws-role-arn <ARN> [--dry-run]

    # Sync Keycloak groups to IAM IRSA trust policies
    python scripts/sync.py irsa [--dry-run]

Environment:
    DOMINO_API_PROXY  - Required. Domino API proxy URL for auth token
    ADMIN_URL         - Optional. Admin service URL (default: http://domino-irsa-lite-admin.domino-field.svc.cluster.local:8000)
"""

import argparse
import json
import os
import sys

import requests

ADMIN_URL = os.getenv("ADMIN_URL", "http://domino-irsa-lite-admin.domino-field.svc.cluster.local:8000")


def get_domino_token() -> str:
    proxy = os.environ.get("DOMINO_API_PROXY")
    if not proxy:
        raise RuntimeError("DOMINO_API_PROXY environment variable is not set")
    resp = requests.get(f"{proxy}/access-token")
    resp.raise_for_status()
    return resp.text.strip()


def call_admin_api(endpoint: str, body: dict) -> dict:
    token = get_domino_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(f"{ADMIN_URL}{endpoint}", json=body, headers=headers, timeout=30)
    if resp.status_code >= 400:
        print(f"ERROR: {endpoint} failed ({resp.status_code})", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)
    return resp.json()


def cmd_users(args):
    """Sync Domino users to K8s service accounts."""
    result = call_admin_api("/k8s/users/sync", {
        "default_aws_role_arn": args.default_aws_role_arn,
        "dry_run": args.dry_run,
    })
    print(json.dumps(result, indent=2))


def cmd_irsa(args):
    """Sync Keycloak groups to IAM IRSA trust policies."""
    result = call_admin_api("/iam/irsa/sync", {"dry_run": args.dry_run})
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync utilities for cloud identity management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # users subcommand
    p_users = subparsers.add_parser("users", help="Sync Domino users to K8s service accounts")
    p_users.add_argument("--default-aws-role-arn", required=True, help="Default AWS IAM role ARN for IRSA annotation")
    p_users.add_argument("--dry-run", action="store_true", help="Print planned actions without modifying resources")
    p_users.set_defaults(func=cmd_users)

    # irsa subcommand
    p_irsa = subparsers.add_parser("irsa", help="Sync Keycloak groups to IAM IRSA trust policies")
    p_irsa.add_argument("--dry-run", action="store_true", help="Print planned actions without modifying resources")
    p_irsa.set_defaults(func=cmd_irsa)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())