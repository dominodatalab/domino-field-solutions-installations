#!/usr/bin/env python3
"""
bootstrap_keycloak_client.py

Creates a Keycloak OIDC client with a SERVICE ACCOUNT and assigns realm-management
client roles:
  - query-groups
  - query-users
  - manage-users

IMPORTANT BEHAVIOR:
  - CLIENT_ID must be explicitly provided
  - If CLIENT_ID already exists, script TERMINATES (no mutation)

Environment (Domino / your setup):
  KC_URL="https://<host>/auth"
  REALM="DominoRealm"
  KC_ADMIN_USER="admin"
  KC_ADMIN_PASS="***"
  CLIENT_ID="irsa-mapping-manager"

Optional:
  ROLE_NAMES="query-groups,query-users,manage-users"
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

import requests


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        die(f"Missing required env var: {name}")
    return v


KC_URL = env("KC_URL").rstrip("/")
REALM = env("REALM")
KC_ADMIN_USER = env("KC_ADMIN_USER")
KC_ADMIN_PASS = env("KC_ADMIN_PASS")
CLIENT_ID = env("CLIENT_ID")

ROLE_NAMES = os.getenv("ROLE_NAMES", "query-groups,query-users,manage-users")
ROLE_LIST: List[str] = [r.strip() for r in ROLE_NAMES.split(",") if r.strip()]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def post_form(url: str, data: dict) -> dict:
    r = requests.post(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
        verify=False
    )
    if r.status_code >= 400:
        die(f"{r.status_code} error POST {url}: {r.text[:500]}")
    print(r.status_code)
    print("----")
    print(r.text)
    print("----")
    return r.json()


def get(url: str, token: str) -> dict:
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        verify=False
    )
    if r.status_code >= 400:
        die(f"{r.status_code} error GET {url}: {r.text[:500]}")
    return r.json()


def post_json(url: str, token: str, payload: dict) -> None:
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
        verify=False
    )
    if r.status_code not in (200, 201, 204):
        die(f"{r.status_code} error POST {url}: {r.text[:500]}")


# ---------------------------------------------------------------------------
# Step 1: Mint admin token (master realm, admin-cli)
# ---------------------------------------------------------------------------

print(">> Minting admin token (master realm, admin-cli)…", file=sys.stderr)

token_resp = post_form(
    f"{KC_URL}/realms/master/protocol/openid-connect/token",
    {
        "grant_type": "password",
        "client_id": "admin-cli",
        "username": KC_ADMIN_USER,
        "password": KC_ADMIN_PASS,
    },
)

ADMIN_TOKEN = token_resp.get("access_token")
if not ADMIN_TOKEN:
    die("Failed to mint admin token. Check admin credentials.")


# ---------------------------------------------------------------------------
# Step 2: Ensure client does NOT exist
# ---------------------------------------------------------------------------

print(f">> Checking if client '{CLIENT_ID}' already exists…", file=sys.stderr)

clients = get(
    f"{KC_URL}/admin/realms/{REALM}/clients?clientId={CLIENT_ID}",
    ADMIN_TOKEN,
)

if clients:
    die(f"Client '{CLIENT_ID}' already exists (uuid={clients[0]['id']}). Terminating.")


# ---------------------------------------------------------------------------
# Step 3: Create client
# ---------------------------------------------------------------------------

print(f">> Creating client '{CLIENT_ID}'…", file=sys.stderr)

create_payload = {
    "clientId": CLIENT_ID,
    "enabled": True,
    "protocol": "openid-connect",
    "publicClient": False,
    "serviceAccountsEnabled": True,
    "standardFlowEnabled": False,
    "directAccessGrantsEnabled": False,
    "implicitFlowEnabled": False,
}

post_json(
    f"{KC_URL}/admin/realms/{REALM}/clients",
    ADMIN_TOKEN,
    create_payload,
)

clients = get(
    f"{KC_URL}/admin/realms/{REALM}/clients?clientId={CLIENT_ID}",
    ADMIN_TOKEN,
)

if not clients:
    die("Client creation appeared to succeed but lookup failed.")

CLIENT_UUID = clients[0]["id"]


# ---------------------------------------------------------------------------
# Step 4: Get service-account user
# ---------------------------------------------------------------------------

print(">> Fetching service-account user…", file=sys.stderr)

sa_user = get(
    f"{KC_URL}/admin/realms/{REALM}/clients/{CLIENT_UUID}/service-account-user",
    ADMIN_TOKEN,
)

SA_USER_ID = sa_user.get("id")
if not SA_USER_ID:
    die("Could not resolve service-account user.")


# ---------------------------------------------------------------------------
# Step 5: Assign realm-management roles
# ---------------------------------------------------------------------------

print(">> Resolving realm-management client…", file=sys.stderr)

rm_clients = get(
    f"{KC_URL}/admin/realms/{REALM}/clients?clientId=realm-management",
    ADMIN_TOKEN,
)
if not rm_clients:
    die("realm-management client not found.")

RM_UUID = rm_clients[0]["id"]

all_roles = get(
    f"{KC_URL}/admin/realms/{REALM}/clients/{RM_UUID}/roles",
    ADMIN_TOKEN,
)

roles_to_assign = []
for role_name in ROLE_LIST:
    role = next((r for r in all_roles if r["name"] == role_name), None)
    if not role:
        die(f"Missing expected role '{role_name}' in realm-management.")
    roles_to_assign.append(role)

print(">> Assigning roles to service-account user…", file=sys.stderr)

post_json(
    f"{KC_URL}/admin/realms/{REALM}/users/{SA_USER_ID}/role-mappings/clients/{RM_UUID}",
    ADMIN_TOKEN,
    roles_to_assign,
)


# ---------------------------------------------------------------------------
# Step 6: Fetch client secret
# ---------------------------------------------------------------------------

print(">> Fetching client secret…", file=sys.stderr)

secret = get(
    f"{KC_URL}/admin/realms/{REALM}/clients/{CLIENT_UUID}/client-secret",
    ADMIN_TOKEN,
)

CLIENT_SECRET = secret.get("value")
if not CLIENT_SECRET:
    die("Failed to fetch client secret.")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print(
    json.dumps(
        {
            "kc_url": KC_URL,
            "realm": REALM,
            "client_id": CLIENT_ID,
            "client_uuid": CLIENT_UUID,
            "client_secret": CLIENT_SECRET,
            "service_account_user_id": SA_USER_ID,
            "granted_roles_client": "realm-management",
            "granted_roles": ROLE_LIST,
        },
        indent=2,
    )
)
