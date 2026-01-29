# Cloud Identity Management Utils - Design

## Overview

This project provides tools for managing cloud identity mappings between Domino, Keycloak, Kubernetes, and AWS IAM. It enables IRSA (IAM Roles for Service Accounts) integration where Keycloak groups define which users can assume specific AWS IAM roles.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Entry Points                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │  FastAPI REST Service       │    │  Command Line Scripts               │ │
│  │  (src/admin.py)             │    │  (scripts/)                         │ │
│  │                             │    │                                     │ │
│  │  GET  /k8s/serviceaccounts  │    │  bootstrap_keycloak_client.py       │ │
│  │  POST /k8s/users/sync       │◄───│  kc_apply_group_spec.py             │ │
│  │  GET  /keycloak/groups      │    │                                     │ │
│  │  POST /keycloak/groups/member    │  sync.py users ──────────────────►  │ │
│  │  DELETE /keycloak/groups/member  │  sync.py irsa  ──────────────────►  │ │
│  │  GET  /iam/irsa/trust       │    │       (these call REST API)         │ │
│  │  POST /iam/irsa/sync        │    │                                     │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
│                                                                             │
│  Note: bootstrap_keycloak_client.py and kc_apply_group_spec.py are         │
│  standalone scripts that talk directly to Keycloak (not via REST API)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Core Modules                                      │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│  kc_irsa_core.py │ kc_admin_core.py │ iam_irsa_core.py │   irsa_client.py   │
│                  │                  │                  │                    │
│  AWS config      │  KC group        │  IAM trust       │  Sidecar for       │
│  generation      │  validation      │  policy mutation │  workspaces        │
└──────────────────┴──────────────────┴──────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          External Systems                                    │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│     Domino       │    Keycloak      │   Kubernetes     │      AWS IAM       │
│                  │                  │                  │                    │
│  /v4/users       │  Groups API      │  ServiceAccounts │  Trust Policies    │
│  /v4/auth        │  Users API       │  in compute ns   │  for IRSA roles    │
└──────────────────┴──────────────────┴──────────────────┴────────────────────┘
```

## Entry Points

### 1. FastAPI REST Service (`src/admin.py`)

A REST API deployed as a Kubernetes service, typically accessed by administrators or automation tools.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/healthz` | Health check |
| `GET` | `/k8s/serviceaccounts` | List K8s service accounts |
| `POST` | `/k8s/users/sync` | Sync Domino users to K8s service accounts |
| `GET` | `/keycloak/groups` | List Keycloak IRSA groups |
| `POST` | `/keycloak/groups/member` | Add user to Keycloak group |
| `DELETE` | `/keycloak/groups/member` | Remove user from Keycloak group |
| `GET` | `/iam/irsa/trust` | Get IRSA trust policy status |
| `POST` | `/iam/irsa/sync` | Sync Keycloak groups to IAM trust policies |

**Authentication:** Requires Domino admin credentials via `Authorization` header or `X-Domino-Api-Key`.

**Usage:**
```bash
# Via curl
curl -X POST http://admin-service:8000/k8s/users/sync \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"default_aws_role_arn": "arn:aws:iam::123:role/default", "dry_run": true}'
```

### 2. Command Line Scripts (`scripts/`)

Standalone scripts for one-time setup and operational tasks.

| Script | Purpose | Target |
|--------|---------|--------|
| `bootstrap_keycloak_client.py` | One-time: Create Keycloak service account client | Keycloak directly |
| `kc_apply_group_spec.py` | Apply Keycloak group structure from JSON spec | Keycloak directly |
| `sync.py users` | Sync Domino users to K8s service accounts | REST API |
| `sync.py irsa` | Sync Keycloak groups to IAM trust policies | REST API |

**Usage:**
```bash
# One-time setup
python scripts/bootstrap_keycloak_client.py

# Apply group spec
python scripts/kc_apply_group_spec.py --spec groups.json --dry-run

# Sync operations (via REST API)
python scripts/sync.py users --default-aws-role-arn $ARN --dry-run
python scripts/sync.py irsa --dry-run
```

## Core Modules

### `kc_irsa_core.py`
- Parses Keycloak group paths into IRSA role mappings
- Generates AWS CLI config files from JWT claims
- Handles direct roles (`arn:aws:iam::123:role/role1`) and chained roles (`proxy->target`)

### `kc_admin_core.py`
- Validates IRSA group paths against root group
- Resolves Keycloak group IDs by path
- Pure functions for testability

### `iam_irsa_core.py`
- Mutates AWS IAM trust policies for IRSA
- Adds/removes `sub` claims for service accounts
- Handles StringEquals/StringLike conditions

### `irsa_client.py`
- Sidecar container for Domino workspaces
- Fetches JWT from Domino API proxy
- Writes AWS config file based on user's Keycloak groups

## Data Flow

### User Sync Flow
```
Domino /v4/users  →  admin.py  →  K8s ServiceAccounts
                                   (with IRSA annotations)
```

### IRSA Sync Flow
```
Keycloak Groups  →  admin.py  →  AWS IAM Trust Policies
(group members)                  (sub claims)
```

### Workspace IRSA Flow
```
Domino JWT  →  irsa_client.py  →  ~/.aws/config
(user_groups)                     (profiles per role)
```

## Keycloak Group Structure

Groups are organized under a root group (default: `domino-irsa-roles`):

```
/domino-irsa-roles/
  ├── arn:aws:iam::123456789012:role/data-scientist
  ├── arn:aws:iam::123456789012:role/ml-engineer
  └── arn:aws:iam::111111111111:role/proxy->arn:aws:iam::222222222222:role/target
```

- **Direct mapping:** Group name is the IAM role ARN
- **Chained mapping:** `proxy_arn->target_arn` for cross-account access

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOMINO_HOST` | Domino API URL | `http://nucleus-frontend.domino-platform:80` |
| `COMPUTE_NAMESPACE` | K8s namespace for service accounts | `domino-compute` |
| `KC_URL` | Keycloak base URL | - |
| `KC_REALM` | Keycloak realm | - |
| `KC_CLIENT_ID` | Keycloak client ID | - |
| `KC_CLIENT_SECRET` | Keycloak client secret | - |
| `KC_IRSA_ROOT_GROUP` | Root group for IRSA roles | `domino-irsa-roles` |
| `OIDC_PROVIDER_URL` | EKS OIDC provider URL | - |
| `AUDIENCE` | OIDC audience | `sts.amazonaws.com` |

## Security Considerations

1. **Admin API Authentication:** All admin endpoints require Domino admin privileges
2. **Keycloak Client:** Uses service account with minimal roles (`query-groups`, `query-users`, `manage-users`)
3. **IAM Permissions:** Service needs IAM permissions to read/update role trust policies
4. **Trust Policy Locking:** Concurrent IRSA sync requests are serialized via mutex

## File Structure

```
src/
├── admin.py           # FastAPI REST service
├── kc_irsa_core.py    # AWS config generation
├── kc_admin_core.py   # Keycloak group validation
├── iam_irsa_core.py   # IAM trust policy mutation
└── irsa_client.py     # Workspace sidecar

scripts/
├── bootstrap_keycloak_client.py  # One-time KC setup
├── kc_apply_group_spec.py        # Apply group spec
└── sync.py                       # CLI for sync operations

tests/
├── test_iam_irsa_core.py
├── test_kc_admin_core.py
└── test_kc_irsa_core.py
```