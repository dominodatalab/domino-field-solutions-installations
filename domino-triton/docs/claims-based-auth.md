# Claims-Based Authentication

This document explains the claims-based authentication mechanism for Triton Inference Server field solutions in Domino.

## Overview

Claims-based authentication provides a consistent way to allow users to assume custom roles based on claims defined in Keycloak. This enables fine-grained access control for admin operations.

## Authorization Model

### 1. Identity Verification

Every request must carry an authorization header:

```
Authorization: Bearer <token>
```

Or the Domino API key header:

```
X-Domino-Api-Key: <api-key>
```

The token/key is validated against the Domino platform:

```
GET /v4/auth/principal
```

If the principal is anonymous, access is denied. This call fails if the token is invalid or expired.

### 2. Domino Platform Admins

Domino platform admins (`isAdmin: true` in the principal response) **always have access** to admin operations, regardless of the `CLAIM_BASED` setting. This ensures platform administrators can manage all services without requiring specific claims.

```json
{
  "isAdmin": true,
  "isAnonymous": false,
  "canonicalId": "user-123",
  ...
}
```

### 3. Claim-Based Admin Enforcement (for non-admin users)

Controlled by the environment variable:

```bash
CLAIM_BASED=true | false
```

When `CLAIM_BASED=true` and the user is **not** a Domino platform admin, the JWT must include a `domino-service-auth` claim like:

```json
"domino-service-auth": [
  {
    "namespace": "domino-inference-test",
    "service": "triton-inference-server",
    "role": {
      "admin": true
    }
  }
]
```

The intent here is to define which namespaces and services a user can act as admin for.

Admin access is granted only if:

- `namespace` matches the pod's namespace
- `service` matches `CLAIM_SERVICE` environment variable
- `role[CLAIM_ROLE] == true`

When `CLAIM_BASED=false`, only Domino platform admins can access admin operations.

## Protected Endpoints

The following endpoints require admin/claims authorization:

### HTTP (REST Proxy)
| Endpoint | Description |
|----------|-------------|
| `POST /v2/repository/models/{model}/load` | Load a model |
| `POST /v2/repository/models/{model}/unload` | Unload a model |
| `POST /v1/models/{model}/pin` | Pin/unpin a model |
| `POST /v1/models/{model}/cordon` | Cordon a model |
| `POST /v1/models/{model}/uncordon` | Uncordon a model |

### gRPC (gRPC Proxy)
| Method | Description |
|--------|-------------|
| `RepositoryModelLoad` | Load a model |
| `RepositoryModelUnload` | Unload a model |

**Note:** Inference endpoints (`/v2/models/{model}/infer`, `ModelInfer`) only require basic authentication, not admin privileges. Auto-load triggered by inference requests bypasses the admin check.

### Authorization Flow

```
User Request
    │
    ▼
Is authenticated? ──No──► 401 Unauthorized
    │ Yes
    ▼
Is anonymous? ──Yes──► 403 Forbidden
    │ No
    ▼
Is Domino admin? ──Yes──► ✓ Access granted
    │ No
    ▼
CLAIM_BASED=true? ──No──► ✓ Access granted (any authenticated user)
    │ Yes
    ▼
Has valid claim? ──No──► 403 "does not have a valid claim"
    │ Yes
    ▼
✓ Access granted
```

## Expanding to Other Services

This same mechanism can be used for other field solutions by defining more claims:

```json
"domino-service-auth": [
  {
    "namespace": "domino-inference-test",
    "service": "triton-inference-server",
    "role": {
      "admin": true
    }
  },
  {
    "namespace": "domino-field",
    "service": "irsa",
    "role": {
      "admin": true
    }
  }
]
```

Or add more roles per service, such as which models deployed in Triton Inference Server a user can invoke:

```json
"domino-service-auth": [
  {
    "namespace": "domino-inference-test",
    "service": "triton-inference-server",
    "role": {
      "admin": true,
      "models_invoke": ["modelA", "modelB:v[1,2,3,7]"]
    }
  },
  {
    "namespace": "domino-field",
    "service": "irsa",
    "role": {
      "admin": true
    }
  }
]
```

## Defining Claims in Keycloak

Keycloak typically generates the `domino-service-auth` claim via a **User Attribute (JSON) mapper**.

### Steps to Configure

1. In Keycloak Admin Console, navigate to your client
2. Go to **Mappers** tab
3. Create a new mapper:
   - **Mapper Type**: User Attribute
   - **Name**: `domino-service-auth`
   - **User Attribute**: `domino-service-auth`
   - **Token Claim Name**: `domino-service-auth`
   - **Claim JSON Type**: JSON

4. Set the user attribute as a JSON array on each user who needs access

### Example User Attribute

In the user's attributes, set:

| Key | Value |
|-----|-------|
| `domino-service-auth` | `[{"namespace":"domino-inference-dev","service":"triton-inference-server","role":{"admin":true}}]` |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAIM_BASED` | Enable claims-based authorization | `false` |
| `CLAIM_SERVICE` | Service name to match in claims | `triton-inference-server` |
| `CLAIM_ROLE` | Role name to check | `admin` |
| `POD_NAMESPACE` | Current namespace (auto-detected in K8s) | - |

## Related Documentation

- [Helm Installation Guide](helm-install.md) - Deployment configuration including `env.claim_based`
- [Runbook](runbook.md) - Usage instructions and testing
