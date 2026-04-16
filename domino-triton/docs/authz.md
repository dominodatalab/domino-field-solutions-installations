# Authentication & Authorization

This document describes how authentication works for the Domino Triton Inference proxies and client scripts.

## Overview

The proxies support two authentication methods:
- **Bearer Token** (JWT) - Validated locally using JWKS public keys
- **API Key** - Validated against Domino's `/v4/auth/principal` endpoint

## Authentication Modes

The proxy supports three authentication modes controlled by environment variables:

| Mode | Environment Variable | Behavior |
|------|---------------------|----------|
| **Required** | (default) | All requests must have valid credentials |
| **Optional** | `AUTH_OPTIONAL=true` | No credentials = allow; credentials provided = must be valid |
| **Disabled** | `SKIP_AUTH=true` | All requests allowed, no validation |

### When to Use Each Mode

- **Required**: Production deployments
- **Optional**: Local development where you want to test both authenticated and unauthenticated flows
- **Disabled**: Local testing only (not recommended)

## Client Authentication

### Resolution Order

Client scripts use `auth_helper.py` which resolves credentials in this order:

1. **`DOMINO_USER_TOKEN`** env var - Manually set Bearer token (for testing outside Domino)
2. **`$DOMINO_API_PROXY/access-token`** - Automatic token fetch (inside Domino workspaces)
3. **`DOMINO_USER_API_KEY`** env var - API key fallback

Inside Domino workspaces, `DOMINO_API_PROXY` is pre-configured, so authentication is fully automatic with no setup required.

### Token Caching

Tokens fetched from `DOMINO_API_PROXY/access-token` are cached to avoid repeated network calls:

- **First request**: Fetches token from endpoint, caches it
- **Subsequent requests**: Uses cached token (no network call)
- **On 401/403 error**: Call `invalidate_token()` to clear cache
- **Next request after invalidation**: Fetches fresh token

### Usage in Client Code

```python
from auth_helper import get_auth_headers, invalidate_token

# Get auth headers (automatically resolved and cached)
headers = get_auth_headers()

# Make request
response = client.infer(model_name, inputs, headers=headers)

# Handle token expiry
if response.status_code in (401, 403):
    invalidate_token()  # Clear cached token
    headers = get_auth_headers()  # Refetches from endpoint
    response = client.infer(model_name, inputs, headers=headers)
```

### Header Formats

| Auth Type | Header |
|-----------|--------|
| Bearer Token | `Authorization: Bearer <token>` |
| API Key | `X-Domino-Api-Key: <key>` |

## Proxy-Side Validation

### JWT Validation (Fast Path)

When `JWKS_URL` is configured, Bearer tokens are validated locally:

1. Token signature verified against cached JWKS public keys
2. Claims validated: `exp`, `iat`, `sub`, `iss` (if configured)
3. No network call required (~0.1ms)

**Configuration:**
```yaml
JWKS_URL: https://<domino-host>/auth/realms/DominoRealm/protocol/openid-connect/certs
JWT_ISSUER: https://<domino-host>/auth/realms/DominoRealm
JWT_AUDIENCE: ""  # Empty = skip audience validation
```

### Principal Endpoint Fallback (Slow Path)

For API keys or when JWT validation is unavailable:

1. Credentials forwarded to `$DOMINO_HOST/v4/auth/principal`
2. Response indicates user identity and admin status
3. Network call required (~50-100ms)

**Note**: Domino returns `isAnonymous: true` for invalid API keys (not 401), so the proxy treats anonymous responses as authentication failures when credentials were explicitly provided.

## Admin Authorization

Admin endpoints (model load/unload) have additional authorization:

| Setting | Behavior |
|---------|----------|
| `ONLY_ADMIN=false` (default) | Any authenticated user can access admin endpoints |
| `ONLY_ADMIN=true` | Only Domino platform admins (`isAdmin=true`) can access |

## Environment Variables Reference

### Proxy Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SKIP_AUTH` | Disable all authentication | `false` |
| `AUTH_OPTIONAL` | Allow unauthenticated requests, validate if credentials provided | `false` |
| `DOMINO_HOST` | Domino platform URL for principal endpoint | Required |
| `JWKS_URL` | Keycloak JWKS endpoint for JWT validation | Optional |
| `JWT_ISSUER` | Expected JWT issuer claim | Optional |
| `JWT_AUDIENCE` | Expected JWT audience claim (empty = skip) | Optional |
| `ONLY_ADMIN` | Restrict admin endpoints to Domino admins | `false` |

### Client Configuration

| Variable | Description | Set By |
|----------|-------------|--------|
| `DOMINO_USER_TOKEN` | Manual Bearer token override | User (testing) |
| `DOMINO_API_PROXY` | Domino API proxy URL | Domino (automatic) |
| `DOMINO_USER_API_KEY` | API key for authentication | User or Domino |

## Troubleshooting

### "Token expired" Error

Token has expired. Inside Domino, `invalidate_token()` will fetch a fresh one. Outside Domino, obtain a new token.

### "Invalid credentials" Error

- For Bearer tokens: Token signature invalid or claims don't match
- For API keys: Key not recognized by Domino

### "missing Authorization or X-Domino-Api-Key header" Error

No credentials provided and proxy is in required auth mode. Either:
- Provide credentials
- Set `AUTH_OPTIONAL=true` for development

### Dashboard Shows No Models (401 errors in logs)

Proxy is requiring auth but dashboard isn't sending credentials:
- For local development: Set `AUTH_OPTIONAL=true` in docker-compose.yml
- For production: Ensure dashboard has valid credentials configured
