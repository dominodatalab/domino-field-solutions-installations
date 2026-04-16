# Design Limitations and Failure Modes

This document describes when and why this Triton proxy architecture fails or underperforms. Understanding these limitations helps operators plan for failure scenarios and make informed deployment decisions.

---

## Table of Contents

1. [Critical Single Points of Failure](#critical-single-points-of-failure)
2. [State Management Issues](#state-management-issues)
3. [Scalability Bottlenecks](#scalability-bottlenecks)
4. [Protocol Limitations](#protocol-limitations)
5. [Authentication Concerns](#authentication-concerns)
6. [Resource Management Issues](#resource-management-issues)
7. [Observability Gaps](#observability-gaps)
8. [Feature Gaps vs Native Triton](#feature-gaps-vs-native-triton)
9. [Summary Table](#summary-table)
10. [Recommendations](#recommendations)

---

## Critical Single Points of Failure

### 1. Backend Triton Dependency

**Severity: CRITICAL**

The proxy layer has a hard dependency on backend Triton connectivity. If Triton becomes unavailable (even briefly), all inference requests fail immediately.

**What fails:**
- Auto-load mechanism calls Triton directly with no fallback
- No circuit breaker pattern implemented
- No retry with exponential backoff

**When this happens:**
- Triton pod restarts
- Network partition between proxy and Triton
- Triton OOM kills or crashes

**Impact:** Total service outage during Triton unavailability.

**Location:** `src/proxy/grpc/servicer.py:220-240`, `src/proxy/http/routes/inference.py:42-59`

**Mitigation:**
1. **Add circuit breaker** using `pybreaker` or `circuitbreaker` library:
   ```python
   from circuitbreaker import circuit

   @circuit(failure_threshold=5, recovery_timeout=30)
   async def call_triton(stub, request):
       return await stub.ModelInfer(request)
   ```
2. **Implement retry with exponential backoff** using `tenacity`:
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
   async def infer_with_retry(stub, request):
       return await stub.ModelInfer(request)
   ```
3. **Deploy multiple Triton replicas** with health-check-aware load balancing
4. **Add Kubernetes liveness/readiness probes** to detect and restart unhealthy Triton pods faster
5. **Return cached responses** for idempotent requests during brief outages (if applicable)

---

### 2. Domino Authentication Endpoint Dependency

**Severity: HIGH**

Every request validates against Domino's `/v4/auth/principal` endpoint. If Domino responds slowly or becomes unavailable, all inference fails.

**What fails:**
- Single-threaded HTTP session for auth calls
- Fixed 3-second timeout (no retry)
- No auth response caching
- Every request pays auth latency cost

**When this happens:**
- Domino platform under heavy load
- Network latency to Domino > 3 seconds
- Domino auth service maintenance

**Impact:** All inference requests fail or timeout when Domino is slow.

**Location:** `src/proxy/common/auth.py:49-69`

**Mitigation:**

#### Option A: Local JWT Validation (Recommended - Eliminates Network Calls)

You can verify JWTs locally without network calls by using the issuer's public key. Keycloak (and most OIDC providers) expose their public keys via a JWKS (JSON Web Key Set) endpoint. You fetch and cache these keys once at startup (or periodically), then verify signatures locally.

**This eliminates the auth network call entirely while still validating tokens cryptographically.**

##### Option 1: Cached JWKS (Recommended)

Fetch public keys from Keycloak's JWKS endpoint once, cache them, and verify locally:

```python
import jwt
from jwt import PyJWKClient
from functools import lru_cache
import os

# Keycloak JWKS URL pattern:
# Inside cluster: http://keycloak-http.domino-platform.svc.cluster.local:80/auth/realms/DominoRealm/protocol/openid-connect/certs
# Outside cluster: https://<domino-host>/auth/realms/DominoRealm/protocol/openid-connect/certs
JWKS_URL = os.getenv("JWKS_URL")
JWT_ISSUER = os.getenv("JWT_ISSUER")  # e.g., https://sameerw116894.cs.domino.tech/auth/realms/DominoRealm
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "")  # Empty = skip audience validation

# Cache the JWKS client
# lifespan=0 means cache forever (until pod restart) - use if keys never rotate
# lifespan=3600 means re-fetch after 1 hour - use if keys may rotate
@lru_cache(maxsize=1)
def get_jwks_client() -> PyJWKClient:
    return PyJWKClient(JWKS_URL, cache_jwk_set=True, lifespan=0)  # Cache forever

def verify_jwt(token: str) -> tuple[bool, dict | None, str | None]:
    """
    Verify JWT locally using cached public keys.
    Returns: (success, claims, error_message)
    """
    try:
        jwks_client = get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Build decode options - audience validation is optional
        decode_options = {
            "verify_exp": True,      # Verify expiration
            "verify_iat": True,      # Verify issued-at
            "verify_iss": True,      # Verify issuer
            "verify_aud": bool(JWT_AUDIENCE),  # Only verify if audience is set
            "require": ["exp", "iat", "sub"]   # Required claims
        }

        # Build decode kwargs
        decode_kwargs = {
            "algorithms": ["RS256", "ES256"],  # Keycloak typically uses RS256
            "issuer": JWT_ISSUER,
            "options": decode_options,
        }

        # Only pass audience if configured (Domino tokens have multiple audiences)
        if JWT_AUDIENCE:
            decode_kwargs["audience"] = JWT_AUDIENCE

        claims = jwt.decode(token, signing_key.key, **decode_kwargs)

        return True, claims, None

    except jwt.ExpiredSignatureError:
        return False, None, "Token expired"
    except jwt.InvalidAudienceError:
        return False, None, "Invalid audience"
    except jwt.InvalidIssuerError:
        return False, None, "Invalid issuer"
    except jwt.InvalidSignatureError:
        return False, None, "Invalid signature"
    except jwt.DecodeError as e:
        return False, None, f"Token decode error: {e}"
    except Exception as e:
        return False, None, f"Validation error: {e}"
```

**Note on Audience Validation:** Domino JWT tokens contain multiple audiences (`flyteadmin`, `domino-platform`, `toolkit-client`, etc.). Since `triton-proxy` is not in the default audience list, we skip audience validation by leaving `JWT_AUDIENCE` empty. The token is still cryptographically verified via signature, expiration, and issuer checks.

##### Hybrid Fallback: JWT with Principal Endpoint Backup (Most Resilient)

For maximum resilience, use JWT as the fast path but fall back to the Domino principal endpoint if JWKS cache expires and Keycloak is unavailable:

```python
import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError
import requests
import logging
import os

logger = logging.getLogger(__name__)

JWKS_URL = os.getenv("JWKS_URL")
JWT_ISSUER = os.getenv("JWT_ISSUER")
DOMINO_HOST = os.getenv("DOMINO_HOST")  # Fallback: http://nucleus-frontend.domino-platform:80

_jwks_client: PyJWKClient | None = None
_domino_session = requests.Session()


def _init_jwks_client():
    """Initialize JWKS client with eager cache warming."""
    global _jwks_client
    if JWKS_URL and _jwks_client is None:
        try:
            # lifespan=0 caches forever - Domino/Keycloak keys rarely rotate
            # This means after startup, JWKS endpoint is never called again
            _jwks_client = PyJWKClient(JWKS_URL, cache_jwk_set=True, lifespan=0)
            _jwks_client.get_signing_keys()  # Fetch once at startup
            logger.info(f"JWKS client initialized, keys cached permanently")
        except Exception as e:
            logger.warning(f"JWKS init failed: {e}, will use fallback")


def _verify_jwt_local(token: str) -> tuple[bool, dict | None, str | None]:
    """Fast path: verify JWT locally using cached keys."""
    if not _jwks_client:
        return False, None, "JWKS client not initialized"

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=JWT_ISSUER,
            options={"verify_exp": True, "verify_aud": False}
        )
        return True, _claims_to_principal(claims), None

    except jwt.ExpiredSignatureError:
        return False, None, "Token expired"  # Don't fallback - token is genuinely expired
    except jwt.InvalidSignatureError:
        return False, None, "Invalid signature"  # Don't fallback - token is invalid
    except PyJWKClientError as e:
        # JWKS fetch failed (Keycloak down, cache expired) - USE FALLBACK
        logger.warning(f"JWKS fetch failed: {e}, falling back to principal endpoint")
        return None, None, "jwks_unavailable"  # Signal to use fallback
    except Exception as e:
        logger.warning(f"JWT verification error: {e}, falling back to principal endpoint")
        return None, None, "jwt_error"  # Signal to use fallback


def _verify_via_principal(token: str) -> tuple[bool, dict | None, str | None]:
    """Slow path fallback: validate via Domino principal endpoint."""
    if not DOMINO_HOST:
        return False, None, "DOMINO_HOST not configured"

    try:
        url = f"{DOMINO_HOST}/v4/auth/principal"
        headers = {"Authorization": f"Bearer {token}"}
        resp = _domino_session.get(url, headers=headers, timeout=5.0)

        if resp.status_code == 200:
            principal = resp.json()
            return True, principal, None
        else:
            return False, None, f"Principal endpoint returned {resp.status_code}"

    except requests.RequestException as e:
        return False, None, f"Principal endpoint error: {e}"


def validate_token(token: str) -> tuple[bool, dict | None, str | None]:
    """
    Validate token with JWT fast path and principal endpoint fallback.

    Flow:
    1. Try JWT local validation (fast, ~0.1ms)
    2. If JWKS unavailable (Keycloak down), fall back to principal endpoint (~50-100ms)
    3. If token is expired/invalid, reject immediately (no fallback)
    """
    if token.startswith("Bearer "):
        token = token[7:]

    # Fast path: JWT local validation
    success, principal, error = _verify_jwt_local(token)

    if success:
        return True, principal, None

    if error in ("jwks_unavailable", "jwt_error"):
        # Infrastructure issue - try fallback
        logger.info("Using principal endpoint fallback")
        return _verify_via_principal(token)

    # Token is genuinely invalid (expired, bad signature) - don't fallback
    return False, None, error


def _claims_to_principal(claims: dict) -> dict:
    """Map JWT claims to Domino principal format."""
    user_groups = claims.get("user_groups", [])
    return {
        "id": claims.get("sub"),
        "canonicalName": claims.get("preferred_username"),
        "email": claims.get("email"),
        "isAdmin": "/roles/SysAdmin" in user_groups,
        "user_groups": user_groups,
    }
```

**Fallback Logic:**

| Scenario | Action | Latency |
|----------|--------|---------|
| JWT valid, JWKS cached | Local validation | ~0.1ms |
| JWT expired | Reject (no fallback) | ~0.1ms |
| JWT invalid signature | Reject (no fallback) | ~0.1ms |
| Startup: Keycloak available | Fetch keys once, cache forever | ~50-100ms (once) |
| Startup: Keycloak down | **Fallback to principal endpoint for all requests** | ~50-100ms |
| Both Keycloak and Domino down at startup | Reject all | Fail |

**With `lifespan=0` (cache forever):**
- Keys are fetched **once at startup**
- After startup, Keycloak is never called again
- If Keycloak is down at startup, fallback handles all requests until pod restart
- Pod restart re-fetches keys

**Why not always fallback?**
- Expired/invalid tokens should fail fast, not retry via slower path
- Only infrastructure failures (JWKS unavailable) warrant fallback
- Prevents attackers from using fallback to bypass JWT validation

##### Option 2: Pre-configured Public Key (Zero Network Calls)

If you want zero network calls even at startup, embed the public key:

```python
import jwt
import os

# Public key can be:
# 1. Environment variable (base64 encoded)
# 2. Mounted file from Kubernetes secret
# 3. ConfigMap

PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY") or open("/etc/secrets/jwt-public-key.pem").read()

def verify_jwt_static_key(token: str) -> tuple[bool, dict | None, str | None]:
    """Verify JWT using pre-configured public key."""
    try:
        claims = jwt.decode(
            token,
            PUBLIC_KEY,
            algorithms=["RS256"],
            audience=os.getenv("JWT_AUDIENCE"),
            issuer=os.getenv("JWT_ISSUER"),
            options={"verify_exp": True, "verify_aud": True, "verify_iss": True}
        )
        return True, claims, None
    except jwt.PyJWTError as e:
        return False, None, str(e)
```

**Kubernetes deployment:**
```yaml
volumes:
  - name: jwt-public-key
    secret:
      secretName: keycloak-jwt-public-key
volumeMounts:
  - name: jwt-public-key
    mountPath: /etc/secrets/jwt-public-key.pem
    subPath: public-key.pem
    readOnly: true
```

##### Option 3: Hybrid - JWKS with Fallback

Best of both worlds - try cached JWKS, fall back to static key:

```python
_jwks_client: PyJWKClient | None = None
_static_key: str | None = None

async def init_jwt_verification():
    """Initialize at startup - try JWKS, fall back to static."""
    global _jwks_client, _static_key

    jwks_url = os.getenv("JWKS_URL")
    static_key_path = os.getenv("JWT_PUBLIC_KEY_PATH")

    if jwks_url:
        try:
            _jwks_client = PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=3600)
            # Warm the cache
            _jwks_client.get_signing_keys()
            logger.info("JWT verification using JWKS")
        except Exception as e:
            logger.warning(f"JWKS init failed: {e}, trying static key")

    if static_key_path and os.path.exists(static_key_path):
        _static_key = open(static_key_path).read()
        logger.info("JWT verification using static public key")

    if not _jwks_client and not _static_key:
        raise RuntimeError("No JWT verification method configured")
```

##### Environment Variables

```yaml
env:
  # JWKS URL for fetching Keycloak public keys (fetched once, cached for 1 hour)
  # Inside Domino cluster:
  - name: JWKS_URL
    value: "http://keycloak-http.domino-platform.svc.cluster.local:80/auth/realms/DominoRealm/protocol/openid-connect/certs"

  # Expected issuer (must match 'iss' claim in JWT exactly)
  - name: JWT_ISSUER
    value: "https://your-domino-host.cs.domino.tech/auth/realms/DominoRealm"

  # Audience validation (optional - leave empty to skip)
  # Domino tokens have multiple audiences: flyteadmin, domino-platform, toolkit-client, etc.
  # Since triton-proxy is not in the default audience list, we skip validation
  - name: JWT_AUDIENCE
    value: ""  # Empty = skip audience validation
```

**Pod Label Requirement:** Pods accessing Keycloak inside Domino must have:
```yaml
labels:
  keycloak-client: "true"
```

##### Trade-offs

| Approach | Network Calls | Key Rotation | Complexity |
|----------|---------------|--------------|------------|
| Current (Domino API) | Every request | N/A | Low |
| JWKS (cached) | Once per hour | Automatic | Medium |
| Static public key | Never | Manual redeploy | Low |
| Hybrid | Once per hour (fallback: never) | Automatic with fallback | Medium |

##### Extracting Claims

Domino JWT token claims structure:

```python
claims = {
    "sub": "69a870b1c5cd3148ba427058",      # User ID (Domino user ID)
    "preferred_username": "integration-test", # Username
    "email": "user@dominodatalab.com",
    "given_name": "John",
    "family_name": "Smith",
    "iss": "https://your-host.cs.domino.tech/auth/realms/DominoRealm",  # Issuer
    "aud": ["flyteadmin", "domino-platform", "toolkit-client", ...],    # Multiple audiences
    "azp": "domino-play",                   # Authorized party
    "exp": 1774628347,                      # Expiration (Unix timestamp)
    "iat": 1774628047,                      # Issued at
    "realm_access": {
        "roles": ["offline_access", "uma_authorization", "default-roles-dominorealm"]
    },
    "resource_access": {
        "toolkit-client": {"roles": ["toolkit_admin"]},
        "grafana-client": {"roles": ["grafana_admin"]},
    },
    "user_groups": [                        # Domino-specific: user roles/groups
        "/roles/Practitioner",
        "/roles/SysAdmin",                  # Check this for admin access
        "/domino-irsa-roles/..."
    ],
}

# Map to principal format
# NOTE: Domino user roles are in 'user_groups' claim, not 'realm_access.roles'
principal = {
    "id": claims["sub"],
    "canonicalName": claims["preferred_username"],
    "email": claims["email"],
    "isAdmin": "/roles/SysAdmin" in claims.get("user_groups", []),
    "isPractitioner": "/roles/Practitioner" in claims.get("user_groups", []),
    "user_groups": claims.get("user_groups", []),  # Contains /roles/*, /domino-irsa-roles/*
}
```

#### Option B: Auth Response Caching (Quick Fix)

If JWT migration is not feasible, add short TTL caching to reduce network calls:

```python
from cachetools import TTLCache

_auth_cache = TTLCache(maxsize=10000, ttl=60)

def validate_domino_auth(headers):
    api_key = headers.get("x-domino-api-key")
    if api_key in _auth_cache:
        return _auth_cache[api_key]
    # ... validate and cache result
    _auth_cache[api_key] = (success, principal, message)
```

#### Additional Mitigations

1. **Use connection pooling** with `httpx` async client instead of `requests.Session`
2. **Add retry logic** for transient failures (503, connection errors)
3. **Implement graceful degradation**: Allow previously-authenticated keys during brief Domino outages
4. **Add circuit breaker** for auth endpoint to fast-fail when Domino is down

---

### 3. No Circuit Breaker

**Severity: HIGH**

There is no protection against cascading failures. When Triton is unhealthy, all requests continue to hit it rather than fast-failing.

**What happens:**
- Unhealthy Triton receives 100% of requests
- Timeouts pile up (default 60s per request)
- Thread/connection pool exhaustion
- Proxy itself becomes unresponsive

**Better behavior:** After N consecutive failures, open circuit and fast-fail new requests for a cooldown period.

**Mitigation:**
1. **Implement circuit breaker pattern** in `src/proxy/common/`:
   ```python
   # circuit_breaker.py
   from circuitbreaker import CircuitBreaker

   triton_breaker = CircuitBreaker(
       failure_threshold=5,
       recovery_timeout=30,
       expected_exception=Exception
   )

   # In servicer.py
   @triton_breaker
   async def call_triton_backend(stub, request):
       return await stub.ModelInfer(request)
   ```
2. **Add health check endpoint** that reflects circuit state
3. **Expose circuit state as Prometheus metric** for alerting
4. **Consider per-model circuit breakers** to isolate failures
5. **Alternative: Use Istio/Linkerd** service mesh which provides circuit breaking out-of-the-box

---

## State Management Issues

### 4. File-Based State Race Conditions

**Severity: HIGH**

When multiple proxy replicas share a filesystem (NFS, EFS), file-based state management has race conditions.

**What fails:**
- Concurrent writes to same model state file
- No distributed locking mechanism
- Read-modify-write operations are not atomic

**Consequences:**
- Access counts become incorrect
- Last access times diverge between proxies
- LRU eviction decisions become inconsistent
- Model unloaded on one pod while another thinks it's loaded

**When this happens:**
- Multiple proxy pods with shared persistent volume
- High request rate to same model

**Location:** `src/proxy/common/state.py:133-172`

**Mitigation:**
1. **Use Redis as primary state store** (already implemented as `RedisStateClient`):
   ```yaml
   # Set environment variable to use Redis
   REDIS_URL: "redis://redis:6379/0"
   ```
2. **Add file locking for file-based state** using `fcntl`:
   ```python
   import fcntl

   def _persist_state_sync(self, state):
       with open(state_file, "r+") as f:
           fcntl.flock(f.fileno(), fcntl.LOCK_EX)
           try:
               # read, modify, write
           finally:
               fcntl.flock(f.fileno(), fcntl.LOCK_UN)
   ```
3. **Use atomic rename pattern** (already partially implemented, but verify NFS compatibility)
4. **For single-replica deployments**: File-based is fine; document this constraint
5. **For multi-replica**: Mandate Redis or implement distributed locking with etcd/Consul

---

### 5. Redis SPOF for Cordon State

**Severity: HIGH**

While Redis adds redundancy, it becomes its own single point of failure for critical operations.

**What fails when Redis is unavailable:**
- `is_cordoned()` returns `False` by default
- Cordoned models incorrectly receive new requests
- LRU eviction returns empty candidates (no eviction happens)
- Admin's intent to drain traffic is violated

**Consequences:**
- Models marked for maintenance receive traffic
- GPU memory pressure as eviction stops
- Potential OOM kills

**Location:** `src/proxy/common/state_client.py:446-478`

**Mitigation:**
1. **Deploy Redis with high availability** (Redis Sentinel or Redis Cluster):
   ```yaml
   # Helm values for Redis HA
   redis:
     architecture: replication
     sentinel:
       enabled: true
       masterSet: mymaster
   ```
2. **Change default behavior**: Return `True` (cordoned) on Redis failure for safety:
   ```python
   async def is_cordoned(self, model_name: str) -> bool:
       try:
           return await self._check_redis_cordon(model_name)
       except RedisError:
           logger.warning("Redis unavailable, assuming cordoned for safety")
           return True  # Fail-safe: reject requests when uncertain
   ```
3. **Add local persistent fallback**: Write cordon state to local file as backup
4. **Implement health checks** that mark proxy unhealthy when Redis is down
5. **Add Prometheus alerts** for Redis connectivity issues

---

### 6. Async/Sync Dual State Paths

**Severity: MEDIUM-HIGH**

HTTP and gRPC proxies use different state management code paths with different semantics.

| Proxy | State Access | Locking |
|-------|--------------|---------|
| HTTP | In-memory with async lock | `asyncio.Lock()` |
| gRPC | Reads from disk each call | No cross-process lock |

**What fails:**
- HTTP proxy has cached state invisible to gRPC proxy
- Both proxies can read old state, overwrite with stale values
- No cross-proxy synchronization

**Location:** `src/proxy/common/state.py:451-556`

**Mitigation:**
1. **Unify on Redis** as single source of truth for all proxies:
   ```python
   # Both HTTP and gRPC should use RedisStateClient
   state_manager = get_state_manager()  # Returns Redis client when REDIS_URL set
   ```
2. **Remove in-memory caching** or add cache invalidation via Redis pub/sub:
   ```python
   # Subscribe to state changes
   async def listen_for_invalidations():
       pubsub = redis_client.pubsub()
       await pubsub.subscribe("state_invalidations")
       async for message in pubsub.listen():
           model_name = message["data"]
           self._states.pop(model_name, None)
   ```
3. **Add version numbers** to state objects for optimistic locking
4. **Document constraint**: Use Redis for multi-proxy deployments

---

### 7. Fire-and-Forget Operations Lose Data

**Severity: MEDIUM**

`mark_accessed()` operations use fire-and-forget patterns for performance.

```python
asyncio.create_task(self._mark_accessed_impl(model_name, principal))
```

**What fails:**
- Proxy crash before async completes: access time not recorded
- After crash: model shows old access time, gets incorrectly evicted
- High throughput: unbounded task queue growth
- Graceful shutdown (5s grace): thousands of pending operations cancelled

**Location:** `src/proxy/common/state_client.py:259-292`

**Mitigation:**
1. **Track pending tasks and flush on shutdown**:
   ```python
   class StateManager:
       def __init__(self):
           self._pending_tasks: set = set()

       async def mark_accessed(self, model_name):
           task = asyncio.create_task(self._mark_accessed_impl(model_name))
           self._pending_tasks.add(task)
           task.add_done_callback(self._pending_tasks.discard)

       async def flush_pending(self):
           if self._pending_tasks:
               await asyncio.gather(*self._pending_tasks, return_exceptions=True)
   ```
2. **Add bounded queue with backpressure**:
   ```python
   from asyncio import Queue

   _update_queue: Queue = Queue(maxsize=10000)

   async def mark_accessed(self, model_name):
       try:
           self._update_queue.put_nowait((model_name, time.time()))
       except asyncio.QueueFull:
           logger.warning("State update queue full, dropping update")
   ```
3. **Batch updates** in background worker (see #10)
4. **Increase shutdown grace period** to 30 seconds for high-traffic deployments
5. **Add metric** for dropped updates to monitor backpressure

---

### 8. Cordon Cache 30-Second TTL

**Severity: MEDIUM**

Local cordon cache with 30-second TTL causes propagation delays.

**Scenario:**
1. Admin cordons model on Proxy A (updates Redis + local cache)
2. Proxy B updates its cache from Redis
3. Proxy C still has old cached value for up to 30 seconds
4. Proxy C continues accepting requests for cordoned model

**Impact:** Up to 30-second delay before cordon takes effect across all proxies.

**Location:** `src/proxy/common/state_client.py:446-479`

**Mitigation:**
1. **Reduce TTL to 5 seconds** (trade-off: more Redis calls):
   ```python
   self._cordon_cache_ttl = float(os.getenv("CORDON_CACHE_TTL_SECS", "5"))
   ```
2. **Use Redis pub/sub for instant invalidation**:
   ```python
   # When cordoning
   async def set_cordoned(self, model_name, cordoned):
       await redis_client.hset(key, "cordoned", str(cordoned))
       await redis_client.publish("cordon_changes", model_name)

   # Background listener invalidates local cache
   async def _listen_cordon_changes(self):
       pubsub = redis_client.pubsub()
       await pubsub.subscribe("cordon_changes")
       async for msg in pubsub.listen():
           self._cordon_cache.pop(msg["data"], None)
   ```
3. **Check Redis on every request** for critical paths (slower but consistent)
4. **Document the delay** so operators know to wait 30s after cordoning before maintenance

---

## Scalability Bottlenecks

### 9. Global State Lock Contention

**Severity: MEDIUM**

All state operations serialize on a single `asyncio.Lock()`.

**What fails at high throughput:**
- Every `mark_accessed()` acquires the same lock
- 1000+ concurrent requests serialize on lock acquisition
- Latency spikes as requests queue on lock

**Location:** `src/proxy/common/state.py:94-104`

**Mitigation:**
1. **Use per-model locks** instead of global lock:
   ```python
   class ModelStateManager:
       def __init__(self):
           self._locks: Dict[str, asyncio.Lock] = {}
           self._locks_lock = asyncio.Lock()  # Only for creating new locks

       async def _get_lock(self, model_name: str) -> asyncio.Lock:
           if model_name not in self._locks:
               async with self._locks_lock:
                   if model_name not in self._locks:
                       self._locks[model_name] = asyncio.Lock()
           return self._locks[model_name]
   ```
2. **Use lock-free data structures** for counters:
   ```python
   from collections import defaultdict
   import atomics  # or use Redis INCR which is atomic

   _access_counts = defaultdict(lambda: atomics.atomic(width=8, atype=atomics.INT))
   ```
3. **Move hot path to Redis** which handles concurrency natively
4. **Sample access updates** at high throughput (update every Nth request)

---

### 10. Redis Pipeline Per Request

**Severity: MEDIUM**

Each `mark_accessed()` call executes a Redis pipeline immediately rather than batching.

```python
pipe = redis_client.pipeline()
pipe.hincrby(key, "access_count", 1)
# ... more operations
await pipe.execute()  # Every request = 1 Redis round-trip
```

**At 1000 req/s:** 1000 Redis round-trips per second, even with pipelining.

**Better:** Buffer N updates, send 1 pipeline per batch window.

**Location:** `src/proxy/common/state_client.py:280-287`

**Mitigation:**
1. **Implement batched updates** with background worker:
   ```python
   class BatchedStateUpdater:
       def __init__(self, flush_interval=1.0, max_batch_size=100):
           self._pending: Dict[str, AccessUpdate] = {}
           self._flush_interval = flush_interval
           self._max_batch_size = max_batch_size

       async def mark_accessed(self, model_name: str):
           self._pending[model_name] = AccessUpdate(
               count=self._pending.get(model_name, AccessUpdate()).count + 1,
               timestamp=time.time()
           )
           if len(self._pending) >= self._max_batch_size:
               await self._flush()

       async def _flush(self):
           if not self._pending:
               return
           pipe = redis_client.pipeline()
           for model_name, update in self._pending.items():
               pipe.hincrby(f"model:{model_name}", "access_count", update.count)
               pipe.hset(f"model:{model_name}", "last_access_time", update.timestamp)
           await pipe.execute()
           self._pending.clear()

       async def run_periodic_flush(self):
           while True:
               await asyncio.sleep(self._flush_interval)
               await self._flush()
   ```
2. **Use Redis Streams** for append-only access log (process in batch)
3. **Coalesce updates** for same model within batch window

---

### 11. No Backpressure for State Updates

**Severity: MEDIUM**

High-throughput scenarios can overwhelm state update queues.

**What fails:**
- Each request creates a new async task
- Task queue grows unbounded
- Memory exhaustion under sustained load
- No throttling or sampling

**Location:** `src/proxy/grpc/servicer.py:243-244`

**Mitigation:**
1. **Use bounded queue** (see #7 mitigation)
2. **Implement sampling** at high throughput:
   ```python
   import random

   _request_count = 0
   _sample_rate = 1.0  # Adjust dynamically

   async def mark_accessed_sampled(self, model_name: str):
       global _request_count, _sample_rate
       _request_count += 1

       # Adjust sample rate based on queue depth
       if self._update_queue.qsize() > 5000:
           _sample_rate = 0.1  # Only 10% of updates
       elif self._update_queue.qsize() < 1000:
           _sample_rate = 1.0

       if random.random() < _sample_rate:
           await self.mark_accessed(model_name)
   ```
3. **Add metrics** for queue depth and dropped updates
4. **Set memory limits** on proxy container to prevent node-level impact

---

### 12. Unbounded Channel Pool

**Severity: MEDIUM**

gRPC channels to Triton pods are cached forever in a dictionary.

**What fails:**
- With placement-aware routing: one channel per Triton pod
- Pods scale up/down: stale channels accumulate
- No cleanup, no connection health checks
- Dead channels used until timeout

**Location:** `src/proxy/grpc/servicer.py:70-87`

**Mitigation:**
1. **Use LRU cache with TTL** for channels:
   ```python
   from cachetools import TTLCache

   class GRPCInferenceServiceServicer:
       def __init__(self):
           self._channels = TTLCache(maxsize=100, ttl=300)  # 5 min TTL
   ```
2. **Add health checking** to channels:
   ```python
   async def _get_channel(self, url: str):
       if url in self._channels:
           channel = self._channels[url]
           try:
               # Quick health check
               state = channel.get_state(try_to_connect=False)
               if state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                   await channel.close()
                   del self._channels[url]
           except:
               del self._channels[url]

       if url not in self._channels:
           self._channels[url] = aio.insecure_channel(url, options=self.CHANNEL_OPTIONS)
       return self._channels[url]
   ```
3. **Periodic cleanup** of unused channels
4. **Use connection pooling library** like `grpcio-channelz` for better visibility

---

### 13. No Model Metadata Caching

**Severity: MEDIUM**

Every metadata/config request forwards to Triton rather than caching.

**What fails:**
- Metadata rarely changes during model lifetime
- Unnecessary latency on every metadata request
- Unnecessary load on Triton backend

**Location:** `src/proxy/grpc/servicer.py:173-191`

**Mitigation:**
1. **Add TTL cache** for metadata:
   ```python
   from cachetools import TTLCache

   _metadata_cache = TTLCache(maxsize=1000, ttl=60)  # 1 min TTL

   async def ModelMetadata(self, request, context):
       cache_key = f"{request.name}:{request.version}"
       if cache_key in _metadata_cache:
           return _metadata_cache[cache_key]

       response = await stub.ModelMetadata(request)
       _metadata_cache[cache_key] = response
       return response
   ```
2. **Invalidate on model load/unload** events
3. **Add cache-control headers** for HTTP proxy responses
4. **Consider longer TTL** (5 min) since metadata rarely changes

---

## Protocol Limitations

### 14. No Request Cancellation Propagation

**Severity: MEDIUM**

Client cancellation is not propagated to the Triton backend.

**Scenario:**
- Client starts inference (60s timeout)
- Client disconnects at 30s
- Proxy continues waiting for Triton
- Triton continues processing cancelled request

**Impact:** Wasted GPU resources on cancelled work.

**Location:** `src/proxy/grpc/servicer.py:195-255`

**Mitigation:**
1. **Check context.cancelled() periodically**:
   ```python
   async def ModelInfer(self, request, context):
       task = asyncio.create_task(stub.ModelInfer(request))

       while not task.done():
           if context.cancelled():
               task.cancel()
               # Note: Triton may still complete the request
               raise grpc.aio.AbortError(grpc.StatusCode.CANCELLED)
           await asyncio.sleep(0.1)

       return await task
   ```
2. **Use gRPC deadline propagation**:
   ```python
   # Forward client deadline to Triton
   deadline = context.deadline()
   if deadline:
       remaining = deadline - time.time()
       response = await stub.ModelInfer(request, timeout=remaining)
   ```
3. **Implement Triton-side cancellation** (requires Triton support, limited)
4. **Document limitation**: GPU work cannot be interrupted mid-inference

---

### 15. Streaming Has No Flow Control

**Severity: MEDIUM**

For streaming requests, cordon checks happen after receiving each request in the stream.

**What fails:**
- Client sends 1000 requests in burst
- All requests received before cordon check fires
- Too late to reject gracefully

**Location:** `src/proxy/grpc/servicer.py:257-298`

**Mitigation:**
1. **Check cordon before accepting stream**:
   ```python
   async def ModelStreamInfer(self, request_iterator, context):
       # Get model name from first message, check cordon
       first_request = await request_iterator.__anext__()
       model_name = first_request.model_name

       if await model_manager.is_cordoned(model_name):
           context.abort(grpc.StatusCode.UNAVAILABLE, "Model cordoned")
           return

       # Proceed with stream
       async def wrapped_iterator():
           yield first_request
           async for request in request_iterator:
               yield request

       async for response in stub.ModelStreamInfer(wrapped_iterator()):
           yield response
   ```
2. **Implement flow control** with bounded buffer:
   ```python
   from asyncio import Queue

   buffer = Queue(maxsize=10)  # Only buffer 10 requests
   # Client blocks when buffer full
   ```
3. **Add periodic cordon checks** within stream (every N requests)

---

### 16. Auto-Load TOCTOU Race

**Severity: MEDIUM**

Time-of-check to time-of-use race in auto-load logic.

```python
ready_response = await stub.ModelReady(request)  # Check
if not ready_response.ready:
    await triton_control.load_model(model_name)  # Load
# Model could unload between check and inference
```

**Location:** `src/proxy/grpc/servicer.py:220-240`

**Mitigation:**
1. **Retry loop with exponential backoff**:
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
   async def infer_with_auto_load(self, model_name, request):
       ready = await stub.ModelReady(ModelReadyRequest(name=model_name))
       if not ready.ready:
           await triton_control.load_model(model_name)

       try:
           return await stub.ModelInfer(request)
       except grpc.RpcError as e:
           if "not found" in str(e) or "not ready" in str(e):
               raise  # Will trigger retry
           raise
   ```
2. **Pin frequently-used models** to prevent unloading during inference
3. **Use model state locking** during inference:
   ```python
   async with model_manager.inference_lock(model_name):
       # Model won't be evicted while lock held
       return await stub.ModelInfer(request)
   ```

---

## Authentication Concerns

### 17. TLS Verification Disabled

**Severity: CRITICAL (Security)**

Auth calls disable TLS certificate verification.

```python
resp = _session.get(url, headers=headers, timeout=3.0, verify=False)
```

**Impact:** Vulnerable to man-in-the-middle attacks on untrusted networks.

**Location:** `src/proxy/common/auth.py:61`

**Mitigation:**
1. **Enable TLS verification** with configurable CA bundle:
   ```python
   import os

   CA_BUNDLE = os.getenv("DOMINO_CA_BUNDLE", True)  # True = system CAs

   resp = _session.get(url, headers=headers, timeout=3.0, verify=CA_BUNDLE)
   ```
2. **Add environment variable** for custom CA:
   ```yaml
   env:
     - name: DOMINO_CA_BUNDLE
       value: "/etc/ssl/certs/domino-ca.crt"
   ```
3. **Mount CA certificate** in Kubernetes:
   ```yaml
   volumes:
     - name: ca-cert
       secret:
         secretName: domino-ca-cert
   volumeMounts:
     - name: ca-cert
       mountPath: /etc/ssl/certs/domino-ca.crt
       subPath: ca.crt
   ```
4. **Document secure deployment** requirements in runbook

---

### 18. gRPC Auth Creates New Session Per Interceptor

**Severity: MEDIUM**

gRPC interceptor creates its own HTTP session rather than sharing the connection pool.

**What fails:**
- Each gRPC auth call creates new TCP connection
- Connection establishment overhead (TCP + TLS handshake)
- Should reuse HTTP connection pool

**Location:** `src/proxy/grpc/servicer.py:198` vs `src/proxy/common/auth.py:24`

**Mitigation:**
1. **Share session across modules**:
   ```python
   # src/proxy/common/auth.py
   _session: Optional[requests.Session] = None

   def get_session() -> requests.Session:
       global _session
       if _session is None:
           _session = requests.Session()
           adapter = requests.adapters.HTTPAdapter(pool_maxsize=100)
           _session.mount("https://", adapter)
       return _session
   ```
2. **Use httpx async client** for better async support:
   ```python
   import httpx

   _client: Optional[httpx.AsyncClient] = None

   async def get_async_client() -> httpx.AsyncClient:
       global _client
       if _client is None:
           _client = httpx.AsyncClient(
               limits=httpx.Limits(max_connections=100),
               timeout=3.0
           )
       return _client
   ```
3. **Import shared session** in gRPC interceptor instead of creating new one

---

## Resource Management Issues

### 19. Unbounded State Dictionary

**Severity: MEDIUM**

The in-memory `_states` dictionary grows unbounded.

**What fails:**
- Every model ever accessed creates an entry
- Entries never cleaned up
- 1000+ models = 1000+ entries in memory forever
- Even deleted models retain state entries

**Location:** `src/proxy/common/state.py:94-105`

**Mitigation:**
1. **Use LRU cache** for state dictionary:
   ```python
   from cachetools import LRUCache

   class ModelStateManager:
       def __init__(self, max_models=1000):
           self._states = LRUCache(maxsize=max_models)
   ```
2. **Periodic sync with Triton** to remove deleted models:
   ```python
   async def cleanup_stale_state(self):
       """Remove state for models no longer in Triton repository"""
       triton_models = await self._get_triton_models()
       for model_name in list(self._states.keys()):
           if model_name not in triton_models:
               del self._states[model_name]
   ```
3. **Add TTL** for state entries (e.g., remove if not accessed in 24 hours)
4. **Monitor state size** with Prometheus metric

---

### 20. Auto-Load 300-Second Timeout

**Severity: MEDIUM**

Auto-load uses the same timeout as batch loads (default 300s).

**What fails:**
- Model load hangs (stuck GPU kernel, corrupt file)
- Request blocks for 5 minutes
- Client typically times out first (30-60s)
- Server continues spinning for remaining time

**Location:** `src/proxy/common/config.py:86`

**Mitigation:**
1. **Add separate timeout for on-demand loads**:
   ```python
   # config.py
   model_load_timeout_secs: int = int(os.getenv("MODEL_LOAD_TIMEOUT_SECS", "300"))
   model_autoload_timeout_secs: int = int(os.getenv("MODEL_AUTOLOAD_TIMEOUT_SECS", "30"))
   ```
2. **Use client deadline** for auto-load timeout:
   ```python
   async def auto_load_model(self, model_name, context):
       deadline = context.deadline()
       if deadline:
           timeout = min(deadline - time.time(), settings.model_autoload_timeout_secs)
       else:
           timeout = settings.model_autoload_timeout_secs

       return await triton_control.load_model(model_name, timeout=timeout)
   ```
3. **Return 503** immediately if model requires loading (let client retry)
4. **Document expected load times** for each model in config

---

## Observability Gaps

### 21. No Metrics for State Operations

Missing Prometheus metrics for:
- State operation latencies
- Lock contention time
- Redis operation latencies
- Eviction frequency and success rate
- Auth validation latencies

**Mitigation:**
1. **Add Prometheus metrics**:
   ```python
   from prometheus_client import Histogram, Counter

   STATE_OP_LATENCY = Histogram(
       "proxy_state_operation_seconds",
       "State operation latency",
       ["operation", "backend"]  # operation: mark_accessed, is_cordoned, etc.
   )

   REDIS_OP_LATENCY = Histogram(
       "proxy_redis_operation_seconds",
       "Redis operation latency",
       ["operation"]
   )

   EVICTION_COUNTER = Counter(
       "proxy_model_evictions_total",
       "Number of models evicted",
       ["reason"]  # lru, manual, etc.
   )

   AUTH_LATENCY = Histogram(
       "proxy_auth_validation_seconds",
       "Auth validation latency"
   )
   ```
2. **Instrument all state operations**:
   ```python
   async def mark_accessed(self, model_name):
       with STATE_OP_LATENCY.labels(operation="mark_accessed", backend="redis").time():
           await self._mark_accessed_impl(model_name)
   ```
3. **Add lock contention metrics** using lock wrapper
4. **Create Grafana dashboard** for proxy operations

---

### 22. No Distributed Tracing

**What's missing:**
- No trace IDs propagated end-to-end
- No request journey visibility: Proxy → Triton → Backend
- Difficult to debug distributed failures

**Better:** Add OpenTelemetry instrumentation.

**Mitigation:**
1. **Add OpenTelemetry instrumentation**:
   ```python
   from opentelemetry import trace
   from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
   from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

   # Instrument gRPC
   GrpcInstrumentorServer().instrument()
   GrpcInstrumentorClient().instrument()

   # Instrument FastAPI
   FastAPIInstrumentor.instrument_app(app)
   ```
2. **Propagate trace context** to Triton:
   ```python
   from opentelemetry.propagate import inject

   metadata = []
   inject(metadata)  # Adds trace headers
   response = await stub.ModelInfer(request, metadata=metadata)
   ```
3. **Configure exporter** (Jaeger, Zipkin, OTLP):
   ```python
   from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

   exporter = OTLPSpanExporter(endpoint=os.getenv("OTLP_ENDPOINT"))
   ```
4. **Add trace IDs to logs** for correlation

---

### 23. Incomplete Error Responses

Error responses lose root cause details.

```python
except httpx.RequestError as e:
    raise HTTPException(status_code=502, detail=f"Failed to connect: {e}")
```

Was it DNS failure? Connection refused? Timeout? Client gets generic message.

**Mitigation:**
1. **Classify errors** and return specific codes:
   ```python
   import httpx

   try:
       response = await client.post(url, ...)
   except httpx.ConnectTimeout:
       raise HTTPException(504, detail="Triton connection timeout")
   except httpx.ConnectError as e:
       if "Name or service not known" in str(e):
           raise HTTPException(503, detail="Triton DNS resolution failed")
       raise HTTPException(502, detail=f"Triton connection refused: {e}")
   except httpx.ReadTimeout:
       raise HTTPException(504, detail="Triton inference timeout")
   ```
2. **Add error codes** for programmatic handling:
   ```python
   raise HTTPException(
       status_code=502,
       detail={
           "error_code": "TRITON_CONNECTION_REFUSED",
           "message": "Failed to connect to Triton backend",
           "triton_url": settings.triton_url
       }
   )
   ```
3. **Log full stack trace** server-side while returning safe message to client
4. **Include request ID** in error responses for debugging

---

## Feature Gaps vs Native Triton

| Feature | Native Triton | This Proxy | Mitigation |
|---------|---------------|------------|------------|
| Request priority/queuing | Yes | No | Forward `priority` header to Triton |
| Request scheduling | Yes | No | Use Triton's native scheduling |
| Ensemble pipelines | Yes | Pass-through only | Works via pass-through |
| Sequence batching | Yes | Pass-through only | Works via pass-through |
| Metrics aggregation | Yes | Limited | Add proxy-level metrics, combine with Triton metrics |
| Model warmup | Yes | Manual only | Add warmup endpoint or startup hook |
| Dynamic batching control | Yes | No proxy-level control | Forward batching params to Triton |

**General Mitigation:** For features that work via pass-through, document that clients should use Triton's native parameters. For missing features, evaluate whether proxy-level implementation is needed or if Triton's implementation suffices.

---

## Summary Table

| # | Issue | Severity | Category | Mitigation Effort |
|---|-------|----------|----------|-------------------|
| 1 | Triton backend SPOF | CRITICAL | Architecture | Medium (circuit breaker) |
| 2 | Auth endpoint SPOF | HIGH | Auth | Medium (caching + retry) |
| 3 | No circuit breaker | HIGH | Resilience | Low (library) |
| 4 | File-based state races | HIGH | State | Low (use Redis) |
| 5 | Redis SPOF for cordon | HIGH | State | Medium (HA + fail-safe) |
| 6 | Async/sync dual paths | MEDIUM-HIGH | State | Medium (unify on Redis) |
| 7 | Fire-and-forget data loss | MEDIUM | State | Medium (flush + queue) |
| 8 | Cordon cache 30s TTL | MEDIUM | State | Low (reduce TTL) |
| 9 | Global state lock | MEDIUM | Scalability | Medium (per-model locks) |
| 10 | Redis pipeline per request | MEDIUM | Scalability | Medium (batching) |
| 11 | No backpressure | MEDIUM | Scalability | Low (bounded queue) |
| 12 | Unbounded channel pool | MEDIUM | Resource | Low (TTL cache) |
| 13 | No metadata cache | MEDIUM | Scalability | Low (TTL cache) |
| 14 | No cancellation propagation | MEDIUM | Protocol | Medium (deadline) |
| 15 | Streaming no flow control | MEDIUM | Protocol | Medium (pre-check) |
| 16 | Auto-load TOCTOU race | MEDIUM | Concurrency | Low (retry loop) |
| 17 | TLS verify=False | CRITICAL | Security | Low (config) |
| 18 | gRPC auth session | MEDIUM | Auth | Low (share session) |
| 19 | Unbounded state dict | MEDIUM | Resource | Low (LRU cache) |
| 20 | Auto-load 300s timeout | MEDIUM | Timeouts | Low (separate timeout) |
| 21 | No state metrics | MEDIUM | Observability | Medium (prometheus) |
| 22 | No distributed tracing | MEDIUM | Observability | Medium (OpenTelemetry) |
| 23 | Incomplete errors | LOW | Observability | Low (error classes) |

---

## Recommendations

### Immediate (Critical)

1. **Replace `verify=False`** with proper TLS configuration (configurable CA bundle)
2. **Add circuit breaker** for Triton backend failures (e.g., using `circuitbreaker` library)
3. **Add auth response caching** with short TTL (30-60s) to reduce Domino dependency
4. **Fix file-based state** with distributed locking (`fcntl.flock`) or migrate to Redis-only

### Short-Term (High Priority)

1. Implement per-model locks instead of global lock
2. Add operation batching for Redis (batch 100 writes per second)
3. Implement graceful shutdown flushing for fire-and-forget operations
4. Add backpressure/sampling to `mark_accessed` at high throughput
5. Reduce cordon cache TTL to 5 seconds

### Medium-Term

1. Add Prometheus metrics for all state operations
2. Implement request cancellation checking and propagation
3. Add metadata caching with TTL
4. Implement distributed tracing (OpenTelemetry)
5. Add health checking for cached gRPC channels

### Long-Term

1. Redesign for truly distributed state (not Redis-dependent)
2. Implement request scheduling/priority support
3. Add placement-aware load balancing with active health checks
4. Consider service mesh (Istio/Linkerd) for circuit breaking and observability

---

## When to Use This Design

This architecture is well-suited for:
- Single-region deployments with low-to-moderate scale
- Teams comfortable with Kubernetes operations
- Use cases where Domino authentication is required
- Environments where Redis is already available

This architecture is **not** well-suited for:
- High-availability requirements (no circuit breaker)
- Ultra-high throughput (>1000 req/s) without modifications
- Multi-region deployments (state synchronization issues)
- Environments requiring strict audit trails (fire-and-forget loses data)
- Zero-downtime maintenance (cordon propagation delays)