# State Management

This document describes the state management system used by the Triton proxy layer, including the Redis-backed implementation for horizontal scaling.

## Overview

The proxy layer maintains state about models to support:
- **LRU Eviction**: Track last access times for automatic model unloading
- **Access Metrics**: Count requests per model for monitoring
- **Pin/Cordon**: Administrative controls for model lifecycle
- **User Attribution**: Track which users access which models

## State Data Model

Each model has the following state tracked:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model name |
| `loaded` | bool | Whether model is currently loaded in Triton |
| `last_access_time` | datetime | Last time model was accessed (for LRU) |
| `load_time` | datetime | When the model was loaded |
| `access_count` | int | Total number of inference requests |
| `load_time_seconds` | float | How long the model took to load |
| `pinned` | bool | Pinned models are never auto-evicted |
| `cordoned` | bool | Cordoned models reject new requests |
| `last_accessed_by` | string | Username of last accessor |
| `last_accessed_by_id` | string | Domino user ID of last accessor |

## State Backend Options

The system supports three state backends, selected automatically based on configuration:

### 1. Redis (Recommended for Production)

**Configuration:** Set `REDIS_URL` environment variable

```yaml
environment:
  REDIS_URL: redis://state:6379
```

**Features:**
- Shared state across multiple proxy replicas
- Sub-millisecond read latency
- Atomic operations for concurrent access
- No persistence (state syncs with Triton on startup)

**Redis Data Model:**

| Key Pattern | Type | Purpose |
|-------------|------|---------|
| `model:state:{model_name}` | HASH | All model state fields |
| `model:config:{model_name}` | HASH | Per-model config (custom TTL) |
| `model:placement:{model_name}` | SET | Triton pod names where model is loaded |
| `model:access_times` | SORTED SET | LRU index (model → timestamp) |

### 2. File-Based (Single Replica)

**Configuration:** Set `MODEL_STATE_DIR` environment variable

```yaml
environment:
  MODEL_STATE_DIR: /triton-repo/model-state
```

**Features:**
- Persists state to JSON files on disk
- Survives proxy restarts
- Not suitable for multiple replicas (race conditions)

### 3. In-Memory (Development Only)

**Configuration:** Neither `REDIS_URL` nor `MODEL_STATE_DIR` set

**Features:**
- State lost on proxy restart
- Suitable for local development only

## Backend Selection Logic

The proxy automatically selects the backend at startup:

```
1. If REDIS_URL is set → Use RedisStateClient
2. Else if MODEL_STATE_DIR is set → Use file-based ModelStateManager
3. Else → Use in-memory ModelStateManager
```

## Architecture

### Single Replica (File-Based)

```
┌─────────────┐
│   Proxy     │
│  (single)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ JSON Files  │
│ (emptyDir)  │
└─────────────┘
```

### Horizontal Scaling (Redis)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ HTTP Proxy  │  │ HTTP Proxy  │  │ gRPC Proxy  │
│ (replica 1) │  │ (replica 2) │  │ (replica N) │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   Redis State   │
              │   (ephemeral)   │
              └─────────────────┘
```

## Performance Characteristics

### Fire-and-Forget Operations

The most frequently called operation (`mark_accessed`) is **non-blocking** to avoid adding latency to inference requests:

**Async (HTTP Proxy):**
```python
async def mark_accessed(self, model_name, principal=None):
    asyncio.create_task(self._mark_accessed_impl(model_name, principal))
```

**Sync (gRPC Proxy):**
```python
def mark_accessed_sync(self, model_name, principal=None):
    self._executor.submit(self._mark_accessed_impl_sync, model_name, principal)
```

### Operation Latency

| Operation | Frequency | Blocking? | Typical Latency |
|-----------|-----------|-----------|-----------------|
| `mark_accessed()` | Every request | **No** (fire-and-forget) | 0ms added to request |
| `is_cordoned()` | Every request | **No** (cached) | 0ms (cache hit), ~0.3ms (cache miss) |
| `mark_loaded()` | Model load | Yes | ~0.5-1ms |
| `mark_unloaded()` | Model unload | Yes | ~0.5-1ms |
| `is_pinned()` | Admin calls | Yes | ~0.1-0.5ms |
| `get_eviction_candidates()` | Periodic | Yes | ~1-5ms |

### Request Path Impact

For each inference request, cordon status is checked via **local cache**:

```python
# Check if model is cordoned (uses local cache, refreshes every 30s)
is_cordoned = await model_manager.is_cordoned(model_name)  # Cache hit: 0ms
```

**Total added latency: 0ms** for 99.9% of requests (cache hits).

The cache is refreshed:
- Every 30 seconds (configurable via `REDIS_CORDON_CACHE_TTL_SECS`)
- Immediately when `set_cordoned()` is called on the same proxy

### Cordon Cache Design

To minimize Redis calls on the hot path, cordon status is cached locally:

```
Every inference request:
  └── Check local cache
        ├── Cache hit (age < 30s): Return cached value (0ms)
        └── Cache miss/expired: HGET from Redis (~0.3ms), update cache
```

**Trade-off**: Up to 30 seconds delay before other proxy replicas see a cordon change. This is acceptable because:
1. Cordon is an admin operation for graceful drain
2. Admin waits for traffic to drain before unloading
3. The proxy that sets cordon sees it immediately (local cache updated)

For a model receiving 100 req/s, this reduces Redis calls from **100/s to ~0.03/s** (3,300x reduction)

### Comparison: Redis vs File-Based

| Aspect | File-Based | Redis |
|--------|------------|-------|
| Read latency | 1-5ms (disk I/O) | 0.1-0.5ms |
| Write latency | 5-20ms (fsync) | 0.5-1ms |
| Concurrent access | File locking overhead | Native atomic ops |
| Cross-replica consistency | Race conditions | Guaranteed |
| Horizontal scaling | Not supported | Supported |

### Pipeline Optimization

Access tracking batches multiple Redis operations into a single round-trip:

```python
pipe = redis_client.pipeline()
pipe.hincrby(key, "access_count", 1)
pipe.hset(key, "last_access_time", now.isoformat())
pipe.zadd(self.ACCESS_TIMES_KEY, {model_name: now.timestamp()})
await pipe.execute()  # Single network round-trip
```

## Graceful Degradation

If Redis becomes unavailable, the system continues functioning:

| Operation | Behavior on Redis Failure |
|-----------|---------------------------|
| `mark_accessed()` | Silent fail (fire-and-forget) |
| `is_cordoned()` | Returns `False` (allows inference) |
| `is_pinned()` | Returns `False` |
| `mark_loaded()` | Logs warning, continues |
| `get_eviction_candidates()` | Returns empty list |

This ensures inference continues even during Redis outages, with the trade-off that:
- LRU tracking is temporarily lost
- Cordoned models may receive requests
- Access counts may be inaccurate

State automatically recovers when Redis comes back online via `sync_with_triton()`.

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | (none) | Redis connection URL (e.g., `redis://state:6379`) |
| `REDIS_CONNECTION_TIMEOUT_SECS` | 5.0 | Timeout for establishing Redis connection |
| `REDIS_OPERATION_TIMEOUT_SECS` | 2.0 | Timeout for Redis operations |
| `REDIS_CORDON_CACHE_TTL_SECS` | 30.0 | TTL for local cordon status cache |
| `MODEL_STATE_DIR` | (none) | Directory for file-based state persistence |
| `MODEL_IDLE_TIMEOUT_SECS` | 1800 | Default idle time before LRU eviction (30 min) |
| `TRITON_HTTP_URL_PATTERN` | (none) | HTTP URL pattern for StatefulSet routing (e.g., `http://{pod}.triton.svc:8000`) |
| `TRITON_GRPC_URL_PATTERN` | (none) | gRPC URL pattern for StatefulSet routing (e.g., `{pod}.triton.svc:8001`) |

### Docker Compose Example

```yaml
services:
  state:
    image: redis:7-alpine
    command: redis-server --save "" --appendonly no --maxmemory 128mb --maxmemory-policy noeviction
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  http-proxy:
    environment:
      REDIS_URL: redis://state:6379
    depends_on:
      state:
        condition: service_healthy

  grpc-proxy:
    environment:
      REDIS_URL: redis://state:6379
    depends_on:
      state:
        condition: service_healthy
```

### Helm Values Example

```yaml
state:
  enabled: true
  maxmemory: "128mb"
  resources:
    requests:
      cpu: "100m"
      memory: "128Mi"
    limits:
      cpu: "500m"
      memory: "256Mi"
```

## API Reference

### Async API (HTTP Proxy)

```python
from proxy.common.state import model_manager

# Track model access (fire-and-forget)
await model_manager.mark_accessed(model_name, principal)

# Check if model is cordoned
if await model_manager.is_cordoned(model_name):
    raise HTTPException(503, "Model is cordoned")

# Mark model as loaded
await model_manager.mark_loaded(model_name, load_time_secs, pinned=False)

# Get eviction candidates
candidates = await model_manager.get_eviction_candidates()

# Pin/unpin a model
await model_manager.set_pinned(model_name, True)

# Cordon/uncordon a model
await model_manager.set_cordoned(model_name, True)
```

### Sync API (gRPC Proxy)

```python
from proxy.common.state import model_manager

# Track model access (fire-and-forget via thread pool)
model_manager.mark_accessed_sync(model_name, principal)

# Check if model is cordoned
if model_manager.is_cordoned_sync(model_name):
    context.abort(grpc.StatusCode.UNAVAILABLE, "Model is cordoned")

# Mark model as loaded
model_manager.mark_loaded_sync(model_name, load_time_secs, pinned=False)
```

### Per-Model TTL API

```python
# Get per-model TTL (returns None if using global default)
ttl = await model_manager.get_model_ttl(model_name)

# Set per-model TTL (overrides global default)
await model_manager.set_model_ttl(model_name, 3600)  # 1 hour

# Remove per-model TTL (revert to global default)
await model_manager.set_model_ttl(model_name, None)
```

### Model Placement API (for Triton StatefulSet)

```python
# Get list of pods where model is loaded
pods = await model_manager.get_placement(model_name)  # ["triton-0", "triton-1"]

# Record model loaded on a pod (call after successful load)
await model_manager.add_placement(model_name, "triton-0")

# Record model unloaded from a pod
await model_manager.remove_placement(model_name, "triton-0")

# Get all placements
all_placements = await model_manager.get_all_placements()
# {"yolov8n": ["triton-0"], "bert": ["triton-0", "triton-1"]}
```

## Per-Model TTL

By default, all models use the global idle timeout (`MODEL_IDLE_TIMEOUT_SECS`, default 1800s/30min). You can override this for individual models that need different eviction policies.

### Use Cases

- **Large models**: Set longer TTL to avoid expensive reload cycles
- **Frequently accessed models**: Use global default or pin instead
- **Experimental models**: Set shorter TTL for aggressive eviction

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models/{name}/ttl` | GET | Get current TTL config |
| `/v1/models/{name}/ttl` | PUT | Set custom TTL |
| `/v1/models/{name}/ttl` | DELETE | Remove custom TTL |

### Example

```bash
# Set 1-hour TTL for a large model
curl -X PUT "$PROXY_URL/v1/models/large-llm/ttl" \
  -H "X-Domino-Api-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"idle_timeout_secs": 3600}'

# Check TTL config
curl "$PROXY_URL/v1/models/large-llm/ttl" \
  -H "X-Domino-Api-Key: $API_KEY"

# Remove custom TTL
curl -X DELETE "$PROXY_URL/v1/models/large-llm/ttl" \
  -H "X-Domino-Api-Key: $API_KEY"
```

## Model Placement (Triton StatefulSet)

When running multiple Triton replicas as a Kubernetes StatefulSet (triton-0, triton-1, etc.), placement tracking records which pods have which models loaded and routes inference requests accordingly.

### How It Works

1. Admin decides which Triton pod should load a model
2. Admin calls the specific pod directly to load the model
3. Admin updates Redis with placement info
4. **Proxy automatically routes inference requests to pods that have the model loaded**

### Enabling Placement-Aware Routing

Set URL patterns in your proxy deployment:

```yaml
environment:
  # Pattern for HTTP requests - {pod} is replaced with pod name
  TRITON_HTTP_URL_PATTERN: "http://{pod}.triton.svc:8000"
  # Pattern for gRPC requests
  TRITON_GRPC_URL_PATTERN: "{pod}.triton.svc:8001"
  # Redis for placement state
  REDIS_URL: redis://state:6379
```

When these patterns are set, the proxy will:
1. Check Redis for model placement
2. Route to one of the pods that has the model (round-robin if multiple)
3. Fall back to default `TRITON_HTTP_URL`/`TRITON_URL` if no placement found

### Routing Behavior

| Scenario | Behavior |
|----------|----------|
| No placement recorded | Uses default Triton URL |
| Model on 1 pod | Routes to that pod |
| Model on multiple pods | Round-robin across pods |
| Placement check fails | Falls back to default URL |
| URL pattern not configured | Always uses default Triton URL (routing disabled) |

### Routing Algorithm

The proxy uses **per-model round-robin** load balancing:

```
Request for model "yolov8n"
    │
    ▼
Check Redis: SMEMBERS model:placement:yolov8n
    │
    ├── Empty set → Use default TRITON_HTTP_URL
    │
    └── {"triton-0", "triton-1"} → Round-robin selection
              │
              ▼
        Request 1 → triton-0
        Request 2 → triton-1
        Request 3 → triton-0
        Request 4 → triton-1
        ...
```

**Key characteristics:**

| Aspect | Behavior |
|--------|----------|
| **Algorithm** | Round-robin per model (not global) |
| **State** | Counter maintained in proxy memory |
| **Consistency** | Different proxy replicas may route to different pods |
| **Failover** | No automatic failover - routes to pod even if unhealthy |

### Routing Decision Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Request                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ TRITON_HTTP_URL_PATTERN set?  │
              └───────────────────────────────┘
                     │              │
                    No             Yes
                     │              │
                     ▼              ▼
            ┌─────────────┐  ┌─────────────────────┐
            │ Use default │  │ Get placement from  │
            │ TRITON_URL  │  │ Redis               │
            └─────────────┘  └─────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    Empty/Error          Has pods
                         │                     │
                         ▼                     ▼
               ┌─────────────┐    ┌────────────────────┐
               │ Use default │    │ Select pod         │
               │ TRITON_URL  │    │ (round-robin)      │
               └─────────────┘    └────────────────────┘
                                           │
                                           ▼
                                  ┌─────────────────────┐
                                  │ Construct URL:      │
                                  │ pattern.replace(    │
                                  │   "{pod}", selected │
                                  │ )                   │
                                  └─────────────────────┘
```

### Performance Characteristics

| Operation | Latency | Frequency |
|-----------|---------|-----------|
| Placement lookup | ~0.3ms (Redis SMEMBERS) | Every inference request |
| URL construction | <0.01ms (string replace) | Every inference request |
| Pod selection | <0.01ms (counter increment) | Every inference request |

**Total routing overhead:** ~0.3ms per request (Redis round-trip)

**Optimization considerations:**
- Placement data is small (set of pod names) - fast Redis operation
- No caching currently - every request queries Redis
- For very high throughput, consider adding TTL cache similar to cordon cache

### gRPC Connection Pooling

For gRPC, the proxy maintains **persistent channels per Triton pod**:

```python
# Channels are created on first use and reused
self._channels = {
    "triton-0.triton.svc:8001": <gRPC Channel>,
    "triton-1.triton.svc:8001": <gRPC Channel>,
}
```

This avoids connection setup overhead on each request while supporting routing to different pods.

### Backward Compatibility

Routing is **fully backward compatible**:

| Configuration | Behavior |
|---------------|----------|
| No URL pattern set | Uses `TRITON_HTTP_URL` / `TRITON_URL` for all requests |
| Pattern set, no placement | Falls back to default URL |
| Pattern set, placement exists | Routes based on placement |

Existing deployments with single Triton instance require **no changes**.

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models/{name}/placement` | GET | Get pods where model is loaded |
| `/v1/models/{name}/placement/{pod}` | POST | Record model loaded on pod |
| `/v1/models/{name}/placement/{pod}` | DELETE | Record model unloaded from pod |
| `/v1/placements` | GET | Get all model placements |

### Example

```bash
# Load model on triton-0 (call Triton directly)
curl -X POST "http://triton-0.triton:8000/v2/repository/models/yolov8n/load"

# Record placement in Redis
curl -X POST "$PROXY_URL/v1/models/yolov8n/placement/triton-0" \
  -H "X-Domino-Api-Key: $API_KEY"

# Check where model is loaded
curl "$PROXY_URL/v1/models/yolov8n/placement" \
  -H "X-Domino-Api-Key: $API_KEY"

# View all placements
curl "$PROXY_URL/v1/placements" \
  -H "X-Domino-Api-Key: $API_KEY"
```

### Redis Data

```bash
# View placement for a model
redis-cli SMEMBERS "model:placement:yolov8n"
# Returns: "triton-0", "triton-1"

# View all placement keys
redis-cli KEYS "model:placement:*"
```

### Testing Placement Routing Locally

To test placement-aware routing with Docker Compose:

**1. Create a multi-backend compose file (`docker-compose.multibackend.yml`):**

```yaml
services:
  state:
    image: redis:7-alpine
    command: redis-server --save "" --appendonly no
    ports:
      - "6379:6379"

  triton-0:
    build:
      context: .
      dockerfile: Dockerfile.triton
    volumes:
      - ./triton-repo:/triton-repo:ro
    environment:
      MODEL_REPO: /triton-repo/models

  triton-1:
    build:
      context: .
      dockerfile: Dockerfile.triton
    volumes:
      - ./triton-repo:/triton-repo:ro
    environment:
      MODEL_REPO: /triton-repo/models

  http-proxy:
    build:
      context: .
      dockerfile: Dockerfile.proxy.http
    ports:
      - "8080:8080"
    environment:
      TRITON_HTTP_URL: http://triton-0:8000
      TRITON_HTTP_URL_PATTERN: "http://{pod}:8000"
      REDIS_URL: redis://state:6379
      SKIP_AUTH: "true"
    depends_on:
      - state
      - triton-0
      - triton-1
```

**2. Start the stack:**

```bash
docker compose -f docker-compose.multibackend.yml up -d
```

**3. Load model on triton-0 and register placement:**

```bash
# Load on triton-0
curl -X POST "http://localhost:8080/v2/repository/models/yolov8n/load"

# Register placement (pod name matches service name)
curl -X POST "http://localhost:8080/v1/models/yolov8n/placement/triton-0"
```

**4. Verify routing:**

```bash
# Check placement
curl "http://localhost:8080/v1/models/yolov8n/placement"
# Should show: {"model_name": "yolov8n", "pods": ["triton-0"], ...}

# Run inference - should route to triton-0
curl -X POST "http://localhost:8080/v2/models/yolov8n/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [...]}'

# Check proxy logs to see routing
docker compose -f docker-compose.multibackend.yml logs http-proxy | grep -i routing
```

**5. Test multi-pod routing:**

```bash
# Load model on triton-1 as well
docker compose -f docker-compose.multibackend.yml exec triton-1 \
  curl -X POST "http://localhost:8000/v2/repository/models/yolov8n/load"

# Register triton-1 placement
curl -X POST "http://localhost:8080/v1/models/yolov8n/placement/triton-1"

# Now requests will round-robin between triton-0 and triton-1
curl "http://localhost:8080/v1/models/yolov8n/placement"
# Should show: {"model_name": "yolov8n", "pods": ["triton-0", "triton-1"], ...}
```

## Monitoring

### Redis Commands for Debugging

```bash
# Check Redis connectivity
redis-cli ping

# View all model states
redis-cli KEYS "model:state:*"

# View specific model state
redis-cli HGETALL "model:state:yolov8n"

# View per-model TTL config
redis-cli HGETALL "model:config:yolov8n"

# View model placement
redis-cli SMEMBERS "model:placement:yolov8n"

# View all placements
redis-cli KEYS "model:placement:*"

# View LRU access times (oldest first)
redis-cli ZRANGE "model:access_times" 0 -1 WITHSCORES

# View eviction candidates (accessed more than 30 min ago)
redis-cli ZRANGEBYSCORE "model:access_times" -inf $(( $(date +%s) - 1800 ))
```

### Metrics to Monitor

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Redis latency | Redis INFO | > 5ms |
| Redis memory | Redis INFO | > 80% of maxmemory |
| Redis connections | Redis INFO | > 100 |
| State sync failures | Proxy logs | Any occurrence |

## Troubleshooting

### Redis Connection Failures

**Symptom:** Logs show "Failed to create Redis client"

**Solutions:**
1. Verify Redis is running: `redis-cli ping`
2. Check network connectivity between proxy and Redis
3. Verify `REDIS_URL` is correct
4. Check Redis logs for errors

### State Inconsistency

**Symptom:** Model shows as loaded in state but not in Triton (or vice versa)

**Solution:** Trigger a state sync:
```python
await model_manager.sync_with_triton(triton_http_url)
```

This happens automatically on proxy startup and can be triggered via the admin API.

### High Redis Latency

**Symptom:** Inference requests slower than expected

**Solutions:**
1. Check Redis memory usage (should be < 80%)
2. Verify Redis is on same network as proxies
3. Check for Redis slowlog: `redis-cli SLOWLOG GET 10`
4. Consider increasing `maxmemory` if needed

## Security Considerations

### Network Isolation

Redis should only be accessible from proxy pods. Use NetworkPolicy in Kubernetes:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: state-network-policy
spec:
  podSelector:
    matchLabels:
      app: state
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: proxy
    ports:
    - port: 6379
```

### No Persistence

Redis is configured with no persistence (`--save "" --appendonly no`) because:
1. State can be reconstructed from Triton on startup
2. Reduces disk I/O and complexity
3. State is ephemeral by design (LRU metrics, not critical data)

### No Authentication (Internal Only)

Redis runs without authentication because it's only accessible within the cluster. For external exposure, configure Redis AUTH.
