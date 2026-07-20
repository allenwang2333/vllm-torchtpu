# PD Disaggregation Gap Analysis

**Date:** 2026-05-08

## Architecture Overview: Two Distinct Paths

The codebase has **two separate disaggregation implementations** that serve different deployment topologies but share no common abstraction:

| | **Path 1: In-Process Orchestrator** | **Path 2: Multi-Host KV Connector** |
|---|---|---|
| **Files** | `core/core_tpu.py`, `core/disagg_executor.py` | `distributed/tpu_connector.py`, `distributed/host_kv_shm.py`, `distributed/kv_scatter.py` |
| **Trigger** | `PREFILL_SLICES` env var | `--kv-transfer-config '{"kv_connector":"TPUConnector",...}'` |
| **Topology** | Single host, chips partitioned into P/D slices | Separate vLLM processes (possibly multi-host), connected via proxy |
| **KV Transfer** | In-process queue + direct memory via `transfer_kv_cache()` / `insert_request_with_kv_cache()` | ZMQ over TCP + shared-memory staging pool |
| **Integration** | Monkey-patches `vllm.v1.engine.core.EngineCore` via `unittest.mock.patch` | Native vLLM `KVConnectorBase_V1` plugin interface |
| **Scheduling** | Custom `_DisaggOrchestrator` with its own prefill/transfer/decode threads | vLLM's native scheduler with connector hooks |

---

## Critical Gaps

### 1. The Monkey-Patch Integration is Fragile (Path 1) -- Severity: High

`DisaggEngineCore`/`DisaggEngineCoreProc` are injected via `unittest.mock.patch`:

```python
# examples/offline_inference.py:154-155
with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), patch(
        "vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
    main(args)
```

This means:
- No platform-level registration -- vLLM doesn't know the TPU platform supports disagg
- Any code path that imports `EngineCore` before the patch is applied will use the wrong class
- The `vllm serve` CLI path doesn't apply this patch, so **Path 1 doesn't work for online serving** without manual intervention
- Patch scope is fragile -- `import` order matters, and future vLLM refactors can silently break it

### 2. Request Abort Handling is Completely Unimplemented (Path 1) -- Severity: High

`core_tpu.py:749-753`:
```python
elif request_type == EngineCoreRequestType.ABORT:
    # TODO(fhzhang): we need to keep track of which engine is processing
    # the request and finish it there.
    # owner_engine.scheduler.finish_requests(request, RequestStatus.FINISHED_ABORTED)
    pass
```

When a client cancels a request mid-flight:
- Request stays in the prefill/transfer/decode pipeline indefinitely
- KV cache blocks are never freed on either the prefill or decode engine
- Memory leaks accumulate over time
- Under load, this will eventually OOM the decode engines

### 3. Single Prefill Engine Routing (Path 1) -- Severity: Medium-High

`core_tpu.py:164`:
```python
self._prefill_engines[0].scheduler.add_request(request)
```

All requests are routed to `_prefill_engines[0]` regardless of how many prefill engines exist. If you configure `PREFILL_SLICES="4,4"` (2 prefill engines), the second one is idle. The transfer thread has a load-balancing strategy for decode engines (least-loaded), but the prefill side is hard-coded to index 0.

### 4. Missing E2E Integration Tests -- Severity: High

The CI config at `.buildkite/features/Single-Host-P-D-disaggregation.yml` references:
```
tests/e2e/test_local_disagg.py::test_disaggregated_serving
tests/e2e/test_local_disagg.py::test_disaggregated_serving_correctness
```

**This file does not exist in the repository.** The CI step is `soft_fail: true`, so it never blocks. The DCN disaggregation CI is even worse -- pure placeholders (`"to be added"`). The only test that exists is `tests/distributed/test_tpu_connector.py`, which is all mocked unit tests with no real TPU or network interaction.

### 5. The Two Paths are Not Unified -- Severity: Medium-High

Path 1 (`_DisaggOrchestrator`) and Path 2 (`TPUConnector`) solve the same problem with completely different mechanisms:

- Path 1 does direct scheduler manipulation (manually calling `scheduler.running.remove()`, `scheduler.requests.pop()`, `kv_cache_manager.free()`) -- this is reaching deep into vLLM internals
- Path 2 uses the proper `KVConnectorBase_V1` interface with `request_finished()`, `get_num_new_matched_tokens()`, etc.
- Block hash / prefix caching is handled differently in each path
- A bug fix in one path doesn't propagate to the other

For production, you need to decide which one is the canonical path and deprecate the other.

### 6. Missing `KVConnectorBase_V1` Methods (Path 2) -- Severity: Medium

The upstream interface has grown significantly. `TPUConnector` is missing implementations for:

| Method | Default Behavior | Risk |
|--------|-----------------|------|
| `get_block_ids_with_load_errors()` | Returns `set()` | Failed loads silently produce wrong output -- no error recovery |
| `shutdown()` | No-op | Leaked ZMQ sockets, shm, threads on graceful shutdown |
| `get_kv_connector_stats()` | Returns `None` | No transfer metrics for monitoring/alerting |
| `get_kv_connector_kv_cache_events()` | Returns `None` | No KV event reporting |
| `get_handshake_metadata()` / `set_xfer_handshake_metadata()` | Returns `None` | Multi-engine handshake won't work |
| `build_connector_worker_meta()` | Returns `None` | Worker->scheduler feedback loop broken |
| `update_connector_output()` | No-op | Scheduler can't react to worker-side transfer events |
| `handle_preemptions()` | Not called | Preempted requests' KV transfers aren't cleaned up |
| `build_prom_metrics()` | Returns `None` | No Prometheus instrumentation |
| `reset_cache()` | Not implemented | Prefix cache reset won't propagate to connector state |

Most of these return safe defaults today, but as vLLM's scheduler starts depending on them (especially `get_block_ids_with_load_errors`, `handle_preemptions`, and `update_connector_output`), things will silently break.

### 7. No Observability Beyond Logging -- Severity: Medium

The `_LatencyTracker` in `tpu_connector.py` writes periodic summary logs, but there are:
- No Prometheus/OpenTelemetry metrics
- No structured latency histograms
- No transfer throughput metrics
- No queue depth / backpressure visibility
- No dashboard or alerting hooks
- No way to monitor KV transfer health programmatically (the `check_health` in `DisaggExecutor` always returns healthy)

### 8. Direct Scheduler Internal Manipulation (Path 1) -- Severity: Medium

`_DisaggOrchestrator._prefill()` reaches directly into scheduler internals:
```python
prefill_engine.scheduler.running.remove(request)
prefill_engine.scheduler.encoder_cache_manager.free(request)
prefill_engine.scheduler.kv_cache_manager.free(request)
prefill_engine.scheduler.requests.pop(req_id)
```

And `_decode()` similarly:
```python
decode_engine.scheduler.running.append(vllm_request)
decode_engine.scheduler.requests[req_id] = vllm_request
```

These bypass the scheduler's own state-management invariants. Any change to the vLLM scheduler's internal data structures (which happen frequently) will break this silently. The scheduler is also being accessed from background threads with no locking -- race conditions are likely.

### 9. Thread Safety Concerns (Path 1) -- Severity: Medium

- `_DisaggOrchestrator._requests` dict is read/written from prefill, transfer, and decode threads without a lock
- `JetThread` kills the entire process (`os.kill(SIGKILL)`) on any exception -- no graceful degradation, no error reporting, no request status updates
- The decode backlog queue sizing uses HBM probing at init time, but HBM usage changes as the model runs, leading to potential OOM or underutilization

### 10. Proxy Server is a Toy Implementation -- Severity: Medium

`examples/disagg/toy_proxy_server.py`:
- Round-robin load balancing (no health checking, no latency-aware routing)
- No retry logic on prefill or decode failure
- No request timeout handling
- No connection pooling beyond httpx defaults
- No authentication between proxy and backends
- No graceful degradation if a P or D instance goes down
- The file is literally named "toy_proxy_server"

For production, you need a proper routing layer (Envoy, a custom load balancer, or vLLM's built-in disagg proxy if it evolves to support TPU).

---

## Moderate Gaps

### 11. No Graceful Degradation on KV Transfer Failure

In `TPUConnector`, when a PULL fails or times out, the request is surfaced as `done_recving` with stale/zero KV data (`tpu_connector.py:1371-1380`). The code comments acknowledge this: *"the KV contents for the affected req are zero/stale, but that's strictly better than a hang."* This is correct for preventing hangs but produces silently wrong output. A production system needs to either re-prefill locally or return an error.

### 12. No Support for `kv_role = "kv_both"` (Path 2)

The upstream connector interface supports `kv_role = "kv_both"` where a single instance acts as both producer and consumer. `TPUConnector` only supports pure producer or pure consumer mode (`is_kv_producer` boolean). This limits deployment flexibility.

### 13. Warmup is Only Worst-Case

`TPUConnectorWorker._warmup_block_sizes()` only compiles for `max_blocks`:
```python
def _warmup_block_sizes(self) -> list[int]:
    max_blocks = (max_model_len + block_size - 1) // block_size
    return [max_blocks]
```

Under dynamic shapes, shorter sequences will trigger recompilation of the XLA gather/scatter kernels on first encounter, causing latency spikes.

### 14. Memory Utilization Hack for Non-v6e

`core_tpu.py:464-465`:
```python
if device_kind != 'TPU7x':
    self.vllm_config.cache_config.gpu_memory_utilization -= 0.1
```

This 10% reduction is a hard-coded heuristic. It should be configurable or based on actual HBM profiling.

### 15. Single KV Cache Group Assumption

`core_tpu.py:202-206`:
```python
# Assume one KV cache group for now.
kv_cache_map[req_id] = (
    prefill_engine.model_executor.driver_worker.
    model_runner.get_kv_cache_for_block_ids(
        block_ids[0]), request.block_hashes)
```

Upstream vLLM now supports multiple KV cache groups (for HMA / heterogeneous memory). This hard-coded `[0]` will break if multi-group KV caching is needed.

---

## What's Solid

- **`TPUConnector`'s coordinator-mode transport** (Path 2): The rank-0 coordinator pattern with shared-memory staging, ZMQ multi-channel parallel PULL, and ipc:// side-channel for TP>1 is well-engineered. Deduplication for preempted requests, expiration sweeping, and pool-exhaustion error surfacing are all handled.
- **The fused Pallas scatter kernel** (`kv_scatter.py`): Single-dispatch multi-layer scatter with donation/aliasing is TPU-native and avoids N per-layer bridge crossings.
- **Shared-memory pool** (`host_kv_shm.py`): Clean layout, zero-copy ZMQ sends via memoryview, proper lifecycle management.
- **Unit tests for the connector**: The mocked tests in `test_tpu_connector.py` cover the state machine transitions well.
- **Latency tracking in the connector**: Phase-level (D2H, wire, unpack, H2D, insert, stage, scatter) breakdown is useful for debugging.

---

## Recommended Priorities

### P0 (Blocking for any production use)
1. Implement request abort handling in `_DisaggOrchestrator`
2. Write the missing `test_local_disagg.py` E2E tests and un-soft-fail the CI
3. Choose a canonical disagg path and invest in it (recommendation: Path 2 / `TPUConnector`, as it uses the upstream interface)

### P1 (Blocking for production quality)
4. Replace monkey-patching with proper platform registration (or deprecate Path 1)
5. Implement `shutdown()` in `TPUConnector` for clean teardown
6. Add Prometheus metrics for transfer latency, throughput, queue depth, failure rate
7. Implement `handle_preemptions()` and `get_block_ids_with_load_errors()` properly
8. Fix prefill engine routing to distribute across all configured prefill engines

### P2 (Important for operational maturity)
9. Replace the toy proxy with a production router (health checks, retries, circuit breaking)
10. Add structured error handling instead of `JetThread`'s SIGKILL
11. Fix thread safety in `_DisaggOrchestrator._requests`
12. Warm up multiple block sizes to avoid first-request recompilation spikes
13. Stop directly manipulating scheduler internals in Path 1
