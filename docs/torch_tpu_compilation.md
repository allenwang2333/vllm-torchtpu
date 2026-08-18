# TorchTPU Compilation Cache

To enable persistent native C++ compilation caching in `vllm-torchtpu`, set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a persistent directory path:

```bash
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/path/to/persistent_cache"
```

## Cache Architecture & Environment Variables

`vllm-torchtpu` unifies compilation caching into a single native C++ pipeline powered by TorchTPU and OpenXLA PJRT:

1. **`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`** (C++ Native PJRT Binaries & Tier-2 In-Memory FX/MLIR Cache):
   - **Purpose**: Controls the persistent root directory for native C++ PJRT compiled binary executables (`.bin` files).
   - **Scope**: Native `torch_tpu` C++ backend. Setting this variable automatically activates in-memory Tier-2 caching (for intermediate FX graph and MLIR modules) and Tier-3 persistent caching on disk.
   - **Performance**: Subsequent warm runs load compiled executables directly from disk (<1ms), bypassing recompilation.

2. **`VLLM_DISABLE_COMPILE_CACHE`** (Enforced `1` by Default on TPU):
   - **Purpose**: Disables redundant vLLM Python-level compile caches (Category 2 pickle handles and Category 3 AOTAutograd files) which were designed for GPU Inductor.
   - **Behavior**: Enforced by default on TPU. Attempting to set `VLLM_DISABLE_COMPILE_CACHE=0` will immediately fail-fast with a `RuntimeError`. All compilation caching is managed natively in C++ via Tier-2 and Tier-3.

## Compilation Debugging & Telemetry

### `VLLM_XLA_CHECK_RECOMPILATION` (Runtime Recompilation Detector)

- **Default**: `0` (Disabled)
- **Purpose**: Detects unexpected compilation during serving — such as an input shape, dtype, or code path that AOT warmup never precompiled and that stalls the engine loop on a cache miss.
- **How It Works**: Queries `torch.tpu._get_cache_stats()` and advances the cached graph count watermark at each checkpoint (including every AOT precompile bucket and runtime forward step), logging newly compiled graphs whenever a cache miss occurs.
- **Overhead**: `0` (zero runtime overhead) vs `1` (queries cache stats and logs per step).
- **Recommendation**:
  - `1`: Enable in CI and test pipelines to catch compilation leaks and un-warmed code paths.
  - `0`: Keep disabled in production and benchmarking for maximum throughput.
