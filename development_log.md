# DeepSeek V4 TPU Inference Development Log

## Overview
Technical log of issue root causes, failure mechanics, and implemented code fixes for running DeepSeek V4 inference on TPU v7x-8 within the `vllm-torchtpu` framework.

## Current "Best" Command
```bash
ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa \
  "sudo docker run --init --privileged --net=host --ipc=host --rm \
  -v /home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/gdn_attention_op.py:/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py \
  -v /home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/gdn_attention_op.py:/usr/local/lib/python3.12/site-packages/vllm_torchtpu/layers/vllm/custom_ops/gdn_attention_op.py \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e VLLM_TP_SIZE=8 -e VLLM_EP_SIZE=8 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_FP4_CACHE_DIR=/dev/shm/dsv4-flash-real-fp4 \
  -e JAX_COMPILATION_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache \
  -v /dev/shm:/dev/shm -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
  bash -c 'rm -rf /tmp/ipc* /tmp/psm* /tmp/vllm* && python3 -u /home/patemotter_google_com/workspace/vllm-torchtpu/examples/run_dsv4_inference.py'"
```

---

## Technical Issues & Resolved Fixes

### 1. Custom `libtpu.so` Integration via `TPU_LIBRARY_PATH`
- **Failure Mechanics**: The container default `/usr/local/lib/python3.12/site-packages/libtpu/libtpu.so` rejected `--xla_tpu_use_dynamic_smem_negotiation`, raising `ERROR: Unknown command line flag`.
- **Resolution**: Specified `-e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so` in Docker container execution commands.

### 2. `VllmQuantConfig` Property Assertion
- **Failure Mechanics**: `tpu_inference` quantization layers queried `quant_config.mesh`. The FP4 quantization config class `VllmQuantConfig` lacked a `.mesh` getter property, raising `AttributeError`.
- **Resolution**: Added `@property def mesh(self): return None` to `VllmQuantConfig` in `src/vllm_torchtpu/layers/vllm/quantization/configs.py`.

### 3. JAX `NamedSharding` Weight Post-Processing
- **Failure Mechanics**: `VllmUnquantizedEmbeddingMethod` and `VllmUnquantizedLinearMethod` attempted JAX array sharding post-processing during PyTorch weight loading.
- **Resolution**: Monkey-patched weight post-processing in `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py` to bypass JAX sharding during PyTorch parameter initialization.

### 4. PyTorch Dynamo MoE Output Reduction Context Assertion
- **Failure Mechanics**: `VllmMoERunner._maybe_reduce_final_output` asserted wrapper context (`vmwc.get_vllm_model_wrapper_context`), which is not present during PyTorch Dynamo graph tracing.
- **Resolution**: Patched `vmwc.get_vllm_model_wrapper_context` and substituted `VllmMoERunner._maybe_reduce_final_output` with `MoERunner._maybe_reduce_final_output` in `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`.

### 5. Pallas Kernel Ceiling Division Tracing Assertion (`cdiv`)
- **Failure Mechanics**: `cdiv(a, b)` contained a raw Python `assert b != 0`. During PyTorch Dynamo FX-to-MLIR export (`torch_tpu_export.fx_to_mlir`), PyTorch passes fake tensors and zero/symbolic shape metadata through kernel parameter evaluation, causing Python `AssertionError` during graph lowering.
- **Resolution**: Updated `cdiv(a, b)` across `mla_swa.py`, `mla.py`, and `streamindex_topk.py` to use safe branch guards:
  ```python
  def cdiv(a, b):
      if isinstance(b, (int, float)):
          if b == 0:
              return 0
          return (a + b - 1) // b
      return jnp.where(b != 0, (a + jnp.maximum(1, b) - 1) // jnp.maximum(1, b), 0)
  ```

### 6. Zero-Size Axis Reduction Exception (`s_rowmax = jnp.max`)
- **Failure Mechanics**: In `flash_attention`, `s_rowmax = jnp.max(s, axis=1, keepdims=True)` evaluated `s` with shape `(bq_sz, 0)` during empty/uninitialized SWA cache shape tracing. In JAX/NumPy, maximum reduction over a 0-length axis raises `ValueError: zero-size array to reduction operation max which has no identity`.
- **Resolution**: Added explicit trace-time dimension checks in `src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py` and `src/vllm_torchtpu/kernels/deepseek_v4/mla.py`:
  ```python
  if s.shape[1] == 0:
      s_rowmax = jnp.full((s.shape[0], 1), -jnp.inf, dtype=s.dtype)
  else:
      s_rowmax = jnp.max(s, axis=1, keepdims=True)
  ```

### 7. Zero-Width Broadcast Concatenation Exception (`broadcast_minor`)
- **Failure Mechanics**: When `s.shape[1] == 0` (e.g. `s.shape = (bq_sz, 0)` during empty SWA cache tracing), `broadcast_minor(m_curr, s.shape)` calculated `target_minor = align_to(0, src.shape[-1]) = 0`. Evaluating `range(0 // src.shape[-1])` yielded an empty list `[]`, causing `jnp.concatenate([], axis=-1)` to raise `ValueError: Need at least one array to concatenate.`.
- **Resolution**: Added zero-shape handling to `broadcast_minor` in `src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py` and `src/vllm_torchtpu/kernels/deepseek_v4/mla.py`:
  ```python
  if shape[-1] == 0:
      return jnp.zeros(shape, dtype=src.dtype)
  ```

---

## Architectural Comparison: PyTorch Dynamo Symbolic Tracing vs. Native JAX Staging

### Why PyTorch Dynamo Tracing Triggers Pallas Kernel Failures

Native JAX kernels developed directly for JAX workloads rarely encounter these errors (`cdiv` assertions, zero-size axis max reductions, empty array concatenations). The structural reasons for this difference in the `vllm-torchtpu` stack are:

1. **Static Shape Staging in Native JAX (`jax.jit` / `pallas_call`)**:
   - Native JAX programs stage graphs using concrete, statically defined array metadata (e.g., `(seq_len=128, dim=4096)`).
   - Array dimensions in pure JAX are evaluated with explicit static integer bounds during compilation. Zero-length or uninitialized shape paths are not executed during tracing.

2. **Symbolic `FakeTensor` Evaluation in PyTorch-TPU (`torch_tpu_export.fx_to_mlir`)**:
   - `vLLM` V1 uses Piecewise Graph Compilation over variable sequence length ranges (e.g., `(16, 16)`, `(64, 64)`).
   - PyTorch Dynamo extracts FX graphs using `FakeTensor` objects with symbolic or zero-width shape bounds (e.g., `seq_len=0` during uninitialized SWA cache blocks or graph warm-up).
   - When `torch_tpu` exports custom FX nodes to MLIR (`EagerLikeFxInterpreter`), it calls the Python implementation of custom Pallas kernels using these zero-sized fake inputs.

3. **Tracing-Time Python Exception Trigger**:
   - Any raw Python assertion (`assert b != 0`), zero-size axis reduction (`jnp.max([])`), or empty array list (`jnp.concatenate([])`) inside a Pallas function executes in Python *during* PyTorch's FX tracing interpreter phase.
   - This aborts PyTorch graph lowering before JAX/XLA graph compilation ever receives the node.

---

### Code Examples: Native JAX vs. PyTorch FX Tracing

#### Example A: Ceiling Division (`cdiv`)
- **Native JAX Execution Path**:
  ```python
  def cdiv(a, b):
      assert b != 0  # Passed: b is static integer (e.g. 128) during jax.jit trace
      return (a + b - 1) // b
  ```
- **PyTorch FX Tracing Failure & Fix**:
  ```python
  # PyTorch FX interpreter passes b=0 / SymInt(0) during fake shape tracing
  def cdiv(a, b):
      if isinstance(b, (int, float)):
          if b == 0:
              return 0
          return (a + b - 1) // b
      return jnp.where(b != 0, (a + jnp.maximum(1, b) - 1) // jnp.maximum(1, b), 0)
  ```

#### Example B: Axis Max Reduction (`flash_attention`)
- **Native JAX Execution Path**:
  ```python
  # s.shape is concrete (bq_sz, kv_len) with kv_len > 0
  s_rowmax = jnp.max(s, axis=1, keepdims=True)  # Works cleanly
  ```
- **PyTorch FX Tracing Failure & Fix**:
  ```python
  # PyTorch FX passes empty cache shape s.shape = (bq_sz, 0)
  # JAX raises: ValueError: zero-size array to reduction operation max which has no identity
  if s.shape[1] == 0:
      s_rowmax = jnp.full((s.shape[0], 1), -jnp.inf, dtype=s.dtype)
  else:
      s_rowmax = jnp.max(s, axis=1, keepdims=True)
  ```

#### Example C: Minor Dimension Broadcast (`broadcast_minor`)
- **Native JAX Execution Path**:
  ```python
  # shape[-1] is positive head dimension (e.g. 256)
  target_minor = align_to(shape[-1], src.shape[-1])
  # range(target_minor // src.shape[-1]) yields non-empty list [src, src, ...]
  return jnp.concatenate([src for _ in range(...)], axis=-1)
  ```
- **PyTorch FX Tracing Failure & Fix**:
  ```python
  # PyTorch FX passes shape[-1] = 0 -> align_to(0, dim) = 0 -> range(0) = []
  # JAX raises: ValueError: Need at least one array to concatenate.
  if shape[-1] == 0:
      return jnp.zeros(shape, dtype=src.dtype)
  ```

---

## 8. MLIR Compiler Failure on Zero-Sized Vector Dimensions (`logical_page_size=0`)
- **Symptom**:
  ```text
  MLIRError: Invalid type: vector types must have positive constant sizes but got 0, 640
  callsite: load_bkv in _mla_sliding_window_ragged_paged_attention_kernel
  ```
- **Root Cause**:
  When `swa_cache` is `None` or uninitialized, `logical_page_size` was assigned `0` in `custom_ops/deepseek_v4.py`. Inside `mla_swa.py`, `page_size_per_kv_packing = 0 // kv_packing = 0`, producing `bkv_sz = 0`. Pallas `bkv.reshape(0, 640)` generates a 0-sized VMEM vector type. The TPU MLIR compiler requires strictly positive constant vector sizes (`dim > 0`), causing FX graph compilation to fail during MLIR lowering.
- **Fix**:
  1. Defaulted `logical_page_size = 16` in `custom_ops/deepseek_v4.py` when `swa_cache` is `None` or has `block_size <= 0`.
  2. Added explicit guard `if logical_page_size <= 0: logical_page_size = kv_packing` in `mla_sliding_window_ragged_paged_attention`.

---

## 9. `NameError: name 'load_with_init' is not defined` in `mla.py`
- **Symptom**:
  ```text
  NameError: name 'load_with_init' is not defined
  callsite: flash_attention in src/vllm_torchtpu/kernels/deepseek_v4/mla.py
  ```
- **Root Cause**:
  In `mla.py:200`, `m_prev = load_with_init(head_m_ref, -jnp.inf)` was invoked inside `flash_attention`, but `load_with_init` had not been defined (unlike `mla_swa.py` where it was defined as a nested closure).
- **Fix**:
  Defined `is_first_bkv = (bkv_idx == 0)` and `def load_with_init(ref, init_val): return jnp.where(is_first_bkv, jnp.full_like(ref[...], init_val), ref[...])` inside `flash_attention` in `mla.py`, and updated `l_prev` and `o_prev` to use `load_with_init`.

---

## 10. `TypeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)` in `mla.py`
- **Symptom**:
  ```text
  TypeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)
  While executing %deepseek_v4_attention_csa
  ```
- **Root Cause**:
  In `mla.py`, `attention_sinks` was constructed by concatenating `attention_sinks_ref[...]` over fixed `range(bq_sz)` (yielding `bq_sz * num_q_heads` = `8 * 8 = 64` rows, shape `(64, 1)`). When `q.shape[0]` corresponds to a partial query tile (e.g. `sz = 1` token, `q.shape[0] = 8`), `m_ref` sliced by `[:q.shape[0]]` has shape `(8, 128)`. Performing `attention_sinks - m_ref[...]` attempted to subtract `(64, 1)` - `(8, 128)`, causing shape mismatch during graph compilation.
- **Fix**:
  1. Preserved static tile shape allocations (`bq_sz * num_q_heads` = 64 rows) for all VMEM registers (`acc_ref[...]`, `m_ref[...]`, `l_ref[...]`, `attention_sinks`, `kv_lens_to_attend_segment`) inside `compute_with_bq`.
  2. Used JAX `lax.dynamic_slice_in_dim` to extract static size `bq_sz` slices from `kv_lens_to_attend_ref` without dynamic shape conversion.
  3. Relied on Pallas's native dynamic DMA transfer `_send_bo(..., pl.ds(0, sz))` to write only the valid `sz` output tokens to HBM, eliminating all non-static shape operations and avoiding `TracerBoolConversionError`.

---

### Issue #11: `TypeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)`

- **Root Cause**:
  In `mla.py` and `mla_swa.py`, `run_mla_kernel` passed hardcoded `static_q_len = chunk_prefill_size` (64) during prefill execution. When vLLM compiled smaller prompt sequence ranges like `(16, 16)` (`q.shape[0] = 2` tokens per TP rank), `m_scratch` VMEM register buffer was allocated with shape `(q.shape[0] * num_q_heads, 128)` = `(8, 128)`. However, `static_q_len = 64` forced `bq_sz = min(8, 64) = 8`, causing `attention_sinks` concatenation inside `_kernel_body` to allocate `8 * 8 = 64` rows, producing a shape mismatch during `attention_sinks` `(64, 1)` - `m_ref` `(8, 128)` subtraction.
- **Fix**:
  Updated `static_q_len` in prefill `run_mla_kernel` calls in [mla.py:L1052](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L1052) and [mla_swa.py:L1294](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py#L1294) to `static_q_len = min(chunk_prefill_size, q.shape[0])`, ensuring `bq_sz` dynamic tile scaling matches actual VMEM array shapes across all sequence compile ranges.





