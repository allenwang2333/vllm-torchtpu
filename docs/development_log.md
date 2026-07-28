# DeepSeek-V4 Integration: Detailed Development & Execution Log

This document serves as a persistent, timestamped log tracking all architectural decisions, code changes, git commits, bug fixes, VM execution logs, and benchmark results for the DeepSeek-V4 integration in `vllm-torchtpu`.

---

## Session Timeline & Log Entries

### [2026-07-20 17:15:00 UTC] Initial Integration & Standalone File Migration
- **Context:** Ported 27 standalone DeepSeek-V4 files from backup repository `torchtpu-vllm-dsv4-backup-2026-07-15` into `vllm-torchtpu` on branch `dsv4-integration`.
- **Git Commit:** `ef2d8ba8` (`feat(dsv4): initial port of DeepSeek-V4 custom ops, kernels, and quantization configs`)
- **Key Registrations:**
  - Registered `DEEPSEEK_V4_FP8 = "deepseek_v4_fp8"` in `src/vllm_torchtpu/layers/common/quant_methods.py`.
  - Registered `@register_quantization_config("deepseek_v4_fp8")` and `"tpu-deepseek_v4_fp8"` on `VllmDeepseekV4Fp8Config` in `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py`.
  - Registered `"deepseek_v4_fp8"` in `TpuPlatform.supported_quantization` in `src/vllm_torchtpu/platforms/tpu_platform.py`.
  - Added JAX/Torch dtype conversion helpers (`get_jax_dtype_from_str_dtype`, `to_jax_dtype`, `to_torch_dtype`, `t2j_dtype`, `j2t_dtype`) to `src/vllm_torchtpu/utils.py`.

---

### [2026-07-20 19:17:00 UTC] Runner Model Patcher Hook & Sharding Clean-up
- **Context:** Resolved `WorkerProc` background initialization failure by hooking `_maybe_patch_for_deepseek_v4` in model loader.
- **Git Commit:** `e3f09c08` (`feat(runner): enable DeepSeek-V4 patch context in load_model and use native itemsize in sharding.py`)
- **Key Changes:**
  - Wrapped `model_loader.load_model(...)` in `with _maybe_patch_for_deepseek_v4(self.vllm_config):` inside `src/vllm_torchtpu/runner/tpu_runner.py`.

---

### [2026-07-20 19:19:00 UTC] Sharding Reversion & Axis Alignment Fix
- **Context:** Ensured `sharding.py` remains 100% untouched and aligned with `main` branch.
- **Git Commit:** `41fb6dce` (`fix: revert sharding.py to main branch state and align deepseek_v4 custom ops with ShardingAxisNameBase`)
- **Key Changes:**
  - Reverted `src/vllm_torchtpu/layers/common/sharding.py` to 0 diffs against `main`.
  - Replaced `ShardingAxisName.BATCH` with `BATCH_AXIS = getattr(ShardingAxisName, "BATCH", ShardingAxisName.ATTN_DATA)` in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.

---

### [2026-07-20 19:21:00 UTC] ZeroDivisionError Guard in MoE Parallel Config
- **Context:** Resolved `ZeroDivisionError: integer division or modulo by zero` in `ep_rank` calculation when DP-attention sharding is active (`attn_dp = tensor_parallel_size`).
- **Git Commit:** `20951e7a` (`fix(patch): guard tp_size and ep_size with max(1, ...) to prevent ZeroDivisionError`)
- **Key Changes:**
  - Guarded `tp_size = max(1, sharding_config.tp_size)` and `ep_size = max(1, sharding_config.expert_size)` in `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`.

---

### [2026-07-20 19:28:00 UTC] Workspace Guidelines & Architecture Rules
- **Context:** Documented localized rules, VM infrastructure details (`alyssa`, `patemotter-v6e-8-3`, `patemotter-cdk`), `rsync` workflows, and testing tiers.
- **Git Commit:** `be292077` (`docs: add AGENTS.md guidelines for DeepSeek-V4 TPU development and VM setups`)
- **Git Commit:** `5970b907` (`docs: document 4-layer vs full model testing tiers in AGENTS.md`)
- **Key Artifacts:** Created `AGENTS.md` in workspace root.

---

### [2026-07-20 19:29:00 UTC] Prebuild Fused MoE Kernel Keyword Argument Alignment
- **Context:** Resolved `TypeError: prebuild_fused_moe_kernel() got an unexpected keyword argument 'has_bias'` during `process_weights_after_loading()`.
- **Git Commit:** `56526aba` (`fix(moe): add optional has_bias parameter to prebuild_fused_moe_kernel`)
- **Key Changes:**
  - Updated `prebuild_fused_moe_kernel()` in `src/vllm_torchtpu/layers/vllm/fused_moe.py` to accept `has_bias: bool = False`.

---

### [2026-07-20 19:30:00 UTC] Eager Mode Full 35-Prompt Execution Run (`task-426`)
- **Execution Target:** Host VM `alyssa` (v7x-8, 8 chips).
- **Command:**
  ```bash
  ssh alyssa "sudo docker run --privileged --net=host --shm-size=16g \
    -v /home/patemotter_google_com/torchtpu-vllm:/root/cloud-devkit/torchtpu-vllm \
    -v /home/patemotter_google_com/workspace:/mnt/workspace \
    -w /root/cloud-devkit/torchtpu-vllm \
    -e PYTHONPATH=/root/cloud-devkit/torchtpu-vllm/src \
    -e DSV4_MODEL_PATH=/mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 \
    -e VLLM_FP4_CACHE_DIR=/mnt/workspace/tmp/dsv4-flash-real-fp4 \
    -e TPU_LIBRARY_PATH=/mnt/workspace/libtpu.so \
    -e ENABLE_QUANTIZED_MATMUL_KERNEL=1 \
    -e REQUANTIZE_BLOCK_SIZE=128 \
    -e MOE_REQUANTIZE_BLOCK_SIZE=512 \
    -e MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
    -e NEW_MODEL_DESIGN=1 \
    -e MODEL_IMPL_TYPE=vllm \
    -e VLLM_ENFORCE_EAGER=1 \
    -e VLLM_PLUGINS=torchtpu \
    -e GPU_MEM_UTIL=0.28 \
    -e XLA_CLIENT_MEM_FRACTION=0.32 \
    us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest \
    python3 -u examples/run_dsv4_inference.py"
  ```
- **Cache Audit:** 467 of 488 files (96%) hit pre-existing cache; remaining missing files currently populating to `/mnt/workspace/tmp/dsv4-flash-real-fp4`.

---

### [2026-07-20 20:33:00 UTC] PyTorch Dynamo Lock Context Manager Fix
- **Context:** Resolved `RuntimeError: Worker failed with error 'Unsupported context manager: Dynamo does not know how to enter a lock context manager'` during forward pass warm-up (`_initialize_kv_caches`).
- **Git Commit:** `250deccc` (`fix(linear): remove Lock context manager from _get_quantized_matmul_op for PyTorch Dynamo compatibility`)
- **Key Changes:**
  - Removed `with _quantized_matmul_kernel_op_lock:` in `src/vllm_torchtpu/layers/vllm/linear_common.py`.
  - Operator is now lazily assigned directly (`_quantized_matmul_kernel_op = op`), allowing PyTorch Dynamo to trace `_get_quantized_matmul_op()` without lock context breaks.
- **Execution Target:** Host VM `alyssa` (`task-630`). All 488 weight files loaded from NVMe cache in < 5 seconds.

---

### [2026-07-20 21:10:00 UTC] PyTorch Dynamo Multiproc Module Alias Fix
- **Context:** Resolved `ModuleNotFoundError: No module named 'patched_pallas_temp'` during multiproc worker execution graph tracing.
- **Git Commit:** `63812d82` (`fix(pallas): register patched_pallas_temp alias in sys.modules for multiproc Dynamo compatibility`)
- **Key Changes:**
  - Added `sys.modules["patched_pallas_temp"] = sys.modules[__name__]` at the top of `src/vllm_torchtpu/patches/pallas_patched.py`.
  - Maps the dynamic Pallas wrapper module name directly in `sys.modules` across all multiproc worker processes.
- **Execution Target:** Host VM `alyssa` (`task-726`).

---

### [2026-07-20 21:20:00 UTC] Hot-Patch Module Spec Registration Fix & Detached Execution Setup
- **Context:** Resolved `Failed to hot-patch pallas.py: 'patched_pallas_temp'` by explicitly assigning `sys.modules["patched_pallas_temp"] = patched_module` prior to `exec_module` in `run_dsv4_inference.py`.
- **Git Commit:** `dfdedda9` (`fix(examples): register patched_pallas_temp in sys.modules prior to exec_module in run_dsv4_inference.py`)
- **Workflow Change:** Switched remote Docker execution on `alyssa` to detached background daemon mode (`sudo docker run -d --name dsv4_eval`). Remote evaluation runs are now 100% independent of local agent restarts.

---

### [2026-07-20 21:26:00 UTC] Eager Import-Time Op Pre-Building Fix
- **Context:** Pre-built `_quantized_matmul_kernel_op` eagerly at module import time inside `linear_common.py`. Replaces the manual runner pre-warm hook from the backup repository (`torchtpu-vllm-dsv4-backup-2026-07-15`) with a self-contained import-time initialization.
- **Git Commit:** `9d51eccc` (`fix(linear): pre-build _quantized_matmul_kernel_op eagerly at import time for PyTorch Dynamo compatibility`)
- **Key Changes:**
  - Added `_build_quantized_matmul_op()` function in `src/vllm_torchtpu/layers/vllm/linear_common.py`.
  - Added explicit docstring and comments explaining Dynamo context manager and JAX JIT graph-break avoidance.
- **Execution Target:** Host VM `alyssa` (`dsv4_eval` detached daemon container).

---

### [2026-07-20 21:34:00 UTC] Forward Context Metadata Dict Lookup Guard
- **Context:** Resolved `torch._dynamo.exc.ObservedKeyError: Dict key lookup failed for ConstantVariable(str: 'model.layers.0.attn.swa_cache')` during attention warm-up.
- **Git Commit:** `fed57414` (`fix(dsv4): guard attn_metadata dictionary lookups in forward_mqa`)
- **Key Changes:**
  - Used `.get()` fallback logic for `swa_layer_name` and `main_layer_name` lookups in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (`dsv4_eval` detached container).

---

### [2026-07-20 21:38:00 UTC] Attention Metadata Attribute Access Safeguard
- **Context:** Resolved `torch._dynamo.exc.ObservedAttributeError: seq_lens` during `forward_mqa()` in `deepseek_v4.py`.
- **Git Commit:** `9f91e4cb` (`fix(dsv4): add getattr safeguards for seq_lens and metadata fields in forward_mqa`)
- **Key Changes:**
  - Wrapped `seq_lens`, `block_tables`, `query_start_loc`, and `request_distribution` attribute accesses in `getattr()` safeguards inside `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (`dsv4_eval` detached container, 60s background timer `task-896`).

---

### [2026-07-20 21:42:00 UTC] Unified AttentionMetadata Representation Support
- **Context:** Resolved `ObservedAttributeError: block_tables` by supporting both dictionary (AMD/Multi-cache) and direct dataclass (`AttentionMetadata`) representations in `forward_mqa()`.
- **Git Commit:** `c5d44db9` (`fix(dsv4): support both dict and dataclass AttentionMetadata in forward_mqa`)
- **Key Changes:**
  - Added type-checking for `isinstance(attn_ctx, dict)` in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
  - Extracted attributes with `getattr()` guards across both dictionary and dataclass metadata instances.
- **Execution Target:** Host VM `alyssa` (`dsv4_eval` detached container, 60s background timer `task-936`).

---

### [2026-07-20 21:53:00 UTC] Data-Dependent Tensor Branching Elimination
- **Context:** Replaced logical `or` on tensor attributes with an explicit `if ... is not None:` identity check to prevent PyTorch Dynamo graph breaks during model warm-up.
- **Git Commit:** `80338e6b` (`fix(dsv4): replace tensor boolean or with explicit if is not None identity check in forward_mqa`)
- **Key Changes:**
  - Restored clean identity check structure in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (`dsv4_eval` detached container, 60s background timer `task-983`).

---

### [2026-07-20 22:03:00 UTC] Standard Direct Attribute Access for AttentionMetadata
- **Context:** Restored direct attribute access (`main_attn_metadata.block_tables`, `main_attn_metadata.seq_lens`, etc.) in `forward_mqa()`, eliminating `getattr()` lookup graph breaks during PyTorch Dynamo tracing.
- **Git Commit:** `a973f984` (`fix(dsv4): restore direct attribute access for AttentionMetadata fields in forward_mqa`)
- **Key Changes:**
  - Replaced `getattr(main_attn_metadata, "field", None)` with standard attribute reads in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1343`).

---

### [2026-07-20 22:38:00 UTC] Pallas Blockwise Kernel Tile Division Fix
- **Context:** Resolved `out_block_size=128 must be divisible by compute_tile_n=256` in `blockwise_kernel.py` when `out_block_size` is 128 (e.g. `fused_wqa_wkv` partitioned under TP=8).
- **Git Commit:** `876e31da` (`fix(pallas): cap n_lane_multiplier to out_block_size // MXU_SIZE in blockwise_kernel`)
- **Key Changes:**
  - Dynamic tile scaling: `n_lane_multiplier = min(tuned_value.n_lane_multiplier, max(1, out_block_size // MXU_SIZE))` in `src/vllm_torchtpu/kernels/quantized_matmul/blockwise_kernel.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1400`).

---

### [2026-07-20 22:42:00 UTC] Lazy Quantized Matmul Operator Instantiation
- **Context:** `_quantized_matmul_kernel_op` was previously built at module import time in `linear_common.py`, capturing unpatched Pallas kernel functions before runtime fixes loaded.
- **Git Commit:** `d50be542` (`fix(linear_common): lazily instantiate _quantized_matmul_kernel_op on first call`)
- **Key Changes:**
  - Updated `_get_quantized_matmul_op()` in `src/vllm_torchtpu/layers/vllm/linear_common.py` to instantiate `_quantized_matmul_kernel_op` lazily on first invocation.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1467`).

---

### [2026-07-20 22:44:00 UTC] Restored Eager Module-Level Quantized Matmul Building
- **Context:** Restored top-level module import pre-building `_quantized_matmul_kernel_op = _build_quantized_matmul_op()` in `linear_common.py` to prevent PyTorch Dynamo C-extension lock graph breaks during forward execution.
- **Git Commit:** `00dbf60c` (`fix(linear_common): restore eager pre-building of _quantized_matmul_kernel_op`)
- **Key Changes:**
  - Restored module-level eager pre-building of `_quantized_matmul_kernel_op` in `src/vllm_torchtpu/layers/vllm/linear_common.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1489`).

---

### [2026-07-20 22:46:00 UTC] Pallas Compute Tile Fallback for Small Output Block Sizes
- **Context:** Resolved `ValueError: out_block_size=128 must be divisible by compute_tile_n=256` in `blockwise_kernel.py` when `out_block_size < compute_tile_n` (e.g. `fused_wqa_wkv` on TP=8).
- **Git Commit:** `e0b12017` (`fix(pallas): add compute_tile_n fallback when out_block_size < compute_tile_n`)
- **Key Changes:**
  - Added fallback check: `if out_block_size < compute_tile_n or out_block_size % compute_tile_n != 0:` to clamp `compute_tile_n = min(out_block_size, MXU_SIZE)` in `src/vllm_torchtpu/kernels/quantized_matmul/blockwise_kernel.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1517`).

---

### [2026-07-20 22:48:00 UTC] Safe Mesh Axis Lookup in Pallas Shape Conversion
- **Context:** Resolved `ValueError: tuple.index(x): x not in tuple` in `pallas_patched.py` when partition specs included mesh axis names not present in the active `mesh.axis_names` tuple.
- **Git Commit:** `9a298ba0` (`fix(pallas): safely lookup mesh axis names in _get_axis_size`)
- **Key Changes:**
  - Added presence check `if name in mesh.axis_names` in `_get_axis_size()` in `src/vllm_torchtpu/patches/pallas_patched.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1561`).

---

### [2026-07-20 22:50:00 UTC] Mesh Partition Spec Validation for Attention Op
- **Context:** Resolved `ValueError: Resource axis: attn_dp of P('attn_dp', 'model', None) is not found in mesh: ('model',)` during `deepseek_v4_attention_swa` tracing.
- **Git Commit:** `69ec6dd0` (`fix(dsv4): wrap attention partition specs with get_valid_spec in _build_attn_op`)
- **Key Changes:**
  - Wrapped all input and output partition specs with `get_valid_spec(mesh, ...)` in `_build_attn_op()` in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1599`).

---

### [2026-07-20 22:53:00 UTC] Pallas CompilerParams Backward Compatibility Patch
- **Context:** Resolved `TypeError: CompilerParams.__init__() got an unexpected keyword argument 'needs_layout_passes'` during MoE GMM kernel execution.
- **Git Commit:** `bcb209a9` (`fix(pallas): patch CompilerParams.__init__ to ignore needs_layout_passes`)
- **Key Changes:**
  - Patched `pltpu.CompilerParams.__init__` in `src/vllm_torchtpu/patches/pallas_patched.py` to pop `needs_layout_passes` safely.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1644`).

---

### [2026-07-20 22:55:00 UTC] Pallas Indirect Attribute Alias Fix
- **Context:** Resolved `AttributeError: module 'jax.experimental.pallas' has no attribute 'Indirect'` during MoE GMM kernel lowering.
- **Git Commit:** `eeeb1134` (`fix(pallas): alias pltpu.Indirect to pl.Indirect if missing`)
- **Key Changes:**
  - Added attribute alias check `if not hasattr(pl, "Indirect") and hasattr(pltpu, "Indirect"): setattr(pl, "Indirect", pltpu.Indirect)` in `src/vllm_torchtpu/patches/pallas_patched.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1682`).

---

### [2026-07-20 22:58:00 UTC] Fallback Indirect Class in Pallas Patch
- **Context:** Resolved `AttributeError: module 'jax.experimental.pallas' has no attribute 'Indirect'` in container JAX environment by providing a fallback `Indirect` class on `pl`.
- **Git Commit:** `b46c5cae` (`fix(pallas): provide fallback Indirect class on jax.experimental.pallas`)
- **Key Changes:**
  - Added fallback `class Indirect` on `jax.experimental.pallas` in `src/vllm_torchtpu/patches/pallas_patched.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1732`).

---

### [2026-07-20 23:00:00 UTC] SparseCore Ragged Gather Version Alignment
- **Context:** Resolved `Unsupported block shape type: <class 'patched_pallas_temp.Indirect'>` in MoE GMM kernel execution by defaulting `RAGGED_GATHER_VERSION=v1` and `RAGGED_GATHER_REDUCE_VERSION=v1`.
- **Git Commit:** `21798b7b` (`fix(envs): default RAGGED_GATHER_VERSION and RAGGED_GATHER_REDUCE_VERSION to v1`)
- **Key Changes:**
  - Configured `envs.py` defaults to `v1` gather and reduce kernels to avoid unsupported JAX `Indirect` block specifications.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1774`).

---

### [2026-07-20 23:02:00 UTC] MoE GMM Scale Tensor Rank Alignment
- **Context:** Resolved `AssertionError: rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)` in `gmm_v2.py`.
- **Git Commit:** `44920c47` (`fix(moe): handle 3D rhs_scale in gmm_v2 by expanding singleton dimension`)
- **Key Changes:**
  - Added 3D `rhs_scale` singleton dimension expansion `if rhs_scale.ndim == 3: rhs_scale = jnp.expand_dims(rhs_scale, axis=2)` in `validate_inputs` and `gmm` in `src/vllm_torchtpu/kernels/megablox/gmm_v2.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1828`).

---

### [2026-07-20 23:05:00 UTC] MXFP4 Cache Collision Fix
- **Context:** Resolved `rhs_scale shape (32, 8, 1, 4096) does not match expected (32, 8, 1, 2048)` caused by loading stale cached parameters from previous 61-layer model runs.
- **Git Commit:** `c2107e40` (`fix(mxfp4): add shape tag to cache filename to prevent cross-model cache collisions`)
- **Key Changes:**
  - Cleared stale cache files in `/mnt/workspace/tmp/dsv4-flash-real-fp4`.
  - Added weight shape signature (`{w13_shape_str}`) to MXFP4 parameter cache filenames in `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1878`).












---

### [2026-07-20 22:05:00 UTC] Debug Print Statement Graph Break Removal
- **Context:** Identified that `ObservedAttributeError: block_tables` occurred inside a leftover debug `print(f"... {swa_attn_metadata.block_tables.shape} ...")` statement on line 1033 of `deepseek_v4.py`.
- **Git Commit:** `d25be9f2` (`fix(dsv4): remove print statement from forward_mqa to prevent PyTorch Dynamo graph break`)
- **Key Changes:**
  - Removed line 1033 debug print from `forward_mqa()` in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1101`).

---

### [2026-07-20 22:13:00 UTC] Local Variable Pre-Extraction for Custom Op Arguments
- **Context:** Pre-extracted `swa_seq_lens`, `swa_block_tables`, `swa_query_start_loc`, `swa_request_distribution` into local variables before passing them into `self.attn_op(...)` to prevent PyTorch Dynamo attribute lookup graph breaks inside custom op argument tuples.
- **Git Commit:** `8c26eeaa` (`fix(dsv4): pre-extract swa_attn_metadata fields into local variables before custom op invocation`)
- **Key Changes:**
  - Added pre-extraction block in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1135`).

---

### [2026-07-20 22:16:00 UTC] Sqrtsoftplus Scoring Function Support in MoE Routing
- **Context:** Resolved `NotImplementedError: FusedMoE does not support sqrtsoftplus scoring function for TPU` during MoE routing in `moe_routing.py`.
- **Git Commit:** `bd2e3c4` (`fix(moe): add sqrtsoftplus scoring function support in moe_routing.py`)
- **Key Changes:**
  - Added `sqrtsoftplus` scoring calculation (`torch.sqrt(F.softplus(scores))`) to `_apply_scoring_fn()` in `src/vllm_torchtpu/layers/vllm/moe_routing.py`, matching backup repository implementation.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1180`).

---

### [2026-07-20 22:18:00 UTC] Compressor Module AttentionMetadata Unpacking
- **Context:** Resolved `ObservedKeyError: Dict key lookup failed for ConstantVariable(str: 'model.layers.2.attn.indexer.compressor.state_cache')` inside `Compressor.forward()` in `deepseek_v4.py`.
- **Git Commit:** `70343295` (`fix(dsv4): support both dict and dataclass AttentionMetadata in Compressor.forward`)
- **Key Changes:**
  - Added dual-representation handling (`dict` or dataclass) and pre-extracted `state_req_dist`, `state_q_start_loc`, `state_seq_lens`, and `state_block_tables` in `Compressor.forward()` inside `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1218`).

---

### [2026-07-20 22:29:00 UTC] Indexer Module AttentionMetadata Unpacking
- **Context:** Resolved `ObservedKeyError: Dict key lookup failed for ConstantVariable(str: 'model.layers.2.attn.indexer.k_cache')` inside `Indexer.forward()` in `deepseek_v4.py`.
- **Git Commit:** `23efbeb3` (`fix(dsv4): support both dict and dataclass AttentionMetadata in Indexer.forward`)
- **Key Changes:**
  - Added dual-representation handling (`dict` or dataclass), removed leftover debug print statement, and pre-extracted `idx_seq_lens`, `idx_block_tables`, `idx_q_start_loc`, and `idx_req_dist` in `Indexer.forward()` inside `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1252`).

---

### [2026-07-20 22:31:00 UTC] Pallas Quantized Matmul Tile Alignment Fix
- **Context:** Graph tracing and model forward execution succeeded 100% through all Python layers. The Pallas quantized matmul kernel raised `out_block_size=128 must be divisible by compute_tile_n=256` due to `-e REQUANTIZE_BLOCK_SIZE=128`.
- **Key Changes:**
  - Adjusted environment variable `-e REQUANTIZE_BLOCK_SIZE=256` in container launch command to align `out_block_size=256` with Pallas `compute_tile_n=256`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1271`).

---

### [2026-07-20 22:33:00 UTC] Dynamic Scale Expansion for wo_a Dequantization
- **Context:** Resolved `RuntimeError: Attempting to broadcast a dimension of length 2048 at -1! Mismatching argument at index 1 had torch.Size([1024, 2048]); but expected shape should be broadcastable to [1024, 4096]` when dequantizing `wo_a.weight` under Tensor Parallelism (TP=8).
- **Git Commit:** `b103b5e4` (`fix(dsv4): dynamically expand wo_a scale to match weight shape under TP`)
- **Key Changes:**
  - Replaced hardcoded `4096` / `128` column expansion with dynamic ratio expansion (`block_n = w.shape[0] // s.shape[0]`, `block_k = w.shape[1] // s.shape[1]`) in `deepseek_v4.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1309`).

---

### [2026-07-20 22:35:00 UTC] Patched Outer Model Forward Debug Print Removal
- **Context:** Identified that `patched_amd_forward` in `deepseek_v4_patch.py` contained a debug `print()` statement that was executed inside `_AmdDeepseekV4ForCausalLM.forward` during PyTorch Dynamo tracing.
- **Git Commit:** `42d77fb8` (`fix(dsv4): remove debug print statement from patched_amd_forward`)
- **Key Changes:**
  - Removed debug print loop from `patched_amd_forward` in `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/mnt/workspace/dsv4-flash-4layer-fp8`, 60s background timer `task-1343`).

---

### [2026-07-20 23:10:00 UTC] Pallas CompilerParams Keyword Argument Patch
- **Context:** Resolved `TypeError: CompilerParams.__init__() got an unexpected keyword argument 'needs_layout_passes'` during Pallas kernel compilation under `pltpu.CompilerParams`.
- **Git Commit:** `2b26090e` (`fix(pallas): pop unsupported needs_layout_passes in CompilerParams.__init__ patch`)
- **Key Changes:**
  - Added compatibility wrapper popping `needs_layout_passes` in `src/vllm_torchtpu/patches/pallas_patched.py`.
- **Execution Target:** Host VM `alyssa` (4-layer model `/home/patemotter_google_com/workspace/dsv4-flash-4layer-fp8`).

---

### [2026-07-20 23:20:00 UTC] SparseCore Ragged Gather Version Alignment
- **Context:** Resolved `Unsupported block shape type: <class 'patched_pallas_temp.Indirect'>` in MoE GMM kernel execution.
- **Git Commit:** `21798b7b` (`fix(envs): default RAGGED_GATHER_VERSION and RAGGED_GATHER_REDUCE_VERSION to v1`)
- **Key Changes:**
  - Defaulted `RAGGED_GATHER_VERSION` and `RAGGED_GATHER_REDUCE_VERSION` to `"v1"` in `src/vllm_torchtpu/envs.py`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-20 23:30:00 UTC] MoE GMM Scale Tensor Rank Alignment
- **Context:** Resolved `AssertionError: rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)` in `gmm_v2.py`.
- **Git Commit:** `44920c47` (`fix(moe): handle 3D rhs_scale in gmm_v2 by expanding singleton dimension`)
- **Key Changes:**
  - Added 3D scale tensor singleton dimension expansion `if rhs_scale.ndim == 3: rhs_scale = jnp.expand_dims(rhs_scale, axis=2)` in `validate_inputs` and `gmm` in `src/vllm_torchtpu/kernels/megablox/gmm_v2.py`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-20 23:45:00 UTC] MXFP4 Cache Collision Fix & Scale Shape Tagging
- **Context:** Resolved `rhs_scale shape (32, 8, 1, 4096) does not match expected (32, 8, 1, 2048)` caused by loading stale cached parameters from previous 61-layer model runs.
- **Git Commit:** `c2107e40` (`fix(mxfp4): add shape tag to cache filename to prevent cross-model cache collisions`)
- **Key Changes:**
  - Formatted both `w13` and `w2` shapes into `cache_path` (`rank_{rank}_layer_{layer_idx}_w13_{w13_shape_str}_w2_{w2_shape_str}.pt`) in `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py`.
  - Cleared stale root-owned `.pt` cache files in `/home/patemotter_google_com/workspace/tmp/dsv4-flash-real-fp4`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 00:03:00 UTC] Editable Package Installation for Quantization Config Registration
- **Context:** Resolved `pydantic_core._pydantic_core.ValidationError: deepseek_v4_fp8 quantization is currently not supported in tpu` when initializing `ModelConfig`.
- **Key Changes:**
  - Added `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-build-isolation --no-deps -e /root/cloud-devkit/torchtpu-vllm` to container execution environment to dynamically register `deepseek_v4_fp8` under `TpuPlatform.supported_quantization`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 00:05:00 UTC] Plugin Isolation (`VLLM_PLUGINS=torchtpu`)
- **Context:** Resolved `RuntimeError: torch.register_privateuse1_backend() has already been set!` caused by third-party plugin `tpu_inference` importing `torchax` and attempting to re-register the `privateuse1` backend.
- **Key Changes:**
  - Set `export VLLM_PLUGINS=torchtpu` in the execution environment to isolate the `torchtpu` platform plugin.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 00:15:00 UTC] Custom `libtpu.so` Environment Override (`TPU_LIBRARY_PATH`)
- **Context:** Resolved `ERROR: Unknown command line flag 'xla_tpu_use_dynamic_smem_negotiation'` when using pinned `libtpu==0.0.42.1` Python wheel.
- **Key Changes:**
  - Set `export TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/libtpu.so` to force PJRT to load the custom build of `libtpu.so`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 00:25:00 UTC] FP4 Packed Weight Scale Dimension Validation (`gmm_v2.py`)
- **Context:** Resolved `rhs_scale shape (32, 128, 1, 4096) does not match expected (32, 128, 1, 2048)` for packed FP4 weights where 2 FP4 values are packed into each uint8 byte along `dim=2`.
- **Key Changes:**
  - Updated `validate_inputs` in `src/vllm_torchtpu/kernels/megablox/gmm_v2.py` to allow `expected_n` in `(size_n, size_n * 2)`.
  - Added detailed shape reporting (`lhs.shape {lhs.shape} != expected {(size_m, size_k)}`) to the `lhs` shape assertion error message.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 00:45:00 UTC] MXFP4 `w2_weight` Dimension Swap Bug (`mxfp4.py`)
- **Context:** Resolved `AssertionError: lhs.shape (96, 1024) != expected (96, 2048)` during `gmm2` MoE execution.
- **Key Changes:**
  - Corrected inverted shape index extraction in `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py` where vLLM passes `w2_weight` (down_proj) with shape `[num_experts, local_intermediate_size, hidden_size]` (`[32, 1024, 4096]`).
  - Set `orig_local_intermediate_size = w2_weight.shape[1]` and `orig_hidden_size = w2_weight.shape[2]`.
  - Fixed `pad_spec_w2` padding to pad hidden dimension (`dim=2`).
  - Purged stale `.pt` parameter cache files in `/home/patemotter_google_com/workspace/tmp/dsv4-flash-real-fp4`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 03:15:00 UTC] MoE Intermediate Contract Alignment & Cache Scope Fix (`gmm_v2.py`, `mxfp4.py`)
- **Context:** Resolved `RuntimeError: Worker failed with error 'lhs.shape (96, 1024) != expected (96, 2048)'` during `gmm2` down-projection execution in Pallas GMM kernel.
- **Key Changes:**
  - In `src/vllm_torchtpu/kernels/megablox/gmm_v2.py`, preserved standard `out_size_n = dims.size_n // 2` and `size_n_per_rhs //= 2` behavior for SwiGLU (`fuse_act="silu"`), which outputs 1024 columns from 2048 input bytes.
  - In `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py`, set `orig_local_intermediate_size = w2_weight.shape[1] // 2` (1024) and sliced `w2_weight = w2_weight[:, :orig_local_intermediate_size, :]`.
  - Updated `w2_processed` to pack into shape `[32, 1024, 2048]` (`size_k = 1024`), achieving 100% contract shape alignment with `gmm1` output (`(96, 1024)`).
  - Fixed variable scope assignment for pre-quantized weight loads (`loaded_from_cache = True`) and enforced `w2_weight_processed[:, :1024, :]` slicing across both cold and warm parameter loading paths.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 03:45:00 UTC] Redundant Slicing Fix in Attention Output (`custom_ops/deepseek_v4.py`)
- **Context:** Resolved `Input hidden dim 128 must match weight hidden dim 1024` compilation crash in `DeepseekV4Attention`.
- **Key Changes:**
  - Removed incorrect second-stage slicing of `wo_a_dequant` in `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py` (lines 1114-1120).
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 04:00:00 UTC] JAX Packed FP4 Scale Validation (`fused_moe_gmm.py`)
- **Context:** Resolved `rhs_scale.shape` mismatches when loading pre-quantized FP4 weights from cache.
- **Key Changes:**
  - Patched `gmm_wrapper` in `src/vllm_torchtpu/layers/common/fused_moe_gmm.py` to identify packed `uint8` tensors as FP4: `is_fp4_weight = (rhs.dtype == jnp.uint8) or ...`
  - Explicitly pass `rhs_quant_dtype = jnp.float4_e2m1fn` to `gmm_v2` to trigger the appropriate input shape doubling and validation paths.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 04:15:00 UTC] Pallas Sharding Mesh Axis Validation (`custom_ops/deepseek_v4.py`)
- **Context:** Resolved `attn_dp` sharding errors on 1D tensor-parallel meshes.
- **Key Changes:**
  - Wrapped partition specifications of `compressor_op` and `indexer_op` in `get_valid_spec(mesh, ...)` to dynamically filter out non-existent axes to `None` on 1D meshes.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 04:30:00 UTC] Compiler Vector Memory (VMEM) Spill Bounds (`gmm_v2.py`, `env_override.py`)
- **Context:** Resolved register allocator VMEM OOM compiler crash (`Used 2.07G of 64.00M vmem`).
- **Key Changes:**
  - Added support for `GMM_MAX_TILE_N` override in `src/vllm_torchtpu/kernels/megablox/gmm_v2.py`.
  - Set default limits `GMM_MAX_TILE_K=1024` and `GMM_MAX_TILE_N=1024` in `src/vllm_torchtpu/env_override.py` to prevent VMEM register spills.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 05:00:00 UTC] Iterative GMM Tile Size Restriction to 512x512 (`env_override.py`)
- **Context:** The GMM compilation with `1024x1024` tile caps still crashed the compiler with `Used 276.29M of 64.00M vmem`, although it achieved a 10x reduction in register allocator spill slots (down from 2.07GB to 276MB).
- **Key Changes:**
  - Lowered default caps to `GMM_MAX_TILE_K=512` and `GMM_MAX_TILE_N=512` in `src/vllm_torchtpu/env_override.py` to further reduce compiler register pressure below the VMEM limit.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 05:10:00 UTC] Iterative GMM Tile Size Restriction to 256x256 (`env_override.py`)
- **Context:** The GMM compilation with `512x512` tile caps still crashed the compiler with `Used 105.11M of 64.00M vmem`, although it achieved a further 2.5x reduction in register allocator spill slots (down to 97.6MB).
- **Key Changes:**
  - Lowered default caps to `GMM_MAX_TILE_K=256` and `GMM_MAX_TILE_N=256` in `src/vllm_torchtpu/env_override.py` to bring the register allocator pressure within the VMEM limit.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 05:15:00 UTC] Guard against `num_n_tiles` Zero-Decrement division by zero (`gmm_v2.py`)
- **Context:** The GMM compilation crashed with `integer division or modulo by zero` inside `align_to`.
- **Analysis:** When the GMM tile caps limit initial `tile_n` below VMEM limits without running the while loop, `num_n_tiles` remains 1. The fallback block `if tile_n < tile_n_limit:` unconditionally decremented `num_n_tiles -= 1` to 0, passing 0 as the alignment stride to `align_to` which performs modulo/division.
- **Key Changes:**
  - Added a `and num_n_tiles > 1` guard to the fallback block in `src/vllm_torchtpu/kernels/megablox/gmm_v2.py`.
- **Execution Target:** Host VM `alyssa`.

---

### [2026-07-21 05:25:00 UTC] Hybrid KV Cache Coordinator Grouping Fix (`deepseek_v4.py`)
- **Context:** Resolved `AssertionError: HybridKVCacheCoordinator requires at least two attention groups` during engine startup.
- **Analysis:** SWA cache `swa_cache_layer` was initialized unconditionally for all layers. For global layers (0 & 1), normal cache was skipped, using only SWA. For compressed layers (2 & 3), both normal MLA cache (`MLAAttentionSpec`) and SWA cache (`SlidingWindowMLASpec`) were active. Because they all shared SWA specs, they were merged into a single attention group, causing the assertion failure.
- **Key Changes:**
  - Added `is_unused` check to `VllmDeepseekV4SWACache.get_kv_cache_spec` to return `None` if SWA cache is not utilized by the layer.
  - Set `is_unused = True` on `swa_cache_layer` when `self.compress_ratio > 1` (compressed layers 2 & 3) inside `VllmDeepseekV4MLAAttention.__init__`.
- **Execution Target:** Host VM `alyssa`.




















---

### [2026-07-21 00:45:57 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 5 processed and cached


---

### [2026-07-21 00:50:06 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 8 processed and cached


---

### [2026-07-21 01:00:17 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 18 processed and cached


---

### [2026-07-21 01:10:18 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 27 processed and cached


---

### [2026-07-21 01:20:16 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 35 processed and cached


---

### [2026-07-21 01:30:06 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 42 processed and cached


---

### [2026-07-21 01:30:19 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 42 processed and cached


---

### [2026-07-21 01:40:09 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  (Worker_TP0_EP0 pid=822) Traceback (most recent call last):
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP0_EP0 pid=822)     self.run()
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP0_EP0 pid=822)     self._target(*self._args, **self._kwargs)
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 173, in _report_usage_worker
(Worker_TP0_EP0 pid=822)     self._report_usage_once(model_architecture, usage_context, extra_kvs)
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 217, in _report_usage_once
(Worker_TP0_EP0 pid=822)     info = cpuinfo.get_cpu_info()
(Worker_TP0_EP0 pid=822)            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/site-packages/cpuinfo/cpuinfo.py", line 2762, in get_cpu_info
(Worker_TP0_EP0 pid=822)     output = json.loads(output, object_hook = _utf_to_str)
(Worker_TP0_EP0 pid=822)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/json/__init__.py", line 359, in loads
(Worker_TP0_EP0 pid=822)     return cls(**kw).decode(s)
(Worker_TP0_EP0 pid=822)            ^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/json/decoder.py", line 338, in decode
(Worker_TP0_EP0 pid=822)     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(Worker_TP0_EP0 pid=822)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=822)   File "/usr/local/lib/python3.12/json/decoder.py", line 356, in raw_decode
(Worker_TP0_EP0 pid=822)     raise JSONDecodeError("Expecting value", s, err.value) from None
(Worker_TP0_EP0 pid=822) json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
(Worker_TP6_EP6 pid=828) Exception in thread Thread-2 (_report_usage_worker):
(Worker_TP6_EP6 pid=828) Traceback (most recent call last):
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP6_EP6 pid=828)     self.run()
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP6_EP6 pid=828)     self._target(*self._args, **self._kwargs)
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 173, in _report_usage_worker
(Worker_TP6_EP6 pid=828)     self._report_usage_once(model_architecture, usage_context, extra_kvs)
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 217, in _report_usage_once
(Worker_TP6_EP6 pid=828)     info = cpuinfo.get_cpu_info()
(Worker_TP6_EP6 pid=828)            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/site-packages/cpuinfo/cpuinfo.py", line 2762, in get_cpu_info
(Worker_TP6_EP6 pid=828)     output = json.loads(output, object_hook = _utf_to_str)
(Worker_TP6_EP6 pid=828)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/json/__init__.py", line 359, in loads
(Worker_TP6_EP6 pid=828)     return cls(**kw).decode(s)
(Worker_TP6_EP6 pid=828)            ^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/json/decoder.py", line 338, in decode
(Worker_TP6_EP6 pid=828)     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(Worker_TP6_EP6 pid=828)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=828)   File "/usr/local/lib/python3.12/json/decoder.py", line 356, in raw_decode
(Worker_TP6_EP6 pid=828)     raise JSONDecodeError("Expecting value", s, err.value) from None
(Worker_TP6_EP6 pid=828) json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
(Worker_TP3_EP3 pid=825) Exception in thread Thread-2 (_report_usage_worker):
(Worker_TP3_EP3 pid=825) Traceback (most recent call last):
(Worker_TP3_EP3 pid=825)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP3_EP3 pid=825)     self.run()
(Worker_TP3_EP3 pid=825)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP3_EP3 pid=825)     self._target(*self._args, **self._kwargs)
  ```


---

### [2026-07-21 02:06:48 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 02:26:59 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  (Worker_TP6_EP6 pid=543) Traceback (most recent call last):
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP6_EP6 pid=543)     self.run()
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP6_EP6 pid=543)     self._target(*self._args, **self._kwargs)
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 173, in _report_usage_worker
(Worker_TP6_EP6 pid=543)     self._report_usage_once(model_architecture, usage_context, extra_kvs)
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 217, in _report_usage_once
(Worker_TP6_EP6 pid=543)     info = cpuinfo.get_cpu_info()
(Worker_TP6_EP6 pid=543)            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/site-packages/cpuinfo/cpuinfo.py", line 2762, in get_cpu_info
(Worker_TP6_EP6 pid=543)     output = json.loads(output, object_hook = _utf_to_str)
(Worker_TP6_EP6 pid=543)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/json/__init__.py", line 359, in loads
(Worker_TP6_EP6 pid=543)     return cls(**kw).decode(s)
(Worker_TP6_EP6 pid=543)            ^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/json/decoder.py", line 338, in decode
(Worker_TP6_EP6 pid=543)     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(Worker_TP6_EP6 pid=543)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP6_EP6 pid=543)   File "/usr/local/lib/python3.12/json/decoder.py", line 356, in raw_decode
(Worker_TP6_EP6 pid=543)     raise JSONDecodeError("Expecting value", s, err.value) from None
(Worker_TP6_EP6 pid=543) json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
(Worker_TP5_EP5 pid=542) Exception in thread Thread-2 (_report_usage_worker):
(Worker_TP5_EP5 pid=542) Traceback (most recent call last):
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP5_EP5 pid=542)     self.run()
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP5_EP5 pid=542)     self._target(*self._args, **self._kwargs)
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 173, in _report_usage_worker
(Worker_TP5_EP5 pid=542)     self._report_usage_once(model_architecture, usage_context, extra_kvs)
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 217, in _report_usage_once
(Worker_TP5_EP5 pid=542)     info = cpuinfo.get_cpu_info()
(Worker_TP5_EP5 pid=542)            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/site-packages/cpuinfo/cpuinfo.py", line 2762, in get_cpu_info
(Worker_TP5_EP5 pid=542)     output = json.loads(output, object_hook = _utf_to_str)
(Worker_TP5_EP5 pid=542)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/json/__init__.py", line 359, in loads
(Worker_TP5_EP5 pid=542)     return cls(**kw).decode(s)
(Worker_TP5_EP5 pid=542)            ^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/json/decoder.py", line 338, in decode
(Worker_TP5_EP5 pid=542)     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(Worker_TP5_EP5 pid=542)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/json/decoder.py", line 356, in raw_decode
(Worker_TP5_EP5 pid=542)     raise JSONDecodeError("Expecting value", s, err.value) from None
(Worker_TP5_EP5 pid=542) json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
(Worker_TP0_EP0 pid=537) Exception in thread Thread-2 (_report_usage_worker):
(Worker_TP0_EP0 pid=537) Traceback (most recent call last):
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP0_EP0 pid=537)     self.run()
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP0_EP0 pid=537)     self._target(*self._args, **self._kwargs)
  ```


---

### [2026-07-21 02:27:31 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 02:47:40 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  (Worker_TP0_EP0 pid=537) Traceback (most recent call last):
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP0_EP0 pid=537)     self.run()
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP0_EP0 pid=537)     self._target(*self._args, **self._kwargs)
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 173, in _report_usage_worker
(Worker_TP0_EP0 pid=537)     self._report_usage_once(model_architecture, usage_context, extra_kvs)
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 217, in _report_usage_once
(Worker_TP0_EP0 pid=537)     info = cpuinfo.get_cpu_info()
(Worker_TP0_EP0 pid=537)            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/site-packages/cpuinfo/cpuinfo.py", line 2762, in get_cpu_info
(Worker_TP0_EP0 pid=537)     output = json.loads(output, object_hook = _utf_to_str)
(Worker_TP0_EP0 pid=537)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/json/__init__.py", line 359, in loads
(Worker_TP0_EP0 pid=537)     return cls(**kw).decode(s)
(Worker_TP0_EP0 pid=537)            ^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/json/decoder.py", line 338, in decode
(Worker_TP0_EP0 pid=537)     obj, end = self.raw_decode(s, idx=_w(s, 0).end())
(Worker_TP0_EP0 pid=537)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0_EP0 pid=537)   File "/usr/local/lib/python3.12/json/decoder.py", line 356, in raw_decode
(Worker_TP0_EP0 pid=537)     raise JSONDecodeError("Expecting value", s, err.value) from None
(Worker_TP0_EP0 pid=537) json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
(Worker_TP5_EP5 pid=542) DEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_compressor_128_4_1: None
(Worker_TP2_EP2 pid=539) DEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_compressor_128_4_1: None
(Worker_TP5_EP5 pid=542) (Worker_TP2_EP2 pid=539) DEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_indexer: NoneDEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_indexer: None

(Worker_TP5_EP5 pid=542) Exception in thread Thread-2 (_report_usage_worker):
(Worker_TP5_EP5 pid=542) (Worker_TP2_EP2 pid=539) Exception in thread Thread-2 (_report_usage_worker)Traceback (most recent call last):
(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
:
(Worker_TP2_EP2 pid=539) Traceback (most recent call last):
(Worker_TP2_EP2 pid=539)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP5_EP5 pid=542)     self.run()
(Worker_TP2_EP2 pid=539)     self.run()(Worker_TP5_EP5 pid=542)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP2_EP2 pid=539) DEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_compressor_512_4_1: None
(Worker_TP5_EP5 pid=542) DEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_compressor_512_4_1: None
(Worker_TP2_EP2 pid=539) [DEBUG_SWA] Layer model.layers.2.attn compress_ratio=4 class=VllmDeepseekV4SWACache is_unused=True
(Worker_TP5_EP5 pid=542) [DEBUG_SWA] Layer model.layers.2.attn compress_ratio=4 class=VllmDeepseekV4SWACache is_unused=True
(Worker_TP1_EP1 pid=538) Exception in thread Thread-2 (_report_usage_worker):
(Worker_TP1_EP1 pid=538) Traceback (most recent call last):
(Worker_TP1_EP1 pid=538)   File "/usr/local/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(Worker_TP1_EP1 pid=538)     self.run()
(Worker_TP1_EP1 pid=538)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run
(Worker_TP2_EP2 pid=539) (Worker_TP5_EP5 pid=542) DEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_attention_csa: NoneDEBUG_PALLAS: registering fake_fn for pallas::deepseek_v4_attention_csa: None
(Worker_TP1_EP1 pid=538)     self._target(*self._args, **self._kwargs)

(Worker_TP1_EP1 pid=538)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 173, in _report_usage_worker
(Worker_TP2_EP2 pid=539)   File "/usr/local/lib/python3.12/threading.py", line 1012, in run

(Worker_TP1_EP1 pid=538)     self._report_usage_once(model_architecture, usage_context, extra_kvs)
(Worker_TP1_EP1 pid=538)   File "/usr/local/lib/python3.12/site-packages/vllm/usage/usage_lib.py", line 217, in _report_usage_once
  ```


---

### [2026-07-21 03:04:56 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 2 processed and cached


---

### [2026-07-21 03:15:11 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 11 processed and cached


---

### [2026-07-21 03:30:04 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 25 processed and cached


---

### [2026-07-21 03:45:03 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 39 processed and cached


---

### [2026-07-21 04:00:04 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  ERROR 07-21 09:49:27 [__init__.py:64] Traceback (most recent call last):
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/vllm/plugins/__init__.py", line 61, in load_plugins_by_group
ERROR 07-21 09:49:27 [__init__.py:64]     func = plugin.load()
ERROR 07-21 09:49:27 [__init__.py:64]            ^^^^^^^^^^^^^
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/importlib/metadata/__init__.py", line 205, in load
ERROR 07-21 09:49:27 [__init__.py:64]     module = import_module(match.group('module'))
ERROR 07-21 09:49:27 [__init__.py:64]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/importlib/__init__.py", line 90, in import_module
ERROR 07-21 09:49:27 [__init__.py:64]     return _bootstrap._gcd_import(name[level:], package, level)
ERROR 07-21 09:49:27 [__init__.py:64]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 07-21 09:49:27 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
ERROR 07-21 09:49:27 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
ERROR 07-21 09:49:27 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
ERROR 07-21 09:49:27 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
ERROR 07-21 09:49:27 [__init__.py:64]   File "<frozen importlib._bootstrap_external>", line 999, in exec_module
ERROR 07-21 09:49:27 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/__init__.py", line 14, in <module>
ERROR 07-21 09:49:27 [__init__.py:64]     from tpu_inference.layers.vllm import backends as backends
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/backends/__init__.py", line 14, in <module>
ERROR 07-21 09:49:27 [__init__.py:64]     from tpu_inference.layers.vllm.backends import flash_attn as flash_attn
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/backends/flash_attn.py", line 9, in <module>
ERROR 07-21 09:49:27 [__init__.py:64]     from torchax.interop import jax_view, torch_view
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/torchax/__init__.py", line 107, in <module>
ERROR 07-21 09:49:27 [__init__.py:64]     torch.utils.rename_privateuse1_backend("jax")
ERROR 07-21 09:49:27 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/torch/utils/backend_registration.py", line 79, in rename_privateuse1_backend
ERROR 07-21 09:49:27 [__init__.py:64]     _rename_privateuse1_backend(backend_name)
ERROR 07-21 09:49:27 [__init__.py:64] RuntimeError: torch.register_privateuse1_backend() has already been set! Current backend: tpu
INFO 07-21 09:49:27 [utils.py:278] non-default args: {'trust_remote_code': True, 'kv_cache_dtype': 'fp8', 'max_model_len': 64, 'tensor_parallel_size': 8, 'enable_expert_parallel': True, 'gpu_memory_utilization': 0.2, 'max_num_batched_tokens': 64, 'disable_log_stats': True, 'quantization': 'deepseek_v4_fp8', 'additional_config': {'compilation_config': {'compile_sizes': [64]}, 'sharding': {'sharding_strategy': {'enable_dp_attention': True, 'expert_parallelism': 8}}}, 'model': '/mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1'}
WARNING 07-21 09:49:27 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_FP4_CACHE_DIR
INFO 07-21 09:49:27 [config.py:803] Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM.
WARNING 07-21 09:49:36 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
WARNING 07-21 09:49:36 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
WARNING 07-21 09:49:36 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
WARNING 07-21 09:49:36 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
INFO 07-21 09:49:36 [model.py:617] Resolved architecture: DeepseekV4ForCausalLM
INFO 07-21 09:49:36 [model.py:635] Defaulting to tokenizer_mode='deepseek_v4' for DeepseekV4ForCausalLM
INFO 07-21 09:49:36 [model.py:1752] Using max model len 64
INFO 07-21 09:49:36 [cache.py:261] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor
INFO 07-21 09:49:36 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=64.
INFO 07-21 09:49:36 [vllm.py:977] Asynchronous scheduling is enabled.
INFO 07-21 09:49:36 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
INFO 07-21 09:49:36 [__init__.py:36] Applied TPU patch: disable vLLM custom collective ops.
INFO 07-21 09:49:36 [__init__.py:57] Applied TPU patch: DefaultMoERunner uses direct _moe_forward.
INFO 07-21 09:49:36 [__init__.py:84] Applied TPU patch: disable dynamic compile_ranges (static compile_sizes only).
INFO 07-21 09:49:36 [__init__.py:158] Applied TPU patch: disable sequence-parallel MoE.
INFO 07-21 09:49:36 [__init__.py:228] Applied TPU patch: scope non-EP MoE TP inside each DP engine.
INFO 07-21 09:49:36 [__init__.py:141] Applied TPU patch: defer RowParallelLinear bias past TP all_reduce.
INFO 07-21 09:49:36 [__init__.py:174] Applied TPU patch: disable DP microbatching.
INFO 07-21 09:49:36 [__init__.py:281] Applied TPU patch: MultiprocExecutor worker binding env.
INFO 07-21 09:49:36 [tpu_platform.py:87] TorchTPU eager mode configured: DEFER_AND_FUSE (previous=DEFER_NEVER)
INFO 07-21 09:49:36 [tpu_platform.py:676] Initialized sharding configuration: ShardingConfigManager(total_devices=0, sharding_strategy=ShardingStrategy(tensor_parallelism=0, expert_parallelism=8, sequence_parallelism=1, data_parallelism=1, attention_data_parallelism=16), device_indexes=None)
  ```


---

### [2026-07-21 04:01:53 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 04:15:02 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Safetensors shard loading: 100% Completed | 46/46


---

### [2026-07-21 04:30:05 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 11 processed and cached


---

### [2026-07-21 04:45:04 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 27 processed and cached


---

### [2026-07-21 05:00:06 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 41 processed and cached


---

### [2026-07-21 05:15:04 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  ERROR 07-21 11:02:53 [__init__.py:64] Traceback (most recent call last):
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/vllm/plugins/__init__.py", line 61, in load_plugins_by_group
ERROR 07-21 11:02:53 [__init__.py:64]     func = plugin.load()
ERROR 07-21 11:02:53 [__init__.py:64]            ^^^^^^^^^^^^^
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/importlib/metadata/__init__.py", line 205, in load
ERROR 07-21 11:02:53 [__init__.py:64]     module = import_module(match.group('module'))
ERROR 07-21 11:02:53 [__init__.py:64]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/importlib/__init__.py", line 90, in import_module
ERROR 07-21 11:02:53 [__init__.py:64]     return _bootstrap._gcd_import(name[level:], package, level)
ERROR 07-21 11:02:53 [__init__.py:64]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 07-21 11:02:53 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
ERROR 07-21 11:02:53 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
ERROR 07-21 11:02:53 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
ERROR 07-21 11:02:53 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
ERROR 07-21 11:02:53 [__init__.py:64]   File "<frozen importlib._bootstrap_external>", line 999, in exec_module
ERROR 07-21 11:02:53 [__init__.py:64]   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/__init__.py", line 14, in <module>
ERROR 07-21 11:02:53 [__init__.py:64]     from tpu_inference.layers.vllm import backends as backends
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/backends/__init__.py", line 14, in <module>
ERROR 07-21 11:02:53 [__init__.py:64]     from tpu_inference.layers.vllm.backends import flash_attn as flash_attn
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/vllm/backends/flash_attn.py", line 9, in <module>
ERROR 07-21 11:02:53 [__init__.py:64]     from torchax.interop import jax_view, torch_view
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/torchax/__init__.py", line 107, in <module>
ERROR 07-21 11:02:53 [__init__.py:64]     torch.utils.rename_privateuse1_backend("jax")
ERROR 07-21 11:02:53 [__init__.py:64]   File "/usr/local/lib/python3.12/site-packages/torch/utils/backend_registration.py", line 79, in rename_privateuse1_backend
ERROR 07-21 11:02:53 [__init__.py:64]     _rename_privateuse1_backend(backend_name)
ERROR 07-21 11:02:53 [__init__.py:64] RuntimeError: torch.register_privateuse1_backend() has already been set! Current backend: tpu
INFO 07-21 11:02:53 [utils.py:278] non-default args: {'trust_remote_code': True, 'kv_cache_dtype': 'fp8', 'max_model_len': 64, 'tensor_parallel_size': 8, 'enable_expert_parallel': True, 'gpu_memory_utilization': 0.2, 'max_num_batched_tokens': 64, 'disable_log_stats': True, 'quantization': 'deepseek_v4_fp8', 'additional_config': {'compilation_config': {'compile_sizes': [64]}, 'sharding': {'sharding_strategy': {'enable_dp_attention': True, 'expert_parallelism': 8}}}, 'model': '/mnt/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1'}
WARNING 07-21 11:02:53 [envs.py:2057] Unknown vLLM environment variable detected: VLLM_FP4_CACHE_DIR
INFO 07-21 11:02:53 [config.py:803] Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM.
WARNING 07-21 11:03:06 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
WARNING 07-21 11:03:06 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
WARNING 07-21 11:03:06 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
WARNING 07-21 11:03:06 [interface.py:247] Failed to import from vllm._C: ImportError('libcudart.so.13: cannot open shared object file: No such file or directory')
INFO 07-21 11:03:07 [model.py:617] Resolved architecture: DeepseekV4ForCausalLM
INFO 07-21 11:03:07 [model.py:635] Defaulting to tokenizer_mode='deepseek_v4' for DeepseekV4ForCausalLM
INFO 07-21 11:03:07 [model.py:1752] Using max model len 64
INFO 07-21 11:03:07 [cache.py:261] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor
INFO 07-21 11:03:07 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=64.
INFO 07-21 11:03:07 [vllm.py:977] Asynchronous scheduling is enabled.
INFO 07-21 11:03:07 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
INFO 07-21 11:03:08 [__init__.py:36] Applied TPU patch: disable vLLM custom collective ops.
INFO 07-21 11:03:08 [__init__.py:57] Applied TPU patch: DefaultMoERunner uses direct _moe_forward.
INFO 07-21 11:03:08 [__init__.py:84] Applied TPU patch: disable dynamic compile_ranges (static compile_sizes only).
INFO 07-21 11:03:08 [__init__.py:158] Applied TPU patch: disable sequence-parallel MoE.
INFO 07-21 11:03:08 [__init__.py:228] Applied TPU patch: scope non-EP MoE TP inside each DP engine.
INFO 07-21 11:03:08 [__init__.py:141] Applied TPU patch: defer RowParallelLinear bias past TP all_reduce.
INFO 07-21 11:03:08 [__init__.py:174] Applied TPU patch: disable DP microbatching.
INFO 07-21 11:03:08 [__init__.py:281] Applied TPU patch: MultiprocExecutor worker binding env.
INFO 07-21 11:03:08 [tpu_platform.py:87] TorchTPU eager mode configured: DEFER_AND_FUSE (previous=DEFER_NEVER)
INFO 07-21 11:03:08 [tpu_platform.py:676] Initialized sharding configuration: ShardingConfigManager(total_devices=0, sharding_strategy=ShardingStrategy(tensor_parallelism=0, expert_parallelism=8, sequence_parallelism=1, data_parallelism=1, attention_data_parallelism=16), device_indexes=None)
  ```


---

### [2026-07-21 05:15:31 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 05:28:51 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 05:30:02 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Safetensors shard loading: 100% Completed | 46/46


---

### [2026-07-21 05:43:32 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 05:45:02 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Safetensors shard loading: 100% Completed | 46/46


---

### [2026-07-21 06:00:04 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Sampler compilation: gather_logprobs in progress


---

### [2026-07-21 06:03:35 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 06:15:02 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Safetensors shard loading: 100% Completed | 46/46


---

### [2026-07-21 06:30:02 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 10 processed and cached


---

### [2026-07-21 06:45:03 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 25 processed and cached


---

### [2026-07-21 07:00:02 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** MoE weight pre-processing: layer 40 processed and cached


---

### [2026-07-21 07:15:03 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Sampler compilation: substitute_placeholder_token in progress


---

### [2026-07-21 07:45:05 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Safetensors shard loading: 100% Completed | 46/46


---

### [2026-07-21 08:00:05 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Sampler compilation: substitute_placeholder_token in progress


---

### [2026-07-21 08:15:04 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Safetensors shard loading: 100% Completed | 46/46


---

### [2026-07-21 08:30:03 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Sampler compilation: gather_logprobs in progress


---

### [2026-07-21 10:46:21 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 10:47:06 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  Traceback (most recent call last):
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 134, in <module>
INFO 07-21 17:46:45 [utils.py:278] non-default args: {'trust_remote_code': True, 'kv_cache_dtype': 'fp8', 'max_model_len': 16, 'tensor_parallel_size': 8, 'enable_expert_parallel': True, 'gpu_memory_utilization': 0.32, 'max_num_batched_tokens': 16, 'disable_log_stats': True, 'quantization': 'deepseek_v4_fp8', 'additional_config': {'compilation_config': {'compile_sizes': [16]}, 'sharding': {'sharding_strategy': {'enable_dp_attention': True, 'expert_parallelism': 8}}}, 'model': '/home/patemotter_google_com/workspace/dsv4-flash-4layer-fp8'}
    main()
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 43, in main
    llm = LLM(
          ^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 162, in from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1709, in create_engine_config
    device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
  File "/usr/local/lib/python3.12/site-packages/vllm/config/device.py", line 78, in __post_init__
    self.device = torch.device(self.device_type)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: tpu
  ```


---

### [2026-07-21 10:49:35 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 10:51:15 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  Traceback (most recent call last):
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 134, in <module>
    main()
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 43, in main
    llm = LLM(
          ^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 162, in from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1709, in create_engine_config
    device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
  File "/usr/local/lib/python3.12/site-packages/vllm/config/device.py", line 78, in __post_init__
    self.device = torch.device(self.device_type)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: tpu
  ```


---

### [2026-07-21 10:51:51 UTC] Full Model Offline Inference Progress Run
- **Container Status:** Running (Up)
- **Progress State:** Initializing...


---

### [2026-07-21 10:52:30 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  Traceback (most recent call last):
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 134, in <module>
    main()
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 43, in main
    llm = LLM(
          ^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 162, in from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1709, in create_engine_config
    device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
  File "/usr/local/lib/python3.12/site-packages/vllm/config/device.py", line 78, in __post_init__
    self.device = torch.device(self.device_type)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: tpu
  ```


---

### [2026-07-21 10:54:56 UTC] Full Model Offline Inference Failed
- **Container Status:** Exited (1) - Failure
- **Traceback:**
  ```
  Traceback (most recent call last):
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 134, in <module>
    main()
  File "/root/cloud-devkit/torchtpu-vllm/examples/run_dsv4_inference.py", line 43, in main
    llm = LLM(
          ^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 162, in from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 2224, in create_engine_config
    config = VllmConfig(
             ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
  Value error, No TPU devices found. [type=value_error, input_value=ArgsKwargs((), {'model_co... 'shutdown_timeout': 0}), input_type=ArgsKwargs]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
  ```

---

### [2026-07-22 14:10:00 UTC] DeepSeek-V4 Correctness Debugging & Synthetic Parity Verification
- **Context:** Resolved compiler shape mismatches, Pallas sharding bounds, compiler VMEM spills, CustomOp negative constraints, and verified FP4 MoE numerical correctness.
- **Key Changes:**
  - **Attention Sinks Sharding Fix**: Synced query attention sinks layout partition spec dynamically under TP=8 to avoid multi-device python indexing gather crashes.
  - **PyTorch-XLA CustomOp Sentinel Fix**: Masked negative expert IDs (-1) to 0 at the PyTorch CustomOp boundary, and recovered the sentinels back to -1 inside the JAX kernel graph.
  - **Weight Scale Parameter Shapes**: Enabled `is_channel_quant = True` on the FP8 quantization config, resolving shape mismatch assertions.
  - **MoE GMM Contracting Axis Shape Bug**: Corrected packed weight contracting axis detection logic (`padded_hidden_size = w1.shape[1] * 2` for `uint8`) in `fused_moe_gmm.py`, resolving negative padding crashes.
  - **Synthetic Correctness Verification**: Created `test_fused_moe_fp4.py` unit test comparing custom Pallas GMM MoE kernel outputs with CPU dequantization reference matrix multiplication.
- **Result:**
  - Parity successfully verified: Max Absolute Difference = 0.125, Mean Absolute Error = 0.014 (within bfloat16 hardware accumulator rounding bounds).
  - Cleaned up local workspace backups and reclaimed 80 GB of disk space on the laptop.

---

### [2026-07-22 14:45:00 UTC] DeepSeek-V4 Kernel Correctness Verification & SWA Softmax Invariant Fix
- **Context:** Verified correctness of core DeepSeek-V4 attention and routing kernels, and resolved sliding window attention (SWA) verification test failure.
- **Key Changes:**
  - **SWA Softmax Denominator Assertion Fix**: Corrected verification test in `test_dsv4_mla_swa.py` to compare the mathematically invariant `logsumexp = log(L) + m` rather than `L` and `m` individually. This bypasses the scaling offset caused by different max-value tracking in block-wise kernel execution versus monolithic CPU reference.
  - **GMM MXFP4 Contracting Axis Packing**: Updated `VllmMxfp4MoEMethod` to pack FP4 weight indices along `axis=1` (contracting dimension K) matching the offline pre-quantization script shape outputs.
  - **FP8 Scale Routing Prefix Helper**: Configured prefix-based linear layer routing in `VllmFp8LinearMethodTPU` to enforce per-channel FP8 scaling on dense layers (attention, shared experts) while routing routed experts to block-wise configuration.
  - **Activation Logs Patcher**: Hooked `DeepseekV4DecoderLayer.forward` to print activation statistics (mean, std, min, max, NaNs) layer-by-layer to help locate numerical divergence during execution.
  - **MoE Weight Loading Correctness Test**: Created `test_dsv4_moe_weight_loading.py` unit test to verify that the FusedMoE weight loader correctly uninterleaves `w1`/`w3` from the checkpoint, pads, re-quantizes to FP4, and packs along axis 1.
- **Result:**
  - `pytest` correctness tests for MLA attention, SWA attention, KV compressor, MoE routing (EP=8), and StreamIndex TopK all completed successfully on TPU.
  - MoE weight loading test passed successfully with exact dequantized parity (MAE = 0.0) compared to raw checkpoint reference.


---

### [2026-07-27 07:07:00 UTC] DeepSeek-V4 Code Correctness Audit & Critical Correctness Fixes
- **Context:** Conducted a comprehensive code correctness and mathematical audit of attention kernels, custom ops, and model weight loading layers, resolving multiple critical bugs causing NaN outputs, loading failures, and index mismatches.
- **Key Changes:**
  - **Attention Sinks Head-Channel Broadcasting Mismatch Fix**: Resolved shape mismatch in `mla.py` where a `(1, 128)` sinks tensor was subtracted from a `(bq_sz * num_q_heads, 128)` max-score matrix, resulting in mathematically invalid column-wise head-sinks subtraction. Modified to tile and expand `sinks` along the head dimension `(bq_sz * num_q_heads, 1)`.
  - **SWA Cache Read mis-indexing Fix**: Fixed a critical division rounding bug in `mla_swa.py` where `(-shift_amount) // kv_packing` rounded down to `-1` when `shift_amount` was positive (i.e. `new_kv_packing_offset < kv_packing_offset`), resulting in reading from the wrong VMEM word. Simplified to read directly from `cdiv(token_offset_in_bkv, kv_packing)`.
  - **SWA Attention Sinks NaN Guard**: Injected `jnp.where` guards in `mla_swa.py` to prevent overflow (`exp_attention_sinks` evaluating to `inf` or `NaN`) during sinks calculation when `m` evaluates to extremely small/masked values (`m <= -1e30` or `l <= 1e-10`).
  - **Model Weight Loader Requantization Ordering Fix**: Corrected weight loading pipeline order inside `patched_load_weights` to execute `WrappedMapper` key mapping *before* `attention_requantizer_generator`. Previously, the generator operated on raw HF names (e.g. `.self_attn.q_a_proj.`) whereas the quant-filtering logic checked for mapped names (e.g. `.attn.wq_a.`), causing the requantization to be silently skipped.
  - **Shared Experts down_proj Mapping**: Added missing Hugging Face `shared_experts.down_proj` mapping to `shared_experts.w2` in `WrappedMapper.apply()`, resolving missing weight key loader issues for shared MoE experts.
  - **Quantization Scale Naming Alignment**: Updated requantization generator to output scale keys with the `.weight_scale` suffix to align with the parameter naming expected by the patched linear layers.
  - **Decoder Layer Weight Loader Hooking**: Explicitly registered `patched_decoder_load_weights` to `DeepseekV4DecoderLayer.load_weights` for both CUDA and AMD classes. Previously, the loader patch was unregistered, which fell back to the default vLLM loader that does not handle TPU fused weights.
  - **Narrow Suffix Checks in Weight Loader**: Restructured substring checks in `patched_decoder_load_weights` (for `wq_b.weight`, `wo_a.weight`, `wo_b.weight`) from broad `"string" in name` checking to strict `name.endswith("string")` matching. This prevents scale keys (e.g. `weight_scale`) from being incorrectly matched and loaded into weight parameters.
  - **Hash-MoE TopK ID Shape Mismatch**: Fixed a 1D tensor shape mismatch in `moe_routing.py` by unsqueezing `topk_ids` to 2D `[num_tokens, 1]` before calling `scores.gather`.
  - **Implicit parameter recovery in select_experts**: Added frame stack inspection inside `moe_routing.select_experts` to dynamically recover `layer` and `input_ids` parameters when unquantized/nvfp4 quant-wrapper calls omit them.
  - **SWA Dummy Cache Dtype alignment**: Unified dummy SWA cache initialization in `forward_mqa` to use `torch.uint8` rather than `q.dtype` (which was bfloat16), preventing tracer type tracing errors in JAX.
  - **Compressor Block Table Safety**: Added safe checks for `block_tables` in `VllmDeepseekCompressor.forward` to prevent crash on dummy/profile runs when `block_tables` is `None`.
- **Result:**
  - Pytest correctness unit tests for MLA attention, sliding window attention, compressor, MoE routing, and StreamIndex TopK all completed successfully.

---

## Appendix: PyTorch Dynamo / Pallas Tracing Issue Reference

_Merged from the repo-root `development_log.md` (session 8 cleanup) — a curated, issue-by-issue technical reference distinct from the chronological log above, focused on PyTorch Dynamo/FX tracing failures encountered wiring Pallas kernels through `torch_tpu`'s TPU compiler path._

### Overview
Technical log of issue root causes, failure mechanics, and implemented code fixes for running DeepSeek V4 inference on TPU v7x-8 within the `vllm-torchtpu` framework.

### Current "Best" Command
```bash
ssh -o IdentitiesOnly=yes -i /Users/patemotter/.ssh/google_compute_engine alyssa \
  "sudo docker run --init --privileged --net=host --ipc=host --rm \
  -v /home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu:/usr/local/lib/python3.12/site-packages/vllm_torchtpu \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e HF_HOME=/home/patemotter_google_com/workspace/hf_cache \
  -e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so \
  -e LIBTPU_INIT_ARGS=\"--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=1000000000 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1000000000\" \
  -e TORCH_DIST_TIMEOUT=120 -e VLLM_TP_SIZE=8 -e VLLM_EP_SIZE=8 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e PYTHONWARNINGS=ignore -e TRANSFORMERS_VERBOSITY=error \
  -e VLLM_FP4_CACHE_DIR=/dev/shm/dsv4-flash-real-fp4 \
  -e JAX_COMPILATION_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache \
  -e DSV4_PROMPT=\"Compare XLA and LLVM in two bullet points.\" \
  -v /dev/shm:/dev/shm -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  vllm-torchtpu-local:latest \
  bash -c 'python3 -u /home/patemotter_google_com/workspace/vllm-torchtpu/examples/run_dsv4_inference.py'"
```

---

### Technical Issues & Resolved Fixes

#### 1. Custom `libtpu.so` Integration via `TPU_LIBRARY_PATH`
- **Failure Mechanics**: The container default `/usr/local/lib/python3.12/site-packages/libtpu/libtpu.so` rejected `--xla_tpu_use_dynamic_smem_negotiation`, raising `ERROR: Unknown command line flag`.
- **Resolution**: Specified `-e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so` in Docker container execution commands.

#### 2. `VllmQuantConfig` Property Assertion
- **Failure Mechanics**: `tpu_inference` quantization layers queried `quant_config.mesh`. The FP4 quantization config class `VllmQuantConfig` lacked a `.mesh` getter property, raising `AttributeError`.
- **Resolution**: Added `@property def mesh(self): return None` to `VllmQuantConfig` in `src/vllm_torchtpu/layers/vllm/quantization/configs.py`.

#### 3. JAX `NamedSharding` Weight Post-Processing
- **Failure Mechanics**: `VllmUnquantizedEmbeddingMethod` and `VllmUnquantizedLinearMethod` attempted JAX array sharding post-processing during PyTorch weight loading.
- **Resolution**: Monkey-patched weight post-processing in `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py` to bypass JAX sharding during PyTorch parameter initialization.

#### 4. PyTorch Dynamo MoE Output Reduction Context Assertion
- **Failure Mechanics**: `VllmMoERunner._maybe_reduce_final_output` asserted wrapper context (`vmwc.get_vllm_model_wrapper_context`), which is not present during PyTorch Dynamo graph tracing.
- **Resolution**: Patched `vmwc.get_vllm_model_wrapper_context` and substituted `VllmMoERunner._maybe_reduce_final_output` with `MoERunner._maybe_reduce_final_output` in `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py`.

#### 5. Pallas Kernel Ceiling Division Tracing Assertion (`cdiv`)
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

#### 6. Zero-Size Axis Reduction Exception (`s_rowmax = jnp.max`)
- **Failure Mechanics**: In `flash_attention`, `s_rowmax = jnp.max(s, axis=1, keepdims=True)` evaluated `s` with shape `(bq_sz, 0)` during empty/uninitialized SWA cache shape tracing. In JAX/NumPy, maximum reduction over a 0-length axis raises `ValueError: zero-size array to reduction operation max which has no identity`.
- **Resolution**: Added explicit trace-time dimension checks in `src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py` and `src/vllm_torchtpu/kernels/deepseek_v4/mla.py`:
  ```python
  if s.shape[1] == 0:
      s_rowmax = jnp.full((s.shape[0], 1), -jnp.inf, dtype=s.dtype)
  else:
      s_rowmax = jnp.max(s, axis=1, keepdims=True)
  ```

#### 7. Zero-Width Broadcast Concatenation Exception (`broadcast_minor`)
- **Failure Mechanics**: When `s.shape[1] == 0` (e.g. `s.shape = (bq_sz, 0)` during empty SWA cache tracing), `broadcast_minor(m_curr, s.shape)` calculated `target_minor = align_to(0, src.shape[-1]) = 0`. Evaluating `range(0 // src.shape[-1])` yielded an empty list `[]`, causing `jnp.concatenate([], axis=-1)` to raise `ValueError: Need at least one array to concatenate.`.
- **Resolution**: Added zero-shape handling to `broadcast_minor` in `src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py` and `src/vllm_torchtpu/kernels/deepseek_v4/mla.py`:
  ```python
  if shape[-1] == 0:
      return jnp.zeros(shape, dtype=src.dtype)
  ```

---

### Architectural Comparison: PyTorch Dynamo Symbolic Tracing vs. Native JAX Staging

#### Why PyTorch Dynamo Tracing Triggers Pallas Kernel Failures

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

#### Code Examples: Native JAX vs. PyTorch FX Tracing

##### Example A: Ceiling Division (`cdiv`)
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

##### Example B: Axis Max Reduction (`flash_attention`)
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

##### Example C: Minor Dimension Broadcast (`broadcast_minor`)
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

### 8. MLIR Compiler Failure on Zero-Sized Vector Dimensions (`logical_page_size=0`)
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

### 9. `NameError: name 'load_with_init' is not defined` in `mla.py`
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

### 10. `TypeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)` in `mla.py`
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

#### Issue #11: `TypeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)`

- **Root Cause**:
  In `mla.py` and `mla_swa.py`, `run_mla_kernel` passed hardcoded `static_q_len = chunk_prefill_size` (64) during prefill execution. When vLLM compiled smaller prompt sequence ranges like `(16, 16)` (`q.shape[0] = 2` tokens per TP rank), `m_scratch` VMEM register buffer was allocated with shape `(q.shape[0] * num_q_heads, 128)` = `(8, 128)`. However, `static_q_len = 64` forced `bq_sz = min(8, 64) = 8`, causing `attention_sinks` concatenation inside `_kernel_body` to allocate `8 * 8 = 64` rows, producing a shape mismatch during `attention_sinks` `(64, 1)` - `m_ref` `(8, 128)` subtraction.
- **Fix**:
  Updated `static_q_len` in prefill `run_mla_kernel` calls in [mla.py:L1052](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L1052) and [mla_swa.py:L1294](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla_swa.py#L1294) to `static_q_len = min(chunk_prefill_size, q.shape[0])`, ensuring `bq_sz` dynamic tile scaling matches actual VMEM array shapes across all sequence compile ranges.

---

#### Issue #12: Build-time `RuntimeError` on non-SWA layers & `sm_scale` attribute resolution

- **Symptom**:
  ```text
  RuntimeError: model.layers.0.attn: swa_cache_layer.block_size unavailable at attention-op build time
  RuntimeError: model.layers.0.attn: attention scale unavailable at attention-op build time
  ```
- **Root Cause**:
  1. In `_build_attn_op()`, an explicit validation check raised `RuntimeError` when `swa_cache_layer.block_size` was absent. However, non-SWA layers (such as CSA layer 0 with `compress_ratio=4`) do not construct `swa_cache_layer` (`swa_cache = None`), causing the validation check to fail on layer 0 during model construction.
  2. `sm_scale` resolution queried `getattr(self.mla_attn, "scale", None)`, whereas `DeepseekV4Attention` defines the attention scaling factor under `self.softmax_scale`.
- **Fix**:
  1. Updated `_build_attn_op` in [custom_ops/deepseek_v4.py:L920-L930](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L920-L930) to guard `if swa_cache is not None:`, validating `swa_cache.block_size` only for SWA layers and providing fallback `16` for non-SWA layers.
  2. Expanded `sm_scale` resolution in [custom_ops/deepseek_v4.py:L933-L940](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L933-L940) to check `scale`, `softmax_scale`, and fallback to `head_dim ** -0.5`.

---

#### Issue #13: `TypeError: TpuCompilerAdaptor.__init__() takes 1 positional argument but 2 were given`

- **Symptom**:
  ```text
  RuntimeError: Worker failed with error 'TpuCompilerAdaptor.__init__() takes 1 positional argument but 2 were given'
  ```
- **Root Cause**:
  In `compilation/tpu_compiler.py`, `TpuCompilerAdaptor.__init__(self)` accepted zero positional arguments. When vLLM v1 initialized the compiler plugin interface or when `compute_tpu_compilation_hash` instantiated `TpuCompilerAdaptor(vllm_config)`, passing `vllm_config` caused a `TypeError`.
- **Fix**:
  Updated `TpuCompilerAdaptor.__init__` in [compilation/tpu_compiler.py:L148](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/compilation/tpu_compiler.py#L148) to `def __init__(self, vllm_config: VllmConfig | None = None):`, storing `self.vllm_config = vllm_config`.

---

#### Issue #14: `RuntimeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)`

- **Symptom**:
  ```text
  RuntimeError: Worker failed with error 'sub got incompatible shapes for broadcasting: (64, 1), (8, 128).
  While executing %deepseek_v4_attention_csa : [num_users=2] = call_function[target=torch.ops.pallas.deepseek_v4_attention_csa.default](args = ...
  ```
- **Root Cause**:
  When static query shapes (e.g., `q.shape[0] = 64` or `positions.shape[0] = 64`) were passed into `attention_impl` for CSA layers, `self.indexer` returned `topk_indices` matching unpadded prompt token count (e.g., 8 rows: shape `(8, 128)`). In `forward_mqa()`, `max_valid_kv` was computed from `positions` (shape `(64, 1)`), leading to an invalid shape broadcast `(8, 128) < (64, 1)` inside XLA.
- **Fix**:
  Replaced conditional padding logic with unconditional tensor alignment in [custom_ops/deepseek_v4.py:L1188-L1189](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L1188-L1189) using `topk_indices = torch.cat([topk_indices, fill_pad], dim=0)[:positions.shape[0]]`. This guarantees `topk_indices` has exactly `positions.shape[0]` rows across both eager warmup and PyTorch Dynamo graph tracing.

#### Issue 15: Dynamo Symbolic Guard Violation and Pallas CSA MLIR Export Shape Mismatch
- **Status**: FIXED
- **Root Cause**:
  Hardcoding integer literal `64` inside `_fake_indexer` during PyTorch Dynamo tracing caused Dynamo to generate constant shape guards (`positions.shape[0] == 64`), triggering `RuntimeError: Constraints violated: You marked L['positions'].size()[0] as dynamic but your code specialized it to be a constant (64)`. Additionally, unpadded indexer outputs led to out-of-bounds reads during Pallas MLIR graph interpretation (`sub got incompatible shapes for broadcasting: (64, 1), (8, 128)`).
- **Fix**:
  1. Preserved dynamic symbolic dimensions in `_fake_indexer` in [custom_ops/deepseek_v4.py:L742](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L742) by returning `(q_rope.shape[0], self.topk_tokens)`.
  2. Dynamically aligned JAX indexer output in `_deepseek_v4_indexer_jax` in [custom_ops/deepseek_v4.py:L668](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L668) using `q_rope.shape[0]`.
  3. Added symbolic row padding via `torch.cat([topk_indices, fill_pad], dim=0)[:positions.shape[0]]` in [custom_ops/deepseek_v4.py:L792-L793](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4.py#L792-L793) to align `topk_indices.shape[0]` with `positions.shape[0]` symbolically without introducing static integer specialization guards.

#### Issue 17: Pallas MLA Kernel Attention Sinks Query-Block Concatenation Broadcast Failure in Decode
- **Status**: FIXED
- **Root Cause**:
  In `_mla_ragged_paged_attention_kernel` inside [kernels/deepseek_v4/mla.py:L613](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L613), `attention_sinks` was constructed via `jnp.concat([attention_sinks_ref[...] for _ in range(bq_sz)])[..., None]`. This concatenated the 8 local head sinks `bq_sz` times along axis 0, yielding shape `(bq_sz * num_q_heads, 1)` (e.g., `(64, 1)` for `bq_sz=8`). When subtracted from FlashAttention running row-max `m` (shape `(bq_sz, num_l_heads)` = `(8, 128)`), JAX raised `TypeError: sub got incompatible shapes for broadcasting: (64, 1), (8, 128)`. During decode execution where `q` is sharded across heads (`data_spec = PartitionSpec(None, "model")`), all queries in the decode batch remain on each chip, causing `bq_sz` to scale up and trigger shape mismatches.
- **Fix**:
  Updated `attention_sinks` evaluation in [kernels/deepseek_v4/mla.py:L613](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L613) to `sinks[None, :]`, converting the 1D head sink tensor into a 2D row tensor `(1, num_l_heads)`. This broadcasts cleanly against `m` (shape `(bq_sz, num_l_heads)`) across all prefill and decode query block sizes (`bq_sz`).

---

#### Issue #18: `TypeError: sub got incompatible shapes for broadcasting: (1, 64), (8, 128)`

- **Status**: FIXED
- **Root Cause**:
  In `mla_ragged_paged_attention` inside [kernels/deepseek_v4/mla.py:L822](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L822), `attention_sinks` was padded to `q.shape[1] - actual_num_q_heads` (which evaluates to 0 since `q.shape[1] == 8`). When `attention_sinks` was un-sharded or sharded differently, its length remained 64 instead of `num_l_heads = 128`. In `_mla_ragged_paged_attention_kernel`, `sinks[None, :]` produced shape `(1, 64)`, failing to broadcast against running row-max `m` (shape `(bq_sz, 128)`).
- **Fix**:
  Updated `attention_sinks` padding in [kernels/deepseek_v4/mla.py:L820-L825](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L820-L825) to `jnp.pad(attention_sinks[:actual_num_q_heads], (0, num_l_heads - actual_num_q_heads), constant_values=-jnp.inf)`. This forces `attention_sinks` to always have shape `(128,)`, enabling clean `(1, 128)` vs `(bq_sz, 128)` broadcasting in the Pallas MLA kernel.

---

#### Issue #19: `RuntimeError: Can only load scalars from SMEM`

- **Status**: FIXED
- **Root Cause**:
  In HCA execution inside `_mla_ragged_paged_attention_kernel` ([kernels/deepseek_v4/mla.py:L538](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L538)), `kv_lens_to_attend_ref` resides in TPU SMEM (Scalar Memory). TPU architecture only permits loading single scalar values from SMEM. Attempting vector slicing via `lax.dynamic_slice_in_dim` or whole-array dereference `kv_lens_to_attend_ref[...]` caused `RuntimeError: Can only load scalars from SMEM`.
- **Fix**:
  Updated `kv_lens_to_attend_ref` reading in [kernels/deepseek_v4/mla.py:L538](file:///Users/patemotter/dsv4/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/mla.py#L538) to unrolled scalar SMEM reads `jnp.stack([kv_lens_to_attend_ref[q_len_start + i] for i in range(bq_sz)])`.
