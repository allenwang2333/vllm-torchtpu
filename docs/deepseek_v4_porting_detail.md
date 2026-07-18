# DeepSeek v4 TPU Porting: Detailed Technical Specification

This document provides a comprehensive, task-by-task technical specification for integrating DeepSeek v4 support into the native PyTorch-based `torchtpu-vllm` codebase. It details the mechanical rationale ("why") and the exact implementation steps ("how") for every task in the integration blueprint.

---

## Phase 1: Porting Pallas Kernels

### Task 1.1: Create target directory `src/vllm_torchtpu/kernels/experimental/deepseek_v4`
*   **Why:** DeepSeek v4 uses Multi-Head Latent Attention (MLA) and a specialized compression layer. The Pallas kernels for these operations must be isolated from the general-purpose attention kernels (e.g., standard Ragged Paged Attention) to maintain modularity and prevent compilation path interference.
*   **How:** 
    1. Create the physical directory: `src/vllm_torchtpu/kernels/experimental/deepseek_v4`.
    2. Create an empty `__init__.py` file inside it to register the directory as a Python package.

### Task 1.2: Port pure JAX/Pallas kernels from `tpu-inference`
*   **Why:** DeepSeek v4 achieves high efficiency by compressing Key-Value states into low-rank latents. This requires custom TPU hardware-level kernels for:
    *   `compressor.py`: Compresses and normalizes query/key/value tensors, applying Rotary Position Embeddings (RoPE).
    *   `compress_norm_rope.py`: Concrete implementation of the norm and RoPE operations.
    *   `compress_store.py`: Memory-store operations to write compressed latents to HBM.
    *   `mla.py`: High-performance Multi-Head Latent Attention kernel.
    *   `mla_swa.py`: Sliding Window Attention variant of the MLA kernel.
*   **How:**
    1. Copy the JAX files directly from the existing [tpu_inference/kernels/experimental/deepseek_v4/](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/) directory to the new target `src/vllm_torchtpu/kernels/experimental/deepseek_v4/` directory.
    2. **Parity Check:** Ensure that imports of shared utilities (e.g., `jax.experimental.pallas`) are resolved. These files contain pure functional JAX code and require no framework-level modifications because they compile directly to XLA HLO.

### Task 1.3: Port unit tests
*   **Why:** Functional and numerical parity must be verified at the kernel level before hooking them into the PyTorch graph. Porting tests ensures we can isolate and debug numerical drift or compilation failures in the Pallas kernels.
*   **How:**
    1. Copy `compressor_test.py`, `mla_test.py`, and `mla_swa_test.py` from the existing [tests/kernels/deepseek_v4/](file:///Users/patemotter/dsv4/tpu-inference/tests/kernels/deepseek_v4/) directory to the new target `tests/kernels/deepseek_v4/` directory.
    2. Resolve the package imports to point to `vllm_torchtpu` instead of `tpu_inference`.
    3. Run the tests using `pytest` to confirm successful compilation and numerical correctness on TPU.

---

## Phase 2: PyTorch Custom Ops & Attention Backend

### Task 2.1 & 2.2: Create and Port `PallasMLAttentionBackend`
*   **Why:** `vllm` abstracts attention execution using backend classes. We must provide a backend (`PallasMLAttentionBackend`) and an implementation class (`PallasMLAttentionBackendImpl`) that conform to `vllm`'s `AttentionBackend` and `MLAAttentionImpl` interfaces respectively.
*   **How:**
    1. Create the new file `src/vllm_torchtpu/layers/vllm/backends/flash_attn_mla.py`.
    2. Port the structure of `PallasMLAttentionBackend` from the existing [tpu-inference file](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py), defining:
        *   `get_name()`: Returns `"FLASH_ATTN_MLA"`.
        *   `get_impl_cls()`: Returns `PallasMLAttentionBackendImpl`.
        *   `get_page_size()`: Returns the static page size (e.g., 1024).

### Task 2.3: Refactor to use `pallas.jax_op` (No `torchax.interop`)
*   **Why:** In the JAX middle-layer codebase (`tpu-inference`), PyTorch-to-JAX execution relied on `torchax.interop` to cast PyTorch-XLA tensors into JAX array views, and vice versa. In `torchtpu-vllm`, we must use `torch_tpu._internal.pallas.jax_op` to register the JAX kernel as a native PyTorch custom operator. This keeps PyTorch's Dynamo tracer from inspecting JAX internals, preventing graph breaks during `torch.compile`.
*   **How:**
    1. In the new `flash_attn_mla.py` file, import the Pallas custom operator builder:
        ```python
        from torch_tpu._internal import pallas
        ```
    2. Instantiate the custom operator with a unique name, passing the JAX entrypoint function and marking the first argument (KV Cache) for in-place donation:
        ```python
        mla_kernel_op = pallas.jax_op(
            "pallas::mla_kernel",
            jax_mla_entrypoint_fn,
            donate_argnums=(0,)
        )
        ```
    3. Register a fake (meta) implementation to allow Dynamo to trace shapes during the compilation phase:
        ```python
        @mla_kernel_op.register_fake
        def _mla_kernel_fake(kv_cache, q_nope, q_pe, *args, **kwargs):
            # Returns empty tensors matching the output shapes of the real kernel
            return torch.empty_like(kv_cache), torch.empty_like(q_nope)
        ```
    4. In `PallasMLAttentionBackendImpl.forward()`, invoke the custom operator:
        ```python
        # Tensors q_nope, q_pe, kv_cache are passed as pure PyTorch-XLA tensors.
        # The operator executes the compiled JAX/Pallas kernel directly on HBM.
        new_kv_cache, output = mla_kernel_op(kv_cache, q_nope, q_pe, ...args)
        kv_cache.copy_(new_kv_cache)  # Apply in-place KV cache update
        return output
        ```

### Task 2.4: Register backend in `TpuPlatform`
*   **Why:** When `vllm` encounters a model utilizing Multi-Head Latent Attention (like DeepSeek v4), the engine queries the active platform to resolve the backend for the `FLASH_ATTN_MLA` enum.
*   **How:**
    1. In the existing [tpu_platform.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/platforms/tpu_platform.py#L142) file, locate `pre_register_and_update()`.
    2. Add the registration hook:
        ```python
        register_backend(
            AttentionBackendEnum.FLASH_ATTN_MLA,
            "vllm_torchtpu.layers.vllm.backends.flash_attn_mla.PallasMLAttentionBackend",
        )
        ```

---

## Phase 3: Quantization & Model Loading Patches

### Task 3.1: Define `DSV4_FP8` Constant
*   **Why:** DeepSeek v4 checkpoints utilize a specialized FP8/FP4 mixed quantization scheme. The format name must be registered as a constant to act as a dispatch key during model loading.
*   **How:**
    1. Open the existing [quant_methods.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/common/quant_methods.py) file.
    2. Add the constant:
        ```python
        DSV4_FP8 = "deepseek_v4_fp8"
        ```

### Task 3.2 & 3.3: Port and Register `VllmDeepseekV4Fp8Config`
*   **Why:** The quantization configuration manages how weights are unpacked and which math kernels are executed. We must port the class that maps model layers to their TPU-optimized quantization methods (such as MXFP4 for MoE experts and block-scaled FP8 for linear layers).
*   **How:**
    1. Create the new file `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py`.
    2. Port `VllmDeepseekV4Fp8Config` from the existing [tpu-inference quantization config file](file:///Users/patemotter/dsv4/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py), ensuring it subclasses `DeepseekV4FP8Config` and `VllmQuantConfig`.
    3. In the existing [__init__.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/layers/vllm/quantization/__init__.py) file, import the new config and add it to `method_to_config`:
        ```python
        from vllm_torchtpu.layers.vllm.quantization.deepseek_v4_fp8 import VllmDeepseekV4Fp8Config
        
        # Inside get_tpu_quantization_config() -> method_to_config
        quant_methods.DSV4_FP8: VllmDeepseekV4Fp8Config
        ```

### Task 3.4: Port `VllmDeepseekV4MLAAttention` and Class Rebinding
*   **Why:** The upstream `DeepseekV4Attention` constructor allocates CUDA events (`torch.cuda.Event`) and streams. This triggers immediate runtime failures on TPU. We must subclass it to bypass these allocations and swap in the `PallasMLAttentionBackend`.
*   **How:**
    1. Create the new file `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4_attention.py`.
    2. Define `VllmDeepseekV4MLAAttention` subclassing `DeepseekV4Attention` (imported from `vllm.models.deepseek_v4.attention`).
    3. Override `__init__` to bypass CUDA allocations, calling `nn.Module.__init__(self)` directly and setting head dimensions.
    4. Implement `get_attn_backend()` to return `PallasMLAttentionBackend`.
    5. Implement `patch_deepseek_v4_mla_cls()`:
        ```python
        def patch_deepseek_v4_mla_cls() -> None:
            import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
            ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention = VllmDeepseekV4MLAAttention
        ```

### Task 3.5: Implement Model Loader Monkey-Patches
*   **Why:** Upstream `vllm`'s model registry defaults to the NVIDIA variant of DeepSeek v4, which relies on CUDA-specific cutedsl operators. We must force `vllm` to load the AMD/ROCm variant instead because the AMD path utilizes standard, extensible modules that allow us to cleanly rebind the attention class symbols. Additionally, we must mock out CUDA streams during construction.
*   **How:**
    1. In the existing [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py) file, define the context manager `_maybe_patch_for_deepseek_v4()`:
        ```python
        @contextmanager
        def _maybe_patch_for_deepseek_v4(vllm_config: VllmConfig):
            architectures = getattr(vllm_config.model_config.hf_config, "architectures", None) or []
            if not any("DeepseekV4ForCausalLM" in arch for arch in architectures):
                yield
                return
            
            # 1. Force the registry to resolve to the AMD/ROCm model variant
            import vllm.models.deepseek_v4 as ds_v4
            from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM as _AmdDeepseekV4ForCausalLM
            ds_v4.DeepseekV4ForCausalLM = _AmdDeepseekV4ForCausalLM
            
            # 2. Clear vLLM's internal loader caches to force reloading
            from vllm.model_executor.model_loader import utils as _ml_utils
            from vllm.model_executor.models.registry import _try_load_model_cls
            _ml_utils._MODEL_ARCH_BY_HASH.clear()
            _try_load_model_cls.cache_clear()
            
            # 3. Apply the class symbol rebinding for the attention layer
            from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4_attention import patch_deepseek_v4_mla_cls
            patch_deepseek_v4_mla_cls()
            
            # 4. Mock torch.cuda.Stream to prevent runtime errors during instantiation
            _orig_stream = torch.cuda.Stream
            torch.cuda.Stream = lambda *args, **kwargs: None
            
            try:
                # 5. Temporarily mock current_platform.device_type to 'cpu' to allow
                # shape/buffer initialization without TPU device-binding errors
                with patch.object(current_platform, "device_type", "cpu"):
                    yield
            finally:
                torch.cuda.Stream = _orig_stream
        ```
    2. Wrap the `model_loader.load_model()` call in `TPUModelRunner.load_model()` using this context manager:
        ```python
        with _maybe_patch_for_deepseek_v4(self.vllm_config), \
             set_vllm_model_wrapper_context(mesh=self.mesh), \
             set_current_vllm_config(self.vllm_config):
            model = model_loader.load_model(...)
        ```

---

## Phase 4: KV Cache & Runner Alignment

### Task 4.1: Update `get_kv_cache_spec()` in `tpu_runner.py`
*   **Why:** The `TPUModelRunner` evaluates the model's layers to determine how much HBM to allocate for the KV cache pages. Since `VllmDeepseekV4MLAAttention` does not subclass the standard `Attention` or `MLAAttention` layers (it subclasses `DeepseekV4Attention` directly), the runner's parser will skip it. We must explicitly intercept this module, extract its specialized MLA cache dimensions, and register them.
*   **How:**
    1. Locate `get_kv_cache_spec()` in the existing [tpu_runner.py](file:///Users/patemotter/dsv4/torchtpu-vllm/src/vllm_torchtpu/runner/tpu_runner.py#L553) file.
    2. Add an explicit check for the DeepSeek v4 attention class:
        ```python
        from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4_attention import VllmDeepseekV4MLAAttention
        
        # Inside the layer loop in get_kv_cache_spec():
        if isinstance(attn_module, VllmDeepseekV4MLAAttention):
            spec = attn_module.get_kv_cache_spec(self.vllm_config)
            assert spec is not None
            # DeepSeek v4 requires head dimensions to be aligned to 128 bytes
            aligned_head_size = common_utils.align_to(spec.head_size, 128)
            kv_cache_spec[layer_name] = MLAAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=1,
                head_size=aligned_head_size,
                dtype=spec.dtype,
                cache_dtype_str=spec.cache_dtype_str,
                page_size_padded=aligned_head_size * spec.block_size, # Or custom formula
            )
            continue
        ```
