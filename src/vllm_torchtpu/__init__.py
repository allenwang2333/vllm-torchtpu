# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm_torchtpu.env_override  # noqa: F401
from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _patch_vllm_aot_compile_cache_key() -> None:
    """Include the TPU compiler hash in vLLM's outer AOT cache key."""
    from vllm.compilation import caching

    original = caching.aot_compile_hash_factors
    if getattr(original, "_tpu_compiler_hash_patch", False):
        return

    def tpu_aot_compile_hash_factors(vllm_config):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash
        return [
            *original(vllm_config),
            compute_tpu_compilation_hash(vllm_config),
        ]

    tpu_aot_compile_hash_factors._tpu_compiler_hash_patch = True
    caching.aot_compile_hash_factors = tpu_aot_compile_hash_factors
    logger.info("Applied TPU patch: include compiler hash in AOT cache key.")


def _patch_vllm_config_hash_ignore_diagnostics() -> None:
    """Keep diagnostics-only ``additional_config`` keys out of cache keys.

    ``VllmConfig.compute_hash()`` unconditionally folds
    ``json.dumps(additional_config)`` into its result, and that hash feeds both
    the AOT compile cache key (``caching.aot_compile_hash_factors``) and the
    piecewise cache dir (``backends.py``). The phased profiler is configured
    through ``additional_config``, so a per-run trace directory would force a
    full recompile on every server start. Upstream takes the same care in
    ``ProfilerConfig.compute_hash()`` ("this config will not affect the
    computation graph"); this is the equivalent carve-out for the plugin-owned
    keys, mirroring ``_TPU_COMPILE_ENV_IGNORED`` for env vars.
    """
    import copy

    from vllm.config import VllmConfig

    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    if getattr(VllmConfig, "_tpu_additional_config_hash_patch", False):
        return

    original = VllmConfig.compute_hash

    def compute_hash(self) -> str:
        additional_config = self.additional_config
        if not isinstance(additional_config, dict):
            return original(self)
        filtered = {
            key: value
            for key, value in additional_config.items()
            if key not in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
        }
        if filtered == additional_config:
            return original(self)
        # Hash a shallow copy with the filtered dict rather than mutating the
        # live config: compute_hash() may run while other code holds the same
        # VllmConfig, and a shallow copy shares every sub-config by reference
        # (cheap, no re-validation -- __post_init__ is not re-run) while giving
        # compute_hash its own additional_config to read.
        proxy = copy.copy(self)
        proxy.additional_config = filtered
        return original(proxy)

    VllmConfig._tpu_upstream_compute_hash = original
    VllmConfig.compute_hash = compute_hash
    VllmConfig._tpu_additional_config_hash_patch = True
    logger.info(
        "Applied TPU patch: exclude diagnostics-only additional_config keys "
        "%s from the compile cache key.",
        sorted(HASH_IGNORED_ADDITIONAL_CONFIG_KEYS))


def _patch_vllm_tpu_group_custom_ops() -> None:
    """Disable vLLM custom collective ops for TPU in this TorchTPU integration.

    vLLM's `torch.ops.vllm.all_reduce/all_gather` custom ops are not available
    on the TorchTPU backend in this repo. Force GroupCoordinator to use
    communicator-backed collectives instead.

    TODO (geyuhao): Do we actually need to support torch.ops.vllm.* on TPU? Currently
    this patch will use torch.distributed.all_reduce/all_gather.
    """

    from vllm.distributed.parallel_state import GroupCoordinator

    if getattr(GroupCoordinator, "_tpu_no_custom_collective_patch", False):
        return

    original_init = GroupCoordinator.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.use_custom_op_call = False

    GroupCoordinator.__init__ = patched_init
    GroupCoordinator._tpu_no_custom_collective_patch = True
    logger.info("Applied TPU patch: disable vLLM custom collective ops.")


def _patch_default_moe_runner_select_forward() -> None:
    """Keep DefaultMoERunner on the direct ``_moe_forward`` path under OOT.

    Upstream gates this on ``is_tpu()``; under our OOT plugin that's False
    and we'd take ``torch.ops.vllm.moe_forward``, which captures weight
    tensors as closures and trips torch_tpu's MLIR builder.
    """
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as _dmr

    if getattr(_dmr.MoERunner, "_tpu_select_forward_patch", False):
        return

    def patched_select(self):
        return (_dmr._moe_forward
                if self.shared_experts is None else _dmr._moe_forward_shared)

    _dmr.MoERunner._select_forward = patched_select
    _dmr.MoERunner._tpu_select_forward_patch = True
    logger.info(
        "Applied TPU patch: DefaultMoERunner uses direct _moe_forward.")


def _patch_expert_map_host_lookup() -> None:
    """Serve expert-map lookups from a host-side copy of the expert map.

    ``ExpertMapManager.map_global_to_local`` reads the device-resident
    expert map with ``.item()`` once per (expert, shard) tensor during MoE
    weight loading. On TPU every ``.item()`` is a blocking sync that
    flushes the deferred graph, so loading dispatches one tiny device
    program per expert copy (~200k per worker on Qwen3.5-397B). Index a
    cached host copy in Python instead; the cache follows reassignment
    and in-place rebalance via the tensor's version counter.
    """
    from vllm.model_executor.layers.fused_moe import expert_map_manager as _em

    cls = _em.ExpertMapManager
    if getattr(cls, "_tpu_host_expert_map_patch", False):
        return

    def map_global_to_local(self, global_id: int) -> int:
        expert_map = self._expert_map
        if expert_map is None:
            return global_id
        if (self._tpu_expert_map_source is not expert_map
                or self._tpu_expert_map_version != expert_map._version):
            self._tpu_expert_map_source = expert_map
            self._tpu_expert_map_version = expert_map._version
            self._tpu_expert_map_host = expert_map.tolist()
        return self._tpu_expert_map_host[global_id]

    cls.map_global_to_local = map_global_to_local
    cls._tpu_expert_map_source = None
    cls._tpu_expert_map_version = None
    cls._tpu_expert_map_host = None
    cls._tpu_host_expert_map_patch = True
    logger.info("Applied TPU patch: host-side expert-map lookup.")


def _patch_vllm_disable_compile_ranges() -> None:
    """Force TPU to compile only fixed bucketed sizes, never dynamic ranges.

    ``VllmConfig._set_compile_ranges()`` always appends
    ``scheduler_config.max_num_batched_tokens`` to ``compile_ranges_endpoints``,
    so ``PiecewiseBackend`` builds a dynamic ``Range(1, max)`` and compiles it
    with backed ``SymInt`` shapes. The torch_tpu backend rejects SymInts ("does
    not support dynamic shape" / "No shape env"). The TPU runner pads every shape
    to an exact ``compile_size``, so dynamic ranges are never needed.

    Patch the consumption point: ``CompilationConfig.get_compile_ranges() -> []``.
    Clearing the field (e.g. in the worker) does not stick --
    ``_set_compile_ranges`` rewrites it on every config reconstruction
    (Qwen3-VL ``with_hf_config`` -> ``replace``), and the compile-time
    compilation_config is not the instance a platform/worker hook can reach --
    so a class-level override is the robust fix.
    """
    from vllm.config.compilation import CompilationConfig

    def get_compile_ranges(self):
        return []

    CompilationConfig.get_compile_ranges = get_compile_ranges
    logger.info("Applied TPU patch: disable dynamic compile_ranges (static "
                "compile_sizes only).")


def _patch_rowparallel_defer_bias() -> None:
    """Defer RowParallelLinear bias past the TP all-reduce on TPU.

    Upstream ``RowParallelLinear.forward`` fuses bias into the GEMM **only on
    rank 0** (``bias_ = None if tp_rank > 0 ... else self.bias``). Under TP>1
    this makes the per-rank FX graphs structurally different, so the torch_tpu
    XLA backend assigns mismatched all_reduce channel IDs and the collective is
    broken.
    """
    from vllm.distributed import (split_tensor_along_last_dim,
                                  tensor_model_parallel_all_reduce)
    from vllm.model_executor.layers.linear import RowParallelLinear

    # Preserve the genuine upstream forward once so re-application (e.g. a
    # test that re-applies after mocking the all-reduce) never captures an
    # already-patched forward as the trunk reference.
    if not hasattr(RowParallelLinear, "_tpu_upstream_forward"):
        RowParallelLinear._tpu_upstream_forward = RowParallelLinear.forward

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        do_all_reduce = self.reduce_results and self.tp_size > 1
        # Defer the (replicated) bias past the all-reduce so every rank runs an
        # identical GEMM graph -> matching XLA all_reduce channel IDs.
        defer_tpu_bias = (do_all_reduce and not self.skip_bias_add
                          and self.bias is not None)
        if defer_tpu_bias:
            bias_ = None
        else:
            bias_ = (None if
                     (self.tp_rank > 0 or self.skip_bias_add) else self.bias)
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if do_all_reduce:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if defer_tpu_bias:
            output = output + self.bias

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    RowParallelLinear.forward = forward
    logger.info(
        "Applied TPU patch: defer RowParallelLinear bias past TP all_reduce.")


def _patch_disable_sequence_parallel_moe() -> None:
    """Disable vLLM's sequence-parallel MoE on TPU.

    The sequence-parallel path is tied to vLLM all2all MoE backends. The TPU
    communicator handles EP with unsplit token buffers.
    """
    from vllm.config.parallel import ParallelConfig

    if getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        return

    ParallelConfig.use_sequence_parallel_moe = property(lambda self: False)
    ParallelConfig._tpu_no_sp_moe_patch = True
    logger.info("Applied TPU patch: disable sequence-parallel MoE.")


def _patch_disable_dp_ubatch() -> None:
    """Disable DP microbatching (DBO) on TPU.

    TPU does not support vLLM DP microbatching yet. Force the post-sync ubatch
    decision off so DP uses one forward per engine step.
    """
    from vllm.v1.worker import dp_utils

    if getattr(dp_utils, "_tpu_no_ubatch_patch", False):
        return

    dp_utils._post_process_ubatch = lambda tensor, num_ubatches: False
    dp_utils._tpu_no_ubatch_patch = True
    logger.info("Applied TPU patch: disable DP microbatching.")


def _patch_torch_sdpa_for_tpu() -> None:
    """Pad head_dim for torch.nn.functional.scaled_dot_product_attention on TPU.

    PyTorch XLA's SDPA C++ kernel requires head_dim % 128 == 0, seq_len % 512 == 0,
    and attn_mask is None to use FLASH_ATTENTION. When any condition is violated
    (e.g., head_dim=72, seq_len=2520 in Gemma 4 vision encoder), it falls back to
    slow Math/SHLO attention which times out under DP on shared TPU hosts.
    For non-causal 4D vision attention on TPU, this patch delegates execution to
    vllm-torchtpu's custom Pallas vision flash attention kernel (TpuMMEncoderAttention._vision_op),
    which uses 128-block sequence padding and segment IDs.
    """
    import torch
    import torch.nn.functional as F

    if getattr(F, "_tpu_sdpa_padded_patch", False):
        return

    _orig_sdpa = F.scaled_dot_product_attention

    def _tpu_padded_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        is_tpu = query.device.type == "xla" or "tpu" in str(query.device)
        if is_tpu and not is_causal and query.dim() == 4:
            from vllm_torchtpu.layers.vllm.vision_attention import (
                TpuMMEncoderAttention,
                _BLOCK_SIZE,
                _vision_flash_core,
            )
            if TpuMMEncoderAttention._vision_op is None:
                from torch_tpu._internal import pallas
                op = pallas.jax_op("pallas::vision_flash_attention", _vision_flash_core)
                op.register_fake(lambda q, k, v, q_seg, kv_seg: torch.empty_like(q))
                TpuMMEncoderAttention._vision_op = op

            bsz, num_heads, seq_len, head_dim = query.shape
            target_dim = ((head_dim + 127) // 128) * 128
            pad_dim = target_dim - head_dim
            if scale is None:
                scale = 1.0 / (head_dim ** 0.5)

            if pad_dim > 0:
                q_4d = F.pad(query, (0, pad_dim))
                k_4d = F.pad(key, (0, pad_dim))
                v_4d = F.pad(value, (0, pad_dim))
            else:
                q_4d, k_4d, v_4d = query, key, value

            q_bshd = q_4d.transpose(1, 2)
            k_bshd = k_4d.transpose(1, 2)
            v_bshd = v_4d.transpose(1, 2)

            pad = (-seq_len) % _BLOCK_SIZE
            padded = seq_len + pad
            seg = torch.zeros((bsz, padded), dtype=torch.int32, device=query.device)
            seg[:, :seq_len] = 1

            def _prep(x):
                if pad:
                    x = F.pad(x, (0, 0, 0, 0, 0, pad))
                return x.permute(0, 2, 1, 3).contiguous()

            op = TpuMMEncoderAttention._vision_op
            out = op(_prep(q_bshd * scale), _prep(k_bshd), _prep(v_bshd), seg, seg)
            out = out.permute(0, 2, 1, 3)[:, :seq_len].contiguous()
            out_bhsd = out.transpose(1, 2)
            return out_bhsd[..., :head_dim]

        head_dim = query.shape[-1]
        if is_tpu and head_dim % 128 != 0:
            target_dim = ((head_dim + 127) // 128) * 128
            pad_amount = target_dim - head_dim
            if scale is None:
                scale = 1.0 / (head_dim ** 0.5)
            q_pad = F.pad(query, (0, pad_amount))
            k_pad = F.pad(key, (0, pad_amount))
            v_pad = F.pad(value, (0, pad_amount))
            out_pad = _orig_sdpa(
                q_pad, k_pad, v_pad,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs
            )
            return out_pad[..., :head_dim]

        return _orig_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs
        )

    F.scaled_dot_product_attention = _tpu_padded_sdpa
    torch.nn.functional.scaled_dot_product_attention = _tpu_padded_sdpa
    F._tpu_sdpa_padded_patch = True
    logger.info("Applied TPU patch: padded SDPA for head_dim %% 128 != 0.")


def _patch_moe_no_ep_tp_scope() -> None:
    """Keep non-EP MoE tensor parallelism scoped to each DP engine.

    Upstream vLLM flattens TP across DP for MoE when expert parallelism is
    disabled. This temporary TPU patch avoids that path until the TPU dummy
    batch path can keep no-EP MoE ranks in the same lockstep as upstream
    model-internal DP.
    """
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_no_ep_tp_scope_patch", False):
        return

    original_make = FusedMoEParallelConfig.make

    # TODO: Remove this patch in a follow-up PR. TorchTPU should match vLLM's
    # no-EP MoE convention by flattening DP*TP and extending the coordinated
    # dummy/lockstep execution used for EP to no-EP MoE as well.
    def patched_make(tp_size_, pcp_size_, dp_size_, sp_size_,
                     vllm_parallel_config):
        use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1
                  and vllm_parallel_config.enable_expert_parallel)
        if use_ep:
            return original_make(tp_size_, pcp_size_, dp_size_, sp_size_,
                                 vllm_parallel_config)

        dp_rank = (parallel_state.get_dp_group().rank_in_group
                   if dp_size_ > 1 else 0)
        pcp_rank = (parallel_state.get_pcp_group().rank_in_group
                    if pcp_size_ > 1 else 0)
        tp_rank = (0 if tp_size_ == 1 else
                   parallel_state.get_tensor_model_parallel_rank())
        return FusedMoEParallelConfig(
            tp_size=tp_size_,
            tp_rank=tp_rank,
            pcp_size=pcp_size_,
            pcp_rank=pcp_rank,
            dp_size=dp_size_,
            dp_rank=dp_rank,
            ep_size=1,
            ep_rank=0,
            sp_size=sp_size_,
            use_ep=False,
            all2all_backend=vllm_parallel_config.all2all_backend,
            enable_eplb=vllm_parallel_config.enable_eplb,
        )

    FusedMoEParallelConfig.make = staticmethod(patched_make)
    FusedMoEParallelConfig._tpu_no_ep_tp_scope_patch = True
    logger.info(
        "Applied TPU patch: scope non-EP MoE TP inside each DP engine.")


def _patch_multiproc_worker_global_rank_env() -> None:
    """Set TorchTPU binding env on spawned workers before torch_tpu import.

    Native vLLM DP passes per-engine rank/local_rank to workers and computes
    global torch.distributed rank later from ParallelConfig. TorchTPU currently
    reads RANK/LOCAL_RANK/WORLD_SIZE earlier for physical chip/PjRt binding, so
    this shim exposes the unified single-host DP*TP slice to TorchTPU without
    changing the worker rank arguments passed through vLLM.
    """
    import os as _os

    from vllm.v1.executor.multiproc_executor import WorkerProc

    from vllm_torchtpu.worker.tpu_rank_binding import get_tpu_worker_binding

    if getattr(WorkerProc, "_tpu_global_rank_env_patch", False):
        return
    _orig = WorkerProc.make_worker_process

    def _wrapped(vllm_config, local_rank, rank, *args, **kwargs):
        pc = vllm_config.parallel_config
        binding = get_tpu_worker_binding(pc, rank, local_rank, env=_os.environ)
        _os.environ.update(binding.as_env())
        logger.info(
            "Applied TPU patch: worker spawn env RANK=%d LOCAL_RANK=%d "
            "WORLD_SIZE=%d LOCAL_WORLD_SIZE=%d "
            "(rank=%d local_rank=%d dp_rank=%d dp_size=%d offset=%d "
            "init_local_rank=%d)", binding.rank, binding.local_rank,
            binding.world_size, binding.local_world_size, rank, local_rank,
            binding.dp_rank, binding.dp_size, binding.local_rank_offset,
            binding.init_local_rank)
        if binding.pcp_local_rank_remap is not None:
            logger.info(
                "Applied TPU patch: PCP native-rank worker spawn binding "
                "native_local_rank=%d local_rank_env=%d "
                "tpu_local_rank_env=%d local_world=%d tpu_local_world=%d "
                "remap=%s source=%s",
                binding.native_local_rank,
                binding.init_local_rank,
                binding.local_rank,
                binding.world_size,
                binding.local_world_size,
                binding.pcp_local_rank_remap,
                binding.pcp_remap_source,
            )
        _patch_tpu_platform_mem_get_info()
        return _orig(vllm_config, local_rank, rank, *args, **kwargs)

    WorkerProc.make_worker_process = staticmethod(_wrapped)
    WorkerProc._tpu_global_rank_env_patch = True
    logger.info("Applied TPU patch: MultiprocExecutor worker binding env.")


def _patch_tpu_platform_mem_get_info() -> None:
    """Provide a general fallback for current_platform.mem_get_info() on TPU.

    Multimodal vision models (Gemma 4 MM, etc.) query mem_get_info() during
    resolution bucket selection. On GPU this calls cudaMemGetInfo. On TPU we
    provide a static high-watermark estimate (64GB free / 96GB total).
    """
    try:
        from vllm.platforms import current_platform
        current_platform.mem_get_info = lambda *a, **kw: (64 * (1024**3), 96 *
                                                          (1024**3))
        logger.info(
            "Applied TPU patch: current_platform.mem_get_info fallback for multimodal models."
        )
    except Exception:
        pass


def _patch_vllm_unify_kv_cache_spec_page_size() -> None:
    """Fix page size unification in vLLM V1 engine for specs with padded page sizes.

    When unifying page sizes across KV cache specs by scaling block_size,
    dataclasses.replace must update page_size_padded to match max_page_size
    if page_size_padded is set; otherwise new_spec.page_size_bytes retains
    the old padded page size and trips an assertion error in page_size_bytes
    (assert self.page_size_padded >= real_page_size).
    """
    import sys
    from dataclasses import replace

    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import KVCacheSpec

    if getattr(kv_cache_utils, "_tpu_unify_page_size_patch", False):
        return

    def unify_kv_cache_spec_page_size(
        kv_cache_spec: dict[str, KVCacheSpec], ) -> dict[str, KVCacheSpec]:
        page_sizes = {
            layer.page_size_bytes
            for layer in kv_cache_spec.values()
        }
        if len(page_sizes) <= 1:
            return kv_cache_spec

        max_page_size = max(page_sizes)
        new_kv_cache_spec = {}
        for layer_name, layer_spec in kv_cache_spec.items():
            if layer_spec.page_size_bytes == max_page_size:
                new_kv_cache_spec[layer_name] = layer_spec
            else:
                layer_page_size = layer_spec.page_size_bytes
                if max_page_size % layer_page_size != 0:
                    raise NotImplementedError(
                        "The page size of the layer is not divisible by the "
                        "maximum page size. Cannot unify by adjusting block_size."
                    )
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                kwargs: dict[str, object] = {"block_size": new_block_size}
                if getattr(layer_spec, "page_size_padded", None) is not None:
                    kwargs["page_size_padded"] = max_page_size
                new_spec = replace(layer_spec, **kwargs)
                assert new_spec.page_size_bytes == max_page_size
                new_kv_cache_spec[layer_name] = new_spec
        return new_kv_cache_spec

    kv_cache_utils.unify_kv_cache_spec_page_size = unify_kv_cache_spec_page_size
    kv_cache_utils._tpu_unify_page_size_patch = True

    for module_name in ("vllm.v1.engine.core", "vllm.v1.core.kv_cache_utils"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "unify_kv_cache_spec_page_size",
                    unify_kv_cache_spec_page_size)

    logger.info(
        "Applied TPU patch: unify_kv_cache_spec_page_size page_size_padded handling."
    )


def _run_engine_core_with_tpu_patches(*args, **kwargs):
    _patch_vllm_hybrid_pcp_block_sizes()
    _patch_vllm_unify_kv_cache_spec_page_size()
    _patch_tpu_platform_mem_get_info()

    from vllm.v1.engine.core import EngineCoreProc

    original_run = EngineCoreProc._tpu_original_run_engine_core
    return original_run(*args, **kwargs)


def _patch_vllm_hybrid_pcp_block_sizes() -> None:
    """Resolve full-attention + Mamba block sizes under TPU PCP.

    Upstream vLLM rejects all multi-group KV cache configs when
    ``pcp_world_size > 1`` because it cannot infer the scheduler block size for
    mixed cache types. TorchTPU's Qwen3.5 path has a narrower, well-defined
    layout: each physical cache page is rank-local, while the scheduler and
    cache coordinator operate on logical pages spanning all PCP ranks. Present
    those logical page sizes to the generic hybrid coordinator instead of
    rejecting the config.
    """
    import math
    import sys
    from dataclasses import replace

    from vllm.v1.core import kv_cache_coordinator, kv_cache_utils
    from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheSpec,
                                            MambaSpec, UniformTypeKVCacheSpecs)

    already_patched = getattr(kv_cache_utils,
                              "_tpu_hybrid_pcp_block_sizes_patch", False)
    original_resolve = getattr(
        kv_cache_utils,
        "_tpu_original_resolve_kv_cache_block_sizes",
        kv_cache_utils.resolve_kv_cache_block_sizes,
    )

    def _iter_leaf_specs(spec: KVCacheSpec):
        if isinstance(spec, UniformTypeKVCacheSpecs):
            yield from spec.kv_cache_specs.values()
        else:
            yield spec

    def _has_spec_type(spec: KVCacheSpec, spec_type: type) -> bool:
        return any(
            isinstance(leaf, spec_type) for leaf in _iter_leaf_specs(spec))

    def _is_attention_mamba_hybrid(groups) -> bool:
        return any(
            _has_spec_type(group.kv_cache_spec, MambaSpec)
            for group in groups) and any(
                _has_spec_type(group.kv_cache_spec, AttentionSpec)
                for group in groups)

    def _with_effective_block_size(spec: KVCacheSpec, pcp: int) -> KVCacheSpec:
        """Restate a rank-local spec in the logical (all-ranks) page size."""
        if isinstance(spec, UniformTypeKVCacheSpecs):
            effective_specs = {
                name: _with_effective_block_size(leaf, pcp)
                for name, leaf in spec.kv_cache_specs.items()
            }
            effective_block_size = math.lcm(
                *(leaf.block_size for leaf in effective_specs.values()))
            return replace(spec,
                           block_size=effective_block_size,
                           kv_cache_specs=effective_specs)
        if isinstance(spec, (AttentionSpec, MambaSpec)):
            return replace(spec, block_size=spec.block_size * pcp)
        return spec

    def _effective_block_size(spec: KVCacheSpec, pcp: int) -> int:
        return _with_effective_block_size(spec, pcp).block_size

    def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        dcp = parallel_config.decode_context_parallel_size
        pcp = parallel_config.prefill_context_parallel_size
        groups = kv_cache_config.kv_cache_groups

        if len(groups) <= 1 or dcp != 1 or pcp == 1:
            return original_resolve(kv_cache_config, vllm_config)

        if not _is_attention_mamba_hybrid(groups):
            return original_resolve(kv_cache_config, vllm_config)

        group_block_sizes = [
            _effective_block_size(group.kv_cache_spec, pcp) for group in groups
        ]
        scheduler_block_size = math.lcm(*group_block_sizes)

        connector_enabled = vllm_config.kv_transfer_config is not None
        if not (cache_config.enable_prefix_caching or connector_enabled):
            hash_block_size = scheduler_block_size
        elif (not hasattr(cache_config, "prefix_match_unit")
              or cache_config.prefix_match_unit is None):
            hash_block_size = math.gcd(*group_block_sizes)
        else:
            hash_block_size = cache_config.prefix_match_unit
            if any(bs % hash_block_size != 0 for bs in group_block_sizes):
                raise ValueError(
                    f"Invalid prefix_match_unit={hash_block_size}; all logical "
                    "PCP KV cache group block sizes must be divisible by "
                    "prefix_match_unit. Got "
                    f"block sizes={group_block_sizes}.")

        logger.info(
            "Applied TPU PCP hybrid KV cache block-size resolution: "
            "pcp=%d group_effective_block_sizes=%s scheduler_block_size=%d "
            "hash_block_size=%d", pcp, group_block_sizes, scheduler_block_size,
            hash_block_size)
        return scheduler_block_size, hash_block_size

    patched_resolve = (kv_cache_utils.resolve_kv_cache_block_sizes
                       if already_patched else resolve_kv_cache_block_sizes)
    if not already_patched:
        kv_cache_utils._tpu_original_resolve_kv_cache_block_sizes = (
            original_resolve)
        kv_cache_utils.resolve_kv_cache_block_sizes = patched_resolve
        kv_cache_utils._tpu_hybrid_pcp_block_sizes_patch = True

    hybrid_coordinator = kv_cache_coordinator.HybridKVCacheCoordinator
    if not getattr(hybrid_coordinator, "_tpu_hybrid_pcp_coordinator_patch",
                   False):
        original_hybrid_init = hybrid_coordinator.__init__

        def _hybrid_init_with_logical_pcp_blocks(self,
                                                 kv_cache_config,
                                                 max_model_len,
                                                 max_num_batched_tokens,
                                                 use_eagle,
                                                 enable_caching,
                                                 enable_kv_cache_events,
                                                 dcp_world_size,
                                                 pcp_world_size,
                                                 scheduler_block_size,
                                                 hash_block_size,
                                                 metrics_collector=None):
            groups = kv_cache_config.kv_cache_groups
            if (pcp_world_size > 1 and _is_attention_mamba_hybrid(groups)):
                if dcp_world_size != 1:
                    raise ValueError("TPU hybrid PCP cache coordination does "
                                     "not support DCP.")
                # The generic coordinator only needs logical token
                # granularity. Encoding PCP into each group spec and then
                # passing pcp_world_size=1 avoids multiplying block sizes
                # twice and keeps its prefix-cache lookup, hashing, and
                # allocation invariants internally consistent.
                kv_cache_config = replace(
                    kv_cache_config,
                    kv_cache_groups=[
                        replace(group,
                                kv_cache_spec=_with_effective_block_size(
                                    group.kv_cache_spec, pcp_world_size))
                        for group in kv_cache_config.kv_cache_groups
                    ])
                pcp_world_size = 1

            original_hybrid_init(
                self,
                kv_cache_config,
                max_model_len,
                max_num_batched_tokens,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
            )

        hybrid_coordinator.__init__ = _hybrid_init_with_logical_pcp_blocks
        hybrid_coordinator._tpu_hybrid_pcp_coordinator_patch = True

    from vllm.v1.engine.core import EngineCoreProc

    # EngineCore imports the function directly, so patch module bindings as
    # well as the source module.
    for module_name in ("vllm.v1.engine.core", "vllm.v1.kv_offload.base"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "resolve_kv_cache_block_sizes", patched_resolve)

    if not getattr(EngineCoreProc, "_tpu_engine_core_patch_wrapper", False):
        EngineCoreProc._tpu_original_run_engine_core = (
            EngineCoreProc.run_engine_core)
        EngineCoreProc.run_engine_core = staticmethod(
            _run_engine_core_with_tpu_patches)
        EngineCoreProc._tpu_engine_core_patch_wrapper = True

    if not already_patched:
        logger.info("Applied TPU patch: hybrid full-attention + Mamba PCP "
                    "block sizes.")


if "proxy" in envs.JAX_PLATFORMS:
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import traceback

        import pathwaysutils
        import vllm
        from vllm.platforms import (resolve_current_platform_cls_qualname,
                                    resolve_obj_by_qualname)
        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")

        # Pathways requires eager resolution of vllm.current_platform instead of
        # lazy resolution in the normal code path. Since this part involves
        # global topology discovery across multiple hosts, the platform
        # resolution must happen before other components are loaded.
        logger.info("Eagerly resolving vLLM current_platform for Pathways.")
        platform_cls_qualname = resolve_current_platform_cls_qualname()
        resolved_platform_instance = resolve_obj_by_qualname(
            platform_cls_qualname)()
        vllm.platforms._current_platform = resolved_platform_instance
        vllm.platforms._init_trace = "".join(traceback.format_stack())
        logger.info(
            f"vLLM platform resolved to: {resolved_platform_instance.__class__.__name__}"
        )

    except Exception as e:
        logger.error(
            f"Error occurred while importing pathwaysutils or logging TPU info: {e}"
        )
