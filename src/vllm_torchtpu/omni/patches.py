# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni diffusion models."""

import dataclasses
import functools
import importlib
from collections.abc import Callable
from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Run ZImageTransformerBlock.forward through torch.compile(backend="tpu")
# (one compiled function shared by all blocks). Falls back to eager if the
# compiled path raises.
_ZIMAGE_BLOCK_COMPILE = True


def _unwrap_tpu_patch(fn: Any) -> Any:
    """Return the underlying unpatched callable if ``fn`` was wrapped by TPU patches."""
    if getattr(fn, "_is_tpu_patch_wrapper", False) is True:
        return fn._orig_tpu_fn
    return fn


def _mark_tpu_patch(wrapper: Any, orig: Any) -> Any:
    """Mark ``wrapper`` with a reference to ``orig`` so re-patching is idempotent.

    ``functools.wraps`` keeps ``__name__``/``__qualname__``/``inspect.signature``
    of the original for callers that introspect patched methods. It is skipped
    when ``orig`` has no string ``__name__`` (e.g. the ``MagicMock`` originals the
    hermetic unit tests install), where ``wraps`` would raise.
    """
    if isinstance(getattr(orig, "__name__", None), str):
        functools.wraps(orig)(wrapper)
    wrapper._is_tpu_patch_wrapper = True
    wrapper._orig_tpu_fn = orig
    return wrapper


@run_once
def apply_omni_tpu_patches() -> None:
    """Extend async output timeout and patch TPU unsupported ops."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Lazy execution attributes failure to sync point: In PyTorch/XLA
    #    and torch_tpu, forward passes are queued lazily in the computation
    #    graph. diffusion_worker emits COMPUTE_DONE prematurely before the TPU
    #    hardware finishes execution. When the async output background loop
    #    triggers host synchronization (.cpu()), it blocks on multi-step diffusion
    #    graph execution and initial compilation.
    # 2. Recompilation storms & JIT compile latency: Initial JIT/AOT
    #    compilation and weight distribution for large diffusion models (e.g.
    #    Wan 14B) take >600s, exceeding vLLM-Omni's default 600s timeout.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed/reduced once:
    # 1. OmniTpuPlatform.record_device_event() implements hardware synchronization
    #    (torch.tpu.synchronize()) to ensure execution completes before COMPUTE_DONE.
    # 2. AOT compilation caching stabilizes startup compile times.
    # -------------------------------------------------------------------------
    import os

    os.environ.setdefault("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", "1800.0")
    logger.info("Applied TPU patch: extended VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT to 1800s.")

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Missing aten::upsample_linear1d.out operator in torch_tpu.
    # Calling torch.nn.functional.interpolate(..., mode="linear") on 3D tensors
    # fails with:
    #   RuntimeError: operator 'aten::upsample_linear1d.out' is not implemented for TPU
    # Used in temporal/audio diffusion alignment (Wan 2.2 S2V, LongCat, audio vocoders).
    #
    # UPSTREAM STATUS & REMOVAL CONDITION:
    # Fixed upstream in torch_tpu commit a0b184ed27d
    # ("Implement aten::upsample_linear1d.out", Sep 14, 2026). This CPU fallback patch
    # can be deleted once vllm-torchtpu bumps to a torch_tpu development wheel
    # >= 0.1.1.dev20260914 containing that commit.
    # -------------------------------------------------------------------------
    orig_interpolate = _unwrap_tpu_patch(torch.nn.functional.interpolate)

    def _safe_interpolate(
        input: torch.Tensor,
        size: Any = None,
        scale_factor: Any = None,
        mode: str = "nearest",
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input.device.type == "tpu" and mode == "linear":
            return orig_interpolate(
                input.cpu(), size, scale_factor, mode, *args, **kwargs
            ).to(input.device)
        return orig_interpolate(input, size, scale_factor, mode, *args, **kwargs)

    torch.nn.functional.interpolate = _mark_tpu_patch(
        _safe_interpolate, orig_interpolate
    )
    logger.info(
        "Applied TPU patch: safe torch.nn.functional.interpolate for 1D linear TPU ops."
    )

    _patch_flow_match_scheduler_step_flush()
    _patch_group_norm_tpu()
    _patch_vae_decoder_conv_out_tpu()
    _patch_rotary_embedding_tpu()
    _patch_z_image_transformer_block_flush()
    _patch_z_image_prompt_encoder_tpu()
    # Engine-host (CPU) output path; OmniTpuPlatform is instantiated in the
    # engine process too, so this reaches DiffusionEngine.postprocess_output.
    _patch_vae_image_processor_pil_output()


@functools.cache
def _rope_matrices(
    head_dim: int, interleaved: bool, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant matrices for ``rotate_half`` and the cos/sin half-dim expansion.

    ``R`` is a signed permutation with ``x @ R == rotate_half(x)``; ``E`` maps
    ``cos`` of shape ``[.., D/2]`` to the per-element ``[.., D]`` factor
    (``repeat(cos, "d -> (d 2)")`` for the interleaved/GPT-J layout,
    ``cat(cos, cos)`` for the NeoX layout). Entries are 0/+-1 so the products
    are exact in any float dtype.
    """
    half = head_dim // 2
    r = torch.zeros(head_dim, head_dim)
    e = torch.zeros(half, head_dim)
    idx = torch.arange(half)
    if interleaved:
        # out[2i] = -x[2i+1], out[2i+1] = x[2i]
        r[2 * idx + 1, 2 * idx] = -1.0
        r[2 * idx, 2 * idx + 1] = 1.0
        e[idx, 2 * idx] = 1.0
        e[idx, 2 * idx + 1] = 1.0
    else:
        # out[:half] = -x[half:], out[half:] = x[:half]
        r[idx + half, idx] = -1.0
        r[idx, idx + half] = 1.0
        e[idx, idx] = 1.0
        e[idx, idx + half] = 1.0
    return r.to(dtype=dtype, device=device), e.to(dtype=dtype, device=device)


def _rope_matrices_const(
    head_dim: int, interleaved: bool, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_rope_matrices`` resolved eagerly while dynamo traces (trace-time constant)."""
    return _rope_matrices(head_dim, interleaved, dtype, device)


_rope_matrices_const = torch._dynamo.assume_constant_result(_rope_matrices_const)


def _tpu_apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> torch.Tensor:
    """``x * cos + rotate_half(x) * sin`` without strided slices or concats.

    ``x``: ``[..., seq, heads, head_dim]``; ``cos``/``sin``: ``[seq, head_dim/2]``
    (already in ``x.dtype``). The rotation is a ``[head_dim, head_dim]`` signed
    permutation applied on the MXU (exact), so the whole op lowers to one
    matmul with a fused elementwise epilogue.
    """
    head_dim = x.shape[-1]
    r, e = _rope_matrices_const(head_dim, interleaved, x.dtype, x.device)
    cos_full = (cos @ e).unsqueeze(-2)  # [seq, 1, head_dim], broadcast over heads
    sin_full = (sin @ e).unsqueeze(-2)
    return x * cos_full + (x @ r) * sin_full


def _patch_rotary_embedding_tpu() -> None:
    # -------------------------------------------------------------------------
    # Root cause (XLA:TPU layout of the generic rotary implementation):
    # vLLM-Omni's ``RotaryEmbedding.forward_native`` builds ``rotate_half`` from
    # ``x[..., ::2]`` / ``x[..., 1::2]`` strided slices, ``stack``, ``rearrange``
    # and ``cat`` on the minor (lane) dimension. On TPU every one of those is a
    # relayout: per DiT layer (q and k) XLA emits ~8 copy/reshape/as_strided ops
    # over ``[4128, 30, 128]`` bf16, ~36 ms of a 158 ms Z-Image-Turbo denoise
    # step on tpu7x (23%) for an op whose HBM roofline is <2 ms.
    #
    # Fix: express rotate_half as ``x @ R`` with a constant signed-permutation
    # matrix (exact), and the half-dim cos/sin expansion as ``cos @ E``; XLA
    # then emits one small matmul with the multiply-add fused as its epilogue.
    # Numerics are identical to the reference formula (same bf16 products).
    #
    # REMOVAL CONDITION: vLLM-Omni grows a TPU/XLA-friendly rotary path (or
    # ``forward_native`` stops slicing the minor dim), or torch_tpu fuses
    # strided minor-dim slices without materialising them.
    # -------------------------------------------------------------------------
    try:
        rope_mod = importlib.import_module("vllm_omni.diffusion.layers.rope")
        rotary_cls = rope_mod.RotaryEmbedding
    except Exception:  # noqa: BLE001 - vllm_omni is an omni extra
        logger.warning("vllm_omni rope module not importable; rotary patch skipped.")
        return

    orig_forward_native = _unwrap_tpu_patch(rotary_cls.forward_native)

    def _forward_native(
        self: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # On TPU, default matmul rounds fp32 inputs to bf16 on the MXU; only
        # route low-precision activations (bf16/fp16, where the 0/+-1 MXU
        # matmul is bit-exact) and preserve full fp32 precision when x is fp32.
        if (
            x.device.type != "tpu"
            or x.dtype not in (torch.bfloat16, torch.float16)
            or x.dim() < 3
        ):
            return orig_forward_native(self, x, cos, sin)
        cos_h, sin_h = self._prepare_half_head_dim_cos_sin(cos, sin)
        if (
            cos_h.dim() != 2
            or cos_h.shape[-1] * 2 != x.shape[-1]
            or cos_h.shape != sin_h.shape
        ):
            # Partial rotary dims or per-batch cos/sin: keep the generic path.
            return orig_forward_native(self, x, cos, sin)
        return _tpu_apply_rotary(
            x, cos_h.to(x.dtype), sin_h.to(x.dtype), bool(self.interleaved)
        )

    rotary_cls.forward_native = _mark_tpu_patch(_forward_native, orig_forward_native)
    logger.info(
        "Applied TPU patch: matmul-based rotate_half in RotaryEmbedding.forward_native "
        "(avoids minor-dim strided-slice relayouts)."
    )


def _tpu_group_norm(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """GroupNorm with statistics reduced over a ``[N, G, C/G, *spatial]`` view.

    Statistics are accumulated in fp32 (as ``native_group_norm`` does) and the
    result is cast back to the input dtype.
    """
    n, c = input.shape[0], input.shape[1]
    spatial = tuple(input.shape[2:])
    x = input.reshape(n, num_groups, c // num_groups, *spatial).float()
    reduce_dims = tuple(range(2, x.dim()))
    mean = x.mean(reduce_dims, keepdim=True)
    xc = x - mean
    var = (xc * xc).mean(reduce_dims, keepdim=True)
    y = (xc * torch.rsqrt(var + eps)).reshape(input.shape)
    affine_shape = (1, c) + (1,) * len(spatial)
    if weight is not None:
        y = y * weight.reshape(affine_shape).float()
    if bias is not None:
        y = y + bias.reshape(affine_shape).float()
    return y.to(input.dtype)


def _patch_group_norm_tpu() -> None:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # torch_tpu lowers ``aten::native_group_norm`` by flattening each group to
    # ``[N, G, C/G*H*W]`` and reducing over that single huge minor dimension.
    # For VAE-decoder activations (e.g. ``[1, 128, 1024, 1024]`` -> 4M-wide
    # minor dim) XLA:TPU emits a serial ``while`` loop of ``dynamic-slice`` /
    # ``dynamic-update-slice`` instead of a fused reduce: ~35 ms per GroupNorm
    # on tpu7x against a ~0.4 ms HBM roofline, i.e. ~390 ms of a 2.1 s
    # Z-Image-Turbo 1024x1024 generate call. Any formulation that merges H*W
    # into the reduced minor dim (``reshape(N, G, -1)``, ``var_mean``,
    # ``layer_norm`` over the flattened group) hits the same lowering.
    #
    # Fix: reduce over a ``[N, G, C/G, H, W]`` view (dims 2..) in fp32, which
    # XLA fuses into ordinary reduces: ~3.3 ms for the same shape (~10x).
    #
    # REMOVAL CONDITION: torch_tpu's ``native_group_norm`` lowering keeps the
    # spatial dims separate (or XLA:TPU stops serialising wide minor-dim
    # reductions), at which point the override is a no-op in cost.
    # -------------------------------------------------------------------------
    orig_group_norm = _unwrap_tpu_patch(torch.nn.functional.group_norm)

    def _group_norm(
        input: torch.Tensor,
        num_groups: int,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if input.device.type == "tpu" and input.dim() >= 2:
            return _tpu_group_norm(input, num_groups, weight, bias, eps)
        return orig_group_norm(input, num_groups, weight, bias, eps)

    torch.nn.functional.group_norm = _mark_tpu_patch(_group_norm, orig_group_norm)
    logger.info(
        "Applied TPU patch: fused-reduce torch.nn.functional.group_norm "
        "(avoids serial while-loop lowering of native_group_norm)."
    )


# Output-channel count a VAE ``conv_out`` is zero-padded to on TPU (one lane tile).
_TPU_CONV_OUT_LANES = 128
# Convs with at most this many output channels take the padded path.
_TPU_CONV_OUT_MAX_CHANNELS = 16


def _tpu_lane_padded_conv2d(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """``conv(x)`` computed with the output channels zero-padded to 128 lanes.

    The extra channels have all-zero weights and bias, so they are exactly zero
    and are sliced away; the real channels see the same per-element dot
    products (the only possible difference is MXU summation order).
    """
    w, b = conv.weight, conv.bias
    c_out = w.shape[0]
    extra = _TPU_CONV_OUT_LANES - c_out
    w_p = torch.cat([w, w.new_zeros((extra, *w.shape[1:]))], dim=0)
    b_p = torch.cat([b, b.new_zeros(extra)], dim=0) if b is not None else None
    y = torch.nn.functional.conv2d(
        x, w_p, b_p, conv.stride, conv.padding, conv.dilation, conv.groups
    )
    return y[:, :c_out]


def _tpu_conv_out_forward(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    if x.device.type == "tpu" and x.dim() == 4:
        return _tpu_lane_padded_conv2d(self, x)
    return torch.nn.Conv2d.forward(self, x)


def _patch_vae_decoder_conv_out_tpu() -> None:
    # -------------------------------------------------------------------------
    # Root cause (XLA:TPU convolution layout, Z-Image / Flux ``AutoencoderKL``
    # decoder ``conv_out = Conv2d(128, 3, 3, padding=1)`` at 1024x1024):
    # XLA assigns the conv output a channel-minor tiled layout
    # (``bf16[1024,8,129,3]{1,3,2,0:T(4,128)}``), so each 128-lane vector
    # holds 3 valid values and the result is stored through masked partial
    # writes. XProf: ``fusion.337`` 6.7 ms at 39 GiB/s and 1.3 TFLOP/s — 25 %
    # of the 26.6 ms VAE decode (0.8 ms/step of an 8-step request, at every
    # SP degree because every rank decodes). Zero-padding the weights to 128
    # output channels turns it into an ordinary dense conv (~0.5 ms incl. the
    # channel slice) with identical values in the 3 real channels.
    #
    # REMOVAL CONDITION: XLA:TPU picks a spatial-minor layout (or packs lanes)
    # for convolutions with a handful of output channels.
    # -------------------------------------------------------------------------
    try:
        vae_mod = importlib.import_module("diffusers.models.autoencoders.vae")
        decoder_cls = vae_mod.Decoder
    except Exception:  # noqa: BLE001 - diffusers is an omni extra
        logger.warning("diffusers not importable; VAE conv_out TPU patch skipped.")
        return

    orig_init = _unwrap_tpu_patch(decoder_cls.__init__)

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        conv: Any = getattr(self, "conv_out", None)
        if not isinstance(conv, torch.nn.Conv2d):
            return
        if conv.groups == 1 and 0 < conv.out_channels <= _TPU_CONV_OUT_MAX_CHANNELS:
            # Instance-level override: parameters and state_dict keys are
            # untouched, nn.Module.__call__ picks up the bound attribute.
            conv.forward = _tpu_conv_out_forward.__get__(conv, type(conv))

    decoder_cls.__init__ = _mark_tpu_patch(_init, orig_init)
    logger.info(
        "Applied TPU patch: lane-padded VAE decoder conv_out "
        "(avoids channel-minor partial stores for 3-channel outputs)."
    )


def _tpu_materialize_async(tensor: torch.Tensor | None = None) -> bool:
    """Dispatch the pending lazy graph that produces ``tensor`` without waiting.

    torch_tpu's ``DEFER_AND_FUSE`` eager mode queues ops until something reads a
    result; ``synchronize(t, wait=False)`` compiles + launches that graph and
    returns immediately, so the host can keep tracing the next step while the
    device runs this one. Passing ``None`` flushes all pending tensors on the
    current stream. Returns ``False`` when the API is unavailable.
    """
    try:
        from torch_tpu._internal.sync import synchronize as _tpu_synchronize
    except Exception:  # noqa: BLE001 - older/newer torch_tpu without the helper
        return False
    _tpu_synchronize(tensor, wait=False)
    return True


def _patch_z_image_transformer_block_flush() -> None:
    # -------------------------------------------------------------------------
    # Root cause (torch_tpu repeated-op heuristic mid-layer kSplitTraversal):
    # After _patch_rotary_embedding_tpu reduces ZImageTransformerBlock to ~65
    # aten ops, two consecutive blocks fit inside torch_tpu's 128-op repetition
    # lookback window (RecordDeferredOpAndDetectRepetition, default mode
    # TORCH_TPU_INTERNAL_DETECT_REPEATED_OPS="safe"). The heuristic triggers
    # mid-layer at ``gate_msa * self.attention_norm2(attn_out)`` while 9
    # intermediate tensors (adaLN gates/scales, residual x, attention_norm2,
    # bf16 cos/sin casts) are simultaneously live on the Python stack, and
    # MaterializeCurrentStream(kSplitTraversal) shatters each DiT block into 9
    # separate XLA modules (~270 modules/step, ~15 ms/step of extra HBM
    # traffic and launch overhead on tpu7x).
    #
    # Fix: flush asynchronously at the clean ZImageTransformerBlock boundary
    # (before layer_id == 0 so pre-loop embeddings/refiners/unified inputs are
    # already materialized, and after each block where only the output hidden
    # state ``x`` is live). Every block in a stack then compiles to a single
    # shared XLA module with one live output tensor, and the stream never
    # accumulates two unmaterialized blocks.
    # In addition, because every block now flushes its 4 Ulysses
    # ``all_to_all_single`` collectives as a separate XLA module instead of
    # accumulating 120+ collectives across all 30 blocks, we can mark
    # ``ZImageTransformer2DModel._sp_plan["unified_prepare"]`` with
    # ``auto_pad=True`` (Z-Image already pads sequences to ``SEQ_MULTI_OF=32``)
    # so ``UlyssesParallelAttention`` skips the 240 per-layer ``_all_gather_int``
    # host syncs in ``advanced_uaa`` mode (sp=4 / sp=8).
    #
    # REMOVAL CONDITION: torch_tpu's repeated-op detector aligns splits to
    # module boundaries (or avoids materializing dead stack temporaries), or
    # vLLM-Omni compiles ZImageTransformer2DModel with torch.compile.
    #
    # Regional torch.compile (second part of this patch): even with one XLA
    # module per block, the eager DEFER_AND_FUSE path still *traces* ~65 aten
    # ops per block on the host each step. At sp>=2 the device finishes a
    # block faster than the host can trace the next one, so the step time
    # (~87 ms at sp=2/4) is the host-tracing floor and does not improve with
    # more chips. vLLM-Omni's own ``regionally_compile`` is skipped on TPU
    # (``supports_torch_inductor()`` is False, and its default
    # ``dynamic=True`` is wrong for XLA), so the block forward is compiled
    # here with ``torch.compile(backend="tpu", dynamic=False)``: one compiled
    # function shared by all blocks, one static graph per distinct block
    # input shape (noise_refiner S=4096, context_refiner S=32, sharded layers,
    # plus warmup shapes), replayed per step with ~zero host tracing. The
    # Ulysses ``all_to_all_single`` collectives compile inside the graph.
    # Any failure of the compiled path logs once and falls back to eager.
    # REMOVAL CONDITION: vLLM-Omni's diffusion_model_runner compiles blocks on
    # TPU itself (with dynamic=False).
    # -------------------------------------------------------------------------
    try:
        z_img_mod = importlib.import_module(
            "vllm_omni.diffusion.models.z_image.z_image_transformer"
        )
        block_cls = z_img_mod.ZImageTransformerBlock
    except Exception:  # noqa: BLE001 - vllm_omni is an omni extra
        logger.warning(
            "vllm_omni z_image_transformer module not importable; "
            "block flush patch skipped."
        )
        return

    model_cls = getattr(z_img_mod, "ZImageTransformer2DModel", None)
    sp_plan = getattr(model_cls, "_sp_plan", None)
    if isinstance(sp_plan, dict):
        unified_plan = sp_plan.get("unified_prepare")
        if isinstance(unified_plan, dict):
            sp_plan["unified_prepare"] = {
                k: dataclasses.replace(v, auto_pad=True)
                if dataclasses.is_dataclass(v) and hasattr(v, "auto_pad")
                else v
                for k, v in unified_plan.items()
            }

    # vLLM-Omni's registry sets ``ctx.sp_plan_hooks_applied = True`` on the
    # temporary ForwardContext active during ``load_model()``, but every
    # inference step constructs a fresh ``ForwardContext`` where
    # ``sp_plan_hooks_applied`` resets to ``False``. Without restoring it on
    # the step's ``ForwardContext``, ``ctx.sp_active`` ignores
    # ``_sp_shard_depth == 0`` before ``unified_prepare`` and runs
    # ``UlyssesParallelAttention`` on unsharded ``noise_refiner`` (S=4096) and
    # ``context_refiner`` (S=32), all-to-all gathering ``4096 * sp`` tokens
    # per rank (S=8192/16384/32768 at sp=2/4/8).
    try:
        fc_mod = importlib.import_module("vllm_omni.diffusion.forward_context")
        is_fc_available = fc_mod.is_forward_context_available
        get_fc = fc_mod.get_forward_context
    except Exception:  # noqa: BLE001
        is_fc_available = None
        get_fc = None

    def _has_sp_plan_hooks(module: Any) -> bool:
        registry = getattr(module, "_hook_registry", None)
        if registry is None:
            return False
        hooks = getattr(registry, "_hooks", getattr(registry, "hooks", None))
        return bool(hooks) if isinstance(hooks, dict) else True

    # Shard ``noise_refiner`` under SP as well. Upstream's ``_sp_plan`` only
    # shards from ``unified_prepare`` on, so at sp=N every rank runs both
    # noise_refiner blocks on the full image sequence (S=4096, ~3 ms each per
    # step on tpu7x) while the main layers run S=4128/N. The refiner blocks are
    # ordinary ``ZImageTransformerBlock``s, so the same Ulysses attention that
    # serves the sharded main layers serves them: shard ``x``/``attn_mask``/
    # ``cos``/``sin`` along the sequence before ``noise_refiner[0]`` and
    # all-gather the output after ``noise_refiner[-1]``. ``_sp_shard_depth`` is
    # raised for the duration (vLLM-Omni only raises it for ``split_output``
    # plan entries) so ``ctx.sp_active`` turns the collectives on, and an
    # equal-length boundary is pushed so the attention layer skips its runtime
    # length all-gather. Sequences that do not divide evenly stay unsharded.
    # REMOVAL CONDITION: upstream ``ZImageTransformer2DModel._sp_plan`` covers
    # ``noise_refiner`` (and the split hook raises ``_sp_shard_depth`` for
    # input splits).
    def _install_noise_refiner_sp(
        model: Any, fc_available: Callable[[], bool], fc_get: Callable[[], Any]
    ) -> None:
        if getattr(model, "_tpu_noise_refiner_sp", False):
            return
        model._tpu_noise_refiner_sp = True
        refiner = getattr(model, "noise_refiner", None)
        if refiner is None or len(refiner) == 0:
            return
        try:
            sp_mod = importlib.import_module(
                "vllm_omni.diffusion.distributed.sp_sharding"
            )
            sp_shard, sp_gather = sp_mod.sp_shard, sp_mod.sp_gather
            sp_world_size = sp_mod.get_sequence_parallel_world_size
        except Exception:  # noqa: BLE001 - keep the unsharded refiner
            logger.warning(
                "vllm_omni sp_sharding unavailable; noise_refiner stays unsharded."
            )
            return
        state = {"sharded": False}
        arg_names = ("x", "attn_mask", "cos", "sin")

        def _make_pre(first: bool) -> Any:
            # ``forward`` re-passes the full-length attn_mask/cos/sin to every
            # refiner block, so each block shards those; ``x`` is sharded once.
            def _pre(
                module: Any, args: tuple, kwargs: dict
            ) -> tuple[tuple, dict] | None:
                if first:
                    state["sharded"] = False
                elif not state["sharded"]:
                    return None
                x = args[0] if args else kwargs.get("x")
                if x is None or not isinstance(x, torch.Tensor):
                    return None
                if x.dim() != 3 or not fc_available():
                    return None
                world = sp_world_size()
                if first:
                    full_len = x.shape[1]
                    if world <= 1 or full_len % world or full_len < world:
                        return None
                else:
                    full_len = x.shape[1] * world
                args_list = list(args)
                for i, name in enumerate(arg_names):
                    if i == 0 and not first:
                        continue
                    if i < len(args_list):
                        t = args_list[i]
                    elif name in kwargs:
                        t = kwargs[name]
                    else:
                        continue
                    if not (
                        isinstance(t, torch.Tensor)
                        and t.dim() >= 2
                        and t.shape[1] == full_len
                    ):
                        continue
                    t = sp_shard(t, 1, validate=False)
                    if i == 0:
                        # ``sp_shard`` hands ``noise_refiner[0]`` the view
                        # ``x[:, lo:hi]`` of the full sequence (storage offset
                        # ``rank * S_local * D``, batch stride of the *full*
                        # tensor; ``.contiguous()`` is a no-op at batch 1).
                        # Through the shared ``torch.compile(backend="tpu")``
                        # block function that view is the one input the
                        # compiled graph gets wrong: NaN from the second step
                        # at sp=8 and a ~10 dB PSNR loss at sp=4, verified
                        # in-process against the eager block on identical
                        # inputs, while the same compiled function is exact
                        # (<= 1 bf16 ulp) on the fresh tensors every later
                        # block receives and on every main layer. A fresh copy
                        # of the shard (S_local x 3840 bf16, a few microseconds
                        # on the device) restores exactness; the sharded mask /
                        # cos / sin views are unaffected.
                        # REMOVAL CONDITION: torch_tpu's compiled path handles
                        # strided-view inputs with a storage offset correctly.
                        t = t.clone(memory_format=torch.contiguous_format)
                    if i < len(args_list):
                        args_list[i] = t
                    else:
                        kwargs[name] = t
                if first:
                    ctx = fc_get()
                    ctx._sp_shard_depth += 1
                    ctx._sp_equal_pad_stack.append(True)
                    state["sharded"] = True
                return tuple(args_list), kwargs

            return _pre

        def _post(module: Any, args: tuple, output: Any) -> Any:
            if not state["sharded"]:
                return None
            state["sharded"] = False
            gathered = output
            if isinstance(output, torch.Tensor):
                gathered = sp_gather(output, 1, validate=False)
            ctx = fc_get()
            ctx._sp_shard_depth = max(0, ctx._sp_shard_depth - 1)
            if ctx._sp_equal_pad_stack:
                ctx._sp_equal_pad_stack.pop()
            return gathered

        for i, block in enumerate(refiner):
            block.register_forward_pre_hook(_make_pre(i == 0), with_kwargs=True)
        refiner[-1].register_forward_hook(_post)
        logger.info(
            "Applied TPU patch: noise_refiner (%d blocks) runs sequence-sharded "
            "under SP.",
            len(refiner),
        )

    if (
        model_cls is not None
        and hasattr(model_cls, "forward")
        and is_fc_available is not None
        and get_fc is not None
    ):
        orig_model_forward = _unwrap_tpu_patch(model_cls.forward)

        def _model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                _has_sp_plan_hooks(getattr(self, "unified_prepare", None))
                and is_fc_available()
            ):
                get_fc().sp_plan_hooks_applied = True
                _install_noise_refiner_sp(self, is_fc_available, get_fc)
            return orig_model_forward(self, *args, **kwargs)

        model_cls.forward = _mark_tpu_patch(_model_forward, orig_model_forward)

    orig_forward = _unwrap_tpu_patch(block_cls.forward)
    compile_state: dict[str, Any] = {"enabled": _ZIMAGE_BLOCK_COMPILE, "fn": None}

    def _compiled_forward() -> Any:
        if compile_state["fn"] is None:
            # dynamic=False gives one static graph per distinct block input
            # shape: vLLM-Omni's warmup shapes, noise_refiner (S=4096),
            # context_refiner (S=32) and the (SP-sharded) layers. That exceeds
            # dynamo's default limit of 8 recompiles per code object, after
            # which new shapes silently run eagerly.
            cfg = torch._dynamo.config
            for attr in ("recompile_limit", "cache_size_limit"):
                if (
                    isinstance(getattr(cfg, attr, None), int)
                    and getattr(cfg, attr) < 64
                ):
                    setattr(cfg, attr, 64)
            compile_state["fn"] = torch.compile(
                orig_forward, backend="tpu", dynamic=False
            )
        return compile_state["fn"]

    def _forward_and_flush(self: Any, *args: Any, **kwargs: Any) -> Any:
        x = args[0] if args else kwargs.get("x")
        on_tpu = (
            isinstance(x, torch.Tensor)
            and getattr(getattr(x, "device", None), "type", None) == "tpu"
        )
        # noise_refiner[0] has layer_id == 1000; context_refiner[0] and
        # layers[0] have layer_id == 0.
        if on_tpu and getattr(self, "layer_id", None) in (0, 1000):
            _tpu_materialize_async(None)
        if on_tpu and compile_state["enabled"]:
            try:
                out = _compiled_forward()(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - keep the model usable
                logger.warning(
                    "torch.compile(backend='tpu') of ZImageTransformerBlock failed "
                    "(%s); running the block eagerly from now on.",
                    type(exc).__name__,
                    exc_info=True,
                )
                compile_state["enabled"] = False
                out = orig_forward(self, *args, **kwargs)
        else:
            out = orig_forward(self, *args, **kwargs)
        if on_tpu and isinstance(out, torch.Tensor):
            _tpu_materialize_async(out)
        return out

    block_cls.forward = _mark_tpu_patch(_forward_and_flush, orig_forward)
    logger.info(
        "Applied TPU patch: async per-block graph flush around "
        "ZImageTransformerBlock.forward (torch.compile(backend='tpu'): %s).",
        "on" if compile_state["enabled"] else "off",
    )


def _patch_flow_match_scheduler_step_flush() -> None:
    # -------------------------------------------------------------------------
    # Root cause (torch_tpu lazy execution + vLLM-Omni's single pipeline call):
    # In DEFER_AND_FUSE mode nothing inside the diffusion pipeline forces the
    # deferred graph to run, so the host traces *all* denoise steps and the VAE
    # decode (tens of thousands of aten ops) while the TPU sits idle, and only
    # the final ``.cpu()`` of the image launches the work. On Z-Image-Turbo /
    # tpu7x that is ~415 ms of exposed host time per image (~17% of E2E).
    #
    # Fix: after every ``FlowMatchEulerDiscreteScheduler.step`` launch the graph
    # that produces the new latents *asynchronously*. Per-step host tracing
    # (~45 ms) is then hidden behind the device's step time, and each step
    # becomes an identical, cache-hit program instead of one giant graph.
    #
    # REMOVAL CONDITION: vLLM-Omni compiles the denoise loop on TPU
    # (``supports_torch_inductor``-equivalent path with ``backend="tpu"``) or
    # torch_tpu flushes deferred graphs on its own size/time heuristics.
    # -------------------------------------------------------------------------
    try:
        schedulers_mod = importlib.import_module("diffusers.schedulers")
        scheduler_cls = schedulers_mod.FlowMatchEulerDiscreteScheduler
    except Exception:  # noqa: BLE001 - diffusers is an omni extra
        logger.warning("diffusers not importable; scheduler step flush patch skipped.")
        return

    orig_step = _unwrap_tpu_patch(scheduler_cls.step)

    # The first ``step`` after ``set_timesteps`` resolves its sigma index with
    # ``(self.timesteps == timestep).nonzero()``: on TPU that is a blocking
    # device->host readback which also forces the whole deferred step-0 graph
    # to run *synchronously*, so the host cannot trace step 1 while step 0
    # executes (~64 ms exposed per image on tpu7x). Pipelines that start
    # mid-schedule (image-to-image) call ``set_begin_index`` beforehand (the
    # diffusers convention, which vLLM-Omni's ``get_timesteps`` follows); when
    # they have not, the first call corresponds to ``timesteps[0]``.
    def _resolve_first_step_index(self: Any, timestep: Any) -> None:
        if getattr(self, "_step_index", None) is not None:
            return
        if getattr(self, "_begin_index", None) is not None:
            return  # orig_step picks it up without a readback
        if not (isinstance(timestep, torch.Tensor) and timestep.device.type == "tpu"):
            return
        timesteps: torch.Tensor | None = getattr(self, "timesteps", None)
        if timesteps is None or not isinstance(timesteps, torch.Tensor):
            return
        if int(timesteps.numel()) == 0:
            return
        self._step_index = 0

    def _step_and_flush(self: Any, *args: Any, **kwargs: Any) -> Any:
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        _resolve_first_step_index(self, timestep)
        out = orig_step(self, *args, **kwargs)
        sample = out[0] if isinstance(out, tuple) else getattr(out, "prev_sample", None)
        device = getattr(sample, "device", None)
        if isinstance(sample, torch.Tensor) and getattr(device, "type", None) == "tpu":
            _tpu_materialize_async(sample)
        return out

    scheduler_cls.step = _mark_tpu_patch(_step_and_flush, orig_step)
    logger.info(
        "Applied TPU patch: async per-step graph flush after "
        "FlowMatchEulerDiscreteScheduler.step."
    )


class _TpuCompiledPromptEncoder:
    """``hidden_states[-2]`` of a HF decoder-only LM as one compiled TPU program.

    Z-Image conditions on the penultimate hidden state of its Qwen3 text
    encoder, so the last decoder layer, the final norm and ``lm_head`` never
    reach the output and are skipped. Positions are ``arange`` and the mask is
    causal & key-padding, which is exactly what ``Qwen3Model.forward`` derives
    for a padded ``attention_mask`` when no ``position_ids`` are given.
    """

    def __init__(self, text_encoder: Any) -> None:
        get_decoder = getattr(text_encoder, "get_decoder", None)
        decoder = get_decoder() if callable(get_decoder) else text_encoder.model
        self._embed_tokens = decoder.embed_tokens
        self._rotary_emb = decoder.rotary_emb
        self._layers = list(decoder.layers)[:-1]
        self._compiled = torch.compile(self._forward, backend="tpu", dynamic=False)

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._embed_tokens(input_ids)
        position_embeddings = self._rotary_emb(hidden, position_ids)
        for layer in self._layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return hidden

    @staticmethod
    def build_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """``[B, 1, S, S]`` bool mask: causal & key padding, diagonal always kept.

        Keeping the diagonal means padded query rows are never fully masked
        (no NaN rows); their outputs are sliced away by the caller anyway.
        """
        seq = attention_mask.shape[-1]
        eye = torch.eye(seq, dtype=torch.bool, device=attention_mask.device)
        causal = torch.ones_like(eye).tril()
        return (causal & attention_mask.bool()[:, None, None, :]) | eye

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq = input_ids.shape
        device = input_ids.device
        mask = self.build_mask(attention_mask).to(device)
        position_ids = torch.arange(seq).unsqueeze(0).expand(bsz, seq).to(device)
        with torch.no_grad():
            return self._compiled(input_ids, mask, position_ids)


def _patch_z_image_prompt_encoder_tpu() -> None:
    # -------------------------------------------------------------------------
    # Root cause (torch_tpu op-by-op lazy tracing of a 36-layer HF model):
    # ``ZImagePipeline._encode_prompt`` runs the Qwen3-4B text encoder eagerly.
    # Each image pays ~55 ms of host time while the 36 decoder layers are traced
    # into ~250 small XLA modules (device busy for only ~10 ms of it), then
    # ``prompt_embeds[i][prompt_masks[i]]`` performs a data-dependent boolean
    # index that forces a device->host readback before the denoise loop can be
    # traced. The encoder input is already a static ``[B, max_sequence_length]``
    # (``padding="max_length"``), so the whole encoder compiles once per shape.
    #
    # Fix: on TPU, compute ``hidden_states[-2]`` with one
    # ``torch.compile(backend="tpu", dynamic=False)`` program over the first
    # ``num_layers - 1`` decoder layers (the last layer, final norm and lm_head
    # do not contribute to that output) and slice the valid tokens with the
    # host-side ``attention_mask`` (no readback). The tokenization mirrors
    # upstream line for line; ``test_z_image_components.py`` carries a canary
    # that fails if upstream's ``_encode_prompt`` changes.
    #
    # REMOVAL CONDITION: vLLM-Omni compiles its diffusion text encoders through
    # the platform compile hook (``supports_torch_inductor`` equivalent for the
    # ``"tpu"`` backend) or drops the boolean-mask gather.
    # -------------------------------------------------------------------------
    try:
        pipe_mod = importlib.import_module(
            "vllm_omni.diffusion.models.z_image.pipeline_z_image"
        )
        pipe_cls = pipe_mod.ZImagePipeline
    except Exception:  # noqa: BLE001 - vllm_omni is an omni extra
        logger.warning(
            "vllm_omni pipeline_z_image module not importable; "
            "prompt encoder patch skipped."
        )
        return

    orig_encode_prompt = _unwrap_tpu_patch(pipe_cls._encode_prompt)

    def _get_encoder(pipe: Any) -> _TpuCompiledPromptEncoder | None:
        encoder = getattr(pipe, "_tpu_prompt_encoder", None)
        if encoder is None:
            try:
                encoder = _TpuCompiledPromptEncoder(pipe.text_encoder)
            except Exception as exc:  # noqa: BLE001 - unexpected encoder layout
                logger.warning(
                    "Z-Image text encoder %s not supported by the compiled TPU "
                    "prompt encoder (%s); using the eager path.",
                    type(pipe.text_encoder).__name__,
                    exc,
                )
                encoder = False
            pipe._tpu_prompt_encoder = encoder
        return encoder or None

    def _encode_prompt(
        self: Any,
        prompt: Any,
        device: torch.device | None = None,
        prompt_embeds: Any = None,
        max_sequence_length: int = 512,
    ) -> Any:
        device = device or self._execution_device
        encoder = None
        if prompt_embeds is None and getattr(device, "type", None) == "tpu":
            encoder = _get_encoder(self)
        if encoder is None:
            return orig_encode_prompt(
                self,
                prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                max_sequence_length=max_sequence_length,
            )

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_item}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt_item in prompts
        ]
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = text_inputs.attention_mask
        try:
            hidden = encoder(text_inputs.input_ids.to(device), attention_mask)
            embeddings_list = []
            for i in range(hidden.shape[0]):
                valid = attention_mask[i].nonzero().flatten()
                start = int(valid[0]) if valid.numel() else 0
                stop = int(valid[-1]) + 1 if valid.numel() else 0
                if stop - start == valid.numel():
                    # Contiguous padding: a static slice, no device->host readback.
                    embeddings_list.append(hidden[i, start:stop])
                else:
                    embeddings_list.append(hidden[i][attention_mask[i].bool()])
        except Exception as exc:  # noqa: BLE001 - keep the pipeline usable
            logger.warning(
                "Compiled Z-Image prompt encoder failed (%s); falling back to "
                "the eager text encoder.",
                exc,
            )
            self._tpu_prompt_encoder = False
            return orig_encode_prompt(
                self,
                prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        return embeddings_list

    pipe_cls._encode_prompt = _mark_tpu_patch(_encode_prompt, orig_encode_prompt)
    logger.info("Applied TPU patch: compiled Qwen3 prompt encoder for ZImagePipeline.")


def _pt_images_to_pil(images: torch.Tensor) -> list[Any]:
    """diffusers' ``numpy_to_pil(pt_to_numpy(images))`` computed in torch.

    ``images`` is the denormalized ``[B, C, H, W]`` tensor (values in ``[0, 1]``).
    Upstream converts it with ``images.cpu().permute(0, 2, 3, 1).float().numpy()``
    and then runs ``(x * 255).round().astype("uint8")`` and ``PIL.Image.fromarray``
    on that *non-C-contiguous* NHWC view: every numpy inner loop is the
    3-element channel axis and PIL falls back to a per-element ``tobytes()``
    copy. The same arithmetic in torch (float32 multiply, round-half-to-even,
    uint8 cast, one parallel NHWC transpose) produces identical bytes.
    """
    # Dynamic import: PIL is a diffusers dependency, not a vllm-torchtpu one.
    Image = importlib.import_module("PIL.Image")

    u8 = images.detach().cpu().to(torch.float32, copy=True).mul_(255).round_()
    u8 = u8.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    if u8.shape[-1] == 1:
        # Grayscale special case, as in ``numpy_to_pil``.
        return [Image.fromarray(im.squeeze(), mode="L") for im in u8]
    return [Image.fromarray(im) for im in u8]


def _patch_vae_image_processor_pil_output() -> None:
    # -------------------------------------------------------------------------
    # Root cause (diffusers ``VaeImageProcessor.postprocess``; runs on the
    # *engine* host in ``DiffusionEngine.postprocess_output`` after every
    # generated image, i.e. on the request's critical path):
    # ``pt_to_numpy`` returns a channels-last *view* of the NCHW tensor, so the
    # numpy ``* 255 / round / astype`` chain iterates over the 3-element
    # channel axis and ``PIL.Image.fromarray`` copies the strided array
    # element by element. For a 1024x1024 image that is 15-23 ms per image on
    # the engine host — ~2 ms/step of an 8-step Z-Image-Turbo request at every
    # SP degree (6 % of E2E at sp=8), and it grows with the host thread count.
    # Doing the identical arithmetic in torch and transposing once (parallel,
    # C-contiguous) takes ~1-2 ms and yields bit-identical PIL images.
    #
    # REMOVAL CONDITION: diffusers' ``pt_to_numpy`` returns a C-contiguous
    # array (or ``numpy_to_pil`` converts in torch), or vLLM-Omni's image
    # pipelines return uint8 images decoded on the device.
    # -------------------------------------------------------------------------
    try:
        ip_mod = importlib.import_module("diffusers.image_processor")
        ip_cls = ip_mod.VaeImageProcessor
    except Exception:  # noqa: BLE001 - diffusers is an omni extra
        logger.warning(
            "diffusers not importable; VaeImageProcessor PIL output patch skipped."
        )
        return

    orig_postprocess = _unwrap_tpu_patch(ip_cls.postprocess)

    def _postprocess(
        self: Any,
        image: Any,
        output_type: str = "pil",
        do_denormalize: Any = None,
    ) -> Any:
        if (
            output_type != "pil"
            or not isinstance(image, torch.Tensor)
            or image.dim() != 4
        ):
            return orig_postprocess(
                self, image, output_type=output_type, do_denormalize=do_denormalize
            )
        # Upstream validation and (conditional) denormalization, then the
        # numpy/PIL tail in torch.
        image = orig_postprocess(
            self, image, output_type="pt", do_denormalize=do_denormalize
        )
        return _pt_images_to_pil(image)

    ip_cls.postprocess = _mark_tpu_patch(_postprocess, orig_postprocess)
    logger.info(
        "Applied TPU patch: torch-native PIL output in VaeImageProcessor.postprocess."
    )


@run_once
def apply_omni_model_specific_patches() -> None:
    """Apply model-specific runtime patches."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Pickling a device tensor aborts process: In torch_tpu, calling
    #    pickle.dumps on a device tensor hits
    #    ABSL_CHECK(tensor.storage().allocator() == nullptr) in
    #    csrc/eager/tensor_to_buffer.cc:260-261, terminating the worker process
    #    with SIGABRT (exit code 134) with no Python traceback.
    # 2. libtpu per-process lockfile prevents cross-process init:
    #    Holding TPU tensors across process boundaries causes the CPU-only
    #    orchestrator process to trigger PJRT backend initialization during
    #    unpickling, colliding with the worker on /tmp/libtpu_lockfile.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed once torch_tpu supports safe inter-process tensor
    # serialization / proxy unpickling without triggering PJRT initialization
    # in host processes that do not own TPU hardware.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion import ipc

    orig_pack_tensor = _unwrap_tpu_patch(ipc._pack_tensor_if_large)

    def _safe_pack_tensor_if_large(
        val: torch.Tensor,
        d2h_stream: Any = None,
    ) -> Any:
        # The tensor is moved to the host first (see above), so upstream's
        # ``d2h_stream`` path (allocate a pinned buffer, ``copy_`` into it,
        # synchronize the stream) would only add a 12 MB pinned allocation
        # and one more copy per 1024x1024 image before the shared-memory
        # write. The bfloat16 -> float32 shim for numpy stays upstream's.
        return orig_pack_tensor(val.cpu(), d2h_stream=None)

    ipc._pack_tensor_if_large = _mark_tpu_patch(
        _safe_pack_tensor_if_large, orig_pack_tensor
    )
    logger.info("Applied TPU patch: safe CPU serialization for diffusion IPC tensors.")

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # In LTX-2, LTXRuntime._decode_output produces a (video, audio) tuple. Audio
    # tensors are sub-threshold (<1MB) and returned directly in the output object,
    # bypassing SHM packing. If audio remains on TPU, unpickling in the orchestrator
    # triggers PJRT re-initialization and /tmp/libtpu_lockfile collision, while
    # pickling in the worker risks tensor_to_buffer.cc ABSL_CHECK aborts.
    # Moving video and audio to CPU prevents device tensors entering the IPC stream.
    #
    # REMOVAL / FIX CONDITION:
    # Can be removed once torch_tpu provides safe cross-process device tensor
    # serialization, or once vllm_omni universally ensures all diffusion output
    # payloads are moved to host CPU before IPC serialization.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    ltx_cls = ltx2_runtime.LTXRuntime
    orig_decode_output = _unwrap_tpu_patch(ltx_cls._decode_output)

    def _safe_decode_output(self: Any, *args: Any, **kwargs: Any):
        res = orig_decode_output(self, *args, **kwargs)
        video, audio = res.output
        res.output = (
            video.cpu() if isinstance(video, torch.Tensor) else video,
            audio.cpu() if isinstance(audio, torch.Tensor) else audio,
        )
        return res

    ltx_cls._decode_output = _mark_tpu_patch(_safe_decode_output, orig_decode_output)
    logger.info("Applied TPU patch: safe LTX-2 CPU decode output.")

    _patch_rotary_embedding_tpu()
    _patch_z_image_transformer_block_flush()
    _patch_z_image_prompt_encoder_tpu()


@run_once
def apply_omni_worker_patches() -> None:
    """Apply patches that must be live before ``WorkerProc`` starts its threads.

    Called from ``OmniTpuPlatform.set_device``. ``DiffusionWorker.__init__``
    calls ``init_device()`` (-> ``set_device``) first, before ``load_model()``
    and before ``WorkerProc.__init__`` starts the ``DiffusionAsyncOutput``
    thread, on both the default path and the ``custom_pipeline_args`` path
    (where ``skip_load_model=True`` defers ``load_model`` until after the
    thread exists). Installing from the attention-backend hook would be too
    late on the custom-pipeline path. ``vllm_omni.diffusion.worker.
    diffusion_worker`` is already imported in the worker process by then, so
    the import below adds no cost.
    """

    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni / torch_tpu Root Cause Tracking:
    # In vllm_omni.diffusion.worker.diffusion_worker, WorkerProc._async_output_loop
    # constructs a device stream using `self.gpu_id`:
    #   device = torch.device(torch.accelerator.current_accelerator().type, self.gpu_id)
    #   d2h_stream = torch.Stream(device=device)
    # Because each TPU worker process isolates a single chip via TPU_VISIBLE_CHIPS,
    # its only addressable local device index is 0. On worker ranks > 0 (where
    # self.gpu_id > 0), torch.Stream(device=torch.device("tpu", self.gpu_id))
    # crashes the background DiffusionAsyncOutput thread with:
    #   RuntimeError: Device index N is out of bounds of addressable devices 1
    # Wrapping `self` in a proxy that exposes `gpu_id = 0` inside `_async_output_loop`
    # fixes the stream device index without mutating `self.gpu_id` used by the
    # main worker loop for rank/output routing.
    #
    # REMOVAL / FIX CONDITION:
    # Can be removed once vLLM-Omni's WorkerProc._async_output_loop uses
    # `current_omni_platform.get_torch_device(self.gpu_id)` instead of raw
    # `torch.device(..., self.gpu_id)`.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.worker import diffusion_worker

    worker_proc_cls = diffusion_worker.WorkerProc
    orig_async_output_loop = _unwrap_tpu_patch(worker_proc_cls._async_output_loop)

    class _ZeroGpuIdProxy:
        """``self`` stand-in for the body of ``_async_output_loop`` only.

        Every attribute read/write is forwarded to the real WorkerProc except
        ``gpu_id``, which reads as the local device index 0. Nothing outside
        that one method body ever sees the proxy, so ``self.gpu_id`` keeps its
        rank meaning everywhere else. Any new ``self.gpu_id`` use *inside* the
        loop (e.g. a log line) would also read 0.
        """

        def __init__(self, target: Any) -> None:
            object.__setattr__(self, "_target", target)

        @property
        def gpu_id(self) -> int:
            return 0

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_target"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(object.__getattribute__(self, "_target"), name, value)

    def _safe_async_output_loop(self: Any, *args: Any, **kwargs: Any) -> Any:
        return orig_async_output_loop(_ZeroGpuIdProxy(self), *args, **kwargs)

    worker_proc_cls._async_output_loop = _mark_tpu_patch(
        _safe_async_output_loop, orig_async_output_loop
    )
    logger.info(
        "Applied TPU patch: local device index 0 for WorkerProc._async_output_loop."
    )

    _patch_diffusion_worker_runtime_threads(diffusion_worker)


def _patch_diffusion_worker_runtime_threads(diffusion_worker: Any) -> None:
    # -------------------------------------------------------------------------
    # Root cause (vLLM-Omni multiproc executor x torch intra-op thread pool):
    # ``DiffusionMultiprocExecutor`` calls vLLM's
    # ``set_multiprocessing_worker_envs()`` with its default
    # ``local_world_size=1`` before spawning the diffusion workers, so every
    # worker starts with ``OMP_NUM_THREADS = <all host CPUs>`` (224 on a
    # tpu7x-8 host) instead of its share. With N workers that is N-fold CPU
    # oversubscription: each small CPU op on the per-image path (tokenizer
    # tensors, the prompt-encoder mask, latents ``randn``, scheduler maths)
    # fans out over a full-host thread pool whose workers then spin-wait,
    # starving every other rank's serial dispatch code. The slowest rank gates
    # all ranks in the first Ulysses ``all_to_all_single`` of each image, which
    # is why the cost grew ~quadratically with the SP degree (≈10 ms/image at
    # sp=2, ≈40 at sp=4, ≈150 at sp=8 for Z-Image-Turbo) and looked like a
    # "compiled text encoder stall" in host timers.
    #
    # Fix: once the worker has loaded its model (weight loading is the only
    # phase that benefits from CPU parallelism) apply vLLM's steady-state
    # policy ``set_torch_threads_for_runtime()`` -- the same call this repo's
    # ``TPUWorker.compile_or_warm_up_model`` makes for LLM workers. It drops
    # torch's intra-op threads to 1 unless the user set OMP_NUM_THREADS
    # explicitly (vLLM marks its own default with
    # ``VLLM_OMP_NUM_THREADS_SET_BY_VLLM``). Measured on tpu7x-8 (Z-Image-Turbo,
    # 1024², 8 steps): sp=8 64.3 -> 39.5 ms/step, sp=4 66.0 -> 59.8, sp=2 and
    # sp=1 within noise.
    #
    # REMOVAL CONDITION: vLLM-Omni passes ``local_world_size=num_gpus`` to
    # ``set_multiprocessing_worker_envs`` and/or calls
    # ``set_torch_threads_for_runtime()`` in ``DiffusionWorker`` after
    # ``load_model``.
    # -------------------------------------------------------------------------
    try:
        from vllm.utils.torch_utils import set_torch_threads_for_runtime
    except ImportError:
        # vLLM < 0.29 has no steady-state policy helper, but its
        # set_multiprocessing_worker_envs already defaults OMP_NUM_THREADS=1.
        logger.debug("vLLM has no set_torch_threads_for_runtime; thread patch skipped.")
        return

    worker_cls = diffusion_worker.DiffusionWorker
    orig_load_model = _unwrap_tpu_patch(worker_cls.load_model)

    def _load_model_then_trim_threads(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = orig_load_model(self, *args, **kwargs)
        set_torch_threads_for_runtime()
        return result

    worker_cls.load_model = _mark_tpu_patch(
        _load_model_then_trim_threads, orig_load_model
    )
    logger.info(
        "Applied TPU patch: steady-state torch CPU threads "
        "(set_torch_threads_for_runtime) after DiffusionWorker.load_model."
    )
