# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Serve a TP-only MoE layer, its shared expert and its TP all-reduce in one kernel.

Under MoE tensor parallelism vLLM computes the shared expert separately, adds
it to the routed output and all-reduces the sum. Expert parallelism with TP
attention (DP 1) ends the same way, so its layers are served too, each rank
with its own expert range. With
``USE_MOE_FUSED_TP_AR=1`` an eligible layer instead runs
``moe_fused_tp.fused_moe_tp_ar`` under shard_map over the TP mesh: shared +
routed experts and the all-reduce in one Pallas call per rank.

Arming happens once after the model is loaded (``arm_fused_tp_ar``), when the
shared expert's linears have their final FP8 per-channel layout. For an armed
layer the runner is told it has no shared expert (the kernel owns it), and
``MoERunner._fused_output_is_reduced`` reports True (see ``VllmMoERunner`` in
``layers/adapter/custom_ops/moe_runner.py``), so vLLM issues neither the shared
MLP nor any all-reduce. Token counts above ``MAX_DECODE_TOKENS`` run the fused
TP prefill kernel (``moe_tp_prefill``) inside the same exported program (the
token count is static there), still returning the reduced sum; branching in
torch instead would give buckets different graphs.

Under MoE sequence parallelism (``TPU_MOE_SEQUENCE_PARALLEL=1``, TP only) a
layer's prefill buckets arrive as ``[T/TP, H]`` per rank. Such a layer is also
armed with a second program (``_build_sp_op``) that all-gathers FP8 rows plus
packed routing metadata, runs the prefill kernel and reduce-scatters straight
back to ``[T/TP, H]``. The decoder layer selects it per call by setting
``_CURRENT_LAYER_USE_SP`` around the MLP call (see ``qwen3_next_patch``).
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh, ep_mesh_index
from vllm_torchtpu.kernels.experimental.moe_fused_tp import (
    fused_moe_tp_ar as _kernel_body,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FUSED_MOE_TP_AR_OP_ATTR = "_fused_moe_tp_ar_op"
FUSED_MOE_TP_SP_OP_ATTR = "_fused_moe_tp_sp_op"
_RUNNER_ARMED_ATTR = "_fused_moe_tp_ar_armed"
# The shared expert's tensors become non-persistent buffers of the routed layer,
# so they stay module state Dynamo traces like any weight (a tensor reached
# through a plain attribute makes traces of the same graph disagree on inputs).
_SHARED_BUFFERS = (
    "_ftp_ar_w1s",
    "_ftp_ar_s1s",
    "_ftp_ar_w2s",
    "_ftp_ar_s2s",
    "_ftp_ar_wg",
)

# One exported program per distinct closure; the mesh never enters a traced
# frame (ops are built at arming time and only looked up in forward).
_OPS: dict[tuple, Any] = {}
_SP_OPS: dict[tuple, Any] = {}
_RANK_BUFFER: Any = None
# Set by the decoder layer around an MLP call whose input is already
# sequence-parallel ([T/TP, H]); read at trace time in fused_moe_tp_ar.
_CURRENT_LAYER_USE_SP: bool | None = None
# Largest token count served by the decode kernel; above it, the prefill kernel.
MAX_DECODE_TOKENS = 128


def fused_moe_tp_ar_supported(quant_method: Any) -> bool:
    return getattr(quant_method, FUSED_MOE_TP_AR_OP_ATTR, None) is not None


def _build_op(mesh, n_ranks: int, topk: int, expert_parallel: bool):
    # The token count is static inside the exported program, so the cap is
    # applied HERE, not in torch: every bucket then traces the same torch op
    # with the same inputs, which vLLM's shape-specialized warmup requires.
    key = (id(mesh), n_ranks, topk, expert_parallel)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = EP_AXIS_NAME
    spec = PartitionSpec(axis)

    def prefill(x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, start):
        """Above MAX_DECODE_TOKENS: expert kernel + combine + chunked psum."""
        from vllm_torchtpu.kernels.experimental.moe_tp_prefill import (
            default_combine_chunks,
            moe_tp_prefill,
        )

        # Trace-time, once per token bucket: the evidence that the prefill
        # kernel, not the GMM branch, serves this shape.
        logger.info("Fused TP MoE prefill kernel traced for %d tokens", x.shape[0])
        return moe_tp_prefill(
            x,
            w1,
            w2,
            s1,
            s2,
            w1s,
            w2s,
            s1s,
            s2s,
            wg.astype(x.dtype),
            tw,
            ti,
            combine_chunks=default_combine_chunks(x.shape[0]),
            reduce_fn=lambda y: jax.lax.psum(y, axis),
            expert_start=start,
        )

    def local(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti):
        s1s, s2s, wg = s1s.reshape(1, -1), s2s.reshape(1, -1), wg.reshape(-1, 1)
        # Expert parallelism places experts linearly: rank r holds the global
        # experts [r * E_local, (r + 1) * E_local).
        start = rank.reshape(()) * w1.shape[0] if expert_parallel else None
        if x.shape[0] > MAX_DECODE_TOKENS:
            return prefill(x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, start)
        return _kernel_body(
            rank.reshape(-1),
            x,
            w1,
            w2,
            s1,
            s2,
            w1s,
            s1s,
            w2s,
            s2s,
            wg,
            tw,
            ti,
            n_ranks=n_ranks,
            expert_start=start,
            num_buffers=envs.VLLM_TPU_FUSED_TP_NUM_BUFFERS or None,
        )

    # torch_tpu infers static arguments from annotations: every operand is a
    # jax.Array. Per-rank operands are sharded on axis 0; the gate is replicated.
    def fn(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        w1s: jax.Array,
        s1s: jax.Array,
        w2s: jax.Array,
        s2s: jax.Array,
        wg: jax.Array,
        tw: jax.Array,
        ti: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        body = jax.shard_map(
            lambda *a: local(a[-1], *a[:-1]),
            mesh=mesh,
            in_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
        return body(
            x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti.astype(jnp.int32), rank
        )

    name = "pallas::fused_moe_tp_ar" + (f"_{len(_OPS)}" if _OPS else "")
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def _build_sp_op(mesh, n_ranks: int, topk: int):
    """Sequence-parallel fused TP MoE op (``[T/TP, H] -> [T/TP, H]``), TP only.

    On prefill buckets (``T > MAX_DECODE_TOKENS``) each rank quantizes its
    local ``[T/TP, H]`` rows to FP8 and computes the shared-expert gate on
    them, then all-gathers the 1-byte FP8 rows and one packed f32 metadata
    block (row scale, gate, topk weights, bitcast topk ids) before running
    ``moe_tp_prefill`` with a chunked ``psum_scatter`` straight back to
    ``[T/TP, H]``. Smaller totals all-gather bf16 rows, run the decode kernel
    (which all-reduces) and keep this rank's slice.
    """
    key = (id(mesh), n_ranks, topk)
    op = _SP_OPS.get(key)
    if op is not None:
        return op
    axis = EP_AXIS_NAME
    spec = PartitionSpec(axis)

    def local_sp(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti):
        from vllm_torchtpu.kernels.experimental.moe_tp_prefill import (
            default_combine_chunks,
            moe_tp_prefill,
            rowquant_rows,
        )

        s1s, s2s, wg = s1s.reshape(1, -1), s2s.reshape(1, -1), wg.reshape(-1, 1)
        t_local = x.shape[0]
        if t_local * n_ranks <= MAX_DECODE_TOKENS:
            x_full = jax.lax.all_gather(x, axis, axis=0, tiled=True)
            tw_full = jax.lax.all_gather(tw, axis, axis=0, tiled=True)
            ti_full = jax.lax.all_gather(ti.astype(jnp.int32), axis, axis=0, tiled=True)
            out_full = _kernel_body(
                rank.reshape(-1),
                x_full,
                w1,
                w2,
                s1,
                s2,
                w1s,
                s1s,
                w2s,
                s2s,
                wg,
                tw_full,
                ti_full,
                n_ranks=n_ranks,
            )
            return jax.lax.dynamic_slice_in_dim(
                out_full, rank.reshape(()) * t_local, t_local, axis=0
            )

        x_q_local, x_s_local = rowquant_rows(x)
        gate_local = jax.nn.sigmoid(
            jnp.dot(x, wg.astype(x.dtype), preferred_element_type=jnp.float32)
        )
        k = tw.shape[1]
        # One all-gather for all per-token metadata instead of four small ones.
        # Packed as int32, never f32: small expert ids bitcast to f32 are
        # denormals, which the TPU flushes to zero on the way through the
        # concat / all-gather (every route would land on expert 0). The f32
        # fields bitcast to int32 survive any copy bit-exactly.
        def as_i32(a):
            return jax.lax.bitcast_convert_type(a.astype(jnp.float32), jnp.int32)

        meta_local = jnp.concatenate(
            [
                as_i32(x_s_local[:, None]),
                as_i32(gate_local),
                as_i32(tw),
                ti.astype(jnp.int32),
            ],
            axis=1,
        )
        x_q = jax.lax.all_gather(x_q_local, axis, axis=0, tiled=True)
        meta = jax.lax.all_gather(meta_local, axis, axis=0, tiled=True)

        def as_f32(a):
            return jax.lax.bitcast_convert_type(a, jnp.float32)

        x_s = as_f32(meta[:, 0])
        gate = as_f32(meta[:, 1:2])
        tw_full = as_f32(meta[:, 2 : 2 + k])
        ti_full = meta[:, 2 + k :]
        n_tok = x_q.shape[0]
        logger.info("Fused TP+SP MoE prefill kernel traced for %d tokens", n_tok)
        return moe_tp_prefill(
            x_q,
            w1,
            w2,
            s1,
            s2,
            w1s,
            w2s,
            s1s,
            s2s,
            wg.astype(x.dtype),
            tw_full,
            ti_full,
            combine_chunks=default_combine_chunks(n_tok),
            reduce_fn=lambda y: jax.lax.psum_scatter(
                y, axis, scatter_dimension=0, tiled=True
            ),
            x_quantized=(x_q, x_s),
            shared_gate=gate,
            scatter_ranks=n_ranks,
        )

    def fn(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        w1s: jax.Array,
        s1s: jax.Array,
        w2s: jax.Array,
        s2s: jax.Array,
        wg: jax.Array,
        tw: jax.Array,
        ti: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        body = jax.shard_map(
            lambda *a: local_sp(a[-1], *a[:-1]),
            mesh=mesh,
            in_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
        return body(
            x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti.astype(jnp.int32), rank
        )

    name = "pallas::fused_moe_tp_sp" + (f"_{len(_SP_OPS)}" if _SP_OPS else "")
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _SP_OPS[key] = op
    return op


def _shared_slice(runner) -> tuple | None:
    """This rank's shared-expert tensors in the routed-slice layout, or None."""
    shared = getattr(runner, "_shared_experts", None)
    mlp = getattr(shared, "_layer", None)
    if mlp is None:
        return None
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    gate = getattr(mlp, "expert_gate", None)
    if gate_up is None or down is None or gate is None:
        return None
    w1s, s1s = getattr(gate_up, "weight", None), getattr(gate_up, "weight_scale", None)
    w2s, s2s = getattr(down, "weight", None), getattr(down, "weight_scale", None)
    wg = getattr(gate, "weight", None)
    if w1s is None or s1s is None or w2s is None or s2s is None or wg is None:
        return None
    # Per-channel FP8 linears are stored K-major ([in, out]) with a 1-D scale.
    if s1s.dim() != 1 or s2s.dim() != 1 or w1s.dtype != torch.float8_e4m3fn:
        return None
    if getattr(gate, "bias", None) is not None or wg.dim() != 2 or wg.shape[0] != 1:
        return None
    # Hand over the parameters themselves, never torch views of them: a view
    # passed to the op is re-materialized by an eager as_strided program on
    # every call (measured: one per layer per step, ~48 us of host dispatch
    # each). The kernel's [1, N] / [H, 1] shapes are made inside the program.
    return (w1s.data, s1s.data, w2s.data, s2s.data, wg.data)


def _admit(runner, n_ranks: int) -> tuple[tuple, bool] | None:
    """(shared-expert slice, expert_parallel) for an eligible layer, else None."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    qm = runner._quant_method
    layer = runner.routed_experts
    expert_parallel = bool(par.use_ep)
    if expert_parallel:
        sharded_ok = par.ep_size == n_ranks and par.dp_size == 1
    else:
        sharded_ok = par.tp_size == n_ranks
    if not isinstance(qm, VllmFp8MoEMethodTPU) or not sharded_ok:
        return None
    # Sequence-parallel layers get the SP program (_build_sp_op), which is
    # TP only; EP + SP keeps the unfused path.
    if getattr(cfg, "is_sequence_parallel", False) and expert_parallel:
        return None
    if getattr(cfg, "skip_final_all_reduce", False):
        return None
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
    ):
        return None
    if (
        getattr(runner, "routed_scaling_factor", 1.0) != 1.0
        or runner.shared_expert_gate is not None
    ):
        return None
    shared = _shared_slice(runner)
    if shared is None:
        return None
    w1s, s1s, w2s, s2s, wg = shared
    s1s, s2s, wg = (
        s1s.reshape(1, -1),
        s2s.reshape(1, -1),
        wg.reshape(-1, 1),
    )  # shapes only
    w1, w2 = layer.w13_weight, layer.w2_weight
    s1, s2 = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    e, h, two_i = w1.shape
    i = two_i // 2
    i_s = w2s.shape[0]  # the shared expert is a TP slice, even under EP
    if (
        w1.dtype != torch.float8_e4m3fn
        or tuple(w2.shape) != (e, i, h)
        or tuple(s1.shape) != (e, 1, 1, two_i)
        or tuple(s2.shape) != (e, 1, 1, h)
        or i % 128
        or h % (n_ranks * 128)
        or i_s % 128
        or i_s > i
        or tuple(w1s.shape) != (h, 2 * i_s)
        or tuple(wg.shape) != (h, 1)
    ):
        return None
    return shared, expert_parallel


def _refuse_unarmed_sharded_sp_shared_expert(runner) -> None:
    """Raise if a sequence-parallel layer kept a TP-sharded shared expert but
    was not armed.

    Under SP vLLM replicates the shared expert; ``qwen3_next_patch`` keeps it
    TP-sharded when the fused TP kernel is requested, because only the armed
    kernel reduces its partial sums. Unarmed, every shared-expert output would
    be one rank's partial sum.
    """
    if not getattr(runner.moe_config, "is_sequence_parallel", False):
        return
    shared = getattr(runner, "_shared_experts", None)
    down = getattr(getattr(shared, "_layer", None), "down_proj", None)
    if down is not None and getattr(down, "tp_size", 1) > 1:
        raise RuntimeError(
            "USE_MOE_FUSED_TP_AR=1 kept a sequence-parallel layer's shared expert "
            "TP-sharded, but the layer could not be armed for the fused TP "
            "kernel; its shared-expert output would never be reduced. Unset "
            "USE_MOE_FUSED_TP_AR or TPU_MOE_SEQUENCE_PARALLEL."
        )


def arm_fused_tp_ar(model: torch.nn.Module) -> int:
    """Arm every eligible MoE layer of `model`; returns how many were armed."""
    if not envs.USE_MOE_FUSED_TP_AR:
        return 0
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    if not runners:
        return 0
    try:
        mesh = build_ep_mesh()
    except Exception as exc:  # refuse rather than serve a wrong reduction
        logger.warning("Fused TP all-reduce MoE not armed: no TP mesh (%s)", exc)
        return 0
    if mesh is None:
        return 0
    n_ranks = mesh.shape[EP_AXIS_NAME]
    global _RANK_BUFFER
    armed = 0
    for runner in runners:
        admitted = _admit(runner, n_ranks)
        if admitted is None:
            _refuse_unarmed_sharded_sp_shared_expert(runner)
            continue
        shared, expert_parallel = admitted
        layer = runner.routed_experts
        if _RANK_BUFFER is None:
            _RANK_BUFFER = torch.tensor(
                [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
            )
        setattr(
            runner._quant_method,
            FUSED_MOE_TP_AR_OP_ATTR,
            _build_op(
                mesh, n_ranks, layer.moe_config.experts_per_token, expert_parallel
            ),
        )
        if getattr(runner.moe_config, "is_sequence_parallel", False):
            # Prefill buckets above the SP threshold arrive [T/TP, H]; smaller
            # buckets still run replicated through the op above.
            setattr(
                runner._quant_method,
                FUSED_MOE_TP_SP_OP_ATTR,
                _build_sp_op(mesh, n_ranks, layer.moe_config.experts_per_token),
            )
        for name, tensor in zip(_SHARED_BUFFERS, shared):
            layer.register_buffer(name, tensor, persistent=False)
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        object.__setattr__(runner, _RUNNER_ARMED_ATTR, True)
        armed += 1
    if armed:
        logger.info(
            "Armed fused TP MoE + shared expert + all-reduce on %d of %d MoE layers "
            "(%d ranks, decode kernel up to %d tokens, prefill kernel above)",
            armed,
            len(runners),
            n_ranks,
            MAX_DECODE_TOKENS,
        )
    return armed


def fused_moe_tp_ar(
    quant_method: Any, layer: Any, x: torch.Tensor, topk_weights, topk_ids
):
    """Forward for an armed layer: the TP-reduced (or SP-scattered) output.

    One torch op for every token count; above MAX_DECODE_TOKENS the exported
    program runs the prefill kernel (with its own psum), so an armed layer
    always returns the reduced sum. When the decoder layer marks the call as
    sequence-parallel (``_CURRENT_LAYER_USE_SP``) on an SP layer, the SP
    program maps ``[T/TP, H] -> [T/TP, H]`` instead.
    """
    from vllm_torchtpu.layers.adapter import sequence_parallel

    w1s, s1s, w2s, s2s, wg = (getattr(layer, name) for name in _SHARED_BUFFERS)
    sp_active = (
        _CURRENT_LAYER_USE_SP
        if _CURRENT_LAYER_USE_SP is not None
        else sequence_parallel.is_active()
    )
    use_sp = bool(sp_active) and getattr(
        layer.moe_config, "is_sequence_parallel", False
    )
    # The replicated program on [T/TP, H] rows would silently reduce the wrong
    # tokens, so a missing SP program is an error, not a fallback.
    op = getattr(
        quant_method, FUSED_MOE_TP_SP_OP_ATTR if use_sp else FUSED_MOE_TP_AR_OP_ATTR
    )
    return op(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        w1s,
        s1s,
        w2s,
        s2s,
        wg,
        topk_weights,
        topk_ids,
        _RANK_BUFFER,
    )


def arm_runner_models(model_runner: Any) -> int:
    """Arm the target model and, when it is a separate module, the drafter's
    model (the MTP layer). Called once after the drafter has loaded, when every
    module has its final weight layout."""
    armed = arm_fused_tp_ar(model_runner.model)
    # Only read ``drafter``: the runner's ``_async_drafter`` property is just
    # ``self.drafter`` behind an isinstance assert, so touching it raises
    # AssertionError (not swallowed by getattr's default) for MTP drafters
    # and when speculative decoding is disabled.
    drafter = getattr(model_runner, "drafter", None)
    draft = getattr(drafter, "draft_model", None) or getattr(drafter, "model", None)
    if isinstance(draft, torch.nn.Module) and draft is not model_runner.model:
        armed += arm_fused_tp_ar(draft)
    return armed
