# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/moe_tp_decode (USE_MOE_TP_DECODE_KERNEL).

Under MoE tensor parallelism vLLM runs the shared expert, scales and adds the
routed output, and all-reduces the sum over the TP group. For an armed layer
one torch op does all of it: at the token counts the kernel admits (16, 32 and 64)
one Pallas call per rank covers routing, the shared expert, the routed experts
and the all-reduce; every other count runs the usual TP path (router top-k,
fused_moe_func, the shared expert's linears) and a psum in the same exported
program. The token count is static there, so one torch op serves every bucket.
The op's output is reduced, so arming also takes the shared expert and the
routed scale off the runner, and VllmMoERunner._fused_output_is_reduced turns
the runner's all-reduce off for that layer.

`arm` runs once at the end of TPUModelRunner.load_model, when the weights,
the MTP drafter and the token-padding mask exist. It is collective over the TP
group: every rank must arm the same layers (a rank that skips the kernel would
leave its peers waiting in its barrier), so the ranks compare an arming
signature and raise on a mismatch, then run the kernel's all-reduce once on
known inputs through the op's mesh and rank operand.

Stacked experts (`stack_experts`, asked for by the GLM-5.2 decode megakernel,
which reads every MoE layer's experts through one operand): the armed target
layers' routed weights and scales move into one tensor per kind,
[n_layers, E, ...], allocated once; each layer keeps a row index buffer and the
op reads its row in place (the kernel's and gmm_v2's in-kernel row index), so
the stock MoE computes from the same bytes, with no second copy and no slice
copy per call.
"""

import dataclasses
import faulthandler
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.experimental import moe_tp_decode as kernel
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.kernels.router_topk import select as router_select
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    fused_moe_func_stacked,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OP_ATTR = "_moe_tp_decode_op"
# Stacked experts: the per-layer routed parameters that move into the stacks,
# and the buffers that replace them (the shared stacks + this layer's row).
_EXPERT_PARAMS = ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv")
_STACK_BUFFERS = (
    "_moe_tp_decode_ew13",
    "_moe_tp_decode_ew2",
    "_moe_tp_decode_es13",
    "_moe_tp_decode_es2",
)
_ROW_BUFFER = "_moe_tp_decode_row"
# Set by stack_experts: the compile-cache key folds it in (tpu_compiler), since the
# stacked form changes the stock graphs' inputs and no env records the choice.
_STACKED_FORM: str | None = None
# Non-persistent buffers of an armed layer: the shared expert's tensors (the
# parameters themselves, not views: a view handed to the op is re-materialized
# on every call), the f32 correction bias and this rank's mesh index.
_BUFFERS = (
    "_moe_tp_decode_sw13",
    "_moe_tp_decode_ss13",
    "_moe_tp_decode_sw2",
    "_moe_tp_decode_ss2",
    "_moe_tp_decode_bias",
    "_moe_tp_decode_rank",
)
# A wrong rank or mesh hangs the self-test's all-reduce: dump every stack and
# exit instead.
SELFTEST_TIMEOUT_S = 300
_SUM_FLOOR = 1e-20

_OPS: dict[tuple, Any] = {}
_SELFTEST_OPS: dict[all_reduce.AllReduceSpec, Any] = {}
_RANK_BUFFER: torch.Tensor | None = None
_SELFTEST_DONE = False


@dataclasses.dataclass(frozen=True)
class _Topology:
    mesh: Any
    mesh_index: int


@dataclasses.dataclass(frozen=True)
class _OpStatics:
    topk: int
    routed_scale: float
    activation: str
    use_sparse_core: bool
    onehot_threshold: int


@dataclasses.dataclass(frozen=True)
class _Admitted:
    shared: tuple[torch.Tensor, ...]  # sw13, ss13, sw2, ss2
    rows: tuple[int, ...]  # the token counts the kernel admits
    hidden: int
    statics: _OpStatics


def moe_tp_decode_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's quant method the op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def _sds(t: torch.Tensor) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        tuple(t.shape), pallas.pallas.TORCH_TO_JAX_DTYPE_MAP[t.dtype]
    )


def _topology() -> _Topology | str:
    """The op's 1-D mesh over the TP group and this rank's index in it, or a
    refusal reason. Collective: build_ep_mesh gathers device ids."""
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if ep is None:
        return "no EP group (the TP group has one device)"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    ep_ranks = tuple(int(r) for r in ep.ranks)
    if tp_ranks != ep_ranks:
        return f"TP ranks {tp_ranks} != EP ranks {ep_ranks} (needs dp 1, pcp 1)"
    world = torch.distributed.get_world_size()
    if len(tp_ranks) != world:
        return f"the TP group has {len(tp_ranks)} of {world} ranks"
    mesh = build_ep_mesh()
    mesh_index = ep_mesh_index()
    assert mesh is not None and mesh_index is not None  # the EP group exists
    return _Topology(mesh=mesh, mesh_index=mesh_index)


def _shared_slice(runner) -> tuple[torch.Tensor, ...] | str:
    """This rank's shared-expert tensors (w13 [D, 2Fs], s13 [2Fs], w2 [Fs, D],
    s2 [D]), or a refusal reason."""
    from vllm_torchtpu.layers.adapter.linear_common import (
        KEEP_VLLM_LAYOUT_ATTR,
        WEIGHT_FLIPPED_ATTR,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    if runner.shared_experts is None:
        return "no shared expert"
    mlp = runner.shared_experts._layer
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if gate_up is None or down is None:
        return "the shared expert has no gate_up_proj / down_proj"
    for lin in (gate_up, down):
        if not isinstance(lin.quant_method, VllmFp8LinearMethodTPU):
            return "a shared-expert linear is not VllmFp8LinearMethodTPU"
        flipped = getattr(lin, WEIGHT_FLIPPED_ATTR, False)
        if not flipped or getattr(lin, KEEP_VLLM_LAYOUT_ATTR, False):
            return "a shared-expert weight is not K-major"
        if lin.bias is not None:
            return "a shared-expert linear has a bias"
        if lin.weight.dtype != torch.float8_e4m3fn or lin.weight_scale.dim() != 1:
            return "the shared expert is not FP8 with per-channel scales"
    if down.reduce_results:
        return "the shared expert reduces its own output"
    return (
        gate_up.weight.data,
        gate_up.weight_scale.data,
        down.weight.data,
        down.weight_scale.data,
    )


def _admit(runner, n_ranks: int) -> _Admitted | str:
    """What arming needs for this runner, or why it keeps the usual path."""
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )

    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    layer = runner.routed_experts
    qm = layer.quant_method
    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    if type(qm) is not VllmFp8MoEMethodTPU:
        return f"quant method {type(qm).__name__} is not VllmFp8MoEMethodTPU"
    if qm._tpu_activation_str != "silu":
        return f"activation {qm._tpu_activation_str!r} is not silu"
    if qm._use_adaptive_fused_moe or fused_moe_ep_supported(qm):
        return "a fused EP MoE kernel serves this layer"
    if envs.FUSED_MOE_DECODE_KERNEL:
        return "FUSED_MOE_DECODE_KERNEL is set; moe_tp_decode stands down"
    # Same test as apply_monolithic's owns_collectives: those paths run their own
    # collectives, so the layer keeps them.
    if enable_pipelined_collective_and_compute() or envs.TPU_MOE_FP8_ALL_GATHER:
        return (
            "the MoE path owns its collectives "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE or TPU_MOE_FP8_ALL_GATHER)"
        )
    if par.use_ep or par.tp_size != n_ranks:
        return f"needs TP MoE over {n_ranks} ranks (ep {par.ep_size}, tp {par.tp_size})"
    if par.dp_size != 1 or par.pcp_size != 1 or cfg.is_sequence_parallel:
        return "needs dp 1, pcp 1 and no sequence-parallel MoE"
    if par.enable_eplb:
        return "EPLB is on"
    if runner.gate is None or runner.shared_expert_gate is not None:
        return "needs the runner to hold the gate and no shared-expert gate"
    if runner.routed_input_transform is not None:
        return "a routed input transform is set"
    if runner.routed_output_transform is not None:
        return "a routed output transform is set"
    if cfg.skip_final_all_reduce:
        return "skip_final_all_reduce is set"
    if isinstance(runner.router, ZeroExpertRouter):
        return "zero-expert router"
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "the routed experts have biases"
    grouped = (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    )
    if layer.scoring_func != "sigmoid" or not layer.renormalize or grouped:
        return "routing is not sigmoid + renormalized ungrouped top-k"
    if layer.e_score_correction_bias is None:
        return "no e_score_correction_bias"
    if layer.custom_routing_function is not None:
        return "custom routing function"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return "routing simulation is on"
    if moe_routing._ROUTER_TOPK != "rowmax":
        return f"TPU_MOE_ROUTER_TOPK={moe_routing._ROUTER_TOPK} (needs rowmax)"
    if layer.routed_scaling_factor != 1.0:
        return f"layer routed_scaling_factor {layer.routed_scaling_factor} != 1"
    if not kernel.exact_in_bf16(runner.routed_scaling_factor):
        return f"routed_scaling_factor {runner.routed_scaling_factor} is not bf16"
    shared = _shared_slice(runner)
    if isinstance(shared, str):
        return shared
    hidden = layer.w13_weight.shape[1]
    experts = layer.global_num_experts
    topk = int(cfg.experts_per_token)
    weights = (
        layer.e_score_correction_bias,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        *shared,
    )
    rows = tuple(
        m
        for m in kernel.ROW_BUCKETS
        if kernel.can_use_moe_tp_decode(
            jax.ShapeDtypeStruct((m, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((m, experts), jnp.float32),
            *(_sds(t) for t in weights),
            topk=topk,
            n_ranks=n_ranks,
        )
    )
    if not rows:
        return (
            f"the kernel admits no token count (w13 {list(layer.w13_weight.shape)} "
            f"{layer.w13_weight.dtype}, shared w13 {list(shared[0].shape)})"
        )
    statics = _OpStatics(
        topk=topk,
        routed_scale=float(runner.routed_scaling_factor),
        activation=qm._tpu_activation_str,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_threshold=resolve_onehot_permute_threshold(),
    )
    return _Admitted(shared=shared, rows=rows, hidden=hidden, statics=statics)


def _tp_chain(
    x, scores, bias, is_pad, w13, w2, s13, s2, sw13, ss13, sw2, ss2, st, row=None
):
    """This rank's partial as the unarmed TP path computes it: select_experts
    (ids from the biased scores, renormalized unbiased weights in bf16),
    zero_routing_weights_for_padding, fused_moe_func, the shared expert's two
    quantized linears around SiluAndMul, and shared + scale * routed in bf16.
    `row`: the routed tensors are layer-stacked, fused_moe_func reads row `row`."""
    _, ids = router_select(scores + bias.astype(jnp.float32)[None, :], st.topk)
    w = jnp.take_along_axis(scores, ids, axis=-1)
    total = jnp.sum(w, axis=-1, keepdims=True)
    # torch.clamp(min=1e-20) keeps NaN; a fused jnp.maximum may not (XLA CPU).
    total = jnp.where(jnp.isnan(total), total, jnp.maximum(total, _SUM_FLOOR))
    w = (w / total).astype(x.dtype)
    w = jnp.where(is_pad[:, None], jnp.zeros_like(w), w)
    if row is None:
        routed = fused_moe_func(
            x,
            w13,
            w2,
            s13,
            s2,
            None,
            None,
            w,
            ids,
            None,
            topk=st.topk,
            activation=st.activation,
            use_ep=False,
            use_sparse_core=st.use_sparse_core,
            onehot_moe_permute_threshold=st.onehot_threshold,
            rhs_quant_dtype=None,
            skip_padded_tokens=True,  # admission requires TPU_MOE_SKIP_PADDED_TOKENS
        )
    else:  # the same computation on row `row` of the layer-stacked tensors
        routed = fused_moe_func_stacked(
            x,
            w13,
            w2,
            s13,
            s2,
            w,
            ids,
            row,
            topk=st.topk,
            activation=st.activation,
            use_sparse_core=st.use_sparse_core,
            onehot_moe_permute_threshold=st.onehot_threshold,
            skip_padded_tokens=True,
        )
    gate_up = xla_quantized_matmul(x, sw13, ss13)
    fs = sw2.shape[0]
    act = jax.nn.silu(gate_up[:, :fs].astype(jnp.float32)).astype(x.dtype)
    shared = xla_quantized_matmul(act * gate_up[:, fs:], sw2, ss2)
    return shared + routed * st.routed_scale


def _layer_row(t):
    """One layer's view of a layer-stacked tensor, for shape-only admission."""
    return jax.ShapeDtypeStruct(tuple(t.shape[1:]), t.dtype)


def _op_body(statics: _OpStatics, n_ranks: int, axis: str, stacked: bool = False):
    """The op's per-rank body under shard_map over `axis`: the fused kernel at
    the token counts it admits, else this rank's partial from the usual path
    and a psum. `stacked`: w13 / w2 / s13 / s2 are the layer-stacked tensors and
    a trailing int32 [1, 1] operand is this layer's row in them."""

    def body(x, scores, bias, pad_mask, w13, w2, s13, s2, sw13, ss13, sw2, ss2, rank, *row):
        n = x.shape[0]  # static: torch_tpu exports one program per token count
        is_pad = pad_mask[:n]  # replicated, so n_valid agrees on every rank
        n_valid = n - jnp.sum(is_pad.astype(jnp.int32))
        weights = (w13, w2, s13, s2, sw13, ss13, sw2, ss2)
        assert len(row) == int(stacked), (len(row), stacked)
        layer = row[0].reshape(()) if stacked else None
        routed = weights[:4]
        if stacked:
            routed = tuple(_layer_row(t) for t in routed)
        if kernel.can_use_moe_tp_decode(
            x, scores, bias, *routed, *weights[4:], topk=statics.topk, n_ranks=n_ranks
        ):
            return kernel.moe_tp_decode(
                x,
                scores,
                bias,
                n_valid,
                rank.reshape(()),
                *weights,
                layer,
                routed_scale=statics.routed_scale,
                topk=statics.topk,
                n_ranks=n_ranks,
            )
        partial = _tp_chain(x, scores, bias, is_pad, *weights, statics, row=layer)
        return lax.psum(partial, axis)

    return body


def _get_op(mesh, statics: _OpStatics, stacked: bool = False):
    """One jax_op per distinct closure; built at arming (stacked: at
    stack_experts), never in a trace."""
    key = (mesh, statics, stacked)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    # Operands as moe_tp_decode_forward passes them; bias and the padding mask
    # are replicated, everything else is per rank (stacked: + this layer's row).
    specs = (P(axis), P(axis), P(), P()) + (P(axis),) * (10 if stacked else 9)
    body = _op_body(statics, int(mesh.size), axis, stacked)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        x: jax.Array,
        scores: jax.Array,
        bias: jax.Array,
        pad_mask: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        sw13: jax.Array,
        ss13: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        args = (
            x,
            scores,
            bias,
            pad_mask,
            w13,
            w2,
            s13,
            s2,
            sw13,
            ss13,
            sw2,
            ss2,
            rank,
        )
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    def fn_stacked(
        x: jax.Array,
        scores: jax.Array,
        bias: jax.Array,
        pad_mask: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        sw13: jax.Array,
        ss13: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        rank: jax.Array,
        row: jax.Array,
    ) -> jax.Array:
        args = (x, scores, bias, pad_mask, w13, w2, s13, s2, sw13, ss13, sw2, ss2)
        args += (rank, row)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    name = "pallas::moe_tp_decode" + (f"_{len(_OPS)}" if _OPS else "")
    # torch_tpu annotates the partition specs as tuples of axis-name tuples, but
    # jax_op takes PartitionSpecs at runtime, as other jax_op call sites pass them.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name,
        fn_stacked if stacked else fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    # vLLM traces with a dynamic token dimension; rows in == rows out.
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _OPS[key] = op
    return op


def _selftest_op(mesh, spec: all_reduce.AllReduceSpec):
    op = _SELFTEST_OPS.get(spec)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis), P())
    call = all_reduce.standalone(spec)

    def body(x, rank, n_valid):
        meta = jnp.stack([n_valid.reshape(()), rank.reshape(())]).astype(jnp.int32)
        return call(meta, x)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    def fn(x: jax.Array, rank: jax.Array, n_valid: jax.Array) -> jax.Array:
        args = (x, rank, n_valid)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # PartitionSpecs, as in _get_op (torch_tpu's annotation is narrower).
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::moe_tp_decode_selftest_{len(_SELFTEST_OPS)}",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _SELFTEST_OPS[spec] = op
    return op


def _rank_tensor(device: torch.device, mesh_index: int) -> torch.Tensor:
    """This rank's mesh index as int32 [1, 1] data (one tensor per process), so
    every rank compiles the same program."""
    global _RANK_BUFFER
    if _RANK_BUFFER is None:
        _RANK_BUFFER = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    return _RANK_BUFFER


def _selftest(topo: _Topology, buckets, device) -> list[tuple[int, int, int]]:
    """Run the kernel's all-reduce alone for every armed (rows, hidden) and row
    path through the op's mesh and rank operand; returns this rank's failing
    (rows, hidden, n_valid). Collective: every rank makes the same calls."""
    n_ranks = int(topo.mesh.size)
    rank = _rank_tensor(device, topo.mesh_index)
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for m, h in buckets:
            spec = all_reduce.make_spec(n_ranks, m, h)
            op = _selftest_op(topo.mesh, spec)
            x = all_reduce.selftest_inputs(spec)
            mine = torch.from_numpy(x[topo.mesh_index]).to(torch.bfloat16).to(device)
            for nv in all_reduce.selftest_row_counts(spec):
                n_valid = torch.tensor([nv], dtype=torch.int32, device=device)
                out = op(mine, rank, n_valid).to(torch.float32).cpu().numpy()
                if not all_reduce.selftest_ok(spec, x, out, nv):
                    failures.append((m, h, nv))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def _gather(obj) -> list:
    """all_gather_object over the TP group's CPU group."""
    tp = parallel_state.get_tp_group()
    if tp.world_size == 1:
        return [obj]
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def _check_ranks_agree(entry: dict) -> None:
    """Raise unless every TP rank arms the same layers with the same statics
    and, when any layer is armed, the mesh indices are 0..n-1 once each (the
    kernel addresses peers, and orders its f32 sum, by mesh index)."""
    seen = _gather(entry)
    problems = []
    if any(s["sig"] != seen[0]["sig"] for s in seen):
        problems.append("arming signatures differ")
    indices = sorted(s["mesh_index"] for s in seen)
    if any(s["armed"] for s in seen) and indices != list(range(len(seen))):
        problems.append(f"mesh indices {indices} are not 0..{len(seen) - 1}")
    if problems:
        raise RuntimeError(
            f"moe_tp_decode: TP ranks disagree ({'; '.join(problems)}); their "
            f"programs and the kernel's barrier would not match. Per rank: {seen}"
        )


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm every admitted MoE layer of `model` and of `draft_model` (the MTP
    drafter); returns how many were armed. Collective over the TP group."""
    global _SELFTEST_DONE
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    topo = _topology()  # collective: first, on every rank
    target = [m for m in model.modules() if isinstance(m, MoERunner)]
    known = {id(r) for r in target}
    drafter = (
        []
        if draft_model is None
        else [
            m
            for m in draft_model.modules()
            if isinstance(m, MoERunner) and id(m) not in known
        ]
    )
    plan, refused = [], {}
    for runner in target + drafter:
        if moe_tp_decode_supported(runner._quant_method):
            continue
        if isinstance(topo, str):
            res = topo
        elif (
            not envs.TPU_MOE_SKIP_PADDED_TOKENS or token_padding._padding_state is None
        ):
            res = "needs TPU_MOE_SKIP_PADDED_TOKENS=1 (n_valid comes from its mask)"
        else:
            res = _admit(runner, int(topo.mesh.size))
        if isinstance(res, str):
            refused[res] = refused.get(res, 0) + 1
        else:
            plan.append((runner, res))
    buckets = sorted({(m, adm.hidden) for _, adm in plan for m in adm.rows})
    entries = [(r.layer_name, repr(adm.statics)) for r, adm in plan]
    sig = hashlib.sha256(
        repr((entries, len(target), len(drafter), buckets)).encode()
    ).hexdigest()[:16]
    mesh_index = -1 if isinstance(topo, str) else topo.mesh_index
    _check_ranks_agree(dict(sig=sig, mesh_index=mesh_index, armed=len(plan)))
    if plan and not _SELFTEST_DONE:
        assert not isinstance(topo, str)
        device = plan[0][0].routed_experts.w13_weight.device
        failures = _selftest(topo, buckets, device)
        seen = _gather(dict(mesh_index=topo.mesh_index, failures=failures))
        if any(s["failures"] for s in seen):
            raise RuntimeError(
                "moe_tp_decode: the all-reduce self-test failed (per rank: mesh "
                f"index, failing (rows, hidden, n_valid)): {seen}"
            )
        _SELFTEST_DONE = True
    for runner, adm in plan:
        assert not isinstance(topo, str)
        layer = runner.routed_experts
        # An alias of the f32 parameter: to() and contiguous() are no-ops.
        bias = layer.e_score_correction_bias.detach().to(torch.float32).contiguous()
        rank = _rank_tensor(layer.w13_weight.device, topo.mesh_index)
        for name, tensor in zip(_BUFFERS, (*adm.shared, bias, rank)):
            layer.register_buffer(name, tensor, persistent=False)
        setattr(layer.quant_method, _OP_ATTR, _get_op(topo.mesh, adm.statics))
        # What stack_experts needs to build the same op over stacked experts.
        layer.quant_method._moe_tp_decode_key = (topo.mesh, adm.statics)
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        runner.routed_scaling_factor = 1.0
    adopt_load_stacks()
    armed_drafter = sum(1 for r, _ in plan if id(r) not in known)
    log = logger.info if plan else logger.warning
    log(
        "moe_tp_decode: armed %d/%d target and %d/%d drafter MoE layers (mesh "
        "index %d, kernel token counts %s, sig %s); refused: %s",
        len(plan) - armed_drafter,
        len(target),
        armed_drafter,
        len(drafter),
        mesh_index,
        sorted({m for m, _ in buckets}),
        sig,
        refused or "none",
    )
    return len(plan)


def moe_tp_decode_forward(
    quant_method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    """Forward of an armed layer: shared + routed_scaling_factor * routed,
    already all-reduced over the TP group."""
    padding = token_padding._padding_state
    assert padding is not None, "moe_tp_decode needs the token-padding mask"
    sw13, ss13, sw2, ss2, bias, rank = (getattr(layer, n) for n in _BUFFERS)
    # The usual path's scoring function; everything after it is the op.
    scores = moe_routing._apply_scoring_fn(layer.scoring_func, router_logits)
    row = getattr(layer, _ROW_BUFFER, None)
    if row is not None:  # stacked experts: the stacks and this layer's row
        routed = tuple(getattr(layer, n) for n in _STACK_BUFFERS)
        tail = (rank, row)
    else:
        routed = tuple(getattr(layer, n) for n in _EXPERT_PARAMS)
        tail = (rank,)
    return getattr(quant_method, _OP_ATTR)(
        x,
        scores,
        bias,
        padding.local_padding_mask,
        *routed,
        sw13,
        ss13,
        sw2,
        ss2,
        *tail,
    )


def stacked_form() -> str | None:
    """The stacked-expert form of this process (layers and stack shapes), or None."""
    return _STACKED_FORM


def expert_stacks(layer: Any) -> tuple[tuple[torch.Tensor, ...], int] | None:
    """(w13, w2, s13, s2 stacks, this layer's row) of a stacked layer, else None."""
    if getattr(layer, _ROW_BUFFER, None) is None:
        return None
    return tuple(getattr(layer, n) for n in _STACK_BUFFERS), int(layer._moe_tp_row)


def _hbm_stats() -> dict:
    try:
        s = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - stats are a report, never a gate
        return {}
    keys = ("bytes_in_use", "peak_bytes_in_use", "largest_free_block_bytes", "bytes_limit")
    return {k: int(s[k]) for k in keys if k in s}


def _install_stacks(layers: list, stacks: list, stacked_op, device) -> None:
    """Point each layer (row order) at the stacks and its row, through the stacked op."""
    for row, layer in enumerate(layers):
        for buf, st in zip(_STACK_BUFFERS, stacks):
            layer.register_buffer(buf, st, persistent=False)
        layer.register_buffer(
            _ROW_BUFFER,
            torch.tensor([[row]], dtype=torch.int32, device=device),
            persistent=False,
        )
        layer._moe_tp_row = row
        setattr(layer.quant_method, _OP_ATTR, stacked_op)
    global _STACKED_FORM
    _STACKED_FORM = f"{len(layers)}:" + ";".join(str(list(t.shape)) for t in stacks)


def adopt_load_stacks() -> dict | None:
    """Hand the armed target layers the load-time expert stacks (layers.adapter.expert_stack), if this process
    built them: buffers, row index and the stacked op, the per-layer placeholders dropped -- the state
    stack_experts leaves, with no stacking program. Returns the report, or None without load-time stacks.

    Raises when a layer with a row is not armed: its per-layer parameters are placeholders with no data (the
    bytes exist only in the stacks), so it has no unarmed path."""
    from vllm_torchtpu.layers.adapter import expert_stack

    plan = expert_stack.current()
    if plan is None:
        return None
    if plan.adopted:
        return plan.report()
    layers = plan.layers
    problems = []
    gap = plan.incomplete()
    if gap:
        problems.append(gap)
    key = None
    for i, layer in enumerate(layers):
        qm = layer.quant_method
        k = getattr(qm, "_moe_tp_decode_key", None)
        if getattr(qm, _OP_ATTR, None) is None or k is None:
            problems.append(f"{getattr(layer, 'layer_name', i)} is not armed")
        elif key is not None and k != key:
            problems.append(f"{getattr(layer, 'layer_name', i)}: op mesh / statics differ")
        key = key or k
    if problems:
        raise RuntimeError(
            "moe_tp_decode: the load-time expert stacks cannot be adopted ("
            + "; ".join(problems[:6])
            + "). Their layers hold no per-layer expert data, so they cannot run "
            "unarmed; serve with GLM52_DECODE_MK_STACK_E=0 (per-layer experts) or "
            "fix the arming refusal logged above."
        )
    mesh, statics = key
    stacked_op = _get_op(mesh, statics, stacked=True)
    stacks = [plan.stacks[k] for k in expert_stack.KINDS]
    with torch.no_grad():
        for layer in layers:
            for name in _EXPERT_PARAMS:  # placeholders: shape only, never materialized
                layer.register_parameter(name, None)
        _install_stacks(layers, stacks, stacked_op, plan.device)
    plan.adopted = True
    plan.hbm["after_adopt"] = _hbm_stats()
    report = plan.report()
    logger.info("moe_tp_decode: adopted the load-time expert stacks of %d layers: %s", len(layers), report)
    return report


def stack_experts(layers: list) -> dict:
    """Move the routed experts of armed layers (target RoutedExperts modules, in
    row order) into one tensor per kind, [len(layers), E, ...], and point each
    layer's op at the stacks and its row. Returns a report (shapes, bytes, HBM
    before / after). Raises ValueError, changing nothing, if a precondition
    fails; the caller then keeps the per-layer form.

    No second copy: one kind at a time, the stack is built with torch.stack
    (a fresh buffer of exactly the summed bytes) and the per-layer parameters
    are dropped right after, so a transient of one kind's bytes exists only
    during the call. The stock MoE reads the same bytes afterwards (the
    kernel's and gmm_v2's in-place row index), so its numerics are unchanged.
    """
    from vllm_torchtpu.layers.core import fused_moe_gmm
    from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

    if not layers:
        raise ValueError("no layers")
    if fused_moe_gmm._DENSE_EXPERT_THRESHOLD > 0:
        raise ValueError(
            "TPU_MOE_DENSE_EXPERT_THRESHOLD > 0 (the dense-expert path takes "
            "per-layer weights)"
        )
    key = None
    for i, layer in enumerate(layers):
        qm = layer.quant_method
        if getattr(qm, _OP_ATTR, None) is None:
            raise ValueError(f"layer #{i} is not armed")
        if getattr(layer, _ROW_BUFFER, None) is not None:
            raise ValueError(f"layer #{i} is already stacked")
        k = getattr(qm, "_moe_tp_decode_key", None)
        if k is None or (key is not None and k != key):
            raise ValueError(f"layer #{i}: op mesh / statics differ from layer #0")
        key = k
        for name in _EXPERT_PARAMS:
            t, t0 = getattr(layer, name), getattr(layers[0], name)
            if t is None or tuple(t.shape) != tuple(t0.shape) or t.dtype != t0.dtype:
                raise ValueError(f"layer #{i} {name}: shape / dtype differ from layer #0")
            if t.storage_offset() != 0 or not t.is_contiguous():
                raise ValueError(f"layer #{i} {name}: not a whole contiguous tensor")
    mesh, statics = key
    stacked_op = _get_op(mesh, statics, stacked=True)
    device = layers[0].w13_weight.device
    if device.type == "tpu":
        # A stacking program that fails puts every deferred computation it was
        # fused with into a failed state (torch_tpu fuses all pending work into
        # the program a synchronize builds; L3c2 probe01 j-ad206c55): run the
        # pending work first, so a failure here can only fail the stack itself.
        synchronize_tensors(None, wait=True)
    report = {"layers": len(layers), "hbm_before": _hbm_stats(), "kinds": {}}
    per_layer_bytes = 0
    stacks = []
    with torch.no_grad():
        try:
            for name in _EXPERT_PARAMS:
                ts = [getattr(layer, name).data for layer in layers]
                nbytes = sum(t.numel() * t.element_size() for t in ts)
                st = torch.stack(ts)
                if st.device.type == "tpu":
                    synchronize_tensors([st])
                del ts
                stacks.append(st)
                for layer in layers:
                    # Drop the per-layer parameter (its HBM returns once the
                    # last reference goes); None keeps the attribute, so a
                    # stray reader fails loudly instead of re-materializing a
                    # view per call.
                    layer.register_parameter(name, None)
                per_layer_bytes += nbytes
                report["kinds"][name] = {
                    "shape": list(st.shape),
                    "dtype": str(st.dtype).replace("torch.", ""),
                    "bytes": st.numel() * st.element_size(),
                    "per_layer_bytes": nbytes,
                    "hbm_after": _hbm_stats(),
                }
        except Exception as e:
            # Put the per-layer parameters back (fresh buffers, not views of
            # the stacks), so a failure leaves the per-layer form.
            for name, st in zip(_EXPERT_PARAMS, stacks):
                for row, layer in enumerate(layers):
                    if getattr(layer, name) is None:
                        layer.register_parameter(
                            name,
                            torch.nn.Parameter(fresh_buffer(st[row]), requires_grad=False),
                        )
            stacks.clear()
            raise ValueError(f"stacking failed ({type(e).__name__}: {e}); per-layer form kept") from e
        _install_stacks(layers, stacks, stacked_op, device)
    report["per_layer_bytes"] = per_layer_bytes
    report["stacked_bytes"] = sum(s.numel() * s.element_size() for s in stacks)
    report["hbm_after"] = _hbm_stats()
    logger.info("moe_tp_decode: stacked the routed experts of %d layers: %s", len(layers), report)
    return report
