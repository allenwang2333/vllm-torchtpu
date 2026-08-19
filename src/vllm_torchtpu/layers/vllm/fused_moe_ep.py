# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for the fused expert-parallel MoE kernel (``USE_MOE_FUSED_EP_V2``).

Token ownership is the whole story here. The kernel wants the EP group's tokens
*sharded* one slice per rank, and returns each rank its own slice, already
combined across every expert. vLLM hands us tokens that are replicated across
the TP group (the MoE input comes out of a RowParallelLinear all-reduce), and
replicated across the DP group only if the DP all-gather ran -- which we
suppress (see ``VllmFp8MoEMethodTPU.supports_internal_mk``).

So each rank takes row slice ``tp_rank`` of ``tp_size`` from its DP-local batch,
runs the kernel, and scatters its result rows back into a zero-filled buffer of
the DP-local shape. vLLM's existing TP all-reduce then does double duty: it
sums the disjoint row slices *and* the TP-sharded shared-expert partials, at
exactly the size it already had. With ``tp_size == 1`` (the DP8_EP layout) none
of that runs: in, kernel, out.

Every rank-derived value crosses into the traced graph as tensor data, never as
a Python constant -- see ``moe_routing.py`` and fa8faaf5.

The kernel needs every rank to bring the *same* number of rows (a shard_map
shard is one shape). That holds because ``TPUModelRunner._dp_coordinated_step``
MAX-all-reduces the token count over the DP CPU group and pads to a shared
bucket before the forward -- the same invariant the DP all-gather relies on
today.
"""

from __future__ import annotations

import functools
import os

import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME, EP_AXIS_NAME_1D,
                                               get_ep_group_layout,
                                               get_or_create_ep_mesh_1d,
                                               validate_ep_mesh)
from vllm_torchtpu.layers.common.fused_moe_ep_v2 import (CAPACITY_SKEW,
                                                         fused_ep_moe_v2_func)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_op_cache: dict[tuple, object] = {}
_op_counter = 0

# tokens, gating, w1, w2, w1_scale, w2_scale -- all sharded on dim 0 (tokens by
# row, weights by expert). Same six operands either way; only the axis name
# differs, because v2's kernel body names its own single mesh axis.
_INPUT_PARTITION_SPECS = (PartitionSpec(EP_AXIS_NAME), ) * 6
# v2 takes a seventh: this process's EP rank, as `int32[1]` sharded over the
# same axis so the exported module sees `arange(ep)`. It has to be an operand
# rather than a constant -- see `ep_ids` in layers/common/fused_moe_ep_v2.
_INPUT_PARTITION_SPECS_1D = (PartitionSpec(EP_AXIS_NAME_1D), ) * 7


def _input_partition_specs():
    return _INPUT_PARTITION_SPECS_1D


# Node-wide tokens above which v2 stops being worth engaging *on the drop-free
# slab*. Measured on an 8-chip v7x node, whole-MoE-layer microseconds, one
# clock, against the production GMM path: 0.88x at 2048, 0.59x at 8192, then
# 4.79x at 16384 -- 15,620 us against 3,264. The cause is the slab itself, so
# `capacity_skew` removes the reason for this cap and `max_tokens` returns none
# when it is set; after that fix 16384 runs in 1,906 us, i.e. 1.71x. The cap
# survives for the deployment that turns the slab sizing off.
_V2_MAX_TOKENS = 8192

# The shipped slab margin. The statistical requirement is ~1.05 -- a shard's
# pooled CV over its 64 local experts is 0.648% -- and the rest is headroom for
# load skew. Exceeding it drops the excess pairs and masks them to zero weight,
# so the cost is quality, not correctness.
_V2_CAPACITY_SKEW = CAPACITY_SKEW


def capacity_skew() -> float:
    """Slab skew margin; <=0 means the drop-free bound (and the cliff)."""
    configured = envs.MOE_FUSED_EP_V2_CAPACITY_SKEW
    if configured >= 0:
        return configured
    return _V2_CAPACITY_SKEW


def max_tokens() -> int:
    """Node-wide token ceiling for engaging the fused kernel; <=0 means none.

    The v2 default exists only because of the cliff, and the cliff is the
    drop-free slab -- so a deployment that sizes the slab by
    ``MOE_FUSED_EP_V2_CAPACITY_SKEW`` has removed the reason for the cap and
    gets no cap. Setting the env explicitly still wins over both.
    """
    configured = envs.MOE_FUSED_EP_KERNEL_MAX_TOKENS
    if configured >= 0:
        return configured
    if capacity_skew():
        return 0
    return _V2_MAX_TOKENS


# What the layer told itself, per distinct forward shape: (rows, coordinated)
# Set by `is_enabled` when a layer accepts the fused path: the DP size that
# layer will key its own decisions on, and how many layers accepted. Both are
# needed to compute the verdict eagerly, before any layer is reached.
_ENABLED_LAYERS: int = 0
_ENABLED_DP_SIZE: int | None = None


def _register_enabled_layer(dp_size: int) -> None:
    global _ENABLED_LAYERS, _ENABLED_DP_SIZE
    if _ENABLED_DP_SIZE is not None and _ENABLED_DP_SIZE != dp_size:
        raise RuntimeError(
            "Fused EP MoE: two layers disagree about dp_size "
            f"({_ENABLED_DP_SIZE} vs {dp_size}); the engagement verdict is a "
            "per-process decision and cannot be computed for both.")
    _ENABLED_DP_SIZE = dp_size
    _ENABLED_LAYERS += 1


def reset_engagement_state() -> None:
    """Forget the armed layers (tests, and a re-loaded model)."""
    global _ENABLED_LAYERS, _ENABLED_DP_SIZE
    _ENABLED_LAYERS = 0
    _ENABLED_DP_SIZE = None


def fused_path_is_live() -> bool:
    """Did at least one layer accept the fused path in this process?"""
    return _ENABLED_LAYERS > 0


def report_engagement(*, strict: bool | None = None) -> None:
    """Say what was armed and what was compiled, and refuse to be quiet.

    Two independent records of the same fact: how many layers *armed* at weight
    load, and the census of MoE custom calls in the graphs the compiler actually
    received (``compilation/moe_census``). An arm is interpretable only if both
    exist and they agree; every serving arm spent on this kernel so far had
    neither. Arming makes it a bicondition rather than a heuristic: an armed
    layer serves every step, so zero fused calls in the capture phase means the
    kernel is not in the program, whatever the flag said.
    ``MOE_FUSED_EP_REQUIRE_ENGAGEMENT=0`` opts out, which is what an arm that
    *wants* the kernel unreachable (a separator control) must say out loud.
    """
    from vllm_torchtpu.compilation import moe_census

    if strict is None:
        strict = bool(envs.MOE_FUSED_EP_REQUIRE_ENGAGEMENT)
    if not envs.USE_MOE_FUSED_EP_V2:
        return
    # Only the phase being reported on: `profile_run`'s graphs are compiled and
    # then discarded, so they can neither prove nor disprove anything.
    phase = moe_census.RECORDER.phase
    compiled = moe_census.RECORDER.total_in_phase("fused_ep", phase)
    if not _ENABLED_LAYERS:
        # `is_enabled` already logged the refusal reason for each layer.
        logger.error(
            "Fused EP MoE: USE_MOE_FUSED_EP_V2=1 but no layer armed, so this "
            "run is the GMM path with extra logging.")
        if compiled:
            raise RuntimeError(
                f"Fused EP MoE: {compiled} fused custom calls were compiled in "
                f"the {phase!r} phase but no layer armed. The arming record and "
                "the graph disagree, so neither can be trusted.")
        return

    logger.info(
        "Fused EP MoE armed: %d layers, dp=%s, band [%s, %s] node-wide tokens.",
        _ENABLED_LAYERS, _ENABLED_DP_SIZE, envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
        max_tokens() or "inf")
    logger.info("%s", moe_census.RECORDER.table())
    if compiled:
        return
    message = (
        f"Fused EP MoE: USE_MOE_FUSED_EP_V2=1 and {_ENABLED_LAYERS} layers "
        f"armed, but the kernel was compiled into ZERO graphs in the {phase!r} "
        f"phase ({moe_census.RECORDER.graphs} graphs seen in all phases). An "
        "armed layer serves every step, so this is the branch and the graph "
        "disagreeing, not a bucket that fell below a threshold. A serving "
        "number measured now is the GMM path and cannot be attributed to the "
        "kernel. Set MOE_FUSED_EP_REQUIRE_ENGAGEMENT=0 if that is deliberate.")
    if strict:
        raise RuntimeError(message)
    logger.error("%s", message)


def max_engine_node_tokens(dp_size: int) -> int | None:
    """The largest node-wide batch this engine can build, or None if unreadable.

    The scheduler's cap is the per-rank ceiling and the runner pads every step
    to a bucket no larger than it (``tpu_platform`` derives the ladder from the
    same number), so ``cap * dp`` is the widest batch the kernel can be handed.
    """
    from vllm.config import get_current_vllm_config_or_none
    scheduler = getattr(get_current_vllm_config_or_none(), "scheduler_config",
                        None)
    cap = getattr(scheduler, "max_num_batched_tokens", None)
    return None if cap is None else int(cap) * max(dp_size, 1)


def in_token_band(node_tokens: int) -> bool:
    """Is a batch of this size worth the kernel's fixed cost?

    The kernel pays an all-device barrier and a weight refill per local expert
    per token block, which a small batch cannot amortise. Asked of the engine's
    largest batch at weight load, never of a step: see `out_of_band_reason`.
    """
    ceiling = max_tokens()
    if node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        return False
    return ceiling <= 0 or node_tokens <= ceiling


def out_of_band_reason(layer) -> str | None:
    """Why this *deployment* cannot use the kernel, or None if it can.

    **The band is a property of the engine, not of a step.** vLLM compiles the
    backbone once with every Dynamo guard dropped
    (``TorchCompileWithNoGuardsWrapper``), so the first trace's answer to any
    branch is the answer for every bucket for the life of the process -- and
    the first trace that survives is the capture ladder's smallest bucket. A
    per-step engagement test therefore does not select the kernel per step; it
    silently selects it never, which is what the last end-to-end attempt
    measured: a v2-flagged server whose compile count went *down*.

    So the question is asked once, here, against the largest batch the engine
    can build; an armed layer then serves every step.
    """
    dp_size = int(getattr(layer.moe_config, "dp_size", 1))
    if dp_size <= 1:
        replica_size, _ = _replica_group()
        return (
            f"dp_size=1 (TP{replica_size}_EP): this kernel exists to remove the "
            "DP dispatch/combine pair, and with no DP transport to remove the "
            "grouped-matmul path is ahead at every token count we measured")
    node_tokens = max_engine_node_tokens(dp_size)
    if node_tokens is None:
        logger.warning_once(
            "Fused EP MoE: the engine's max_num_batched_tokens is unreadable, "
            "so the token band cannot be checked against it; arming on the "
            "flag alone.")
        return None
    if not in_token_band(node_tokens):
        ceiling = max_tokens()
        return (
            f"the engine's largest batch is {node_tokens} node-wide tokens "
            f"(max_num_batched_tokens x dp_size={dp_size}), outside the band "
            f"[{envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS}, "
            f"{ceiling if ceiling > 0 else 'inf'}] this kernel is faster in. "
            "The band is one decision for the whole engine because vLLM drops "
            "Dynamo's guards after the first trace; change "
            "--max-num-batched-tokens, MOE_FUSED_EP_KERNEL_MIN_TOKENS or "
            "MOE_FUSED_EP_KERNEL_MAX_TOKENS to move it")
    return None


def _replica_group() -> tuple[int, int]:
    """(size, rank) of the group across which MoE input tokens are replicated.

    Under expert parallelism vLLM sets the MoE's own ``tp_size`` to 1 and folds
    TP into the EP axis, so the group that replicates tokens is the model's TP
    group, not ``moe_config.tp_size``.
    """
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
    return get_tensor_model_parallel_world_size(
    ), get_tensor_model_parallel_rank()


def unsupported_reason(layer) -> str | None:
    """Why the fused EP kernel cannot serve ``layer``, or None if it can."""
    moe = layer.moe_config
    if not moe.moe_parallel_config.use_ep:
        return "expert parallelism is off"
    if moe.ep_size <= 1:
        return f"ep_size={moe.ep_size}"
    if getattr(moe, "is_sequence_parallel", False):
        return ("sequence parallelism reduces the MoE output itself, leaving "
                "no all-reduce to reassemble the row slices")
    if getattr(moe, "pcp_size", 1) > 1:
        return "PCP adds its own MoE dispatch/combine pair"
    if getattr(layer, "use_grouped_topk", False):
        return "grouped top-k routing is not implemented in the kernel"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing bypasses the gate the kernel reads"
    if os.getenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY"):
        # The kernel routes internally, so the simulation override would be
        # silently ignored rather than applied.
        return "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set"
    if getattr(layer, "e_score_correction_bias", None) is not None:
        return "bias-corrected expert selection is not implemented in the kernel"
    if float(getattr(layer, "routed_scaling_factor", 1.0)) != 1.0:
        return "routed_scaling_factor != 1 is not implemented in the kernel"
    scoring = getattr(layer, "scoring_func", "softmax")
    if scoring != "softmax":
        return f"scoring_func={scoring!r}; the kernel scores with softmax"
    activation = getattr(layer, "_tpu_activation_str", "silu")
    if activation not in ("silu", "swigluoai"):
        return f"activation={activation!r} is not wired to the kernel"
    if getattr(layer, "has_bias", False):
        return "expert biases are not wired through this path"
    replica_size, _ = _replica_group()
    dp_size = int(getattr(moe, "dp_size", 1))
    if dp_size * replica_size != moe.ep_size:
        return (f"ep_size={moe.ep_size} is not dp_size={dp_size} x "
                f"tp_size={replica_size}; token ownership is undefined")
    if moe.num_experts % moe.ep_size:
        return (f"num_experts={moe.num_experts} is not divisible by "
                f"ep_size={moe.ep_size}")
    if replica_size > 1:
        # Not subsumed by the SMEM bound below, and it cannot be: this bridge
        # *slices* the replicated batch across the TP group before the kernel
        # sees it, so the token count the bound tests carries no tp factor and
        # admits the very shapes v2 was seen dying on. Refused on its own terms
        # until a TP arm is measured.
        return (f"tp_size={replica_size}: the row-slicing path across a TP "
                "group is not measured for this kernel")
    if getattr(moe, "skip_final_all_reduce", False):
        # This layer returns each rank's own rows and zero-fills the rest, so
        # the runner's late all-reduce is what reassembles the batch -- and, in
        # the same collective, what sums a TP-sharded shared expert. Without it
        # the layer loses both, silently.
        return (
            "skip_final_all_reduce leaves no late all-reduce to reassemble "
            "the row slices this path returns")
    if getattr(moe, "is_lora_enabled", False):
        return ("LoRA requires moe_kernel to be set wherever the layer claims "
                "supports_internal_mk, and this path leaves it None")
    reason = _smem_reason(moe)
    if reason is not None:
        return reason
    return None


def _smem_reason(moe) -> str | None:
    """Refuse a deployment whose routing tables cannot fit in one chip's SMEM.

    The kernel prefetches its plan as scalar operands, and exactly one of them
    grows with the token count: one int32 per slab row. That makes SMEM, not
    VMEM or HBM, the ceiling on the batch -- and the failure without this check
    is Mosaic's "Ran out of memory in memory space smem" at the first compile of
    a serving run, which names neither the token count nor the setting.
    Answered here, at weight load, against the largest batch the engine can
    build, rather than per call.
    """
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import (
        _SMEM_OVERHEAD_BYTES, _slab_rows)

    dp_size = int(getattr(moe, "dp_size", 1))
    node_tokens = max_engine_node_tokens(dp_size)
    if not node_tokens:
        return None
    smem = _smem_capacity_bytes()
    budget = smem - _SMEM_OVERHEAD_BYTES
    skew = capacity_skew()
    rows = _slab_rows(node_tokens, moe.experts_per_token, moe.num_experts,
                      moe.ep_size, skew)
    if rows * 4 <= budget:
        return None
    return (
        f"v2's routing tables do not fit in SMEM at this engine's largest "
        f"batch: {node_tokens} node-wide tokens need {rows * 4} B of scalar "
        f"memory against a {budget} B budget (skew={skew}). Lower "
        f"--max-num-batched-tokens, or raise MOE_FUSED_EP_V2_CAPACITY_SKEW to "
        f"shrink the slab")


def _smem_capacity_bytes() -> int:
    """This chip's scalar memory, or tpu7x's where there is no chip attached."""
    try:
        from jax.experimental.pallas import tpu as pltpu
        return int(pltpu.get_tpu_info().smem_capacity_bytes)
    except Exception:  # noqa: BLE001 - CPU host, or a device kind we do not know
        return 1024 * 1024


def is_enabled(layer) -> bool:
    """Read the flag once per layer, and log why it did or did not engage."""
    if not envs.USE_MOE_FUSED_EP_V2:
        return False
    if envs.USE_MOE_EP_KERNEL:
        raise ValueError(
            "USE_MOE_FUSED_EP_V2=1 and USE_MOE_EP_KERNEL=1 are both set. They "
            "are different kernels with different weight layouts and different "
            "collective ids, and both would claim this layer's output "
            "contract; set exactly one.")
    reason = unsupported_reason(layer) or out_of_band_reason(layer)
    if reason is not None:
        logger.warning_once(
            "USE_MOE_FUSED_EP_V2=1 but the fused EP MoE kernel cannot serve "
            "this layer (%s); falling back to the GMM path.", reason)
        return False
    validate_ep_mesh(layer, reindexes_expert_order=True)
    # The parallel sizes this path keys every decision on, printed once, because
    # three of them are read off `moe_config` and one of those (`tp_size`) is
    # known to be rewritten under expert parallelism -- so "which number did it
    # actually use" has to be answerable from a boot log rather than from
    # reading the code.
    replica_size, replica_rank = _replica_group()
    logger.info_once(
        "Fused EP MoE parallel sizes: moe_config dp=%s tp=%s ep=%s rank=%s; "
        "model TP group size=%s rank=%s.",
        getattr(layer.moe_config, "dp_size", None),
        getattr(layer.moe_config, "tp_size", None), layer.moe_config.ep_size,
        getattr(layer.moe_config, "ep_rank", None), replica_size, replica_rank)
    dp_size = int(getattr(layer.moe_config, "dp_size", 1))
    ceiling = max_tokens()
    layout = get_ep_group_layout()
    # The engagement gate greps this line, so it says what is true of the
    # *program*: armed layers serve every step, because the band was answered
    # once, above. A layer that declined logs "cannot serve this layer" instead,
    # and the two share no prefix.
    logger.info_once(
        "Fused EP MoE kernel armed for every step: ep_size=%d top_k=%d "
        "dp_size=%d | largest node-wide batch %s in band [%d, %s] | "
        "mesh_index=%d ep_rank_at_mesh_index=%s", layer.moe_config.ep_size,
        layer.moe_config.experts_per_token, dp_size,
        max_engine_node_tokens(dp_size), envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
        ceiling if ceiling > 0 else "inf", layout.mesh_index,
        layout.ep_rank_at_mesh_index)
    _register_enabled_layer(int(getattr(layer.moe_config, "dp_size", 1)))
    return True


def block_size_override() -> tuple[int, ...] | None:
    """Parse ``MOE_FUSED_EP_KERNEL_BLOCKS`` (bt,bf,bd1,bd2,btc,bfc,bd1c,bd2c)."""
    raw = envs.MOE_FUSED_EP_KERNEL_BLOCKS.strip()
    if not raw:
        return None
    blocks = tuple(int(v) for v in raw.replace(" ", "").split(","))
    if len(blocks) != 8:
        raise ValueError(
            "MOE_FUSED_EP_KERNEL_BLOCKS wants 8 comma-separated ints "
            f"(bt,bf,bd1,bd2,btc,bfc,bd1c,bd2c); got {raw!r}")
    return blocks


def mesh_expert_order() -> tuple[int, ...] | None:
    """Which EP rank's expert block sits at each mesh index, or None if they
    agree.

    None keeps the program byte-identical on a node whose devices come up in
    rank order, and is what a pure-JAX caller gets: there, `device_put` splits a
    global array by mesh order, so mesh index i *is* expert block i.
    """
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group_layout

    layout = get_ep_group_layout()
    if layout.mesh_order_is_ep_rank_order:
        return None
    logger.info_once(
        "Fused EP MoE: the node's EP ranks are not in device-id order, so the "
        "expert axis is relabelled into mesh order | device_ids=%s | "
        "ep_rank_at_mesh_index=%s", layout.device_ids,
        layout.ep_rank_at_mesh_index)
    return layout.ep_rank_at_mesh_index


def _v2_partial(*, mesh, top_k: int, renormalize: bool, activation: str,
                **_v1_only):
    """v2 routes and quantizes on its own terms, so the v1 knobs do not apply.

    ``scoring_fn`` is softmax by construction and ``quant_act`` is not optional
    (the wire itself is fp8); both are refused for this version in
    ``unsupported_reason`` rather than silently ignored here. ``subc_quant_*``
    have no analogue: v2 takes per-output-channel weight scales only.
    """
    return functools.partial(
        fused_ep_moe_v2_func,
        mesh=mesh,
        top_k=top_k,
        renormalize=renormalize,
        activation=activation,
        sharded_plan=envs.MOE_FUSED_EP_V2_SHARDED_PLAN,
        capacity_skew=capacity_skew() or None,
        mesh_ep_ranks=mesh_expert_order(),
    )


def op_cache_key(*, top_k: int, renormalize: bool, activation: str,
                 scoring_fn: str, subc_quant_w1_sz: int | None,
                 subc_quant_w2_sz: int | None, quant_act: bool) -> tuple:
    """What ``_get_op`` caches under, for the config the forward passes.

    Shared rather than spelled twice: a key built here and a key built at the
    prebuild that drift apart move the op *build* into the Dynamo-compiled
    forward, which fails on a chip and nowhere else. That skew is what
    ``test_the_forward_never_builds_the_op`` exists to catch, so the two sides
    read it from one place.
    """
    return (top_k, renormalize, activation, scoring_fn, subc_quant_w1_sz,
            subc_quant_w2_sz, quant_act, block_size_override())


def _get_op(*, top_k: int, renormalize: bool, activation: str, scoring_fn: str,
            subc_quant_w1_sz: int | None, subc_quant_w2_sz: int | None,
            quant_act: bool):
    blocks = block_size_override()
    key = op_cache_key(top_k=top_k,
                       renormalize=renormalize,
                       activation=activation,
                       scoring_fn=scoring_fn,
                       subc_quant_w1_sz=subc_quant_w1_sz,
                       subc_quant_w2_sz=subc_quant_w2_sz,
                       quant_act=quant_act)
    op = _op_cache.get(key)
    if op is not None:
        return op

    global _op_counter
    name = f"pallas::fused_ep_moe_kernel_{_op_counter}"
    _op_counter += 1
    mesh = get_or_create_ep_mesh_1d()
    fn = _v2_partial(mesh=mesh,
                     top_k=top_k,
                     renormalize=renormalize,
                     activation=activation,
                     scoring_fn=scoring_fn,
                     subc_quant_w1_sz=subc_quant_w1_sz,
                     subc_quant_w2_sz=subc_quant_w2_sz,
                     quant_act=quant_act,
                     blocks=blocks)
    op = pallas.jax_op(name,
                       fn,
                       mesh=mesh,
                       input_partition_specs=_input_partition_specs())

    # The default fake kernel re-exports to learn the output shape and refuses
    # symbolic dimensions; the output is always the token operand's shape.
    op.register_fake(lambda tokens, *_args: torch.empty_like(tokens))
    _op_cache[key] = op
    return op


def prebuild_fused_ep_moe_kernel(**kwargs) -> None:
    """Register the custom op outside compile-time tracing."""
    _get_op(**kwargs)


def register_ep_rank_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._ep_rank_ids`` as this process's EP rank, ``int32[1]``.

    The kernel needs each shard to know its own index, and the two obvious
    spellings both fail through the bridge: ``lax.axis_index`` lowers to an XLA
    `partition-id` and the SPMD partitioner refuses the exported module, and an
    `arange(ep)` built inside the traced function is a *sharded constant* whose
    partitioning the compiler implements by emitting the same `partition-id`.
    Only a real parameter survives, because the compiler cannot fold what it
    cannot see. Both failures were found on device, a commit apart.

    Sharded over the EP axis, so the global operand is ``arange(ep)`` and this
    process contributes its own element.

    The value is this process's **mesh index**, not its EP rank. The kernel
    uses it for two things at once -- which expert block it owns, and which
    peer coordinate others push to -- and the second is a mesh coordinate by
    construction (it is compared against `lax.axis_index`). Since the mesh is
    laid out in device-id order, the two differ on a node whose devices do not
    come up in rank order, and shipping the EP rank there deadlocks the group.
    The *first* role is what `mesh_ep_ranks` compensates for.
    """
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group_layout

    layer.register_buffer("_ep_rank_ids",
                          torch.tensor([get_ep_group_layout().mesh_index],
                                       dtype=torch.int32,
                                       device=device),
                          persistent=False)


def register_replica_rank_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._ep_replica_rank`` as a persistent 0-d int32 buffer.

    Same reasoning as ``moe_routing.register_experts_start_buffer``: this rank's
    index within the token-replica (TP) group decides which rows it owns, and it
    must reach the compiled graph as module state, not as a Python int folded
    into a constant.
    """
    replica_size, replica_rank = _replica_group()
    buffer = (torch.tensor(replica_rank, dtype=torch.int32, device=device)
              if replica_size > 1 else None)
    layer.register_buffer("_ep_replica_rank", buffer, persistent=False)


# Smallest per-rank row count the kernel is happy with: its default block sizes
# derive `btc = min(local_rows // 2, 64)`, which must stay >= 1, and 8 keeps the
# rows on the sublane granularity. AOT-gated at 8 in
# tests/kernels/fused_moe/test_fused_ep_aot.py.
_MIN_LOCAL_ROWS = 8


def _padded_rows(num_rows: int, replica_size: int) -> int:
    """Row count to pad the DP-local batch to before slicing it per replica."""
    per_replica = -(-num_rows // replica_size)
    per_replica = max(_MIN_LOCAL_ROWS,
                      -(-per_replica // _MIN_LOCAL_ROWS) * _MIN_LOCAL_ROWS)
    return per_replica * replica_size


def _row_slice_index(num_rows: int, replica_size: int,
                     replica_rank: torch.Tensor) -> torch.Tensor:
    rows_per_replica = num_rows // replica_size
    return replica_rank * rows_per_replica + torch.arange(
        rows_per_replica, dtype=torch.int32, device=replica_rank.device)


def _scatter_replica_rows(local_rows: torch.Tensor, replica_size: int,
                          replica_rank: torch.Tensor) -> torch.Tensor:
    """``[S, H]`` -> ``[replica_size * S, H]``, zero outside this rank's block.

    A masked broadcast rather than a scatter: ``index_copy_`` demands an int64
    index, and ``narrow`` with a tensor start calls ``.item()`` (a lazy-tensor
    sync, and it would fold the rank back into a constant).
    """
    mask = (torch.arange(replica_size,
                         dtype=torch.int32,
                         device=local_rows.device) == replica_rank).to(
                             local_rows.dtype)
    blocks = local_rows.unsqueeze(0) * mask.view(replica_size, 1, 1)
    return blocks.reshape(-1, local_rows.shape[-1])


def fused_moe_ep(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    *,
    top_k: int,
    renormalize: bool,
    activation: str,
    scoring_fn: str,
    subc_quant_w1_sz: int | None,
    subc_quant_w2_sz: int | None,
    quant_act: bool,
    replica_rank: torch.Tensor | None,
    ep_rank_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused EP MoE for this rank's DP-local batch.

    ``hidden_states`` / ``router_logits`` are the DP-local rows; the returned
    tensor has the same row count and is zero outside this rank's slice when
    ``tp_size > 1`` (vLLM's TP all-reduce reassembles it).
    """
    op = _get_op(top_k=top_k,
                 renormalize=renormalize,
                 activation=activation,
                 scoring_fn=scoring_fn,
                 subc_quant_w1_sz=subc_quant_w1_sz,
                 subc_quant_w2_sz=subc_quant_w2_sz,
                 quant_act=quant_act)

    replica_size, _ = _replica_group()
    num_rows = hidden_states.shape[0]
    pad_rows = _padded_rows(num_rows, replica_size) - num_rows
    if pad_rows:
        hidden_states = torch.nn.functional.pad(hidden_states,
                                                (0, 0, 0, pad_rows))
        router_logits = torch.nn.functional.pad(router_logits,
                                                (0, 0, 0, pad_rows))

    if replica_size > 1:
        assert replica_rank is not None, (
            "register_replica_rank_buffer() was not called for this layer")
        index = _row_slice_index(num_rows + pad_rows, replica_size,
                                 replica_rank)
        local_states = torch.index_select(hidden_states, 0, index)
        local_logits = torch.index_select(router_logits, 0, index)
    else:
        local_states, local_logits = hidden_states, router_logits

    assert ep_rank_ids is not None, (
        "register_ep_rank_buffer() was not called for this layer; the kernel "
        "reads its shard index out of an operand")
    operands = [
        local_states, local_logits, w1, w2, w1_scale, w2_scale, ep_rank_ids
    ]
    local_out = op(*operands)

    if replica_size > 1:
        out = _scatter_replica_rows(local_out, replica_size, replica_rank)
    else:
        out = local_out
    return out[:num_rows] if pad_rows else out
