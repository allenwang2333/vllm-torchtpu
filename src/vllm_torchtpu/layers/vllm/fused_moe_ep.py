# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for the fused expert-parallel MoE kernel (``USE_MOE_EP_KERNEL``).

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

import contextlib
import functools
import os
from typing import NamedTuple

import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME, EP_AXIS_NAME_1D,
                                               get_or_create_ep_mesh,
                                               get_or_create_ep_mesh_1d,
                                               validate_ep_mesh)
from vllm_torchtpu.layers.common.fused_moe_ep import fused_ep_moe_func
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


def kernel_version() -> int:
    """Which fused-EP program serves this process: 1 (default) or 2.

    One per process, not one per layer: the two versions take different meshes
    and different collective ids, and the choice is a serving decision rather
    than a model property.
    """
    version = int(envs.MOE_FUSED_EP_KERNEL_VERSION)
    if version not in (1, 2):
        raise ValueError(
            f"MOE_FUSED_EP_KERNEL_VERSION={version}; the fused EP MoE kernel "
            "ships v1 (token-stationary, measured -1.4% end to end) and v2 "
            "(weight-stationary, the prefill kernel).")
    return version


def _input_partition_specs(version: int):
    return _INPUT_PARTITION_SPECS if version == 1 else _INPUT_PARTITION_SPECS_1D


def engages_for(layer, hidden_states: torch.Tensor) -> bool:
    """Is this step big enough for the fused kernel to be worth it?

    The kernel pays a fixed per-token-block cost -- an all-device barrier and a
    weight refill per local expert -- that a decode-sized batch cannot amortize,
    so below the threshold the step stays on the GMM path. Both paths read the
    same weights, so this is a per-step choice; and because vLLM compiles one
    program per token bucket, the branch is resolved at trace time.

    Counted node-wide over distinct tokens (``rows x dp_size``): rows are
    replicated across the TP group, and it is the node's total that decides how
    much work each expert sees.

    It is a BAND, not a floor, because v2 is fast in the middle and slow at
    both ends -- see ``max_tokens``.

    A step that answers False must still be dispatched and combined -- see
    ``dispatch_for_fallback``.
    """
    return engaged_at(dp_agreed_tokens(layer, hidden_states))


def engaged_at(node_tokens: int | None) -> bool:
    """The engagement rule. **Comparisons only** -- this one is traced.

    Nothing here may *format* ``node_tokens``. It is a Dynamo `SymInt` derived
    from `hidden_states.shape[0]`, and comparing it adds the shape guard the
    bucket already has, but stringifying it forces the symbol to a constant and
    the compile dies with
    ``Constraints violated (L['input_ids'].size()[0], ...)`` -- inside
    `capture_model`, on device, with the boot lost. That is not hypothetical:
    an f-string in the *reason* text did exactly this and voided the v2 arm of
    job `j-2ca2a7d7`. The reason belongs in `engagement_verdict`, which is
    eager-only.

    Same family as the two failures before it -- a `logging.Logger` call and a
    tensor comparison in the traced region. The rule that covers all three:
    **the traced path may compute a bool and nothing else.**
    """
    if node_tokens is None:
        return False
    lo = envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS
    hi = max_tokens()
    return (lo <= 0 or node_tokens >= lo) and (hi <= 0 or node_tokens <= hi)


class Verdict(NamedTuple):
    """Whether the fused path takes a step, and the sentence that says why."""

    engaged: bool
    reason: str


def engagement_verdict(node_tokens: int | None) -> Verdict:
    """``engaged_at`` plus the sentence that explains it. **EAGER CODE ONLY.**

    Built on the same predicate the trace applies, so the ledger cannot drift
    from the graph -- but it formats the token count, which a traced caller may
    not do (see `engaged_at`). The one caller is `_record_engagement`.
    """
    engaged = engaged_at(node_tokens)
    if node_tokens is None:
        return Verdict(engaged, "the DP ranks have not agreed on this step")
    lo = envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS
    hi = max_tokens()
    if not engaged:
        bound = (f"< MIN_TOKENS={lo}"
                 if lo > 0 and node_tokens < lo else f"> MAX_TOKENS={hi}")
        return Verdict(False, f"{node_tokens} node-wide tokens {bound}")
    band = f"[{lo if lo > 0 else 0}, {hi if hi > 0 else 'inf'}]"
    return Verdict(True, f"{node_tokens} node-wide tokens in {band}")


def dp_agreed_tokens(layer, hidden_states: torch.Tensor) -> int | None:
    """Node-wide tokens **all ranks compute the same answer for**, or None.

    This is a rendezvous predicate, not an accounting detail. The kernel's
    barrier is a counting semaphore, so it tolerates a peer arriving late but
    not a peer that never arrives -- and ``engages_for`` is evaluated
    independently on every rank. A step where one rank engages and another does
    not is a deadlock, and the way it presents is not a hang with a useful
    message: the TensorCore parks in ``semaphore_wait`` while the routing
    plan's SparseCore-offloaded scatter/gather is still outstanding, and the
    unit that trips is the SparseCore -- `RuntimeUnexpectedCoreHalt ...
    SparseCoreSequencer halted`, no HLO mapping, 25 minutes after boot (W6, jobs
    j-ef24b962 and j-d68b0dd5, both layouts, deterministic).

    Where the agreement comes from:

    * ``dp_size == 1``: the TP ranks hold the same tensor, so its row count is
      the same number everywhere. Nothing to coordinate.
    * ``dp_size > 1``: only from the runner. ``_dp_coordinated_step``
      MAX-all-reduces the token count over the DP CPU group and pads to a shared
      bucket, and it hands the result to ``set_forward_context`` as
      ``num_tokens_across_dp`` -- so **``dp_metadata`` being present IS the
      runner's promise that the ranks agree**. ``_dummy_run`` defaults to
      ``dp_lockstep=False`` and passes ``num_tokens_across_dp=None``
      (``runner/tpu_runner.py:4205-4207``), which is why the profile run is
      where this bites first, and why an idle DP engine's dummy batch would bite
      the same way on a real serving step.

    Absent that promise we decline and the GMM path runs the step: it has no
    barrier, so a rank that declines alone costs correctness nothing (the
    fallback dispatch/combine pair is reinstated by ``dispatch_for_fallback``).
    """
    # No logging on this path: it runs inside the Dynamo-traced region, and a
    # `logging.Logger` call in a graph being captured is fatal --
    # "logging.Logger method not supported for non-export cases" (W6,
    # j-cf06d716). The decline is announced from `dp_agreement`, which is
    # eager. Nothing else here may log either, for the same reason.
    dp_size = int(getattr(layer.moe_config, "dp_size", 1))
    if not dp_step_is_coordinated(dp_size):
        return None
    return hidden_states.shape[0] * dp_size


# Node-wide tokens above which v2 stops being worth engaging *on the drop-free
# slab*. Measured on an 8-chip v7x node, whole-MoE-layer microseconds, one
# clock, against the production GMM path (W1, job j-f52bceac): v2 is 0.88x at
# 2048 and 0.59x at 8192, and then 4.79x at 16384 -- 15,620 us against 3,264.
# The cause is the slab itself, so `capacity_skew` removes the reason for this
# cap and `max_tokens` returns none when it is set; measured after the fix
# (job j-9dd50718), 16384 runs in 1,906 us, i.e. 1.71x production rather than
# 4.79x. The cap survives for the deployment that turns the slab sizing off.
_V2_MAX_TOKENS = 8192

# Skew margin for v2's per-shard slab, and the default is ON: the drop-free
# slab is what makes the 16384-token bucket cost 15,620 us/layer against
# production's 3,264, and sizing it for the occupancy routing delivers takes it
# to ~1,874 (W1, job j-55b4be57, measured at the equivalent slab). The
# *statistical* requirement is ~1.05 -- W5's pooled CV over a shard's 64 local
# experts is 0.648% -- so all of this margin is headroom for load skew, and
# three independent bounds say the real skew is far under it: seeded draws show
# 1.11x per-rank spread at 64 tokens and 1.004x at 256; the shipping server's
# xprof puts the EP straggler wait at 48% of the MoE combine, which no 3x shard
# imbalance is consistent with; and the prefill capture's expert-load solve has
# all 64 local experts active. If it is ever exceeded the excess is dropped and
# those pairs are masked to zero weight, so the cost is quality, not
# correctness. A real routing capture would let someone tighten this.
_V2_CAPACITY_SKEW = CAPACITY_SKEW


def capacity_skew() -> float:
    """Slab skew margin; <=0 means the drop-free bound (and the cliff)."""
    configured = envs.MOE_FUSED_EP_V2_CAPACITY_SKEW
    if configured >= 0:
        return configured
    return _V2_CAPACITY_SKEW if kernel_version() == 2 else 0.0


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
    if kernel_version() != 2 or capacity_skew():
        return 0
    return _V2_MAX_TOKENS


# Whether the DP ranks agreed on the shape of the step currently in flight, as
# a plain Python bool set in EAGER code by whoever launched it. Not derived
# inside the model: `supports_internal_mk` is read from inside the
# Dynamo-traced region (`moe_runner.py:786`), so a tensor comparison there
# becomes an unbacked symint and full-graph capture fails outright --
# "Could not guard on data-dependent expression u4 > 0" (W6, j-1da07731).
# The runner is also the honest place for it: it is what does or does not
# coordinate, so it can simply say so rather than have the layer infer it from
# a DPMetadata that vLLM manufactures either way.
#
# None means nobody said, which is treated as "not agreed": a forward launched
# by a path that does not declare its coordination gets the GMM path, and the
# cost of that mistake is throughput rather than a hung device.
_DP_AGREED: bool | None = None


# What the layer told itself, per distinct forward shape: (rows, coordinated)
# -> (verdict, forwards). Written in EAGER code from `dp_agreement`, using the
# same rule `engages_for` applies inside the trace, so the log describes the
# graph rather than the flag. Read by `report_engagement`.
_LEDGER: dict[tuple[int, bool], tuple[Verdict, int]] = {}

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
    """Forget the layers and the ledger (tests, and a re-loaded model)."""
    global _ENABLED_LAYERS, _ENABLED_DP_SIZE
    _ENABLED_LAYERS = 0
    _ENABLED_DP_SIZE = None
    _LEDGER.clear()


@contextlib.contextmanager
def dp_agreement(coordinated: bool, local_rows: int | None = None):
    """Declare, for the duration of one forward, whether the DP ranks agree.

    Saves and restores rather than assigning, because a stale verdict leaking
    from one step into the next is the same class of bug as the layer-vs-step
    property this whole path has been chasing, one level down.

    ``local_rows`` is what the caller is about to launch. Given it, the
    engagement verdict for this forward is computed and recorded *here*, in
    eager code, where it can be logged -- the traced region cannot log at all
    (a `logging.Logger` call inside a captured graph is fatal, W6 j-cf06d716)
    and that is why three serving arms were spent on numbers nobody could
    attribute to a path.
    """
    global _DP_AGREED
    if not coordinated and _ENABLED_LAYERS:
        # Announced here rather than at the decline itself, which is traced.
        # Only when a layer actually took the fused path: a control run has
        # nothing to decline, and a warning that appears in both arms is not
        # an observable.
        logger.warning_once(
            "Fused EP MoE: this forward's DP ranks have not agreed on a token "
            "count, so the fused path declines it and the GMM path runs. "
            "Expected during profile_run and for an idle DP engine's dummy "
            "batch; the kernel's barrier needs every rank to enter.")
    if local_rows is not None:
        _record_engagement(int(local_rows), bool(coordinated))
    previous, _DP_AGREED = _DP_AGREED, bool(coordinated)
    try:
        yield
    finally:
        _DP_AGREED = previous


def _record_engagement(local_rows: int, coordinated: bool) -> None:
    """Ledger one forward's verdict, and announce each new shape once."""
    if _ENABLED_DP_SIZE is None:
        return  # no layer took the fused path; there is nothing to decide.
    node_tokens = (local_rows * _ENABLED_DP_SIZE if
                   (coordinated or _ENABLED_DP_SIZE <= 1) else None)
    verdict = engagement_verdict(node_tokens)
    key = (local_rows, coordinated)
    previous = _LEDGER.get(key)
    _LEDGER[key] = (verdict, (previous[1] if previous else 0) + 1)
    if previous is None:
        logger.info(
            "Fused EP MoE v%d: %d local rows (dp=%d, coordinated=%s) -> %s "
            "(%s)", kernel_version(), local_rows, _ENABLED_DP_SIZE,
            coordinated, "FUSED" if verdict.engaged else "GMM",
            verdict.reason)


def fused_path_is_live() -> bool:
    """Did at least one layer accept the fused path in this process?"""
    return _ENABLED_LAYERS > 0


def engagement_ledger() -> dict[tuple[int, bool], tuple[Verdict, int]]:
    """A copy of what the layer decided, per forward shape."""
    return dict(_LEDGER)


def report_engagement(*, strict: bool | None = None) -> None:
    """Print what was decided and what was compiled, and refuse to be quiet.

    Two independent records of the same fact: the *intent* ledger, written in
    eager code by the rule the trace applies, and the *census* of MoE custom
    calls in the graphs the compiler actually received
    (``compilation/moe_census``). An arm is interpretable only if both exist
    and they agree; every serving arm spent on this kernel so far had neither.

    Raises when the operator asked for the kernel and it reached zero compiled
    graphs -- the failure mode that has already produced three unattributable
    jobs. ``MOE_FUSED_EP_REQUIRE_ENGAGEMENT=0`` opts out, which is what an arm
    that *wants* the kernel unreachable (a separator control) must say out loud.
    """
    from vllm_torchtpu.compilation import moe_census

    if strict is None:
        strict = bool(envs.MOE_FUSED_EP_REQUIRE_ENGAGEMENT)
    if not envs.USE_MOE_EP_KERNEL:
        return
    if not _ENABLED_LAYERS:
        # `is_enabled` already logged the refusal reason for each layer.
        logger.error(
            "Fused EP MoE: USE_MOE_EP_KERNEL=1 but no layer accepted the fused "
            "path, so this run is the GMM path with extra logging.")
        return

    lines = [
        f"Fused EP MoE v{kernel_version()} engagement ledger "
        f"({_ENABLED_LAYERS} layers, dp={_ENABLED_DP_SIZE}, "
        f"MIN_TOKENS={envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS}, "
        f"MAX_TOKENS={max_tokens() or 'none'}):"
    ]
    for (rows, coordinated), (verdict, forwards) in sorted(_LEDGER.items()):
        lines.append(f"  {rows:>6} local rows, coordinated={coordinated!s:<5} "
                     f"x{forwards:<4} -> "
                     f"{'FUSED' if verdict.engaged else 'GMM  '}  "
                     f"({verdict.reason})")
    if not _LEDGER:
        lines.append("  EMPTY -- no forward declared its shape. The ledger is "
                     "not instrumented on this path, so nothing here can be "
                     "attributed to the kernel.")
    logger.info("\n".join(lines))
    logger.info("%s", moe_census.RECORDER.table())

    intended = any(v.engaged for v, _ in _LEDGER.values())
    # Only the phase being reported on: `profile_run`'s graphs are compiled
    # before the DP coordinator exists and then discarded, so they can neither
    # prove nor disprove anything about serving.
    phase = moe_census.RECORDER.phase
    compiled = moe_census.RECORDER.total_in_phase("fused_ep", phase)
    if compiled and not intended:
        raise RuntimeError(
            f"Fused EP MoE: {compiled} fused custom calls were compiled but "
            "the engagement ledger says no forward engaged. The ledger and "
            "the graph disagree, so neither can be trusted -- ledger=" +
            repr({k: v[0]
                  for k, v in _LEDGER.items()}))
    if compiled:
        return
    message = (
        "Fused EP MoE: USE_MOE_EP_KERNEL=1 and "
        f"{_ENABLED_LAYERS} layers accepted it, but the kernel was compiled "
        f"into ZERO graphs in the {phase!r} phase "
        f"({moe_census.RECORDER.graphs} graphs seen in all phases). "
        + ("The ledger says a forward should have engaged, so the branch and "
           "the graph disagree. " if intended else
           "No forward's shape reached the engagement band. ") +
        "A serving number measured now is the GMM path and cannot be "
        "attributed to the kernel. Set MOE_FUSED_EP_REQUIRE_ENGAGEMENT=0 if "
        "that is deliberate.")
    if strict:
        raise RuntimeError(message)
    logger.error("%s", message)

def dp_step_is_coordinated(dp_size: int) -> bool:
    """Have the DP ranks agreed on this step's shape?

    Everything this layer takes over from the runner is keyed on this one
    predicate, evaluated once, because taking over half of it is worse than
    taking over none: the kernel's barrier and the DP dispatch/combine pair both
    need every rank to arrive, and on an uncoordinated step no rank can know
    what the others will do. A step where one rank engages and another does not
    hangs the TensorCore in `semaphore_wait`, and what trips is the SparseCore
    with an offloaded collective outstanding -- 25 minutes after boot, naming
    neither cause nor remedy (W6, jobs j-ef24b962, j-d68b0dd5, j-2846842c).

    At `dp_size <= 1` there is nothing to coordinate: the TP ranks hold the
    same tensor.
    """
    if dp_size <= 1:
        return True
    return _DP_AGREED is True


def _needs_fallback_dispatch(layer) -> bool:
    """Did we suppress the runner's pair on this step, so must reinstate it?

    Only where the takeover actually happened. ``supports_internal_mk`` is now
    the same predicate, so on an uncoordinated step the runner keeps ownership
    and reinstating here would issue the collectives **twice**.
    """
    dp_size = int(getattr(layer.moe_config, "dp_size", 1))
    return dp_size > 1 and dp_step_is_coordinated(dp_size)


def dispatch_for_fallback(
        layer, hidden_states: torch.Tensor,
        router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the DP all-gather the runner skipped, for a step that fell back.

    ``supports_internal_mk`` is a property of the *layer*, so once the fused
    path is active vLLM stops dispatching and combining for every step -- but
    ``engages_for`` is per step. On a step that falls back to the GMM path, the
    expert contributions of the other DP ranks would simply be missing (the TP
    all-reduce only spans the TP group), so the pair has to be reinstated here.
    """
    if not _needs_fallback_dispatch(layer):
        return hidden_states, router_logits
    from vllm.distributed.parallel_state import get_ep_group
    dispatched = get_ep_group().dispatch_router_logits(hidden_states,
                                                       router_logits, False)
    # Same arity check the runner makes: a communicator that starts returning
    # extra tensors (e.g. a detached scale) must not have them silently dropped.
    assert len(dispatched) == 2, (
        f"dispatch_router_logits returned {len(dispatched)} tensors; the GMM "
        "fallback only forwards hidden_states and router_logits")
    return dispatched[0], dispatched[1]


def combine_for_fallback(layer, hidden_states: torch.Tensor) -> torch.Tensor:
    """The matching DP reduce-scatter; see ``dispatch_for_fallback``."""
    if not _needs_fallback_dispatch(layer):
        return hidden_states
    from vllm.distributed.parallel_state import get_ep_group
    return get_ep_group().combine(hidden_states, False)


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
    version = kernel_version()
    scoring = getattr(layer, "scoring_func", "softmax")
    supported_scoring = ("softmax",
                         "sigmoid") if version == 1 else ("softmax", )
    if scoring not in supported_scoring:
        return f"scoring_func={scoring!r} on fused EP kernel v{version}"
    activation = getattr(layer, "_tpu_activation_str", "silu")
    supported_act = ("silu", "gelu",
                     "swigluoai") if version == 1 else ("silu", "swigluoai")
    if activation not in supported_act:
        return f"activation={activation!r} on fused EP kernel v{version}"
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
    return None


def is_enabled(layer) -> bool:
    """Read the flag once per layer, and log why it did or did not engage."""
    if not envs.USE_MOE_EP_KERNEL:
        return False
    reason = unsupported_reason(layer)
    if reason is not None:
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but the fused EP MoE kernel cannot serve this "
            "layer (%s); falling back to the GMM path.", reason)
        return False
    validate_ep_mesh(layer)
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
    threshold = envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS
    logger.info_once(
        "Fused EP MoE kernel v%d enabled (ep_size=%d, top_k=%d); engages from "
        "%s node-wide tokens up, smaller steps use the GMM path.",
        kernel_version(), layer.moe_config.ep_size,
        layer.moe_config.experts_per_token,
        threshold if threshold > 0 else "0 (EVERY step, including decode -- "
        "this was measured at -86% end to end)")
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


def _v1_partial(*, mesh, top_k: int, renormalize: bool, activation: str,
                scoring_fn: str, subc_quant_w1_sz: int | None,
                subc_quant_w2_sz: int | None, quant_act: bool, blocks):
    return functools.partial(
        fused_ep_moe_func,
        mesh=mesh,
        ep_axis_name=EP_AXIS_NAME,
        top_k=top_k,
        renormalize=renormalize,
        activation=activation,
        scoring_fn=scoring_fn,
        subc_quant_w1_sz=subc_quant_w1_sz,
        subc_quant_w2_sz=subc_quant_w2_sz,
        quant_act=quant_act,
        blocks=blocks,
    )


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
    return (kernel_version(), top_k, renormalize, activation,
            scoring_fn, subc_quant_w1_sz, subc_quant_w2_sz, quant_act,
            block_size_override())


def _get_op(*, top_k: int, renormalize: bool, activation: str, scoring_fn: str,
            subc_quant_w1_sz: int | None, subc_quant_w2_sz: int | None,
            quant_act: bool):
    blocks = block_size_override()
    version = kernel_version()
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
    build = _v1_partial if version == 1 else _v2_partial
    mesh = (get_or_create_ep_mesh()
            if version == 1 else get_or_create_ep_mesh_1d())
    fn = build(mesh=mesh,
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
                       input_partition_specs=_input_partition_specs(version))

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
    """
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group_layout

    layer.register_buffer("_ep_rank_ids",
                          torch.tensor([get_ep_group_layout().rank_in_group],
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

    operands = [local_states, local_logits, w1, w2, w1_scale, w2_scale]
    if kernel_version() == 2:
        assert ep_rank_ids is not None, (
            "register_ep_rank_buffer() was not called for this layer; v2 reads "
            "its shard index out of an operand")
        operands.append(ep_rank_ids)
    local_out = op(*operands)

    if replica_size > 1:
        out = _scatter_replica_rows(local_out, replica_size, replica_rank)
    else:
        out = local_out
    return out[:num_rows] if pad_rows else out
