# SPDX-License-Identifier: Apache-2.0
"""Per-layer MoE routing histogram, accumulated on device.

Inert unless ``TPU_MOE_DUMP_EXPERT_LOAD=1``. It exists to measure the
per-layer, per-expert load that a *static* expert permutation needs
(``vllm_torchtpu/expert_permutation.py``). The xprof profile only carries the
load's **time image** -- the per-rank ``gmm_v2`` spread, 8 numbers per layer --
which is not enough to repartition 512 experts, and nothing in the job-output
tree carries a 512-bin histogram. So the number has to come from a dedicated
run.

**Why this is not a host callback.** The obvious implementation is
``jax.debug.callback`` on ``topk_ids``. It cannot work: ``torch_tpu`` lowers
every JAX region through ``jax.export`` (``pallas.py`` -> ``do_export`` ->
``_check_lowering``), which raises ``NotImplementedError: serialization of
host_callbacks is not yet implemented``. That is architectural, not a flag --
job ``j-b7f40703-7881-4383-9178`` died at engine-core init on all 8 ranks
proving it. Note ``pallas.jax_op.__call__`` does have an eager, non-exported
branch for wrapper tensors, but the server takes the FX-export path.

So the histogram is accumulated the way this stack already accumulates the KV
cache: the op returns the updated counter functionally, the caller writes it
back with ``copy_``, and XLA aliases the output onto the donated input. See
``layers/vllm/attention.py`` ("Plain donation + copy_ writeback: XLA lifts the
copy_ input mutation and aliases the op output onto the donated pool").

Two details that decide whether the numbers mean anything:

* **Layer identity is explicit**, taken from ``layer.layer_name``, not inferred
  from call order. An earlier draft folded firing ``i`` into layer
  ``i % num_layers``; that is only correct if every trace visits exactly
  ``num_layers`` MoE layers in order, and a silent phase slip would produce a
  plausible-looking histogram for the wrong layers.
* **Padded tokens are excluded.** Under ``TPU_MOE_SKIP_PADDED_TOKENS`` the
  padding rows keep their expert ids and only their routing weights are
  zeroed, so counting ids alone would attribute phantom load to whichever
  experts the padding happened to select. The mask is on ``topk_weights > 0``,
  which is what the kernel itself honours.

This is measurement scaffolding for a throwaway run, not a production path.
"""

import inspect
import json
import os
import threading

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal.pallas.pallas import _infer_static_argnums

_LOCK = threading.Lock()
# layer_name -> the int32[global_num_experts] device counter for that layer.
_BUFFERS: dict[str, torch.Tensor] = {}
_OP = None
_DUMPS = 0


def enabled() -> bool:
    return os.getenv("TPU_MOE_DUMP_EXPERT_LOAD", "0") not in ("0", "", "false",
                                                              "False")


def _dump_every() -> int:
    return int(os.getenv("TPU_MOE_DUMP_EXPERT_LOAD_EVERY") or "500")


def _rank() -> int:
    try:
        return int(jax.process_index())
    except Exception:  # pragma: no cover - defensive
        return -1


def _accumulate(counts: jax.Array, topk_ids: jax.Array,
                topk_weights: jax.Array) -> jax.Array:
    """``counts + histogram(topk_ids)``, ignoring padding and out-of-range."""
    num_experts = counts.shape[0]
    ids = topk_ids.reshape(-1).astype(jnp.int32)
    weights = topk_weights.reshape(-1)
    valid = (ids >= 0) & (ids < num_experts) & (weights > 0)
    safe = jnp.where(valid, ids, 0)
    ones = jnp.where(valid, 1, 0).astype(counts.dtype)
    hist = jax.ops.segment_sum(ones,
                               safe,
                               num_segments=num_experts,
                               indices_are_sorted=False)
    return counts + hist.astype(counts.dtype)


def _get_op():
    global _OP
    if _OP is None:
        # `_infer_static_argnums` classifies an argument as *static* unless its
        # annotation `is jax.Array` -- an identity check against the class. A
        # `from __future__ import annotations` anywhere in this module turns
        # every annotation into a string, all three tensors become static, and
        # the op dies at engine-core init with
        # `TypeError: 'getset_descriptor' object is not iterable`, from inside
        # the compiled MoE forward where it looks like a Dynamo bug rather than
        # an annotation bug. Job j-ba60be15-fc34-4868-bdc9 lost a slot to it.
        # This module therefore must not use postponed annotations, and this
        # asserts it rather than trusting the next edit.
        static = _infer_static_argnums(inspect.signature(_accumulate))
        if static:
            raise RuntimeError(
                "expert_load: _accumulate's tensor args were inferred static "
                f"{static}; annotations are probably strings. Remove any "
                "`from __future__ import annotations` from this module.")
        op = pallas.jax_op("pallas::expert_load_accum",
                           _accumulate,
                           donate_argnums=(0, ))

        def _fake(counts, topk_ids, topk_weights):
            return torch.empty_like(counts)

        op.register_fake(_fake)
        _OP = op
    return _OP


def register_expert_load_buffer(layer) -> None:
    """Allocate this layer's counter. Call where ``experts_start`` is set up."""
    if not enabled():
        return
    name = getattr(layer, "layer_name", "") or ""
    if not name:
        print("EXPERTLOAD_ERROR layer has no layer_name; tap disabled for it",
              flush=True)
        return
    buf = torch.zeros(layer.global_num_experts,
                      dtype=torch.int32,
                      device=layer.w13_weight.device)
    layer.register_buffer("_expert_load_counts", buf, persistent=False)
    with _LOCK:
        _BUFFERS[name] = buf
    # Build the op HERE, eagerly. `maybe_tap` runs inside the compiled MoE
    # forward, so if the op were still unbuilt at the first tap Dynamo would
    # try to symbolically execute `inspect.signature`, `pallas.jax_op` and
    # `torch.library.custom_op` as part of the graph -- which fails with
    # `TypeError: 'getset_descriptor' object is not iterable` from somewhere
    # inside torch's schema inference, reported against qwen3_next.py. Job
    # j-75b2b2a8-3cec-41ac-8258 died that way with all 480 buffers registered
    # and every standalone eager and fullgraph check passing, because the
    # standalone checks happened to build the op before compiling.
    _get_op()
    print(f"EXPERTLOAD_REGISTER rank={_rank()} layer={name} "
          f"experts={layer.global_num_experts}",
          flush=True)


def maybe_tap(layer, topk_ids: torch.Tensor,
              topk_weights: torch.Tensor) -> None:
    """Fold this call's routing into the layer's counter. In-graph, no-op off.

    Must be called *after* padding weights are zeroed, so the mask sees them.

    **Unconditional on purpose.** An earlier version armed only on the
    full-fill prefill shape (``topk_ids.shape[0] >= 2048``) so the histogram
    would not mix prefill and decode routing. That branch reads a *symbolic*
    dimension: Dynamo turns it into a guard, a compile range whose token
    dimension straddles the threshold gets specialised into two traces, and
    the two disagree about whether this layer's counter is a graph input.
    vLLM's artifact cache rejects the second one:

        RuntimeError: artifact_compile_range_4_4_subgraph_0 was compiled by
        another trace of this model whose inputs differ

    Job j-f4752055-9b99-4402-9b07 died that way. Every trace-visible branch
    here has to be trace-invariant, so the only conditions left are the env
    flag and two ``is None`` checks that are fixed for the process.

    The cost is that the counters are a prefill+decode mixture. At
    ``--prefill-schedule-interval=8`` a full prefill step routes 2048 rows
    against ~32 per decode step, so prefill is ~89% of the rows -- and the
    two distributions correlate at Pearson 0.982-0.996 anyway, which is the
    measurement that made a *static* permutation worth trying in the first
    place. The mixture is arguably the better target: the permutation has to
    serve both step types.
    """
    if not enabled():
        return
    buf = getattr(layer, "_expert_load_counts", None)
    if buf is None:
        return
    # Read the already-built op rather than calling `_get_op()`: constructing
    # it here would put op registration inside the traced graph. See
    # `register_expert_load_buffer`.
    op = _OP
    if op is None:
        return
    new_counts = op(buf, topk_ids, topk_weights)
    buf.copy_(new_counts)


def reset() -> None:
    """Zero every counter. Host side, between steps."""
    with _LOCK:
        items = list(_BUFFERS.values())
    for buf in items:
        buf.zero_()


def dump(tag: str = "") -> None:
    """Print every layer's counter as one JSON line. Host side, off-graph.

    Called from the worker between steps -- never from inside the traced
    forward, where ``.cpu()`` would be a graph break.
    """
    global _DUMPS
    if not enabled():
        return
    with _LOCK:
        items = sorted(_BUFFERS.items())
    if not items:
        return
    try:
        counts = {name: buf.cpu().numpy().tolist() for name, buf in items}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"EXPERTLOAD_ERROR dump: {exc!r}", flush=True)
        return
    _DUMPS += 1
    # One line, so it survives interleaving with the other 7 ranks' output.
    print("EXPERTLOAD_JSON " + json.dumps(
        {
            "kind": "expert_load",
            "rank": _rank(),
            "dump": _DUMPS,
            "tag": tag,
            "steps": _STEPS,
            "gate": "unconditional",
            "counts": counts,
        },
        separators=(",", ":")),
          flush=True)


_STEPS = 0


def maybe_dump_after_step() -> None:
    """Periodic host-side dump, driven by the worker's step counter.

    The first call also clears the counters. Now that the tap is
    unconditional it also fires during ``capture_model``'s dummy runs, whose
    routing comes from whatever junk the dummy inputs produce -- counting it
    would be counting noise. ``capture_model`` does not go through
    ``TPUWorker.execute_model``, so ``_STEPS == 1`` is the first *real*
    forward. The pre-reset counters are dumped first, tagged ``warmup``, so
    the discarded amount is visible rather than merely asserted.
    """
    global _STEPS
    if not enabled():
        return
    _STEPS += 1
    if _STEPS == 1:
        dump(tag="warmup")
        reset()
        return
    if _STEPS % _dump_every() == 0:
        dump(tag="serving")
