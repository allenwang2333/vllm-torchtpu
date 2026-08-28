# SPDX-License-Identifier: Apache-2.0
"""Host-side tap on the global MoE routing histogram.

Inert unless ``TPU_MOE_DUMP_EXPERT_LOAD=1``. It exists to measure the
per-layer, per-expert load that a *static* expert permutation needs. The
xprof profile only carries the load's **time image** -- the per-rank
``gmm_v2`` spread -- and nothing in the job-output tree carries a 512-bin
histogram, so the number has to come from a dedicated run.

Placement: the tap sits on ``topk_ids`` in ``fused_moe_gmm`` *before* the
``experts_start`` remap, where the ids are still global and already gathered
across the EP group. Every rank therefore sees identical values and a single
rank's dump is the whole picture. The tap is nonetheless emitted on **every**
rank: gating the graph on rank identity would make ranks compile structurally
different programs, which is the exact failure mode ``experts_start`` is a
traced array to avoid (see ``fused_moe_gmm``'s docstring).

Layer identity is deliberately not plumbed through the JIT. Passing it as a
static argument would force one compilation per layer; passing it as traced
data would mean a signature change down the whole custom-op path. Instead the
callback is ``ordered=True`` and the tap only fires on one static token shape,
so every executed step contributes exactly ``num_layers`` firings in order and
the Nth firing is layer ``N % num_layers``. The dump carries the raw firing
index so that assumption is checkable offline: a correct phase shows the high
cross-step rank correlation the profile already established, a wrong one shows
none.

This is measurement scaffolding for a throwaway run, not a production path --
the histogram is a scatter-add over ~160k ids and the callback is a host
round trip per layer per step.
"""

from __future__ import annotations

import atexit
import json
import os
import threading

import jax
import jax.numpy as jnp
import numpy as np

_STATE_LOCK = threading.Lock()
_STATE: dict = {
    "started": False,
    "firings": 0,
    "num_layers": 0,
    "num_experts": 0,
    # layer -> np.int64[num_experts]
    "counts": None,
    # layer -> number of steps accumulated
    "steps": None,
    # Raw per-firing histograms for the first few steps. The accumulator
    # above already folds firings into layers via ``i % num_layers``; these
    # let that phase be *checked* offline rather than assumed, by correlating
    # firing i against firing i + num_layers. A correct phase reproduces the
    # high cross-step per-layer correlation the profile measured; a wrong one
    # shows none. Bounded to AUDIT_STEPS steps so it cannot grow.
    "audit_raw": [],
}

AUDIT_STEPS = 3


def enabled() -> bool:
    return os.getenv("TPU_MOE_DUMP_EXPERT_LOAD", "0") not in ("0", "", "false",
                                                              "False")


def _num_layers() -> int:
    return int(os.getenv("TPU_MOE_DUMP_EXPERT_LOAD_LAYERS") or "0")


def _num_experts() -> int:
    return int(os.getenv("TPU_MOE_DUMP_EXPERT_LOAD_EXPERTS") or "0")


def _min_tokens() -> int:
    """Gathered token count at which the tap arms.

    Default 16384 = the full-fill prefill step (2048 local x 8 EP ranks),
    which is the step the re-placement prize is measured on. Restricting to a
    single shape also keeps steps from interleaving across compiled programs,
    which is what the ``N % num_layers`` phase relies on.
    """
    return int(os.getenv("TPU_MOE_DUMP_EXPERT_LOAD_MIN_TOKENS") or "16384")


def _period_s() -> float:
    return float(os.getenv("TPU_MOE_DUMP_EXPERT_LOAD_PERIOD_S") or "60")


def _rank() -> int:
    try:
        return int(jax.process_index())
    except Exception:  # pragma: no cover - defensive
        return -1


def _snapshot() -> dict | None:
    with _STATE_LOCK:
        if _STATE["counts"] is None:
            return None
        return {
            "kind": "expert_load",
            "rank": _rank(),
            "firings": _STATE["firings"],
            "num_layers": _STATE["num_layers"],
            "num_experts": _STATE["num_experts"],
            "steps": _STATE["steps"].tolist(),
            "audit_raw": [[i, c.tolist()] for i, c in _STATE["audit_raw"]],
            "counts": _STATE["counts"].tolist(),
        }


def _emit() -> None:
    snap = _snapshot()
    if snap is None:
        return
    # One line, so it survives log interleaving from the other EP ranks.
    print("EXPERTLOAD_JSON " + json.dumps(snap, separators=(",", ":")),
          flush=True)


def _printer_loop() -> None:
    import time
    period = _period_s()
    while True:
        time.sleep(period)
        try:
            _emit()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"EXPERTLOAD_ERROR printer: {exc!r}", flush=True)


def _ensure_started(num_layers: int, num_experts: int) -> None:
    if _STATE["started"]:
        return
    _STATE["started"] = True
    _STATE["num_layers"] = num_layers
    _STATE["num_experts"] = num_experts
    _STATE["counts"] = np.zeros((num_layers, num_experts), dtype=np.int64)
    _STATE["steps"] = np.zeros((num_layers, ), dtype=np.int64)
    threading.Thread(target=_printer_loop, daemon=True,
                     name="expert-load-dump").start()
    atexit.register(_emit)
    print(
        f"EXPERTLOAD_START rank={_rank()} num_layers={num_layers} "
        f"num_experts={num_experts} min_tokens={_min_tokens()}",
        flush=True)


def _record(counts, num_layers: int) -> None:
    """Host callback. Must never raise -- it runs inside the model step."""
    try:
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        with _STATE_LOCK:
            _ensure_started(num_layers, int(counts.shape[0]))
            i = _STATE["firings"]
            _STATE["firings"] = i + 1
            layer = i % num_layers
            _STATE["counts"][layer] += counts
            _STATE["steps"][layer] += 1
            if i < AUDIT_STEPS * num_layers:
                _STATE["audit_raw"].append((i, counts.copy()))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"EXPERTLOAD_ERROR record: {exc!r}", flush=True)


def maybe_tap_global_topk_ids(topk_ids: jax.Array) -> None:
    """Accumulate a global expert histogram for this MoE call.

    ``topk_ids`` must be the gathered, still-global ids -- call this before
    the ``experts_start`` remap. No-op unless the dump is enabled and this
    graph is the armed token shape. Returns nothing and creates no data
    dependency; ``jax.debug.callback`` is side-effecting so XLA keeps it.

    The layer count and expert count come from the environment rather than
    the call site so that no signature on the custom-op path has to change
    for a one-off measurement run.
    """
    if not enabled():
        return
    num_layers = _num_layers()
    global_num_experts = _num_experts()
    if num_layers <= 0 or global_num_experts <= 0:
        # Nothing sane to attribute the firings to; say so once rather than
        # silently recording an unusable dump.
        print(
            "EXPERTLOAD_ERROR TPU_MOE_DUMP_EXPERT_LOAD=1 but "
            "TPU_MOE_DUMP_EXPERT_LOAD_LAYERS / _EXPERTS unset; tap disabled",
            flush=True)
        return
    num_tokens = topk_ids.shape[0]
    if num_tokens < _min_tokens():
        return

    flat = topk_ids.reshape(-1).astype(jnp.int32)
    counts = jax.ops.segment_sum(jnp.ones_like(flat),
                                 flat,
                                 num_segments=global_num_experts,
                                 indices_are_sorted=False)
    jax.debug.callback(lambda c: _record(c, num_layers), counts, ordered=True)
