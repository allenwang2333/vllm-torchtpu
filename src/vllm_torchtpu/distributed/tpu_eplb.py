# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Online expert-parallel load balancing (EPLB) for the fused EP MoE kernel.

Off unless ``VLLM_TPU_EPLB=1``. With it off nothing here is imported by the
serving path and the fused op is built exactly as before.

The server rebalances from load it measures itself, the way vLLM GPU's EPLB
(``vllm/distributed/eplb``) does; no placement is ever read from a file.

* **Counters.** The fused kernel already all-gathers every rank's per-expert
  pair counts (``rows_by_dest``) to build its routing plan, so the global
  per-physical-slot load of a step is free inside the op and is IDENTICAL on
  every rank. The op returns it as a second output and the layer adds it into
  a persistent int32 ``[E]`` buffer inside the compiled graph. The host reads
  the buffers only at a rebalance.
* **Remap.** Placement is a per-layer logical->physical table, an int32 ``[E]``
  device buffer passed to the op (a runtime operand, so a rebalance does not
  recompile). It is applied to the selected ids AFTER top-k, so the router
  logits, the score-correction bias and the tie order are untouched, and the
  combine sums each token's experts in top-k slot order, so the output is
  bit-identical to the unbalanced layer.
* **Placement.** Permutation only: every rank keeps exactly E/ep slots.
  Upstream's ``balanced_packing`` (greedy LPT with equal-cardinality bins)
  over the window's per-logical-expert load, then the packs are matched to
  ranks to keep as many experts where they already are as possible, then the
  experts of each rank are ordered by their current slot (which is the order a
  ragged all-to-all delivers them in).
* **Weights.** Each changed layer's four expert tensors go through one
  ``ragged_all_to_all`` over the EP mesh: a local gather into send order, one
  collective that moves only the experts that change rank, written straight
  into the new slot order.
* **Trigger.** Deterministic on every rank without a collective: counted in
  non-megakernel forwards, which every rank runs in lockstep (each one carries
  the kernel's EP collectives), and decided from the global load table, which
  every rank holds identically.

Knobs (VLLM_ prefix so Ray forwards them to the workers):

  VLLM_TPU_EPLB               1 to arm (default 0)
  VLLM_TPU_EPLB_INTERVAL      forwards between rebalances (default 128)
  VLLM_TPU_EPLB_FIRST         forward count of the first rebalance (default 32)
  VLLM_TPU_EPLB_DECAY         weight of the previous windows' load in the next
                              plan (default 0.5; 0 = last window only)
  VLLM_TPU_EPLB_MIN_TOKENS    skip a rebalance whose window saw fewer routed
                              tokens than this (default 65536)
  VLLM_TPU_EPLB_MIN_GAIN      move a layer only if its modelled busiest-rank
                              load falls by at least this fraction (default 0.10)
  VLLM_TPU_EPLB_LOG_DIR       write one npz per rebalance (loads, placements)
  VLLM_TPU_EPLB_DEBUG_SKIP    negative control: "weights" moves the map but not
                              the weights, "map" moves the weights but not the
                              map. Output must then differ. Never set in serving.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Module-level on purpose: sharded_jax_callable resolves the string annotations
# (``jax.Array``) of make_exchange_fn's closure against these globals.
import jax
import numpy as np


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    return float(v) if v.strip() else default


def enabled() -> bool:
    return os.environ.get("VLLM_TPU_EPLB", "0").strip().lower() in ("1", "true", "yes")


def _log(msg: str) -> None:
    # stderr: Ray drops worker logger.info lines once serving starts.
    print(f"TPU_EPLB pid={os.getpid()} {msg}", file=sys.stderr, flush=True)


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# Stall diagnostics (VLLM_TPU_EPLB_TRACE=1, default off): compile counters
# read around the rebalance and the forwards right after it.
# --------------------------------------------------------------------------

_JAX_COMPILES = [0, 0.0]  # count, seconds
_JAX_LISTENER = [False]


def _jax_compile_listener(event, duration_secs=0.0, *a, **kw) -> None:
    if "compile" in str(event) and "cache" not in str(event):
        _JAX_COMPILES[0] += 1
        _JAX_COMPILES[1] += float(duration_secs)


def _install_jax_listener() -> None:
    if _JAX_LISTENER[0]:
        return
    try:
        import jax.monitoring

        jax.monitoring.register_event_duration_secs_listener(_jax_compile_listener)
        _JAX_LISTENER[0] = True
    except Exception as e:  # diagnostic only
        _log(f"trace: no jax compile listener ({e!r})")


def _compile_snapshot() -> tuple:
    """(torch_tpu deferred-graph cache misses, jax compiles, jax compile s)."""
    misses = -1
    try:
        from torch_tpu._internal.device import _device_ops_backend

        misses = int(_device_ops_backend._get_cache_misses())
    except Exception:
        pass
    return misses, _JAX_COMPILES[0], _JAX_COMPILES[1]


def _compile_delta(a: tuple, b: tuple) -> str:
    return (f"tt_cache_miss={b[0] - a[0] if a[0] >= 0 else 'na'} "
            f"jax_compiles={b[1] - a[1]} jax_compile_ms={(b[2] - a[2]) * 1e3:.1f}")


# --------------------------------------------------------------------------
# Placement algorithm (pure numpy; identical input -> identical output on
# every rank, which is what lets every rank decide alone).
# --------------------------------------------------------------------------


def balanced_packing(weight: np.ndarray, num_packs: int) -> np.ndarray:
    """Pack items into ``num_packs`` bins of exactly ``n / num_packs`` items.

    Upstream vLLM ``DefaultEplbPolicy.balanced_packing`` for one layer: items
    by descending weight, each to the lightest bin that still has room.
    ``np.argsort(kind="stable")`` and ``np.argmin`` (first minimum) make the
    result a pure function of ``weight``. Returns ``pack[item]``.
    """
    n = weight.shape[0]
    assert n % num_packs == 0
    per = n // num_packs
    order = np.argsort(-weight.astype(np.float64), kind="stable")
    pack = np.full(n, -1, np.int64)
    load = np.zeros(num_packs, np.float64)
    cnt = np.zeros(num_packs, np.int64)
    for item in order:
        p = int(np.argmin(load))
        pack[item] = p
        load[p] += float(weight[item])
        cnt[p] += 1
        if cnt[p] == per:
            load[p] = np.inf
    return pack


def _match_packs_to_ranks(overlap: np.ndarray) -> np.ndarray:
    """rank_of_pack maximizing sum(overlap[pack, rank]); deterministic.

    Hungarian when scipy is importable (it is in the serving image), else a
    greedy largest-overlap-first matching. Either way a pure function of the
    integer overlap matrix.
    """
    n = overlap.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-overlap.astype(np.int64))
        out = np.empty(n, np.int64)
        out[rows] = cols
        return out
    except ImportError:  # pragma: no cover - fallback
        out = np.full(n, -1, np.int64)
        used = np.zeros(n, bool)
        flat = np.argsort(-overlap.reshape(-1), kind="stable")
        for f in flat:
            p, r = divmod(int(f), n)
            if out[p] < 0 and not used[r]:
                out[p] = r
                used[r] = True
        return out


def plan_layer(
    logical_load: np.ndarray, phys2log: np.ndarray, ep: int, min_gain: float
) -> tuple[np.ndarray | None, dict]:
    """New ``phys2log`` (mesh-order slots) for one layer, or None to keep it.

    ``logical_load[e]`` is the window's routed pairs of logical expert e and
    ``phys2log[q]`` the logical expert now in physical slot q (slot q lives on
    mesh index q // g). The new placement keeps every expert whose pack stays
    on its rank on that rank, and orders each rank's experts by their current
    slot, which is the order ``exchange_plan``'s ragged all-to-all delivers.
    """
    E = phys2log.shape[0]
    g = E // ep
    log2phys = np.empty(E, np.int64)
    log2phys[phys2log] = np.arange(E)
    old_rank = log2phys // g
    pack = balanced_packing(logical_load, ep)
    overlap = np.zeros((ep, ep), np.int64)
    np.add.at(overlap, (pack, old_rank), 1)
    rank_of_pack = _match_packs_to_ranks(overlap)
    new_rank = rank_of_pack[pack]  # [E] by logical id
    old_max = float(np.bincount(old_rank, weights=logical_load, minlength=ep).max())
    new_max = float(np.bincount(new_rank, weights=logical_load, minlength=ep).max())
    mean = float(logical_load.sum()) / ep
    info = {
        "old_max_over_mean": old_max / mean if mean > 0 else 1.0,
        "new_max_over_mean": new_max / mean if mean > 0 else 1.0,
        "moved": int((new_rank != old_rank).sum()),
    }
    if mean <= 0 or new_max > old_max * (1.0 - min_gain):
        info["moved"] = 0
        return None, info
    # Order by (new rank, current slot): slots of rank d are filled with the
    # experts arriving at d in ascending current-slot order.
    key = new_rank * E + log2phys
    new_phys2log = np.argsort(key, kind="stable").astype(np.int64)
    return new_phys2log, info


@dataclass
class ExchangePlan:
    """Per-rank operands of the ragged all-to-all that realizes a placement.

    Row r is mesh index r's: ``send_idx[r]`` is the local gather that puts
    its experts in send order (by destination, then current slot);
    ``in_off/send_sz/out_off/recv_sz[r]`` are the ragged_all_to_all operands
    in units of whole experts.
    """

    send_idx: np.ndarray  # [ep, g]
    in_off: np.ndarray  # [ep, ep]
    send_sz: np.ndarray  # [ep, ep]
    out_off: np.ndarray  # [ep, ep]
    recv_sz: np.ndarray  # [ep, ep]
    moved_experts: int = 0
    per_rank_out: np.ndarray = field(default_factory=lambda: np.zeros(0))


def exchange_plan(old_phys2log: np.ndarray, new_phys2log: np.ndarray, ep: int) -> ExchangePlan:
    E = old_phys2log.shape[0]
    g = E // ep
    old_l2p = np.empty(E, np.int64)
    old_l2p[old_phys2log] = np.arange(E)
    new_l2p = np.empty(E, np.int64)
    new_l2p[new_phys2log] = np.arange(E)
    src_rank = old_l2p // g
    dst_rank = new_l2p // g
    # The convention `plan_layer` produces and the collective delivers: within
    # a destination, experts sit in ascending current-slot order.
    for d in range(ep):
        slots = new_phys2log[d * g:(d + 1) * g]
        assert np.all(np.diff(old_l2p[slots]) > 0), "placement not in delivery order"
    cnt = np.zeros((ep, ep), np.int64)  # [src, dst]
    np.add.at(cnt, (src_rank, dst_rank), 1)
    send_idx = np.zeros((ep, g), np.int64)
    for s in range(ep):
        local = old_phys2log[s * g:(s + 1) * g]  # logical ids in local slot order
        order = np.lexsort((np.arange(g), dst_rank[local]))  # by dst, then slot
        send_idx[s] = order
    in_off = np.concatenate([np.zeros((ep, 1), np.int64), np.cumsum(cnt, axis=1)[:, :-1]], 1)
    # out_off[s, d]: where s's chunk lands on d = experts reaching d from sources < s
    out_off = np.concatenate([np.zeros((1, ep), np.int64), np.cumsum(cnt, axis=0)[:-1]], 0)
    recv_sz = cnt.T.copy()  # recv_sz[d, s]
    moved = int((src_rank != dst_rank).sum())
    per_rank_out = cnt.sum(1) - np.diag(cnt)
    return ExchangePlan(
        send_idx=send_idx,
        in_off=in_off,
        send_sz=cnt,
        out_off=out_off,
        recv_sz=recv_sz,
        moved_experts=moved,
        per_rank_out=per_rank_out,
    )


def simulate_exchange(old_local: list, plan: ExchangePlan) -> list:
    """Host model of the collective: per-rank lists of items -> new lists."""
    ep = len(old_local)
    g = len(old_local[0])
    new = [[None] * g for _ in range(ep)]
    for s in range(ep):
        sb = [old_local[s][i] for i in plan.send_idx[s]]
        for d in range(ep):
            n = plan.send_sz[s, d]
            for k in range(n):
                new[d][plan.out_off[s, d] + k] = sb[plan.in_off[s, d] + k]
    return new


def ep_ids_from_mesh(log2phys_mesh: np.ndarray, order: tuple[int, ...] | None, g: int) -> np.ndarray:
    """Logical -> physical id in EP-RANK order, the space the op's relabel reads.

    The op maps an EP-rank-order id ``r * g + j`` to mesh slot
    ``mesh_index_of(r) * g + j`` after the map, so the map must hand it the
    EP-rank-order id of the mesh slot the expert sits in.
    """
    q = np.asarray(log2phys_mesh, np.int64)
    if order is None:
        return q.astype(np.int32)
    order = np.asarray(order, np.int64)
    return (order[q // g] * g + q % g).astype(np.int32)


def initial_phys2log_mesh(E: int, ep: int, order: tuple[int, ...] | None) -> np.ndarray:
    """Load-time placement: mesh index m holds EP rank order[m]'s contiguous block."""
    g = E // ep
    q = np.arange(E)
    if order is None:
        return q.astype(np.int64)
    order = np.asarray(order, np.int64)
    return (order[q // g] * g + q % g).astype(np.int64)


# --------------------------------------------------------------------------
# Serving state
# --------------------------------------------------------------------------

MAP_ATTR = "_tpu_eplb_map"
LOAD_ATTR = "_tpu_eplb_load"
_LAYERS: dict[int, Any] = {}  # decoder layer id -> RoutedExperts layer
_EXCHANGE_OPS: dict = {}


def layer_id_of(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name or "")
    return int(m.group(1)) if m else -1


def register_layer(layer, num_experts: int, device) -> None:
    """Called from process_weights_after_loading for an armed fused-EP layer."""
    import torch

    layer.register_buffer(
        MAP_ATTR, torch.arange(num_experts, dtype=torch.int32, device=device), persistent=False
    )
    layer.register_buffer(
        LOAD_ATTR, torch.zeros(num_experts, dtype=torch.int32, device=device), persistent=False
    )
    lid = layer_id_of(getattr(layer, "layer_name", ""))
    _LAYERS[lid] = layer


def map_operand(layer):
    return getattr(layer, MAP_ATTR, None)


def make_exchange_fn(mesh, axis_name):
    """The JAX function that realizes one ``ExchangePlan`` on one [E, ...] tensor.

    Operands are global arrays sharded on ``axis_name``: the tensor, and the
    plan's five per-rank rows (``send_idx`` [ep, g], the four offset/size
    tables [ep, ep]). Each shard gathers its experts into send order, then
    one ``ragged_all_to_all`` moves every expert that changes rank (an
    expert that stays is a self-send, a local copy) and writes it at its new
    slot.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    spec = P(axis_name)

    def body(w_l, si, io, ss, oo, rs):
        # Move bits, not values: fp8 goes over the collective as int8.
        dt = w_l.dtype
        if jnp.dtype(dt).itemsize == 1 and dt != jnp.int8:
            w_l = lax.bitcast_convert_type(w_l, jnp.int8)
        sb = jnp.take(w_l, si[0], axis=0)
        out = lax.ragged_all_to_all(
            sb, jnp.zeros_like(w_l), io[0], ss[0], oo[0], rs[0], axis_name=axis_name
        )
        return out if out.dtype == dt else lax.bitcast_convert_type(out, dt)

    def exchange(
        w: jax.Array,
        send_idx: jax.Array,
        in_off: jax.Array,
        send_sz: jax.Array,
        out_off: jax.Array,
        recv_sz: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 6, out_specs=spec, check_vma=False
        )(w, send_idx, in_off, send_sz, out_off, recv_sz)

    return exchange


def _exchange_op(mesh, shape, dtype):
    """One torch-callable per (local shape, dtype) of the exchange."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_callable

    key = (tuple(shape), str(dtype))
    op = _EXCHANGE_OPS.get(key)
    if op is not None:
        return op
    spec = P(EP_AXIS_NAME)
    op = sharded_jax_callable(
        f"tpu_eplb::exchange_{len(_EXCHANGE_OPS)}",
        make_exchange_fn(mesh, EP_AXIS_NAME),
        mesh=mesh,
        input_partition_specs=(spec,) * 6,
        output_partition_specs=spec,
    )
    _EXCHANGE_OPS[key] = op
    return op


def _expert_tensors(layer) -> list:
    """Every per-expert tensor the fused op or the megakernel reads, [g, ...]."""
    names = ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv")
    out = []
    for n in names:
        t = getattr(layer, n, None)
        if t is not None:
            out.append((n, t))
    return out


def step_trigger(
    count: int, next_at: int, interval: int, all_busy: bool = True
) -> tuple[bool, int]:
    """Pure trigger rule: rebalance at the first step whose agreed count of
    completed non-megakernel forwards reaches ``next_at``; the next one is due
    ``interval`` forwards later. Every input is a DP-wide agreed value, so every
    rank computes the same answer.

    ``all_busy`` (MIN over ranks of the step's tokens > 0, when the busy gate
    is on) defers a due rebalance to the first step where no DP rank is idle:
    with idle ranks the balance buys nothing and the exchange is pure stall."""
    if count > 0 and count >= next_at and all_busy:
        return True, count + interval
    return False, next_at


class EplbRebalancer:
    """Owns the placement of every registered fused-EP layer of one model."""

    # Diagnostics defaults (overridden in __init__ from the environment).
    trace = False
    busy_gate = False
    trace_fwd_left = 0
    trace_steps_left = 0
    _last_step_t = None
    _step_ring: list = []
    _saver = None

    def __init__(self, model, megakernel=None):
        from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, ep_rank_order

        self.layers = {}
        for m in model.modules():
            if getattr(m, LOAD_ATTR, None) is not None:
                self.layers[layer_id_of(getattr(m, "layer_name", ""))] = m
        self.mesh = build_ep_mesh()
        self.ep = int(self.mesh.devices.size)
        self.me = int(ep_mesh_index())
        self.order = ep_rank_order()
        if self.order is not None and tuple(self.order) == tuple(range(self.ep)):
            self.order = None
        any_layer = next(iter(self.layers.values()))
        self.E = int(getattr(any_layer, MAP_ATTR).shape[0])
        self.g = self.E // self.ep
        self.phys2log = {
            lid: initial_phys2log_mesh(self.E, self.ep, self.order) for lid in self.layers
        }
        self.interval = _env_int("VLLM_TPU_EPLB_INTERVAL", 128)
        self.first = _env_int("VLLM_TPU_EPLB_FIRST", 32)
        self.min_tokens = _env_int("VLLM_TPU_EPLB_MIN_TOKENS", 65536)
        self.min_gain = _env_float("VLLM_TPU_EPLB_MIN_GAIN", 0.10)
        self.decay = _env_float("VLLM_TPU_EPLB_DECAY", 0.5)
        # Logical-expert load the next plan sees: this window plus `decay` x
        # the previous plan's. Kept in LOGICAL space, since placements change.
        self.window = {lid: np.zeros(self.E, np.float64) for lid in self.layers}
        self.log_dir = os.environ.get("VLLM_TPU_EPLB_LOG_DIR", "")
        self.debug_skip = os.environ.get("VLLM_TPU_EPLB_DEBUG_SKIP", "")
        # Diagnostics / stall fixes, all default off.
        self.trace = _flag("VLLM_TPU_EPLB_TRACE")
        self.warm_exchange = _flag("VLLM_TPU_EPLB_WARM_EXCHANGE")
        # On whenever EPLB is: VLLM_TPU_EPLB_BUSY_GATE=0 restores the ungated trigger.
        self.busy_gate = os.environ.get("VLLM_TPU_EPLB_BUSY_GATE", "1").strip().lower() not in (
            "0", "false", "no")
        self.trace_fwd_left = 0
        self.trace_steps_left = 0
        self._last_step_t = None
        self._step_ring: list = []
        self._saver = None
        self.megakernel = megakernel
        self.count = 0
        self.next_at = self.first
        self.armed = False
        self.epoch = 0
        self.topk = 8
        _log(
            f"armed layers={len(self.layers)} E={self.E} ep={self.ep} me={self.me} "
            f"order={'identity' if self.order is None else self.order} interval={self.interval} "
            f"first={self.first} min_tokens={self.min_tokens} min_gain={self.min_gain} decay={self.decay} "
            f"debug_skip={self.debug_skip!r} busy_gate={self.busy_gate} "
            f"warm={self.warm_exchange} trace={self.trace}"
        )

    # Counting --------------------------------------------------------------

    def arm(self) -> None:
        """Start counting: after warmup, at the same step on every rank."""
        for layer in self.layers.values():
            getattr(layer, LOAD_ATTR).zero_()
        self.count = 0
        self.next_at = self.first
        if self.trace:
            _install_jax_listener()
        if self.warm_exchange:
            self._warm_exchange()
        self.armed = True
        _log("counting from here (warmup loads discarded)")

    def _warm_exchange(self) -> None:
        """Compile every exchange program at boot instead of inside the first
        rebalance (~4 s at epoch 1). Runs an all-self-send identity plan whose
        result is discarded: collective, so it is called on every rank at the
        same point (arm(), the end of capture_model)."""
        import torch

        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        layer = next(iter(self.layers.values()))
        ident = initial_phys2log_mesh(self.E, self.ep, self.order)
        xp = exchange_plan(ident, ident, self.ep)
        dev = getattr(layer, MAP_ATTR).device
        ops = [
            torch.tensor(a[self.me:self.me + 1], dtype=torch.int32, device=dev)
            for a in (xp.send_idx, xp.in_off, xp.send_sz, xp.out_off, xp.recv_sz)
        ]
        outs = []
        for _, t in _expert_tensors(layer):
            outs.append(_exchange_op(self.mesh, tuple(t.shape), t.dtype)(t, *ops))
        synchronize_tensors(outs)
        del outs
        t1 = time.perf_counter()
        if self.megakernel is not None and self.debug_skip != "map":
            # Identity placement: the same router columns / expert tensors /
            # smalls it already holds (initial_phys2log_mesh is the kernel's
            # load-time perm), so this only compiles the rebuild's deferred
            # graphs (1.2 s at the first all-layer rebalance in j-c03687df).
            self.megakernel.set_expert_placement(dict(self.phys2log))
        _log(f"warm exchange programs={len(_EXCHANGE_OPS)} ms={(t1 - t0) * 1e3:.1f} "
             f"mk_placement_ms={(time.perf_counter() - t1) * 1e3:.1f}")

    def on_step(
        self, agreed_count: int, min_count: int, n_forwards: int, allreduce_ms: float = 0.0,
        min_tokens: int = 1,
    ) -> None:
        """Called once per engine step on every DP rank, right after the step's
        CPU all-reduce and before any of the step's device work.

        ``agreed_count``/``min_count`` are the MAX/MIN over ranks of
        ``self.count`` riding that all-reduce, and ``n_forwards`` is the number
        of non-megakernel forwards this step runs -- derived from the reduced
        chunk count, so identical on every rank. The trigger therefore depends
        only on globally agreed values, and the busy and the idle (DP dummy)
        paths reach the rebalance at the same point, with no rank-specific
        device work pending in front of it.
        """
        if not self.armed:
            return
        if agreed_count != min_count:
            _log(
                f"WARNING forward counts disagree across ranks: min={min_count} "
                f"max={agreed_count} local={self.count}; following max"
            )
        now = time.perf_counter()
        if self.trace:
            period = (now - self._last_step_t) * 1e3 if self._last_step_t is not None else -1.0
            self._step_ring = (self._step_ring + [(agreed_count, period, allreduce_ms, n_forwards)])[-12:]
            if self.trace_steps_left > 0:
                k = 12 - self.trace_steps_left
                _log(f"trace post_step epoch={self.epoch} k={k} count={agreed_count} "
                     f"n_fwd={n_forwards} period_ms={period:.1f} allreduce_ms={allreduce_ms:.1f}")
                self.trace_steps_left -= 1
        due, self.next_at = step_trigger(
            agreed_count, self.next_at, self.interval,
            all_busy=(min_tokens > 0) or not self.busy_gate,
        )
        if due:
            if self.trace:
                ring = " ".join(f"{c}:{p:.0f}/{a:.0f}/{n}" for c, p, a, n in self._step_ring)
                _log(f"trace pre_steps (count:period_ms/allreduce_ms/n_fwd) {ring}")
            self.rebalance(agreed_count)
            if self.trace:
                self.trace_fwd_left = 3
                self.trace_steps_left = 12
            now = time.perf_counter()
        self._last_step_t = now
        self.count = agreed_count + n_forwards

    def traced_forward(self, fn, mk: bool, n: int):
        """One of the first forwards after a rebalance, timed alone with its
        compile counters (VLLM_TPU_EPLB_TRACE)."""
        from vllm_torchtpu.utils import synchronize_tensors

        k = 3 - self.trace_fwd_left
        self.trace_fwd_left -= 1
        t0 = time.perf_counter()
        synchronize_tensors()
        t1 = time.perf_counter()
        c0 = _compile_snapshot()
        out = fn()
        t2 = time.perf_counter()
        synchronize_tensors()
        t3 = time.perf_counter()
        c1 = _compile_snapshot()
        _log(f"trace post_fwd epoch={self.epoch} k={k} mk={mk} toks={n} "
             f"drain_ms={(t1 - t0) * 1e3:.1f} host_ms={(t2 - t1) * 1e3:.1f} "
             f"total_ms={(t3 - t1) * 1e3:.1f} {_compile_delta(c0, c1)}")
        return out

    # Rebalance -------------------------------------------------------------

    def _read_loads(self) -> dict:
        import torch

        lids = sorted(self.layers)
        host = torch.stack([getattr(self.layers[l], LOAD_ATTR) for l in lids]).cpu().numpy()
        return {l: host[i].astype(np.int64) for i, l in enumerate(lids)}

    def rebalance(self, at: int) -> None:
        import torch

        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        c_start = _compile_snapshot() if self.trace else None
        # Drain everything this rank has queued (the step's input prep on a
        # busy rank, nothing on an idle one) so the exchange programs below are
        # compiled from the same ops on every rank.
        synchronize_tensors()
        t_drain = time.perf_counter() - t0
        _log(f"rebalance enter at_forward={at}")
        phys_loads = self._read_loads()  # mesh-order physical slot -> pairs
        t_read = time.perf_counter() - t0
        tokens = int(next(iter(phys_loads.values())).sum()) // self.topk
        if tokens < self.min_tokens:
            _log(f"skip at forward {at}: window saw {tokens} tokens < {self.min_tokens}")
            return
        plans = {}
        infos = {}
        log_loads = {}
        for lid, pl in phys_loads.items():
            p2l = self.phys2log[lid]
            ll = np.zeros(self.E, np.int64)
            ll[p2l] = pl  # physical slot q holds logical p2l[q]
            log_loads[lid] = ll
            w = self.window[lid] + ll
            new, info = plan_layer(w, p2l, self.ep, self.min_gain)
            self.window[lid] = self.decay * w
            infos[lid] = info
            if new is not None:
                plans[lid] = new
        t_plan = time.perf_counter() - t0 - t_read

        moved_total = 0
        bytes_out_me = 0
        t_x0 = time.perf_counter()
        tm = {"xenq": 0.0, "xsync": 0.0, "map": 0.0, "mk_rebuild": 0.0, "mk_sync": 0.0,
              "zero_sync": 0.0, "save": 0.0}
        for lid, new in plans.items():
            layer = self.layers[lid]
            xp = exchange_plan(self.phys2log[lid], new, self.ep)
            moved_total += xp.moved_experts
            ta = time.perf_counter()
            if self.debug_skip != "weights":
                dev = getattr(layer, MAP_ATTR).device
                ops = [
                    torch.tensor(a[self.me:self.me + 1], dtype=torch.int32, device=dev)
                    for a in (xp.send_idx, xp.in_off, xp.send_sz, xp.out_off, xp.recv_sz)
                ]
                for name, t in _expert_tensors(layer):
                    op = _exchange_op(self.mesh, tuple(t.shape), t.dtype)
                    new_t = op(t, *ops)
                    with torch.no_grad():
                        t.copy_(new_t)
                    del new_t
                    per_expert = t[0].numel() * t.element_size()
                    bytes_out_me += int(xp.per_rank_out[self.me]) * per_expert
                tb = time.perf_counter()
                tm["xenq"] += tb - ta
                synchronize_tensors([t for _, t in _expert_tensors(layer)])
                tm["xsync"] += time.perf_counter() - tb
            tc = time.perf_counter()
            if self.debug_skip != "map":
                l2p = np.empty(self.E, np.int64)
                l2p[new] = np.arange(self.E)
                ep_ids = ep_ids_from_mesh(l2p, self.order, self.g)
                getattr(layer, MAP_ATTR).copy_(
                    torch.tensor(ep_ids, dtype=torch.int32, device=getattr(layer, MAP_ATTR).device)
                )
            tm["map"] += time.perf_counter() - tc
            old = self.phys2log[lid]
            self.phys2log[lid] = new
            infos[lid]["old_phys2log"] = old
        if plans and self.megakernel is not None and self.debug_skip != "map":
            self.megakernel.set_expert_placement(
                {lid: self.phys2log[lid] for lid in plans}
            )
            mk_t = getattr(self.megakernel, "last_placement_timing", None)
            if mk_t:
                tm["mk_rebuild"], tm["mk_sync"] = mk_t
        td = time.perf_counter()
        for layer in self.layers.values():
            getattr(layer, LOAD_ATTR).zero_()
        synchronize_tensors()
        tm["zero_sync"] = time.perf_counter() - td
        t_x = time.perf_counter() - t_x0
        self.epoch += 1
        om = np.array([infos[l]["old_max_over_mean"] for l in sorted(infos)])
        nm = np.array(
            [infos[l]["new_max_over_mean"] if l in plans else infos[l]["old_max_over_mean"]
             for l in sorted(infos)]
        )
        _log(
            f"rebalance epoch={self.epoch} at_forward={at} tokens={tokens} "
            f"layers_moved={len(plans)}/{len(infos)} experts_moved={moved_total} "
            f"window max/mean median {np.median(om):.3f} -> {np.median(nm):.3f} "
            f"(mean {om.mean():.3f} -> {nm.mean():.3f}) bytes_out_this_rank={bytes_out_me} "
            f"read_ms={t_read * 1e3:.1f} plan_ms={t_plan * 1e3:.1f} exchange_ms={t_x * 1e3:.1f}"
        )
        te = time.perf_counter()
        if self.log_dir:
            lids = sorted(infos)
            path = os.path.join(self.log_dir, f"eplb_me{self.me}_pid{os.getpid()}_e{self.epoch}.npz")
            self._save_npz(
                path,
                epoch=self.epoch,
                at_forward=at,
                t=time.time(),
                tokens=tokens,
                layers=np.array(lids, np.int32),
                logical_load=np.stack([log_loads[l] for l in lids]),
                phys_load=np.stack([phys_loads[l] for l in lids]),
                phys2log=np.stack([self.phys2log[l] for l in lids]),
                moved_layers=np.array(sorted(plans), np.int32),
                old_max_over_mean=om,
                new_max_over_mean=nm,
                exchange_ms=t_x * 1e3,
                bytes_out_this_rank=bytes_out_me,
                experts_moved=moved_total,
                drain_ms=t_drain * 1e3,
                **{f"{k}_ms": v * 1e3 for k, v in tm.items() if k != "save"},
            )
        tm["save"] = time.perf_counter() - te
        if self.trace:
            _log(f"trace rebalance epoch={self.epoch} layers_moved={len(plans)} "
                 f"drain_ms={t_drain * 1e3:.1f} read_ms={(t_read - t_drain) * 1e3:.1f} "
                 f"plan_ms={t_plan * 1e3:.1f} "
                 + " ".join(f"{k}_ms={v * 1e3:.1f}" for k, v in tm.items())
                 + f" total_ms={(time.perf_counter() - t0) * 1e3:.1f} "
                 + _compile_delta(c_start, _compile_snapshot()))

    def _save_npz(self, path: str, **arrays) -> None:
        """Write the epoch record off the serving thread (VLLM_TPU_EPLB_ASYNC_SAVE,
        default off: then it is written inline, as before)."""
        def write():
            t = time.perf_counter()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, **arrays)
            if self.trace:
                _log(f"trace npz_written ms={(time.perf_counter() - t) * 1e3:.1f} path={path}")

        if not _flag("VLLM_TPU_EPLB_ASYNC_SAVE"):
            write()
            return
        if self._saver is None:
            from concurrent.futures import ThreadPoolExecutor

            self._saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tpu_eplb_save")
        self._saver.submit(write)


_REBALANCER: EplbRebalancer | None = None


def create(model, megakernel=None) -> EplbRebalancer | None:
    global _REBALANCER
    if not enabled() or not any(
        getattr(m, LOAD_ATTR, None) is not None for m in model.modules()
    ):
        return None
    _REBALANCER = EplbRebalancer(model, megakernel)
    return _REBALANCER


def current() -> EplbRebalancer | None:
    return _REBALANCER
