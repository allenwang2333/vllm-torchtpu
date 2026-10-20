# SPDX-License-Identifier: Apache-2.0
"""Opt-in serving tap: MoE expert ids and CSA top-k indices per steady decode
step, plus each worker's NUMA placement. Measurement scaffolding only.

Installed by this directory's sitecustomize.py when DSV41_ROUTING_TAP_OUT is
set; nothing under src/ changes. Three post-import patches:

* VllmDeepseekV4Mxfp4MoEMethod: at weight load, a fused-EP layer gets an
  int32 ring [MOE_RING + 1, rows * topk]; every forward whose router logits
  are exactly [rows, experts] (the steady verify / draft shape) writes the
  layer's global top-k expert ids into ring row counter % MOE_RING and bumps
  the counter in [MOE_RING, 0]. The selection is recomputed from the same
  logits and staged score bias the kernel gets (scoring fn, + bias, top-k;
  lax.top_k, so only exact f32 ties may differ from the kernel's selector).
* DeepseekV41Attention: a layer that owns an indexer gets an int32 ring of
  TOPK_RING steps x [288, 512], filled once after each host reset
  (benchmark_begin, profile arm); every `_select_paged_indices` /
  `_select_indices` result of exactly that shape is written the same way.
* server_step_worker.ServerStepWorker: benchmark_begin snapshots counters
  and NUMA placement; benchmark_end and the profile stop dump every
  ring (with the counters that bracket the window) to
  $DSV41_ROUTING_TAP_OUT/rank-<r>-<label>.npz. The synchronized profile is
  dumped at the step where it stops, with its start and stop counters.

The in-graph writes follow the torch_tpu bridge rules: no host callback
(jax.export cannot serialize one), every jax_op is built at weight load and
never inside the traced forward, the jax functions annotate tensors as
jax.Array (no postponed annotations in this module), and the torch side has
no shape-dependent branch -- shape selection happens inside the jax function,
which is traced per concrete shape. Rings are donated and written back with
copy_, the way the KV cache is.
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time

import numpy as np

OUT_ENV = "DSV41_ROUTING_TAP_OUT"
MOE_RING = int(os.environ.get("DSV41_ROUTING_TAP_RING", "192"))
TOPK_RING = int(os.environ.get("DSV41_ROUTING_TAP_TOPK_RING", "8"))
TOPK_ROWS = int(os.environ.get("DSV41_ROUTING_TAP_TOPK_ROWS", "288"))
TOPK_WIDTH = int(os.environ.get("DSV41_ROUTING_TAP_TOPK_WIDTH", "512"))
# global experts -> rows of the steady decode forward recorded for that layer
# (target verify: 48 requests x 6; DSpark draft: 48 x 5).
MOE_ROWS = {
    int(e): int(r)
    for e, r in (
        kv.split(":")
        for kv in os.environ.get("DSV41_ROUTING_TAP_ROWS", "384:288,128:240").split(",")
    )
}

MOE_ATTR = "_dsv41_tap_moe"
TOPK_ATTR = "_dsv41_tap_topk"
# name -> (ring tensor, meta dict). Filled at weight load, read by dumps.
MOE_RINGS: dict = {}
TOPK_RINGS: dict = {}
_OPS: dict = {}


# --------------------------------------------------------------------------- #
# The in-graph recorders (pure JAX; unit-tested on CPU in test_routing_tap.py).
# --------------------------------------------------------------------------- #
def moe_record_fn(*, rows, experts, topk, scoring_fn, has_bias, ring=None):
    """(ring, logits[, bias]) -> ring with this call's top-k ids recorded."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    ring = MOE_RING if ring is None else ring
    width = rows * topk

    def record(buf, logits, bias):
        if tuple(logits.shape) != (rows, experts):
            return buf
        scores = logits.astype(jnp.float32)
        if scoring_fn == "softmax":
            scores = jax.nn.softmax(scores, axis=-1)
        elif scoring_fn == "sigmoid":
            scores = jax.nn.sigmoid(scores)
        elif scoring_fn == "sqrtsoftplus":
            scores = jnp.sqrt(jax.nn.softplus(scores))
        else:
            raise NotImplementedError(scoring_fn)
        if bias is not None:
            scores = scores + bias.astype(jnp.float32)[None, :]
        _, ids = lax.top_k(scores, topk)
        ctr = buf[ring, 0]
        buf = lax.dynamic_update_slice(
            buf, ids.reshape(1, width).astype(jnp.int32), (lax.rem(ctr, ring), 0)
        )
        return buf.at[ring, 0].add(1)

    if has_bias:

        def fn(buf: jax.Array, logits: jax.Array, bias: jax.Array) -> jax.Array:
            return record(buf, logits, bias)

    else:

        def fn(buf: jax.Array, logits: jax.Array) -> jax.Array:
            return record(buf, logits, None)

    return fn


def topk_record_fn(*, rows=None, width=None, ring=None):
    """(ring, selected) -> ring with one [rows, width] selection recorded.

    Not a wrap-around ring: slots fill once, from the last host reset, so a dump
    holds the FIRST `ring` steady steps of a window (the last ones of a wave
    carry finished requests). The counter keeps counting past `ring`.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    rows = TOPK_ROWS if rows is None else rows
    width = TOPK_WIDTH if width is None else width
    ring = TOPK_RING if ring is None else ring

    def fn(buf: jax.Array, selected: jax.Array) -> jax.Array:
        if tuple(selected.shape) != (rows, width):
            return buf
        ctr = buf[ring * rows, 0]
        slot = jnp.minimum(ctr, ring - 1) * rows
        keep = lax.dynamic_slice(buf, (slot, 0), (rows, width))
        new = jnp.where(ctr < ring, selected.astype(jnp.int32), keep)
        buf = lax.dynamic_update_slice(buf, new, (slot, 0))
        return buf.at[ring * rows, 0].add(1)

    return fn


def _jax_op(name, fn):
    import torch
    from torch_tpu._internal import pallas

    if name in _OPS:
        return _OPS[name]
    op = pallas.jax_op(name, fn, donate_argnums=(0,))

    def fake(buf, *args):
        return torch.empty_like(buf)

    op.register_fake(fake)
    _OPS[name] = op
    return op


# --------------------------------------------------------------------------- #
# Patches.
# --------------------------------------------------------------------------- #
def _patch_mxfp4(module) -> None:
    import torch

    cls = module.VllmDeepseekV4Mxfp4MoEMethod
    load = cls.process_weights_after_loading
    forward = cls._forward_monolithic_tpu

    def process_weights_after_loading(self, layer):
        load(self, layer)
        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            FUSED_MOE_EP_OP_ATTR,
            _routing_config,
            score_bias_operand,
        )

        if getattr(self, FUSED_MOE_EP_OP_ATTR, None) is None:
            return
        experts = int(layer.global_num_experts)
        rows = MOE_ROWS.get(experts)
        if rows is None:
            return
        topk = int(layer.moe_config.experts_per_token)
        scoring_fn, _, _ = _routing_config(layer)
        has_bias = score_bias_operand(layer) is not None
        name = getattr(layer, "layer_name", "") or f"moe@{id(layer):x}"
        op = _jax_op(
            f"dsv41_tap::moe_e{experts}_k{topk}_r{rows}_{scoring_fn}_{int(has_bias)}",
            moe_record_fn(
                rows=rows,
                experts=experts,
                topk=topk,
                scoring_fn=scoring_fn,
                has_bias=has_bias,
            ),
        )
        ring = torch.full(
            (MOE_RING + 1, rows * topk),
            -1,
            dtype=torch.int32,
            device=layer.w13_weight.device,
        )
        ring[MOE_RING].zero_()
        layer.register_buffer(MOE_ATTR, ring, persistent=False)
        layer._dsv41_tap_moe_op = op
        MOE_RINGS[name] = (
            layer,
            dict(
                rows=rows,
                experts=experts,
                topk=topk,
                scoring_fn=scoring_fn,
                has_bias=has_bias,
                ring=MOE_RING,
            ),
        )
        print(
            f"DSV41_ROUTING_TAP moe {name} experts={experts} topk={topk} rows={rows}",
            flush=True,
        )

    def _forward_monolithic_tpu(self, layer, x, router_logits, *args, **kwargs):
        ring = getattr(layer, MOE_ATTR, None)
        if ring is not None:
            from vllm_torchtpu.layers.adapter.fused_moe_ep import score_bias_operand

            op = layer._dsv41_tap_moe_op
            bias = score_bias_operand(layer)
            new = (
                op(ring, router_logits)
                if bias is None
                else op(ring, router_logits, bias)
            )
            ring.copy_(new)
        return forward(self, layer, x, router_logits, *args, **kwargs)

    cls.process_weights_after_loading = process_weights_after_loading
    cls._forward_monolithic_tpu = _forward_monolithic_tpu


def _patch_attention(module) -> None:
    import torch

    cls = module.DeepseekV41Attention
    load = cls.process_weights_after_loading

    def process_weights_after_loading(self, act_dtype):
        load(self, act_dtype)
        if getattr(self, "indexer", None) is None:
            return
        op = _jax_op(
            f"dsv41_tap::csa_topk_r{TOPK_ROWS}_w{TOPK_WIDTH}", topk_record_fn()
        )
        ring = torch.full(
            (TOPK_RING * TOPK_ROWS + 1, TOPK_WIDTH),
            -1,
            dtype=torch.int32,
            device=self.wo_a.weight.device,
        )
        ring[TOPK_RING * TOPK_ROWS].zero_()
        self.register_buffer(TOPK_ATTR, ring, persistent=False)
        self._dsv41_tap_topk_op = op
        name = getattr(self, "prefix", "") or f"attn@{id(self):x}"
        TOPK_RINGS[name] = (
            self,
            dict(rows=TOPK_ROWS, width=TOPK_WIDTH, ring=TOPK_RING),
        )
        print(f"DSV41_ROUTING_TAP csa {name}", flush=True)

    def tapped(original):
        def select(self, *args, **kwargs):
            selected = original(self, *args, **kwargs)
            ring = getattr(self, TOPK_ATTR, None)
            if ring is not None:
                ring.copy_(self._dsv41_tap_topk_op(ring, selected))
            return selected

        select.__wrapped__ = original
        return select

    cls.process_weights_after_loading = process_weights_after_loading
    cls._select_paged_indices = tapped(cls._select_paged_indices)
    cls._select_indices = tapped(cls._select_indices)


# --------------------------------------------------------------------------- #
# Host side: counters, dumps, NUMA.
# --------------------------------------------------------------------------- #
def _counter(name, kind):
    owner, meta = (MOE_RINGS if kind == "moe" else TOPK_RINGS)[name]
    buf = getattr(owner, MOE_ATTR if kind == "moe" else TOPK_ATTR).cpu()
    row = meta["ring"] if kind == "moe" else meta["ring"] * meta["rows"]
    return int(buf[row, 0])


def reset_topk() -> None:
    """Refill every CSA ring from the host (outside any traced forward)."""
    import torch

    for owner, m in TOPK_RINGS.values():
        ring = getattr(owner, TOPK_ATTR)
        fresh = torch.full(tuple(ring.shape), -1, dtype=torch.int32)
        fresh[m["ring"] * m["rows"]] = 0
        ring.copy_(fresh.to(ring.device))


def counters() -> dict:
    return {
        "moe": {n: _counter(n, "moe") for n in MOE_RINGS},
        "csa": {n: _counter(n, "csa") for n in TOPK_RINGS},
    }


def dump(rank: int, label: str, extra: dict) -> str:
    import numpy as np

    out = os.environ[OUT_ENV]
    os.makedirs(out, exist_ok=True)
    arrays, meta = (
        {},
        {
            "rank": rank,
            "label": label,
            "wall_ns": time.time_ns(),
            "moe": {},
            "csa": {},
            **extra,
        },
    )
    for i, (name, (owner, m)) in enumerate(sorted(MOE_RINGS.items())):
        arrays[f"moe_{i}"] = getattr(owner, MOE_ATTR).cpu().numpy()
        meta["moe"][name] = {**m, "key": f"moe_{i}"}
    for i, (name, (owner, m)) in enumerate(sorted(TOPK_RINGS.items())):
        arrays[f"csa_{i}"] = getattr(owner, TOPK_ATTR).cpu().numpy()
        meta["csa"][name] = {**m, "key": f"csa_{i}"}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    path = os.path.join(out, f"rank-{rank}-{label}.npz")
    np.savez_compressed(path, **arrays)
    print(f"DSV41_ROUTING_TAP dumped {path}", flush=True)
    return path


def _vmas():
    """[(start, end, numa_maps fields)] for this process."""
    spans = {}
    with open("/proc/self/maps") as f:
        for line in f:
            a, b = line.split()[0].split("-")
            spans[int(a, 16)] = int(b, 16)
    out = []
    with open("/proc/self/numa_maps") as f:
        for line in f:
            parts = line.split()
            start = int(parts[0], 16)
            fields = {"policy": parts[1]}
            for p in parts[2:]:
                k, _, v = p.partition("=")
                fields[k] = v if v else True
            out.append((start, spans.get(start, start), fields))
    return out


def page_nodes(ptr: int, nbytes: int, samples: int = 2048) -> dict:
    """NUMA node of up to `samples` evenly spaced pages of [ptr, ptr + nbytes).

    numa_maps is per VMA, and the kernel merges adjacent anonymous mappings, so
    a staging buffer shares its VMA with the tables around it; move_pages(2)
    with no target nodes reports where each page actually is.
    """
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    page = os.sysconf("SC_PAGE_SIZE")
    first, last = ptr // page, (ptr + nbytes - 1) // page
    idx = np.unique(
        np.linspace(first, last, min(samples, last - first + 1)).astype(np.int64)
    )
    n = len(idx)
    pages = (ctypes.c_void_p * n)(*[int(i) * page for i in idx])
    status = (ctypes.c_int * n)()
    sys_move_pages = 279  # x86_64
    rc = libc.syscall(sys_move_pages, 0, ctypes.c_ulong(n), pages, None, status, 0)
    if rc != 0:
        return {"error": f"move_pages errno {ctypes.get_errno()}"}
    out = {}
    for v in status:
        key = f"N{v}" if v >= 0 else f"err{-v}"
        out[key] = out.get(key, 0) + 1
    return out


def numa_snapshot(worker) -> dict:
    """Placement of every large host tensor the model holds, and of this process."""
    import torch

    model = worker.model_runner.model
    seen, tensors = set(), []
    for mname, mod in model.named_modules():
        for attr, value in (
            list(vars(mod).items())
            + list(mod._buffers.items())
            + list(mod._parameters.items())
        ):
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                continue
            try:
                ptr = value.data_ptr()
                nbytes = value.untyped_storage().nbytes()
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                tensors.append({"name": f"{mname}.{attr}", "error": repr(exc)})
                continue
            if nbytes < (1 << 20) or ptr in seen:
                continue
            seen.add(ptr)
            tensors.append({"name": f"{mname}.{attr}", "ptr": ptr, "bytes": nbytes})
    vmas = _vmas()
    for t in tensors:
        if "ptr" not in t:
            continue
        lo, hi = t["ptr"], t["ptr"] + t["bytes"]
        t["vmas"] = [
            {"start": hex(s), "end": hex(e), **f}
            for s, e, f in vmas
            if s < hi and e > lo
        ]
        nodes = {}
        for v in t["vmas"]:
            for k, val in v.items():
                if k.startswith("N") and k[1:].isdigit():
                    nodes[k] = nodes.get(k, 0) + int(val)
        t["vma_pages_by_node"] = nodes
        t["sampled_page_nodes"] = page_nodes(t["ptr"], t["bytes"])
    status = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("Cpus_allowed_list", "Mems_allowed_list", "Threads", "VmRSS"):
                status[k] = v.strip()
    topo = {}
    for node in sorted(os.listdir("/sys/devices/system/node")):
        if node.startswith("node") and node[4:].isdigit():
            with open(f"/sys/devices/system/node/{node}/cpulist") as f:
                topo[node] = f.read().strip()
    accel = {}
    for root in ("/sys/class/accel", "/sys/class/vfio-dev"):
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                p = os.path.join(root, d, "device", "numa_node")
                if os.path.exists(p):
                    with open(p) as f:
                        accel[f"{root}/{d}"] = f.read().strip()
    return {
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "status": status,
        "nodes": topo,
        "accel_numa_node": accel,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("TPU_", "TORCH_TPU_DP", "CLOUD_TPU"))
            or k in ("OMP_NUM_THREADS",)
        },
        "tensors": tensors,
    }


def _patch_worker(module) -> None:
    cls = module.ServerStepWorker
    begin, end = cls.benchmark_begin, cls.benchmark_end
    arm = cls.synchronized_profile_arm

    def rank(self):
        return int(self.vllm_config.parallel_config.data_parallel_rank)

    def benchmark_begin(self, scenario):
        n = getattr(self, "_dsv41_tap_rounds", 0)
        self._dsv41_tap_rounds = n + 1
        reset_topk()
        self._dsv41_tap_begin = counters()
        try:
            snap = numa_snapshot(self)
        except Exception as exc:  # noqa: BLE001 - recorded in the dump, not swallowed
            snap = {"error": repr(exc)}
        self._dsv41_tap_numa = snap
        out = os.environ[OUT_ENV]
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"rank-{rank(self)}-numa-round{n}.json"), "w") as f:
            json.dump(snap, f, indent=1)
        return begin(self, scenario)

    def benchmark_end(self):
        result = end(self)
        n = self._dsv41_tap_rounds - 1
        dump(
            rank(self),
            f"round{n}",
            {"counters_begin": self._dsv41_tap_begin, "counters_end": counters()},
        )
        return result

    def synchronized_profile_arm(self, output_dir):
        reset_topk()
        result = arm(self, output_dir)
        runner = self.model_runner
        boundary = runner._dp_coordinated_step
        state = self._sync_capture
        marks = self._dsv41_tap_profile = {}

        def tapped_boundary(*args, **kwargs):
            was_active, was_done = state["active"], state["completed"]
            out = boundary(*args, **kwargs)
            if state["active"] and not was_active:
                marks["start"] = counters()
            if state["completed"] and not was_done:
                # boundary has restored the runner's own step on completion.
                # Dump now: the rest of the wave would overwrite the window.
                marks["stop"] = counters()
                dump(
                    rank(self),
                    "profile",
                    {"profile_counters": marks, "counters_end": marks["stop"]},
                )
            return out

        runner._dp_coordinated_step = tapped_boundary
        return result

    cls.benchmark_begin = benchmark_begin
    cls.benchmark_end = benchmark_end
    cls.synchronized_profile_arm = synchronized_profile_arm


# --------------------------------------------------------------------------- #
# Post-import hooks.
# --------------------------------------------------------------------------- #
PATCHES = {
    "vllm_torchtpu.layers.adapter.quantization.mxfp4": _patch_mxfp4,
    "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention": _patch_attention,
    "server_step_worker": _patch_worker,
}


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self._inner, self._patch = inner, patch

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._patch(module)
        print(
            f"DSV41_ROUTING_TAP patched {module.__name__} (pid {os.getpid()})",
            flush=True,
        )


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = PATCHES.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, patch)
        return spec


def install() -> None:
    for name, patch in PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
