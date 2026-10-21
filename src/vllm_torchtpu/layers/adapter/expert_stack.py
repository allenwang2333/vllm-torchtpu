# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Layer-stacked routed-expert storage built while the checkpoint loads (L3-c experts, load-time form).

The GLM-5.2 decode megakernel reads every MoE layer's routed experts through one operand per kind
([n_layers, E, ...]), and the stock moe_tp_decode op reads its own row of the same tensors in place
(`moe_tp_decode.stack_experts`). Stacking after `load_model` (torch.stack per kind) needs the stack and the
per-layer tensors at once: in the served config that program could not reserve its 28.12 GiB output on any
rank (j-e2afec10). Here the stacks are allocated right after the model is constructed, before any expert byte
reaches the device, and nothing per layer ever holds expert HBM:

- load: the expert staging patch (model_loader_patches.ExpertParamStager) uploads each completed w13 / w2
  parameter into its row of the stack (`write_raw`), in <= 256 MiB slices through a donated
  dynamic_update_slice, instead of into a per-layer device tensor. The checkpoint layout [E, N, K] has the
  served row's bytes ([E, K, N]), so the row holds the checkpoint bytes until the layer is processed.
  The per-layer parameter stays the never-materialized tensor create_weights made: a loader target only.
- process_weights_after_loading (fp8.VllmFp8MoEMethodTPU): the layer's experts are re-quantized chunk by
  chunk as before, each chunk read back from its row (`read_raw`) and written over the same bytes
  (`write_rows`) once requantized; the per-channel scales go to their own stacks (`write_scale`). The
  per-layer parameters become placeholders of the served shapes (never materialized).
- moe_tp_decode.arm: the armed layers adopt the stacks (buffers + row index + the stacked op) and the
  placeholders are dropped, exactly the state `stack_experts` leaves.

So the load holds the stacks plus at most one requantization chunk, which is the per-layer path's own
working set; bytes_in_use after load equals the per-layer form's.

Planned only when it can complete: real weights through the staging patch (DefaultModelLoader), the
megakernel on with stacked experts (GLM52_DECODE_MK=1, GLM52_DECODE_MK_STACK_E unset or "1"), and every
static condition moe_tp_decode's arming needs (TP MoE over the whole world, no EP / DP / PCP, the TP decode
kernel on, FP8 block-quantized checkpoint requantized per channel, ...). Anything else, or a failed stack
allocation, leaves the per-layer path untouched (logged). Once rows hold data there is no per-layer copy to
fall back to, so a layer that is planned but not armed fails the load with the reason.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

KINDS = ("w13", "w2", "s13", "s2")
_PARAM_KIND = {"w13_weight": "w13", "w2_weight": "w2"}
_UPLOAD_CHUNK_BYTES = 256 << 20

_PLAN: LoadStacks | None = None
_REFUSED: str | None = None
_OPS: dict[str, Any] = {}


def hbm_stats() -> dict:
    """bytes_in_use / peak / largest free block of this process's first device (GiB-free ints), or {}."""
    try:
        import jax

        s = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - a report, never a gate
        return {}
    keys = ("bytes_in_use", "peak_bytes_in_use", "largest_free_block_bytes", "bytes_limit")
    return {k: int(s[k]) for k in keys if k in s}


def current() -> LoadStacks | None:
    return _PLAN


def refused() -> str | None:
    return _REFUSED


def row_of(layer: Any) -> int | None:
    """This RoutedExperts module's row in the load-time stacks, or None (per-layer form)."""
    return None if _PLAN is None else _PLAN.row.get(id(layer))


def owns(param: Any) -> bool:
    return _PLAN is not None and id(param) in _PLAN.params


def wanted() -> str | None:
    """None if the environment asks for load-time stacks, else why not."""
    if os.environ.get("GLM52_DECODE_MK", "0") != "1":
        return "GLM52_DECODE_MK is not 1"
    form = os.environ.get("GLM52_DECODE_MK_STACK_E", "1")
    if form != "1":
        return f"GLM52_DECODE_MK_STACK_E={form!r} (load-time stacks need unset or '1')"
    return None


def _layer_index(layer: Any) -> int | None:
    m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")) + ".")
    return int(m.group(1)) if m else None


def _refusal(model: Any, vllm_config: Any, loader: Any) -> str | list:
    """The target MoE layers in row order, or why the load-time stacks cannot be built and armed."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.layers.adapter import moe_routing
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
    from vllm_torchtpu.layers.core import fused_moe_gmm

    if not isinstance(loader, DefaultModelLoader):
        return f"loader {type(loader).__name__} (rows are written by the staged expert upload of DefaultModelLoader)"
    if not getattr(RoutedExperts, "_tpu_expert_staging_patch", False):
        return "the expert staging patch is not applied"
    if not envs.USE_MOE_TP_DECODE_KERNEL:
        return "USE_MOE_TP_DECODE_KERNEL is off (the stacked stock op is moe_tp_decode's)"
    if not envs.TPU_MOE_SKIP_PADDED_TOKENS:
        return "TPU_MOE_SKIP_PADDED_TOKENS is off (moe_tp_decode would not arm)"
    if envs.FUSED_MOE_DECODE_KERNEL or envs.TPU_MOE_FP8_ALL_GATHER or envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0:
        return "FUSED_MOE_DECODE_KERNEL / TPU_MOE_FP8_ALL_GATHER / TPU_MOE_COLLECTION_CHUNK_SIZE (moe_tp_decode would not arm)"
    if envs.USE_MOE_FUSED_EP_KERNEL or envs.MOE_FUSED_EP_KERNEL_IMPL == "adaptive":
        return "a fused EP MoE kernel is on"
    if fused_moe_gmm._DENSE_EXPERT_THRESHOLD > 0:
        return "TPU_MOE_DENSE_EXPERT_THRESHOLD > 0 (the dense-expert path takes per-layer weights)"
    if moe_routing._SIMULATION_STRATEGY is not None or moe_routing._ROUTER_TOPK != "rowmax":
        return "routing simulation / TPU_MOE_ROUTER_TOPK != rowmax (moe_tp_decode would not arm)"
    if envs.MOE_REQUANTIZE_BLOCK_SIZE is not None or envs.MOE_REQUANTIZE_WEIGHT_DTYPE != "float8_e4m3fn":
        return "MoE requantization is not per-channel float8_e4m3fn"
    pc = vllm_config.parallel_config
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if bool(getattr(pc, "enable_expert_parallel", False)):
        return "expert parallel is on"
    if (int(pc.tensor_parallel_size) != world or world < 2 or int(pc.pipeline_parallel_size) != 1
            or int(getattr(pc, "data_parallel_size", 1)) != 1
            or int(getattr(pc, "prefill_context_parallel_size", 1)) != 1):
        return f"needs TP over the whole world (tp {pc.tensor_parallel_size}, world {world}), pp 1, dp 1, pcp 1"
    hf = vllm_config.model_config.hf_text_config
    n_layers = int(getattr(hf, "num_hidden_layers", 0))
    first_dense = int(getattr(hf, "first_k_dense_replace", 0) or 0)
    found = []
    for m in model.modules():
        if not isinstance(m, RoutedExperts):
            continue
        idx = _layer_index(m)
        if idx is None or idx >= n_layers:
            continue                                    # the MTP drafter's layer (index num_hidden_layers)
        found.append((idx, m))
    found.sort(key=lambda t: t[0])
    want = list(range(first_dense, n_layers))
    if [i for i, _ in found] != want:
        return f"MoE layers {[i for i, _ in found][:8]}... are not layers {first_dense}..{n_layers - 1}"
    layers = [m for _, m in found]
    l0 = layers[0]
    for i, layer in zip(want, layers):
        qm = layer.quant_method
        if type(qm) is not VllmFp8MoEMethodTPU:
            return f"layer {i}: quant method {type(qm).__name__}"
        if not qm.quant_config.is_checkpoint_fp8_serialized or getattr(qm, "weight_block_size", None) is None:
            return f"layer {i}: not a block-quantized FP8 checkpoint"
        if getattr(qm.moe, "has_bias", False):
            return f"layer {i}: biased experts"
        mc = layer.moe_config
        unp = getattr(mc, "intermediate_size_per_partition_unpadded", None)
        if unp is not None and unp != mc.intermediate_size_per_partition:
            return f"layer {i}: padded intermediate size"
        par = mc.moe_parallel_config
        if par.use_ep or par.tp_size != world or par.dp_size != 1 or par.pcp_size != 1 or par.enable_eplb:
            return f"layer {i}: MoE parallel config (needs TP over {world})"
        act = qm._resolve_tpu_activation(layer)
        if act != "silu":
            return f"layer {i}: activation {act!r} (the stacked requantization is the silu path)"
        for name in ("w13_weight", "w2_weight"):
            t, t0 = getattr(layer, name, None), getattr(l0, name)
            if t is None or tuple(t.shape) != tuple(t0.shape) or t.dtype != t0.dtype:
                return f"layer {i} {name}: shape / dtype differ from layer {want[0]}"
            if t.dtype != torch.float8_e4m3fn or t.device.type != "tpu":
                return f"layer {i} {name}: {t.dtype} on {t.device.type} (needs float8_e4m3fn on tpu)"
    return layers


def maybe_plan(model: Any, vllm_config: Any, loader: Any) -> None:
    """Allocate the stacks for `model` right after construction, if every condition holds. Once per process: a
    later model (the MTP drafter) keeps the per-layer path."""
    global _PLAN, _REFUSED
    if _PLAN is not None:
        return
    why = wanted()
    if why is not None:
        return
    res = _refusal(model, vllm_config, loader)
    if isinstance(res, str):
        _REFUSED = res
        logger.warning("expert_stack: no load-time expert stacks (%s); experts load per layer", res)
        return
    plan = LoadStacks(res)
    try:
        plan.allocate()
    except Exception as e:  # noqa: BLE001 - per-layer path stays intact: no row has data yet
        plan.release()
        _REFUSED = f"allocation failed ({type(e).__name__}: {str(e)[:300]})"
        logger.warning("expert_stack: no load-time expert stacks (%s); experts load per layer", _REFUSED)
        return
    _PLAN = plan


def _put_fn(stk, x, idx):
    from jax import lax
    import jax.numpy as jnp

    starts = (idx[0], idx[1]) + (jnp.int32(0),) * (stk.ndim - 2)
    return lax.dynamic_update_slice(stk, x[None], starts)


def _get_fn(stk, idx, n):
    from jax import lax
    import jax.numpy as jnp

    starts = (idx[0], idx[1]) + (jnp.int32(0),) * (stk.ndim - 2)
    s = lax.dynamic_slice(stk, starts, (1, n) + tuple(stk.shape[2:]))
    # the checkpoint layout [n, N, K] of a served [n, K, N] row: the same bytes, row-major
    return s[0].reshape((n,) + tuple(stk.shape[2:-2]) + (stk.shape[-1], stk.shape[-2]))


def _ops():
    if not _OPS:
        import jax
        from torch_tpu._internal import pallas

        def put(stk: jax.Array, x: jax.Array, idx: jax.Array) -> jax.Array:
            return _put_fn(stk, x, idx)

        def get(stk: jax.Array, idx: jax.Array, n: int) -> jax.Array:
            return _get_fn(stk, idx, n)

        # The stack is donated: XLA updates it in place (no second stack-sized buffer); the caller rebinds.
        op_put = pallas.jax_op("expert_stack::put_rows", put, donate_argnums=(0,))
        op_put.register_fake(lambda stk, *a, **k: torch.empty_like(stk))
        op_get = pallas.jax_op("expert_stack::get_rows", get)
        _OPS["put"], _OPS["get"] = op_put, op_get
    return _OPS


class LoadStacks:
    """The four stacks of one model's target MoE layers and the bookkeeping of what has been written."""

    def __init__(self, layers: list):
        self.layers = layers
        self.row = {id(layer): r for r, layer in enumerate(layers)}
        self.params = {}
        for r, layer in enumerate(layers):
            for name, kind in _PARAM_KIND.items():
                self.params[id(getattr(layer, name))] = (r, kind)
        w13, w2 = layers[0].w13_weight, layers[0].w2_weight
        L = len(layers)
        self.device = w13.device
        self.raw_shape = {"w13": tuple(w13.shape), "w2": tuple(w2.shape)}
        e, n13, h = w13.shape
        e2, h2, i2 = w2.shape
        self.shapes = {"w13": (L, e, h, n13), "w2": (L, e2, i2, h2), "s13": (L, e, 1, 1, n13), "s2": (L, e2, 1, 1, h2)}
        self.dtypes = {"w13": w13.dtype, "w2": w2.dtype, "s13": torch.float32, "s2": torch.float32}
        self.stacks: dict[str, torch.Tensor] = {}
        self.raw_written: set = set()
        self.scales_written: set = set()
        self.processed: set = set()
        self.writes = 0
        self.hbm = {"before_alloc": hbm_stats()}
        self.adopted = False

    def nbytes(self, kind: str) -> int:
        n = 1
        for d in self.shapes[kind]:
            n *= d
        return n * torch.empty((), dtype=self.dtypes[kind]).element_size()

    def allocate(self) -> None:
        from vllm_torchtpu.utils import synchronize_tensors

        for kind in KINDS:
            t = torch.zeros(self.shapes[kind], dtype=self.dtypes[kind], device=self.device)
            if t.device.type == "tpu":
                synchronize_tensors([t], wait=True)
            self.stacks[kind] = t
        self.hbm["after_alloc"] = hbm_stats()
        logger.info(
            "expert_stack: allocated load-time expert stacks for %d MoE layers (%s .. %s): %s, %d bytes; hbm %s -> %s",
            len(self.layers), getattr(self.layers[0], "layer_name", "?"), getattr(self.layers[-1], "layer_name", "?"),
            {k: list(v) for k, v in self.shapes.items()}, sum(self.nbytes(k) for k in KINDS),
            self.hbm["before_alloc"], self.hbm["after_alloc"])

    def release(self) -> None:
        self.stacks.clear()

    def _put(self, kind: str, row: int, e0: int, x: torch.Tensor) -> None:
        from vllm_torchtpu.utils import synchronize_tensors

        stk = self.stacks[kind]
        if tuple(x.shape[1:]) != tuple(stk.shape[2:]) or x.dtype != stk.dtype or e0 + x.shape[0] > stk.shape[1]:
            raise ValueError(f"expert_stack: {kind} rows {tuple(x.shape)} {x.dtype} do not fit {tuple(stk.shape)} "
                             f"{stk.dtype} at expert {e0}")
        if stk.device.type != "tpu":                     # CPU (tests): a plain in-place slice write
            stk[row, e0:e0 + x.shape[0]].copy_(x)
            return
        idx = torch.tensor([row, e0], dtype=torch.int32, device=stk.device)
        out = _ops()["put"](stk, x, idx)
        # the donated input is invalid now; the stack is the op's (aliased) output, materialized before the next
        # read or write of this stack is issued
        self.stacks[kind] = out
        synchronize_tensors([out], wait=True)
        self.writes += 1

    def write_raw(self, param: Any, host: torch.Tensor) -> None:
        """Upload a completed checkpoint-layout parameter (its host stand-in) into its row: the same bytes,
        reshaped to the served row's shape, in <= 256 MiB slices."""
        row, kind = self.params[id(param)]
        e, a, b = self.shapes[kind][1:]
        src = host.detach()
        if tuple(src.shape) != self.raw_shape[kind]:
            raise ValueError(f"expert_stack: {kind} row {row}: host {tuple(src.shape)} != {self.raw_shape[kind]}")
        per = max(1, src[0].numel() * src.element_size())
        step = max(1, _UPLOAD_CHUNK_BYTES // per)
        for s in range(0, e, step):
            part = src[s:s + step]
            x = torch.Tensor._make_subclass(torch.Tensor, part.contiguous(), False).reshape(part.shape[0], a, b)
            self._put(kind, row, s, x.to(self.device))
        self.raw_written.add((row, kind))
        if len(self.raw_written) == 2 * len(self.layers):
            self.hbm["after_raw_rows"] = hbm_stats()
            logger.info("expert_stack: checkpoint experts of all %d layers are in their rows; hbm %s",
                        len(self.layers), self.hbm["after_raw_rows"])

    def read_raw(self, layer: Any, kind: str, start: int, end: int) -> torch.Tensor:
        """Experts [start, end) of `layer`'s row in the checkpoint layout, as a fresh tensor (materialized, so a
        later donated write of the same row cannot race it)."""
        from vllm_torchtpu.utils import synchronize_tensors

        row = self.row[id(layer)]
        if (row, kind) not in self.raw_written:
            raise RuntimeError(f"expert_stack: {kind} of {getattr(layer, 'layer_name', row)} was never loaded")
        stk = self.stacks[kind]
        if stk.device.type != "tpu":
            t = stk[row, start:end]
            return t.reshape((end - start,) + tuple(t.shape[1:-2]) + (t.shape[-1], t.shape[-2])).clone()
        idx = torch.tensor([row, start], dtype=torch.int32, device=stk.device)
        out = _ops()["get"](stk, idx, end - start)
        synchronize_tensors([out], wait=True)
        return out

    def write_rows(self, layer: Any, kind: str, start: int, x: torch.Tensor) -> None:
        """Requantized experts [start, start + len(x)) of `layer`, over the checkpoint bytes they came from."""
        self._put(kind, self.row[id(layer)], start, x)

    def write_scale(self, layer: Any, kind: str, x: torch.Tensor) -> None:
        row = self.row[id(layer)]
        self._put(kind, row, 0, x)
        self.scales_written.add((row, kind))

    def placeholder(self, kind: str) -> torch.Tensor:
        """A per-layer tensor of the served shape that is never materialized (shape / dtype / device only)."""
        return torch.empty(self.shapes[kind][1:], dtype=self.dtypes[kind], device=self.device)

    def mark_processed(self, layer: Any) -> None:
        self.processed.add(self.row[id(layer)])
        if len(self.processed) == len(self.layers):
            self.hbm["after_processing"] = hbm_stats()
            logger.info("expert_stack: all %d layers requantized in place (%d row writes); hbm %s",
                        len(self.layers), self.writes, self.hbm["after_processing"])

    def incomplete(self) -> str | None:
        L = len(self.layers)
        miss_raw = sorted({r for r in range(L) for k in ("w13", "w2") if (r, k) not in self.raw_written})
        miss_s = sorted({r for r in range(L) for k in ("s13", "s2") if (r, k) not in self.scales_written})
        miss_p = sorted(set(range(L)) - self.processed)
        if miss_raw or miss_s or miss_p:
            return f"rows never loaded {miss_raw[:8]}, scales missing {miss_s[:8]}, not requantized {miss_p[:8]}"
        return None

    def report(self) -> dict:
        # what the per-layer form holds for these layers: one served tensor per (layer, kind)
        per_layer = sum(len(self.layers) * self.nbytes(k) // self.shapes[k][0] for k in KINDS)
        return {
            "layers": len(self.layers),
            "form": "load-time",
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "stacked_bytes": sum(int(t.numel() * t.element_size()) for t in self.stacks.values()),
            "per_layer_bytes": per_layer,
            "row_writes": self.writes,
            "hbm": dict(self.hbm),
        }
