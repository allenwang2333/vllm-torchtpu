"""Diagnostic: per-layer, per-step routed-expert histograms of the fused EP MoE.

Off unless VLLM_PROBE_ROUTE_DIR is set (VLLM_ prefix so Ray copies it to the
workers). Each MoE layer that takes the fused EP path owns a persistent f32
[E] buffer. Inside the compiled forward the layer recomputes the router's
top-k from the same logits and score bias the kernel receives and writes the
per-expert pair count of THIS rank's rows into the buffer (same persistent
buffer + in-graph write pattern as the KV cache and probe_topk). The runner
reads every buffer on the host after each real forward of >= MIN_TOKENS
scheduled tokens and appends one row per step to an npz per worker.

  VLLM_PROBE_ROUTE_DIR         output dir (npz per worker, rewritten every 8 rows)
  VLLM_PROBE_ROUTE_MIN_TOKENS  only record steps with at least this many tokens (default 256)
"""

import os
import re
import sys
import time

import numpy as np
import torch

_DIR = os.environ.get("VLLM_PROBE_ROUTE_DIR", "")
_MIN = int(os.environ.get("VLLM_PROBE_ROUTE_MIN_TOKENS", "256"))

BUFS: dict = {}  # decoder layer id -> torch f32 [E]
_rows: list = []
_meta: list = []
_last_flush = 0


def enabled() -> bool:
    return bool(_DIR)


def layer_of(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name or "")
    return int(m.group(1)) if m else -1


def register(layer, num_experts: int, device) -> None:
    buf = torch.zeros(num_experts, dtype=torch.float32, device=device)
    layer.register_buffer("_tpu_probe_route", buf, persistent=False)
    lid = layer_of(getattr(layer, "layer_name", ""))
    BUFS[lid] = layer._tpu_probe_route
    print(f"PROBE_ROUTE buffer layer={lid} E={num_experts} dev={buf.device}",
          file=sys.stderr, flush=True)


def count(router_logits, score_bias, scoring_fn: str, topk: int, num_experts: int):
    """Per-expert pair counts of this rank's rows, as the kernel would route them.

    Mirrors fused_ep_moe_v2's selection: top-k over scoring_fn(logits) + bias,
    dead rows (every logit -inf, the padding mask) route nowhere.
    """
    from vllm_torchtpu.layers.adapter.moe_routing import _apply_scoring_fn

    logits = router_logits.float()
    valid = torch.isfinite(logits).any(dim=-1).to(torch.float32)  # [T]
    scores = _apply_scoring_fn(scoring_fn, logits)
    sel = scores if score_bias is None else scores + score_bias.float()[None, :]
    sel = torch.nan_to_num(sel, nan=-1e30, neginf=-1e30)
    idx = torch.topk(sel, k=topk, dim=-1).indices  # [T, k]
    ar = torch.arange(num_experts, device=idx.device, dtype=idx.dtype)
    hit = (idx.unsqueeze(-1) == ar.view(1, 1, -1)).to(torch.float32)  # [T, k, E]
    return (hit * valid.view(-1, 1, 1)).sum(dim=(0, 1))


def after_forward(runner, num_tokens: int, positions_np, req_id) -> None:
    global _last_flush
    if not BUFS or num_tokens < _MIN:
        return
    lids = sorted(BUFS)
    host = torch.stack([BUFS[l] for l in lids]).cpu().numpy()  # [L, E] f32
    pc = getattr(runner, "parallel_config", None)
    dp = int(getattr(pc, "data_parallel_rank", -1)) if pc is not None else -1
    p0 = int(positions_np[0]) if positions_np is not None else -1
    _rows.append(np.rint(host).astype(np.uint16))
    _meta.append((time.time(), num_tokens, p0, dp, str(req_id)))
    sums = host.sum(axis=1)
    print(f"PROBE_ROUTE dp={dp} pid={os.getpid()} step={len(_rows)} ntok={num_tokens} p0={p0} "
          f"req={str(req_id)[-24:]} pairs/layer min={sums.min():.0f} max={sums.max():.0f} "
          f"L{lids[0]}max={host[0].max():.0f}", file=sys.stderr, flush=True)
    if len(_rows) - _last_flush >= 8 or len(_rows) <= 2:
        flush(dp)


def flush(dp=None) -> None:
    global _last_flush
    if not _rows:
        return
    os.makedirs(_DIR, exist_ok=True)
    if dp is None:
        dp = _meta[-1][3]
    path = os.path.join(_DIR, f"route_dp{dp}_pid{os.getpid()}.npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        counts=np.stack(_rows),  # [S, L, E] uint16, this rank's rows only
        layers=np.array(sorted(BUFS), dtype=np.int32),
        t=np.array([m[0] for m in _meta]),
        ntok=np.array([m[1] for m in _meta], dtype=np.int32),
        p0=np.array([m[2] for m in _meta], dtype=np.int64),
        dp=np.array([m[3] for m in _meta], dtype=np.int32),
        req=np.array([m[4] for m in _meta]),
    )
    os.replace(tmp, path)
    _last_flush = len(_rows)
