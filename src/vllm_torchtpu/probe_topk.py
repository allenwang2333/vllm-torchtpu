"""Diagnostic: dump sparse-indexer top-k sets for selected layers / prefill steps.

Off unless VLLM_PROBE_TOPK_LAYERS is set. Env (VLLM_ prefix so Ray copies it):
  VLLM_PROBE_TOPK_LAYERS  comma list of decoder layer ids whose indexer output is kept
  VLLM_PROBE_TOPK_STARTS  comma list of prefill-step start positions to dump
  VLLM_PROBE_TOPK_DIR     output dir (raw rows + stats json)
  VLLM_PROBE_TOPK_ROWS    raw rows saved per (layer, step); default 256
Each indexer layer in the set copies its top-k into a persistent buffer inside
the compiled graph (same pattern as `topk_indices_buffer`); the runner reads the
buffers on the host after a real forward whose first position is in STARTS.
"""

import json
import os
import re
import sys
import time

import numpy as np

_LAYERS = {int(x) for x in os.environ.get("VLLM_PROBE_TOPK_LAYERS", "").split(",") if x.strip()}
_STARTS = {int(x) for x in os.environ.get("VLLM_PROBE_TOPK_STARTS", "").split(",") if x.strip()}
_DIR = os.environ.get("VLLM_PROBE_TOPK_DIR", "/tmp/probe_topk")
_ROWS = int(os.environ.get("VLLM_PROBE_TOPK_ROWS", "256"))

BUFS: dict = {}  # layer id -> torch tensor [max_tokens, topk]
_seen: dict = {}


def enabled() -> bool:
    return bool(_LAYERS)


def layer_of(prefix: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", prefix)
    return int(m.group(1)) if m else None


def want_layer(prefix: str) -> bool:
    lid = layer_of(prefix)
    return lid is not None and lid in _LAYERS


def _log(msg: str) -> None:
    print(f"PROBE_TOPK pid={os.getpid()} {msg}", file=sys.stderr, flush=True)


def _stats(rows: np.ndarray) -> dict:
    """rows: int [n, k], -1 padded. Union / Jaccard statistics."""
    n, k = rows.shape
    sets = [np.unique(r[r >= 0]) for r in rows]
    valid = np.array([s.size for s in sets])
    out = {"n": int(n), "k": int(k), "valid_min": int(valid.min()), "valid_max": int(valid.max()),
           "valid_mean": float(valid.mean())}
    for g in (64, 128, 256):
        us = []
        for s in range(0, n - g + 1, g):
            blk = rows[s:s + g]
            us.append(int(np.unique(blk[blk >= 0]).size))
        out[f"union_{g}"] = us
    jac = []
    for i in range(n - 1):
        a, b = sets[i], sets[i + 1]
        inter = np.intersect1d(a, b, assume_unique=True).size
        uni = a.size + b.size - inter
        jac.append(inter / uni if uni else 1.0)
    jac = np.array(jac)
    out["jac_adj"] = {"mean": float(jac.mean()), "p10": float(np.percentile(jac, 10)),
                      "p50": float(np.percentile(jac, 50)), "p90": float(np.percentile(jac, 90))}
    # union growth over the first m queries of each 256-row group
    growth = {}
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        vals = []
        for s in range(0, n - 256 + 1, 256):
            blk = rows[s:s + m]
            vals.append(int(np.unique(blk[blk >= 0]).size))
        growth[m] = float(np.mean(vals))
    out["growth"] = growth
    return out


def after_forward(runner, num_tokens: int, positions_np: np.ndarray, req_id) -> None:
    if not BUFS or num_tokens <= 0:
        return
    p0 = int(positions_np[0])
    if p0 not in _STARTS:
        return
    if int(positions_np[num_tokens - 1]) != p0 + num_tokens - 1:
        _log(f"skip p0={p0}: non-contiguous positions n={num_tokens}")
        return
    key = (str(req_id), p0)
    if key in _seen:
        return
    _seen[key] = 1
    os.makedirs(_DIR, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", str(req_id))[-40:]
    t0 = time.perf_counter()
    for lid in sorted(BUFS):
        rows = BUFS[lid][:num_tokens].cpu().numpy().astype(np.int32)
        st = _stats(rows)
        st.update(layer=lid, p0=p0, req=str(req_id), num_tokens=num_tokens,
                  maxidx=int(rows.max()), minidx=int(rows.min()))
        base = os.path.join(_DIR, f"{tag}_p{p0}_L{lid}")
        raw = np.sort(rows[:_ROWS], axis=1)
        np.savez_compressed(base + ".npz", rows=raw)
        with open(base + ".json", "w") as f:
            json.dump(st, f)
        u = {g: (int(np.median(st[f"union_{g}"])), int(min(st[f"union_{g}"])),
                 int(max(st[f"union_{g}"]))) for g in (64, 128, 256)}
        _log(f"req={tag} p0={p0} L={lid} valid={st['valid_min']}-{st['valid_max']} "
             f"maxidx={st['maxidx']} U64(med,min,max)={u[64]} U128={u[128]} U256={u[256]} "
             f"jac_adj_mean={st['jac_adj']['mean']:.3f} p50={st['jac_adj']['p50']:.3f} "
             f"growth={json.dumps(st['growth'])}")
    _log(f"dumped req={tag} p0={p0} layers={sorted(BUFS)} in {time.perf_counter()-t0:.1f}s -> {_DIR}")
