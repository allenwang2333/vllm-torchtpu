"""Correctness (and on TPU, speed) of batched_rpa.decode_v2 against a float64
reference and against the batched RPA DCP CACHE_ONLY kernel.

  python3 scripts/attn/test_decode_v2.py [--interpret] [--case small|engine]

Prints TEST {json} lines: max abs error vs reference for both kernels, max abs
difference between them, and NaN checks. Tokens past every sequence's end are
filled with fp8 NaN bytes, so a kernel that reads them unmasked fails.
"""

import argparse
import functools
import json
import os
import sys
import types

REPO = os.environ.get("REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--interpret", action="store_true")
AP.add_argument("--case", default="small")
AP.add_argument("--ranks", default="0,1,2,3")
AP.add_argument("--old", type=int, default=1, help="also run the batched RPA kernel")
AP.add_argument("--bkv", type=int, default=0)
AP.add_argument("--slots", type=int, default=4)
AP.add_argument("--groups", default="1,2")
AP.add_argument("--depth", type=int, default=2)
AP.add_argument("--issues", default="runs,pages,loop")
AP.add_argument("--pvs", default="bf16,fp8x2")
A = AP.parse_args()

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as C  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import decode_v2 as V2  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as W  # noqa: E402

if A.interpret:
    V2._INTERPRET = True

PAGE, D, H, DCP = 128, 256, 16, 4
FP8 = jnp.float8_e4m3fn


def emit(**kw):
    print("TEST " + json.dumps(kw), flush=True)


def make_case(name, seed=0):
    rng = np.random.default_rng(seed)
    if name == "small":
        Tq, pps = 4, 24
        # global cache lengths: empty, short, page-misaligned, multi-block
        cache = np.array([0, 5, 300, 1000, 9000, 3 * 512 * DCP + 7], np.int64)
        q_lens = np.array([4, 1, 4, 4, 2, 4], np.int64)
        N = len(cache)
        bkv = A.bkv or 512
    else:  # engine-like: 16 sequences of ~80K context
        Tq, pps = 4, 176
        N = 16
        cache = np.full(N, 80000, np.int64) + rng.integers(-3000, 3000, N)
        q_lens = np.full(N, 4, np.int64)
        bkv = A.bkv or 4096
    T = int(q_lens.sum())
    T_pad = ((T + 15) // 16) * 16
    cu = np.zeros(N + 1, np.int64)
    cu[1:] = np.cumsum(q_lens)
    P = (N + 1) * pps + 16
    shape = W.get_kv_cache_shape(P, PAGE, 1, D, FP8, kv_layout=C.KVLayout.SEQ_ALONG_LANE)
    kvf = rng.standard_normal(shape).astype(np.float32) * 0.5
    kv = np.asarray(jnp.asarray(kvf).astype(FP8))
    # Fill everything with NaN bytes first; real tokens are written below.
    nan_byte = np.array([0x7F], np.uint8).view(np.uint8)
    kv_bytes = kv.view(np.uint8).copy()
    kv_bytes[...] = nan_byte
    # Even sequences: scattered pages; odd sequences: runs of 5 consecutive
    # pages (vLLM's 640-token blocks), in shuffled run order.
    pages = np.zeros((N, pps), np.int32)
    perm = rng.permutation(np.arange(1, P))
    used = 0
    for s in range(N):
        chunk = np.sort(perm[used:used + pps]) if s % 2 else perm[used:used + pps]
        if s % 2:
            nr = -(-pps // 5)
            base = 1 + s * pps
            order = rng.permutation(nr)
            runs = [np.arange(base + 5 * r, base + 5 * r + 5) for r in order]
            chunk = np.concatenate(runs)[:pps]
        pages[s] = chunk
        used += pps
    assert pages.max() < P
    q = (rng.standard_normal((T_pad, H, D)) * 2.0).astype(np.float32)
    q[T:] = 0
    return dict(Tq=Tq, pps=pps, N=N, cache=cache, q_lens=q_lens, T=T, T_pad=T_pad, cu=cu,
                kv_real=kv, kv_bytes=kv_bytes, pages=pages, q=q, bkv=bkv)


def local_len(g, rank):
    sp = DCP * PAGE
    return (g // sp) * PAGE + np.clip(g % sp - rank * PAGE, 0, PAGE)


def build_rank(c, rank):
    """KV bytes for this rank: real values on the pages the rank owns, NaN elsewhere."""
    kvb = c["kv_bytes"].copy()
    real = c["kv_real"].view(np.uint8)
    for s in range(c["N"]):
        ln = int(local_len(c["cache"][s], rank))
        for p in range(-(-ln // PAGE)):
            pg = c["pages"][s, p]
            n = min(PAGE, ln - p * PAGE)
            kvb[pg, :, :, :, :n] = real[pg, :, :, :, :n]
    return kvb.view(FP8)


def reference(c, rank, sm_scale):
    """float64 attention of every query row over this rank's cached tokens."""
    kv = c["kv_real"].astype(np.float64)
    T_pad = c["T_pad"]
    out = np.zeros((T_pad, H, D))
    lse = np.full((T_pad, H), -np.inf)
    qb = np.asarray(jnp.asarray(c["q"]).astype(jnp.bfloat16).astype(np.float32)).astype(np.float64)
    for s in range(c["N"]):
        ln = int(local_len(c["cache"][s], rank))
        if ln == 0:
            continue
        toks = np.arange(ln)
        pg = c["pages"][s, toks // PAGE]
        off = toks % PAGE
        # kv[page, 2, dp, pack, page_off] -> K [ln, D] with dim = dp * pack + j
        Kt = kv[pg, 0, :, :, off].reshape(ln, D)
        Vt = kv[pg, 1, :, :, off].reshape(ln, D)
        for t in range(int(c["q_lens"][s])):
            r = int(c["cu"][s]) + t
            sc = qb[r] @ Kt.T * sm_scale  # [H, ln]
            m = sc.max(-1, keepdims=True)
            p = np.exp(sc - m)
            l = p.sum(-1, keepdims=True)
            out[r] = (p @ Vt) / l
            lse[r] = (m + np.log(l))[:, 0]
    return out, lse


def run_v2(c, kv, rank, sm_scale, group, issue, pv):
    q = jnp.asarray(c["q"]).astype(jnp.bfloat16)
    kv_lens = jnp.asarray(c["cache"] + c["q_lens"], jnp.int32)
    pidx = jnp.asarray(c["pages"].reshape(-1))
    cu = jnp.asarray(c["cu"], jnp.int32)
    out, lse = V2.decode_cache_pass(
        q, kv, kv_lens, pidx, cu, jnp.array([rank], jnp.int32),
        sm_scale=sm_scale, k_scale=1.0, v_scale=1.0, cp_group_size=DCP,
        decode_query_size=c["Tq"], bkv=c["bkv"], group=group, depth=A.depth, issue_mode=issue, pv_mode=pv)
    return np.asarray(out.astype(jnp.float32), np.float64), np.asarray(lse, np.float64)


def run_old(c, kv, rank, sm_scale):
    q = jnp.asarray(c["q"]).astype(jnp.bfloat16)
    T_pad = c["T_pad"]
    knew = jnp.zeros((T_pad, 1, D), FP8)
    vnew = jnp.zeros((T_pad, 1, D), FP8)
    kv_lens = jnp.asarray(c["cache"] + c["q_lens"], jnp.int32)
    pidx = jnp.asarray(c["pages"].reshape(-1))
    cu = jnp.asarray(c["cu"], jnp.int32)
    N = c["N"]
    dist = jnp.array([N, N, N], jnp.int32)
    out, _, lse = W.ragged_paged_attention(
        q, knew, vnew, jnp.asarray(kv), kv_lens, pidx, cu, dist,
        sm_scale=sm_scale, k_scale=1.0, v_scale=1.0, cp_group_size=DCP,
        cp_rank=jnp.array([rank], jnp.int32), attention_scope=C.AttentionScope.CACHE_ONLY,
        decode_query_size=c["Tq"], return_lse=True, kv_layout=C.KVLayout.SEQ_ALONG_LANE)
    return np.asarray(out.astype(jnp.float32), np.float64), np.asarray(lse.astype(jnp.float32), np.float64)


def compare(name, out, lse, ref_out, ref_lse, rows):
    o, ro = out[rows], ref_out[rows]
    l, rl = lse[rows], ref_lse[rows]
    fin = np.isfinite(rl)
    err_o = np.abs(o - ro)[fin]
    err_l = np.abs(l - rl)[fin]
    nan_bad = bool(np.isnan(o).any())
    inf_mismatch = bool((np.isfinite(l) != fin).any())
    return {f"{name}_max_abs_o": float(err_o.max()) if err_o.size else 0.0,
            f"{name}_mean_abs_o": float(err_o.mean()) if err_o.size else 0.0,
            f"{name}_max_abs_lse": float(err_l.max()) if err_l.size else 0.0,
            f"{name}_nan": nan_bad, f"{name}_lse_inf_mismatch": inf_mismatch}


c = make_case(A.case)
sm_scale = 1.0 / 16
rows = np.arange(c["T"])
for rank in [int(x) for x in A.ranks.split(",")]:
    kv = jnp.asarray(build_rank(c, rank))
    ref_out, ref_lse = reference(c, rank, sm_scale)
    rec = dict(case=A.case, rank=rank, bkv=c["bkv"], local=[int(local_len(g, rank)) for g in c["cache"]][:8])
    o2 = None
    for group, issue, pv in [(int(g), i, v) for g in A.groups.split(",") for i in A.issues.split(",")
                             for v in A.pvs.split(",")]:
        tag = f"v2g{group}{issue[0]}{pv[0]}"
        try:
            o2, l2 = run_v2(c, kv, rank, sm_scale, group, issue, pv)
            rec.update(compare(tag, o2, l2, ref_out, ref_lse, rows))
        except Exception as e:  # noqa: BLE001
            rec[tag + "_error"] = f"{type(e).__name__}: {str(e)[:1500]}"
            o2 = None
    if A.old:
        try:
            o1, l1 = run_old(c, kv, rank, sm_scale)
            rec.update(compare("old", o1, l1, ref_out, ref_lse, rows))
            if o2 is not None:
                fin = np.isfinite(ref_lse[rows])
                rec["v2_vs_old_max_abs_o"] = float(np.abs(o2[rows] - o1[rows])[fin].max())
        except Exception as e:  # noqa: BLE001
            rec["old_error"] = f"{type(e).__name__}: {str(e)[:600]}"
    emit(**rec)
print("TEST_DONE", flush=True)
