"""Standalone TPU microbenchmark of the batched-RPA DCP CACHE_ONLY decode pass (pure JAX).

Geometry of the Qwen3.5-397B decode leg (TP8, DCP4, MTP K=3): per device 16 q heads (4 own
heads all-gathered over the DCP group), 1 KV head, head_dim 256, fp8 KV in the SEQ_ALONG_LANE
("nhs") layout, page 128, 4 query rows per sequence, cp_group_size 4 with a 128-token
interleave. One call = one full-attention layer's context pass on one device.

Usage (repo mapped at /workspace/vllm-torchtpu):
  python3 scripts/bench_rpa_decode.py --n 64,128 --ctx 80000 --cfg default,b16,b4,nfp8
Prints one JSON line per (n, cfg): {"n":..,"cfg":..,"us":..,"GBps":..,"md5":..}.
"""

import argparse
import functools
import hashlib
import json
import os
import statistics
import sys
import time
import types

REPO = os.environ.get("REPO", os.path.join(os.path.dirname(__file__), "..", "src"))
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as C  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as W  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--n", default="64,128")
AP.add_argument("--ctx", type=int, default=80000)
AP.add_argument("--T", type=int, default=4)
AP.add_argument("--dcp", type=int, default=4)
AP.add_argument("--qh", type=int, default=16, help="q heads per device after the DCP gather")
AP.add_argument("--cfg", default="default")
AP.add_argument("--iters", type=int, default=20)
AP.add_argument("--max_pages", type=int, default=515)
AP.add_argument("--shared", default="0", help="comma list of 0/1: all sequences read one page list")
AP.add_argument("--run", default="1", help="comma list: contiguous pages per allocation run (0: whole seq)")
A = AP.parse_args()

PAGE = 128
D = 256
dev = jax.devices()[0]
print("device:", dev, flush=True)


def cfg_kwargs(name):
    """name: '+'-joined tokens: default | b<N> (decode batch) | v<MiB> (vmem limit) |
    nfp8 | nfp8x2 | k<bkv> (decode bkv override) | diet0."""
    kw = {}
    env = {}
    blocks = {}
    for tok in name.split("+"):
        if tok == "default":
            continue
        if tok.startswith("b") and tok[1:].isdigit():
            kw["decode_batch_size"] = int(tok[1:])
        elif tok.startswith("v") and tok[1:].isdigit():
            kw["vmem_limit_bytes"] = int(tok[1:]) << 20
        elif tok == "nfp8":
            kw["native_fp8"] = "qkpv"
        elif tok == "nfp8x2":
            kw["native_fp8"] = "qk2pv2"
        elif tok.startswith("k") and tok[1:].isdigit():
            blocks["bkv_sz"] = int(tok[1:])
        elif tok == "diet0":
            env["TPU_RPA_VPU_DIET"] = "0"
        else:
            raise ValueError(tok)
    return kw, env, blocks


def build(n, shared=False, run=1):
    T = A.T
    local = -(-A.ctx // A.dcp)
    num_pages = n * (-(-local // PAGE) + 2) * 2 + 64
    shape = W.get_kv_cache_shape(num_pages, PAGE, 1, D, jnp.float8_e4m3fn,
                                 kv_layout=C.KVLayout.SEQ_ALONG_LANE)
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    kv = (jax.random.normal(k1, shape, jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    q = jax.random.normal(k2, (n * T, A.qh, D), jnp.float32).astype(jnp.bfloat16)
    # New K/V arrive already in the cache dtype (the adapter quantizes them).
    knew = (jax.random.normal(k3, (n * T, 1, D), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    vnew = (jax.random.normal(k4, (n * T, 1, D), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    kv_lens = jnp.full((n,), A.ctx + T, jnp.int32)
    pages = np.zeros((n, A.max_pages), np.int32)
    per = -(-local // PAGE) + 1
    R = per if run <= 0 else run
    nruns = -(-per // R)
    slots = (num_pages - 1) // R
    if slots < n * nruns:
        raise ValueError("not enough pages for runs")
    starts = np.random.default_rng(0).permutation(slots)[: n * nruns].reshape(n, nruns) * R + 1
    allp = (starts[:, :, None] + np.arange(R)[None, None, :]).reshape(n, nruns * R)[:, :per]
    pages[:, :per] = allp
    if shared:
        pages[:, :per] = pages[0, :per]
    page_indices = jnp.asarray(pages.reshape(-1))
    cu = jnp.arange(n + 1, dtype=jnp.int32) * T
    dist = jnp.array([n, n, n], jnp.int32)
    bytes_kv = n * local * 2 * D  # K + V, fp8, one head
    return kv, q, knew, vnew, kv_lens, page_indices, cu, dist, bytes_kv


def run(n, name, data, shared=False, runlen=1):
    kw, env, blocks = cfg_kwargs(name)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    kv, q, knew, vnew, kv_lens, pidx, cu, dist, nbytes = data
    if blocks:
        # Start from the wrapper's own pick and override the named fields.
        model = C.ModelConfigs(num_q_heads=A.qh, num_kv_heads=1, head_dim=D,
                               mask_value=float(jnp.finfo(jnp.bfloat16).min), sm_scale=1 / 16)
        serve = C.ServingConfigs(
            native_fp8=kw.get("native_fp8", ""), num_seqs=n, num_page_indices=pidx.shape[0],
            total_q_tokens=q.shape[0], dtype_q=q.dtype, dtype_kv=kv.dtype, dtype_out=q.dtype,
            page_size=PAGE, kv_layout=C.KVLayout.SEQ_ALONG_LANE, decode_query_size=A.T,
            cp_group_size=A.dcp, attention_scope=C.AttentionScope.CACHE_ONLY, return_lse=True)
        from jax.experimental.pallas import tpu as pltpu
        vmem = kw.get("vmem_limit_bytes", pltpu.get_tpu_info().vmem_capacity_bytes)
        dflt, _ = W.calculate_block_sizes(
            model, serve, vmem, decode_query_size=A.T,
            decode_batch_size=kw.get("decode_batch_size", 8))
        import dataclasses
        kw["decode_block_sizes"] = dataclasses.replace(dflt, **blocks)
        print("  blocks:", kw["decode_block_sizes"], flush=True)

    # The cache is donated, as in the engine: without it XLA copies the whole
    # cache to produce the aliased output every call.
    @functools.partial(jax.jit, donate_argnums=(0,))
    def f(kv, q, knew, vnew, kv_lens, pidx, cu, dist):
        out, new_kv, lse = W.ragged_paged_attention(
            q, knew, vnew, kv, kv_lens, pidx, cu, dist,
            sm_scale=1 / 16, k_scale=1.0, v_scale=1.0,
            cp_group_size=A.dcp, cp_rank=jnp.array([0], jnp.int32),
            attention_scope=C.AttentionScope.CACHE_ONLY, decode_query_size=A.T,
            return_lse=True, kv_layout=C.KVLayout.SEQ_ALONG_LANE, **kw)
        return out, lse, new_kv

    try:
        kv = jnp.copy(kv)
        t0 = time.time()
        out, lse, kv = f(kv, q, knew, vnew, kv_lens, pidx, cu, dist)
        jax.block_until_ready(out)
        comp = time.time() - t0
        ts = []
        for _ in range(A.iters):
            t0 = time.perf_counter()
            o, l2, kv = f(kv, q, knew, vnew, kv_lens, pidx, cu, dist)
            jax.block_until_ready(o)
            ts.append((time.perf_counter() - t0) * 1e6)
        us = statistics.median(ts)
        md5 = hashlib.md5(np.asarray(out.astype(jnp.float32)).tobytes()).hexdigest()[:12]
        finite = bool(jnp.isfinite(out.astype(jnp.float32)).all())
        rec = {"n": n, "cfg": name, "shared": shared, "run": runlen, "us": round(us, 1), "us_min": round(min(ts), 1),
               "GBps": round(nbytes / us / 1e3, 1), "compile_s": round(comp, 1),
               "md5": md5, "finite": finite}
    except Exception as e:  # noqa: BLE001
        rec = {"n": n, "cfg": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("RPA_RESULT " + json.dumps(rec), flush=True)


for rl in [int(x) for x in A.run.split(",")]:
    for sh in [x == "1" for x in A.shared.split(",")]:
        for n in [int(x) for x in A.n.split(",")]:
            data = build(n, sh, rl)
            for name in A.cfg.split(","):
                run(n, name, data, sh, rl)
