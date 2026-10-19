"""Device time of decode_v2 vs the batched RPA DCP CACHE_ONLY kernel at the
decode leg's geometry (16 q heads, 1 fp8 KV head, head_dim 256, DCP 4, page
128, 4 query rows per sequence, 80K context, pages in runs of 5 = 640 tokens).

PROBE {json} per (n, config): device us per call, GB/s of K/V bytes read.
"""

import functools
import json
import os
import sys
import types

REPO = os.environ.get("REPO", "/workspace/vllm-torchtpu/src")
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devtime  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import configs as C  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import decode_v2 as V2  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as W  # noqa: E402

PAGE, D, T, DCP, QH = 128, 256, 4, 4, 16
CTX = int(os.environ.get("BV_CTX", "80000"))
MAXP = int(os.environ.get("BV_MAXP", "512"))


def emit(**kw):
    print("PROBE " + json.dumps(kw), flush=True)


def build(n, run=5, seed=0):
    local = -(-CTX // DCP)
    per = -(-local // PAGE) + 1
    num_pages = n * per * 2 + 64
    shape = W.get_kv_cache_shape(num_pages, PAGE, 1, D, jnp.float8_e4m3fn, kv_layout=C.KVLayout.SEQ_ALONG_LANE)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 2)
    kv = (jax.random.normal(k1, shape, jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    q = (jax.random.normal(k2, (n * T, QH, D), jnp.float32) * 2.0).astype(jnp.bfloat16)
    kv_lens = jnp.full((n,), CTX + T, jnp.int32)
    pages = np.zeros((n, MAXP), np.int32)
    nruns = -(-per // run)
    slots = (num_pages - 1) // run
    starts = np.random.default_rng(seed).permutation(slots)[: n * nruns].reshape(n, nruns) * run + 1
    allp = (starts[:, :, None] + np.arange(run)[None, None, :]).reshape(n, nruns * run)[:, :per]
    pages[:, :per] = allp
    cu = jnp.arange(n + 1, dtype=jnp.int32) * T
    nbytes = n * local * 2 * D
    return kv, q, kv_lens, jnp.asarray(pages.reshape(-1)), cu, nbytes


@functools.partial(jax.jit, donate_argnums=(0,))
def old_fn(kv, q, kv_lens, pidx, cu):
    n = kv_lens.shape[0]
    knew = jnp.zeros((q.shape[0], 1, D), jnp.float8_e4m3fn)
    out, new_kv, lse = W.ragged_paged_attention(
        q, knew, knew, kv, kv_lens, pidx, cu, jnp.array([n, n, n], jnp.int32),
        sm_scale=1 / 16, k_scale=1.0, v_scale=1.0, cp_group_size=DCP, cp_rank=jnp.array([0], jnp.int32),
        attention_scope=C.AttentionScope.CACHE_ONLY, decode_query_size=T, return_lse=True,
        kv_layout=C.KVLayout.SEQ_ALONG_LANE)
    return new_kv, out, lse.astype(jnp.float32)


def v2_fn(bkv, group, depth, issue, dbg, pv="bf16", chunk=4096):
    @functools.partial(jax.jit, donate_argnums=(0,))
    def f(kv, q, kv_lens, pidx, cu):
        out, lse = V2.decode_cache_pass(
            q, kv, kv_lens, pidx, cu, jnp.array([0], jnp.int32), sm_scale=1 / 16, k_scale=1.0, v_scale=1.0,
            cp_group_size=DCP, decode_query_size=T, bkv=bkv, group=group, depth=depth, issue_mode=issue,
            pv_mode=pv, chunk=chunk, dbg=dbg)
        return kv, out, lse
    return f


def parse_cfg(cfg):
    """v2:<group>:<depth>:<bkv>:<issue>[:dbg+dbg]"""
    parts = cfg.split(":")
    toks = parts[5].split("+") if len(parts) > 5 else []
    pv = "fp8x2" if "pv8x2" in toks else "bf16"
    chunk = int(next((t[1:] for t in toks if t.startswith("c") and t[1:].isdigit()), 4096))
    dbg = ",".join(t for t in toks if t != "pv8x2" and not (t.startswith("c") and t[1:].isdigit()))
    return v2_fn(int(parts[3]), int(parts[1]), int(parts[2]), parts[4], dbg, pv, chunk)


def carry(out, args):
    return (out[0],) + tuple(args[1:])


NS = [int(x) for x in os.environ.get("BV_N", "64,128").split(",")]
CFGS = os.environ.get("BV_CFGS", ",".join([
    "old",
    "v2:8:2:4096:runs", "v2:8:2:4096:pages", "v2:8:2:4096:loop", "v2:4:2:4096:runs",
    "v2:8:2:4096:runs:nodma", "v2:8:2:4096:runs:nocompute",
])).split(",")
for n in NS:
    kv, q, kv_lens, pidx, cu, nbytes = build(n)
    outs = {}
    for cfg in CFGS:
        try:
            if cfg == "old":
                f, pat = old_fn, r"^RPAd-"
            else:
                f, pat = parse_cfg(cfg), r"^RPAd2"
            args = (jnp.copy(kv), q, kv_lens, pidx, cu)
            t, out, _ = devtime.device_times(f, args, iters=10, carry=carry)
            us, cnt = devtime.summarize(t, pat)
            other = sum(sum(v) for k, v in t.items() if not k.startswith("RPAd")) / 10
            top_other = sorted(((k, round(sum(v) / 10, 1)) for k, v in t.items() if not k.startswith("RPAd")),
                               key=lambda r: -r[1])[:6]
            o = np.asarray(out[1].astype(jnp.float32))
            l = np.asarray(out[2])
            outs[cfg] = (o, l)
            rec = dict(kind="bench", n=n, cfg=cfg, us=round(us, 1), cnt=cnt, GBps=round(nbytes / us / 1e3, 1),
                       pct_of_3700=round(nbytes / us / 1e3 / 3700 * 100, 1), other_us_per_call=round(other, 1), top_other=top_other,
                       nan=bool(np.isnan(o).any()))
            if "old" in outs and cfg != "old" and ":no" not in cfg:
                o0, l0 = outs["old"]
                rec["max_abs_vs_old_o"] = float(np.abs(o - o0).max())
                rec["max_abs_vs_old_lse"] = float(np.abs(l - l0).max())
            emit(**rec)
        except Exception as e:  # noqa: BLE001
            emit(kind="bench", n=n, cfg=cfg, error=f"{type(e).__name__}: {str(e)[:1500]}")
print("PROBE_DONE", flush=True)
