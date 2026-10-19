"""Integration check of the DCP CACHE_ONLY entry (`cp_attention._pallas_rpa_kernel_dcp`)
with use_decode_v2=True against use_decode_v2=False, under jit with the KV
cache donated as in the engine's jax_op.

- all-decode step: the v2 branch runs; outputs close to the batched RPA ones;
- step with a MIXED sequence: the batched RPA branch runs; outputs identical;
- no large copy in the device profile (the cache must stay in place through
  the lax.cond).

Prints CPTEST {json} lines.
"""

import functools
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devtime  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import configs as C  # noqa: E402
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as W  # noqa: E402
from vllm_torchtpu.layers.adapter import cp_attention as CP  # noqa: E402

PAGE, D, T, DCP, QH = 128, 256, 4, 4, 16
N = int(os.environ.get("CP_N", "64"))
CTX = 80000


def emit(**kw):
    print("CPTEST " + json.dumps(kw), flush=True)


def build(n, seed=0):
    local = -(-CTX // DCP)
    per = -(-local // PAGE) + 1
    num_pages = n * per + 64
    shape = W.get_kv_cache_shape(num_pages, PAGE, 1, D, jnp.float8_e4m3fn, kv_layout=C.KVLayout.SEQ_ALONG_LANE)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    kv = (jax.random.normal(k1, shape, jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    tokens = n * T
    q = (jax.random.normal(k2, (tokens, QH, D), jnp.float32) * 2.0).astype(jnp.bfloat16)
    knew = (jax.random.normal(k3, (tokens, 1, D), jnp.float32) * 0.5).astype(jnp.float8_e4m3fn)
    rng = np.random.default_rng(seed)
    lens = (CTX + rng.integers(-2000, 2000, n)).astype(np.int32)
    pages = np.zeros((n, 512), np.int32)
    perm = rng.permutation(np.arange(1, num_pages))
    for s in range(n):
        pages[s, :per] = perm[s * per:(s + 1) * per]
    cu = np.arange(n + 1, dtype=np.int32) * T
    return kv, q, knew, jnp.asarray(lens + T), jnp.asarray(pages.reshape(-1)), jnp.asarray(cu)


def entry(use_v2, rank):
    fn = functools.partial(
        CP._pallas_rpa_kernel_dcp,
        q_scale=None, k_scale=1.0, v_scale=1.0,
        sliding_window=None, sm_scale=1 / 16, soft_cap=None,
        cp_group_size=DCP, cp_rank_val=rank,
        attention_scope=C.AttentionScope.CACHE_ONLY,
        kv_layout=C.KVLayout.SEQ_ALONG_LANE, decode_query_size=T,
        use_decode_v2=use_v2)

    @functools.partial(jax.jit, donate_argnums=(0,))
    def f(kv, q, knew, lens, pidx, cu, dist):
        new_kv, out, lse = fn(kv, q, knew, knew, lens, pidx, cu, dist)
        return new_kv, out, lse.astype(jnp.float32)

    return f


def carry(out, args):
    return (out[0],) + tuple(args[1:])


# The engine builds the ops through build_dcp_kernels; with the flag set it
# must log once that the v2 path is used.
os.environ["TPU_RPA_DECODE_V2"] = "1"
_ops = CP.build_dcp_kernels(
    sliding_window=None, sm_scale=1 / 16, logits_soft_cap=None, q_scale=None, k_scale=1.0,
    v_scale=1.0, cp_group_size=DCP, cp_rank=0, kv_layout=C.KVLayout.SEQ_ALONG_LANE,
    decode_query_size=T)
emit(built_ops=[type(o).__name__ for o in _ops], registry=[k[0] for k in CP._DCP_KERNEL_REGISTRY])

kv, q, knew, lens, pidx, cu = build(N)
for rank in (() if os.environ.get("CP_MULTI_ONLY") == "1" else (0, 3)):
    for name, dist in (("decode", jnp.array([N, N, N], jnp.int32)),
                       ("mixed", jnp.array([N - 1, N - 1, N], jnp.int32))):
        res = {}
        for use_v2 in (False, True):
            args = (jnp.copy(kv), q, knew, lens, pidx, cu, dist)
            t, out, _ = devtime.device_times(entry(use_v2, rank), args, iters=3, carry=carry)
            big = sorted(((k, round(max(v), 1)) for k, v in t.items() if max(v) > 50.0), key=lambda r: -r[1])
            res[use_v2] = (np.asarray(out[1].astype(jnp.float32)), np.asarray(out[2]), big)
        o0, l0, big0 = res[False]
        o1, l1, big1 = res[True]
        if name == "mixed":
            # Only sequences below distribution[0] are in DECODE; compare those rows.
            rows = slice(0, (N - 1) * T)
        else:
            rows = slice(0, N * T)
        fin = np.isfinite(l0[rows])
        emit(rank=rank, step=name,
             max_abs_o=float(np.abs(o1[rows] - o0[rows]).max()),
             max_abs_lse=float(np.abs(l1[rows] - l0[rows])[fin].max()) if fin.any() else 0.0,
             identical=bool(np.array_equal(o1[rows], o0[rows])),
             nan_v2=bool(np.isnan(o1[rows]).any()),
             ops_over_50us_old=big0, ops_over_50us_v2=big1)
print("CPTEST_DONE", flush=True)


# Four "layers" in one program (same metadata, different queries), as in the
# engine's forward: device time per layer including all glue, old vs v2.
def multi(use_v2, rank, layers=4):
    one = entry(use_v2, rank).__wrapped__ if hasattr(entry(use_v2, rank), "__wrapped__") else None
    fn = functools.partial(
        CP._pallas_rpa_kernel_dcp,
        q_scale=None, k_scale=1.0, v_scale=1.0,
        sliding_window=None, sm_scale=1 / 16, soft_cap=None,
        cp_group_size=DCP, cp_rank_val=rank,
        attention_scope=C.AttentionScope.CACHE_ONLY,
        kv_layout=C.KVLayout.SEQ_ALONG_LANE, decode_query_size=T,
        use_decode_v2=use_v2)
    del one

    @functools.partial(jax.jit, donate_argnums=(0,))
    def f(kv, q, knew, lens, pidx, cu, dist):
        outs = []
        for i in range(layers):
            kv, out, lse = fn(kv, q * (1.0 + 0.01 * i), knew, knew, lens, pidx, cu, dist)
            lf = lse.astype(jnp.float32)
            outs.append(out.astype(jnp.float32).sum() + jnp.where(jnp.isfinite(lf), lf, 0.0).sum())
        return kv, jnp.stack(outs)

    return f


dist = jnp.array([N, N, N], jnp.int32)
for use_v2, sel in ((False, "1"), (True, "0"), (True, "1")):
    os.environ["TPU_RPA_DECODE_V2_SELECT"] = sel
    jax.clear_caches()
    args = (jnp.copy(kv), q, knew, lens, pidx, cu, dist)
    t, out, _ = devtime.device_times(multi(use_v2, 0), args, iters=3, carry=carry)
    total = sum(sum(v) for v in t.values()) / 3
    nested = sum(sum(v) for k, v in t.items() if k.startswith("cond") or k.startswith("conditional")) / 3
    top = sorted(((k, round(sum(v) / 3, 1), len(v) // 3) for k, v in t.items()), key=lambda r: -r[1])[:12]
    emit(mode="4layers", use_v2=use_v2, select=sel, us_per_layer_total=round(total / 4, 1),
         us_per_layer_excl_cond_wrappers=round((total - nested) / 4, 1), top=top)
print("CPTEST_MULTI_DONE", flush=True)
