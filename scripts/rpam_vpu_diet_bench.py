"""RPAm (batched_rpa MIXED) microbench: old vs VPU-diet kernel, + block sweep.

Qwen3.5-397B TP8 per-device geometry: 4 q heads, 1 KV head, head_dim 256,
fp8 e4m3 SEQ_ALONG_LANE (HND) cache, page 128, sm_scale 1/16, k_scale 1.0.
One sequence per call: `ctx` cached tokens + a `T`-token prefill chunk, so the
wrapper runs only the (cond) MIXED kernel plus the shared schedule kernel.

Run on ONE TPU chip (self-contained; loads batched_rpa from $REPO, no install):
  REPO=/path/to/rpam_bench/new_src python3 rpam_vpu_diet_bench.py [--quick]
  REPO=/path/to/rpam_bench/old_src python3 rpam_vpu_diet_bench.py --variants old --quick
Writes /tmp/rpam_bench_<tag>.json and prints a table.
"""
import argparse
import json
import os
import statistics
import sys
import time
import types

REPO = os.environ["REPO"]
for _name, _sub in [("vllm_torchtpu", ""), ("vllm_torchtpu.kernels", "kernels"),
                    ("vllm_torchtpu.kernels.experimental", "kernels/experimental")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, "vllm_torchtpu", _sub)]
    sys.modules[_name] = _m

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

P = argparse.ArgumentParser()
P.add_argument("--quick", action="store_true", help="fewer iters, no sweep on T=512")
P.add_argument("--variants", default="old,new,new_k2048_c64,new_k1024_c128,new_k2048_c128")
P.add_argument("--T", default="512,2048,8192")
P.add_argument("--ctx", default="0,32768,100000")
P.add_argument("--iters", type=int, default=5)
P.add_argument("--reps", type=int, default=20, help="chained calls per timing")
P.add_argument("--k_scale", type=float, default=1.0)
P.add_argument("--tag", default="")
A = P.parse_args()

PAGE, NQ, NKV, D = 128, 4, 1, 256
KV_DT = jnp.float8_e4m3fn
LAYOUT = configs.KVLayout.SEQ_ALONG_LANE
MAX_SEQS = 16
PAGES_PER_SEQ = 262144 // PAGE
Ts = [int(x) for x in A.T.split(",")]
CTXs = [int(x) for x in A.ctx.split(",")]
NUM_PAGES = (max(CTXs) + max(Ts)) // PAGE + 8
ITERS = 3 if A.quick else A.iters
# variant -> (TPU_RPA_VPU_DIET, bkv, bq_c) ; bkv None = calculate_block_sizes default.
VARIANTS = {
    "old": ("0", None, None),
    "new": ("1", None, None),
    "old_k2048_c64": ("0", 2048, 64),
    "new_k2048_c64": ("1", 2048, 64),
    "new_k1024_c128": ("1", 1024, 128),
    "new_k2048_c128": ("1", 2048, 128),
}
dev = jax.devices()[0]
print("device:", dev.device_kind, "jax", jax.__version__, "REPO", REPO, flush=True)

cache_shape = wrapper.get_kv_cache_shape(NUM_PAGES, PAGE, NKV, D, KV_DT, kv_layout=LAYOUT)
key = jax.random.PRNGKey(0)
k0, k1, k2, k3, k4 = jax.random.split(key, 5)
base_cache = (jax.random.normal(k0, cache_shape, jnp.float32) * 0.5).astype(KV_DT)
page_indices = jnp.zeros((MAX_SEQS * PAGES_PER_SEQ,), jnp.int32)
page_indices = page_indices.at[:NUM_PAGES].set(jnp.arange(NUM_PAGES, dtype=jnp.int32))


def inputs(T, ctx):
    q = (jax.random.normal(k1, (T, NQ, D), jnp.float32) * 0.5).astype(jnp.bfloat16)
    k = (jax.random.normal(k2, (T, NKV, D), jnp.float32) * 0.5).astype(KV_DT)
    v = (jax.random.normal(k3, (T, NKV, D), jnp.float32) * 0.5).astype(KV_DT)
    kv_lens = jnp.zeros((MAX_SEQS,), jnp.int32).at[0].set(ctx + T)
    cu = jnp.full((MAX_SEQS + 1,), T, jnp.int32).at[0].set(0)
    dist = jnp.array([0, 0, 1], jnp.int32)
    return q, k, v, kv_lens, cu, dist


def run(T, ctx, blocks, cache, args=None):
    q, k, v, kv_lens, cu, dist = args if args is not None else inputs(T, ctx)
    return wrapper.ragged_paged_attention(
        q, k, v, cache, kv_lens, page_indices, cu, dist,
        sm_scale=D ** -0.5, k_scale=A.k_scale, v_scale=1.0, kv_layout=LAYOUT,
        decode_query_size=4, mask_value=-1e30, prefill_block_sizes=blocks)


def reference_rows(T, ctx, cache, rows):
    """f32 reference for selected query rows, K/V read back from the cache."""
    q = inputs(T, ctx)[0].astype(jnp.float32)[rows]  # [n, NQ, D]
    n_tok = ctx + T
    npg = -(-n_tok // PAGE)
    c = cache[:npg].astype(jnp.float32)  # [npg, 2, D/4, 4, PAGE]
    kk = c[:, 0].reshape(npg, D, PAGE).transpose(0, 2, 1).reshape(-1, D)[:n_tok] * A.k_scale
    vv = c[:, 1].reshape(npg, D, PAGE).transpose(0, 2, 1).reshape(-1, D)[:n_tok]
    s = jnp.einsum("nhd,td->nht", q, kk, precision="highest") * D ** -0.5
    pos = ctx + jnp.asarray(rows)
    s = jnp.where(jnp.arange(n_tok)[None, None, :] <= pos[:, None, None], s, -jnp.inf)
    w = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("nht,td->nhd", w, vv, precision="highest")


results = []
old_out = {}
for vname in A.variants.split(","):
    diet, bkv, bqc = VARIANTS[vname]
    os.environ["TPU_RPA_VPU_DIET"] = diet
    jax.clear_caches()
    for T in Ts:
        if A.quick and bkv is not None and T == 512:
            continue
        blocks = None
        if bkv is not None:
            bq = min(1024, -(-T // 256) * 256)
            blocks = configs.BlockSizes(bq_sz=bq, bq_c_sz=bqc, bkv_sz=bkv,
                                        batch_size=2, n_buffer=3)
        for ctx in CTXs:
            rec = dict(variant=vname, T=T, ctx=ctx, diet=diet, bkv=bkv, bq_c=bqc)
            try:
                cache = jnp.copy(base_cache)
                t0 = time.perf_counter()
                out, cache = jax.block_until_ready(run(T, ctx, blocks, cache))
                rec["first_call_s"] = round(time.perf_counter() - t0, 2)
                # Dispatch REPS chained calls (the cache is donated and carried)
                # before blocking, so host dispatch overlaps device work and the
                # per-call time is ~device time.
                args = inputs(T, ctx)
                times = []
                for it in range(ITERS + 1):
                    t0 = time.perf_counter()
                    for _ in range(A.reps):
                        out, cache = run(T, ctx, blocks, cache, args)
                    jax.block_until_ready((out, cache))
                    if it:
                        times.append((time.perf_counter() - t0) / A.reps)
                rec["ms_median"] = round(1e3 * statistics.median(times), 3)
                rec["ms_min"] = round(1e3 * min(times), 3)
                o = np.asarray(out.astype(jnp.float32))
                rec["finite"] = bool(np.isfinite(o).all())
                rows = sorted(set(list(range(min(32, T))) + list(range(max(0, T - 32), T))))
                ref = np.asarray(reference_rows(T, ctx, cache, np.asarray(rows)))
                rec["max_abs_err_vs_f32_ref"] = float(np.abs(o[rows] - ref).max())
                if (T, ctx) in old_out:
                    rec["max_abs_diff_vs_old"] = float(np.abs(o - old_out[(T, ctx)]).max())
                elif vname.startswith("old") and bkv is None:
                    old_out[(T, ctx)] = o
            except Exception as e:  # VMEM OOM / compile failures are results too.
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(json.dumps(rec), flush=True)
            results.append(rec)

print("\n%-16s %5s %7s %10s %10s %12s %12s" % (
    "variant", "T", "ctx", "ms_med", "speedup", "err_vs_ref", "diff_vs_old"))
base = {(r["T"], r["ctx"]): r.get("ms_median") for r in results if r["variant"] == "old"}
for r in results:
    b = base.get((r["T"], r["ctx"]))
    sp = (b / r["ms_median"]) if b and r.get("ms_median") else float("nan")
    print("%-16s %5d %7d %10s %10.3f %12s %12s" % (
        r["variant"], r["T"], r["ctx"], r.get("ms_median", r.get("error", "")[:10]), sp,
        "%.2e" % r["max_abs_err_vs_f32_ref"] if "max_abs_err_vs_f32_ref" in r else "-",
        "%.2e" % r["max_abs_diff_vs_old"] if "max_abs_diff_vs_old" in r else "-"))
path = f"/tmp/rpam_bench{('_' + A.tag) if A.tag else ''}.json"
with open(path, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", path)
