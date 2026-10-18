"""Prefill attention bench for the Qwen3.5 prefill megakernel at a TP1 pipeline stage.

The production call (RPAm-snh): one sequence of Q new tokens behind a CTX-token cached context,
32 q heads / 2 kv heads / head_dim 256, fp8 KV cache in SEQ_ALONG_LANE pages of 128 tokens,
bf16 q, causal. `sweep` runs every arm in its own process on one clock and writes
attn_bench.json; an arm is `name:key=value,...` (keys: bq, bqc, bkv, batch, nbuf, vmem_mb, impl).

    python3 attn_bench.py sweep --ctx 87040 --q 8192 --arms "default:" "b512:bq=512,bqc=512,bkv=512"
"""

import argparse
import json
import os
import runpy
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.environ.get("MK_CHECKOUT", os.path.abspath(os.path.join(HERE, "..", "..")))
NQ, NKV, HD, PAGE = 32, 2, 256, 128


def serving_env():
    # vllm-torchtpu's libtpu flags and Pallas lowering patches (2,682 us -> 162 us for batched_rpa
    # without them); must run before jax is imported.
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.path.insert(0, os.path.join(CHECKOUT, "src"))
    runpy.run_path(os.path.join(CHECKOUT, "src", "vllm_torchtpu", "env_override.py"))


def parse_arm(spec):
    name, _, kv = spec.partition(":")
    opts = {}
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        opts[k] = v
    return name, opts


def build_inputs(ctx, q_len, num_seqs, pages_per_seq, seed):
    import ml_dtypes
    import numpy as np

    rng = np.random.default_rng(seed)
    total = ctx + q_len
    k_all = rng.standard_normal((total, NKV, HD), dtype=np.float32)
    v_all = rng.standard_normal((total, NKV, HD), dtype=np.float32)
    q = rng.standard_normal((q_len, NQ, HD), dtype=np.float32)
    fp8 = ml_dtypes.float8_e4m3fn
    k8, v8 = k_all.astype(fp8), v_all.astype(fp8)
    # SEQ_ALONG_LANE page: (2*NKV [K0 V0 K1 V1], HD/4, 4, PAGE), as prepare_inputs lays new tokens.
    n_pages = -(-total // PAGE) + 1
    kv = np.zeros((n_pages * PAGE, 2 * NKV, HD), dtype=fp8)
    inter = np.stack([k8[:ctx], v8[:ctx]], axis=2).reshape(ctx, 2 * NKV, HD)
    kv[:ctx] = inter
    cache = kv.reshape(n_pages, PAGE, 2 * NKV, HD // 4, 4).transpose(0, 2, 3, 4, 1).copy()
    page_indices = np.zeros((num_seqs * pages_per_seq,), np.int32)
    page_indices[:n_pages] = np.arange(n_pages, dtype=np.int32)
    kv_lens = np.zeros((num_seqs,), np.int32)
    kv_lens[0] = total
    cu = np.full((num_seqs + 1,), q_len, np.int32)
    cu[0] = 0
    dist = np.array([0, 0, 1], np.int32)
    return dict(q=q.astype(ml_dtypes.bfloat16), k=k_all[ctx:].astype(ml_dtypes.bfloat16),
                v=v_all[ctx:].astype(ml_dtypes.bfloat16), cache=cache, kv_lens=kv_lens,
                page_indices=page_indices, cu=cu, dist=dist, k8=k8, v8=v8)


def reference_rows(inp, ctx, rows):
    """fp32 attention for the given query rows, every head, against the fp8 K/V the cache holds."""
    import numpy as np

    k = inp["k8"].astype(np.float32)
    v = inp["v8"].astype(np.float32)
    q = inp["q"].astype(np.float32)
    out = np.zeros((len(rows), NQ, HD), np.float32)
    for i, r in enumerate(rows):
        n = ctx + r + 1
        for h in range(NQ):
            g = h // (NQ // NKV)
            s = (k[:n, g] @ q[r, h]) / np.sqrt(HD)
            p = np.exp(s - s.max())
            out[i, h] = (p / p.sum()) @ v[:n, g]
    return out


def run_arm(args):
    serving_env()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    name, opts = parse_arm(args.arm)
    inp = build_inputs(args.ctx, args.q, args.num_seqs, args.pages_per_seq, args.seed)
    kw = {}
    if "bq" in opts:
        kw["prefill_block_sizes"] = configs.BlockSizes(
            bq_sz=int(opts["bq"]), bq_c_sz=int(opts.get("bqc", opts["bq"])),
            bkv_sz=int(opts["bkv"]), batch_size=int(opts.get("batch", 1)),
            n_buffer=int(opts.get("nbuf", 3)))
    if "vmem_mb" in opts:
        kw["vmem_limit_bytes"] = int(opts["vmem_mb"]) << 20
    layout = configs.KVLayout.SEQ_ALONG_LANE

    def call(q, k, v, cache, kv_lens, pi, cu, dist):
        return wrapper.ragged_paged_attention(
            q, k, v, cache, kv_lens, pi, cu, dist, sm_scale=HD ** -0.5, kv_layout=layout, **kw)

    dev = jax.devices()[0]
    put = lambda x: jax.device_put(x, dev)
    q, k, v = put(inp["q"]), put(inp["k"]), put(inp["v"])
    kv_lens, pi, cu, dist = put(inp["kv_lens"]), put(inp["page_indices"]), put(inp["cu"]), put(inp["dist"])
    zeros = put(np.zeros((64,), np.int32))

    @jax.jit
    def once(cache):
        return call(q, k, v, cache, kv_lens, pi, cu, dist)

    def loop(n):
        @jax.jit
        def f(cache, zeros):
            def body(i, carry):
                cache, acc = carry
                o, cache = call(q, k, v, cache, kv_lens + zeros[i % 64], pi, cu, dist)
                return (cache, acc + o[0, 0, 0].astype(jnp.float32))
            return jax.lax.fori_loop(0, n, body, (cache, jnp.float32(0)))
        return f

    res = {"arm": name, "opts": opts, "ctx": args.ctx, "q": args.q}
    try:
        t0 = time.time()
        out, _ = once(put(inp["cache"]))
        out = np.asarray(jax.device_get(out)).astype(np.float32)
        res["compile_s"] = round(time.time() - t0, 1)
        rows = [0, args.q // 3, args.q // 2, args.q - 2, args.q - 1]
        ref = reference_rows(inp, args.ctx, rows)
        got = out[rows]
        err = np.linalg.norm(got - ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-6)
        res["rel_l2_max"] = float(err.max())
        res["checksum"] = float(np.abs(out).sum())
        res["finite"] = bool(np.isfinite(out).all())
        np.save(os.path.join(args.workdir, f"out_{name}.npy"), out[:: max(1, args.q // 512)])
        times = {}
        for n in (args.k1, args.k2):
            f = loop(n)
            jax.block_until_ready(f(put(inp["cache"]), zeros))  # compile + warm
            best = 1e9
            for _ in range(args.reps):
                c = put(inp["cache"])
                jax.block_until_ready(c)
                t = time.perf_counter()
                jax.block_until_ready(f(c, zeros))
                best = min(best, time.perf_counter() - t)
            times[n] = best
        res["t_k1_ms"], res["t_k2_ms"] = times[args.k1] * 1e3, times[args.k2] * 1e3
        res["ms_per_call"] = round((times[args.k2] - times[args.k1]) / (args.k2 - args.k1) * 1e3, 3)
        res["ratio"] = round(times[args.k2] / times[args.k1], 2)
        pairs = NQ * (args.ctx * args.q + args.q * (args.q + 1) / 2)
        res["tflops"] = round(4 * HD * pairs / (res["ms_per_call"] * 1e-3) / 1e12, 1)
        res["ok"] = res["finite"] and res["rel_l2_max"] < args.tol
    except Exception as e:  # noqa: BLE001 - a failed arm is a result
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["ok"] = False
    print("ARM_RESULT " + json.dumps(res), flush=True)


def sweep(args):
    results = []
    for spec in args.arms:
        cmd = [sys.executable, os.path.abspath(__file__), "arm", "--arm", spec]
        for k in ("ctx", "q", "num_seqs", "pages_per_seq", "seed", "k1", "k2", "reps", "tol", "workdir"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        print(f"[attn_bench] {spec}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("ARM_RESULT ")]
        r = json.loads(line[-1][11:]) if line else {"arm": spec, "ok": False,
                                                       "error": (p.stderr or p.stdout)[-1500:]}
        print(f"[attn_bench] {json.dumps(r)}", flush=True)
        results.append(r)
    out = os.environ.get("CDK_OUTPUT_DIR", args.workdir)
    with open(os.path.join(out, "attn_bench.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n| arm | ok | ms/call | TFLOP/s | ratio | rel_l2 | compile s | error |\n|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['arm']} | {r.get('ok')} | {r.get('ms_per_call', '')} | {r.get('tflops', '')} | "
              f"{r.get('ratio', '')} | {r.get('rel_l2_max', '')} | {r.get('compile_s', '')} | "
              f"{r.get('error', '')[:120]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "arm"])
    ap.add_argument("--arms", nargs="*", default=["default:"])
    ap.add_argument("--arm", default="default:")
    ap.add_argument("--ctx", type=int, default=87040)
    ap.add_argument("--q", type=int, default=8192)
    ap.add_argument("--num-seqs", type=int, default=16)
    ap.add_argument("--pages-per-seq", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k1", type=int, default=2)
    ap.add_argument("--k2", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--workdir", default="/tmp")
    args = ap.parse_args()
    (sweep if args.mode == "sweep" else run_arm)(args)


if __name__ == "__main__":
    main()
