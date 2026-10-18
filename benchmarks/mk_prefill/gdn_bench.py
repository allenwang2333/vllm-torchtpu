"""GDN prefill bench for the Qwen3.5 prefill megakernel at a TP1 pipeline stage.

One sequence of Q new tokens with an initial state (a cached context), all 16 k/q heads and 64
v heads (head dims 128), conv kernel 4: one GatedDeltaNet layer's fused_conv1d_gdn call on a
PP8 x TP1 stage, with dense f32 states. `sweep` runs each arm in its own process; an arm is
`name:key=value,...` with keys prec (f32|bf16), tile (mixed_tile_size) and env_<VAR>=<value>.
Arms are compared with the first arm's output; xprof is captured per arm.

    python3 gdn_bench.py sweep --q 8192 --arms "default:" "bf16:prec=bf16"
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
NKQ, NV, DK, DV, KS = 16, 64, 128, 128, 4
DIM = 2 * NKQ * DK + NV * DV
# The prefill leg's GDN env (j-5dc80dd3-ab82-4334-986b).
SERVING_ENV = {
    "TPU_ENABLE_GDN_DYNAMIC_TILING": "1",
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": "1",
    "TPU_GDN_CONV_STATE_TILE_PAD": "1",
    "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_v3_pd",
}


def parse_arm(spec):
    name, _, kv = spec.partition(":")
    opts = {}
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        opts[k] = v
    return name, opts


def serving_env(opts):
    for k, v in SERVING_ENV.items():
        os.environ[k] = v
    for k, v in opts.items():
        if k.startswith("env_"):
            os.environ[k[4:]] = v
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.path.insert(0, os.path.join(CHECKOUT, "src"))
    runpy.run_path(os.path.join(CHECKOUT, "src", "vllm_torchtpu", "env_override.py"))


def run_arm(args):
    name, opts = parse_arm(args.arm)
    serving_env(opts)
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    rng = np.random.default_rng(args.seed)
    S, T = args.num_seqs, args.q
    dev = jax.devices()[0]
    put = lambda x: jax.device_put(x, dev)
    qkv = put(jnp.asarray(rng.standard_normal((T, DIM), np.float32), jnp.bfloat16))
    b = put(jnp.asarray(rng.standard_normal((T, NV), np.float32), jnp.bfloat16))
    a = put(jnp.asarray(rng.standard_normal((T, NV), np.float32), jnp.bfloat16))
    conv_state = put(jnp.asarray(0.1 * rng.standard_normal((S + 1, KS - 1, DIM), np.float32), jnp.bfloat16))
    rec_state = put(jnp.asarray(0.01 * rng.standard_normal((S + 1, NV, DK, DV), np.float32)))
    conv_w = put(jnp.asarray(0.3 * rng.standard_normal((DIM, 1, KS), np.float32), jnp.bfloat16))
    a_log = put(jnp.asarray(np.log(rng.uniform(1, 16, (NV,))), jnp.float32))
    dt_bias = put(jnp.asarray(rng.standard_normal((NV,)), jnp.float32))
    qsl = np.full((S + 1,), T, np.int32); qsl[0] = 0
    seq_lens = np.zeros((S,), np.int32); seq_lens[0] = args.ctx + T
    qsl, state_idx = put(qsl), put(np.arange(S, dtype=np.int32))
    dist, seq_lens = put(np.array([0, 0, 1], np.int32)), put(seq_lens)
    zeros = put(np.zeros((64,), np.int32))
    kw = {}
    if opts.get("prec") == "bf16":
        kw["compute_precision"] = jnp.dtype(jnp.bfloat16)
    if "tile" in opts:
        kw["mixed_tile_size"] = int(opts["tile"])

    def call(conv, rec, lens):
        return wrapper.fused_conv1d_gdn(
            qkv, b, a, conv, rec, conv_w, None, a_log, dt_bias, qsl, state_idx, dist, lens,
            n_kq=NKQ, n_v=NV, d_k=DK, d_v=DV, kernel_size=KS, **kw)

    once = jax.jit(call)
    if args.mode == "profile":
        jax.block_until_ready(once(conv_state, rec_state, seq_lens))
        d = os.path.join(args.xprof_dir, name)
        os.makedirs(d, exist_ok=True)
        with jax.profiler.trace(d):
            for _ in range(3):
                jax.block_until_ready(once(conv_state, rec_state, seq_lens))
        print("PROFILE_DONE " + d, flush=True)
        return

    def loop(n):
        @jax.jit
        def f(conv, rec, zeros):
            def body(i, carry):
                conv, rec, acc = carry
                (conv, rec), out = call(conv, rec, seq_lens + zeros[i % 64])
                return conv, rec, acc + out[0, 0].astype(jnp.float32)
            return jax.lax.fori_loop(0, n, body, (conv, rec, jnp.float32(0)))
        return f

    res = {"arm": name, "opts": opts, "ctx": args.ctx, "q": T}
    try:
        t0 = time.time()
        (_, rec_out), out = once(conv_state, rec_state, seq_lens)
        out = np.asarray(jax.device_get(out)).astype(np.float32)
        rec_out = np.asarray(jax.device_get(rec_out[0])).astype(np.float32)
        res["compile_s"] = round(time.time() - t0, 1)
        res["finite"] = bool(np.isfinite(out).all() and np.isfinite(rec_out).all())
        res["checksum"] = float(np.abs(out).sum())
        ref_path = os.path.join(args.workdir, "gdn_ref.npz")
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            res["out_rel_l2"] = float(np.linalg.norm(out - ref["out"]) / np.linalg.norm(ref["out"]))
            res["state_rel_l2"] = float(np.linalg.norm(rec_out - ref["rec"]) / np.linalg.norm(ref["rec"]))
        else:
            np.savez(ref_path, out=out, rec=rec_out)
            res["reference"] = True
        times = {}
        for n in (args.k1, args.k2):
            f = loop(n)
            jax.block_until_ready(f(conv_state, rec_state, zeros))
            best = 1e9
            for _ in range(args.reps):
                t = time.perf_counter()
                jax.block_until_ready(f(conv_state, rec_state, zeros))
                best = min(best, time.perf_counter() - t)
            times[n] = best
        res["ms_per_call"] = round((times[args.k2] - times[args.k1]) / (args.k2 - args.k1) * 1e3, 3)
        res["ratio"] = round(times[args.k2] / times[args.k1], 2)
        res["ok"] = res["finite"] and res.get("out_rel_l2", 0.0) < args.tol
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["ok"] = False
    print("ARM_RESULT " + json.dumps(res), flush=True)


def sweep(args):
    results = []
    ref = os.path.join(args.workdir, "gdn_ref.npz")
    if os.path.exists(ref):
        os.remove(ref)
    for spec in args.arms:
        cmd = [sys.executable, os.path.abspath(__file__), "arm", "--arm", spec]
        for k in ("ctx", "q", "num_seqs", "seed", "k1", "k2", "reps", "tol", "workdir", "xprof_dir"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        print(f"[gdn_bench] {spec}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("ARM_RESULT ")]
        r = json.loads(line[-1][11:]) if line else {"arm": spec, "ok": False,
                                                       "error": (p.stderr or p.stdout)[-1500:]}
        if args.xprof and r.get("ms_per_call"):
            pc = [c if c != "arm" else "profile" for c in cmd]
            pp = subprocess.run(pc, capture_output=True, text=True)
            done = [l for l in pp.stdout.splitlines() if l.startswith("PROFILE_DONE ")]
            r["xprof"] = done[-1][13:] if done else f"failed rc={pp.returncode}: {(pp.stderr or '')[-300:]}"
        print(f"[gdn_bench] {json.dumps(r)}", flush=True)
        results.append(r)
    out = os.environ.get("CDK_OUTPUT_DIR", args.workdir)
    with open(os.path.join(out, "gdn_bench.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n| arm | ok | ms/call | ratio | out rel_l2 | state rel_l2 | error |\n|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['arm']} | {r.get('ok')} | {r.get('ms_per_call', '')} | {r.get('ratio', '')} | "
              f"{r.get('out_rel_l2', '')} | {r.get('state_rel_l2', '')} | {r.get('error', '')[:120]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "arm", "profile"])
    ap.add_argument("--arms", nargs="*", default=["default:"])
    ap.add_argument("--arm", default="default:")
    ap.add_argument("--ctx", type=int, default=87040)
    ap.add_argument("--q", type=int, default=8192)
    ap.add_argument("--num-seqs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k1", type=int, default=2)
    ap.add_argument("--k2", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--workdir", default="/tmp")
    ap.add_argument("--xprof", type=int, default=1)
    ap.add_argument("--xprof-dir", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "xprof"))
    args = ap.parse_args()
    (sweep if args.mode == "sweep" else run_arm)(args)


if __name__ == "__main__":
    main()
