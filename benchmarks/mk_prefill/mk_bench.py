"""Qwen3.5 prefill megakernel bench: one pipeline stage's whole step in one pallas_call, on TPU.

    check: the megakernel vs the pure-JAX reference (ragged_dot MoE) at full model dimensions on
           a short sequence, judged against the reference's own noise floor (1% of the inputs
           moved by one bf16 ulp), plus per-layer state errors.
    time:  wall time of the stage step (best of --reps after a warm call), then one xprof
           capture of a step in its own process (--xprof).

    python3 mk_bench.py check --layers gdn,attn --t 1024 --ctx 4096
    python3 mk_bench.py time --layers gdn,gdn,gdn,attn,gdn,gdn,gdn,attn --t 8192 --ctx 87040
"""

import argparse
import dataclasses
import json
import os
import runpy
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.environ.get("MK_CHECKOUT", os.path.abspath(os.path.join(HERE, "..", "..")))


def serving_env():
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.path.insert(0, os.path.join(CHECKOUT, "src"))
    runpy.run_path(os.path.join(CHECKOUT, "src", "vllm_torchtpu", "env_override.py"))


def tiles_from(args, MK):
    kw = {}
    for item in filter(None, args.tiles.split(",")):
        k, _, v = item.partition("=")
        kw[k] = int(v)
    return MK.MKTiles(**kw)


def setup(args):
    serving_env()
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.qwen35_prefill_mk import megakernel as MK
    from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R

    c = R.StageConfig(layer_types=tuple(args.layers.split(",")))
    s_max = args.ctx + args.t
    weights, states, h, r = jax.jit(
        lambda k: R.random_stage_device(c, args.t, args.ctx, k), static_argnums=())(
        jax.random.PRNGKey(args.seed))
    kw = [MK.kernel_weight_layout(c, w, lt) for w, lt in zip(weights, c.layer_types)]
    ks = [dict(st) if lt == "gdn" else {"k": jnp.swapaxes(st["k"], 0, 1),
                                        "v": jnp.swapaxes(st["v"], 0, 1)}
          for st, lt in zip(states, c.layer_types)]
    tiles = tiles_from(args, MK)
    step = jax.jit(MK.build(c, args.t, s_max, tiles))
    rope = MK.rope_table(c, s_max)
    return jax, jnp, MK, R, c, tiles, weights, states, h, r, kw, ks, step, rope


def check(args):
    jax, jnp, MK, R, c, tiles, weights, states, h, r, kw, ks, step, rope = setup(args)
    rel = lambda a, b: float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32))
                             / jnp.linalg.norm(b.astype(jnp.float32)))
    ref = jax.jit(lambda w, h, r, s: R.stage_step(c, w, h, r, s, args.ctx, flash_bkv=tiles.bkv,
                                                  moe_fn=R.moe_ragged))
    hr, rr, sr = jax.block_until_ready(ref(weights, h, r, jax.tree.map(jnp.copy, states)))
    mask = jax.random.uniform(jax.random.PRNGKey(1), h.shape) < 0.01
    h_eps = jnp.where(mask, jnp.nextafter(h, jnp.full_like(h, jnp.inf)), h)
    hn, rn, _ = jax.block_until_ready(ref(weights, h_eps, r, jax.tree.map(jnp.copy, states)))
    t0 = time.time()
    h2, r2, s2 = jax.block_until_ready(step(args.ctx, h, r, kw, jax.tree.map(jnp.copy, ks), rope))
    res = {"mode": "check", "layers": args.layers, "t": args.t, "ctx": args.ctx,
           "first_call_s": round(time.time() - t0, 1),
           "hidden_rel": rel(h2, hr), "hidden_floor": rel(hn, hr),
           "residual_rel": rel(r2, rr), "residual_floor": rel(rn, rr), "layers_state": []}
    for li, lt in enumerate(c.layer_types):
        if lt == "gdn":
            res["layers_state"].append({"rec_rel": rel(s2[li]["rec"], sr[li]["rec"]),
                                        "conv_eq": bool(jnp.all(s2[li]["conv"] == sr[li]["conv"]))})
        else:
            res["layers_state"].append({"k_rel": rel(jnp.swapaxes(s2[li]["k"], 0, 1), sr[li]["k"]),
                                        "v_rel": rel(jnp.swapaxes(s2[li]["v"], 0, 1), sr[li]["v"])})
    res["finite"] = bool(jnp.isfinite(h2.astype(jnp.float32)).all())
    res["pass"] = (res["finite"] and res["hidden_rel"] < 3 * res["hidden_floor"] + 1e-3
                   and res["residual_rel"] < 3 * res["residual_floor"] + 1e-3)
    print("MK_RESULT " + json.dumps(res), flush=True)


def time_step(args):
    jax, jnp, MK, R, c, tiles, weights, states, h, r, kw, ks, step, rope = setup(args)
    del weights, states
    t0 = time.time()
    out = jax.block_until_ready(step(args.ctx, h, r, kw, ks, rope))
    first = time.time() - t0
    if args.mode == "profile":
        d = os.path.join(args.xprof_dir, "stage")
        os.makedirs(d, exist_ok=True)
        with jax.profiler.trace(d):
            jax.block_until_ready(step(args.ctx, h, r, kw, out[2], rope))
        print("PROFILE_DONE " + d, flush=True)
        return
    best, ks_cur = 1e9, out[2]
    for _ in range(args.reps):
        t1 = time.perf_counter()
        out = jax.block_until_ready(step(args.ctx, h, r, kw, ks_cur, rope))
        best = min(best, time.perf_counter() - t1)
        ks_cur = out[2]
    res = {"mode": "time", "layers": args.layers, "t": args.t, "ctx": args.ctx,
           "tiles": dataclasses.asdict(tiles), "first_call_s": round(first, 1),
           "step_ms": round(best * 1e3, 2),
           "finite": bool(jnp.isfinite(out[0].astype(jnp.float32)).all())}
    print("MK_RESULT " + json.dumps(res), flush=True)
    if args.xprof:
        cmd = [sys.executable, os.path.abspath(__file__), "profile"] + [
            a for a in sys.argv[2:] if a != "--xprof"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        done = [l for l in p.stdout.splitlines() if l.startswith("PROFILE_DONE ")]
        print("XPROF " + (done[-1][13:] if done else f"failed rc={p.returncode}: {p.stderr[-500:]}"),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "time", "profile"])
    ap.add_argument("--layers", default="gdn,attn")
    ap.add_argument("--t", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tiles", default="", help="MKTiles overrides, e.g. tm=256,bkv=1024")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--xprof", action="store_true")
    ap.add_argument("--xprof-dir", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "xprof"))
    args = ap.parse_args()
    {"check": check, "time": time_step, "profile": time_step}[args.mode](args)


if __name__ == "__main__":
    main()
