#!/usr/bin/env python3
"""Isolated attribution of the fused-EP W4A8 MoE layer at the DSv4.1-Flash
DP8 C384 decode shape: 8-device EP, 288 local tokens (2304 global), 384
experts top-6, hidden 5120, intermediate 2560 (2304 padded), fp4 e2m1
weights with f32 scales per 512-row contraction block.

Pure JAX, one process, all eight TensorCores; it calls the serving entry
point `fused_ep_moe_v2` with the serving arguments. Every per-layer number
is a SLOPE between two chain depths in one jit, so dispatch and the jit's
fixed cost cancel:

  layer   L layers chained (x -> moe -> rms-normalize -> next layer), slope
          over L: the whole layer, glue included.
  kernel  one layer whose pallas_call is issued R times back to back (each
          call's activation rows take one element of the previous call's
          arrivals, so nothing is merged), slope over R: the kernel alone.
  glue    as `layer`, with the pallas_call replaced by zeros that depend on
          one element of every kernel operand: everything but the kernel.
  norm    the harness's own per-layer rms-normalize, chained alone.
  ag      the dispatch all-gather f8[288,40,128] -> [2304,40,128] alone.

Kernel ablation arms are named by `ablate.py` ('now', 'nodq+now', ...); they
are timing probes whose outputs are garbage.

Run inside the dsv41 image, e.g.
  python bench_moe_attr.py --arms base,now --mode kernel --out res.jsonl
  python bench_moe_attr.py --check
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ablate  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, host  # noqa: E402
from vllm_torchtpu.kernels.fused_moe.v2 import layer as served_layer  # noqa: E402
from vllm_torchtpu.kernels.megablox.gmm_v2 import silu_and_mul_with_clamp  # noqa: E402

EP = 8
GEO = dict(tokens=2304, hidden=5120, inter=2560, e_total=384, topk=6, qb=512)
CAPACITY = 128
ACT = "silu_and_mul_with_clamp"
FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
# Weight sets the chains cycle through. HBM has no cache in front of it, so
# re-reading a set costs the same as a fresh one; two keeps a layer's weights
# from being the previous layer's.
N_WEIGHT_SETS = 2


def mesh8():
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))


def make_fp4_weights(mesh, seed, *, hidden, inter, e_total, qb):
    """(w1, w2, s1, s2): N(0, 0.1) weights quantized to fp4 e2m1 with one f32
    scale per qb contraction rows, built expert by expert on each shard."""
    g_local = e_total // EP

    def quant(w):
        k, n = w.shape
        grouped = w.reshape(k // qb, qb, n)
        scale = jnp.maximum(jnp.max(jnp.abs(grouped), axis=1) / 6.0, 1e-12)
        q = (grouped / scale[:, None, :]).reshape(k, n).astype(FP4)
        return q, scale

    def local(_):
        base = jax.random.fold_in(jax.random.key(seed), lax.axis_index(AXIS))

        def one(e):
            k1, k2 = jax.random.split(jax.random.fold_in(base, e))
            w1, s1 = quant(jax.random.normal(k1, (hidden, 2 * inter)) * 0.1)
            w2, s2 = quant(jax.random.normal(k2, (inter, hidden)) * 0.1)
            return w1, w2, s1, s2

        return lax.map(one, jnp.arange(g_local))

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(AXIS),),
            out_specs=(P(AXIS),) * 4,
            check_vma=False,
        )
    )(jnp.zeros((EP,), jnp.float32))


def make_inputs(
    mesh, seed, *, tokens, hidden, e_total, n_layers, skew, corr=0.0, rows_per_req=6
):
    """x [T, H] bf16, per-layer gate logits [T, E] f32 and a score bias [E].

    `skew` is the sd of a per-expert logit offset shared by every token, which
    makes some experts hot; 0 is near-uniform routing. `corr` correlates the
    logits of each run of `rows_per_req` consecutive rows (one request's
    target + draft rows under speculative decode): each row's logits are
    corr * request noise + sqrt(1 - corr^2) * own noise."""
    shard = NamedSharding(mesh, P(AXIS))
    key = jax.random.key(seed)
    kx, kg, kb = jax.random.split(key, 3)
    x = jax.device_put(
        (jax.random.normal(kx, (tokens, hidden)) * 0.1).astype(jnp.bfloat16), shard
    )
    gates = []
    for layer in range(n_layers):
        kl, ko, kr = jax.random.split(jax.random.fold_in(kg, layer), 3)
        offset = jax.random.normal(ko, (1, e_total)) * skew
        req = jnp.repeat(
            jax.random.normal(kr, (tokens // rows_per_req, e_total)), rows_per_req, 0
        )
        own = jax.random.normal(kl, (tokens, e_total))
        logits = corr * req + jnp.sqrt(1.0 - corr * corr) * own + offset
        gates.append(jax.device_put(logits, shard))
    bias = jax.random.normal(kb, (e_total,)) * 0.01
    return x, gates, bias


def routing_stats(gate, bias, *, topk, e_total):
    """Load statistics of one layer's routing, as the kernel will see it."""
    scores = jnp.sqrt(jax.nn.softplus(gate.astype(jnp.float32)))
    _, ids = lax.top_k(scores + bias[None, :], topk)
    return routing_stats_from_ids(np.asarray(ids), e_total=e_total)


def routing_stats_from_ids(ids, *, e_total):
    g_local = e_total // EP
    per_expert = np.bincount(ids.reshape(-1), minlength=e_total)
    per_dev = per_expert.reshape(EP, g_local)
    rows_dev = per_dev.sum(1)
    return dict(
        rows_per_device_mean=float(rows_dev.mean()),
        rows_per_device_max=int(rows_dev.max()),
        rows_per_expert_mean=float(per_expert.mean()),
        rows_per_expert_max=int(per_expert.max()),
        experts_visited_per_device_min=int((per_dev > 0).sum(1).min()),
        tiles_per_device_max=int(np.ceil(per_dev / CAPACITY).sum(1).max()),
        rung_histogram={
            str(r): int(((per_expert > r - 32) & (per_expert <= r)).sum())
            for r in (32, 64, 96, 128)
        },
        experts_over_one_tile=int((per_expert > CAPACITY).sum()),
    )


# The layer module the current arm runs through (ablate.install's choice).
moe_layer = served_layer


def moe_call(
    x, w, gate, bias, mesh, scoring_fn="sqrtsoftplus", scaling=1.5, hash_ids=None
):
    """One served MoE layer. `hash_ids` [T, topk] replays a given selection
    (the router's hash path; the weights are still the scores there)."""
    w1, w2, s1, s2 = w
    return moe_layer.fused_ep_moe_v2(
        x,
        w1,
        w2,
        s1,
        s2,
        gate,
        topk=GEO["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=GEO["qb"],
        act_fn=ACT,
        sharded_plan=True,
        scoring_fn=scoring_fn,
        score_bias=bias if scoring_fn == "sqrtsoftplus" and hash_ids is None else None,
        routed_scaling_factor=scaling if scoring_fn == "sqrtsoftplus" else 1.0,
        hash_topk_ids=hash_ids,
    )


def rms_normalize(y):
    y32 = y.astype(jnp.float32)
    return (y32 * lax.rsqrt(jnp.mean(y32 * y32, -1, keepdims=True) + 1e-6)).astype(
        jnp.bfloat16
    )


# --------------------------------------------------------------------------- #
# Kernel wrappers installed between the layer and the arm's builder.
# --------------------------------------------------------------------------- #
# Position of the gathered activation rows in the kernel function's arguments
# (kernel.py `fn(tables, token_gather, expert_rows, expert_base,
# act_q_gathered, ...)`).
ACT_ROWS_ARG = 4
WEIGHT_ARGS = 6


def depend(dst, src):
    """dst with its first element replaced by src's: a real data dependency.

    An optimization barrier is not enough on this toolchain -- repeated
    identical calls chained through one were still merged (zero slope) -- so
    every chained unit takes one element of its predecessor's output.
    """
    one = lax.slice(src, (0,) * src.ndim, (1,) * src.ndim)
    one = one.reshape((1,) * dst.ndim).astype(dst.dtype)
    return lax.dynamic_update_slice(dst, one, (jnp.int32(0),) * dst.ndim)


def repeat_builder(builder, repeats):
    """The arm's builder, with each built kernel issued `repeats` times."""

    def build(**kw):
        kfn = builder(**kw)

        def rep(*args, recv_rows, **arrays):
            out = kfn(*args, recv_rows=recv_rows, **arrays)
            for _ in range(repeats - 1):
                args = (
                    *args[:ACT_ROWS_ARG],
                    depend(args[ACT_ROWS_ARG], out[0]),
                    *args[ACT_ROWS_ARG + 1 :],
                )
                out = kfn(*args, recv_rows=recv_rows, **arrays)
            return out

        return rep

    return build


def stub_builder(**kw):
    """A kernel that does nothing but hold its operands live."""
    lanes = host.HIDDEN_LANE_BLOCK

    def fn(*args, recv_rows, **arrays):
        recv = jnp.zeros((recv_rows, kw["hidden"] // lanes, lanes), FP8)
        mirror = host.arrival_mirror_rows(
            recv_rows, kw["ep"] * kw["g_local"], kw["capacity"]
        )
        rscl = jnp.zeros((mirror, kw["capacity"]), jnp.float32)
        # One element of every operand the layer computes, so nothing feeding
        # the kernel is dead. The weights (args[6:10]) are parameters.
        leaves = [a for a in jax.tree.leaves((args[:WEIGHT_ARGS], arrays)) if a.size]
        probe = sum(
            lax.slice(a, (0,) * a.ndim, (1,) * a.ndim).reshape(()).astype(jnp.float32)
            for a in leaves
        )
        return recv, depend(rscl, probe)

    return fn


# --------------------------------------------------------------------------- #
# Timing.
# --------------------------------------------------------------------------- #
def pipelined_ms(fn, args, *, n=20, bursts=3):
    """Median over bursts of (n async calls, one block) / n, in ms."""
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(bursts):
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1e3 / n)
    return statistics.median(samples), samples


def count_kernel_calls(compiled_text):
    return compiled_text.count("tpu_custom_call")


def slope(fn_by_depth, args, depths, n):
    """Per-unit ms between two depths, plus the raw readings."""
    lo, hi = depths
    raw = {}
    for d in depths:
        f = fn_by_depth(d)
        t_compile = time.perf_counter()
        lowered = f.lower(*args)
        # Counted pre-optimization: the compiled executable's HLO text is not
        # retrievable on this runtime (async-update proto RET_CHECK). A CSE'd
        # repeat would show up as a zero slope instead.
        n_calls = count_kernel_calls(lowered.as_text())
        compiled = lowered.compile()
        t_compile = time.perf_counter() - t_compile
        ms, samples = pipelined_ms(compiled, args, n=n)
        raw[d] = dict(
            ms=ms, samples=samples, pallas_calls=n_calls, compile_s=round(t_compile, 1)
        )
    return (raw[hi]["ms"] - raw[lo]["ms"]) / (hi - lo), raw


# --------------------------------------------------------------------------- #
# Modes.
# --------------------------------------------------------------------------- #
def run_arm(arm, mode, mesh, x, gates, bias, wsets, depths, n, ids=None):
    global moe_layer
    mod, moe_layer = ablate.install(arm)
    if mode == "glue":
        moe_layer.build_fused_ep_moe_kernel = stub_builder

    if mode in ("layer", "glue"):

        def by_depth(depth):
            def chain(x, gates, wsets, bias, ids):
                for layer in range(depth):
                    x = rms_normalize(
                        moe_call(
                            x,
                            wsets[layer % len(wsets)],
                            gates[layer],
                            bias,
                            mesh,
                            hash_ids=None if ids is None else ids[layer],
                        )
                    )
                return x

            return jax.jit(chain)

        per, raw = slope(by_depth, (x, gates, wsets, bias, ids), depths, n)
    elif mode == "kernel":

        def by_depth(repeats):
            moe_layer.build_fused_ep_moe_kernel = repeat_builder(
                mod.build_fused_ep_moe_kernel, repeats
            )
            moe_layer._LAYER_SM_CACHE.clear()

            def one(x, gates, wsets, bias, ids):
                return moe_call(
                    x,
                    wsets[0],
                    gates[0],
                    bias,
                    mesh,
                    hash_ids=None if ids is None else ids[0],
                )

            return jax.jit(one)

        per, raw = slope(by_depth, (x, gates, wsets, bias, ids), depths, n)
    else:
        raise ValueError(mode)
    return per, raw


def run_norm(mesh, x, depths, n):
    def by_depth(depth):
        def chain(x):
            for _ in range(depth):
                x = rms_normalize(x)
            return x

        return jax.jit(chain)

    return slope(by_depth, (x,), depths, n)


def run_allgather(mesh, x, depths, n):
    """The layer's dispatch collective: f8 [t_local, H/128, 128], tiled."""
    lanes = host.HIDDEN_LANE_BLOCK
    t, h = x.shape
    q = jax.device_put(
        x.astype(FP8).reshape(t, h // lanes, lanes), NamedSharding(mesh, P(AXIS))
    )

    def by_depth(depth):
        def local(q_l):
            for _ in range(depth):
                g = lax.all_gather(q_l, AXIS, axis=0, tiled=True)
                q_l = depend(q_l, lax.slice_in_dim(g, g.shape[0] - 1, g.shape[0]))
            return q_l

        return jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(P(AXIS),),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )

    return slope(by_depth, (q,), depths, n)


def dense_reference(mesh, x, w, gating, *, topk, qb):
    """softmax router, top-k, renormalize, every local expert dense in f32 on
    the whole batch (weights decoded from the same fp4 + scales), weighted sum
    over shards."""
    w1, w2, s1, s2 = w
    g_local = w1.shape[0] // EP
    inter = w1.shape[2] // 2
    hi = lax.Precision.HIGHEST

    def decode(q, s):
        k, n = q.shape
        return (q.astype(jnp.float32).reshape(k // qb, qb, n) * s[:, None, :]).reshape(
            k, n
        )

    def local(x_g, gate_g, w1_l, w2_l, s1_l, s2_l):
        scores = jax.nn.softmax(gate_g.astype(jnp.float32), axis=-1)
        tw, ti = lax.top_k(scores, topk)
        tw = tw / tw.sum(-1, keepdims=True)
        first = lax.axis_index(AXIS) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = decode(w1_l[e], s1_l[e])
            w2e = decode(w2_l[e], s2_l[e])
            weight = jnp.sum(jnp.where(ti == first + e, tw, 0.0), -1)[:, None]
            a1 = jnp.dot(x32, w1e, precision=hi)
            mid = silu_and_mul_with_clamp(a1[:, :inter], a1[:, inter:])
            return acc + weight * jnp.dot(mid, w2e, precision=hi)

        return lax.psum(lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32)), AXIS)

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(), P(), P(AXIS), P(AXIS), P(AXIS), P(AXIS)),
            out_specs=P(),
            check_vma=False,
        )
    )(x, gating, w1, w2, s1, s2)


def rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def worst_token_rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    per = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)
    return float(per.max())


def run_check(mesh, seed):
    """The served build at the served shape against the dense reference."""
    global moe_layer
    _, moe_layer = ablate.install("base")
    w = make_fp4_weights(
        mesh,
        seed,
        hidden=GEO["hidden"],
        inter=GEO["inter"],
        e_total=GEO["e_total"],
        qb=GEO["qb"],
    )
    x, gates, _ = make_inputs(
        mesh,
        seed + 1,
        tokens=GEO["tokens"],
        hidden=GEO["hidden"],
        e_total=GEO["e_total"],
        n_layers=1,
        skew=0.0,
    )
    out = jax.jit(lambda x, w, g: moe_call(x, w, g, None, mesh, scoring_fn="softmax"))(
        x, w, gates[0]
    )
    out = np.asarray(out.astype(jnp.float32))
    ref = np.asarray(
        dense_reference(mesh, x, w, gates[0], topk=GEO["topk"], qb=GEO["qb"])
    )
    rolled = tuple(jnp.roll(a, 1, axis=0) for a in w)
    wrong = np.asarray(
        dense_reference(mesh, x, rolled, gates[0], topk=GEO["topk"], qb=GEO["qb"])
    )
    res = dict(
        mode="check",
        finite=bool(np.isfinite(out).all() and np.isfinite(ref).all()),
        out_checksum=float(np.abs(out).sum()),
        ref_checksum=float(np.abs(ref).sum()),
        rel_l2=rel_l2(out, ref),
        worst_token_rel_l2=worst_token_rel_l2(out, ref),
        rotated_control_rel_l2=rel_l2(out, wrong),
        band=dict(rel_l2=0.06, worst_token=0.075, rotated_factor=10),
    )
    res["pass"] = bool(
        res["finite"]
        and res["rel_l2"] < 0.06
        and res["worst_token_rel_l2"] < 0.075
        and res["rotated_control_rel_l2"] > 10 * res["rel_l2"]
    )
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", default="kernel", choices=("kernel", "layer", "glue", "norm", "ag")
    )
    ap.add_argument("--arms", default="base", help="comma-separated ablate.py arms")
    ap.add_argument(
        "--depths", default=None, help="two depths (default: kernel 2,8; others 4,12)"
    )
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument(
        "--corr",
        type=float,
        default=0.0,
        help="logit correlation within each request's 6 rows",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--routing-npz",
        default=None,
        help="replay expert ids from an npz with topk_ids [L, T, topk]",
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None, help="append one JSON line per result")
    args = ap.parse_args()

    mesh = mesh8()
    meta = dict(
        jax=jax.__version__,
        devices=[str(d) for d in mesh.devices.flat],
        geo=GEO,
        capacity=CAPACITY,
        skew=args.skew,
        corr=args.corr,
        seed=args.seed,
        libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
    )

    def emit(rec):
        rec = {**meta, **rec}
        line = json.dumps(rec)
        print("RESULT", line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")

    if args.check:
        emit(run_check(mesh, args.seed))
        return

    depths = tuple(
        int(v)
        for v in (args.depths or ("2,8" if args.mode == "kernel" else "4,12")).split(
            ","
        )
    )
    n_layers = max(depths) if args.mode in ("layer", "glue") else 1
    x, gates, bias = make_inputs(
        mesh,
        args.seed + 1,
        tokens=GEO["tokens"],
        hidden=GEO["hidden"],
        e_total=GEO["e_total"],
        n_layers=n_layers,
        skew=args.skew,
        corr=args.corr,
    )
    if args.mode == "norm":
        per, raw = run_norm(mesh, x, depths, args.iters)
        emit(dict(mode="norm", arm="-", depths=depths, ms_per_unit=per, raw=raw))
        return
    if args.mode == "ag":
        per, raw = run_allgather(mesh, x, depths, args.iters)
        emit(dict(mode="ag", arm="-", depths=depths, ms_per_unit=per, raw=raw))
        return
    ids = None
    if args.routing_npz:
        # [layers, T, topk] expert ids, e.g. dumped from a served step; layer
        # l of the chain replays entry l % layers.
        table = np.load(args.routing_npz)["topk_ids"].astype(np.int32)
        if table.shape[1:] != (GEO["tokens"], GEO["topk"]):
            raise ValueError(
                f"routing table {table.shape} is not [L, "
                f"{GEO['tokens']}, {GEO['topk']}]"
            )
        shard = NamedSharding(mesh, P(AXIS))
        ids = [jax.device_put(table[i % len(table)], shard) for i in range(n_layers)]
        stats = routing_stats_from_ids(table[0], e_total=GEO["e_total"])
    else:
        stats = routing_stats(gates[0], bias, topk=GEO["topk"], e_total=GEO["e_total"])
    n_sets = N_WEIGHT_SETS if args.mode in ("layer", "glue") else 1
    wsets = [
        make_fp4_weights(
            mesh,
            args.seed + 100 + i,
            hidden=GEO["hidden"],
            inter=GEO["inter"],
            e_total=GEO["e_total"],
            qb=GEO["qb"],
        )
        for i in range(n_sets)
    ]
    for arm in args.arms.split(","):
        per, raw = run_arm(
            arm, args.mode, mesh, x, gates, bias, wsets, depths, args.iters, ids
        )
        emit(
            dict(
                mode=args.mode,
                arm=arm,
                depths=depths,
                ms_per_unit=per,
                routing_npz=args.routing_npz,
                routing_layer0=stats,
                raw=raw,
            )
        )


if __name__ == "__main__":
    main()
