#!/usr/bin/env python3
"""Fused-EP MoE v2 kernel variants against the default build: the pipelined
visit loop (MOE_FUSED_EP_V2_PIPELINED_VISITS, "level" 0/1 below) and unpadded
expert weights (--weights unpadded, intermediate 2304 instead of 2560).

  --mode kernel   kernel-only slope (R=2 vs 8 chained pallas_calls, as in
                  bench_moe_attr.py) for each --levels entry; with
                  --routing-npz, for each of --layers from one compile.
  --mode bytes    one layer call per level on fixed inputs; each shard's raw
                  arrival rows and arrival scales are copied to the host and
                  compared byte for byte with the first level, together with
                  the layer output. The first level runs twice, which is the
                  determinism control for the raw buffers (rows and mirror
                  lanes no transport writes are whatever the allocation held).
                  --compare-unpadded compares the unpadded build with the
                  zero-padded one instead.

Run inside the dsv41 image under `perfkit device run`, with the serving
LIBTPU_INIT_ARGS exported, e.g.
  python bench_pipeline.py --mode kernel --levels 0,1 --skew 0.5
  python bench_pipeline.py --mode bytes --levels 0,1 --tokens 16384
  python bench_pipeline.py --mode kernel --routing-npz step.npz --layers all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_moe_attr as B  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v2 import kernel as kmod  # noqa: E402
from vllm_torchtpu.kernels.fused_moe.v2 import layer as lmod  # noqa: E402

ENV = "MOE_FUSED_EP_V2_PIPELINED_VISITS"


def set_level(level):
    os.environ[ENV] = str(level)
    lmod._LAYER_SM_CACHE.clear()


def capture_builder(store):
    """The tree's builder, with each shard's raw arrivals sent to the host."""

    def build(**kw):
        kfn = kmod.build_fused_ep_moe_kernel(**kw)

        def fn(*args, recv_rows, rank, **arrays):
            recv, rscl = kfn(*args, recv_rows=recv_rows, rank=rank, **arrays)

            def put(r, a, s):
                store[int(np.asarray(r).reshape(-1)[0])] = (
                    np.asarray(a).view(np.uint8).copy(),
                    np.asarray(s).view(np.uint8).copy(),
                )

            jax.debug.callback(put, rank, recv, rscl)
            return recv, rscl

        return fn

    return build


TRUE_INTER = (
    2304  # DeepSeek-V4.1-Flash moe_intermediate_size; GEO["inter"] is it padded
)


def make_padded_pair(mesh, seed):
    """The same fp4 expert weights twice: zero-padded to GEO["inter"] (what
    the loader serves today) and unpadded at TRUE_INTER.

    Returns (padded (w1, w2, s1, s2), unpadded (w1, w2, s1, s2)). The scale
    blocks are taken over the padded contraction, so the down matmul's last
    block scale covers its 256 real rows and 256 zeros -- the same value as a
    scale over the real rows alone; the unpadded tables are the padded ones
    with the dead w13 columns dropped.
    """
    g = B.GEO
    hidden, inter, qb, e_total = g["hidden"], g["inter"], g["qb"], g["e_total"]
    g_local = e_total // B.EP
    real = TRUE_INTER

    def quant(w):
        """(normalized f32, scale): cast after any slicing, never before."""
        k, n = w.shape
        grouped = w.reshape(k // qb, qb, n)
        scale = jnp.maximum(jnp.max(jnp.abs(grouped), axis=1) / 6.0, 1e-12)
        return (grouped / scale[:, None, :]).reshape(k, n), scale

    def local(_):
        base = jax.random.fold_in(jax.random.key(seed), jax.lax.axis_index(B.AXIS))

        def one(e):
            k1, k2, k3 = jax.random.split(jax.random.fold_in(base, e), 3)
            gate = jax.random.normal(k1, (hidden, real)) * 0.1
            up = jax.random.normal(k2, (hidden, real)) * 0.1
            down = jax.random.normal(k3, (real, hidden)) * 0.1
            pad_n = jnp.zeros((hidden, inter - real))
            w1, s1 = quant(jnp.concatenate([gate, pad_n, up, pad_n], axis=1))
            w2, s2 = quant(
                jnp.concatenate([down, jnp.zeros((inter - real, hidden))], axis=0)
            )

            def live(a):  # the gate and up columns without their padding
                return jnp.concatenate([a[:, :real], a[:, inter : inter + real]], 1)

            return (
                w1.astype(B.FP4),
                w2.astype(B.FP4),
                s1,
                s2,
                live(w1).astype(B.FP4),
                w2[:real].astype(B.FP4),
                live(s1),
            )

        return jax.lax.map(one, jnp.arange(g_local))

    w1, w2, s1, s2, w1u, w2u, s1u = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(B.AXIS),),
            out_specs=(P(B.AXIS),) * 7,
            check_vma=False,
        )
    )(jnp.zeros((B.EP,), jnp.float32))
    return (w1, w2, s1, s2), (w1u, w2u, s1u, s2)


def capture_layer(call, args):
    """One layer call: its output bytes and each shard's raw arrival bytes."""
    store = {}
    lmod.build_fused_ep_moe_kernel = capture_builder(store)
    lmod._LAYER_SM_CACHE.clear()
    # A new function per call, for the reason by_depth gives.
    out = jax.jit(lambda *a: call(*a))(*args)
    out = np.asarray(jax.device_get(out)).view(np.uint8).copy()
    jax.effects_barrier()
    return out, dict(store)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("kernel", "bytes"), default="kernel")
    ap.add_argument("--levels", default="0,1")
    ap.add_argument("--tokens", type=int, default=B.GEO["tokens"])
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routing-npz", default=None)
    ap.add_argument(
        "--layers",
        default="0",
        help="layers of --routing-npz: comma list or 'all' (kernel mode times "
        "each; bytes mode uses the first)",
    )
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--weights",
        choices=("random", "padded", "unpadded"),
        default="random",
        help="random: bench_moe_attr's weights, padding included; "
        "padded/unpadded: the two forms of make_padded_pair",
    )
    ap.add_argument(
        "--scale-form",
        choices=("flat", "3d"),
        default="flat",
        help="fp4 scale-table operand form; flat is what serving passes",
    )
    ap.add_argument(
        "--compare-unpadded",
        action="store_true",
        help="bytes mode: compare the unpadded build with the zero-padded one "
        "at each level, instead of levels with each other",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    levels = [int(v) for v in args.levels.split(",")]
    mesh = B.mesh8()
    g = B.GEO
    x, gates, bias = B.make_inputs(
        mesh,
        args.seed + 1,
        tokens=args.tokens,
        hidden=g["hidden"],
        e_total=g["e_total"],
        n_layers=1,
        skew=args.skew,
    )
    # layer -> (hash ids or None, routing stats)
    if args.routing_npz:
        table = np.load(args.routing_npz)["topk_ids"].astype(np.int32)
        if table.shape[1:] != (args.tokens, g["topk"]):
            raise ValueError(f"routing table {table.shape} is not [L, T, topk]")
        layers = (
            range(len(table))
            if args.layers == "all"
            else [int(v) for v in args.layers.split(",")]
        )
        shard = NamedSharding(mesh, P(B.AXIS))
        cases = {
            i: (
                jax.device_put(table[i], shard),
                B.routing_stats_from_ids(table[i], e_total=g["e_total"]),
            )
            for i in layers
        }
    else:
        cases = {
            None: (
                None,
                B.routing_stats(gates[0], bias, topk=g["topk"], e_total=g["e_total"]),
            )
        }
    if args.weights == "random" and not args.compare_unpadded:
        w = B.make_fp4_weights(
            mesh,
            args.seed + 100,
            hidden=g["hidden"],
            inter=g["inter"],
            e_total=g["e_total"],
            qb=g["qb"],
        )
        w = B.to_scale_form(mesh, w, args.scale_form)
    else:
        w_padded, w_unpadded = (
            B.to_scale_form(mesh, ws, args.scale_form)
            for ws in make_padded_pair(mesh, args.seed + 100)
        )
        w = w_unpadded if args.weights == "unpadded" else w_padded
    meta = dict(
        tokens=args.tokens,
        weights=args.weights,
        scale_form=args.scale_form,
        skew=args.skew,
        seed=args.seed,
        routing_npz=args.routing_npz,
        libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
    )

    def emit(rec):
        line = json.dumps({**meta, **rec})
        print("RESULT", line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")

    B.moe_layer = lmod

    def call(x, w, gate, bias, hid):
        return B.moe_call(x, w, gate, bias, mesh, hash_ids=hid)

    first_layer = next(iter(cases))
    hid, stats = cases[first_layer]
    if args.mode == "kernel":
        depths = (2, 8)
        for level in levels:
            set_level(level)
            compiled = {}
            for repeats in depths:
                lmod.build_fused_ep_moe_kernel = B.repeat_builder(
                    kmod.build_fused_ep_moe_kernel, repeats
                )
                lmod._LAYER_SM_CACHE.clear()
                # A new function per depth: jit caches the trace on the
                # function, and the env level and builder are read at trace.
                compiled[repeats] = (
                    jax.jit(lambda *a: call(*a))
                    .lower(x, w, gates[0], bias, hid)
                    .compile()
                )
            for layer_i, (hid_i, stats_i) in cases.items():
                operands = (x, w, gates[0], bias, hid_i)
                raw = {}
                for repeats in depths:
                    ms, samples = B.pipelined_ms(
                        compiled[repeats], operands, n=args.iters
                    )
                    raw[repeats] = dict(ms=ms, samples=samples)
                per = (raw[8]["ms"] - raw[2]["ms"]) / (8 - 2)
                emit(
                    dict(
                        mode="kernel",
                        level=level,
                        layer=layer_i,
                        routing=stats_i,
                        us_per_layer=per * 1e3,
                        raw=raw,
                    )
                )
        return

    if args.compare_unpadded:
        for level in levels:
            set_level(level)
            o0, st0 = capture_layer(call, (x, w_padded, gates[0], bias, hid))
            o1, st1 = capture_layer(call, (x, w_unpadded, gates[0], bias, hid))
            emit(
                dict(
                    mode="bytes-unpadded",
                    layer=first_layer,
                    routing=stats,
                    level=level,
                    out_equal=bool(np.array_equal(o0, o1)),
                    out_diff_bytes=int((o0 != o1).sum()),
                    recv_all_equal=all(
                        np.array_equal(st0[r][0], st1[r][0]) for r in st0
                    ),
                    rscl_all_equal=all(
                        np.array_equal(st0[r][1], st1[r][1]) for r in st0
                    ),
                )
            )
        return

    results = {}
    for level in [levels[0]] + levels:
        set_level(level)
        results.setdefault(level, []).append(
            capture_layer(call, (x, w, gates[0], bias, hid))
        )
    ref_out, ref_store = results[levels[0]][0]
    for level, runs in results.items():
        for i, (out, store) in enumerate(runs):
            if level == levels[0] and i == 0:
                continue
            per_shard = {}
            for r in sorted(ref_store):
                a0, s0 = ref_store[r]
                a1, s1 = store[r]
                per_shard[r] = dict(
                    recv_equal=bool(np.array_equal(a0, a1)),
                    recv_diff_bytes=int((a0 != a1).sum()),
                    rscl_equal=bool(np.array_equal(s0, s1)),
                    rscl_diff_bytes=int((s0 != s1).sum()),
                )
            emit(
                dict(
                    mode="bytes",
                    layer=first_layer,
                    routing=stats,
                    level=level,
                    run=i,
                    vs_level=levels[0],
                    out_equal=bool(np.array_equal(ref_out, out)),
                    out_diff_bytes=int((ref_out != out).sum()),
                    recv_all_equal=all(v["recv_equal"] for v in per_shard.values()),
                    rscl_all_equal=all(v["rscl_equal"] for v in per_shard.values()),
                    shards=per_shard,
                )
            )


if __name__ == "__main__":
    main()
