#!/usr/bin/env python3
"""Fused-EP MoE kernel time with the REAL DeepSeek-V4.1 expert weights vs random ones.

Replays served routing ids (one npz per step, topk_ids [40, 2304, 6]) for the
given model layers, each with (a) that layer's real experts converted to the
runtime form exactly as the serving loader does (real_weights.py; checked byte
for byte by verify_real_weights.py) and (b) bench_moe_attr's random fp4
weights. Kernel-only slope (R = 2 vs 8 chained pallas_calls), compiled once per
level (MOE_FUSED_EP_V2_PIPELINED_VISITS) and reused for every operand set; the
real/random arms alternate per step so drift hits both.

  python bench_realweights.py --kernel-rev HEAD_SHA --layers 10,23 \\
      --steps ~/dsv41-work/shared/routing-counts/d186106c4/steps/round1_step0{0,1,2}.npz
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ablate  # noqa: E402
import bench_moe_attr as B  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import real_weights as rw  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

ENV = "MOE_FUSED_EP_V2_PIPELINED_VISITS"
DEPTHS = (2, 8)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kernel-rev", required=True)
    ap.add_argument("--layers", default="10,23")
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--levels", default="0,1")
    ap.add_argument("--ckpt-dir", required=True, help="absolute host path (mounted)")
    ap.add_argument(
        "--index", default="/models/DeepSeek-V4.1-Flash/model.safetensors.index.json"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ablate.pin_source_rev(args.kernel_rev)
    kmod, lmod = ablate.install("base")
    B.moe_layer = lmod
    mesh = B.mesh8()
    g = B.GEO
    layers = [int(v) for v in args.layers.split(",")]
    levels = [int(v) for v in args.levels.split(",")]
    x, gates, bias = B.make_inputs(
        mesh,
        args.seed + 1,
        tokens=g["tokens"],
        hidden=g["hidden"],
        e_total=g["e_total"],
        n_layers=1,
        skew=0.0,
    )
    shard = NamedSharding(mesh, P(B.AXIS))
    tables = [np.load(s)["topk_ids"].astype(np.int32) for s in args.steps]
    weights = {
        "random": B.to_scale_form(
            mesh,
            B.make_fp4_weights(
                mesh,
                args.seed + 100,
                hidden=g["hidden"],
                inter=g["inter"],
                e_total=g["e_total"],
                qb=g["qb"],
            ),
            "flat",
        )
    }
    for layer in layers:
        t0 = time.time()
        path = Path(args.ckpt_dir) / rw.shard_for_layer(Path(args.index), layer)
        raw = rw.read_layer_raw(path, layer, g["e_total"])
        weights[f"real{layer}"] = B.to_scale_form(
            mesh, rw.runtime_weights(mesh, B.AXIS, raw), "flat"
        )
        print(f"layer {layer} weights converted in {time.time() - t0:.0f}s", flush=True)

    meta = dict(
        kernel_rev=args.kernel_rev,
        tree_head=ablate.tree_head(),
        libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
    )

    def emit(rec):
        line = json.dumps({**meta, **rec})
        print("RESULT", line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")

    def call(x, w, gate, bias, hid):
        return B.moe_call(x, w, gate, bias, mesh, hash_ids=hid)

    for level in levels:
        os.environ[ENV] = str(level)
        compiled = {}
        hid0 = jax.device_put(tables[0][layers[0]], shard)
        for repeats in DEPTHS:
            lmod.build_fused_ep_moe_kernel = B.repeat_builder(
                kmod.build_fused_ep_moe_kernel, repeats
            )
            lmod._LAYER_SM_CACHE.clear()
            compiled[repeats] = (
                jax.jit(lambda *a: call(*a))
                .lower(x, weights["random"], gates[0], bias, hid0)
                .compile()
            )
        for step_i, table in enumerate(tables):
            for layer in layers:
                hid = jax.device_put(table[layer], shard)
                stats = B.routing_stats_from_ids(table[layer], e_total=g["e_total"])
                order = ("random", f"real{layer}")
                if step_i % 2:
                    order = order[::-1]
                for wname in order:
                    operands = (x, weights[wname], gates[0], bias, hid)
                    raw = {
                        r: B.pipelined_ms(compiled[r], operands, n=args.iters)[0]
                        for r in DEPTHS
                    }
                    emit(
                        dict(
                            level=level,
                            layer=layer,
                            step=args.steps[step_i],
                            weights="real" if wname.startswith("real") else "random",
                            us_per_layer=(raw[8] - raw[2]) / (8 - 2) * 1e3,
                            raw_ms=raw,
                            routing=stats,
                        )
                    )


if __name__ == "__main__":
    main()
