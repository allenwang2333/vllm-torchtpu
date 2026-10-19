"""Compile the prefill megakernel for tpu7x on a host without a TPU (compile-only topology):
catches Mosaic lowering and VMEM errors before spending a TPU job.

    python3 aot_compile.py --layers gdn,attn --t 1024 --ctx 4096 [--tiles tm=256]
"""

import argparse
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="gdn,attn")
    ap.add_argument("--t", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tiles", default="")
    ap.add_argument("--pooled", action="store_true", help="the serving form (build_stage)")
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--n-buf", type=int, default=8192, help="the runner's padded token count")
    ap.add_argument("--first", action="store_true")
    ap.add_argument("--last", action="store_true")
    args = ap.parse_args()
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.path.insert(0, os.path.join(CHECKOUT, "src"))
    import types
    # Import the MK modules without the vllm_torchtpu package __init__ (it needs torch, which a
    # compile-only host need not have).
    pkg = types.ModuleType("vllm_torchtpu")
    pkg.__path__ = [os.path.join(CHECKOUT, "src", "vllm_torchtpu")]
    sys.modules.setdefault("vllm_torchtpu", pkg)
    k = types.ModuleType("vllm_torchtpu.kernels")
    k.__path__ = [os.path.join(CHECKOUT, "src", "vllm_torchtpu", "kernels")]
    sys.modules.setdefault("vllm_torchtpu.kernels", k)
    try:
        runpy.run_path(os.path.join(CHECKOUT, "src", "vllm_torchtpu", "env_override.py"))
    except ImportError as e:  # the Pallas lowering patches only change speed, not legality
        print(f"env_override skipped: {e}", flush=True)
    import jax
    import jax.numpy as jnp
    from jax._src import mesh as mesh_lib
    from jax.experimental import topologies

    from vllm_torchtpu.kernels.qwen35_prefill_mk import megakernel as MK
    from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R

    topo = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
    device = topo.devices[0]
    shard = jax.sharding.SingleDeviceSharding(device)
    abstract = mesh_lib.AbstractMesh((), (), abstract_device=mesh_lib.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"))
    os.environ.setdefault("VLLM_KV_CACHE_LAYOUT", "HND")
    os.environ.setdefault("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    c = R.StageConfig(layer_types=tuple(args.layers.split(",")))
    kw = {}
    for item in filter(None, args.tiles.split(",")):
        a, _, b = item.partition("=")
        kw[a] = int(b)
    tiles = MK.MKTiles(**kw)
    s_max = args.ctx + args.t
    sd = jax.ShapeDtypeStruct
    with mesh_lib.use_abstract_mesh(abstract):
        w, st, h, r = jax.eval_shape(lambda key: R.random_stage_device(c, args.t, args.ctx, key),
                                     jax.random.PRNGKey(0))
        kws = [jax.eval_shape(lambda x, lt=lt: MK.kernel_weight_layout(c, x, lt), wi)
               for wi, lt in zip(w, c.layer_types)]
        kss = [si if lt == "gdn" else {"k": sd((c.NKV, s_max, c.HD), R.FP8),
                                       "v": sd((c.NKV, s_max, c.HD), R.FP8)}
               for si, lt in zip(st, c.layer_types)]
        rope = sd((s_max, c.ROT), jnp.float32)
        t0 = time.time()
        if args.pooled:
            from vllm_torchtpu.kernels.qwen35_prefill_mk import pool_plan
            types = c.layer_types
            regions = [i // 4 for i in range(len(types))]
            layout = MK.StageLayout(types, tuple(regions), tuple(i % 4 for i in range(len(types))))
            pps = -(-s_max // 128)
            pool_shape = (4096 * 17, 2 * c.NKV, c.HD // 4, 4, 128)
            plan, stride = pool_plan.build_gdn_plan(pool_shape, R.FP8, conv_dim=c.QKV, n_v=c.VH,
                                                    d_k=c.KD, d_v=c.VD, kernel_size=c.CONV,
                                                    pool_block_tokens=2176)
            step = MK.build_stage(c, layout, args.t, max_seqs=args.max_seqs, pps=pps,
                                  pool_shape=pool_shape, plan=plan, stride=stride,
                                  n_buf=args.n_buf, first=args.first, last=args.last,
                                  tiles=tiles)
            S, nb = args.max_seqs, args.n_buf
            i32 = lambda *sh: sd(sh, jnp.int32)
            meta = {"qsl": i32(S + 1), "seq_lens": i32(S), "dist": i32(3), "bt": i32(S * pps),
                    "gslot0": i32(S), "gslot1": i32(S), "gslot2": i32(S)}
            if args.first:
                meta["ids"] = i32(nb)
                x, rr = i32(248320, c.H // 2), sd((1, c.H), R.BF16)
            else:
                x, rr = sd((nb, c.H), R.BF16), sd((nb, c.H), R.BF16)
            pools = [sd(pool_shape, R.FP8) for _ in range(layout.n_regions)]
            lowered = jax.jit(step, in_shardings=shard, out_shardings=shard).trace(
                meta, x, rr, sd((1, c.H), R.BF16), sd((pps * 128, c.ROT), jnp.float32), kws,
                pools).lower(lowering_platforms=("tpu",))
        else:
            step = MK.build(c, args.t, s_max, tiles)
            lowered = jax.jit(lambda ctx, h, r, kw_, ks_, rope_: step(ctx, h, r, kw_, ks_, rope_),
                              in_shardings=shard, out_shardings=shard).trace(
                sd((), jnp.int32), h, r, kws, kss, rope).lower(lowering_platforms=("tpu",))
        print(f"lowered in {time.time() - t0:.1f}s", flush=True)
    compiled = lowered.compile(device_assignment=(device,))
    print(f"AOT_OK compiled in {time.time() - t0:.1f}s; temp {compiled.memory_analysis().temp_size_in_bytes / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
