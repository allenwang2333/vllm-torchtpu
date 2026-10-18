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
        step = MK.build(c, args.t, s_max, tiles)
        rope = sd((s_max, c.ROT), jnp.float32)
        t0 = time.time()
        lowered = jax.jit(lambda ctx, h, r, kw_, ks_, rope_: step(ctx, h, r, kw_, ks_, rope_),
                          in_shardings=shard, out_shardings=shard).trace(
            sd((), jnp.int32), h, r, kws, kss, rope).lower(lowering_platforms=("tpu",))
        print(f"lowered in {time.time() - t0:.1f}s", flush=True)
    compiled = lowered.compile(device_assignment=(device,))
    print(f"AOT_OK compiled in {time.time() - t0:.1f}s; temp {compiled.memory_analysis().temp_size_in_bytes / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
