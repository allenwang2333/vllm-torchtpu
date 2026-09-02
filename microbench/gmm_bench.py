"""gmm_v2 microbenchmark at the two Qwen3.5-397B-A17B-FP8 prefill shapes.

From the profile of job j-80ee1b01-98d7-498e-a834 (tpu7x, DP8/EP8, 60 layers):
  gmm_v2-g_64-m_163840-k_4096-act_silu   21.496 ms/step/core  (gate+up, w1)
  gmm_v2-g_64-m_163840-k_1024-act_None   12.818 ms/step/core  (down,     w2)
  -> 34.314 ms/step = 11.67% of a 294.09 ms step, over 60 layers
  -> per layer per core: 21.496/60 = 358.3 us and 12.818/60 = 213.6 us

Shapes here are per layer per core: 64 local experts, m padded to 163840 routed
rows of which 20480 are real (topk 10 / EP8 over 512 global experts x 16384
tokens), hidden 4096, intermediate 1024 (so w1 n = 2*1024 = 2048).

Also prints get_metadata(cfgs) for each config, which names the dtype the MXU
actually sees (lhs_cfgs.quant_dtype / rhs_cfgs.quant_dtype / acc_dtype).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _bench_util import banner, bench, env_report  # noqa: E402

from vllm_torchtpu.kernels.megablox import gmm_v2 as gmm_mod  # noqa: E402
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2  # noqa: E402

G = 64
M_PAD = 163840
REAL_ROWS = 20480
HIDDEN = 4096
INTER = 1024

SHAPES = {
    # name: (k, n, fuse_act)
    "w1 gate+up k4096 n2048 act_silu": (HIDDEN, 2 * INTER, "silu"),
    "w2 down     k1024 n4096 act_None": (INTER, HIDDEN, None),
}
PROFILE_US = {
    "w1 gate+up k4096 n2048 act_silu": 21496.0 / 60,
    "w2 down     k1024 n4096 act_None": 12818.0 / 60,
}


def make_inputs(k, n, dev, lhs_dtype=jnp.bfloat16, rhs_dtype=jnp.float8_e4m3fn,
                balanced=True, seed=0):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    lhs = (jax.random.normal(k1, (M_PAD, k), jnp.float32) *
           0.05).astype(lhs_dtype)
    w = jax.random.normal(k2, (G, k, n), jnp.float32) * 0.05
    if rhs_dtype == jnp.float8_e4m3fn:
        amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)  # per (group, n)
        scale = amax / float(jnp.finfo(jnp.float8_e4m3fn).max)
        rhs = (w / scale).astype(jnp.float8_e4m3fn)
        # [size_group, num_blocks=1, 1, size_n]
        rhs_scale = scale.reshape(G, 1, 1, n).astype(jnp.float32)
    else:
        rhs = w.astype(rhs_dtype)
        rhs_scale = None
    if balanced:
        gs = np.full(G, REAL_ROWS // G, dtype=np.int32)
    else:
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.full(G, 2.0))
        gs = np.floor(p * REAL_ROWS).astype(np.int32)
        gs[0] += REAL_ROWS - int(gs.sum())
    gs = jnp.asarray(gs)
    put = lambda x: None if x is None else jax.device_put(x, dev)  # noqa: E731
    return put(lhs), put(rhs), put(rhs_scale), put(gs)


def show_metadata(tag, lhs, rhs, rhs_scale, gs, fuse_act, maybe_quantize_lhs,
                  acc_dtype=None):
    try:
        from jax.experimental.pallas import tpu as pltpu
        # gmm_v2 itself defaults None -> 90% of VMEM before calling this;
        # calculate_tiling compares against it with >, so None raises.
        vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
        cfgs = gmm_mod.make_gmm_configs(
            lhs, rhs, rhs_scale, None, gs,
            jnp.array([0], dtype=jnp.int32),
            tile_info=gmm_mod.calculate_tiling,
            vmem_limit_bytes=vmem,
            out_dtype=None,
            acc_dtype=acc_dtype,
            maybe_quantize_lhs=maybe_quantize_lhs,
            zero_initialize=False,
            fuse_act=fuse_act,
            rhs_quant_dtype=None,
        )
        md = gmm_mod.get_metadata(cfgs)
        keep = [kk for kk in md
                if any(s in kk for s in ("quant_dtype", "acc_dtype",
                                         "out_dtype", "cfgs.dtype",
                                         "quant_block_size", "tile", "dims"))]
        print(f"META {tag}", flush=True)
        for kk in sorted(keep):
            print(f"     {kk} = {md[kk]}", flush=True)
        print(f"     rhs_cfgs.should_dequantize_before_matmul = "
              f"{cfgs.rhs_cfgs.should_dequantize_before_matmul}", flush=True)
        print(f"     rhs_cfgs.should_dequantize_after_matmul  = "
              f"{cfgs.rhs_cfgs.should_dequantize_after_matmul}", flush=True)
        print(f"     lhs_cfgs.should_quantize = {cfgs.lhs_cfgs.should_quantize}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"META {tag} FAILED: {type(e).__name__}: {str(e)[:400]}",
              flush=True)


def main():
    dev = env_report()
    try:
        from jax.experimental.pallas import tpu as pltpu
        info = pltpu.get_tpu_info()
        print(f"tpu_info: bf16={info.bf16_ops_per_second/1e12:.2f} TF/s/core, "
              f"fp8={info.fp8_ops_per_second/1e12:.2f} TF/s/core, "
              f"int8={info.int8_ops_per_second/1e12:.2f}, "
              f"mem_bw={info.mem_bw_bytes_per_second/1e9:.1f} GB/s/core",
              flush=True)
        peak_fp8 = info.fp8_ops_per_second
        peak_bf16 = info.bf16_ops_per_second
    except Exception as e:  # noqa: BLE001
        print(f"tpu_info unavailable: {e}", flush=True)
        peak_fp8 = peak_bf16 = None

    for name, (k, n, fuse_act) in SHAPES.items():
        flop = 2.0 * REAL_ROWS * k * n
        banner(f"{name}   real FLOP/call = {flop/1e9:.1f} G   "
               f"profile = {PROFILE_US[name]:.1f} us/layer/core")

        # --- production config: bf16 lhs, fp8 rhs, per-channel scale ---
        lhs, rhs, rhs_scale, gs = make_inputs(k, n, dev)
        show_metadata(f"{name} | production (fp8 rhs, maybe_quantize_lhs=True)",
                      lhs, rhs, rhs_scale, gs, fuse_act, True)

        variants = [
            ("fp8 rhs, quant_lhs=True  (PRODUCTION)",
             dict(maybe_quantize_lhs=True)),
            ("fp8 rhs, quant_lhs=False (bf16 MXU)",
             dict(maybe_quantize_lhs=False)),
            ("fp8 rhs, quant_lhs=True, acc=f32",
             dict(maybe_quantize_lhs=True, acc_dtype=jnp.float32)),
        ]
        for vname, kw in variants:
            def f(a, b, s, g, kw=kw):
                return gmm_v2(lhs=a, rhs=b, rhs_scale=s, group_sizes=g,
                              group_offset=jnp.array([0], jnp.int32),
                              zero_initialize=False, fuse_act=fuse_act, **kw)
            us = bench(vname, f, (lhs, rhs, rhs_scale, gs), dev, iters=30)
            if us:
                tf = flop / (us * 1e-6) / 1e12
                line = f"       -> {tf:8.1f} TFLOP/s per core (real rows)"
                if peak_fp8:
                    line += (f"  = {100*tf*1e12/peak_fp8:5.1f}% of fp8 peak"
                             f", {100*tf*1e12/peak_bf16:5.1f}% of bf16 peak")
                print(line, flush=True)
        for _a in (lhs, rhs, rhs_scale):
            _a.delete()

        # --- bf16 weights, no quantization at all ---
        lhs2, rhs2, _, gs2 = make_inputs(k, n, dev, rhs_dtype=jnp.bfloat16)
        show_metadata(f"{name} | bf16 rhs, no scale", lhs2, rhs2, None, gs2,
                      fuse_act, True)

        def fb(a, b, g):
            return gmm_v2(lhs=a, rhs=b, rhs_scale=None, group_sizes=g,
                          group_offset=jnp.array([0], jnp.int32),
                          zero_initialize=False, fuse_act=fuse_act)
        us = bench("bf16 rhs, no quant (bf16 MXU)", fb, (lhs2, rhs2, gs2), dev,
                   iters=30)
        if us:
            tf = flop / (us * 1e-6) / 1e12
            print(f"       -> {tf:8.1f} TFLOP/s per core (real rows)"
                  + (f"  = {100*tf*1e12/peak_bf16:5.1f}% of bf16 peak"
                     if peak_bf16 else ""), flush=True)
        for _a in (lhs2, rhs2):
            _a.delete()

        # --- imbalanced groups + padded-m sensitivity, production config ---
        lhs3, rhs3, sc3, gs3 = make_inputs(k, n, dev)
        _, _, _, gs_imb = make_inputs(k, n, dev, balanced=False)

        def f3(a, b, s, g):
            return gmm_v2(lhs=a, rhs=b, rhs_scale=s, group_sizes=g,
                          group_offset=jnp.array([0], jnp.int32),
                          zero_initialize=False, fuse_act=fuse_act,
                          maybe_quantize_lhs=True)

        bench("fp8 rhs, PRODUCTION, imbalanced groups", f3,
              (lhs3, rhs3, sc3, gs_imb), dev, iters=30)

        banner(f"{name}: sensitivity to the padded m (real rows fixed at "
               f"{REAL_ROWS})")
        for mpad in (40960, 81920, 163840):
            sub = jax.device_put(lhs3[:mpad], dev)
            bench(f"m_pad={mpad}", f3, (sub, rhs3, sc3, gs3), dev, iters=30)
            sub.delete()
        for arr in (lhs3, rhs3, sc3):
            arr.delete()


if __name__ == "__main__":
    main()
