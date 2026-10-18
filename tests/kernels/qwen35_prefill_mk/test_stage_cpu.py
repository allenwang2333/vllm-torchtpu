"""Interpret-mode test: the whole stage in ONE pallas_call vs reference.stage_step."""
import os, sys, types
sys.path.insert(0, os.path.dirname(__file__)); import cpu_tpu7x  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R, megakernel as MK

c = R.StageConfig(layer_types=("gdn", "attn", "gdn"), H=256, NH=4, NKV=2, HD=128, ROT=32, KH=2, VH=4,
                  E=16, TOPK=4, I=128, IS=128)
T, CTX = 128, 64
S = CTX + T
tiles = MK.MKTiles(tm_norm=64, tm=64, tn=128, chunk=64, bq=64, bkv=64, rt=16, tm_route=64,
                   tm_combine=64, inflight=8, vmem_mib=64)
types_, ws, sts, h, r = R.random_stage(c, T, CTX)
kw = [MK.kernel_weight_layout(c, w, lt) for w, lt in zip(ws, types_)]
ks = [dict(st) if lt == "gdn" else {"k": jnp.swapaxes(st["k"], 0, 1), "v": jnp.swapaxes(st["v"], 0, 1)}
      for st, lt in zip(sts, types_)]
# Reference first: the kernel updates its (aliased) state operands in place.
hr, rr, sr = jax.block_until_ready(R.stage_step(c, ws, h, r, sts, CTX, flash_bkv=tiles.bkv))
ks = jax.tree.map(jnp.copy, ks)
step = MK.build(c, T, S, tiles, interpret=True)
h2, r2, s2 = jax.jit(step)(CTX, h, r, kw, ks, MK.rope_table(c, S))
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / jnp.linalg.norm(b.astype(R.F32)))
# Noise floor: fp8 activation quantization and top-k routing make the stage chaotic, so compare
# with how far the reference itself moves when 1% of its input elements move by one bf16 ulp.
mask = jax.random.uniform(jax.random.PRNGKey(1), h.shape) < 0.01
h_eps = jnp.where(mask, jnp.nextafter(h, jnp.full_like(h, jnp.inf)), h)
hn, rn, _ = jax.block_until_ready(R.stage_step(c, ws, h_eps, r, jax.tree.map(jnp.copy, sts), CTX, flash_bkv=tiles.bkv))
floor_h, floor_r = rel(hn, hr), rel(rn, rr)
print(f"hidden rel {rel(h2, hr):.3e} (noise floor {floor_h:.3e})  residual rel {rel(r2, rr):.3e} (floor {floor_r:.3e})")
print("PASS" if rel(h2, hr) < 3 * floor_h + 1e-3 and rel(r2, rr) < 3 * floor_r + 1e-3 else "FAIL")
for li, lt in enumerate(types_):
    if lt == "gdn":
        print(f"  layer {li} gdn: rec rel {rel(s2[li]['rec'], sr[li]['rec']):.2e} conv eq {bool(jnp.all(s2[li]['conv'] == sr[li]['conv']))}")
    else:
        print(f"  layer {li} attn: k rel {rel(jnp.swapaxes(s2[li]['k'], 0, 1), sr[li]['k']):.2e} v rel {rel(jnp.swapaxes(s2[li]['v'], 0, 1), sr[li]['v']):.2e}")
