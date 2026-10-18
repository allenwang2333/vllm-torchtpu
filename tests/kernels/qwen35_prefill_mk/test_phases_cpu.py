"""Interpret-mode test of the prefill MK's dense phases: the shared-expert path of one layer."""
import os, sys, types
sys.path.insert(0, os.path.dirname(__file__)); import cpu_tpu7x  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R, phases as P

c = R.StageConfig(layer_types=("gdn",), H=256, E=128, IS=128, I=128)
T, tm, tn = 256, 128, 128
_, ws, _, h, r = R.random_stage(c, T, 0, n_layers=1)
w = ws[0]
ANY = pl.BlockSpec(memory_space=pltpu.HBM)

def kernel(h_ref, r_ref, nw_ref, gw_ref, gs_ref, sdw_ref, sds_ref, sgw_ref,
           ro_ref, xb_ref, xq_ref, xs_ref, lg_ref, sgu_ref, hq_ref, hs_ref, sd_ref, sg_ref):
    P.norm_add_quant(h_ref, r_ref, nw_ref, ro_ref, xb_ref, xq_ref, xs_ref, eps=c.eps, tm=tm)
    P.qmatmul(xq_ref, xs_ref, gw_ref, gs_ref, lg_ref, tm=tm, tn=tn, n0=0, nn=c.E // tn)
    P.qmatmul(xq_ref, xs_ref, gw_ref, gs_ref, sgu_ref, tm=tm, tn=tn, n0=c.E // tn, nn=2 * c.IS // tn)
    P.swiglu_quant(sgu_ref, hq_ref, hs_ref, inter=c.IS, tm=tm)
    P.qmatmul(hq_ref, hs_ref, sdw_ref, sds_ref, sd_ref, tm=tm, tn=tn)
    P.bmatmul(xb_ref, sgw_ref, sg_ref, tm=tm)

S = jax.ShapeDtypeStruct
outs = [S((T, c.H), R.BF16), S((T, c.H), R.BF16), S((T, c.H), R.FP8), S((T, 1), R.F32),
        S((T, c.E), R.F32), S((T, 2 * c.IS), R.BF16), S((T, c.IS), R.FP8), S((T, 1), R.F32),
        S((T, c.H), R.F32), S((T, 1), R.F32)]
call = pl.pallas_call(kernel, out_shape=outs, in_specs=[ANY] * 8, out_specs=[ANY] * len(outs),
                      interpret=pltpu.InterpretParams())
args = (h, r, w["in_norm"][None], w["gate_gu_w"], w["gate_gu_s"][None], w["sd_w"], w["sd_s"][None], w["sgate_w"])
ro, xb, xq, xs, lg, sgu, hq, hs, sd, sg = jax.jit(call)(*args)

# Reference pieces.
res = (h.astype(R.F32) + r.astype(R.F32)).astype(R.BF16)
x = R.rmsnorm_zc(res, w["in_norm"], c.eps).astype(R.BF16)
fused = R.qlinear(x, w["gate_gu_w"], w["gate_gu_s"])
sh = (jax.nn.silu(fused[:, c.E:][:, :c.IS]) * fused[:, c.E:][:, c.IS:]).astype(R.BF16)
sd_ref = R.qlinear(sh, w["sd_w"], w["sd_s"])
sg_ref = jax.lax.dot_general(x, w["sgate_w"], (((1,), (0,)), ((), ())), preferred_element_type=R.F32)
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / jnp.linalg.norm(b.astype(R.F32)))
print("residual eq", bool(jnp.all(ro == res)), " x eq", bool(jnp.all(xb == x)))
print("logits rel", rel(lg, fused[:, :c.E]), " shared gu rel", rel(sgu, fused[:, c.E:]))
print("shared down rel", rel(sd, sd_ref), " shared gate rel", rel(sg, sg_ref))
