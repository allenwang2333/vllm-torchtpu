"""Interpret-mode test of the prefill MK's MoE phases against reference.moe."""
import os, sys, types
sys.path.insert(0, os.path.dirname(__file__)); import cpu_tpu7x  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R, phases as P

c = R.StageConfig(layer_types=("gdn",), H=256, E=16, TOPK=4, I=128, IS=128)
T, tm, tn, RT = 128, 64, 128, 16
_, ws, _, h, r = R.random_stage(c, T, 0, n_layers=1)
w = ws[0]
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
NPAD = T * c.TOPK + c.E * (RT - 1)

def kernel(h_ref, r_ref, nw, gw, gs, sdw, sds, sgw, w13, w13s, w2, w2s,
           ro, xb, xq, xs, lg, sgu, hq, hs, sd, sg, topi, topw, rank, cnt, ys, pos, out):
    P.norm_add_quant(h_ref, r_ref, nw, ro, xb, xq, xs, eps=c.eps, tm=tm)
    # Router logits: E=16 < tn, so they take one 16-wide tile.
    P.qmatmul(xq, xs, gw, gs, lg, tm=tm, tn=c.E, n0=0, nn=1)
    P.qmatmul(xq, xs, gw, gs, sgu, tm=tm, tn=c.E, n0=1, nn=2 * c.IS // c.E)
    P.swiglu_quant(sgu, hq, hs, inter=c.IS, tm=tm)
    P.qmatmul(hq, hs, sdw, sds, sd, tm=tm, tn=tn)
    P.bmatmul(xb, sgw, sg, tm=tm)
    P.moe_route(lg, topi, topw, rank, cnt, e=c.E, topk=c.TOPK, tm=tm)
    P.moe_experts(xb, topi, rank, cnt, w13, w13s, w2, w2s, ys, pos, e=c.E, topk=c.TOPK, inter=c.I,
                  rt=RT, inflight=8, tm=tm)
    P.moe_combine(ys, pos, topw, sd, sg, out, topk=c.TOPK, tm=tm, inflight=8)

SD = jax.ShapeDtypeStruct
outs = [SD((T, c.H), R.BF16), SD((T, c.H), R.BF16), SD((T, c.H), R.FP8), SD((T, 1), R.F32),
        SD((T, c.E), R.BF16), SD((T, 2 * c.IS), R.BF16), SD((T, c.IS), R.FP8), SD((T, 1), R.F32),
        SD((T, c.H), R.BF16), SD((T, 1), R.F32), SD((T, c.TOPK), jnp.int32), SD((T, c.TOPK), R.F32),
        SD((T, c.TOPK), jnp.int32), SD((1, c.E), jnp.int32), SD((NPAD, c.H), R.BF16),
        SD((T, c.TOPK), jnp.int32), SD((T, c.H), R.BF16)]
call = pl.pallas_call(kernel, out_shape=outs, in_specs=[HBM] * 12, out_specs=[HBM] * len(outs),
                      interpret=pltpu.InterpretParams())
args = (h, r, w["in_norm"][None], w["gate_gu_w"], w["gate_gu_s"][None], w["sd_w"], w["sd_s"][None],
        w["sgate_w"], w["w13"], w["w13_s"][:, None], w["w2"], w["w2_s"][:, None])
o = jax.jit(call)(*args)
out, topi, cnt = o[-1], o[10], o[13]

res = (h.astype(R.F32) + r.astype(R.F32)).astype(R.BF16)
x = R.rmsnorm_zc(h.astype(R.F32) + r.astype(R.F32), w["in_norm"], c.eps).astype(R.BF16)
ref = R.moe(c, w, x).astype(R.BF16)
fused = R.qlinear(x, w["gate_gu_w"], w["gate_gu_s"])
_, ti_ref = jax.lax.top_k(jax.nn.softmax(fused[:, :c.E], -1), c.TOPK)
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / jnp.linalg.norm(b.astype(R.F32)))
print("topk ids eq", bool(jnp.all(jnp.sort(topi, -1) == jnp.sort(ti_ref, -1))), " counts sum", int(cnt.sum()), "==", T * c.TOPK)
print("moe out rel", rel(out, ref))
