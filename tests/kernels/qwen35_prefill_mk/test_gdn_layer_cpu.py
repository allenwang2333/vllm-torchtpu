"""Interpret-mode test of the prefill MK's GDN layer phases against reference.gdn_layer."""
import os, sys, types
sys.path.insert(0, os.path.dirname(__file__)); import cpu_tpu7x  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R, phases as P

c = R.StageConfig(layer_types=("gdn",), H=256, KH=2, VH=4, E=128, IS=128, I=128)
T, tm, tn, C = 128, 64, 128, 64
_, ws, sts, h, r = R.random_stage(c, T, 0, n_layers=1)
w, st = ws[0], sts[0]
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
KDIM = c.KH * c.KD

def kernel(h_ref, r_ref, nw, qw, qs, baw, cs, cw, al, dt, s0, gw, ow, os_,
           ro, xb, xq, xs, qkvz, ba, q, kk, v, gb, conv_out, core, s_out, oq, oscale, y):
    P.norm_add_quant(h_ref, r_ref, nw, ro, xb, xq, xs, eps=c.eps, tm=tm)
    P.qmatmul(xq, xs, qw, qs, qkvz, tm=tm, tn=tn)
    P.bmatmul(xb, baw, ba, tm=tm)
    P.gdn_prep(qkvz, ba, cs, cw, al, dt, q, kk, v, gb, conv_out, kq=c.KH, vh=c.VH, kd=c.KD,
               vd=c.VD, ks=c.CONV, eps=c.eps, tm=tm)
    P.gdn_chunks(q, kk, v, gb, s0, core, s_out, kq=c.KH, vh=c.VH, kd=c.KD, vd=c.VD, chunk=C)
    P.gdn_gated_norm_quant(core, qkvz, gw, oq, oscale, vh=c.VH, vd=c.VD, qkv=c.QKV, eps=c.eps, tm=tm)
    P.qmatmul(oq, oscale, ow, os_, y, tm=tm, tn=tn)

S = jax.ShapeDtypeStruct
outs = [S((T, c.H), R.BF16), S((T, c.H), R.BF16), S((T, c.H), R.FP8), S((T, 1), R.F32),
        S((T, c.QKV + c.Z), R.BF16), S((T, 2 * c.VH), R.BF16),
        S((T, KDIM), R.F32), S((T, KDIM), R.F32), S((T, c.VH * c.VD), R.F32), S((T, 2 * c.VH), R.F32),
        S((c.CONV - 1, c.QKV), R.BF16), S((T, c.VH * c.VD), R.BF16), S((c.VH, c.KD, c.VD), R.F32),
        S((T, c.Z), R.FP8), S((T, 1), R.F32), S((T, c.H), R.BF16)]
call = pl.pallas_call(kernel, out_shape=outs, in_specs=[HBM] * 14, out_specs=[HBM] * len(outs),
                      interpret=pltpu.InterpretParams())
args = (h, r, w["in_norm"][None], w["qkvz_w"], w["qkvz_s"][None], w["ba_w"], st["conv"],
        w["conv_w"].T, w["a_log"][None], w["dt_bias"][None], st["rec"], w["gnorm_w"][None],
        w["out_w"], w["out_s"][None])
o = jax.jit(call)(*args)
y, conv_out, s_out = o[-1], o[10], o[12]

res = (h.astype(R.F32) + r.astype(R.F32)).astype(R.BF16)
x = R.rmsnorm_zc(h.astype(R.F32) + r.astype(R.F32), w["in_norm"], c.eps).astype(R.BF16)
y_ref, conv_ref, rec_ref = R.gdn_layer(c, w, x, st["conv"], st["rec"])
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / jnp.linalg.norm(b.astype(R.F32)))
print("gdn out rel", rel(y, y_ref), " state rel", rel(s_out, rec_ref), " conv eq", bool(jnp.all(conv_out == conv_ref)))
