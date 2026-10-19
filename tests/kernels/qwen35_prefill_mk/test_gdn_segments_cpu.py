"""Interpret-mode test: GDN layer phases over two sequences in the padded segment layout."""
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
T, tm, tn, C = 512, 64, 128, 64
lens, starts, slots_of = [150, 200], [0, 256], [2, 0]
NSLOT = 3
_, ws, _, h, r = R.random_stage(c, T, 0, n_layers=1)
w = ws[0]
rng = np.random.default_rng(5)
conv0 = jnp.asarray(rng.standard_normal((NSLOT, c.CONV - 1, c.QKV)) * 0.5, R.BF16)
rec0 = jnp.asarray(rng.standard_normal((NSLOT, c.VH, c.KD, c.VD)) * 0.05, R.F32)
tile_seg = np.full((T // P.BLK,), -1, np.int32)
for si, (st, ln) in enumerate(zip(starts, lens)):
    tile_seg[st // P.BLK:(st + -(-ln // P.SEG_ALIGN) * P.SEG_ALIGN) // P.BLK] = si
meta = [jnp.asarray(tile_seg), jnp.asarray(starts, jnp.int32), jnp.asarray(lens, jnp.int32),
        jnp.zeros((2,), jnp.int32), jnp.asarray(slots_of, jnp.int32)]
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
KDIM = c.KH * c.KD

def kernel(ts, sst, sln, sctx, sslot, h_ref, r_ref, nw, qw, qs, baw, cs, cw, al, dt, s0, gw, ow, os_,
           ro, xb, xq, xs, qkvz, ba, q, kk, v, gb, core, oq, oscale, y, conv_out, s_out):
    segs = P.Segments(ts, sst, sln, sctx, sslot)
    P.norm_add_quant(h_ref, r_ref, nw, ro, xb, xq, xs, eps=c.eps, tm=tm)
    P.qmatmul(xq, xs, qw, qs, qkvz, tm=tm, tn=tn)
    P.bmatmul(xb, baw, ba, tm=tm)
    P.gdn_prep(qkvz, ba, cs, cw, al, dt, q, kk, v, gb, conv_out, segs, kq=c.KH, vh=c.VH, kd=c.KD,
               vd=c.VD, ks=c.CONV, eps=c.eps, tm=tm)
    P.gdn_chunks(q, kk, v, gb, s0, core, s_out, segs, kq=c.KH, vh=c.VH, kd=c.KD, vd=c.VD, chunk=C)
    P.gdn_gated_norm_quant(core, qkvz, gw, oq, oscale, vh=c.VH, vd=c.VD, qkv=c.QKV, eps=c.eps, tm=tm)
    P.qmatmul(oq, oscale, ow, os_, y, tm=tm, tn=tn)

S = jax.ShapeDtypeStruct
outs = [S((T, c.H), R.BF16), S((T, c.H), R.BF16), S((T, c.H), R.FP8), S((T, 1), R.F32),
        S((T, c.QKV + c.Z), R.BF16), S((T, 2 * c.VH), R.BF16),
        S((T, KDIM), R.F32), S((T, KDIM), R.F32), S((T, c.VH * c.VD), R.F32), S((T, 2 * c.VH), R.F32),
        S((T, c.VH * c.VD), R.BF16), S((T, c.Z), R.FP8), S((T, 1), R.F32), S((T, c.H), R.BF16),
        S(conv0.shape, R.BF16), S(rec0.shape, R.F32)]
call = pl.pallas_call(
    kernel, out_shape=outs,
    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=5, grid=(), in_specs=[HBM] * 14,
                                           out_specs=[HBM] * len(outs)),
    input_output_aliases={5 + 6: len(outs) - 2, 5 + 10: len(outs) - 1},
    interpret=pltpu.InterpretParams())
args = (*meta, h, r, w["in_norm"][None], w["qkvz_w"], w["qkvz_s"][None], w["ba_w"], jnp.copy(conv0),
        w["conv_w"].T, w["a_log"][None], w["dt_bias"][None], jnp.copy(rec0), w["gnorm_w"][None],
        w["out_w"], w["out_s"][None])
o = jax.jit(call)(*args)
y, conv_out, s_out = o[13], o[14], o[15]
x = R.rmsnorm_zc(h.astype(R.F32) + r.astype(R.F32), w["in_norm"], c.eps).astype(R.BF16)
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / jnp.linalg.norm(b.astype(R.F32)))
for st, ln, sl in zip(starts, lens, slots_of):
    y_ref, conv_ref, rec_ref = R.gdn_layer(c, w, x[st:st + ln], conv0[sl], rec0[sl])
    print(f"seq@{st} len {ln} slot {sl}: out rel {rel(y[st:st + ln], y_ref):.2e} state rel {rel(s_out[sl], rec_ref):.2e} conv eq {bool(jnp.all(conv_out[sl] == conv_ref))}")
print("untouched slot 1:", bool(jnp.all(s_out[1] == rec0[1])) and bool(jnp.all(conv_out[1] == conv0[1])))
