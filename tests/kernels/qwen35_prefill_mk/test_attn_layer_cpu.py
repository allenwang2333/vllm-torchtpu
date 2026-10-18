"""Interpret-mode test of the prefill MK's attention layer phases against reference.attn_layer."""
import os, sys, types
sys.path.insert(0, os.path.dirname(__file__)); import cpu_tpu7x  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
pkg = types.ModuleType("vllm_torchtpu"); pkg.__path__ = [SRC + "/vllm_torchtpu"]; sys.modules["vllm_torchtpu"] = pkg
k = types.ModuleType("vllm_torchtpu.kernels"); k.__path__ = [SRC + "/vllm_torchtpu/kernels"]; sys.modules["vllm_torchtpu.kernels"] = k
from vllm_torchtpu.kernels.qwen35_prefill_mk import reference as R, phases as P

c = R.StageConfig(layer_types=("attn",), H=256, NH=4, NKV=2, HD=128, ROT=32, E=128, IS=128, I=128)
T, CTX, tm, bq, bkv = 128, 64, 64, 64, 64
S = CTX + T
_, ws, sts, h, r = R.random_stage(c, T, CTX, n_layers=1)
w, st = ws[0], sts[0]
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
half = c.ROT // 2
inv = 1.0 / (c.theta ** (np.arange(half) * 2.0 / c.ROT))
ang = np.arange(S)[:, None] * inv[None]
rope = jnp.asarray(np.concatenate([np.cos(ang), np.sin(ang)], -1), jnp.float32)
NT = 2 * c.NH + 2 * c.NKV

def kernel(ctx_ref, h_ref, r_ref, nw, qw, qs, qnw, knw, rope_ref, kc_in, vc_in, ow, os_,
           ro, xb, xq, xs, att, qn, o, gq, gs, y, kc, vc):
    ctx = ctx_ref[0]
    P.norm_add_quant(h_ref, r_ref, nw, ro, xb, xq, xs, eps=c.eps, tm=tm)
    P.qmatmul_heads(xq, xs, qw, qs, att, tm=tm, nh=c.NH, nkv=c.NKV, hd=c.HD)
    P.att_qk_norm_rope_kv(att, qnw, knw, rope_ref, ctx, qn, kc, vc, nh=c.NH, nkv=c.NKV, hd=c.HD,
                          rot=c.ROT, eps=c.eps, tm=tm)
    P.att_flash(qn, kc, vc, ctx, o, nh=c.NH, nkv=c.NKV, hd=c.HD, bq=bq, bkv=bkv, s_max=S)
    P.att_gate_quant(o, att, gq, gs, nh=c.NH, hd=c.HD, tm=tm)
    P.qmatmul(gq, gs, ow, os_, y, tm=tm, tn=128)

SD = jax.ShapeDtypeStruct
outs = [SD((T, c.H), R.BF16), SD((T, c.H), R.BF16), SD((T, c.H), R.FP8), SD((T, 1), R.F32),
        SD((NT, T, c.HD), R.BF16), SD((c.NH, T, c.HD), R.BF16), SD((c.NH, T, c.HD), R.F32),
        SD((T, c.NH * c.HD), R.FP8), SD((T, 1), R.F32), SD((T, c.H), R.BF16),
        SD((c.NKV, S, c.HD), R.FP8), SD((c.NKV, S, c.HD), R.FP8)]
call = pl.pallas_call(
    kernel, out_shape=outs,
    grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(), in_specs=[HBM] * 12,
                                           out_specs=[HBM] * len(outs)),
    input_output_aliases={9: len(outs) - 2, 10: len(outs) - 1},   # kc, vc (indices count the scalar)
    interpret=pltpu.InterpretParams())
kc0 = jnp.swapaxes(st["k"], 0, 1); vc0 = jnp.swapaxes(st["v"], 0, 1)
args = (jnp.array([CTX], jnp.int32), h, r, w["in_norm"][None], w["qkv_w"], w["qkv_s"][None],
        w["q_norm"][None], w["k_norm"][None], rope, kc0, vc0, w["o_w"], w["o_s"][None])
o = jax.jit(call)(*args)
y, kc, vc = o[9], o[10], o[11]

res = (h.astype(R.F32) + r.astype(R.F32)).astype(R.BF16)
x = R.rmsnorm_zc(h.astype(R.F32) + r.astype(R.F32), w["in_norm"], c.eps).astype(R.BF16)
y_ref, k_ref, v_ref = R.attn_layer(c, w, x, st["k"], st["v"], CTX)
rel = lambda a, b: float(jnp.linalg.norm(a.astype(R.F32) - b.astype(R.F32)) / jnp.linalg.norm(b.astype(R.F32)))
print("attn out rel", rel(y, y_ref), " k cache rel", rel(jnp.swapaxes(kc, 0, 1), k_ref),
      " v cache eq", bool(jnp.all(jnp.swapaxes(vc, 0, 1).astype(R.F32) == v_ref.astype(R.F32))))
