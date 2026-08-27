"""CPU numerics check for mega_moe via Pallas interpret mode.

Runs the kernel with interpret=True (no TPU needed) and gates all outputs
against the same jnp reference ops moe_bench uses, including the tail path
(at num_devices=1 the all-reduce degenerates to identity). Timing is
meaningless here; this only catches functional bugs before burning a
dev-node slot.

    PYTHONPATH=src .venv/bin/python scripts/k3/test_mega_moe.py
"""
import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.kimi_k3.mega_moe import (blockify, dequant_fp4,
                                                    mega_moe, prepare_gate,
                                                    quantize_fp4)

T, H, E, K_TOP = 8, 7168, 224, 16
L_SH, L, SI = 1792, 3584, 768  # latent shard / full latent / shared inter
E_LOCAL, I_E = 8, 384
EPS, BETA, LBETA = 1e-5, 4.0, 25.0

key = jax.random.key(0)


def mk(i, shape):
    return (jax.random.normal(jax.random.fold_in(key, i), shape, jnp.bfloat16))


x = mk(0, (T, H))
wg = mk(1, (E, H))
wd = mk(2, (L_SH, H))
wgu = mk(3, (2 * SI, H))
wdn = mk(4, (H, SI))
w13q, w13s = quantize_fp4(mk(5, (E_LOCAL, L_SH, 2 * I_E)))
w2q, w2s = quantize_fp4(mk(6, (E_LOCAL, I_E, L)))
wup = mk(7, (L, L))
gamma = jnp.ones((1, H), jnp.bfloat16)
gamma2 = mk(8, (1, L))

# --- reference (mirrors moe_bench ref_gate/ref_downproj/ref_sharedmlp/
#     ref_experts/ref_tail) ---
logits = jax.lax.dot_general(x.astype(jnp.float32), wg.astype(jnp.float32),
                             (((1, ), (1, )), ((), ())))
scores = jax.nn.sigmoid(logits)
r_wt, r_ids = jax.lax.top_k(scores, K_TOP)
r_wt = r_wt / jnp.sum(r_wt, axis=-1, keepdims=True)
r_lat = jax.lax.dot_general(x,
                            wd, (((1, ), (1, )), ((), ())),
                            preferred_element_type=jnp.float32).astype(x.dtype)
gu = jax.lax.dot_general(x,
                         wgu, (((1, ), (1, )), ((), ())),
                         preferred_element_type=jnp.float32).astype(x.dtype)
g = gu[:, :SI].astype(jnp.float32)
u = gu[:, SI:].astype(jnp.float32)
g = BETA * jnp.tanh(g / BETA) * jax.nn.sigmoid(g)
u = LBETA * jnp.tanh(u / LBETA)
a = (g * u).astype(x.dtype)
r_sh = jax.lax.dot_general(a,
                           wdn, (((1, ), (1, )), ((), ())),
                           preferred_element_type=jnp.float32).astype(x.dtype)
acc = jnp.zeros((T, L), jnp.float32)
ids_local = r_ids % E_LOCAL
for e in range(E_LOCAL):
    m = jnp.sum(jnp.where(ids_local == e, r_wt, 0.0), axis=1)
    w13 = dequant_fp4(w13q[e], w13s[e]).astype(x.dtype)
    w2 = dequant_fp4(w2q[e], w2s[e]).astype(x.dtype)
    ge = jax.lax.dot_general(r_lat,
                             w13, (((1, ), (0, )), ((), ())),
                             preferred_element_type=jnp.float32)
    gg, uu = ge[:, :I_E], ge[:, I_E:]
    gg = BETA * jnp.tanh(gg / BETA) * jax.nn.sigmoid(gg)
    uu = LBETA * jnp.tanh(uu / LBETA)
    ye = jax.lax.dot_general((gg * uu).astype(x.dtype),
                             w2, (((1, ), (0, )), ((), ())),
                             preferred_element_type=jnp.float32)
    acc = acc + ye * m[:, None]
r_exp = acc.astype(x.dtype)
# tail: nd=1 -> all-reduce is identity; then RMS norm + up_proj.
# NB the kernel normalizes the bf16-rounded partial (the nd>1 path sums
# bf16-staged buffers), so the reference must round first too.
f = r_exp.astype(jnp.float32)
inv = jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + EPS)
r_tl = jax.lax.dot_general(
    (f * inv * gamma2.astype(jnp.float32)).astype(x.dtype),
    wup, (((1, ), (1, )), ((), ())),
    preferred_element_type=jnp.float32).astype(x.dtype)

# --- fused, no tail ---
f_wt, f_ids, f_lat, f_sh, f_exp = mega_moe(x,
                                           prepare_gate(wg),
                                           blockify(wd.T, 1792),
                                           blockify(wgu.T, 384),
                                           blockify(wdn.T, 3584),
                                           w13q,
                                           w13s,
                                           w2q,
                                           w2s,
                                           gamma,
                                           n_experts=E,
                                           top_k=K_TOP,
                                           interpret=True)

# --- fused, with tail (nd=1: identity all-reduce) ---
out = mega_moe(x,
               prepare_gate(wg),
               blockify(wd.T, 1792),
               blockify(wgu.T, 384),
               blockify(wdn.T, 3584),
               w13q,
               w13s,
               w2q,
               w2s,
               gamma,
               n_experts=E,
               top_k=K_TOP,
               tail=dict(red_ws=jnp.zeros((2, 1, T, L), jnp.bfloat16),
                         step=jnp.zeros((1, ), jnp.int32),
                         w_up=blockify(wup.T, 512),
                         gamma2=gamma2,
                         axis_name="",
                         num_devices=1,
                         bn_up=512,
                         ahead_up=1),
               interpret=True)
f_tl, f_rw, f_step = out[5], out[6], out[7]

# --- same, but bn_d=256 + explicit union so the down<->up ring union runs ---
out_u = mega_moe(x,
                 prepare_gate(wg),
                 blockify(wd.T, 256),
                 blockify(wgu.T, 384),
                 blockify(wdn.T, 3584),
                 w13q,
                 w13s,
                 w2q,
                 w2s,
                 gamma,
                 n_experts=E,
                 top_k=K_TOP,
                 bn_d=256,
                 ahead_d=2,
                 tail=dict(red_ws=jnp.zeros((2, 1, T, L), jnp.bfloat16),
                           step=jnp.zeros((1, ), jnp.int32),
                           w_up=blockify(wup.T, 512),
                           gamma2=gamma2,
                           axis_name="",
                           num_devices=1,
                           bn_up=512,
                           ahead_up=1,
                           union_up=True),
                 interpret=True)
f_tl_u = out_u[5]


def rel(f, r):
    return (float(
        jnp.max(
            jnp.abs(f.astype(jnp.float32) - r.astype(jnp.float32)) /
            jnp.maximum(jnp.abs(r.astype(jnp.float32)), 1e-2))))


ids_match = float(jnp.mean(f_ids == r_ids))
print(f"topk_weights max_rel={rel(f_wt, r_wt):.3e}")
print(f"topk_ids      match={ids_match:.4f}")
print(f"latent        max_rel={rel(f_lat, r_lat):.3e}")
print(f"shared_out    max_rel={rel(f_sh, r_sh):.3e}")
print(f"expert_out    max_rel={rel(f_exp, r_exp):.3e}")
print(f"tail_out      max_rel={rel(f_tl, r_tl):.3e}")
print(f"tail_out(union) max_rel={rel(f_tl_u, r_tl):.3e}")
print(f"step counter  ={int(f_step[0])} (want 1)")
ok = (rel(f_wt, r_wt) < 5e-3 and ids_match == 1.0 and rel(f_lat, r_lat) < 1e-2
      and rel(f_sh, r_sh) < 5e-3 and rel(f_exp, r_exp) < 1e-2
      and rel(f_tl, r_tl) < 1e-2 and rel(f_tl_u, r_tl) < 1e-2
      and int(f_step[0]) == 1 and bool(jnp.all(jnp.isfinite(f_tl)))
      and bool(jnp.all(jnp.isfinite(f_tl_u))))
print("OK" if ok else "FAIL")
