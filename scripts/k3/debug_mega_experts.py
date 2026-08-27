"""TPU debug probe for the mega_moe expert phase.

Compares fused vs reference expert_out for a few (E_local, ahead_e) combos and
prints error structure to localize ring/streaming bugs. CPU interpret mode is
bit-exact everywhere, so anything wrong here is async-execution related.

    srun env PYTHONPATH=src .venv/bin/python scripts/k3/debug_mega_experts.py
"""
import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.kimi_k3.mega_moe import (blockify, mega_moe,
                                                    prepare_gate)

T, H, E, K_TOP = 8, 7168, 224, 16
L, SI = 1792, 768
I_E = 384
BETA, LBETA = 4.0, 25.0

key = jax.random.key(0)


def mk(i, shape):
    return (jax.random.normal(jax.random.fold_in(key, i), shape, jnp.bfloat16))


x = mk(0, (T, H))
wg = mk(1, (E, H))
wd = mk(2, (L, H))
wgu = mk(3, (2 * SI, H))
wdn = mk(4, (H, SI))
gamma = jnp.ones((1, H), jnp.bfloat16)

logits = jax.lax.dot_general(x.astype(jnp.float32), wg.astype(jnp.float32),
                             (((1, ), (1, )), ((), ())))
scores = jax.nn.sigmoid(logits)
r_wt, r_ids = jax.lax.top_k(scores, K_TOP)
r_wt = r_wt / jnp.sum(r_wt, axis=-1, keepdims=True)
r_lat = jax.lax.dot_general(x,
                            wd, (((1, ), (1, )), ((), ())),
                            preferred_element_type=jnp.float32).astype(x.dtype)


def ref_exp(w13, w2, e_local):
    acc = jnp.zeros((T, L), jnp.float32)
    ids_local = r_ids % e_local
    for e in range(e_local):
        m = jnp.sum(jnp.where(ids_local == e, r_wt, 0.0), axis=1)
        ge = jax.lax.dot_general(r_lat,
                                 w13[e], (((1, ), (0, )), ((), ())),
                                 preferred_element_type=jnp.float32)
        gg, uu = ge[:, :I_E], ge[:, I_E:]
        gg = BETA * jnp.tanh(gg / BETA) * jax.nn.sigmoid(gg)
        uu = LBETA * jnp.tanh(uu / LBETA)
        ye = jax.lax.dot_general((gg * uu).astype(x.dtype),
                                 w2[e], (((1, ), (0, )), ((), ())),
                                 preferred_element_type=jnp.float32)
        acc = acc + ye * m[:, None]
    return acc.astype(x.dtype)


def rel(f, r):
    f, r = f.astype(jnp.float32), r.astype(jnp.float32)
    return float(jnp.max(jnp.abs(f - r) / jnp.maximum(jnp.abs(r), 1e-2)))


def np_row(v):
    return "[" + " ".join(f"{float(x):.1e}" for x in v) + "]"


for e_local, ahead_e, bn_d in ((1, 1, 1792), (8, 2, 1792), (8, 2, 256), (8, 2,
                                                                         896)):
    w13 = mk(5, (e_local, L, 2 * I_E))
    w2 = mk(6, (e_local, I_E, L))
    try:
        f = mega_moe(x,
                     prepare_gate(wg),
                     blockify(wd.T, bn_d),
                     blockify(wgu.T, 384),
                     blockify(wdn.T, 3584),
                     w13,
                     w2,
                     gamma,
                     n_experts=E,
                     top_k=K_TOP,
                     bn_d=bn_d,
                     bn1=384,
                     ahead1=1,
                     ahead_e=ahead_e,
                     vmem_limit_bytes=110 * 1024 * 1024)
        r = ref_exp(w13, w2, e_local)
        fexp = f[4]
        per_row = jnp.max(
            jnp.abs(fexp.astype(jnp.float32) - r.astype(jnp.float32)) /
            jnp.maximum(jnp.abs(r.astype(jnp.float32)), 1e-2),
            axis=1)
        lat_row = jnp.max(
            jnp.abs(f[2].astype(jnp.float32) - r_lat.astype(jnp.float32)) /
            jnp.maximum(jnp.abs(r_lat.astype(jnp.float32)), 1e-2),
            axis=1)
        scale = float(jnp.max(jnp.abs(r.astype(jnp.float32))))
        nrel = float(
            jnp.max(jnp.abs(fexp.astype(jnp.float32) - r.astype(jnp.float32)))
            / scale)
        print(f"E={e_local} ahead_e={ahead_e} bn_d={bn_d}: "
              f"exp_rel={rel(fexp, r):.3e} exp_normrel={nrel:.3e}")
        print(f"  exp per_token={np_row(per_row)}")
        print(f"  lat per_token={np_row(lat_row)}")
    except Exception as ex:
        print(f"E={e_local} ahead_e={ahead_e}: FAIL {type(ex).__name__} "
              f"{str(ex).splitlines()[0][:100]}")
