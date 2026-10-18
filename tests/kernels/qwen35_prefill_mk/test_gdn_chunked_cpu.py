import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "0"
import importlib.util
import jax, jax.numpy as jnp, numpy as np
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "vllm_torchtpu", "kernels", "qwen35_prefill_mk", "gdn_chunked.py")
spec = importlib.util.spec_from_file_location("gdn_chunked", P); gc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)

def ref(qkv, b, a, conv_state, rec, conv_w, a_log, dt_bias, n_kq, n_v, d_k, d_v, ks):
    q, k, v, g, beta, new_conv = gc.prepare(qkv, b, a, conv_state, conv_w, None, a_log, dt_bias,
                                             n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, kernel_size=ks)
    def step(s, xs):
        q_t, k_t, v_t, beta_t, g_t = xs
        e = jnp.exp(g_t)
        ks_ = jnp.einsum("hd,hdm->hm", k_t, s)
        vn = beta_t[:, None] * (v_t - e[:, None] * ks_)
        o = e[:, None] * jnp.einsum("hd,hdm->hm", q_t, s) + jnp.sum(q_t * k_t, -1, keepdims=True) * vn
        s = s * e[:, None, None] + jnp.einsum("hd,hm->hdm", k_t, vn)
        return s, o
    s, o = jax.lax.scan(step, rec, (q, k, v, beta, g))
    return new_conv, s, o.reshape(qkv.shape[0], -1)

rng = np.random.default_rng(0)
n_kq, n_v, d_k, d_v, ks, T, C = 2, 8, 128, 128, 4, 256, 64
dim = 2 * n_kq * d_k + n_v * d_v
qkv = jnp.asarray(rng.standard_normal((T, dim)), jnp.float32)
b = jnp.asarray(rng.standard_normal((T, n_v)), jnp.float32)
a = jnp.asarray(rng.standard_normal((T, n_v)), jnp.float32)
conv_state = jnp.asarray(0.1 * rng.standard_normal((ks - 1, dim)), jnp.float32)
rec = jnp.asarray(0.1 * rng.standard_normal((n_v, d_k, d_v)), jnp.float32)
conv_w = jnp.asarray(0.3 * rng.standard_normal((dim, 1, ks)), jnp.float32)
a_log = jnp.asarray(np.log(rng.uniform(1, 16, (n_v,))), jnp.float32)
dt_bias = jnp.asarray(rng.standard_normal((n_v,)), jnp.float32)
rc, rs, ro = ref(qkv, b, a, conv_state, rec, conv_w, a_log, dt_bias, n_kq, n_v, d_k, d_v, ks)
for name, kw in [("scan", dict(use_pallas=False)), ("pallas-interpret", dict(use_pallas=True, interpret=True))]:
    (c2, s2), o2 = gc.gdn_prefill(qkv, b, a, conv_state, rec, conv_w, None, a_log, dt_bias,
                                  n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, kernel_size=ks, chunk=C, **kw)
    rel = lambda x, y: float(jnp.linalg.norm(x - y) / jnp.linalg.norm(y))
    print(f"{name}: out rel {rel(o2, ro):.2e}  state rel {rel(s2, rs):.2e}  conv eq {bool(jnp.all(c2 == rc))}")
