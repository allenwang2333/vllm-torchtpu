import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import absltest

# Set JAX to CPU mode
jax.config.update("jax_platform_name", "cpu")

def cdiv(a, b):
    return (a + b - 1) // b

def align_to(x, a):
    return cdiv(x, a) * a

DEFAULT_MASK_VALUE = -1e9

def ref_implementation(
    q,
    cache_kv,
    kv_lens,
    kv_lens_to_attend,
    page_indices,
    cu_q_lens,
    distribution,
    attention_sinks,
    swa_accumution,
    swa_l,
    swa_m,
    topk_indices=None,
    *,
    sm_scale=1.0,
    mask_value=DEFAULT_MASK_VALUE,
):
    actual_lkv_dim = q.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)), constant_values=0)

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    kv_c_cache = cache_kv[..., :lkv_dim].reshape(total_num_pages, page_size, lkv_dim)

    outputs = []
    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        kv_len = kv_lens[i]
        q_i = q[q_start:q_end]
        swa_accumution_i = swa_accumution[q_start:q_end]
        swa_l_i = swa_l[q_start:q_end]
        swa_m_i = swa_m[q_start:q_end]

        indices_start = i * pages_per_seq
        num_pages_i = cdiv(kv_len, page_size)
        indices_end = indices_start + num_pages_i
        indices = page_indices[indices_start:indices_end]

        gathered_kv_c = kv_c_cache[indices]
        flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)
        k_i = flat_kv_c[:kv_len]
        v_i = flat_kv_c[:kv_len]

        attn = jnp.einsum("qnh,kh->nqk", q_i, k_i, preferred_element_type=jnp.float32)
        attn *= sm_scale

        if topk_indices is not None:
            topk_indices_i = topk_indices[q_start:q_end]
            csa_mask = (topk_indices_i[:, :, None] == jnp.arange(kv_len)[None, None, :]).any(axis=1)
            mask = ~csa_mask[None, :, :]
        else:
            kv_lens_to_attend_i = kv_lens_to_attend[q_start:q_end]
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = kv_lens_to_attend_i[None, :, None] <= kv_span
        attn = jnp.where(mask, mask_value, attn)

        m_2 = jnp.max(attn, axis=-1, keepdims=True)
        m_1 = jnp.transpose(swa_m_i, (1, 0))[:, :, None]
        m = jnp.maximum(m_1, m_2)

        l_1 = jnp.transpose(swa_l_i, (1, 0))[:, :, None]
        l_1_scaled = l_1 * jnp.exp(m_1 - m)
        l_2 = jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
        L = l_1_scaled + l_2

        # sinks contribution
        sinks_clean = attention_sinks[:q_i.shape[1]]
        attention_sinks_expanded = sinks_clean[:, None, None]
        exp_cond = (m <= -1e30) | (L <= 1e-10)
        safe_m = jnp.where(exp_cond, 0.0, m)
        safe_sinks = jnp.where(exp_cond, 0.0, attention_sinks_expanded)
        safe_exponent = jnp.where(exp_cond, 0.0, safe_sinks - safe_m)
        exp_attention_sinks = jnp.where(exp_cond, 0.0, jnp.exp(safe_exponent))
        L = L + exp_attention_sinks
        L = jnp.maximum(L, 1.0)

        # accumulated outputs
        o_prev = swa_accumution_i.transpose(1, 0, 2)
        exp_m_diff = jnp.exp(m_1 - m)
        exp_m_diff_safe = jnp.where(m_1 <= -1e30, 0.0, exp_m_diff)
        o_curr = exp_m_diff_safe * o_prev + jnp.einsum("nqk,kh->nqh", jnp.exp(attn - m), v_i)

        out_i = o_curr / L
        outputs.append(out_i.transpose(1, 0, 2))

    return jnp.concatenate(outputs, axis=0)

def test():
    # Setup inputs similar to correctness test
    total_tokens = 64
    num_heads = 8
    head_dim = 512
    batch_size = 4
    page_size = 16
    kv_lens = jnp.array([20, 30, 40, 50], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 16, 32, 48, 64], dtype=jnp.int32)
    q = jax.random.normal(jax.random.PRNGKey(0), (total_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
    
    kv_lens_to_attend = jnp.array([10]*total_tokens, dtype=jnp.int32)
    page_indices = jnp.arange(batch_size * 32, dtype=jnp.int32)
    cache_kv = jax.random.normal(jax.random.PRNGKey(1), (128, 4, 4, 512), dtype=jnp.bfloat16)
    
    swa_accumution = jnp.ones_like(q) * 5000
    swa_l = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 200
    swa_m = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 500
    
    attention_sinks = jax.random.normal(jax.random.PRNGKey(2), (num_heads,), dtype=jnp.float32)
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
    
    out = ref_implementation(
        q,
        cache_kv,
        kv_lens,
        kv_lens_to_attend,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m
    )
    print("Output shape:", out.shape)
    print("Output mean:", out.mean().item(), "std:", out.std().item())
    print("Output max:", out.max().item(), "min:", out.min().item())
    print("First few values:", out[0, 0, :10].tolist())

if __name__ == '__main__':
    test()
