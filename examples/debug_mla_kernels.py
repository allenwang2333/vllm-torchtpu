import jax
import jax.numpy as jnp
import numpy as np
from vllm_torchtpu.kernels.deepseek_v4 import mla

print("JAX default backend:", jax.default_backend())

# Setup exact same sizes as test_dsv4_mla.py test_correctness0
batch_size = 12
num_heads = 64
head_dim = 512
page_size = 16
num_decode_seqs = batch_size // 2
is_csa = True
topk = 1024

rng = np.random.default_rng(1234)

new_kv_lens = jnp.array([1, 1, 1, 1, 1, 1, 48, 12, 32, 8, 24, 16], dtype=jnp.int32)
cu_q_lens = jnp.concatenate([jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
kv_lens = new_kv_lens + jnp.array([120, 150, 100, 80, 90, 110, 50, 60, 70, 80, 90, 100], dtype=jnp.int32)
total_tokens = jnp.sum(new_kv_lens)

kv_dtype = jnp.bfloat16
q_dtype = jnp.bfloat16

q = jnp.array(rng.random(size=(total_tokens, num_heads, head_dim), dtype=np.float32)).astype(q_dtype)

topk_indices = []
for i in range(batch_size):
    kv_len_i = int(kv_lens[i])
    for _ in range(new_kv_lens[i]):
        perm = rng.permutation(kv_len_i)
        indices = list(perm[:topk])
        if len(indices) < topk:
            indices.extend([-1] * (topk - len(indices)))
        topk_indices.append(indices)
topk_indices = jnp.array(topk_indices, dtype=jnp.int32)

pages_per_seq = (500 + page_size - 1) // page_size
page_indices = jnp.arange(batch_size * pages_per_seq, dtype=jnp.int32)
total_pages = batch_size * pages_per_seq

cache_kv_base = jnp.array(rng.random(size=(total_pages, page_size, head_dim), dtype=np.float32) * 40.0 - 20.0).astype(kv_dtype)

# Quantize cache_kv_base to create cache_kv_agent in DSV4 FP8 format.
kv_c_flat = cache_kv_base.reshape(total_pages, page_size, head_dim)
fp8_part = kv_c_flat[..., :448]
bf16_part = kv_c_flat[..., 448:512]

fp8_blocked = fp8_part.reshape(total_pages, page_size, 7, 64)
fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
x_amax = jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True)
x_amax = jnp.clip(x_amax, 1e-4, None)
sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))

fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
fp8_uint8 = jax.lax.bitcast_convert_type(fp8_quant.reshape(total_pages, page_size, 448), jnp.uint8)
bf16_uint8 = jax.lax.bitcast_convert_type(bf16_part, jnp.uint8).reshape(total_pages, page_size, 128)
scales_uint8 = (jnp.log2(sf).reshape(total_pages, page_size, 7) + 127.0).astype(jnp.uint8)
pad_uint8 = jnp.zeros((total_pages, page_size, 57), dtype=jnp.uint8)

flat_cache_agent = jnp.concatenate([fp8_uint8, bf16_uint8, scales_uint8, pad_uint8], axis=-1)
kernel_cache = flat_cache_agent.reshape(total_pages, page_size // 4, 4, 640)

distribution = jnp.array([num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32)
attention_sinks = jnp.array(rng.random(size=(num_heads, ), dtype=np.float32) * 300.0 + 200.0)

swa_accumution = jnp.ones_like(q) * 5000
swa_l = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 200
swa_m = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 500

# Run reference
def ref_implementation(q, cache_kv, kv_lens, topk_indices, page_indices, cu_q_lens, distribution, attention_sinks, swa_accumution, swa_l, swa_m):
    lkv_dim = 512
    fp8_part = cache_kv[..., :448]
    bf16_part = cache_kv[..., 448:512]
    fp8_blocked = fp8_part.reshape(total_pages, page_size, 7, 64)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    x_amax = jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True)
    x_amax = jnp.clip(x_amax, 1e-4, None)
    sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))
    fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
    scales_quant = sf.reshape(total_pages, page_size, 7)
    fp8_dequant = (fp8_quant.astype(jnp.bfloat16) * scales_quant[..., None].astype(jnp.bfloat16)).reshape(total_pages, page_size, 448)
    kv_c_cache = jnp.concatenate([fp8_dequant, bf16_part], axis=-1)
    
    print("DEBUG_REF inside: kv_c_cache has NaNs =", jnp.any(jnp.isnan(kv_c_cache)))
    
    outputs = []
    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        kv_len = kv_lens[i]
        q_i = q[q_start:q_end]
        swa_accumution_i = swa_accumution[q_start:q_end]
        swa_l_i = swa_l[q_start:q_end]
        swa_m_i = swa_m[q_start:q_end]
        indices_start = i * pages_per_seq
        num_pages_i = (kv_len + page_size - 1) // page_size
        indices_end = indices_start + num_pages_i
        indices = page_indices[indices_start:indices_end]
        gathered_kv_c = kv_c_cache[indices]
        flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)
        k_i = flat_kv_c[:kv_len]
        v_i = flat_kv_c[:kv_len]
        
        attn = jnp.einsum("qnh,kh->nqk", q_i, k_i, preferred_element_type=jnp.float32)
        attn *= 1.0
        
        topk_indices_i = topk_indices[q_start:q_end]
        csa_mask = (topk_indices_i[:, :, None] == jnp.arange(kv_len)[None, None, :]).any(axis=1)
        mask = ~csa_mask[None, :, :]
        attn = jnp.where(mask, -1e9, attn)
        
        print(f"DEBUG_REF inside seq {i}: attn has NaNs =", jnp.any(jnp.isnan(attn)))
        
        m_2 = jnp.max(attn, axis=-1, keepdims=True)
        m_1 = jnp.transpose(swa_m_i, (1, 0))[:, :, None]
        m = jnp.maximum(m_1, m_2)
        l_1 = jnp.transpose(swa_l_i, (1, 0))[:, :, None]
        l_1_scaled = l_1 * jnp.exp(m_1 - m)
        l_2 = jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
        l_sinks = jnp.exp(attention_sinks[..., None, None] - m)
        L = l_1_scaled + l_2 + l_sinks
        
        print(f"DEBUG_REF inside seq {i}: L has NaNs =", jnp.any(jnp.isnan(L)))
        print(f"DEBUG_REF inside seq {i}: L minimum value =", jnp.min(L))
        
        p_2 = jnp.exp(attn - m)
        acc_2 = jnp.einsum("nqk,kl->qnl", p_2, v_i)
        exp_m1_diff = jnp.transpose(jnp.exp(m_1 - m), (1, 0, 2))
        acc_1_scaled = swa_accumution_i * exp_m1_diff
        acc = acc_1_scaled + acc_2
        
        out_i = (acc / jnp.transpose(L, (1, 0, 2))).astype(q_i.dtype)
        print(f"DEBUG_REF inside seq {i}: out_i has NaNs =", jnp.any(jnp.isnan(out_i)))
        outputs.append(out_i)
        
    return jnp.concatenate(outputs, axis=0)

print("Running Reference...")
out_base = ref_implementation(q, cache_kv_base, kv_lens, topk_indices, page_indices, cu_q_lens, distribution, attention_sinks, swa_accumution, swa_l, swa_m)

print("Running Agent...")
out_agent = mla.mla_ragged_paged_attention(
    q,
    kernel_cache,
    kv_lens,
    None,
    topk_indices,
    page_indices,
    cu_q_lens,
    distribution,
    attention_sinks,
    swa_accumution,
    swa_l,
    swa_m,
    sm_scale=1.0,
    num_kv_pages_per_block=1,
    num_queries_per_block=16,
)

out_base.block_until_ready()
out_agent.block_until_ready()

print("Reference out shape:", out_base.shape, "NaNs:", jnp.any(jnp.isnan(out_base)))
print("Agent out shape:", out_agent.shape, "NaNs:", jnp.any(jnp.isnan(out_agent)))
print("Max Diff:", jnp.max(jnp.abs(out_base - out_agent)))
