import sys
sys.path.insert(0, '/home/patemotter_google_com/workspace/vllm-torchtpu')
import jax
import jax.numpy as jnp
import numpy as np
from tests.kernels.test_dsv4_mla import ref_implementation, generate_attention_sinks, gen_random, gen_random_int, cdiv, get_kv_cache_shape
import vllm_torchtpu.kernels.deepseek_v4.mla as mla

# Force TPU
jax.config.update('jax_platform_name', 'tpu')

rng = np.random.default_rng(1234)
batch_size = 12
num_heads = 64
head_dim = 512
page_size = 16
num_decode_seqs = batch_size // 2

new_kv_lens = gen_random_int(rng, (batch_size - num_decode_seqs,), 4, 60)
new_kv_lens = jnp.concatenate([jnp.ones((num_decode_seqs,), dtype=jnp.int32), new_kv_lens])
cu_q_lens = jnp.concatenate([jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
kv_lens = new_kv_lens + gen_random_int(rng, (batch_size,), 30, 200)
total_tokens = jnp.sum(new_kv_lens)

kv_dtype = jnp.bfloat16
q_dtype = jnp.bfloat16
q = gen_random(rng, (total_tokens, num_heads, head_dim), q_dtype)

kv_lens_to_attend = []
for i in range(batch_size):
    for _ in range(new_kv_lens[i]):
        kv_lens_to_attend.append(rng.integers(low=0, high=kv_lens[i]))
kv_lens_to_attend = jnp.array(kv_lens_to_attend, dtype=jnp.int32)

pages_per_seq = cdiv(500, page_size)
page_indices = jnp.arange(batch_size * pages_per_seq, dtype=jnp.int32)
total_pages = batch_size * pages_per_seq

cache_kv_base = (gen_random(rng, get_kv_cache_shape(total_pages, page_size, head_dim, kv_dtype), jnp.float32) * 40.0 - 20.0).astype(kv_dtype)

# Quantize
kv_c_flat = cache_kv_base.reshape(total_pages, page_size, head_dim)
fp8_part = kv_c_flat[..., :448]
bf16_part = kv_c_flat[..., 448:512]
fp8_blocked = fp8_part.reshape(total_pages, page_size, 7, 64)
fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
x_amax = jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True)
x_amax = jnp.clip(x_amax, 1e-4, None)
sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))

cpu_device = jax.devices("cpu")[0]
fp8_quant = jax.device_put((fp8_blocked * (1.0 / sf)), cpu_device).astype(jnp.float8_e4m3fn)
fp8_uint8_cpu = jax.lax.bitcast_convert_type(fp8_quant.reshape(total_pages, page_size, 448), jnp.uint8)
fp8_uint8 = jax.device_put(fp8_uint8_cpu, jax.devices("tpu")[0])
bf16_uint8 = jax.lax.bitcast_convert_type(bf16_part, jnp.uint8).reshape(total_pages, page_size, 128)
scales_uint8 = (jnp.log2(sf).reshape(total_pages, page_size, 7) + 127.0).astype(jnp.uint8)
pad_uint8 = jnp.zeros((total_pages, page_size, 57), dtype=jnp.uint8)

flat_cache_agent = jnp.concatenate([fp8_uint8, bf16_uint8, scales_uint8, pad_uint8], axis=-1)
kernel_cache = flat_cache_agent.reshape(total_pages, page_size // 4, 4, 640)

distribution = jnp.array([num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32)
attention_sinks = generate_attention_sinks(rng, num_heads)

swa_accumution = jnp.ones_like(q) * 5000
swa_l = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 200
swa_m = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 500

print("Running reference...")
out_base = ref_implementation(
    q, cache_kv_base, kv_lens, kv_lens_to_attend, page_indices, cu_q_lens,
    distribution, attention_sinks, swa_accumution, swa_l, swa_m, sm_scale=1.0
)

print("Running agent kernel...")
out_agent = mla.mla_ragged_paged_attention(
    q, kernel_cache, kv_lens, kv_lens_to_attend, None, page_indices, cu_q_lens,
    distribution, attention_sinks, swa_accumution, swa_l, swa_m, sm_scale=1.0,
    num_kv_pages_per_block=1, num_queries_per_block=16
)

out_base.block_until_ready()
out_agent.block_until_ready()

print("Analyzing debug outputs per sequence:")
for i in [0, 1, 6]:
    start, end = int(cu_q_lens[i]), int(cu_q_lens[i+1])
    # Extract debug VMEM fields from agent output shape (total_tokens, num_heads, 512)
    # head 0 values:
    l_vals = out_agent[start, 0, :5]
    m_vals = out_agent[start, 0, 128:133]
    out_vals = out_agent[start, 0, 256:261]
    print(f"Sequence {i} (len={end-start}):")
    print("  -> VMEM l (cols 0-4):", l_vals)
    print("  -> VMEM m (cols 128-132):", m_vals)
    print("  -> VMEM out (cols 256-260):", out_vals)
