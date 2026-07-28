"""Empirical repro: fill SWA cache with a poison byte pattern, run a single
fresh-sequence prefill write, and check which in-range byte positions still
show the poison pattern afterward (meaning the write kernel never touched
them, even though they're within the nominally-valid kv_len range).
"""
import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4 import mla_swa
from vllm_torchtpu.kernels.deepseek_v4.mla import get_dtype_packing, cdiv


def main():
    print(f"JAX Backend: {jax.default_backend()}")

    kv_dtype = jnp.bfloat16
    q_dtype = jnp.bfloat16
    kv_packing = get_dtype_packing(kv_dtype)

    num_heads = 8
    head_dim = 512
    sliding_window = 128  # matches HCA-ish window sizes in the real config
    page_size = 16  # matches the real dump's page layout
    batch_size = 1

    pages_per_seq = cdiv(sliding_window * 2, page_size)
    # Match the real dump's actual first physical page index (1, not 0) --
    # test whether the specific starting page number matters.
    page_indices = jnp.arange(1, 1 + batch_size * pages_per_seq, dtype=jnp.int32)
    total_pages_needed = int(page_indices.max()) + 1

    sw_physical_page_size = page_size + 4

    # POISON FILL: 0xFF everywhere, instead of the test harness's jnp.zeros.
    poison_byte = 0xFF
    swc_cache = jnp.full(
        (total_pages_needed, sw_physical_page_size // kv_packing,
         get_dtype_packing(jnp.uint8), 640),
        poison_byte,
        dtype=jnp.uint8,
    )

    attention_sinks = jnp.zeros((num_heads,), dtype=jnp.float32)

    # Single fresh sequence, 15-token prefill, but the scheduler pads the
    # batch dimension to 16 (next power of 2) -- matches the real failing
    # dump exactly (q.shape=[16,...], kv_lens=[15,...]).
    real_tokens = 15
    padded_tokens = 16
    kv_lens = jnp.array([real_tokens], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, real_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)

    rng = np.random.default_rng(1234)
    q = jnp.array(
        rng.random(size=(padded_tokens, num_heads, head_dim), dtype=np.float32)
    ).astype(q_dtype)
    new_kv = jnp.array(
        rng.random(size=(padded_tokens, head_dim), dtype=np.float32)
    ).astype(kv_dtype)
    total_tokens = real_tokens

    out, updated_cache, L, m = mla_swa.mla_sliding_window_ragged_paged_attention(
        q,
        new_kv,
        swc_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        sm_scale=1.0,
        sliding_window=sliding_window,
        num_queries_per_block=8,
        num_kv_pages_per_block=2,
        logical_page_size=page_size,
    )
    out.block_until_ready()
    updated_cache.block_until_ready()

    print("out has NaN:", bool(jnp.isnan(out).any()))

    # Check the first page: which rows (token positions 0..page_size-1) still
    # have poison bytes anywhere in the nope region (bytes 0:448) after the
    # write? Every row < total_tokens SHOULD have been overwritten.
    page0 = np.asarray(updated_cache[int(page_indices[0])]).reshape(-1, 640)  # [page_size, 640]
    print(f"page0 shape: {page0.shape}")
    for row in range(page0.shape[0]):
        nope = page0[row, :448]
        still_poison = np.sum(nope == poison_byte)
        marker = "  <-- SHOULD BE FULLY WRITTEN" if row < total_tokens else "  (beyond kv_len, ok to be poison)"
        print(f"row={row:2d} poison_byte_count_in_nope={still_poison:3d}/448{marker}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("FAILED:", e)
