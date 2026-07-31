"""A5: run compress_norm_rope_store standalone and compare CPU vs TPU.

Established:
  * On TPU the compressor emits rope ~1e29 (decode) / ~1e31 (prefill).
  * RMSNorm cannot emit those from any finite input (it normalises to O(1)).
  * quantize_fp8_ue8m0 cannot emit the observed byte combination (A1).
  * The identical assembly path produces a perfect record on CPU (A2).

=> the defect is in the TPU lowering of the write path, not the arithmetic.

This runs the real compress_norm_rope_store with a healthy synthetic window and
prints the written row. Run twice:
    JAX_PLATFORMS=cpu ...   -> expect a healthy record (448/128/7, scale ~119)
    JAX_PLATFORMS=tpu ...   -> if rope blows up, bug reproduced with NO model
Then bisect inside the function.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import (
    compress_norm_rope, compress_norm_rope_store, gather_state_windows,
    unpack_sparse_kv_cache)

HEAD_DIM, ROPE_DIM, CR, QB = 512, 64, 4, 64
OVERLAP, COFF = True, 2
SB = 4  # state_block_size
STATE_DIM = 2 * COFF * HEAD_DIM  # 2048
MAX_BLOCKS = 32
TARGET_KV_SLOT = 259
POS0 = 15


def build():
    rng = np.random.default_rng(0)
    n_tok = 8
    positions = jnp.asarray([POS0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
    tok2req = jnp.zeros((n_tok, ), dtype=jnp.int32)

    bt = np.zeros((1, MAX_BLOCKS), dtype=np.int32)
    bt[0, :4] = [6, 7, 8, 9]
    block_table = jnp.asarray(bt)

    # State view [num_state_pages, SB, STATE_DIM], pages 8 and 9 populated so
    # the whole 8-wide window (positions 8..15) is healthy O(1) data.
    st = np.zeros((16, SB, STATE_DIM), dtype=np.float32)
    for pg in (8, 9):
        st[pg] = rng.normal(0, 1.0, (SB, STATE_DIM)).astype(np.float32)
    state_cache = jnp.asarray(st)

    slot_mapping = jnp.asarray([9 * SB + (POS0 % SB)] + [-1] * 7,
                               dtype=jnp.int32)
    kv_slot_mapping = jnp.asarray([TARGET_KV_SLOT] + [-1] * 7, dtype=jnp.int32)

    cache = jnp.zeros((4, 64, 4, 640), dtype=jnp.uint8)
    rms_w = jnp.asarray(rng.normal(1.0, 0.05, (HEAD_DIM, )).astype(np.float32))
    cos_sin = jnp.asarray(
        rng.normal(0, 1.0, (64, ROPE_DIM)).astype(np.float32))
    return (cache, state_cache, positions, slot_mapping, block_table, tok2req,
            kv_slot_mapping, rms_w, cos_sin)


def main():
    plat = os.environ.get("JAX_PLATFORMS", "?")
    print(f"--- backend: {plat}  (jax default: {jax.default_backend()}) ---")
    (cache, state_cache, positions, slot_mapping, block_table, tok2req,
     kv_slot_mapping, rms_w, cos_sin) = build()

    # Stage 1: the gather, in isolation.
    kvw, scw, vm = gather_state_windows(state_cache=state_cache,
                                        positions=positions,
                                        block_table=block_table,
                                        token_to_req_indices=tok2req,
                                        block_size=SB,
                                        head_dim=HEAD_DIM,
                                        compress_ratio=CR,
                                        overlap=OVERLAP)
    kv0 = np.asarray(kvw)[0]
    print(f"gather : kv_window absmax={np.abs(kv0).max():.6g} "
          f"nonzero_rows={int((np.abs(kv0).max(axis=-1) > 0).sum())}/"
          f"{kv0.shape[0]} valid={np.asarray(vm)[0].tolist()}")

    # Stage 2: the pooled/normed/roped vector, before quantisation.
    comp = compress_norm_rope(kv_window=kvw,
                              score_window=scw,
                              valid_mask=vm,
                              rms_weight=rms_w,
                              cos_sin_cache=cos_sin,
                              compressed_pos=(positions // CR) * CR,
                              rms_eps=1e-6,
                              rope_head_dim=ROPE_DIM)
    c0 = np.asarray(comp)[0]
    print(f"compress: nope_absmax={np.abs(c0[:448]).max():.6g} "
          f"rope_absmax={np.abs(c0[448:]).max():.6g}")

    # Stage 3: the full store (quantise + pack + scatter).
    out = compress_norm_rope_store(cache=cache,
                                   state_cache=state_cache,
                                   positions=positions,
                                   slot_mapping=slot_mapping,
                                   block_table=block_table,
                                   token_to_req_indices=tok2req,
                                   kv_slot_mapping=kv_slot_mapping,
                                   rms_weight=rms_w,
                                   cos_sin_cache=cos_sin,
                                   state_block_size=SB,
                                   head_dim=HEAD_DIM,
                                   rope_head_dim=ROPE_DIM,
                                   compress_ratio=CR,
                                   overlap=OVERLAP,
                                   rms_eps=1e-6,
                                   quant_block=QB)
    row = np.asarray(out).reshape(-1, 640)[TARGET_KV_SLOT].astype(np.uint8)
    print(f"stored : nz[nope/rope/scale/pad]={int((row[:448] != 0).sum())}/"
          f"{int((row[448:576] != 0).sum())}/{int((row[576:583] != 0).sum())}/"
          f"{int((row[583:640] != 0).sum())}  scale={row[576:583].tolist()}")

    n, r, s = unpack_sparse_kv_cache(jnp.asarray(row.reshape(1, -1)), 448,
                                     ROPE_DIM, QB)
    n = np.asarray(n.astype(jnp.float32))
    r = np.asarray(r.astype(jnp.float32))
    print(f"decoded: nope_absmax={np.abs(n).max():.6g} "
          f"rope_absmax={np.abs(r).max():.6g} "
          f"scale_max={float(np.asarray(s).max()):.6g}")

    bad = (not np.isfinite(r).all()) or np.abs(r).max() > 1e6
    print(f"\n=== {'BUG REPRODUCED' if bad else 'healthy'} on {plat} ===")


if __name__ == "__main__":
    main()
