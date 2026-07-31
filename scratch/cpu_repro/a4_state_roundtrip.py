"""A4: does pack_state_cache -> unpack_state_cache round-trip losslessly?

This is the one link never tested. A5 passed a pre-populated state_cache
straight into compress_norm_rope_store, bypassing the round-trip entirely.

Why it fits the observed asymmetry exactly:
  * PREFILL  -- save_partial_states writes state_view in memory and
    compress_norm_rope_store gathers from that SAME in-memory array. The
    round-trip is bypassed. Observed: a complete, well-formed 583-byte record.
  * DECODE   -- window positions 8..14 must come back THROUGH
    pack -> (buffer) -> unpack. Observed: a 12-nonzero-byte garbage record.

Earlier probing confirmed the state BYTES are present in the buffer, but never
that they decode back to the right VALUES at the right offsets -- which is
precisely what a broken round-trip looks like.

Real geometry (from the live run): state_cache_buf [1922, 4, 1, 4, 8192],
squeezed to [1922, 4, 4, 8192]; state_block_size=4; state_dim=2*coff*head_dim
= 2*2*512 = 2048. _state_chunk_dims then gives kv_slots=16, rpt=4, fpr=512,
bpr=2048.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src python3 scratch/cpu_repro/a4_...py
"""
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import (
    _state_chunk_dims, pack_state_cache, unpack_state_cache)
from vllm_torchtpu.kernels.deepseek_v4.compressor import save_partial_states

SB, STATE_DIM, HEAD_DIM, CR = 4, 2048, 512, 4
NPAGES, ROWS, PACKING, WIDTH = 16, 4, 4, 8192


def main():
    shape = (NPAGES, ROWS, PACKING, WIDTH)
    kv_slots, rpt, fpr, bpr = _state_chunk_dims(shape, SB, STATE_DIM)
    print(f"geometry: kv_slots={kv_slots} rpt={rpt} fpr={fpr} bpr={bpr} "
          f"width={WIDTH}")
    print(f"  bytes used per state token = rpt*bpr = {rpt * bpr}")
    print(f"  bytes spanned per state token = rpt*width = {rpt * WIDTH}")

    rng = np.random.default_rng(0)
    buf = jnp.zeros(shape, dtype=jnp.uint8)
    state = jnp.asarray(
        rng.normal(0, 1.0, (NPAGES, SB, STATE_DIM)).astype(np.float32))

    packed = pack_state_cache(buf, state)
    back = unpack_state_cache(packed, SB, STATE_DIM)

    s0, b0 = np.asarray(state), np.asarray(back)
    same = np.array_equal(s0, b0)
    print(f"\nround-trip exact: {same}")
    if not same:
        bad = np.abs(s0 - b0) > 0
        print(f"  mismatching elements: {int(bad.sum())}/{s0.size}")
        idx = np.argwhere(bad)
        print(f"  first 5 mismatches (page,off,dim): {idx[:5].tolist()}")
        for p, o, d in idx[:5]:
            print(f"    [{p},{o},{d}] wrote {s0[p, o, d]:+.6g} "
                  f"read {b0[p, o, d]:+.6g}")
        # Is it a permutation (right values, wrong places)?
        print(
            f"  sorted-value sets equal (permutation?): "
            f"{np.array_equal(np.sort(s0, axis=None), np.sort(b0, axis=None))}"
        )

    # Now the realistic path: scatter via save_partial_states, then round-trip.
    n_tok = 16
    kv_score = jnp.asarray(
        rng.normal(0, 1.0, (n_tok, STATE_DIM)).astype(np.float32))
    ape = jnp.zeros((CR, 2 * HEAD_DIM), dtype=jnp.float32)
    positions = jnp.arange(n_tok, dtype=jnp.int32)
    slot_mapping = jnp.arange(n_tok, dtype=jnp.int32)  # slots 0..15

    view0 = unpack_state_cache(jnp.zeros(shape, dtype=jnp.uint8), SB,
                               STATE_DIM)
    view1 = save_partial_states(kv_score, ape, positions, view0, slot_mapping,
                                HEAD_DIM, True, CR)
    buf1 = pack_state_cache(jnp.zeros(shape, dtype=jnp.uint8), view1)
    view2 = unpack_state_cache(buf1, SB, STATE_DIM)

    v1, v2 = np.asarray(view1), np.asarray(view2)
    ok = np.array_equal(v1, v2)
    print(f"\nsave_partial_states -> pack -> unpack exact: {ok}")
    if not ok:
        bad = np.abs(v1 - v2) > 0
        print(f"  mismatching elements: {int(bad.sum())}/{v1.size} "
              f"({100.0 * bad.sum() / v1.size:.1f}%)")
        print(f"  in-memory absmax={np.abs(v1).max():.6g}  "
              f"after-roundtrip absmax={np.abs(v2).max():.6g}")

    print("\n=== VERDICT ===")
    if same and ok:
        print("round-trip is lossless; the state buffer is NOT the defect.")
    else:
        print("ROUND-TRIP IS BROKEN -- state written in one step cannot be")
        print("read back in the next. Explains prefill-ok / decode-garbage.")


if __name__ == "__main__":
    main()
