"""A2: does the record assembly path preserve the fp8 nope region?

Measured on TPU (layer 2, decode step, flat row 259):
    BEFORE compressor op: nz[nope/rope/scale]=448/128/7, scale bytes 119-120
                          (a healthy, fully-populated record)
    AFTER  compressor op: nope 0/448 nonzero, rope absmax 1.585e29, scale 109

So the compressor overwrites good data with a malformed record. A1 already
showed those bytes are unreachable from any single quantize_fp8_ue8m0 call, and
RMSNorm cannot emit 1.585e29 from any finite input (it normalises to O(1)).

Prime suspect: `_to_byte_lane(q)` on float8_e4m3fn. It is the only component
bitcast at itemsize 1, so it skips the reshape branch that bf16/f32 take. If
that bitcast degrades, the nope region zeroes while rope/scale survive --
exactly the observed damage.

FALSIFICATION:
  * fp8 bytes survive here -> the CPU path is sound; the divergence is
    TPU-specific lowering, and the fix is to route fp8 through the same
    explicit-width path bf16/f32 use.
  * fp8 bytes are lost here -> a plain logic bug, fixable and testable offline.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src python3 scratch/cpu_repro/a2_...py
"""
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import (
    _from_byte_lane, _to_byte_lane, quantize_fp8_ue8m0, unpack_sparse_kv_cache)

NOPE, ROPE, QB, WIDTH = 448, 64, 64, 640


def main():
    rng = np.random.default_rng(0)
    # A realistic post-RMSNorm compressed vector: O(1) values.
    nope = jnp.asarray(rng.normal(0, 1.0, (1, NOPE)).astype(np.float32))
    rope = jnp.asarray(rng.normal(0, 1.0, (1, ROPE)).astype(np.float32))

    q, scale_u8 = quantize_fp8_ue8m0(nope, QB)
    rope_q = rope.astype(jnp.bfloat16)

    q_b = _to_byte_lane(q)
    rope_b = _to_byte_lane(rope_q)
    sc_b = _to_byte_lane(scale_u8)
    print("component byte widths:")
    print(f"  q     dtype={q.dtype}  -> bytes {q_b.shape}  "
          f"nonzero={int(np.count_nonzero(np.asarray(q_b)))}")
    print(f"  rope  dtype={rope_q.dtype} -> bytes {rope_b.shape}  "
          f"nonzero={int(np.count_nonzero(np.asarray(rope_b)))}")
    print(f"  scale dtype={scale_u8.dtype}  -> bytes {sc_b.shape}  "
          f"nonzero={int(np.count_nonzero(np.asarray(sc_b)))}")

    record = jnp.concatenate([q_b, rope_b, sc_b], axis=-1)
    record = jnp.pad(record, ((0, 0), (0, WIDTH - record.shape[-1])))
    r = np.asarray(record).reshape(-1).astype(np.uint8)
    print(f"\nrecord width={record.shape[-1]} "
          f"nz[nope/rope/scale/pad]={int((r[:448] != 0).sum())}/"
          f"{int((r[448:576] != 0).sum())}/{int((r[576:583] != 0).sum())}/"
          f"{int((r[583:640] != 0).sum())}")
    print(f"scale bytes = {r[576:583].tolist()}")

    n2, r2, s2 = unpack_sparse_kv_cache(jnp.asarray(record), NOPE, ROPE, QB)
    n2 = np.asarray(n2.astype(jnp.float32))
    r2 = np.asarray(r2.astype(jnp.float32))
    print(f"\nround-trip: nope_absmax={np.abs(n2).max():.6g} "
          f"rope_absmax={np.abs(r2).max():.6g} "
          f"scale_max={float(np.asarray(s2).max()):.6g}")

    # Direct fp8 bitcast fidelity check across the representable range.
    vals = jnp.asarray(
        np.array([[0.0, 1.0, -1.0, 0.5, 448.0, -448.0, 2.0, 256.0]],
                 dtype=np.float32))
    f8 = vals.astype(jnp.float8_e4m3fn)
    via_lane = np.asarray(_to_byte_lane(f8)).reshape(-1)
    via_view = np.asarray(jnp.asarray(f8).view(jnp.uint8)).reshape(-1)
    print(f"\nfp8 bitcast: _to_byte_lane={via_lane.tolist()}")
    print(f"fp8 bitcast: .view(uint8) ={via_view.tolist()}")
    print(f"agree={np.array_equal(via_lane, via_view)}")
    back = _from_byte_lane(jnp.asarray(via_lane.reshape(1, -1)),
                           jnp.float8_e4m3fn)
    print(f"round-trip values={np.asarray(back.astype(jnp.float32)).tolist()}")

    ok = (r[:448] != 0).sum() > 0
    print("\n=== VERDICT ===")
    print("fp8 nope region survives assembly on CPU."
          if ok else "fp8 nope region LOST during assembly on CPU.")


if __name__ == "__main__":
    main()
