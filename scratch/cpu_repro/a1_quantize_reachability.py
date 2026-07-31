"""A1: is (all-zero fp8 bytes, scale byte 109) reachable from ANY input?

Observed on TPU for compressed-KV slot 3 (flat row 259), layer 2, decode step:
    nope region [0:448]   -> 0 nonzero bytes
    scale region [576:583]-> all 7 bytes == 109  (=> exponent -18 => 2**-18)

Those look mutually exclusive for a single quantize_fp8_ue8m0 call:
  - all-zero input clips amax to the 1e-4 floor -> exponent -22 -> byte 105
  - byte 109 implies amax ~ 1e-3 -> q = x * 2**18 peaks at 224..448 -> nonzero

FALSIFICATION:
  * If some input yields (q_nonzero == 0 and scale_byte == 109), the row IS a
    single coherent record and the defect is upstream in compress_norm_rope.
  * If NO input yields it, the row is NOT one coherent record -- which breaks
    the "one atomic .at[dest].set(record)" assumption that has been driving the
    whole investigation, and re-opens record assembly / scatter / byte packing.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src python3 scratch/cpu_repro/a1_...py
"""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import \
    quantize_fp8_ue8m0

NOPE_DIM, QBLOCK = 448, 64
TARGET_BYTE = 109


def probe(x_np, label):
    """Return (n_nonzero_fp8_bytes, sorted set of scale bytes)."""
    x = jnp.asarray(x_np.reshape(1, NOPE_DIM).astype(np.float32))
    q, scale_u8 = quantize_fp8_ue8m0(x, QBLOCK)
    qb = np.asarray(jnp.asarray(q).view(jnp.uint8)).reshape(-1)
    sb = np.asarray(scale_u8).reshape(-1).astype(int)
    return int(np.count_nonzero(qb)), sorted(set(sb.tolist())), label


def main():
    rng = np.random.default_rng(0)
    cases = []

    cases.append((np.zeros(NOPE_DIM), "exact zeros"))
    for e in range(-30, 8):
        v = float(2.0**e)
        cases.append((np.full(NOPE_DIM, v), f"constant 2**{e} = {v:.3g}"))
    # Straddle the 1e-4 amax clip and the band byte 109 would imply.
    for v in [
            1e-8, 1e-6, 9.9e-5, 1e-4, 1.01e-4, 8.5e-4, 1.2e-3, 1.709e-3,
            1.71e-3, 1e-2
    ]:
        cases.append((np.full(NOPE_DIM, v), f"constant {v:.4g}"))
    # amax exactly in the byte-109 band, everything else zero.
    for v in [8.6e-4, 1.0e-3, 1.7e-3]:
        a = np.zeros(NOPE_DIM)
        a[0] = v
        cases.append((a, f"single spike {v:.3g}, rest 0"))
    # Denormals / extremes / non-finite.
    cases.append((np.full(NOPE_DIM, 1e-45), "denormal 1e-45"))
    cases.append((np.full(NOPE_DIM, np.nan), "all NaN"))
    cases.append((np.full(NOPE_DIM, np.inf), "all +Inf"))
    cases.append((np.full(NOPE_DIM, -np.inf), "all -Inf"))
    m = np.zeros(NOPE_DIM)
    m[0] = np.nan
    cases.append((m, "one NaN, rest 0"))
    m2 = np.zeros(NOPE_DIM)
    m2[0] = np.inf
    cases.append((m2, "one Inf, rest 0"))
    cases.append((rng.normal(0, 1e-3, NOPE_DIM), "gaussian sigma=1e-3"))
    cases.append((rng.normal(0, 1.0, NOPE_DIM), "gaussian sigma=1"))

    results = [probe(x, lab) for x, lab in cases]

    print(f"{'nonzero_fp8':>12} {'scale_bytes':>24}  input")
    print("-" * 78)
    hits = []
    for nz, sbs, lab in results:
        mark = ""
        if nz == 0 and sbs == [TARGET_BYTE]:
            mark = "   <<< MATCHES OBSERVED"
            hits.append(lab)
        print(f"{nz:>12} {str(sbs):>24}  {lab}{mark}")

    # Exhaustive scan over the reachable scale-byte axis: for every byte value
    # we can produce, record whether q can simultaneously be all-zero.
    print("\nreachability scan (constant inputs, exponent sweep):")
    reach = {}
    for e in np.arange(-40.0, 10.0, 0.25):
        v = float(2.0**e)
        nz, sbs, _ = probe(np.full(NOPE_DIM, v), "")
        for b in sbs:
            reach.setdefault(b, set()).add(nz == 0)
    z_only = sorted(b for b, s in reach.items() if s == {True})
    print(f"  scale bytes reachable at all: {min(reach)}..{max(reach)}")
    print(f"  scale bytes reachable WITH all-zero q: {z_only}")
    print(f"  is {TARGET_BYTE} reachable with all-zero q? "
          f"{TARGET_BYTE in z_only}")

    print("\n=== VERDICT ===")
    if hits:
        print("REACHABLE via:", hits)
        print("-> row 259 can be one coherent record; defect is upstream.")
    else:
        print(f"NOT reachable: no input gives all-zero fp8 AND scale byte "
              f"{TARGET_BYTE}.")
        print("-> row 259 is NOT a single coherent quantize_fp8_ue8m0 record.")
        print("-> the 'one atomic .at[dest].set(record)' assumption is FALSE.")


if __name__ == "__main__":
    main()
