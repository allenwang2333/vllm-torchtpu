"""Cheap pre-bench checks for O8. No timing here -- nothing this prints is a result.

  python3 run_selfcheck.py --rows B-D4-256k,B-D1-1M

Four things, in the order a failure is cheapest to find:
  1. P agrees between this variant's copy of the policy and the shipping one.
  2. The source this copy was taken from is the source in the image (sha256 of
     the shipping module, printed so the report can quote it).
  3. The paired kernel compiles and its two outputs are slot-aligned: for every
     selected slot, the emitted key equals `monotone_key(bits[selected index])`.
  4. The SELECTED INDEX SET matches the incumbent's, exactly, per row. O8 is an
     exact-semantics change, so the symmetric difference must be 0. A nonzero
     value is a finding, not a rounding detail, and is printed either way.
"""
import argparse
import hashlib
import inspect
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import tiling  # noqa: E402
from api import (check_policy_agrees, flat_topk, inc_copy_topk,  # noqa: E402
                 inc_topk, o8_topk)
from pallas_mosaic_tpu_kernel import (KEY_NEG_INF, MONO_MASK,  # noqa: E402
                                      sc_topk_scores_paired)

from bench import synth_scores  # noqa: E402
from shapes import K, ROWS, padded_n  # noqa: E402

import vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk as shipping  # noqa: E402


def _live_lens(name):
    """Per-token-row effective lengths, the same derivation `bench.py` uses in
    its `--decompose` path (which mirrors `_effective_row_lengths`)."""
    r = ROWS[name]
    n = padded_n(r["psz"], r["pages_per_seq"])
    T, R, qlen = r["T"], r["R"], r.get("qlen", 1)
    lens = np.full(R, r["seqlen"], dtype=np.int64)
    if r.get("seqlens_override") == "ragged":
        lens = np.full(R, 4096, dtype=np.int64)
        lens[::4] = r["seqlen"]
    out = np.zeros(T, dtype=np.int32)
    t = 0
    for i in range(R):
        for u in range(qlen):
            out[t] = min(int(lens[i]) - qlen + u + 1, n)
            t += 1
    return T, n, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--skip-pairing", action="store_true")
    a = ap.parse_args()
    rows = [r.strip() for r in a.rows.split(",")]
    fails = []

    print("== provenance ==", flush=True)
    ship_src = inspect.getsource(shipping)
    print(f"  shipping sparsecore_topk sha256 "
          f"{hashlib.sha256(ship_src.encode()).hexdigest()}", flush=True)
    for f in ("pallas_mosaic_tpu_kernel.py", "api.py"):
        h = hashlib.sha256(open(os.path.join(_HERE, f), "rb").read())
        print(f"  O8/{f:32} sha256 {h.hexdigest()}", flush=True)

    print("== 1. P policy agreement ==", flush=True)
    probe = []
    for name in rows:
        T, n, _ = _live_lens(name)
        probe.append((T, n, K))
    bad = check_policy_agrees(probe)
    for b in bad:
        fails.append(f"policy: {b}")
        print(f"  MISMATCH {b}", flush=True)
    for (b, n, k), name in zip(probe, rows):
        p, n_p, k_p, c = tiling.stage_extents(b, n, k)
        note = "p==1 -> two-stage path NOT taken, O8 is a no-op here" if p == 1 \
            else f"stage2 reads C={c} of B*N={b*n}"
        print(f"  {name:16} B={b:5} N={n:8} p={p:3} NP={n_p:7} KP={k_p:5}  "
              f"{note}", flush=True)

    print("== 2/3/4. per-row checks ==", flush=True)
    for name in rows:
        T, n, lens = _live_lens(name)
        p = tiling.pick_partition(T, n, K)
        sc = synth_scores(T, n, lens)
        jax.block_until_ready(sc)
        rl = jnp.asarray(lens, jnp.int32)

        if p > 1 and not a.skip_pairing:
            n_p = n // p
            k_p = min(K, n_p)
            offs = jnp.arange(p, dtype=jnp.int32) * n_p
            lp = jnp.clip(rl[:, None] - offs[None, :], 0, n_p).reshape(T * p)
            # Done on device: at B-P1024-256k the stage-1 slice view is 4096 x
            # 65536 i32, and pulling it to host numpy to gather is a gigabyte
            # this check does not need.
            @jax.jit
            def _pairing(s, L):
                sp = s.reshape(T * p, n_p)
                li_, lk_ = sc_topk_scores_paired(sp, k_p, L)
                bits = jax.lax.bitcast_convert_type(sp, jnp.int32)
                g = jnp.take_along_axis(bits, jnp.maximum(li_, 0), axis=1)
                m = jnp.bitwise_and(jnp.right_shift(g, 31),
                                    jnp.int32(MONO_MASK))
                want_ = jnp.where(li_ >= 0, jnp.bitwise_xor(g, m),
                                  jnp.int32(KEY_NEG_INF))
                return ((want_ != lk_).sum(), (li_ < 0).sum(), lk_.size)

            nmis, nempty, ntot = [int(x) for x in _pairing(sc, lp)]
            print(f"  [{'PASS' if nmis == 0 else 'FAIL'}] {name:16} pairing: "
                  f"{nmis} of {ntot} emitted keys disagree with "
                  f"monotone_key(score[index]); empties={nempty}",
                  flush=True)
            if nmis:
                fails.append(f"{name}: {nmis} mispaired keys")

        got = {}
        for label, fn in (("inc", inc_topk), ("inc_copy", inc_copy_topk),
                          ("flat", flat_topk), ("o8", o8_topk)):
            got[label] = np.asarray(fn(sc, K, rl))
        ref = got["inc"]
        for label in ("inc_copy", "flat", "o8"):
            symdiff = 0
            for t in range(T):
                s0 = set(ref[t][ref[t] >= 0].tolist())
                s1 = set(got[label][t][got[label][t] >= 0].tolist())
                symdiff += len(s0 ^ s1)
            tot = int((ref >= 0).sum())
            st = "PASS" if symdiff == 0 else "DIFF"
            print(f"  [{st}] {name:16} {label:9} vs inc: symmetric difference "
                  f"{symdiff} over {tot} selected ({symdiff / max(tot,1):.3%})",
                  flush=True)
            if symdiff:
                fails.append(f"{name}/{label}: symdiff {symdiff}/{tot}")

    print(f"\nselfcheck: {'PASS' if not fails else 'FAIL'}", flush=True)
    for f in fails:
        print(f"  {f}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
