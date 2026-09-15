#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Self-test for ``check_batched_rpa_decode_batch_size.py``'s ``--compare``.

The gate that script implements is only worth anything if it goes red on a bad
arm, and a comparator that cannot fail is the specific way a correctness gate
becomes decorative. This fabricates npz files in the emit format -- **no TPU,
no JAX, seconds on a laptop** -- and drives every comparator branch, healthy
and broken:

    healthy 3-arm / pairwise                        -> rc 0
    injected drop_last_seqs / drop_first_seqs       -> caught  (rc 0)
    injected scale_one_seq:1.05                     -> caught  (rc 0)
    injected scale_one_seq:1.001                    -> NOT caught (rc 1),
                                                       the tolerance's floor,
                                                       asserted so it is on
                                                       the record
    same file compared with itself                  -> INVALID (rc 2)
    an arm that reads past kv_len   (A1 red)        -> rc 1
    an arm that drops 3 sequences   (A2 red)        -> rc 1
    two arms sharing one build hash (GUARD1 red)    -> rc 2
    arms fed different caches       (GUARD2 red)    -> rc 2
    the env var not honoured        (GUARD1 red)    -> rc 2

Synthetic data only: this proves the comparator discriminates, and nothing
whatever about the kernel. The hardware half of that argument is the
``oob_poison`` / ``inband_poison`` / ``overread_ctl`` cases the check itself
runs on the pod.

    python scripts/selftest_check_batched_rpa_decode_batch_size.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK = HERE / "check_batched_rpa_decode_batch_size.py"
PY = sys.executable
_TMPCTX = tempfile.TemporaryDirectory(prefix="rpa-bs-selftest-")
TMP = Path(_TMPCTX.name)

CASES = ("base", "oob_poison", "inband_poison", "overread_ctl")
NSEQ, NH, HD = 64, 4, 128
rng = np.random.default_rng(0)
REF = {c: rng.normal(size=(NSEQ, NH, HD)).astype(np.float32) for c in CASES}
# inband/overread must differ from base on every seq; oob must equal base.
REF["oob_poison"] = REF["base"].copy()
REF["inband_poison"] = REF["base"] + 0.3
REF["overread_ctl"] = REF["base"] + 0.2


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(path, arm, bs, *, jitter, break_a1=False, break_a2=False,
          same_hash_as=None, bad_input=False, unhonoured=False):
    blob, man = {}, {}
    # One jitter draw per arm, shared by base and oob_poison: within an arm
    # those two are bit-identical (the poison is out of range), and only the
    # *arm* reassociates differently from arm 8.
    shared = (rng.normal(size=(NSEQ, NH, HD)).astype(np.float32) * jitter
              if jitter else 0.0)
    for c in CASES:
        out = REF[c].copy()
        if jitter:
            out = out + shared
        if break_a1 and c == "oob_poison":
            out = out + 0.5          # kernel read past kv_len
        if break_a2 and c == "inband_poison":
            out[-3:] = REF["base"][-3:]   # 3 sequences never attended
        blob[f"{c}__out"] = out.astype(np.float32)
        man[c] = dict(
            kv_len_fed=8704 if c != "overread_ctl" else 8960,
            poison_pages=None, in_cache_sha="ic" + ("X" if bad_input else ""),
            q_sha="q", k_sha="k", v_sha="v", out_sha=_sha(out),
            out_shape=list(out.shape), out_dtype="float32",
            cache_valid_sha="cv" + ("Z" if (break_a1 and c == "oob_poison")
                                    else ""),
            out_finite=True, out_std=float(out.std()),
            per_seq_absmax=[float(x) for x in
                            np.abs(out).reshape(NSEQ, -1).max(axis=1)])
    meta = dict(arm=arm, resolved_batch_size=(int(arm) + 1) if unhonoured
                else bs, kernel_name=f"RPAd-p256-b{bs}-q1-k{3840 * 8 // bs}",
                bkv_sz=3840 * 8 // bs, bq_sz=1, n_buffer=3,
                build_hash=same_hash_as or f"hash{bs:02d}", layout="head_along_sublane",
                page_size=256, cache_shape=[2304, 256, 1, 4, 128],
                max_num_seqs=64, max_model_len=9216, decode_kv_len=8704,
                num_live=64, poison_value=6.0, overread_tokens=256, seed=0,
                cases=list(CASES), manifest=man)
    np.savez(path, _meta=np.frombuffer(json.dumps(meta).encode(),
                                       dtype=np.uint8), **blob)


def run(label, argv, want):
    r = subprocess.run([PY, str(CHECK), *argv], capture_output=True, text=True)
    ok = r.returncode == want
    print(f"[{'ok ' if ok else 'BAD'}] {label:<52} rc={r.returncode} "
          f"want={want}")
    if not ok:
        print(r.stdout[-4000:])
        print(r.stderr[-2000:])
    return ok


def main():
    JIT = 2e-4   # legitimate reassociation, well under the 2e-2 tolerance
    a8, a4, a2 = TMP / "b8.npz", TMP / "b4.npz", TMP / "b2.npz"
    write(a8, "8", 8, jitter=0.0)
    write(a4, "4", 4, jitter=JIT)
    write(a2, "2", 2, jitter=JIT)

    ok = True
    ok &= run("healthy: 3 arms", ["--compare", str(a8), str(a4), str(a2)], 0)
    ok &= run("healthy: arm 4 alone", ["--compare", str(a8), str(a4)], 0)
    ok &= run("healthy: arm 2 alone", ["--compare", str(a8), str(a2)], 0)
    ok &= run("neg ctl drop_last_seqs:2",
              ["--compare", str(a8), str(a4), str(a2),
               "--inject-fault", "drop_last_seqs:2", "--inject-into", "2"], 0)
    ok &= run("neg ctl drop_first_seqs:2",
              ["--compare", str(a8), str(a4), str(a2),
               "--inject-fault", "drop_first_seqs:2", "--inject-into", "4"], 0)
    ok &= run("neg ctl scale_one_seq:1.05",
              ["--compare", str(a8), str(a4), str(a2),
               "--inject-fault", "scale_one_seq:1.05", "--inject-into", "2"], 0)
    ok &= run("floor probe scale_one_seq:1.001 (must NOT be caught)",
              ["--compare", str(a8), str(a4), str(a2),
               "--inject-fault", "scale_one_seq:1.001", "--inject-into", "2"], 1)
    ok &= run("vacuity: same file twice", ["--compare", str(a8), str(a8)], 2)

    # Kernel-bug simulations: the gate must go red.
    bad1 = TMP / "b2_a1.npz"
    write(bad1, "2", 2, jitter=JIT, break_a1=True)
    ok &= run("A1 red: arm 2 reads past kv_len",
              ["--compare", str(a8), str(bad1)], 1)
    bad2 = TMP / "b2_a2.npz"
    write(bad2, "2", 2, jitter=JIT, break_a2=True)
    ok &= run("A2 red: arm 2 drops 3 sequences",
              ["--compare", str(a8), str(bad2)], 1)
    bad3 = TMP / "b2_null.npz"
    write(bad3, "2", 2, jitter=JIT, same_hash_as="hash08")
    ok &= run("GUARD1 red: arm 2 shares arm 8's build hash",
              ["--compare", str(a8), str(bad3)], 2)
    bad4 = TMP / "b2_input.npz"
    write(bad4, "2", 2, jitter=JIT, bad_input=True)
    ok &= run("GUARD2 red: arms fed different caches",
              ["--compare", str(a8), str(bad4)], 2)
    bad5 = TMP / "b2_unhon.npz"
    write(bad5, "2", 2, jitter=JIT, unhonoured=True)
    ok &= run("GUARD1 red: env var not honoured",
              ["--compare", str(a8), str(bad5)], 2)

    print("\nSELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
