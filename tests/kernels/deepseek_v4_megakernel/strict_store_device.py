# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device check of the strict-causal indexer bound and the record stores.

    python tests/kernels/deepseek_v4_megakernel/strict_store_device.py

Compiles owner_attention_phase (served uint8 slabs) for a CSA and an HCA
layer over the requests of test_strict_causal.SEQ and
test_store_granules.CASES in one call, once per TPU_STREAMIDX_STRICT_CAUSAL
value: caches bit-exact vs the references, every top-k row equal to stock
rope_quant + streamindex_topk (SparseCore top-k) with the same flag,
attention within the phase tolerance. Controls: stock top-k with the other
flag must differ on some row; attn_phase.STORE_FAULT = "two_runs" is run and
its corrupted requests printed (on a device the race's winner is timing).
Prints RESULT lines; exits 1 if a check fails. `--interpret`: a CPU dry run
of the same script (word arrays, no stock SparseCore comparison).
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402
import test_store_granules as TSG  # noqa: E402
import test_strict_causal as TSC  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402

Q = 7
SEQ = TSC.SEQ + TSG.SEQ


INTERPRET = False


def run(ls, kind, layer, inp, strict):
    fn, ops = H.build_call(ls, kind, layer, inp, interpret=INTERPRET,
                           served_dtypes=not INTERPRET,
                           debug_topk=kind == "csa", strict_causal=strict)
    res = jax.jit(fn)(*ops)
    n_c = len(ls.arrays)
    u8 = set() if INTERPRET else H.u8_roles(ls)
    caches = [np.asarray(csa_cache_layout.slabs_to_words(c)) if i in u8 else np.asarray(c)
              for i, c in enumerate(res[:n_c])]
    tk = np.asarray(res[n_c + 1])[:, :Q] if kind == "csa" else None
    return caches, np.asarray(res[n_c], np.float32), tk


def att_err(o, e):
    R = e.shape[0]
    o = o[:R, :Q * 128].reshape(R, Q, 128, 512)
    err = np.abs(o - np.asarray(e))
    return float(err.max() / np.abs(e).max()), float(err.mean())


def main():
    global INTERPRET
    p = argparse.ArgumentParser()
    p.add_argument("--interpret", action="store_true")
    INTERPRET = p.parse_args().interpret
    ok = True
    rng = np.random.default_rng(31)
    ls = H.build(rng, ("hca", "csa"), SEQ, 8192)
    inp = {k: H.make_inputs(rng, ls, k) for k in ("hca", "csa")}
    for strict in (False, True):
        caches, out, _ = run(ls, "hca", 0, inp["hca"], strict)
        exp_c, exp_o, _ = H.reference(ls, "hca", 0, inp["hca"])
        bad = [i for i, (c, x) in enumerate(zip(caches, exp_c)) if not np.array_equal(c, x)]
        e = att_err(out, exp_o)
        good = not bad and e[0] <= 1e-2 and e[1] <= 1e-3
        ok &= good
        print("RESULT " + json.dumps(dict(kind="hca", strict=strict, caches_differ=bad,
                                          att_maxrel_meanabs=e, ok=good)), flush=True)

        caches, out, tk = run(ls, "csa", 1, inp["csa"], strict)
        exp_c, exp_o, _ = H.reference(ls, "csa", 1, inp["csa"], strict_causal=strict)
        bad = [i for i, (c, x) in enumerate(zip(caches, exp_c)) if not np.array_equal(c, x)]
        e = att_err(out, exp_o)
        if INTERPRET:  # no SparseCore: the reference's lists stand in for stock's
            same = other = -1
        else:
            same = H.stock_topk_equal(ls, 1, inp["csa"], exp_c, tk, strict_causal=strict)
            other = H.stock_topk_equal(ls, 1, inp["csa"], exp_c, tk, strict_causal=not strict)
        lens = TSC.stock_lens(SEQ, Q, strict)
        zero_rows = [(r, t) for r in range(len(SEQ)) for t in range(Q) if lens[r, t] == 0]
        zero_ok = all((tk[r, t] == -1).all() for r, t in zero_rows)
        stock_ok = INTERPRET or (same == len(SEQ) * Q and other < same)
        good = (not bad and e[0] <= 1e-2 and e[1] <= 1e-3 and stock_ok and zero_ok
                and bool(np.isfinite(out).all()))
        ok &= good
        print("RESULT " + json.dumps(dict(
            kind="csa", strict=strict, caches_differ=bad, att_maxrel_meanabs=e,
            topk_rows_equal_stock_same_flag=same, topk_rows_equal_stock_other_flag=other,
            rows=len(SEQ) * Q, zero_bound_rows=len(zero_rows), zero_bound_rows_all_minus1=zero_ok,
            ok=good)), flush=True)

    AP.STORE_FAULT = "two_runs"
    caches, _, _ = run(ls, "csa", 1, inp["csa"], False)
    AP.STORE_FAULT = ""
    exp_c, _, _ = H.reference(ls, "csa", 1, inp["csa"])
    corrupted = {role: sorted(TSG._diff_requests(ls, 1, role, caches, exp_c))
                 for role in ("kv", "kv_rope", "indexer_k")}
    print("RESULT " + json.dumps(dict(control="store_fault_two_runs",
                                      corrupted_requests=corrupted,
                                      first_store_case_request=len(TSC.SEQ))), flush=True)
    print(f"RESULT all_ok={ok}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
