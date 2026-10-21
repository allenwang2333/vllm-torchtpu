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
"""On device: owner_attention_phase per layer kind, served cache dtypes.

Time per layer = slope between 1 and 11 in-kernel repetitions of the phase
(idempotent), median of 20 calls each; numerics vs the harness references.
Usage: bench_attn_phase_device.py [--ctx 32768,131072] [--req 1]
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout  # noqa: E402


def timed(fn, ops, n=20):
    jf = jax.jit(fn)
    jax.block_until_ready(jf(*ops))
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        jax.block_until_ready(jf(*ops))
        ts.append(time.perf_counter() - s)
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32768,131072")
    ap.add_argument("--req", default="1")
    ap.add_argument("--kinds", default="hca,csa")
    ap.add_argument("--lease_mb", type=float, default=0, help="run the phase on a pool lease")
    ap.add_argument("--lease_extra", default="", help="extra LEASE_ORDER keys, e.g. un_scr:0,scores")
    ap.add_argument("--attn_heads", type=int, default=128, help="group TP: heads attended per row")
    ap.add_argument("--transpose_keys", type=int, default=-1, help="attn_phase.TRANSPOSE_KEYS")
    ap.add_argument("--csa_slots", type=int, default=0, help="attn_phase.CSA_SLOTS override")
    ap.add_argument("--csa_blk", type=int, default=0, help="attn_phase.CSA_BLK override")
    ap.add_argument("--max_ctx", type=int, default=0, help="compiled max context (0: = ctx)")
    ap.add_argument("--layout", choices=("pair", "dsv4"), default="pair",
                    help="cache operands: one layer per kind, or all 61 DSv4-Pro layers "
                         "(the step's per-layer dispatch depth)")
    ap.add_argument("--round", default="rows")
    ap.add_argument("--csa_mode", default="union")
    ap.add_argument("--idx_corr", default="0.0", help="comma list of indexer query correlations")
    ap.add_argument("--ablate", default="", help="comma list of ablation sets joined by +, e.g. kvw,scan+topk")
    a = ap.parse_args()
    print(jax.devices()[0], flush=True)
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP0
    AP0.ROUND_MODE = a.round
    AP0.CSA_MODE = a.csa_mode
    if a.lease_extra:
        AP0.LEASE_ORDER = AP0.LEASE_ORDER + tuple(
            (k.split(":")[0], int(k.split(":")[1])) if ":" in k else k
            for k in a.lease_extra.split(","))
    if a.transpose_keys >= 0:
        AP0.TRANSPOSE_KEYS = bool(a.transpose_keys)
    if a.csa_slots:
        AP0.CSA_SLOTS = a.csa_slots
    if a.csa_blk:
        AP0.CSA_BLK = a.csa_blk
    for ctx in map(int, a.ctx.split(",")):
        for R in map(int, a.req.split(",")):
            rng = np.random.default_rng(0)
            seq = [ctx - 3 - 17 * i for i in range(R)]
            kinds = ("hca", "csa") if a.layout == "pair" else tuple(
                "hca" if i in (0, 1) or i % 2 == 1 else "csa" for i in range(61))
            ls = H.build(rng, kinds, seq, a.max_ctx or ctx, share_tables=a.layout == "dsv4")
            for kind, corr in [(k, float(c)) for k in a.kinds.split(",")
                               for c in (a.idx_corr.split(",") if k == "csa" else ["0"])]:
                layer = (0 if kind == "hca" else 1) if a.layout == "pair" else (
                    31 if kind == "hca" else 30)
                inp = H.make_inputs(rng, ls, kind, idx_corr=corr)
                fn, ops = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=a.attn_heads, lease_bytes=int(a.lease_mb * 2**20), served_dtypes=True,
                                       debug_topk=kind == "csa")
                res = jax.jit(fn)(*ops)
                n_c = len(ls.arrays)
                caches = [csa_cache_layout.slabs_to_words(c) if c.dtype == jnp.uint8 else c
                          for c in res[:n_c]]
                exp_c, exp_o, _ = H.reference(ls, kind, layer, inp)
                hh = a.attn_heads  # heads [0, hh) attended
                o = np.asarray(res[n_c], np.float32)[:, :7 * hh].reshape(R, 7, hh, 512)
                e = np.asarray(exp_o)[:, :, :hh]
                topk_note = ""
                if kind == "csa":
                    # The fp8 MXU head sums (as stock) are not f32-exact, so the
                    # f32 reference's top-k differs in a few boundary entries;
                    # check the kernel's lists against stock and the attention
                    # against the reference fed the kernel's lists.
                    tk = np.asarray(res[n_c + 1])[:, :7]
                    union = [len(set(tk[r_].ravel().tolist()) - {-1}) for r_ in range(R)]
                    topk_note_u = f" union={union} corr={corr}"
                    e = np.asarray(H.reference_with_topk(ls, layer, inp, exp_c, tk))[:, :, :hh]
                    topk_note = (f" topk_eq_stock={H.stock_topk_equal(ls, layer, inp, exp_c, tk)}"
                                 + topk_note_u)
                err = (np.abs(o - e).max() / np.abs(e).max(), np.abs(o - e).mean())
                bad = [i for i, (c, x) in enumerate(zip(caches, exp_c))
                       if not np.array_equal(np.asarray(c), x)]
                fn11, ops11 = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=a.attn_heads, lease_bytes=int(a.lease_mb * 2**20),
                                           served_dtypes=True, reps=11)
                t1, t11 = timed(fn, ops), timed(fn11, ops11)
                per = (np.median(t11) - np.median(t1)) / 10
                lo = (np.percentile(t11, 25) - np.percentile(t1, 75)) / 10
                hi = (np.percentile(t11, 75) - np.percentile(t1, 25)) / 10
                for ab in [x for x in a.ablate.split(",") if x]:
                    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP
                    AP.ABLATE = frozenset(ab.split("+"))
                    f1, o1 = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=a.attn_heads, lease_bytes=int(a.lease_mb * 2**20), served_dtypes=True)
                    f11, o11 = H.build_call(ls, kind, layer, inp, interpret=False, attn_heads=a.attn_heads, lease_bytes=int(a.lease_mb * 2**20),
                                            served_dtypes=True, reps=11)
                    p_ = (np.median(timed(f11, o11)) - np.median(timed(f1, o1))) / 10
                    AP.ABLATE = frozenset()
                    print(f"RESULT kind=attn_phase_ablate layer={kind} R={R} ctx={ctx} "
                          f"skip={ab} us_per_layer={p_*1e6:.1f}", flush=True)
                print(f"RESULT kind=attn_phase heads={a.attn_heads} layer={kind} round={a.round} mode={a.csa_mode} R={R} ctx={ctx} "
                      f"us_per_layer={per*1e6:.1f} [{lo*1e6:.1f},{hi*1e6:.1f}] "
                      f"single_call_us={np.median(t1)*1e6:.0f} err_max_rel={err[0]:.2e} "
                      f"err_mean={err[1]:.2e} caches_bitexact={not bad} bad={bad}{topk_note}",
                      flush=True)


if __name__ == "__main__":
    main()
