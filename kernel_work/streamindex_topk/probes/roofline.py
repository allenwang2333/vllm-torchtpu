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
"""Modelled floor for `streamindex_topk`. Runs on the host; touches no TPU.

EVERYTHING THIS PRINTS IS MODELLED, NOT MEASURED. It never goes near the shared
experiment log and must never be quoted anywhere it could be read as a
measurement.

The model, stated so it can be argued with
------------------------------------------
Two HBM figures, because the difference between them is the whole question of
whether headroom lives in the design or in the execution:

  t_hbm_ideal       every byte the block logically needs crosses HBM exactly
                    once. This is the floor a perfect implementation of *this
                    computation* would hit, whatever its loop structure.
  t_hbm_structured  the bytes THIS loop structure moves. It differs from ideal
                    in two places: the KV is re-read once per query block
                    (`nbq` times, streamindex_topk.py:418/420 -- bq is the
                    outer loop), and the scores buffer is filled with -inf at
                    :740 before the kernel overwrites it.

  t_mxu             score FLOPs / peak, derated for a short M. The einsum at
                    :304 contracts `(b, bq_sz*num_q_heads, 128) x (b, bkv_sz,
                    128)`, so its M is `bq_sz * num_q_heads`. At decode that is
                    1*32 = 32 rows into a 128-row MXU tile, so three quarters
                    of every pass is thrown away and the model charges 4x.

  floor = max(t_hbm_ideal, t_mxu)

That formula embeds two assumptions in opposite directions and neither is
visible in the table: HBM traffic is assumed to overlap compute perfectly
(optimistic), and the per-DMA fixed cost is assumed fully hidden by the
kernel's double buffering (`bkv_x2_ref`, `bq_x2_ref` -- also optimistic).
Together they make this a genuine LOWER bound rather than a prediction.

What carries no price here
--------------------------
- **The VPU.** After the einsum the kernel does a ReLU, a per-head scale and a
  sum over the head axis on an f32 array of `num_tokens x 32 x kv_len`
  elements -- 2.68e8 elements at `mixed_prod`, which is 1.07 GB of VMEM/register
  traffic, larger than every HBM term in the table combined. `V7X.md` carries
  no VPU peak, so this is left unpriced rather than guessed. It is the single
  largest gap in the model and the most likely reason a measured time exceeds
  the floor below.
- **`sparsecore_topk`.** Runs on the SparseCore. Its time overlaps TensorCore
  time and is charged separately by the microbenchmark's control arm.
- **Relayouts.** No roofline term exists for them and they do not appear in
  profiles either.

MXU precision is reported both ways because which one the einsum actually gets
is not settleable from the source: the operands are `float8_e4m3fn`, but
whether Mosaic emits a native fp8 pass or promotes to bf16 is a lowering
question. Step 2 owns probing it. Where the two answers disagree about what the
block is bound by, the table says so.
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402

# tpu7x, per TensorCore. Source: V7X.md.
HBM_BW = 3364.9e9        # bytes/s
MXU_BF16 = 892.8e12      # FLOP/s
MXU_FP8 = 1785.6e12      # FLOP/s
MXU_TILE_M = 128         # rows per MXU pass


def bytes_for(r: S.Regime):
    """Return (ideal_bytes, structured_bytes, breakdown) for one call, one layer."""
    sub = S.num_sublanes_total(r.max_model_len)
    scores_b = r.num_tokens * sub * 128 * 4
    # Sequences the kernel actually loops KV for. instruments/inputs.py puts
    # every token of a mixed batch in one sequence (the profiled shape), while
    # a decode batch has one token in each of `max_num_seqs`.
    seqs_read = r.max_num_seqs if r.phase == "decode" else 1
    kv_b = seqs_read * r.kv_len * S.LKV_DIM
    q_b = r.num_tokens * S.INDEX_N_HEADS * S.INDEX_HEAD_DIM * 1
    w_b = r.num_tokens * S.INDEX_N_HEADS * 4
    out_b = r.num_tokens * S.INDEX_TOPK * 4

    ideal = kv_b + q_b + w_b + scores_b + scores_b + out_b
    #       KV once                      write    read by topk
    structured = (kv_b * r.num_bq + q_b + w_b
                  + scores_b          # the -inf fill at :740
                  + scores_b          # the kernel's own writes
                  + scores_b          # sparsecore_topk's read
                  + out_b)
    bd = dict(kv_MB=kv_b / 1e6, kv_reread_x=r.num_bq, scores_MB=scores_b / 1e6,
              q_MB=q_b / 1e6, out_MB=out_b / 1e6)
    return ideal, structured, bd


def mxu_for(r: S.Regime):
    """Return (flops, m_rows, derate, t_bf16_s, t_fp8_s)."""
    flops = r.num_tokens * S.INDEX_N_HEADS * S.INDEX_HEAD_DIM * r.kv_len * 2
    bq_sz = 1 if r.phase == "decode" else S.NUM_QUERIES_PER_BLOCK[2]
    m = bq_sz * S.INDEX_N_HEADS
    derate = MXU_TILE_M / min(MXU_TILE_M, m)
    eff = flops * derate
    return flops, m, derate, eff / MXU_BF16, eff / MXU_FP8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", default="",
                    help="regime=us pairs, comma separated, to add an "
                         "'achieved' column, e.g. decode_prod=76.42")
    a = ap.parse_args()
    meas = {}
    for kv in [x for x in a.measured.split(",") if x]:
        n, v = kv.split("=")
        meas[n] = float(v)

    print("MODELLED. tpu7x, one TensorCore, one layer, one call. "
          "All times in microseconds.\n")
    hdr = (f"{'regime':<13}{'kvMB':>8}{'x re':>5}{'scMB':>8}"
           f"{'hbm_id':>9}{'hbm_st':>9}{'M':>6}{'drt':>5}"
           f"{'mxu_bf16':>10}{'mxu_fp8':>9}{'floor_bf16':>11}{'floor_fp8':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for r in S.REGIMES:
        ideal, struct, bd = bytes_for(r)
        flops, m, derate, tb, tf = mxu_for(r)
        hi, hs = ideal / HBM_BW * 1e6, struct / HBM_BW * 1e6
        tb, tf = tb * 1e6, tf * 1e6
        fb, ff = max(hi, tb), max(hi, tf)
        rows.append((r, hi, hs, tb, tf, fb, ff))
        print(f"{r.name:<13}{bd['kv_MB']:>8.1f}{bd['kv_reread_x']:>5}"
              f"{bd['scores_MB']:>8.1f}{hi:>9.1f}{hs:>9.1f}{m:>6}"
              f"{derate:>5.0f}{tb:>10.1f}{tf:>9.1f}{fb:>11.1f}{ff:>10.1f}")

    print("\ncolumn definitions")
    print("  kvMB     unique cache_kv bytes for one call (MB, 1e6)")
    print("  x re     times the loop structure re-reads that KV (= nbq)")
    print("  scMB     one f32 scores buffer (MB). Crossed twice ideally, "
          "three times as structured.")
    print("  hbm_id   ideal_bytes / 3364.9e9   -- every byte crosses once")
    print("  hbm_st   structured_bytes / 3364.9e9 -- what this loop moves")
    print("  M        rows of the einsum's left operand = bq_sz * 32 heads")
    print("  drt      MXU derate, 128 / min(128, M)")
    print("  mxu_*    flops * drt / peak, peak = 892.8 / 1785.6 TFLOP/s")
    print("  floor_*  max(hbm_id, mxu_*).  NOT the largest column -- HBM is "
          "assumed to overlap compute.")

    if meas:
        print("\nagainst measurement (measured is REAL, floors are MODELLED)")
        h2 = (f"{'regime':<13}{'measured':>10}{'floor_bf16':>11}{'ach_bf16':>10}"
              f"{'floor_fp8':>10}{'ach_fp8':>9}{'bound by':>22}")
        print(h2)
        print("-" * len(h2))
        for r, hi, hs, tb, tf, fb, ff in rows:
            if r.name not in meas:
                continue
            mv = meas[r.name]
            bound = ("HBM" if hi >= tb else "MXU(bf16)")
            bound += " / " + ("HBM" if hi >= tf else "MXU(fp8)")
            print(f"{r.name:<13}{mv:>10.1f}{fb:>11.1f}{fb / mv * 100:>9.0f}%"
                  f"{ff:>10.1f}{ff / mv * 100:>8.0f}%{bound:>22}")
        print("  ach_*    floor / measured, as a percentage. Higher is closer "
              "to the hardware. >=80% means leave it alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
