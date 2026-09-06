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
"""Regimes added after step 1. AN INSTRUMENT ADDITION, NOT PART OF ANY VARIANT.

Committed separately from the Option B diff on purpose: nothing here is a
change to the kernel, and every one of these shapes is reusable by a later
option. `instruments/shapes.py` is READ-ONLY after step 1 and is not touched --
`REGIMES` and `REGIMES_BY_NAME` still hold exactly the five step-1 rows, so a
variant cannot be measured against a moved yardstick. `microbench.run_regime`
and `test_correctness.run` both take a `Regime` INSTANCE, so a sixth shape is
handed to the read-only instruments without editing them.

Design section 12 item 4 sanctions exactly this: "the five regimes must not be
touched; a sixth may be added".

  mixed_confound     the kv_len << max_model_len shape. Option B and Option C
                     both have their entire prize on this ratio and there was
                     no regime in which it was anything but ~1.
  mask_mixed_8000    kv_len not a multiple of bkv_sz. Correctness only.
  mask_decode_8000   the same, through the batched decode band.
"""

from kernel_work.streamindex_topk.instruments import shapes as S

# ---------------------------------------------------------------------------
# THE SIXTH REGIME -- an instrument ADDITION, not part of Option B.
#
# `measurement_and_design.md` section 12 item 4: the five step-1 regimes all run
# `kv_len` within one page of `max_model_len`, so in every one of them the
# incumbent's `max_model_len`-derived scores width is already within 3-8% of the
# live width. Option B's entire prize is the gap between those two numbers, so
# at all five regimes B is measuring almost nothing -- section 4.3(b)'s headline
# is quoted as `<= -5.40%` precisely because it had no shape to be measured at
# and was normalised against `mixed_64k`'s call as an upper-bound denominator.
#
# `mixed_confound` is that missing shape: a 64k-capable deployment serving an
# 8k-context batch, which section 8.1 says is the *common* case (mean live
# fraction ~50%, the first sixteen chunks of a 64k prompt below 25%).
#
#   incumbent width  align_to(65, 2) * 1024 / 128        = 528 sublanes
#   right-sized      cdiv(min(8192, 66560), 128)         =  64 sublanes
#
# It is defined HERE and not appended to `instruments/shapes.py` because the
# five existing regimes must come out byte-identical: `microbench.run_regime`
# and `test_correctness.run` take a `Regime` INSTANCE, so a sixth can be handed
# to them without the read-only file changing at all. `REGIMES` and
# `REGIMES_BY_NAME` in `shapes.py` are untouched, so nothing that iterates them
# sees a new row either.
#
# It is reusable and is committed separately from the variant diff. Any later
# option whose prize is on the `kv_len / max_model_len` ratio -- Option C's is
# on the same bytes -- should drive it from here.
# ---------------------------------------------------------------------------
CONFOUND = S.Regime(
    name="mixed_confound",
    phase="mixed",
    max_num_seqs=6,
    num_tokens=S.PROD_MAX_NUM_BATCHED_TOKENS,
    kv_len=8192,
    max_model_len=66560,
    note="THE CONFOUND SHAPE (design section 12 item 4). Same deployment flag "
    "as mixed_64k (max_model_len=66560) but an 8k live context, so the "
    "incumbent's scores buffer is 8.25x wider than the live width. This is "
    "the only regime in which a kv_len-vs-max_model_len optimisation has "
    "anything to measure; the five step-1 regimes are all within one page of "
    "their max_model_len by construction.",
)

# ---------------------------------------------------------------------------
# Two CORRECTNESS-ONLY shapes. Not perf regimes, and not part of the sweep.
#
# At every one of the five step-1 regimes and at CONFOUND, `kv_len` is 8192 or
# 65536 -- both exact multiples of `bkv_sz = 2048` -- so the right-sized width
# is a whole number of `bkv` slabs, `n_tail_sublanes` is 0, and the masked
# partial-block store is NEVER EMITTED. Measuring B at those six shapes
# therefore says nothing about whether the masked store is correct or whether
# it lowers.
#
#   kv_len = 8000 -> width cdiv(8000, 128) = 63 sublanes
#                 -> n_full_slabs = 3, n_tail_sublanes = 15
#                 -> num_bkv = cdiv(8000, 2048) = 4, clamped to max_bkv = 4
# so the fourth slab computes 16 sublanes and stores 15. Both bands are
# covered: `mixed` exercises the MIXED invocation, `decode` the batched DECODE
# one, and they run different `seq_batch_size` through the same store.
# ---------------------------------------------------------------------------
MASK_MIXED = S.Regime(
    name="mask_mixed_8000",
    phase="mixed",
    max_num_seqs=S.PROD_MAX_NUM_SEQS,
    num_tokens=S.PROD_MAX_NUM_BATCHED_TOKENS,
    kv_len=8000,
    max_model_len=S.PROD_MAX_MODEL_LEN,
    note="CORRECTNESS ONLY. kv_len not a multiple of bkv_sz, so the masked "
    "partial-block store fires on the MIXED band. 63 sublanes = 3 full slabs "
    "+ a 15-sublane tail.",
)

MASK_DECODE = S.Regime(
    name="mask_decode_8000",
    phase="decode",
    max_num_seqs=S.PROD_MAX_NUM_SEQS,
    num_tokens=S.PROD_MAX_NUM_SEQS,
    kv_len=8000,
    max_model_len=S.PROD_MAX_MODEL_LEN,
    note="CORRECTNESS ONLY. Same tail as mask_mixed_8000 but through the "
    "batched DECODE band (seq_batch_size=4).",
)

EXTRA_REGIMES = {r.name: r for r in (CONFOUND, MASK_MIXED, MASK_DECODE)}
