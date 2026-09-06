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
"""Shared plumbing for the step-4 composed-candidate drivers.

Everything here is loading and binding. No measurement, no shapes of its own
beyond the four correctness-only shapes at the bottom, which are additions in
the sense `measurement_and_design.md` section 12 item 4 sanctions:
`instruments/shapes.py` is READ-ONLY and is not touched, and the sixth perf
regime is IMPORTED from `variants/right-size/extra_regimes.py` rather than
copied.

THE FOUR ARMS
-------------
  incumbent   `vllm_torchtpu.kernels.deepseek_v4.streamindex_topk`, untouched,
              packed cache, tuple (2,2,2).
  B           `variants/right-size/streamindex_topk_rs.py`, the Option B
              variant exactly as step 3 measured it, packed cache, (2,2,2),
              `kv_len_bucket` bound. Present so any composition loss can be
              ATTRIBUTED rather than merely observed.
  BA          `final/streamindex_topk_composed.py`, split cache, (2,2,2),
              `kv_len_bucket` bound.
  BAE         the same file, the same bucket, tuple (4,2,4).

Plus a DRIFT CONTROL: a second `incumbent` arm, identical code, measured as if
it were a comparison. Its delta is the noise floor for an arm-to-arm
comparison and is a larger quantity than the within-arm spread.

HOW THE ARMS REACH THE READ-ONLY INSTRUMENTS
--------------------------------------------
Three mechanisms, none of which edits an instrument:

1.  `bind()` wraps a module in a `SimpleNamespace` whose `streamindex_topk`
    has the extra static kwargs applied. `microbench.run_regime` and
    `test_correctness.check` touch exactly one attribute.
2.  `patched_inputs()` (Option A's, imported from `variants/narrow-record`)
    rebinds `instruments.inputs.build` to the splitting builder for the
    duration of a with-block. The split is a pure reindexing of the array the
    read-only builder returned, so both arms see bit-identical key values,
    bit-identical scales, the same page permutation and the same q/weights.
3.  `tuple_bound()` rebinds `instruments.shapes.NUM_KV_PAGES_PER_BLOCK`, which
    is the module attribute `run_regime` reads at trace time. This is Option
    E. Note `shapes.num_sublanes_total`'s default argument was bound at def
    time and therefore does NOT move, so the instrument's reported
    `scores_MB` and its `inner` budget stay identical across arms -- which is
    what makes the arms comparable.

THE BUCKET IS THE LADDER'S, NOT THE EXACT LENGTH
-------------------------------------------------
Step 3 measured Option B with the driver passing the EXACT `kv_len`, which its
own RESULTS.md already calls a ceiling. A trace of the call chain (recorded in
`final_report.md`) found that ceiling is not merely optimistic, it is
unreachable: `kv_len_bucket` has to be a static Python int, vLLM's backbone is
Dynamo-traced ONCE with all guards dropped
(`vllm/compilation/wrapper.py:104-122`, and this repo's own
`compilation/shape_variants.py:1-31` says so in as many words), so a Python int
reaching the custom op is frozen at its first-traced value and cannot vary per
serving step. The warmed ladder that does exist keys on TOKEN COUNT, not on kv
length.

So the two modes below are a shipping number and a ceiling, and they are
labelled that way everywhere:

  static  `bucket = min(max_model_len, MAX_SLICE_WORDS)`, a constant derived
          from config at op-build time. THIS IS WHAT SHIPS. It still collects
          the whole `_pick_partition` threshold prize at the A/B shape -- the
          incumbent's 67584 columns are stuck at P=2 and 65536 is P=1 -- which
          step 3 identified as where most of B's win comes from.
  exact   `bucket = align(kv_len, 8192)`, capped the same way. THIS IS A
          CEILING, not a candidate: it needs a per-step static int the
          compilation model does not currently admit. Run at `mixed_confound`
          only, to price what that plumbing would be worth if it existed.

The difference is not cosmetic, and it is not uniform either. At
`mixed_confound` (kv_len 8192 inside max_model_len 66560) `exact` allocates 64
sublanes and `static` allocates 512 -- so `exact` is the one that could buy
step 3's 3.725x, while `static` gets only the part of the win that comes from
crossing the `_pick_partition` threshold (512 sublanes is P=1; the incumbent's
528 is P=2). Both arms are run at that regime for exactly this reason: the gap
between them is the price of the plumbing that does not exist.
"""

import contextlib
import functools
import importlib
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_VARIANTS = os.path.abspath(os.path.join(_HERE, os.pardir, "variants"))

from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402

INCUMBENT_MODULE = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_incumbent():
    return importlib.import_module(INCUMBENT_MODULE)


def load_b_only():
    """Option B exactly as step 3 measured it. The attribution arm."""
    return _load_path(
        "streamindex_topk_rs",
        os.path.join(_VARIANTS, "right-size", "streamindex_topk_rs.py"))


def load_composed():
    """B + A, built by `compose.py`."""
    return _load_path("streamindex_topk_composed",
                      os.path.join(_HERE, "streamindex_topk_composed.py"))


def load_narrow_inputs():
    """Option A's splitting input builder. IMPORTED, not copied."""
    return _load_path(
        "narrow_inputs",
        os.path.join(_VARIANTS, "narrow-record", "narrow_inputs.py"))


def load_bucket_probe():
    """The B-alone `bucket > kv_len` gate (job j-6454262e-acd2-4371-b3ba).

    IMPORTED, not copied. It owns the two pieces that make a poison arm
    falsifiable rather than decorative, and rebuilding them here would mean
    the composed candidate is judged by a second, unreviewed implementation of
    the same shim:

      `_lanecount_shim`      counts sublanes still entirely equal to the
                             marker -- the EMPIRICAL unwritten-lane count,
                             read off the device buffer rather than predicted,
                             and cross-checked against `predict`.
      `_full_rowlen_shim`    `neg_control_unwritten`: the same poisoned buffer
                             with only the reachability gate removed. It is
                             REQUIRED TO FAIL wherever the measured unwritten
                             count is > 0. Without it the poison arm is once
                             again a check that cannot fail.
      `predict`              the analytic geometry, printed beside the
                             measurement and never in place of it.
    """
    return _load_path(
        "run_bucket_correctness",
        os.path.join(_HERE, os.pardir, "probes", "bucket_correctness",
                     "run_bucket_correctness.py"))


def bind(impl, **extra):
    """A module-like object whose `streamindex_topk` has `extra` bound."""
    return types.SimpleNamespace(
        streamindex_topk=functools.partial(impl.streamindex_topk, **extra),
        __name__=getattr(impl, "__name__", "impl"))


@contextlib.contextmanager
def tuple_bound(tup):
    """Option E: rebind the tuple the read-only instruments read at trace time."""
    orig = S.NUM_KV_PAGES_PER_BLOCK
    S.NUM_KV_PAGES_PER_BLOCK = tup
    try:
        yield
    finally:
        S.NUM_KV_PAGES_PER_BLOCK = orig


@contextlib.contextmanager
def nullctx():
    yield


# ---------------------------------------------------------------------------
# Regimes. The five step-1 rows come from the READ-ONLY `shapes.py`; the sixth
# and the two masked-store shapes are IMPORTED from the Option B agent's
# instrument addition; the four below are new.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(_VARIANTS, "right-size"))
from extra_regimes import EXTRA_REGIMES  # noqa: E402

# ---------------------------------------------------------------------------
# FOUR CORRECTNESS-ONLY SHAPES WHERE `kv_len_bucket > kv_len`.
#
# This is the configuration Option B actually SHIPS in under any bucket ladder,
# and it is the only configuration in which the risk B carries -- a reachable
# unwritten lane, created by deleting the `-1` fill -- exists at all. Every
# correctness arm run in step 3 bound `kv_len_bucket = r.kv_len`
# (`variants/right-size/run_correctness.py:121-122`), so the right-sized buffer
# had ZERO unwritten lanes and the `+1e30` poison was 100% overwritten before
# selection could read it. The check could not have failed.
#
#   written sublanes = align(kv_len, bkv_sz) / 128   (what the kernel stores)
#   buffer sublanes  = kv_len_bucket / 128           (what B allocates)
#   unwritten        = the difference, and it is > 0 only when bucket > kv_len
#
# The two shapes below are chosen so that the SHIPPING bucket -- `static`,
# `min(max_model_len, 65536)`, no per-step plumbing assumed -- lands on exactly
# the two configurations the review named: 24576 under a 32768 bucket (64
# unwritten sublanes) and 8000 under a 16384 bucket (64 unwritten sublanes).
# ---------------------------------------------------------------------------
UNWRITTEN_MIXED = S.Regime(
    name="unwritten_mixed_24576",
    phase="mixed",
    max_num_seqs=12,
    num_tokens=S.PROD_MAX_NUM_BATCHED_TOKENS,
    kv_len=24576,
    max_model_len=32768,
    note="CORRECTNESS ONLY, AND THE ONE THE RECOMMENDATION RESTS ON. Static "
    "bucket 32768 against kv_len 24576: buffer 256 sublanes, kernel writes "
    "align(24576,2048)/128 = 192, so 64 sublanes (8192 columns) per row are "
    "NEVER WRITTEN. MIXED band.",
)

UNWRITTEN_DECODE = S.Regime(
    name="unwritten_decode_8000",
    phase="decode",
    max_num_seqs=8,
    num_tokens=8,
    kv_len=8000,
    max_model_len=16384,
    note="CORRECTNESS ONLY. Static bucket 16384 against kv_len 8000: buffer "
    "128 sublanes, kernel writes align(8000,2048)/128 = 64, so 64 sublanes "
    "are NEVER WRITTEN. Batched DECODE band (8 seqs, seq_batch_size=4).",
)

# The masked partial-block store needs the bucket to NOT be a whole number of
# `bkv_sz = 2048` slabs, AND `num_bkv` to reach `max_bkv` so the tail slab is
# actually visited. `max_model_len = 9216` -- the CHECKED-IN PRODUCTION VALUE
# -- gives a static bucket of 9216 = 4 slabs + an 8-sublane tail, and kv_len
# 8300 makes num_bkv = cdiv(8300,2048) = 5 = max_bkv, so the tail slab runs.
#
# Note this is structural, and worth knowing when reading the verdicts: the
# masked tail store and unwritten lanes are MUTUALLY EXCLUSIVE. The tail slab
# is only reached when num_bkv == max_bkv, and that is exactly the case in
# which the kernel writes the buffer's whole width. One shape cannot cover
# both risks, which is why there are three here and not one.
TAIL_LADDER = S.Regime(
    name="tail_static_mixed_8300",
    phase="mixed",
    max_num_seqs=12,
    num_tokens=S.PROD_MAX_NUM_BATCHED_TOKENS,
    kv_len=8300,
    max_model_len=S.PROD_MAX_MODEL_LEN,
    note="CORRECTNESS ONLY. Static bucket 9216 = 4 slabs + an 8-sublane tail, "
    "so the masked partial-block store fires at a bucket the SHIPPING "
    "configuration actually produces at the production max_model_len.",
)

NEW_REGIMES = {r.name: r for r in (UNWRITTEN_MIXED, UNWRITTEN_DECODE,
                                   TAIL_LADDER)}


def regime(name: str) -> S.Regime:
    if name in NEW_REGIMES:
        return NEW_REGIMES[name]
    if name in EXTRA_REGIMES:
        return EXTRA_REGIMES[name]
    return S.REGIMES_BY_NAME[name]


def incumbent_cols(r: S.Regime) -> int:
    """The scores width the SHIPPED kernel builds, in columns."""
    return S.num_sublanes_total(r.max_model_len) * 128


def bucket_for(r: S.Regime, composed, mode: str) -> int:
    """The `kv_len_bucket` an arm passes. See the module docstring.

    `static` is the only mode a call site can actually deliver today; `exact`
    is a ceiling that would need a per-step static int, and is measured to
    price the plumbing rather than to be shipped. Both go through the composed
    kernel's own `kv_len_bucket_for`, so the driver cannot drift from what the
    candidate would compute for itself.
    """
    if mode == "static":
        return composed.kv_len_bucket_for(r.max_model_len, r.max_model_len,
                                          S.INDEX_TOPK)
    if mode == "exact":
        return composed.kv_len_bucket_for(r.kv_len, r.max_model_len,
                                          S.INDEX_TOPK)
    if mode == "kvlen":
        # Step 3's configuration, reproduced only so the hollow-check finding
        # can be shown side by side with the shipping one. NOT a candidate.
        return max(r.kv_len, S.INDEX_TOPK)
    raise ValueError(f"unknown bucket mode {mode!r}")


def lane_accounting(r: S.Regime, bucket: int) -> dict:
    """How many sublanes the buffer has, the kernel writes, and nobody writes.

    `unwritten_sublanes > 0` is the whole point of the `bucket > kv_len` arms:
    it is the count of lanes the deleted `-1` fill used to cover and nothing
    now does. Reported so the poison arm's verdict can be read against a
    number rather than against a claim.
    """
    inc_cols = incumbent_cols(r)
    pps_cols = S.pages_per_seq(r.max_model_len) * S.PAGE_SIZE
    live = min(bucket, pps_cols)
    buf_sub = max(S.cdiv(live, 128), S.cdiv(S.INDEX_TOPK, 128))
    bkv_sz = S.PAGE_SIZE * S.NUM_KV_PAGES_PER_BLOCK[
        0 if r.phase == "decode" else 2]
    num_sub_bkv = bkv_sz // 128
    max_bkv = S.cdiv(buf_sub, num_sub_bkv)
    n_full = buf_sub // num_sub_bkv
    n_tail = buf_sub - n_full * num_sub_bkv
    num_bkv = min(max(1, S.cdiv(r.kv_len, bkv_sz)), max_bkv)
    written_sub = min(num_bkv * num_sub_bkv, buf_sub)
    return dict(
        incumbent_cols=inc_cols,
        bucket=bucket,
        buffer_sublanes=buf_sub,
        buffer_cols=buf_sub * 128,
        written_sublanes=written_sub,
        unwritten_sublanes=buf_sub - written_sub,
        unwritten_cols=(buf_sub - written_sub) * 128,
        n_tail_sublanes=n_tail,
        tail_store_emitted=bool(n_tail) and num_bkv > n_full,
        num_bkv=num_bkv,
        max_bkv=max_bkv,
    )


def banner():
    import hashlib

    import jax
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    # The incumbent's path comes from the imported module, not from a
    # constructed one: in the job the repo lives in the image at
    # /root/cloud-devkit/vllm-torchtpu while only kernel_work is mapped to
    # /workspace, so a path built from `_REPO` would point at nothing.
    for label, path in (
        ("incumbent", load_incumbent().__file__),
        ("B_only", os.path.join(_VARIANTS, "right-size",
                                "streamindex_topk_rs.py")),
        ("composed", os.path.join(_HERE, "streamindex_topk_composed.py")),
    ):
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        print(f"{label:<10} {path}  md5={h}", flush=True)
    inst = os.path.join(_HERE, os.pardir, "instruments", "shapes.py")
    h = hashlib.md5(open(inst, "rb").read()).hexdigest()
    print(f"{'shapes.py':<10} md5={h}  (must be byte-identical to step 1)",
          flush=True)
