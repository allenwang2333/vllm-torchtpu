#!/usr/bin/env python3
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
"""Shared config grid and arm definitions for the merged-op A/B.

Imported by `bench_merged_trace.py` (device self time) and
`correctness_merged.py` (index equality).  Nothing here executes on import
beyond building the config list, so it is safe to import under
`JAX_PLATFORMS=cpu` for a dry run.

The five arms
-------------

| arm       | module    | decode `bkv_p`   | early-exit cond | Option C     |
| --------- | --------- | ---------------- | --------------- | ------------ |
| `inc`     | incumbent | 2 (production)   | live            | off          |
| `bkvp`    | incumbent | `choose_bkv_p()` | live            | off          |
| `d1`      | merged    | 2                | deleted         | off          |
| `bkvp_d1` | merged    | `choose_bkv_p()` | deleted         | off          |
| `merged`  | merged    | `choose_bkv_p()` | deleted         | prefill only |

`inc` and `merged` are the headline pair; the middle three exist so a
composition surprise is attributable to a pair rather than merely visible.
The cond-live arms have to come from the incumbent module, because the merged
module deleted the cond outright and cannot synthesize it back.

`num_positions` is the same in every arm, at every config in the grid
----------------------------------------------------------------------

`num_sublanes_total = max over the 3-tuple of align_to(pages_per_seq, bkv_p)
* page_size // 128` (streamindex_topk.py:603-605), so the *whole tuple* sets
the score matrix width, and a `bkv_p` change can silently resize stage B and
the `-inf` init along with stage A.  It does not here: every config in
`CONFIGS` has `pages_per_seq` a multiple of 2 or equal to 9, and
`align_to(pps, P) == align_to(pps, 2)` at all of them when `P = cdiv(mml,
page_size)`.  Checked, not assumed -- `check_like_for_like()` recomputes it
and the probes refuse to run an arm that fails it.

The VMEM ladder
---------------

`recommended_num_kv_pages_per_block` returns `cdiv(max_model_len, page_size)`
flat, and at long context that tuple **does not compile**: the bkv double
buffer alone is `2 * decode_req_batch_size * bkv_p * page_size * lkv_dim`,
which at `bkv_p=32` is 64 MiB against a 63.94 MiB physical TensorCore VMEM
(`results.md` 6.4).  `choose_bkv_p` walks the block-count ladder
`p = cdiv(pages_per_seq, nb)` for `nb = 1, 2, 3, ...` and takes the first
entry that fits, which is the right fallback under `results.md` 6.2's
two-regime model: fewer bkv blocks is better, and among tuples with the same
block count the smallest `p` overfetches least.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_INSTR = os.path.join(_ROOT, "instruments", "bench_streamidx_topk.py")
_MERGED = os.path.join(_HERE, "streamindex_topk_merged.py")

ARMS = ("inc", "bkvp", "d1", "bkvp_d1", "merged")
# Arms whose decode slot is the recommended `bkv_p` rather than the constant 2.
BKVP_ARMS = frozenset({"bkvp", "bkvp_d1", "merged"})
# Arms that come from the merged module (cond deleted).
MERGED_MODULE_ARMS = frozenset({"d1", "bkvp_d1", "merged"})


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


@dataclasses.dataclass(frozen=True)
class Cfg:
    name: str
    phase: str  # "decode" | "prefill"
    num_seqs: int
    q_len: int
    ctx: int
    max_model_len: int = 0  # 0 => the block table is sized to ctx

    @property
    def block_table_len(self) -> int:
        return self.max_model_len or self.ctx


DECODE_BATCHES = (8, 32, 64)
DECODE_CONTEXTS = (4096, 16384, 32768, 65536)
PREFILL_Q_LENS = (1024, 4096)
PREFILL_CONTEXTS = (4096, 16384)

# The GLM-5.2 nightly deployment: max_model_len 9216 (8192 in + 1024 out),
# max_num_seqs 64, page 1024.  ctx 8704 is mid-decode, the row `report.md`
# models and the row every earlier number in `results.md` was taken at.
PROD = Cfg("dec_b64_ctx8704_mml9216", "decode", 64, 1, 8704, 9216)


def _configs() -> list[Cfg]:
    out: list[Cfg] = [PROD]
    for b in DECODE_BATCHES:
        for ctx in DECODE_CONTEXTS:
            out.append(Cfg(f"dec_b{b}_ctx{ctx}", "decode", b, 1, ctx))
    for q in PREFILL_Q_LENS:
        for ctx in PREFILL_CONTEXTS:
            out.append(Cfg(f"pre_q{q}_ctx{ctx}", "prefill", 1, q, ctx))
    return out


CONFIGS: list[Cfg] = _configs()
CONFIG_BY_NAME = {c.name: c for c in CONFIGS}


# ---------------------------------------------------------------------------
# Module loading.  Both modules define a symbol called `streamindex_topk`, so
# they are loaded under distinct names and never via a bare `import`.
# ---------------------------------------------------------------------------


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_instrument():
    """The read-only step-1 instrument: Shape, geometry, build_inputs."""
    return _load(_INSTR, "bench_streamidx_topk")


def load_merged():
    """The merged module under test."""
    return _load(_MERGED, "streamindex_topk_merged")


def load_incumbent():
    """The shipping kernel, imported from the mapped `src/` tree."""
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as mod
    return mod


def to_shape(B, cfg: Cfg):
    """The instrument's `Shape` for a `Cfg`, so geometry/build_inputs apply."""
    return B.Shape(
        name=cfg.name,
        phase=cfg.phase,
        num_seqs=cfg.num_seqs,
        q_len=cfg.q_len,
        context_len=cfg.ctx,
        enable_early_exit=True,  # production: mla_attention_op.py:57
        max_model_len=cfg.max_model_len,
    )


# ---------------------------------------------------------------------------
# bkv_p selection.
# ---------------------------------------------------------------------------


def bkv_p_ladder(pages_per_seq: int) -> list[int]:
    """Candidate decode `bkv_p`, best first.

    `cdiv(pages_per_seq, nb)` for nb = 1, 2, 3, ... -- i.e. "cover the block
    table in nb bkv blocks".  Deduplicated, descending, terminated at the
    incumbent's 2.

    Rungs that would widen the score matrix are dropped, matching the module's
    `recommended_num_kv_pages_per_block`.  `num_sublanes_total` is a max over
    the whole tuple, so a decode `p` with `align_to(pages_per_seq, p) >
    align_to(pages_per_seq, 2)` gives every launch a wider matrix: more
    `-inf` init and more positions for stage B.  That is not a like-for-like
    comparison against the incumbent, and it is not a win either -- it is
    strictly more work.  At `pages_per_seq = 32` this drops 11, 7, 6, ... and
    leaves [32, 16, 8, 4, 2].
    """
    baseline = align_to(pages_per_seq, PREFILL_BKV_P)
    out: list[int] = []
    for nb in range(1, pages_per_seq + 1):
        p = cdiv(pages_per_seq, nb)
        if p < PREFILL_BKV_P:
            break
        if align_to(pages_per_seq, p) > baseline:
            continue
        if p not in out:
            out.append(p)
    if PREFILL_BKV_P not in out:
        out.append(PREFILL_BKV_P)
    return out


def num_sublanes_total(pages_per_seq: int, tup) -> int:
    return max(align_to(pages_per_seq, p) * 1024 // 128 for p in tup)


def check_like_for_like(pages_per_seq: int, tup_a, tup_b) -> bool:
    return (num_sublanes_total(pages_per_seq, tup_a) ==
            num_sublanes_total(pages_per_seq, tup_b))


def bkv_vmem_bytes(bkv_p: int, page_size: int, lkv_dim: int,
                   seq_batch_size: int) -> int:
    """The bkv double buffer alone -- the term that hits the VMEM wall."""
    return 2 * seq_batch_size * bkv_p * page_size * lkv_dim


INCUMBENT_TUPLE = (2, 2, 2)
PREFILL_BKV_P = 2


def arm_spec(arm: str, decode_bkv_p: int) -> dict:
    """The kwargs that distinguish one arm, given a resolved decode `bkv_p`."""
    assert arm in ARMS, arm
    tup = ((decode_bkv_p, PREFILL_BKV_P, PREFILL_BKV_P)
           if arm in BKVP_ARMS else INCUMBENT_TUPLE)
    if arm in MERGED_MODULE_ARMS:
        return {
            "module": "merged",
            "num_kv_pages_per_block": tup,
            "head_major_prefill": arm == "merged",
        }
    return {
        "module": "incumbent",
        "num_kv_pages_per_block": tup,
        "enable_early_exit": True,
    }


def make_call(B, mods, arm: str, inputs: dict, decode_bkv_p: int):
    """A zero-argument closure that runs one arm on `inputs`."""
    spec = arm_spec(arm, decode_bkv_p)
    static = dict(
        k=B.K_IDX,
        compression_ratio=B.COMPRESSION_RATIO,
        num_kv_pages_per_block=spec["num_kv_pages_per_block"],
        num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
    )
    if spec["module"] == "merged":
        fn = mods["merged"].streamindex_topk
        static["head_major_prefill"] = spec["head_major_prefill"]
    else:
        fn = mods["incumbent"].streamindex_topk
        static["enable_early_exit"] = spec["enable_early_exit"]

    def call():
        return fn(**inputs, **static)

    return call, spec


def resolve_decode_bkv_p(B, mods, cfg: Cfg, inputs: dict, log=print):
    """The largest ladder entry that actually compiles, plus what it cost.

    Returns `(bkv_p, record)`.  `record["ladder"]` lists every entry tried and
    whether it lowered, so a config that fell back is visible in the results
    rather than silently measured at a different parameter than the one the
    recommendation names.
    """
    pages_per_seq = cdiv(cfg.block_table_len, 1024)
    ideal = mods["merged"].recommended_num_kv_pages_per_block(
        cfg.block_table_len, 1024)[0]
    full_ladder = bkv_p_ladder(pages_per_seq)
    assert ideal in full_ladder, (full_ladder, ideal)
    # Start ONE RUNG ABOVE the module's recommendation, then descend.  The
    # module clamps analytically on a half-VMEM budget for the bkv double
    # buffer, and that budget is a judgement call, not a measurement: at
    # pages_per_seq >= 16 the rung above the clamp is p=16, whose 32.00 MiB
    # buffer misses the 31.97 MiB budget by 0.03 MiB.  Trying it costs one
    # compile and turns "we assumed it would not fit" into evidence either
    # way, which is the difference between a tuned clamp and a superstition.
    start = max(0, full_ladder.index(ideal) - 1)
    ladder = full_ladder[start:]

    tried = []
    chosen = None
    for p in ladder:
        tup = (p, PREFILL_BKV_P, PREFILL_BKV_P)
        if not check_like_for_like(pages_per_seq, tup, INCUMBENT_TUPLE):
            tried.append({"bkv_p": p, "status": "rejected_not_like_for_like",
                          "num_sublanes_total":
                          num_sublanes_total(pages_per_seq, tup)})
            continue
        call, _ = make_call(B, mods, "bkvp", inputs, p)
        try:
            import jax
            out = call()
            jax.block_until_ready(out)
            del out
            tried.append({"bkv_p": p, "status": "ok"})
            chosen = p
            break
        except Exception as err:  # noqa: BLE001 - a tuple that will not lower is a result
            tried.append({"bkv_p": p, "status": "failed",
                          "error": repr(err)[:300]})
            log(f"  bkv_p={p} did not lower: {repr(err)[:200]}")
    record = {
        "pages_per_seq": pages_per_seq,
        "recommended_bkv_p": ideal,
        "chosen_bkv_p": chosen,
        "fell_back": chosen is not None and chosen < ideal,
        "clamp_too_tight": chosen is not None and chosen > ideal,
        "analytic_ladder": full_ladder,
        "tried_from": full_ladder[start],
        "vmem_clamped": ideal != full_ladder[0],
        "bkv_double_buffer_MiB": (
            mods["merged"].bkv_double_buffer_bytes(ideal, 1024) / 2**20),
        "ladder": tried,
    }
    return chosen, record
