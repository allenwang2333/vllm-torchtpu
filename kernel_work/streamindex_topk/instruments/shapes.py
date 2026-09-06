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
"""Derived shapes for the GLM-5.2 DSA lightning indexer (`streamindex_topk`).

Every constant here has exactly one source in code or in the checkpoint config,
quoted as a `file:line` comment beside it. Nothing in this file is a round
number chosen for convenience; see `kernel_work/streamindex-topk/report.md`
section "Shapes the block is really called at" for the derivation chain.

Vocabulary, because three of these words mean something else elsewhere in the
stack:

* **compressed KV position** -- one row of the *indexer* KV cache. GLM-5.2 runs
  `compression_ratio=1`, so one compressed position is one token; the code still
  divides by it everywhere and a future model may not.
* **page** -- a `page_size`-row unit of the paged indexer KV cache
  (`page_size = 1024` here). NOT a Pallas grid block and NOT vLLM's KV block for
  the main MLA cache, which is a different size.
* **bkv block** -- `bkv_p` pages' worth of compressed KV positions processed in
  one iteration of the kernel's inner loop. This is the Pallas-level tile.
* **chunk** -- vLLM chunked prefill's `max_num_batched_tokens`. NOT the
  chunkwise-parallel "chunk" of linear-attention kernels.
"""

import dataclasses

# --------------------------------------------------------------------------
# Checkpoint constants -- GLM-5.2-FP8 `config.json`
# (gs://tpu-commons-ci/glm/GLM-5.2-FP8/config.json)
# --------------------------------------------------------------------------
INDEX_N_HEADS = 32  # config.json: index_n_heads
INDEX_HEAD_DIM = 128  # config.json: index_head_dim
INDEX_TOPK = 2048  # config.json: index_topk           -> k
NUM_HIDDEN_LAYERS = 78  # config.json: num_hidden_layers
COMPRESSION_RATIO = 1  # mla_attention_op.py:114 passes compression_ratio=1

# --------------------------------------------------------------------------
# Deployment flags -- scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh
# --------------------------------------------------------------------------
PROD_MAX_MODEL_LEN = 9216  # glm-5.2-fp8-dp16-ep.sh: MAX_MODEL_LEN
PROD_MAX_NUM_BATCHED_TOKENS = 1024  # ..sh: MAX_NUM_BATCHED_TOKENS
PROD_MAX_NUM_SEQS = 64  # ..sh: MAX_NUM_SEQS  (per DP engine)
PROD_ISL = 8192  # ..sh: ISL_OSL_CONFIGS="8192:1024"
PROD_OSL = 1024

# --------------------------------------------------------------------------
# Runner-derived -- computed by the platform, not chosen here
# --------------------------------------------------------------------------
# layers/vllm/attention.py:1018 hardcodes the indexer page size.
PAGE_SIZE = 1024
# streamindex_topk.py:592-593  page_size = page_size_per_kv_packing * kv_packing
KV_PACKING = 4
PAGE_SIZE_PER_KV_PACKING = PAGE_SIZE // KV_PACKING  # 256
# mla_attention_op.py:230-231  head_dim = 128 + 128 // 128 = 129, padded to a
# 128-multiple for the lane axis -> 256. Bytes 0..127 are the fp8 values, byte
# 128 is the e8m0 scale, bytes 129..255 are padding that is still transferred.
LKV_DIM = 256
CACHE_RECORD_USED_BYTES = INDEX_HEAD_DIM + 1  # 129

# --------------------------------------------------------------------------
# Block-tuning knobs -- mla_attention_op.py:119-120
# Tuple order is (DECODE, PREFILL, MIXED); see MlaCase in streamindex_topk.py:49.
# --------------------------------------------------------------------------
NUM_KV_PAGES_PER_BLOCK = (2, 2, 2)  # mla_attention_op.py:119
NUM_QUERIES_PER_BLOCK = (1, 128, 128)  # mla_attention_op.py:120
DECODE_REQ_BATCH_SIZE = 4  # streamindex_topk.py:536 default; call site does not override
# mla_attention_op.py:57 sets this True; the kernel's own default is False
# (streamindex_topk.py:537).
CALLSITE_ENABLE_EARLY_EXIT = True

# The PREFILL band is empty by construction: mla_attention_op.py:183 builds
# `distribution = torch.stack([rd[0], rd[0], rd[2]])`, so entry [1] equals entry
# [0] and the prefill-only kernel invocation
# (streamindex_topk.py:775-791, start=decode_batch_end, end=distribution[1])
# spans zero sequences. Only the DECODE and MIXED cases ever execute for
# GLM-5.2, which is why only `StreamIdxTC-d-*` and `StreamIdxTC-m-*` scopes
# appear in a profile.
PREFILL_BAND_IS_EMPTY = True


def cdiv(a, b):
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def pages_per_seq(max_model_len: int) -> int:
    """Mirror of the runner's block-table width.

    `page_indices` is `i32[max_num_seqs * pages_per_seq]`
    (streamindex_topk.py:527) and the kernel recovers pages_per_seq by integer
    division at streamindex_topk.py:104.
    """
    return cdiv(max_model_len, PAGE_SIZE)


def num_sublanes_total(max_model_len: int,
                       bkv_ps=NUM_KV_PAGES_PER_BLOCK) -> int:
    """Width of the f32 scores buffer, in sublanes of 128 columns.

    Mirrors streamindex_topk.py:603-605 exactly. The scores array is
    `f32[max_num_tokens, num_sublanes_total, 128]`, so its column count is
    `num_sublanes_total * 128` -- which is `align_to(pages_per_seq, bkv_p)`
    pages' worth of compressed positions, i.e. rounded up past the real
    `max_model_len`.
    """
    pps = pages_per_seq(max_model_len)
    return max(align_to(pps, bkv_p) * PAGE_SIZE // 128 for bkv_p in bkv_ps)


def scores_bytes(max_num_tokens: int, max_model_len: int) -> int:
    """Bytes in one f32 scores buffer. This is the block's largest tensor."""
    return max_num_tokens * num_sublanes_total(max_model_len) * 128 * 4


def check_legal(max_model_len: int, bkv_ps=NUM_KV_PAGES_PER_BLOCK) -> None:
    """Refuse a shape that cannot actually run.

    streamindex_topk.py:596-601 rejects any `bkv_p` whose `bkv_sz` is not a
    multiple of 128. `page_size` is 1024 here so every positive `bkv_p` passes,
    but the guard is kept because `page_size` is a runner-derived value
    (attention.py:1018) and a future change to it makes this reachable.
    """
    for bkv_p in bkv_ps:
        bkv_sz = PAGE_SIZE * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"illegal shape: bkv_sz ({PAGE_SIZE} * {bkv_p} = {bkv_sz}) "
                "must be a multiple of 128 -- streamindex_topk.py:598")
    if max_model_len % PAGE_SIZE != 0:
        # Not illegal, but it means the last page is partly padding and the
        # scores buffer is wider than the model length. Worth saying out loud.
        pass


@dataclasses.dataclass(frozen=True)
class Regime:
    """One reachable operating point of the block.

    Fields:
      name           short label used in results tables
      phase          'decode' or 'mixed' -- the two MlaCase values GLM-5.2 reaches
      max_num_seqs   sequences in the batch (padded lane count, not real occupancy)
      num_tokens     rows of `q`, i.e. max_num_batched_tokens for the engine
      kv_len         compressed KV positions resident for each sequence
      max_model_len  deployment flag; sets pages_per_seq and the scores width
      note           why this point is in the sweep
    """

    name: str
    phase: str
    max_num_seqs: int
    num_tokens: int
    kv_len: int
    max_model_len: int
    note: str

    @property
    def num_bq(self) -> int:
        """Inner-loop trip count over query blocks, per sequence."""
        idx = 0 if self.phase == "decode" else 2
        bq_sz = NUM_QUERIES_PER_BLOCK[idx]
        if self.phase == "decode":
            bq_sz = min(bq_sz, 1)  # static_q_len=1 -> streamindex_topk.py:633
            return 1
        return cdiv(self.num_tokens, bq_sz)

    @property
    def num_bkv(self) -> int:
        """Inner-loop trip count over KV blocks, per query block."""
        idx = 0 if self.phase == "decode" else 2
        bkv_sz = PAGE_SIZE * NUM_KV_PAGES_PER_BLOCK[idx]
        return max(1, cdiv(self.kv_len, bkv_sz))


# --------------------------------------------------------------------------
# The sweep. Two production points read off the checked-in config, and two
# long-context points for the regime the end-to-end A/B is being run at.
#
# Production decode and production mixed are the two the profile
# (optimization_work/glm-5-2-2026-09-05/profile_notes.md) actually contains, so
# they are the rows the microbenchmark is cross-checked against.
# --------------------------------------------------------------------------
REGIMES = [
    Regime(
        name="decode_prod",
        phase="decode",
        max_num_seqs=PROD_MAX_NUM_SEQS,
        num_tokens=PROD_MAX_NUM_SEQS,
        kv_len=PROD_ISL,
        max_model_len=PROD_MAX_MODEL_LEN,
        note="checked-in config, steady-state decode; matches the profile's "
        "StreamIdxTC-d-bq_1-bkvp_2 scope at f32[64,80,128]",
    ),
    Regime(
        name="mixed_prod",
        phase="mixed",
        max_num_seqs=PROD_MAX_NUM_SEQS,
        num_tokens=PROD_MAX_NUM_BATCHED_TOKENS,
        kv_len=PROD_ISL,
        max_model_len=PROD_MAX_MODEL_LEN,
        note="checked-in config, last prefill chunk of an 8192-token prompt; "
        "matches StreamIdxTC-m-bq_128-bkvp_2 at f32[1024,80,128]",
    ),
    Regime(
        name="mixed_32k",
        phase="mixed",
        max_num_seqs=12,
        num_tokens=PROD_MAX_NUM_BATCHED_TOKENS,
        kv_len=32768,
        max_model_len=33792,
        note="long-context arm, 4x the indexer work per chunk; regime boundary "
        "check against mixed_prod",
    ),
    Regime(
        name="mixed_64k",
        phase="mixed",
        max_num_seqs=6,
        num_tokens=PROD_MAX_NUM_BATCHED_TOKENS,
        kv_len=65536,
        max_model_len=66560,
        note="long-context arm, the shape the end-to-end A/B targets",
    ),
    Regime(
        name="decode_64k",
        phase="decode",
        max_num_seqs=6,
        num_tokens=6,
        kv_len=65536,
        max_model_len=66560,
        note="long-context decode; included because the indexer's decode cost "
        "is NOT invariant to context length even though the resident token "
        "count per rank is capped by HBM",
    ),
]

REGIMES_BY_NAME = {r.name: r for r in REGIMES}
