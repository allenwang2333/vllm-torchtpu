#!/usr/bin/env bash
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

set -euo pipefail

# Optionally activate a venv.
if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

PORT="${PORT:-8000}"
SHARDING="${SHARDING:-DP8_EP}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
case "$MODEL" in
  gs://*) LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}" ;;
esac
# Headroom over ISL+OSL. A kilotoken, so the 8k1k point serves at a round
# 10240. The reference jobs all serve at 70000 regardless of the workload, but
# max_model_len past the longest request only widens the block table the
# attention kernel walks, and there is no reason to carry their number.
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-1024}"

# Scheduler steps between prefill admissions. Empty means don't pass the flag.
SHARDING_PREFILL_SCHEDULE_INTERVAL=""

case "$SHARDING" in
  DP8_EP)
    # Data-parallel attention across all 8 cores + expert-parallel MoE.
    DP_SIZE=8
    # Admit prefills every 2 scheduler steps, unless the point table below
    # names its own interval.
    SHARDING_PREFILL_SCHEDULE_INTERVAL=2
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
    )
    # Disable the locally ported coordinated scheduler for this reproduction.
    export DP_SCHED_ENABLED=0
    export TPU_DP_PREFILL_FLUSH_TIMEOUT_MS=0
    export TPU_DP_PREFILL_QUORUM_FRACTION=1.0
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
    # Pick the MoE path per compiled token bucket: fused EP for prefill, vLLM's
    # own dispatch onto the fused GMM for decode, which is the cheaper of the
    # two once a step is only a few rows per expert. DP only, the contract
    # patch returns early at dp_size 1.
    #
    # A decode-only step carries one token per running sequence, so its width
    # is the point's concurrency divided over the 8 ranks: 16 at c128 and 32 at
    # c256. Buckets here are 4 8 16 32 48 64 128 256 and the test is
    # shape[0] >= threshold, so 33 is the smallest value that leaves both decode
    # widths on the GMM. 32 would not, it puts c256 back on fused EP by one
    # token. Going higher, to max_num_seqs + 1, would also pull the 48 and 64
    # buckets over, and those carry prefill tails that we have never measured on
    # the GMM, so keep the split as tight as the two swept points need.
    #
    # The width it compares against is DP-wide, not per rank. EP-DP runs in
    # lockstep, an all_reduce MAX over the ranks' scheduled tokens picks one
    # bucket and every rank runs it. A step reaches the decode buckets only when
    # all eight ranks are decode-only at once, which a saturating client rarely
    # allows, so expect this to matter at low load and in the drain tail rather
    # than in the middle of a sweep point.
    #
    # Forking the backbone trace this way used to kill warmup, because the two
    # paths lifted different graph inputs and a bucket could not be handed
    # between them (job j-a0850720). The base stabilizes those inputs now, so
    # the fork is safe to take, but moving where it falls is what that failure
    # was sensitive to.
    if [ "$USE_MOE_FUSED_EP_KERNEL" = "1" ]; then
      export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS="${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-33}"
    fi
    ;;
  TP8_EP)
    # Tensor-parallel attention across all 8 chips + expert-parallel MoE.
    DP_SIZE=1
    SHARDING_ARGS=(
      --tensor-parallel-size=8
      --data-parallel-size=1
      --enable-expert-parallel
    )
    # The fused EP kernel refuses tensor parallelism and cannot help here.
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-0}"
    ;;
  DP4TP2_EP)
    # Data-parallel attention across all 4 chips with tensor-parallel across 2 cores
    # on the same chip + expert-parallel MoE.
    DP_SIZE=4
    SHARDING_PREFILL_SCHEDULE_INTERVAL=256
    SHARDING_ARGS=(
      --tensor-parallel-size=2
      --data-parallel-size=4
      --enable-expert-parallel
    )
    # Keep the coordinated DP scheduler disabled for reproduction runs.
    export DP_SCHED_ENABLED=0
    export TPU_DP_PREFILL_FLUSH_TIMEOUT_MS=0
    export TPU_DP_PREFILL_QUORUM_FRACTION=1.0
    # The fused EP kernel also refuses the TP2 component of this sharding.
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-0}"
    ;;
  *) echo "ERROR: unknown SHARDING='$SHARDING' (DP8_EP, TP8_EP, DP4TP2_EP are wired up)" >&2; exit 1 ;;
esac

# Gathering hidden states before recomputing routing logits would save one
# collective per MoE layer on small DP buckets, but its token threshold forks
# the backbone trace exactly like the fused EP threshold above. Left unset for
# the same reason. Export TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS to try it.

MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))

# Global num of batched tokens == max(ISL, GLOBAL_BATCHED_TOKENS_MIN).
GLOBAL_BATCHED_TOKENS_MIN="${GLOBAL_BATCHED_TOKENS_MIN:-24576}"
GLOBAL_BATCHED_TOKEN=$(( ISL > GLOBAL_BATCHED_TOKENS_MIN ? ISL : GLOBAL_BATCHED_TOKENS_MIN ))

# Fallbacks for a point nobody has tuned: divide the node-wide token target
# across DP ranks, and give the engine twice the client concurrency in
# sequence slots so it can admit a new request while the batch drains.
DEFAULT_MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_BATCHED_TOKEN + DP_SIZE - 1) / DP_SIZE))
DEFAULT_MAX_NUM_SEQS=$(((2 * CONC + DP_SIZE - 1) / DP_SIZE))
DEFAULT_PREFILL_SCHEDULE_INTERVAL="$SHARDING_PREFILL_SCHEDULE_INTERVAL"

# The swept 8k1k points are not a formula. Each one below is the scheduler
# configuration of the job that holds the record at that concurrency, read off
# its serve line, so a run here differs from the record in the engine and not
# in how the engine was asked to batch. Fields are max_num_seqs,
# max_num_batched_tokens and prefill_schedule_interval, the last per DP rank.
#
#   conc  job                          total tok/s
#      4  j-c6283d9b-ca2e-44a6-9a6d         4349.6
#      8  j-edf3640d-b658-440d-815f         6886.2
#     16  j-9cc87140-d980-4274-b154         9473.9
#     32  j-09e874c0-0ea0-4749-8029        12491.6
#     64  j-c68bdb9a-dbf9-4ec4-b8d2        16216.0
#    128  j-71ee7d19-3bcb-42f2-8e3c        21858.7
#    256  j-2386a47f-6bfb-4a1d-879a        32244.6
#
# The interval is inert at DP=1, where the throttle only engages in a
# DPEngineCoreProc, but the TP8 records pass it and carrying it costs nothing.
#
# c4 alone does not carry the record's max_num_seqs of 256. The attention
# kernels here size their metadata off it directly -- kv_lens is
# [max_num_seqs] and page_indices is [max_num_seqs, pages_per_seq], and the
# decode step walks every slot -- so 256 makes each step do 256 sequences of
# work to serve 4. The record's build buckets that dimension, through
# ATTN_BUCKETIZED_NUM_REQS and ATTN_CUSTOM_NUM_REQS_BUCKETS, neither of which
# is read anywhere in this tree.
#
# c8 is the crossover and goes the other way, so both were measured rather
# than derived:
#            max_num_seqs 2*conc   max_num_seqs 256
#   c4              4284.8               4204.3
#   c8              6639.4               6732.9
# Eight live sequences already hide enough of the padding that the scheduler
# having room to admit the next request is worth more than the empty slots
# cost, and by c16 the record's own numbers are small again.
if [ "$ISL:$OSL" = "8192:1024" ]; then
  case "$SHARDING:$CONC" in
    TP8_EP:4)   REFERENCE_POINT="8   16384 256" ;;
    TP8_EP:8)   REFERENCE_POINT="256 16384 256" ;;
    TP8_EP:16)  REFERENCE_POINT="24   8192  64" ;;
    TP8_EP:32)  REFERENCE_POINT="48   8192  64" ;;
    TP8_EP:64)  REFERENCE_POINT="64   8192  16" ;;
    DP8_EP:128) REFERENCE_POINT="64    256   2" ;;
    # 256, not the record's 320, for the same reason c4 and c8 keep their own
    # batch size. Buckets here are powers of two plus TPU_TOKEN_BUCKET_EXTRA,
    # so 320 buys one more bucket and a longer step, while the record's build
    # lays down a linear ladder (TPU_TOKEN_BUCKET_LINEAR_INTERVAL,
    # TPU_TOKEN_BUCKET_LINEAR_UNTIL, neither read here) that 320 sits on
    # neatly. Measured: 33595.5 at 256 against 31093.8 at 320.
    DP8_EP:256) REFERENCE_POINT="64    256   2" ;;
    *)          REFERENCE_POINT="" ;;
  esac
  if [ -n "$REFERENCE_POINT" ]; then
    read -r DEFAULT_MAX_NUM_SEQS DEFAULT_MAX_NUM_BATCHED_TOKENS \
      DEFAULT_PREFILL_SCHEDULE_INTERVAL <<<"$REFERENCE_POINT"
  fi
fi

MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_NUM_BATCHED_TOKENS}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEFAULT_MAX_NUM_SEQS}"
# Colon-dash on purpose: the recipe exports PREFILL_SCHEDULE_INTERVAL="" when
# its PSI knob is unset, and that has to mean "you decide", not "no flag".
PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-$DEFAULT_PREFILL_SCHEDULE_INTERVAL}"
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1
if [ -n "$PREFILL_SCHEDULE_INTERVAL" ]; then
  SHARDING_ARGS+=(--prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL")
fi

# Increase API-server frontend wait time since cold init may take long.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export TPU_GDN_RECURRENT_STATE_BF16="${TPU_GDN_RECURRENT_STATE_BF16:-0}"

export TPU_ACCELERATOR_TYPE=tpu7x
export USE_MOE_SPARSE_CORE=1
# The selector counts routed rows (tokens * top-k). Qwen3.5's c128 decode
# shape has 128 * 10 = 1280 rows, so 2048 retains its fast dense one-hot
# matmul while larger grouped-GMM shapes can use the SparseCore gather.
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"

# The XLA flags every record-holding job passes and we never did. In order:
# shard a trivially-shardable major dimension on the minor one instead; keep
# the reduce-scatter of a collective off SparseCore, which the MoE wants for
# its own gathers; combine all-reduces regardless of size; and leave async
# collectives unmerged. Set to empty to serve with stock XLA defaults.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false}"

# Add extra padding
# 4, 8 for low concurrency 4 and 8
# 48 for concurrency 256 when DP8+EP is enabled to reduce padding from 32->64.
# 3072 adds a finer prefill bucket between 2048 and 4096 where the local cap permits.
# Set it to empty to opt out of the extra buckets.
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8,48,3072}"

# Shrink the rotary cos_sin caches to max_model_len, text-only, to minimize
# xla layout data copy overhead.
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"

# Skip padded tokens in the fused MoE so they activate no experts.
export TPU_MOE_SKIP_PADDED_TOKENS=1

# Measured on this benchmark at 8k1k c256 DP8_EP (2560 prompts): 29,163 ->
# 31,101 total tok/s, median TPOT 60.15 -> 56.21 ms, GSM8K 0.980 strict.
#
# Destination-major SparseCore combine. The MoE combine is ~13% of device time
# at prefill and v3 roughly halves it.
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"
# One Pallas call for gmm1 -> activation -> gmm2, intermediate held in VMEM.
# Bit-exact against the two-call path; -8% on the MoE layer at the decode shape.
export USE_FUSED_MOE_GMM="${USE_FUSED_MOE_GMM:-1}"
# Stacked scheduling in the batched-RPA kernel. Worth +1.6% end to end on a
# mixed batch even though every fixed-shape microbenchmark says it is slower --
# the win is in how it schedules a mixed batch, not in any single shape.
# Leave USE_BATCHED_RPA_SEQ_ON_LANE off: that layout is slower here and, unlike
# this one, is not bit-identical to the mainline kernel.
export USE_BATCHED_RPA_LONGCTX="${USE_BATCHED_RPA_LONGCTX:-1}"
# Spend the packing slots that num_kv_heads * 2 leaves empty on head_dim.
# TP8 gives each rank a single KV head of this model, and an fp8 cache packs
# four values into a 32-bit word, so half of every cached token is structural
# zeros. The runner already says so: "KV cache pages are 2.00x larger than
# num_kv_heads=1 at head_size=256 ... requires". Decode rereads that cache
# every step, which is where the TP8 c32 point trails the reference by 4% on
# median ITL at an identical median TTFT. kv_head_dim_fold() returns 1 when
# there is nothing to reclaim, so DP8, whose ranks keep all eight KV heads,
# is unaffected, and tpu_platform refuses the fold outright on the layouts
# that cannot read it.
export TPU_RPA_FOLD_KV_HEAD_DIM="${TPU_RPA_FOLD_KV_HEAD_DIM:-1}"

args=(
  "$MODEL"
  --served-model-name="$SERVED_NAME"
  --max-model-len="$MAX_MODEL_LEN"
  --max-num-batched-tokens="$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs="$MAX_NUM_SEQS"
  --no-enable-prefix-caching
  --gpu-memory-utilization=0.92
  --block-size=256
  --async-scheduling
  --port="$PORT"
  --language-model-only
  --limit-mm-per-prompt='{"image": 0, "video": 0}'
  --quantization=fp8
  --kv-cache-dtype=fp8
  --attention-backend CUSTOM
  "${SHARDING_ARGS[@]}"
)
# Pass --load-format when streaming from GCS.
[ -n "${LOAD_FORMAT:-}" ] && args+=(--load-format="$LOAD_FORMAT")

# Word-split extra flags (profiler config, one-off overrides) onto the command
# line. Matches how run_benchmarks.sh consumes EXTRA_SERVE_ARGS.
if [ -n "${EXTRA_SERVE_ARGS:-}" ]; then
  read -r -a extra_serve_args <<< "$EXTRA_SERVE_ARGS"
  args+=("${extra_serve_args[@]}")
fi

# Print the resolved configuration and exit without serving. sweep.sh uses this
# to decide whether two points can share one running server. Most of these are
# derived from CONC, so a key the sweep builds from its own variables would call
# two points identical and benchmark the second against the first one's server.
if [ -n "${PRINT_CONFIG:-}" ]; then
  echo "${args[*]}" \
    "USE_MOE_FUSED_EP_KERNEL=${USE_MOE_FUSED_EP_KERNEL:-}" \
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-}"
  exit 0
fi

set -x
vllm serve "${args[@]}"
set +x
