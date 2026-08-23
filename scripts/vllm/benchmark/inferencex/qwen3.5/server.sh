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
# Headroom over ISL+OSL (matches InferenceX qwen3.5: ISL+OSL+20)
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-20}"

case "$SHARDING" in
  DP8_EP)
    # Data-parallel attention across all 8 cores + expert-parallel MoE.
    DP_SIZE=8
    # Match the measured-best DP8 jobs: admit prefills every 2 scheduler steps.
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-2}"
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
    )
    # Disable the locally ported coordinated scheduler for this reproduction.
    export DP_SCHED_ENABLED=0
    export TPU_DP_PREFILL_FLUSH_TIMEOUT_MS=0
    export TPU_DP_PREFILL_QUORUM_FRACTION=1.0
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
    # Picking the MoE path per compiled token bucket forks the backbone trace,
    # and the two forks lift different graph inputs, so warmup dies handing a
    # bucket between them (job j-a0850720). Keep every bucket on the fused EP
    # kernel until both paths lift the same inputs. Override for a single run.
    export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS="${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-0}"
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
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-256}"
    SHARDING_ARGS=(
      --tensor-parallel-size=2
      --data-parallel-size=4
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
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

# The global token target is normally divided across DP ranks. The measured
# DP8 high-concurrency points use smaller per-rank chunks instead.
DEFAULT_MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_BATCHED_TOKEN + DP_SIZE - 1) / DP_SIZE))
case "$SHARDING:$CONC" in
  DP8_EP:128|DP8_EP:256) DEFAULT_MAX_NUM_BATCHED_TOKENS=256 ;;
esac
# Max sequences is a node-wide target divided across DP ranks, at twice the
# client concurrency. Sizing it to the concurrency exactly leaves the engine no
# slot to admit a new request while the batch drains, and none of the top TP8
# results others recorded at 8k1k do it: the c16, c32 and c64 record holder
# runs 24, 48 and 64, and the c4 and c8 record holder runs 256 throughout. Two
# is a uniform stand-in for that hand-tuned table, close to the 1.5x those
# middle points use, and long over the 1x the c64 point wants.
DEFAULT_MAX_NUM_SEQS=$(((2 * CONC + DP_SIZE - 1) / DP_SIZE))
case "$SHARDING:$CONC" in
  DP8_EP:128|DP8_EP:256) DEFAULT_MAX_NUM_SEQS=64 ;;
esac
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_NUM_BATCHED_TOKENS}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEFAULT_MAX_NUM_SEQS}"
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1

# Increase API-server frontend wait time since cold init may take long.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

export TPU_ACCELERATOR_TYPE=tpu7x
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768

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
