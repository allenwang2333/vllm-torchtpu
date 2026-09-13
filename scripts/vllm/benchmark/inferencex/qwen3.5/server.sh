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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
EXTRA_ARGS=()
case "$SHARDING" in
  DP8_EP)
    # Data-parallel attention across all 8 cores + expert-parallel MoE, with
    # the settings measured at 64 to 1024 users (sweep.sh's table).
    DP_SIZE=8
    # New prefills are admitted every step; the router's release rules below
    # line the engines' prompt steps up instead of a longer interval.
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-1}"
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
      # One API-server process, so the router's release rules below see
      # every arrival (vLLM would start one per engine).
      --api-server-count=1
    )
    export DP_SCHED_ENABLED=1
    # The measured points ran with 1024 tokens of headroom over ISL+OSL.
    MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-1024}"
    # The fused expert-parallel MoE kernel for compiled shapes of
    # TPU_MOE_FUSED_EP_STEP_MIN_TOKENS rows per engine and above; smaller
    # shapes take the grouped-matmul path.
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
    export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS="${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-128}"
    # The DP router holds new prompts and releases a group together: an
    # idle-engine trigger, a 5 s deadline, a 2 s quiet period.
    export DP_PLACEMENT="${DP_PLACEMENT:-release}"
    export DP_SCHED_BUFFER_THRESHOLD="${DP_SCHED_BUFFER_THRESHOLD:-8}"
    export DP_SCHED_BUFFER_FLUSH_MS="${DP_SCHED_BUFFER_FLUSH_MS:-5000}"
    export DP_SCHED_BUFFER_IDLE_TRIGGER="${DP_SCHED_BUFFER_IDLE_TRIGGER:-1}"
    export DP_SCHED_BUFFER_QUIET_MS="${DP_SCHED_BUFFER_QUIET_MS:-2000}"
    # The static expert placement (vLLM's EPLB policy on a routing recording
    # at 128 users), applied when the checkpoint is loaded. Set it to empty
    # for the checkpoint's own order.
    export TPU_EXPERT_PLACEMENT="${TPU_EXPERT_PLACEMENT-$SCRIPT_DIR/expert_placement_dp8_ep8.json}"
    # The vocabulary projection split across the engines: on up to 128 users,
    # off above (sweep.sh sets it per point).
    export TPU_SHARDED_LM_HEAD="${TPU_SHARDED_LM_HEAD:-1}"
    # The grouped-matmul path's two expert matmuls as one call, the
    # counting-sort routing prep, the norm weights computed once, the GDN
    # output's gated norm and fp8 quantisation as one pass.
    export USE_FUSED_MOE_GMM="${USE_FUSED_MOE_GMM:-1}"
    export USE_MOE_COUNTING_SORT="${USE_MOE_COUNTING_SORT:-1}"
    export TPU_NORM_WEIGHT_HOIST="${TPU_NORM_WEIGHT_HOIST:-1}"
    export TPU_GDN_FUSED_NORM_QUANT="${TPU_GDN_FUSED_NORM_QUANT:-1}"
    export USE_BATCHED_RPA_LONGCTX="${USE_BATCHED_RPA_LONGCTX:-1}"
    export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"
    # The compiler flags of the measured points; the async all-gather lines
    # let the fused EP kernel's row gather run on the sparse cores. The
    # plugin prepends its own defaults and skips any flag already set here.
    DP8_LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false"
    if [ "$USE_MOE_FUSED_EP_KERNEL" = "1" ]; then
      DP8_LIBTPU_INIT_ARGS+=" --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
    fi
    export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS-$DP8_LIBTPU_INIT_ARGS}"
    # The GDN recurrent state in bf16: half the state memory, and the decode
    # step's state loads and stores through the pool's packed 16-bit view.
    EXTRA_ARGS=(--mamba-ssm-cache-dtype bfloat16)
    ;;
  TP8_EP)
    # Tensor-parallel attention across all 8 chips + expert-parallel MoE.
    DP_SIZE=1
    SHARDING_ARGS=(
      --tensor-parallel-size=8
      --data-parallel-size=1
      --enable-expert-parallel
    ) ;;
  DP4TP2_EP)
    # Data-parallel attention across all 4 chips with tensor-parallel across 2 cores
    # on the same chip + expert-parallel MoE.
    DP_SIZE=4
    # New prefills are admitted every this many steps. Larger value trades
    # TTFT for throughput. Default 8 keeps TTFT low.
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-8}"
    SHARDING_ARGS=(
      --tensor-parallel-size=2
      --data-parallel-size=4
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
    )
    export DP_SCHED_ENABLED=1
    ;;
  *) echo "ERROR: unknown SHARDING='$SHARDING' (DP8_EP, TP8_EP, DP4TP2_EP are wired up)" >&2; exit 1 ;;
esac

# Headroom over ISL+OSL (matches InferenceX qwen3.5: ISL+OSL+20; the DP8_EP
# case set its own above)
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-20}"
MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))

if [ "$SHARDING" = "DP8_EP" ]; then
  # Per point (sweep.sh's table): the tokens per engine per forward step
  # (the prompt chunk) and the requests per engine, with room for the
  # router to place a prompt on the engine with the fewest in flight. The
  # defaults are the 64-user point's.
  GLOBAL_MAX_NUM_BATCHED_TOKENS="${GLOBAL_MAX_NUM_BATCHED_TOKENS:-8192}"
  MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_MAX_NUM_BATCHED_TOKENS + DP_SIZE - 1) / DP_SIZE))
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-$((CONC * 2 / DP_SIZE))}"
else
  # Global num of batched tokens == max(ISL, GLOBAL_BATCHED_TOKENS_MIN).
  GLOBAL_BATCHED_TOKENS_MIN="${GLOBAL_BATCHED_TOKENS_MIN:-16384}"
  GLOBAL_BATCHED_TOKEN=$(( ISL > GLOBAL_BATCHED_TOKENS_MIN ? ISL : GLOBAL_BATCHED_TOKENS_MIN ))

  MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_BATCHED_TOKEN + DP_SIZE - 1) / DP_SIZE))

  MAX_NUM_SEQS=$((CONC / DP_SIZE))
fi
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1

# Increase API-server frontend wait time since cold init may take long.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

export TPU_ACCELERATOR_TYPE=tpu7x
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"

# Default to v3 which is a bit faster than v2.
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"

# Add extra padding
# 4, 8 for low concurrency 4 and 8
# 48 for concurrency 256 when DP8+EP is enabled to reduce padding from 32->64.
# Set it to empty to opt out of the extra buckets.
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8,48}"

# Shrink the rotary cos_sin caches to max_model_len, text-only, to minimize
# xla layout data copy overhead.
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"

# Skip padded tokens in the fused MoE so they activate no experts.
export TPU_MOE_SKIP_PADDED_TOKENS=1

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
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)
# Pass --load-format when streaming from GCS.
[ -n "${LOAD_FORMAT:-}" ] && args+=(--load-format="$LOAD_FORMAT")

set -x
vllm serve "${args[@]}"
set +x
