#!/bin/bash
# Shared environment for DeepSeek-V4-Flash runs on this TPU node.
# Source it; don't execute it.
#
# Every value here is overridable from the caller's environment, so a run can
# be varied without editing the file:
#   GPU_MEM_UTIL=0.7 scripts/dsv4/run_mmlu_pro.sh
#
# The env vars below are the same set the tpu-inference reference run uses
# (MOE_REQUANTIZE_*, NEW_MODEL_DESIGN, MODEL_IMPL_TYPE) plus the plugin/TP
# wiring that vllm-torchtpu needs and tpu-inference does not.

: "${WORKSPACE:=/home/patemotter_google_com/workspace}"
: "${REPO:=$WORKSPACE/vllm-torchtpu}"
: "${VENV:=$WORKSPACE/uv_venv_dsv4}"
: "${MODEL_PATH:=$WORKSPACE/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1}"
: "${SERVED_MODEL_NAME:=deepseek-ai/DeepSeek-V4-Flash}"

# --- model/quantization config (matches the tpu-inference reference run) ---
export MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-fp4}"
export MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-512}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"
export MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-vllm}"

# --- vllm-torchtpu plugin wiring (no tpu-inference equivalent) ---
export VLLM_PLUGINS="${VLLM_PLUGINS:-torchtpu,torchtpu_general}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
export VLLM_EP_SIZE="${VLLM_EP_SIZE:-8}"
export PYTHONPATH="${PYTHONPATH:-$REPO/src}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TORCH_DIST_TIMEOUT="${TORCH_DIST_TIMEOUT:-1200}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1200000}"
export VLLM_SHM_BROADCAST_TIMEOUT_S="${VLLM_SHM_BROADCAST_TIMEOUT_S:-1200}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-1200}"

# Requantized-FP4 weight cache. The cache key encodes block size and dtype
# (mxfp4.py `_fp4_cache_path`), so a cache built under a different
# MOE_REQUANTIZE_* config is no longer silently reused. Leave empty to disable.
export VLLM_FP4_CACHE_DIR="${VLLM_FP4_CACHE_DIR:-/dev/shm/dsv4-fp4-fp4x512}"

# vLLM's compile-artifact caches (VLLM_CACHE_ROOT, VLLM_XLA_CACHE_PATH) default
# to ~/.cache/vllm, which lives on the small root disk (/dev/root, ~145GiB)
# rather than the workspace disk (/dev/nvme1n1, ~2TiB, mounted at $WORKSPACE).
# The per-instance compile-cache isolation fix (compile_prefix keyed per
# layer) multiplies the number of on-disk cache directories -- 43 real layers
# x several bucket sizes x 8 ranks -- so this needs headroom. Root-disk-full
# has already taken down tpu-runtime.service once (its Docker overlay2 layer
# got wiped); keep compile caches off that disk entirely.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKSPACE/vllm_cache}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$VLLM_CACHE_ROOT/xla_cache}"

# Do NOT set LIBTPU_INIT_ARGS here. env_override.py computes the chip-correct
# sparse-core offload thresholds only when the user has not already supplied
# those flags; hardcoding them silently disables that logic.

# --- serving shape (see README for the mapping to the reference command) ---
: "${MAX_MODEL_LEN:=9216}"
: "${MAX_NUM_BATCHED_TOKENS:=1024}"
: "${MAX_NUM_SEQS:=128}"
: "${KV_CACHE_DTYPE:=fp8_e4m3}"
# Squeezed between two failures, not tuned for throughput. Floor: vLLM's DSv4
# KV budget check over-estimates the per-request requirement ~62x, demanding
# 42.86 GiB available at 9216, so util must exceed (42.86 + 19.2 weights) /
# 94.74 = 0.655 or the engine refuses to start. Ceiling: vLLM fills whatever is
# available with KV blocks, so 0.75 allocated a 52 GiB pool and left too little
# HBM for compile/activation buffers (RuntimeBufferAllocationFailure).
# 0.67 clears the floor with ~31 GiB left over. If weights or context change,
# recompute the floor rather than nudging this. See docs/dsv4_mmlu_pro_bringup.md.
: "${GPU_MEM_UTIL:=0.67}"
: "${TP_SIZE:=8}"
: "${PORT:=8000}"
: "${BLOCK_SIZE:=}"          # empty => let the TPU platform derive it
# NOTE: vLLM 0.26 enables async scheduling by DEFAULT on this path -- the
# server log says "Asynchronous scheduling is enabled" even with this at 0.
# This flag only forces the CLI switch on; it cannot turn the feature off.
# Matches the reference, which passes --async-scheduling explicitly.
: "${ASYNC_SCHEDULING:=1}"   # 1 => pass --async-scheduling
: "${VLLM_ENFORCE_EAGER:=0}" # 1 => --enforce-eager (slow; only verified path)
export VLLM_ENFORCE_EAGER

# vLLM auto-sets VLLM_USE_BREAKABLE_CUDAGRAPH=1 for DeepseekV4ForCausalLM
# (vllm/config/vllm.py), which forces compilation_config.mode=NONE and
# silently no-ops every @support_torch_compile decoration in
# deepseek_v4_patch.py -- the model runs eager regardless of enforce_eager.
# Opt back out so the per-layer/head compile path actually engages.
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"

# vLLM auto-enables VLLM_USE_AOT_COMPILE (torch>=2.10, envs.use_aot_compile())
# unless told otherwise. Its cache key (decorators.py _model_hash_key /
# aot_compile_hash_factors) hashes only vllm.__version__, forward.__qualname__
# and forward.__code__.co_firstlineno -- none of it instance-specific. Every
# DeepseekV4DecoderLayer instance shares one class, so layer 0's AOT artifact
# gets silently reused by layers 1..N via _try_load_aot_compiled_fn, closing
# over layer 0's swa_cache shape and crashing (e0102 RuntimeProgramInputMismatch)
# the moment a layer with a different overlay shape actually runs. This
# bypasses TorchCompileWithNoGuardsWrapper.__call__ entirely (and therefore
# the per-instance compile_prefix isolation in
# _patch_vllm_compile_prefix_isolation), so that patch can't fix it -- confirmed
# via DSV4_DEBUG_CACHE_SHAPES=1 showing zero calls to __call__ and the log
# alternating "saved AOT compiled function" / "Directly load AOT compilation
# from path" on the same hash across different layers. Disable AOT compile so
# per-layer/head compilation actually goes through the (already
# instance-isolated) TorchCompileWithNoGuardsWrapper path.
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-0}"

# Kill leftover servers/workers WITHOUT killing ourselves.
#
# A plain `pkill -f "vllm serve"` is a live grenade here: the caller's own
# command line frequently contains that string (a wrapper that echoes the
# command, a shell invoked with it inline), so pkill matches the very process
# running this cleanup and the run dies silently the moment it starts. This
# walks the match list and skips this process and its ancestors.
dsv4_kill_server() {
    local pat pid ancestors p
    ancestors=" "
    p=$$
    while [[ -n "$p" && "$p" != "0" && "$p" != "1" ]]; do
        ancestors+="$p "
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    done
    for pat in "vllm serve" "VLLM::" "vllm.entrypoints.openai.api_server"; do
        for pid in $(pgrep -f "$pat" 2>/dev/null); do
            [[ "$ancestors" == *" $pid "* ]] && continue
            kill -9 "$pid" 2>/dev/null || true
        done
    done
    rm -rf /tmp/libtpu_lockfile* /tmp/ipc* /tmp/psm* /dev/shm/psm_* 2>/dev/null || true
    sleep 5
}
