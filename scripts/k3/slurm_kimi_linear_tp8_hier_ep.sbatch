#!/usr/bin/env bash
#SBATCH --job-name=kimi-linear-tp8-hierep
#SBATCH --partition=dev-2x2x1
#SBATCH --nodelist=tpu-rack1-w-014
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out
#
# Kimi-Linear-48B-A3B-Instruct at TP8 on one tpu7x host: does the hierarchical
# expert-parallel correctness failure reproduce on a single node?
#
# Context: K3_HIERARCHICAL_EP_PLAN.md, "CORRECTNESS FAILURE at TP32 (job 482)".
# Hierarchical EP (EP between chips, TP within a chip) at TP32 on full K3 made
# 47% of GSM8K responses degenerate into repetition loops and CJK spam, while
# the same code path on pruned K3 at TP8 showed none. Every TP32 diagnosis costs
# a 4-node allocation and ~an hour of load; this job asks the same question for
# ~15 minutes on the idle dev node.
#
# WHY KIMI-LINEAR IS A USABLE VEHICLE. It is served by the same code: vllm_torchtpu
# registers KimiLinearForCausalLM out of models/vllm/kimi_k3/, so its MoE is the
# same KimiMoE, wrapped in the same hierarchical_moe_parallel_config() context.
# Geometry at TP8 (256 experts, topk 8, moe_intermediate 1024):
#
#     flat EP8       32 experts/rank, intermediate 1024 (tp_size=1)
#     hierarchical   64 experts/chip, intermediate  512 per device (EP4 x TP2)
#
# and 92 GB bf16 loads in minutes rather than the ~40 the 443 GB K3 takes.
#
# WHAT IT CANNOT TELL YOU -- read this before interpreting a clean result.
# Two things the TP32 run exercised are absent here, and one of them is the
# plan's own prime suspect:
#
#   1. mxfp4. Kimi-Linear is bf16. The leading hypothesis for job 482 is w2_scale
#      sharding in compressed_tensors_moe_w4an_mxfp4.py, whose padded/unpadded
#      zero-fill is keyed on intermediate_size_per_partition -- which halves
#      under the split, on a path that has only ever run with tp_size=1 under EP.
#      That code does not execute at all in this job.
#   2. KimiRoutedOutputTransform. Kimi-Linear's config has no
#      routed_expert_hidden_size, so there is no latent MoE, no pre-norm
#      all-reduce and no /tp_size compensation. K3 has all three.
#
# So a CLEAN run here does not exonerate hierarchical EP; it narrows the suspect
# list to {mxfp4 scales, latent-MoE transform, genuinely TP32-only effects}.
# A DIRTY run is the far more valuable outcome: it would rule the mxfp4 path out,
# put the bug in the generic expert-block/width sharding, and hand the
# investigation a single-node reproducer.
#
# What it does exercise: chip-block expert ownership (determine_expert_map,
# get_experts_start), the realigned --enable-ep-weight-filter, the width split of
# w13/w2 inside an expert, and the 8-way all-reduce that has to sum 4 chips x 2
# partial halves.
#
# Usage
#   sbatch scripts/k3/slurm_kimi_linear_tp8_hier_ep.sbatch
#   sbatch --export=ALL,ARMS="flat hier hier-nofilter",DOCS=200 \
#          scripts/k3/slurm_kimi_linear_tp8_hier_ep.sbatch
#   ARMS="hier" scripts/k3/slurm_kimi_linear_tp8_hier_ep.sbatch   # on an salloc
set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}}"
MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
TP="${TP:-8}"
PORT="${PORT:-8010}"

# Arms, run sequentially in one allocation so they share a node, a build and a
# checkpoint copy. Recognised names:
#   flat           vLLM's own EP8 -- the control
#   hier           TPU_MOE_HIERARCHICAL_EP=1, EP weight filter ON  (the TP32 config)
#   hier-nofilter  TPU_MOE_HIERARCHICAL_EP=1, EP weight filter OFF
# The third arm exists because the filter realignment (commit 1309b755) is
# itself a suspect: it rebinds the loader's ownership function, and if that
# substitution is wrong a rank silently keeps randomly-initialized experts.
# Running hier without it separates "the loader read the wrong weights" from
# "the sharded compute is wrong" -- one bit of information the TP32 runs never
# collected.
#
# A "-b2" suffix on any arm adds the size-2 compile bucket to COMPILE_SIZES.
# That is the second variable job 482 changed at the same time as switching on
# hierarchical EP, and it was never accuracy-tested on its own: jobs 480/481
# isolated it for throughput under --ignore-eos, where the output text is not
# read. Run it at EVAL_CONC=1 -- at concurrency 1 a decode step holds one real
# sequence, so with a size-2 bucket EVERY step runs one real row beside one pad
# row. At EVAL_CONC=8 the steps land in the size-8 bucket and the arm tests
# nothing.
ARMS="${ARMS:-flat hier}"

DOCS="${DOCS:-100}"
EVAL_CONC="${EVAL_CONC:-8}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$EVAL_CONC}"
BATCHED_TOKENS="${BATCHED_TOKENS:-1024}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"

SP="${SP:-$REPO/.runtime-cache/kimi-linear-tp8-hierep}"
RUN="${SP}/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN"

echo "== node:  $(hostname -s)"
echo "== repo:  $REPO"
echo "== git:   $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "== model: $MODEL  TP=$TP"
echo "== arms:  $ARMS   (${DOCS} gsm8k docs each)"
echo "== out:   $RUN"

# ---------------------------------------------------------------------------
# Checkpoint location.
#
# /mnt/scratch is node-local NVMe and differs per node; /mnt/lustre is shared and
# slower. The two mounts also use DIFFERENT layouts -- the scratch copy puts
# models--* at the root of /mnt/scratch/hf-hub, the lustre copy under
# /mnt/lustre/hf-models/hub -- so probe and set HF_HUB_CACHE explicitly rather
# than deriving it from HF_HOME. An inherited HF_HUB_CACHE from
# /etc/profile.d/shared-env.sh has silently redirected loads before.
# ---------------------------------------------------------------------------
MODEL_DIR="models--${MODEL//\//--}"
# A snapshots/ directory is NOT evidence of a usable checkpoint. On w-013 the
# Kimi-Linear snapshot holds config.json and the index but none of the 20
# safetensors shards, and being probed first it shadowed the complete lustre
# copy and failed the load with "Cannot find any model weights". Require at
# least one shard before accepting a root.
has_weights() {
  compgen -G "$1/$MODEL_DIR/snapshots/*/*.safetensors" >/dev/null 2>&1
}
if has_weights /mnt/scratch/hf-hub; then
  export HF_HUB_CACHE=/mnt/scratch/hf-hub
elif has_weights /mnt/scratch/hf-hub/hub; then
  export HF_HUB_CACHE=/mnt/scratch/hf-hub/hub
elif [ -d "/mnt/lustre/hf-models/hub/$MODEL_DIR/snapshots" ]; then
  export HF_HUB_CACHE=/mnt/lustre/hf-models/hub
  # Optional: stage to node-local NVMe. Worth it when several arms each reload
  # the checkpoint; skip it (STAGE=0) for a single arm, where the copy costs
  # more than it saves. As of 2026-08-11 the 92 GB Kimi-Linear snapshot lives
  # only on lustre while /mnt/scratch on w-014 has ~2.9 TB free.
  if [ "${STAGE:-1}" = "1" ] && [ "$(df --output=avail -BG /mnt/scratch | tail -1 | tr -dc 0-9)" -gt 150 ]; then
    echo "== staging $MODEL_DIR to node-local nvme (once; ~92 GB)"
    mkdir -p /mnt/scratch/hf-hub
    if cp -a "/mnt/lustre/hf-models/hub/$MODEL_DIR" /mnt/scratch/hf-hub/ ; then
      export HF_HUB_CACHE=/mnt/scratch/hf-hub
      echo "== staged"
    else
      echo "!! staging failed; falling back to lustre"
      rm -rf "/mnt/scratch/hf-hub/$MODEL_DIR"
    fi
  fi
else
  echo "!! $MODEL not found on any known cache root" >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$(dirname "$HF_HUB_CACHE")}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "== checkpoint: $HF_HUB_CACHE/$MODEL_DIR"

# The eval client resolves GSM8K through the same HF_HUB_CACHE, as
# datasets--openai--gsm8k. Both mounts happen to have it on w-014, but pin the
# fallback so an unstaged node does not fail the eval after a successful load.
GSM8K_HF_HUB_CACHE="${GSM8K_HF_HUB_CACHE:-$HF_HUB_CACHE}"
[ -d "$GSM8K_HF_HUB_CACHE/datasets--openai--gsm8k" ] || \
  GSM8K_HF_HUB_CACHE=/mnt/lustre/hf-models/hub

# ---------------------------------------------------------------------------
# Environment.
#
# $HOME is NOT writable on these compute nodes -- /home/george is the login
# node's. Anything defaulting to ~ dies partway through startup, and not
# cleanly: it has surfaced as "[Errno 13] Permission denied: /home/george" from
# inside determine_available_memory, which has nothing to do with the cause.
# Redirect HOME itself as well as the specific cache roots so nothing has to be
# enumerated correctly.
# ---------------------------------------------------------------------------
export PATH="$REPO/.venv/bin:$PATH"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export VLLM_TARGET_DEVICE=tpu
export JAX_PLATFORMS=""

# Single host: 4 chips x 2 cores = 8 devices. No Ray, no multihost backend --
# this is the one structural difference from slurm_launch_k3_tp32_tp.sbatch.
export TPU_ACCELERATOR_TYPE=tpu7x
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,1,2}"
unset TPU_MULTIHOST_BACKEND || true

export HOME="$SP/home"
export XDG_CACHE_HOME="$SP/xdg-cache"
export VLLM_CACHE_ROOT="$SP/vllm"
export VLLM_CONFIG_ROOT="$SP/vllm-config"
export HF_MODULES_CACHE="$SP/hf_modules"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
         "$HF_MODULES_CACHE"

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TPU_HBM_REPORT_ENABLE=0
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"
export TF_CPP_MIN_LOG_LEVEL=1
export TPU_STDERR_LOG_LEVEL=3
export TPU_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# KV blocks are PINNED, and on this model that is mandatory rather than tuning.
# With utilisation-based sizing each rank measures its own free HBM and derives
# its own block count; executors/tpu_multiproc_executor.py then asserts
#
#     max(overrides) - min(overrides) <= _NUM_BLOCKS_OVERRIDE_TOL   (=4)
#
# The tolerance is an ABSOLUTE block count. It holds for K3 (a few hundred
# blocks) and cannot hold for a 92 GB model that leaves most of HBM free: the
# eight ranks landed on 516517..516527, a spread of 10 -- 0.002% measurement
# jitter, not a disagreement. 4096 blocks x 16 tokens = 65k, against the 32k
# this eval needs at concurrency 8 x 4096 tokens.
#
# It is a per-MODEL setting and it is fatal in both directions. Pass
# KV_BLOCKS="" for Kimi-K3 and let vLLM profile instead: K3's blocks cost
# ~56 MB each (the hybrid KDA state dominates, not the MLA KV), so after
# 71.8 GiB of weights and workspace only 13.5 GiB is left and vLLM sizes 244
# blocks. Forcing 4096 there asks for ~231 GiB and dies mid-load with
#
#     Error allocating device buffer: Attempting to allocate 2.97G.
#     That was not possible. There are 1.62G free.
#
# which reads like a model that does not fit rather than a bad flag (job 551).
# Note "${KV_BLOCKS-...}", not ":-": an explicitly empty value must survive.
KV_BLOCKS="${KV_BLOCKS-4096}"

SERVER_PID=""
stop_server() {
  [ -z "$SERVER_PID" ] && return 0
  kill "$SERVER_PID" 2>/dev/null
  # Workers are named VLLM::Worker_TPn_EPn and outlive the launcher often
  # enough to strand the TPUs for the next arm.
  for _ in $(seq 1 60); do
    pgrep -f 'VLLM::Worker' >/dev/null 2>&1 || break
    sleep 2
  done
  pkill -9 -f 'VLLM::Worker' 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  # libtpu does not always release the device the instant the process dies.
  sleep 20
}
trap 'stop_server; exit 143' TERM
trap 'stop_server; exit 130' INT
trap 'stop_server' EXIT

run_arm() {
  local arm="$1"
  local log="$RUN/$arm.server.log"
  local hier=0 filter=1 bucket2=0
  local base="${arm%-b2}"
  [ "$base" != "$arm" ] && bucket2=1

  case "$base" in
    flat)          hier=0; filter=1 ;;
    flat-nofilter) hier=0; filter=0 ;;
    hier)          hier=1; filter=1 ;;
    hier-nofilter) hier=1; filter=0 ;;
    *) echo "!! unknown arm '$arm'" >&2; return 1 ;;
  esac

  local sizes="${COMPILE_SIZES:-}"
  [ "$bucket2" = 1 ] && sizes="2${sizes:+,$sizes}"
  # A prefill step whose shape has no compiled bucket recompiles mid-run, which
  # VLLM_XLA_CHECK_RECOMPILATION=1 makes fatal. Worth checking rather than
  # discovering after the load: `sbatch --export=ALL,COMPILE_SIZES="8,16,512"`
  # silently truncates the value to 8, because --export splits on commas.
  if [ -n "$sizes" ]; then
    case ",$sizes," in
      *",$BATCHED_TOKENS,"*) ;;
      *) echo "!! COMPILE_SIZES=[$sizes] has no bucket for "
         echo "!! --max-num-batched-tokens=$BATCHED_TOKENS. Pass commas escaped"
         echo "!! (COMPILE_SIZES=8\\,16\\,512) or export the variable before"
         echo "!! sbatch instead of listing it in --export."
         return 1 ;;
    esac
  fi

  echo
  echo "############################################################"
  echo "== ARM $arm  (hierarchical=$hier, ep_weight_filter=$filter, "
  echo "==            compile_sizes=[${sizes:-default}])"
  echo "############################################################"

  local args=(
    serve "$MODEL"
    --tensor-parallel-size="$TP"
    # Hierarchical EP only engages on top of real expert parallelism: the patch
    # in moe_hierarchical.py bails out with a warning when use_ep is false,
    # because forcing a split onto a flat TP config would change what it means.
    --enable-expert-parallel
    --max-model-len="$MAX_MODEL_LEN"
    --max-num-batched-tokens="$BATCHED_TOKENS"
    --max-num-seqs="$MAX_NUM_SEQS"
    --block-size=16
    --attention-backend=CUSTOM
    --trust-remote-code
    --no-enable-prefix-caching
    --async-scheduling
    --gpu-memory-utilization="${GPU_MEM_UTIL:-0.85}"
    --safetensors-load-strategy=prefetch
    --port="$PORT"
  )
  # KV_BLOCKS="" opts out of the override and lets vLLM profile the device.
  # Which one is right is per-model, and getting it backwards is fatal both
  # ways -- see the KV_BLOCKS comment above.
  [ -n "$KV_BLOCKS" ] && args+=(--num-gpu-blocks-override="$KV_BLOCKS")
  [ "$filter" = 1 ] && args+=(--enable-ep-weight-filter)
  [ -n "$sizes" ] && args+=("--compilation-config={\"compile_sizes\":[$sizes]}")

  # Kimi-K3 needs four things Kimi-Linear does not. It is multimodal, so the
  # vision tower has to be skipped; and its packaged chat template is the
  # multimodal XTML one, so a text-only template must be supplied or the
  # /v1/chat/completions the probe uses renders wrong. Keeping both models in
  # one harness is deliberate -- "the same harness" is the whole claim being
  # made when their degeneration rates are compared.
  case "$MODEL" in
    *Kimi-K3*|*kimi-k3*)
      args+=(
        --language-model-only
        '--limit-mm-per-prompt={"image":0,"video":0}'
        --chat-template="$REPO/scripts/kimi_k3_text_chat_template.jinja"
        '--default-chat-template-kwargs={"thinking_effort":"low"}'
      )
      ;;
  esac

  # Per-arm cache root. Sharing one would be fine for correctness -- the cache
  # keys on the compiled graph -- but the arms produce different MoE shapes and
  # keeping them apart makes "did this arm recompile" answerable from the log.
  VLLM_CACHE_ROOT="$SP/vllm-$arm" TPU_MOE_HIERARCHICAL_EP="$hier" \
    vllm "${args[@]}" >"$log" 2>&1 &
  SERVER_PID=$!

  echo "== waiting up to ${READY_TIMEOUT}s for /v1/models  (log: $log)"
  local start; start=$(date +%s)
  while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!! server exited early; tail:"; tail -60 "$log"
      SERVER_PID=""; return 1
    fi
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    if [ $(( $(date +%s) - start )) -gt "$READY_TIMEOUT" ]; then
      echo "!! timeout; tail:"; tail -60 "$log"; stop_server; return 1
    fi
    sleep 5
  done
  echo "== SERVER_READY after $(( $(date +%s) - start ))s"

  # Prove the sharding the arm claims is the sharding it got. A hierarchical arm
  # that silently fell back to flat -- unreadable coords, a world size that does
  # not match the visible devices -- would produce a clean result for the wrong
  # reason, and the fallback is only a WARNING in the log.
  # grep -c prints 0 and EXITS 1 on no match, so an `|| echo 0` fallback would
  # append a second line and make the numeric tests below throw.
  count_in_log() { local n; n=$(grep -c "$1" "$log" 2>/dev/null); echo "${n:-0}"; }
  local engaged fallback realigned
  engaged=$(count_in_log "Hierarchical MoE parallelism")
  fallback=$(count_in_log "falling back to flat expert parallelism\|leaving the MoE parallel config alone")
  realigned=$(count_in_log "EP weight filter realigned")
  echo "== split check: hierarchical-config lines=$engaged  fallbacks=$fallback  filter-realigned=$realigned"
  grep -m1 "Chip topology:" "$log" || true
  if [ "$hier" = 1 ] && { [ "$engaged" -eq 0 ] || [ "$fallback" -gt 0 ]; }; then
    echo "!! ARM $arm did NOT get the hierarchical split; its result means nothing."
  fi
  if [ "$hier" = 0 ] && [ "$engaged" -gt 0 ]; then
    echo "!! ARM $arm was supposed to be flat but the split engaged."
  fi

  # Canary first: one line, ~10 seconds, and at TP32 it was enough to show the
  # failure. No reason to spend the GSM8K run to find out.
  echo "== canary"
  HF_HUB_CACHE="$GSM8K_HF_HUB_CACHE" python "$REPO/scripts/k3/gsm8k_degeneration_probe.py" \
    --canary --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" --summary "$RUN/$arm.canary.json" || true

  echo "== gsm8k: $DOCS docs, concurrency $EVAL_CONC"
  HF_HUB_CACHE="$GSM8K_HF_HUB_CACHE" python "$REPO/scripts/k3/gsm8k_degeneration_probe.py" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
    --limit "$DOCS" --concurrency "$EVAL_CONC" --max-tokens "$MAX_TOKENS" \
    --label "$arm" --output "$RUN/$arm.jsonl" --summary "$RUN/$arm.json" \
    | tail -30

  stop_server
}

for arm in $ARMS; do
  run_arm "$arm" || echo "!! ARM $arm failed; continuing"
done

echo
echo "############################################################"
echo "== SUMMARY   $RUN"
echo "############################################################"
python - "$RUN" <<'PY'
import json, pathlib, sys

run = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(run.glob("*.json")):
    if path.name.endswith(".canary.json"):
        continue
    try:
        rows.append((path.stem, json.loads(path.read_text())))
    except Exception as error:
        print(f"  {path.name}: unreadable ({error})")

if not rows:
    print("  no arm summaries written")
    raise SystemExit(0)

head = f"{'arm':<16}{'acc':>8}{'degen':>8}{'cjk':>6}{'len%':>7}{'mean_tok':>10}{'n':>5}"
print(head)
print("-" * len(head))
for arm, s in rows:
    def pct(v):
        return f"{100 * v:.1f}" if isinstance(v, (int, float)) else "-"
    print(f"{arm:<16}{pct(s.get('accuracy')):>8}{pct(s.get('degeneration_rate')):>8}"
          f"{s.get('cjk_tail', '-'):>6}{pct(s.get('finish_length_rate')):>7}"
          f"{s.get('mean_completion_tokens') or 0:>10.1f}{s.get('completed', 0):>5}")

print()
print("Read `degen` first, not `acc`. The TP32 failure moved degeneration from")
print("0% to 47% while accuracy fell only 4pp, which 100 docs cannot resolve.")
PY
