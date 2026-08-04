#!/bin/bash
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
#
# Run lm_eval against an already-running local vLLM server.
#
# This is the POST_READY_CMD half of the pair: glm_serve.sh brings the server
# up and health-polls it, then invokes this. Keeping them separate means there
# is one server implementation, and an eval failure cannot be mistaken for a
# load failure.
#
# Reference invocation this adapts:
#
#   lm_eval --model local-completions \
#     --model_args base_url=http://localhost:8000/v1/completions,\
#   model=nvidia/GLM-5.2-NVFP4,tokenizer=nvidia/GLM-5.2-NVFP4,\
#   max_gen_toks=4096,max_length=32768,num_concurrent=32,timeout=7200 \
#     --apply_chat_template --tasks mmlu_pro --seed 42 --limit 100 \
#     --gen_kwargs "max_new_tokens=4096,until=['<|im_end|>']" &
#
# Adaptations, all forced by the environment:
#   * no trailing `&` -- the container must run the eval to completion and
#     then exit, or the four chips stay held until TTL cleanup;
#   * lm_eval is installed from the NFS checkout copied to /tmp, because the
#     mount is read-only and a pip build wants to write egg-info next to the
#     source;
#   * results are written to the weights PVC, since --mount-gcs is off (the
#     GCS FUSE sidecar crash-loops on these nodes) and the container fs is
#     ephemeral.
set -uo pipefail

# Same stale-handle hazard as glm_serve.sh: bash reads the script
# incrementally, so an edit to the NFS copy mid-run aborts it. Re-exec local.
if [ "${GLM_EVAL_LOCAL:-0}" != "1" ]; then
  cp "${BASH_SOURCE[0]}" /tmp/glm_eval.local.sh || {
    echo "### FATAL: could not stage the eval script locally"; exit 6; }
  export GLM_EVAL_LOCAL=1
  exec bash /tmp/glm_eval.local.sh "$@"
fi

MODEL="${MODEL_ID:-nvidia/GLM-5.2-NVFP4}"
PORT="${SERVE_PORT:-8000}"
TASKS="${EVAL_TASKS:-mmlu_pro}"
LIMIT="${EVAL_LIMIT:-100}"
CONC="${EVAL_CONCURRENT:-32}"
MAX_GEN="${EVAL_MAX_GEN_TOKS:-4096}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${EVAL_SEED:-42}"
# No `until` override: mmlu_pro stops on "Question:" and gpqa on "</s>", and a
# global override replaces both. The reference command used until=['<|im_end|>'],
# which is a ChatML token GLM does not have -- its EOS is <|endoftext|> and its
# turn markers are <|user|>/<|assistant|>/<|observation|> -- so it never matched
# and only served to discard the tasks' own stops.
#
# max_gen_toks does need overriding: mmlu_pro's default is 2048, and GLM-5.2 is
# a reasoning model whose chain of thought can exceed that and get truncated
# before it emits "The answer is X", which the scoring regex needs. 30000
# matches what the working multi-node GLM recipe uses.
GEN_KWARGS="${EVAL_GEN_KWARGS:-max_gen_toks=30000}"
LM_EVAL_SRC="${LM_EVAL_SRC:-/nfs/lm-eval}"
OUT_DIR="${EVAL_OUT_DIR:-/hf-cache/lm_eval_out}"
LOG_SAMPLES="${EVAL_LOG_SAMPLES:-0}"

echo "### lm_eval: tasks=$TASKS limit=$LIMIT concurrent=$CONC seed=$SEED"
echo "### model=$MODEL port=$PORT out=$OUT_DIR"
# `mmlu_pro` is a group of 14 subtasks and --limit applies PER SUBTASK, so
# limit=100 is 1400 generations, not 100. Worth knowing before reading the
# wall-clock, and worth knowing when comparing against a published number.
case "$TASKS" in
  *mmlu_pro*) [ "$LIMIT" != "0" ] &&
    echo "### note: mmlu_pro is 14 subtasks; --limit $LIMIT => ~$((LIMIT * 14)) samples" ;;
esac

# ---------------------------------------------------------------- install ---
# lm_eval does not depend on torch, but `datasets`/`evaluate` can drag numpy
# or pyarrow upward, and a numpy bump is enough to break jax and torch_tpu in
# a way that only shows up as a cryptic import error mid-run. Pin whatever is
# already installed so pip cannot move it.
freeze_constraints() {
  python3 - "$1" <<'PY'
import importlib.metadata as md, sys
out = sys.argv[1]
with open(out, "w") as fh:
    for name in ("torch", "torch_tpu", "torch-tpu", "numpy", "jax", "jaxlib",
                 "transformers", "pyarrow", "tokenizers", "safetensors"):
        try:
            fh.write(f"{name}=={md.version(name)}\n")
        except md.PackageNotFoundError:
            pass
print(open(out).read().strip().replace("\n", "  "))
PY
}

if ! command -v lm_eval >/dev/null 2>&1; then
  CONSTRAINTS=/tmp/lm_eval_constraints.txt
  echo "### pinning: $(freeze_constraints "$CONSTRAINTS")"
  if [ -d "$LM_EVAL_SRC" ]; then
    # Read-only NFS mount: build from a scratch copy so pip can write
    # egg-info next to the source.
    echo "### installing lm_eval from $LM_EVAL_SRC"
    rm -rf /tmp/lm-eval && cp -r "$LM_EVAL_SRC" /tmp/lm-eval || {
      echo "### FATAL: could not stage $LM_EVAL_SRC"; exit 3; }
    SPEC="/tmp/lm-eval[api]"
  else
    echo "### $LM_EVAL_SRC not mounted; falling back to PyPI"
    SPEC="lm_eval[api]"
  fi
  pip install --quiet --no-cache-dir -c "$CONSTRAINTS" "$SPEC" 2>&1 | tail -8
  rc=${PIPESTATUS[0]}
  command -v lm_eval >/dev/null 2>&1 || {
    echo "### FATAL: lm_eval still not on PATH (pip rc=$rc)"; exit 3; }
  # The pin is only useful if we check it held.
  python3 -c "
import torch, numpy
print(f'### post-install: torch {torch.__version__} numpy {numpy.__version__}')
import torch_tpu  # noqa: F401
print('### torch_tpu still imports')
" || { echo "### FATAL: install broke the TPU stack"; exit 3; }
fi
echo "### lm_eval $(lm_eval --version 2>&1 | head -1 || python3 -c 'import lm_eval;print(lm_eval.__version__)' 2>/dev/null)"

# ------------------------------------------------------------- readiness ---
# BASE is localhost when this runs as glm_serve.sh's POST_READY_CMD, and the
# JobSet headless DNS name when it runs as a separate CPU pod. In the split
# case the server is not up yet -- this pod schedules immediately while the
# TPU pod is still pulling and loading 433 GiB.
BASE="${EVAL_BASE_URL:-http://localhost:${PORT}}"

# "discover" resolves the JobSet headless service instead of a per-pod DNS
# name. Per-pod names need network.enableDNSHostnames, which makes JobSet
# assign each pod a hostname/subdomain -- and with that set, every TPU worker
# in the server pod aborts during device init with
#   RAW: absl::container_internal::raw_hash_map<>::at
# before the model is even loaded (reproduced twice; the identical single-pod
# job serves fine). The headless service still publishes A records for all
# pods without it, so find the one answering /health on the serving port.
if [ "$BASE" = "discover" ]; then
  echo "### discovering the server via the ${JS_NAME} headless service"
  found=""
  for attempt in $(seq 1 "${EVAL_DISCOVER_TRIES:-360}"); do
    for ip in $(getent ahostsv4 "${JS_NAME}" 2>/dev/null | awk '{print $1}' | sort -u); do
      if curl -sf -o /dev/null --connect-timeout 2 "http://${ip}:${PORT}/health"; then
        found="http://${ip}:${PORT}"; break 2
      fi
    done
    [ $((attempt % 20)) -eq 0 ] && echo "### still discovering ... ${attempt}0s"
    sleep 10
  done
  [ -n "$found" ] || { echo "### FATAL: no pod answered /health on :${PORT}"; exit 4; }
  BASE="$found"
fi
echo "### server base: $BASE"

WAIT_HEALTH="${EVAL_WAIT_HEALTH:-5400}"
waited=0
until curl -sf -o /dev/null --connect-timeout 5 "$BASE/health"; do
  if [ "$waited" -ge "$WAIT_HEALTH" ]; then
    echo "### FATAL: no /health from $BASE within ${WAIT_HEALTH}s"
    exit 4
  fi
  sleep 10
  waited=$((waited + 10))
  [ $((waited % 300)) -eq 0 ] && echo "### waiting for /health ... ${waited}s"
done
echo "### /health up after ${waited}s"

# /health answers before the engine has finished compiling its graphs, so gate
# on a real completion too. Without this the first lm_eval batch races the
# warmup and 32 concurrent requests all time out at once.
WAIT_GEN="${EVAL_WAIT_GEN:-1800}"
probed=0
until curl -sf -o /dev/null "$BASE/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"hello\", \"max_tokens\": 4}"; do
  if [ "$probed" -ge "$WAIT_GEN" ]; then
    echo "### FATAL: no completion from $BASE within ${WAIT_GEN}s"
    exit 4
  fi
  sleep 10
  probed=$((probed + 10))
  [ $((probed % 300)) -eq 0 ] && echo "### waiting for a completion ... ${probed}s"
done
echo "### engine served a completion after ${probed}s"

mkdir -p "$OUT_DIR" 2>/dev/null || OUT_DIR=/tmp/lm_eval_out && mkdir -p "$OUT_DIR"

MODEL_ARGS="base_url=${BASE}/v1/completions"
MODEL_ARGS="${MODEL_ARGS},model=${MODEL}"
MODEL_ARGS="${MODEL_ARGS},tokenizer=${MODEL}"
MODEL_ARGS="${MODEL_ARGS},max_gen_toks=${MAX_GEN}"
MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LEN}"
MODEL_ARGS="${MODEL_ARGS},num_concurrent=${CONC}"
MODEL_ARGS="${MODEL_ARGS},timeout=7200"

EVAL_CMD=(lm_eval
  --model local-completions
  --model_args "$MODEL_ARGS"
  --apply_chat_template
  --seed "$SEED"
  --gen_kwargs "$GEN_KWARGS"
  --output_path "$OUT_DIR")
# limit=0 means "the whole task"; lm_eval has no sentinel for that, so the
# flag is simply omitted.
[ "$LIMIT" != "0" ] && EVAL_CMD+=(--limit "$LIMIT")
[ "$LOG_SAMPLES" = "1" ] && EVAL_CMD+=(--log_samples)

# One invocation per task, not one for all of them. lm_eval only writes
# results when the whole run finishes, so a full mmlu_pro (12,032 items) that
# fails or gets killed hours in would take a completed gpqa down with it.
# Sequential runs bank each result as it lands; put the cheap task first.
rc=0
T0=$(date +%s)
: > /tmp/lm_eval.log
for task in ${TASKS//,/ }; do
  echo "### ===== task: $task ====="
  t0=$(date +%s)
  "${EVAL_CMD[@]}" --tasks "$task" 2>&1 | tee -a /tmp/lm_eval.log
  trc=${PIPESTATUS[0]}
  echo "### task $task exit=$trc after $(( $(date +%s) - t0 ))s"
  [ "$trc" -ne 0 ] && rc=$trc
done
echo "### lm_eval total $(( $(date +%s) - T0 ))s (worst exit=$rc)"

echo "### --- results ---"
# lm_eval prints a markdown table at the end; re-emit it so the score is in
# the job log even if the PVC copy is never synced.
sed -n '/|*Tasks/,$p' /tmp/lm_eval.log | tail -40 | sed 's/^/### /'
find "$OUT_DIR" -name "results*.json" -newermt "-1 hour" 2>/dev/null | tail -3 | sed 's/^/### wrote /'

echo "### EVAL_RESULT=$([ $rc -eq 0 ] && echo OK || echo FAIL)"
exit $rc
