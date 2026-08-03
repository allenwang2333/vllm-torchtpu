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
# Price the MLA KV-cache quantization schemes on real activations.
#
# Runs one prefill through the real vLLM engine, captures the unquantized
# `kv_c_normed` / `k_pe` that `PallasMLAttentionBackendImpl.forward` is about
# to fp8-quantize, and reports what each cache scheme costs -- see
# kv_precision_probe.py and kv_precision_analysis.py.
#
#   cdk job create ttpu-dev \
#     SCRIPT=../../.claude/worktrees/glm-kv-probe/scripts/cdk/kv_precision_probe \
#     PROBE_MODEL=deepseek-ai/DeepSeek-V2-Lite MEM=64Gi
set -uo pipefail

MODEL="${PROBE_MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
export PROBE_MODEL="$MODEL"

# Knobs set HERE, not via `cdk job create KEY=VALUE`. cdk renders KEY=VALUE
# into the recipe template, so a key the recipe does not declare in its `env:`
# block never reaches the container and is silently ignored -- PROBE_DRY_RUN=1
# was dropped exactly that way on the first run. Only the knobs ttpu-dev
# already declares (PROBE_MODEL, MAX_MODEL_LEN, PROMPT_LEN, TP_SIZE,
# GPU_MEM_UTIL, ...) can come from the command line. Everything below is edited
# in this file, which is live on NFS, so it costs a re-submit and no rebuild.
export PROBE_DRY_RUN="${PROBE_DRY_RUN:-0}"
export CAPTURE_LAYERS="${CAPTURE_LAYERS:-1,13,25}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
export MAX_Q_ROWS="${MAX_Q_ROWS:-64}"
export ANALYSIS_TOPK="${ANALYSIS_TOPK:-2048}"

echo "### kv_precision_probe: model=$MODEL kv_dtype=$KV_CACHE_DTYPE"
echo "### layers=$CAPTURE_LAYERS prompt_len=${PROMPT_LEN:-2048} dry_run=$PROBE_DRY_RUN"

# Same workarounds the other TPU scripts use for current TorchTPU builds.
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

# The capture hook patches PallasMLAttentionBackendImpl in *this* process. With
# V1 multiprocessing on, the model forward runs in a separate EngineCore
# process that never sees the patch, and the probe would exit with an empty
# capture. Keeping everything in-process is also why this runs at TP=1: a
# multi-rank run is multi-process by construction.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Fetch once, up front, single process. Every TP worker would otherwise call
# snapshot_download on the same repo concurrently and HF starts dropping
# connections mid-file. With the weights already on the SSD-backed HF_HOME the
# workers just read them.
if [ "${MODEL#/}" = "$MODEL" ]; then   # not an absolute path -> HF repo id
  echo "### pre-fetching $MODEL into ${HF_HOME:-~/.cache/huggingface}"
  python3 - <<'FETCHEOF'
import os, time
from huggingface_hub import snapshot_download
repo = os.environ["PROBE_MODEL"]
for attempt in range(1, 11):
    try:
        p = snapshot_download(repo, max_workers=8)
        print("### fetched to", p, flush=True)
        break
    except Exception as e:
        print(f"### fetch attempt {attempt} failed: {type(e).__name__}: "
              f"{str(e)[:160]}", flush=True)
        time.sleep(min(30, 5 * attempt))
else:
    raise SystemExit("### FATAL: could not fetch weights")
FETCHEOF
  rc=$?
  [ $rc -eq 0 ] || { echo "### FETCH_EXIT=$rc"; exit $rc; }
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The worktree's src must win over the main checkout's, which the recipe put
# on PYTHONPATH ahead of us.
export PYTHONPATH="${HERE}/../../src:${PYTHONPATH:-}"
echo "### PYTHONPATH=$PYTHONPATH"

python3 "${HERE}/kv_precision_probe.py"
rc=$?
echo "### PROBE_EXIT=$rc"
exit $rc
