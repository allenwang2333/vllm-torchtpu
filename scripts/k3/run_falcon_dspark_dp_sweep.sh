#!/usr/bin/env bash
# Falcon-only Kimi-K3 DSpark throughput sweep with TP32 per DP replica.
set -euo pipefail
ulimit -c 0

: "${K3_DP_SIZE:?set K3_DP_SIZE}"
: "${K3_TP_SIZE:?set K3_TP_SIZE}"
: "${K3_COORD_PREFIX:?set K3_COORD_PREFIX}"
: "${K3_HOST_BOUNDS:?set K3_HOST_BOUNDS}"
: "${K3_TORCH_TPU_TOPOLOGY:?set K3_TORCH_TPU_TOPOLOGY}"
EXPECTED_WORLD=$((K3_DP_SIZE * K3_TP_SIZE / 8))
if (( K3_TP_SIZE != 32 || K3_DP_SIZE * K3_TP_SIZE % 8 != 0 )); then
  echo "expected TP32 and a whole number of v7x-8 Falcon workers" >&2
  exit 2
fi

export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jax-cache
export VLLM_CACHE_ROOT=/tmp/tpu_logs/vllm-cache
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/tmp/tpu_logs/torch-tpu-tier3
export HF_HOME=/tmp/tpu_logs/huggingface
export HF_HUB_CACHE=/tmp/tpu_logs/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_TARGET_DEVICE=empty
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS="$K3_HOST_BOUNDS"
export TORCH_TPU_TOPOLOGY="$K3_TORCH_TPU_TOPOLOGY"
export JAX_PLATFORMS=""
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_TORCHTPU_KDA_FUSED=0
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=mega
export TPU_TOKEN_BUCKET_EXTRA=1,2,4,6,8,12,16,24,32,48,96,192,512
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_MOE_SKIP_PADDED_TOKENS=1
unset DP_SCHED_ENABLED USE_PHASED_PROFILER || true

MODEL_ID=moonshotai/Kimi-K3
MODEL_PATH=/models/Kimi-K3-a590ce09
MODEL_REVISION=a590ce090cb049c93a33dfe8c208ec652aa20503
SPEC_MODEL_PATH=/models/Inferact/Kimi-K3-DSpark-Block5
WHEEL_PATH=/wheels/rank-0/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl
VLLM_TORCHTPU_COMMIT="$(git -C /workspace/vllm-torchtpu rev-parse HEAD)"
VLLM_COMMIT=4bdc8a788d2e2ce9165d552b3d4d8b72604626bf
RUNBOOK_COMMIT=de2c67b2a9ad20746d78340c881f98a87e51417c
RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
WORLD="${FALCON_WORLD_SIZE:-4}"
POD_HOST="$(hostname)"
JOB_BASE="${POD_HOST%-*}"
MASTER_ADDR="${JOB_BASE}-0.${JOB_BASE}"
OUT_ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
OUT="$OUT_ROOT/rank-$RANK"
VENV=/tmp/tpu_logs/venv
SERVER_LOG="/tmp/tpu_logs/server-dspark-dp${K3_DP_SIZE}-rank${RANK}.log"
DONE_MARKER="/coord/${K3_COORD_PREFIX}_DONE"
SERVER_PID=0
TAIL_PID=0

if [[ "$WORLD" != "$EXPECTED_WORLD" || "$RANK" -lt 0 || "$RANK" -ge "$WORLD" ]]; then
  echo "unexpected Falcon topology rank=$RANK world=$WORLD" >&2
  exit 1
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR" "$VLLM_CACHE_ROOT" \
  "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" "$HF_HOME" "$OUT"
if [[ "$RANK" == 0 ]]; then
  rm -f "$DONE_MARKER"
fi
exec > >(tee -a "$OUT/task.log") 2>&1

for _ in $(seq 1 60); do
  [[ -s "$WHEEL_PATH" ]] && break
  sleep 5
done
test -s "$WHEEL_PATH"
test "$(git -C /workspace/vllm rev-parse HEAD)" = "$VLLM_COMMIT"
test "$(git -C /workspace/k3-launch-scripts rev-parse HEAD)" = "$RUNBOOK_COMMIT"
test -s "$MODEL_PATH/config.json"
test -s "$SPEC_MODEL_PATH/config.json"

HUB_MODEL_ROOT="$HF_HUB_CACHE/models--moonshotai--Kimi-K3"
mkdir -p "$HUB_MODEL_ROOT/refs" "$HUB_MODEL_ROOT/snapshots"
ln -sfn "$MODEL_PATH" "$HUB_MODEL_ROOT/snapshots/$MODEL_REVISION"
printf '%s\n' "$MODEL_REVISION" > "$HUB_MODEL_ROOT/refs/main"

python3 -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip 'setuptools>=77,<81' wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu'
python -m pip install "$WHEEL_PATH"
python - /workspace/vllm-torchtpu/pyproject.toml <<'PY'
import pathlib
import subprocess
import sys
import tomllib

project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
skip = ("vllm ", "torch==", "torchvision==", "torch-tpu==", "tpu-raiden-torch==")
deps = [d for d in project["project"]["dependencies"] if not d.startswith(skip)]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--pre", *deps])
PY
VLLM_TARGET_DEVICE=empty python -m pip install -e /workspace/vllm
python -m pip install --no-deps -e /workspace/vllm-torchtpu

python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("transformers.models.deepseek_v2.configuration_deepseek_v2")
assert spec is not None and spec.origin is not None
path = Path(spec.origin)
source = path.read_text()
old = "        if self.hidden_size % self.num_attention_heads != 0:\n"
new = (
    "        if (\n"
    "            self.model_type != \"k3_dspark\"\n"
    "            and self.hidden_size % self.num_attention_heads != 0\n"
    "        ):\n"
)
assert source.count(old) == 1
path.write_text(source.replace(old, new))
PY

stop_server() {
  set +e
  if (( TAIL_PID > 0 )); then
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
  fi
  if (( SERVER_PID > 0 )); then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  cp "$SERVER_LOG" "$OUT/$(basename "$SERVER_LOG")" 2>/dev/null || true
  set -e
}
trap stop_server EXIT

export VLLM_TARGET_DEVICE=tpu
serve_args=(
  "$MODEL_PATH"
  --served-model-name "$MODEL_ID"
  --tensor-parallel-size "$K3_TP_SIZE"
  --data-parallel-size "$K3_DP_SIZE"
  --data-parallel-backend mp
  --enable-expert-parallel
  --enable-ep-weight-filter
  --nnodes "$WORLD"
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port 29500
  --quantization compressed-tensors
  --safetensors-load-strategy lazy
  --kv-cache-dtype fp8
  --attention-backend CUSTOM
  --async-scheduling
  --max-model-len 32768
  --max-num-seqs 16
  --max-num-batched-tokens 512
  --block-size 16
  --gpu-memory-utilization 0.82
  --no-enable-prefix-caching
  --language-model-only
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --trust-remote-code
  --tokenizer "$MODEL_PATH"
  --tokenizer-mode kimi_k3
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$SPEC_MODEL_PATH\",\"num_speculative_tokens\":5,\"draft_tensor_parallel_size\":$K3_TP_SIZE,\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":4.0}"
  '--compilation-config={"compile_sizes":[1,2,4,6,8,12,16,24,32,48,96,192,512]}'
)

if [[ "$RANK" == 0 ]]; then
  vllm serve "${serve_args[@]}" --host 0.0.0.0 --port 8000 >"$SERVER_LOG" 2>&1 &
else
  vllm serve "${serve_args[@]}" --headless >"$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!
tail -n +1 -F "$SERVER_LOG" &
TAIL_PID=$!

ready=0
for i in $(seq 1 1440); do
  kill -0 "$SERVER_PID" 2>/dev/null || exit 1
  if curl --connect-timeout 2 --max-time 5 -sf "http://$MASTER_ADDR:8000/health" >/dev/null; then
    echo "KIMI_K3_DSPARK_DP_READY dp=$K3_DP_SIZE tp=$K3_TP_SIZE elapsed_seconds=$((i * 10))"
    ready=1
    break
  fi
  sleep 10
done
test "$ready" = 1

if [[ "$RANK" == 0 ]]; then
  bench_script=/workspace/k3-launch-scripts/scripts/k3/bench_isl1_osl1k.sh
  for concurrency in 4 8 16; do
    point_dir="$OUT/benchmark/c_${concurrency}"
    num_prompts=$((4 * concurrency))
    num_warmups=$((2 * concurrency))
    mkdir -p "$point_dir"
    curl --fail --silent --show-error "http://$MASTER_ADDR:8000/metrics" > "$point_dir/metrics-before.prom"
    OUTPUT_DIR="$point_dir" MODEL="$MODEL_ID" MAX_CONCURRENCY="$concurrency" \
      NUM_PROMPTS="$num_prompts" NUM_WARMUPS="$num_warmups" SEED=42 \
      VLLM="$VENV/bin/vllm" KIMI_K3_HF_HOME="$HF_HOME" \
      KIMI_K3_HF_HUB_CACHE="$HF_HUB_CACHE" \
      "$bench_script" "$MASTER_ADDR" 2>&1 | tee "$point_dir/bench.log"
    curl --fail --silent --show-error "http://$MASTER_ADDR:8000/metrics" > "$point_dir/metrics-after.prom"
  done
  grep -F "No tuned parameters found for the given tuning key" "$SERVER_LOG" \
    > "$OUT/mla-tuning-fallbacks.log" || true
  grep -E "TuningKey\(case='(batched_decode|mixed)'" "$SERVER_LOG" \
    > "$OUT/mla-tuning-keys.log" || true
  python - "$OUT" "$K3_DP_SIZE" "$K3_TP_SIZE" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
dp, tp = map(int, sys.argv[2:])

def counters(path):
    values = {}
    wanted = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    }
    for line in path.read_text().splitlines():
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([^ ]+)$", line)
        if match is None:
            continue
        name, value = match.groups()
        if name.endswith("_total"):
            name = name[:-6]
        if name in wanted:
            values[name] = values.get(name, 0.0) + float(value)
    return values

points = []
for point_dir in sorted((root / "benchmark").glob("c_*"), key=lambda p: int(p.name[2:])):
    raw_path = next(point_dir.glob("kimi-k3_isl1_osl1000_concurrency*.json"))
    raw = json.loads(raw_path.read_text())
    before = counters(point_dir / "metrics-before.prom")
    after = counters(point_dir / "metrics-after.prom")
    steps = after.get("vllm:spec_decode_num_drafts", 0.0) - before.get(
        "vllm:spec_decode_num_drafts", 0.0
    )
    accepted = after.get("vllm:spec_decode_num_accepted_tokens", 0.0) - before.get(
        "vllm:spec_decode_num_accepted_tokens", 0.0
    )
    acceptance_length = 1.0 + accepted / steps if steps else 0.0
    if not 3.95 <= acceptance_length <= 4.05:
        raise RuntimeError(
            f"C={point_dir.name[2:]} acceptance={acceptance_length}, expected 4"
        )
    points.append({
        "concurrency": int(point_dir.name[2:]),
        "result_file": raw_path.name,
        "draft_steps": steps,
        "measured_acceptance_length": acceptance_length,
        "raw": raw,
    })
summary = {
    "model": "moonshotai/Kimi-K3",
    "spec_model": "Inferact/Kimi-K3-DSpark-Block5",
    "hardware": f"v7x-{dp * tp}",
    "data_parallel_size": dp,
    "tensor_parallel_size": tp,
    "expert_parallel_size": dp * tp,
    "draft_tensor_parallel_size": tp,
    "input_tokens_per_request": 1,
    "output_tokens_per_request": 1000,
    "concurrency_points": [4, 8, 16],
    "num_prompts_rule": "4 * concurrency",
    "num_warmups_rule": "2 * concurrency",
    "num_speculative_tokens": 5,
    "target_verify_tokens_per_request": 6,
    "synthetic_acceptance_length": 4.0,
    "points": points,
}
(root / "benchmark-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("KIMI_K3_DSPARK_DP_SWEEP_PASS " + json.dumps(summary, sort_keys=True))
PY
  touch "$DONE_MARKER"
fi

for _ in $(seq 1 4320); do
  [[ -f "$DONE_MARKER" ]] && break
  kill -0 "$SERVER_PID" 2>/dev/null || exit 1
  sleep 10
done
test -f "$DONE_MARKER"
sleep 10
