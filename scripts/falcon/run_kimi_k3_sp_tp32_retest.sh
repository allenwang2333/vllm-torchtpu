#!/usr/bin/env bash
# Multi-host Kimi-K3 sequence-parallel benchmark/profile runner for v7x-32.
# Falcon launches four v7x-8 workers that form one TP32 slice.  One variant
# runs per artifact so Falcon can compare SP1 and SP4 without mixing traces.

set -euo pipefail
ulimit -c 0

readonly VARIANT="${FALCON_K3_SP_VARIANT:?FALCON_K3_SP_VARIANT is required}"
readonly MODE="${FALCON_K3_SP_TP32_MODE:?FALCON_K3_SP_TP32_MODE is required}"
readonly ROUNDS="${FALCON_K3_SP_ROUNDS:-3}"
readonly EXPECTED_SOURCE_COMMIT="${EXPECTED_TORCHTPU_COMMIT:?EXPECTED_TORCHTPU_COMMIT is required}"

readonly MODEL_NAME="moonshotai/Kimi-K3"
readonly MODEL_PATH="/models/Kimi-K3-a590ce09"
readonly MODEL_REVISION="a590ce090cb049c93a33dfe8c208ec652aa20503"
readonly VLLM_COMMIT="4bdc8a788d2e2ce9165d552b3d4d8b72604626bf"
readonly TORCH_TPU_VERSION="0.1.1.dev20260813160135"
readonly TORCH_TPU_SHA256="e1e326fd4efbc85e965618e6a97a653eb256f00f817d2955f1fffaa59d0e525a"
readonly WHEEL_PATH="/wheels/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl"

readonly RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
readonly WORLD="${FALCON_WORLD_SIZE:-4}"
readonly POD_HOST="$(hostname)"
readonly JOB_BASE="${POD_HOST%-*}"
readonly MASTER_ADDR="${JOB_BASE}-0.${JOB_BASE}"
readonly ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/kimi-k3-sp-tp32-artifact}"
readonly OUT="$ROOT/rank-$RANK"
readonly VARIANT_OUT="$OUT/$VARIANT"
readonly BARRIER_ROOT="$ROOT/barriers"
readonly CACHE_ROOT="/tmp/tpu_logs_k3_sp_tp32_${VARIANT}_${MODE}_${EXPECTED_SOURCE_COMMIT:0:8}"
readonly VENV="$CACHE_ROOT/venv"
readonly PROFILE_DIR="$CACHE_ROOT/profile"

case "$VARIANT" in
  sp1) readonly SP_SIZE=1 ;;
  sp4) readonly SP_SIZE=4 ;;
  *) printf 'Unsupported FALCON_K3_SP_VARIANT=%s\n' "$VARIANT" >&2; exit 1 ;;
esac
case "$MODE" in
  benchmark|profile) ;;
  *) printf 'Unsupported FALCON_K3_SP_TP32_MODE=%s\n' "$MODE" >&2; exit 1 ;;
esac
case "$ROUNDS" in
  ''|*[!0-9]*) printf 'FALCON_K3_SP_ROUNDS must be an integer\n' >&2; exit 1 ;;
esac
test "$ROUNDS" -ge 1
if test "$WORLD" != 4 || test "$RANK" -lt 0 || test "$RANK" -gt 3; then
  printf 'UNEXPECTED_FALCON_TOPOLOGY rank=%s world=%s\n' "$RANK" "$WORLD" >&2
  exit 1
fi

export TMPDIR="$CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip-cache"
export UV_CACHE_DIR="$CACHE_ROOT/uv-cache"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax-cache"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm-cache"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/tier3"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_TARGET_DEVICE=empty
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export TPU_MULTIHOST_BACKEND=mp
export TPU_TOKEN_BUCKET_EXTRA=4,8,16,32,64,128
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_MOE_SKIP_PADDED_TOKENS=1
# Pin the latest-main KDA backend for both SP1 and SP4.
export VLLM_TORCHTPU_KDA_FUSED=1
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=chunked

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" \
  "$HF_HOME" \
  "$PROFILE_DIR" \
  "$OUT/profiling/xprof" \
  "$OUT/benchmark" \
  "$VARIANT_OUT/correctness" \
  "$VARIANT_OUT/benchmarks" \
  "$BARRIER_ROOT"

exec > >(tee -a "$OUT/task.log") 2>&1
printf 'FALCON_MULTIHOST rank=%s world=%s pod=%s master=%s variant=%s mode=%s\n' \
  "$RANK" "$WORLD" "$POD_HOST" "$MASTER_ADDR" "$VARIANT" "$MODE"
getent hosts "$MASTER_ADDR"

WHEEL_TO_INSTALL="$WHEEL_PATH"
if test -s "$WHEEL_TO_INSTALL"; then
  printf '%s  %s\n' "$TORCH_TPU_SHA256" "$WHEEL_TO_INSTALL" \
    | sha256sum -c -
else
  WHEEL_TO_INSTALL=""
  printf 'Mounted torch-tpu wheel unavailable; using verified registry fallback\n'
fi
test "$(git -C /workspace/vllm-torchtpu rev-parse HEAD)" = \
  "$EXPECTED_SOURCE_COMMIT"
test "$(git -C /workspace/vllm rev-parse HEAD)" = "$VLLM_COMMIT"

export MODEL_PATH MODEL_REVISION
python3 - <<'PY' | tee "$OUT/model-validation.json"
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_PATH"])
expected_revision = os.environ["MODEL_REVISION"]
required = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"missing required model files: {missing}")
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(root.glob("model-*-of-000096.safetensors"))
referenced = sorted(set(index["weight_map"].values()))
missing_referenced = [name for name in referenced if not (root / name).is_file()]
record = {
    "model_path": str(root),
    "model_revision": expected_revision,
    "architectures": config.get("architectures"),
    "safetensors_shards": len(shards),
    "index_referenced_shards": len(referenced),
    "missing_index_referenced_shards": missing_referenced,
}
print(json.dumps(record, indent=2, sort_keys=True))
if record["architectures"] != ["KimiK3ForConditionalGeneration"]:
    raise SystemExit(f"unexpected architecture: {record['architectures']}")
if len(shards) != 96 or len(referenced) != 96 or missing_referenced:
    raise SystemExit(f"incomplete model shards: {record}")
print("KIMI_K3_OFFICIAL_WEIGHT_VALIDATION_OK")
PY

python3 -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip 'setuptools>=77,<81' wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu'
if test -z "$WHEEL_TO_INSTALL"; then
  readonly REGISTRY_WHEEL_DIR="$CACHE_ROOT/registry-wheel"
  mkdir -p "$REGISTRY_WHEEL_DIR"
  python -m pip install keyring keyrings.google-artifactregistry-auth
  PIP_INDEX_URL='https://oauth2accesstoken@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/' \
    python -m pip download --no-deps --pre \
      "torch-tpu==$TORCH_TPU_VERSION" -d "$REGISTRY_WHEEL_DIR"
  WHEEL_TO_INSTALL=$(find "$REGISTRY_WHEEL_DIR" -maxdepth 1 -type f \
    -name 'torch_tpu-*.whl' -print -quit)
  test -n "$WHEEL_TO_INSTALL"
  printf '%s  %s\n' "$TORCH_TPU_SHA256" "$WHEEL_TO_INSTALL" \
    | sha256sum -c -
fi
python -m pip install "$WHEEL_TO_INSTALL"
python -m pip install --pre 'runai-model-streamer[gcs]'

python - /workspace/vllm-torchtpu/pyproject.toml <<'PY'
import pathlib
import subprocess
import sys
import tomllib

project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
skip_prefixes = (
    "vllm ",
    "torch==",
    "torchvision==",
    "torch-tpu==",
    "tpu-raiden-torch==",
)
dependencies = [
    dependency for dependency in project["project"]["dependencies"]
    if not dependency.startswith(skip_prefixes)
]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--pre", *dependencies]
)
PY

VLLM_TARGET_DEVICE=empty python -m pip install -e /workspace/vllm
python -m pip install --no-deps -e /workspace/vllm-torchtpu

python - <<'PY' | tee "$OUT/runtime-versions.txt"
import importlib.metadata as md
import torch
import torch_tpu

for package in (
    "torch", "torchvision", "torch-tpu", "vllm", "vllm-torchtpu",
    "jax", "jaxlib", "libtpu", "runai-model-streamer",
):
    try:
        print(f"{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}=MISSING")
x = torch.ones((8, 8), device="tpu")
value = (x @ x).cpu()[0, 0].item()
assert value == 8.0
print(f"tpu_matmul={value}")
print("TORCH_TPU_DEVICE_SMOKE_OK")
PY

barrier() {
  local name="$1"
  local timeout_seconds="${2:-1800}"
  local marker="$BARRIER_ROOT/$name.$RANK"
  touch "$marker"
  local waited=0
  while test "$waited" -lt "$timeout_seconds"; do
    local count
    count=$(find "$BARRIER_ROOT" -maxdepth 1 -type f -name "$name.*" | wc -l)
    if test "$count" -eq "$WORLD"; then
      printf 'BARRIER_OK name=%s rank=%s waited=%s\n' "$name" "$RANK" "$waited"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  printf 'BARRIER_TIMEOUT name=%s rank=%s\n' "$name" "$RANK" >&2
  return 1
}

SERVER_PID=0
SERVER_LOG="$VARIANT_OUT/server.log"

cleanup_server() {
  set +e
  if test "$SERVER_PID" -gt 0; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=0
  rm -f /tmp/libtpu_lockfile* 2>/dev/null || true
  set -e
}

cleanup_all() {
  cleanup_server
  tail -500 "$SERVER_LOG" >"$VARIANT_OUT/last-server-lines.log" \
    2>/dev/null || true
}
trap cleanup_all EXIT

serve_args=(
  "$MODEL_PATH"
  --served-model-name "$MODEL_NAME"
  --tensor-parallel-size 32
  --nnodes 4
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port 29500
  --enable-expert-parallel
  --enable-ep-weight-filter
  --max-model-len 32768
  --max-num-seqs 4
  --max-num-batched-tokens 4096
  --max-num-scheduled-tokens 4096
  --long-prefill-token-threshold 4096
  --enable-chunked-prefill
  --block-size 256
  --gpu-memory-utilization 0.81
  --no-enable-prefix-caching
  --language-model-only
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --trust-remote-code
  --load-format runai_streamer
  --model-loader-extra-config '{"concurrency":16,"memory_limit":8589934592}'
  --enable-logging-iteration-details
  '--compilation-config={"compile_sizes":[4,4096]}'
  --additional-config "{\"sequence_parallel_size\":$SP_SIZE}"
)
if test "$MODE" = profile; then
  serve_args+=(
    --profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$PROFILE_DIR"
    --profiler-config.ignore_frontend true
  )
fi
if test "$RANK" = 0; then
  serve_args+=(--host 0.0.0.0 --port 8000)
else
  serve_args+=(--headless)
fi

vllm serve "${serve_args[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 1440); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'VLLM_PROCESS_DIED rank=%s variant=%s mode=%s\n' \
      "$RANK" "$VARIANT" "$MODE" >&2
    tail -500 "$SERVER_LOG" || true
    exit 1
  fi
  if curl --connect-timeout 2 --max-time 5 -sf \
    "http://$MASTER_ADDR:8000/health" >/dev/null; then
    printf 'KIMI_K3_SP_TP32_SERVER_READY rank=%s variant=%s mode=%s elapsed_seconds=%s\n' \
      "$RANK" "$VARIANT" "$MODE" "$((i * 10))"
    ready=1
    break
  fi
  sleep 10
done
test "$ready" = 1

grep -F 'using the fused conv1d + GDN v3 path' "$SERVER_LOG" \
  >"$VARIANT_OUT/kda-backend.log"
if test "$SP_SIZE" = 4; then
  grep -F 'Sequence parallelism enabled' "$SERVER_LOG" \
    >"$VARIANT_OUT/sequence-parallel.log"
else
  if grep -Fq 'Sequence parallelism enabled' "$SERVER_LOG"; then
    printf 'Sequence parallelism unexpectedly enabled for SP1 rank=%s\n' \
      "$RANK" >&2
    exit 1
  fi
fi

barrier server-ready 1800

if test "$RANK" = 0; then
  python - "$ROOT/manifest.json" "$VARIANT" "$MODE" \
    "$EXPECTED_SOURCE_COMMIT" "$SP_SIZE" <<'PY'
import json
import pathlib
import sys

path, variant, mode, commit, sp_size = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "workflow": "operator-optimization",
    "operator_family": "kimi_k3_attention_residual_sp",
    "operator_name": "vllm_kimi_k3_sequence_parallel",
    "model": "moonshotai/Kimi-K3",
    "model_revision": "a590ce090cb049c93a33dfe8c208ec652aa20503",
    "source_commit": commit,
    "vllm_commit": "4bdc8a788d2e2ce9165d552b3d4d8b72604626bf",
    "torch_tpu": "0.1.1.dev20260813160135",
    "hardware": {
        "name": "v7x-32",
        "device_type": "v7x",
        "device_count": 8,
        "device_topo": "2x2x4",
        "replica": 4,
    },
    "dimensions": {
        "variant": variant,
        "mode": mode,
        "tp_size": 32,
        "sequence_parallel_size": int(sp_size),
        "isl": 4096,
        "osl": 1,
        "concurrency": 1,
        "kda_backend": "fused-conv1d-gdn-v3",
    },
}
pathlib.Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
PY
  env | sort | grep -E \
    '^(VLLM_TORCHTPU_|FALCON_K3_SP_|EXPECTED_TORCHTPU_COMMIT)=' \
    >"$ROOT/profiling-env.txt" || true
fi

fixed_request() {
  local label="$1"
  local output="$2"
  python - "$VARIANT" "$MODEL_NAME" "$MASTER_ADDR" "$label" "$output" <<'PY'
import json
import pathlib
import sys

import requests

variant, model, host, label, output_path = sys.argv[1:]
prompt = [1 + ((index * 17 + 23) % 10000) for index in range(4096)]
response = requests.post(
    f"http://{host}:8000/v1/completions",
    json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 42,
        "logprobs": 20,
        "ignore_eos": True,
    },
    timeout=3600,
)
response.raise_for_status()
choice = response.json()["choices"][0]
logprobs = choice["logprobs"]
record = {
    "variant": variant,
    "label": label,
    "tokens": 4096,
    "sampled_text": choice["text"],
    "sampled_token": logprobs["tokens"][0],
    "sampled_logprob": logprobs["token_logprobs"][0],
    "top_logprobs": logprobs["top_logprobs"][0],
}
pathlib.Path(output_path).write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n"
)
print("KIMI_K3_SP_TP32_CORRECTNESS=" +
      json.dumps(record, ensure_ascii=False, sort_keys=True))
PY
}

run_benchmark_round() {
  local round="$1"
  local round_dir="$VARIANT_OUT/benchmarks/isl-4096/round-$round"
  local result="$round_dir/benchmark.json"
  mkdir -p "$round_dir"
  vllm bench serve \
    --backend vllm \
    --model "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --trust-remote-code \
    --base-url "http://$MASTER_ADDR:8000" \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 1 \
    --random-range-ratio 0 \
    --num-prompts 8 \
    --num-warmups 2 \
    --max-concurrency 1 \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-filename "$result" \
    --temperature 0 \
    --seed 42 \
    --metadata variant="$VARIANT" logical_isl=4096 logical_osl=1 \
      round="$round" \
    2>&1 | tee "$round_dir/benchmark.log"
  python - "$round" "$result" "$OUT/benchmark/metrics.jsonl" <<'PY'
import json
import pathlib
import sys

round_index, result_path, metrics_path = sys.argv[1:]
data = json.loads(pathlib.Path(result_path).read_text())
if data.get("completed") != 8 or data.get("failed", 0) != 0:
    raise SystemExit(f"incomplete benchmark: {data}")
if set(data.get("input_lens", [])) != {4096}:
    raise SystemExit(f"unexpected input lengths: {data.get('input_lens')}")
if set(data.get("output_lens", [])) != {1}:
    raise SystemExit(f"unexpected output lengths: {data.get('output_lens')}")
row = {
    "round": int(round_index),
    "tokens": 4096,
    "seq_len": 4096,
    "batch_size": 1,
    "concurrency": 1,
    "latency_ms": data["mean_ttft_ms"],
}
with pathlib.Path(metrics_path).open("a") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

if test "$RANK" = 0; then
  fixed_request "$MODE" "$VARIANT_OUT/correctness/isl-4096.json"
  if test "$MODE" = benchmark; then
    for round in $(seq 1 "$ROUNDS"); do
      run_benchmark_round "$round"
    done
  else
    curl --fail-with-body -sS -X POST \
      "http://$MASTER_ADDR:8000/start_profile" \
      | tee "$VARIANT_OUT/start-profile.json"
    fixed_request profile "$VARIANT_OUT/correctness/profile-isl-4096.json"
    curl --fail-with-body -sS -X POST \
      "http://$MASTER_ADDR:8000/stop_profile" \
      | tee "$VARIANT_OUT/stop-profile.json"
  fi
fi

barrier workload-finished 3600

if test "$MODE" = profile; then
  local_count=0
  for _ in $(seq 1 180); do
    local_count=$(find "$PROFILE_DIR" -type f -name '*.xplane.pb' -size +0c | wc -l)
    if test "$local_count" -ge 8; then
      break
    fi
    sleep 5
  done
  if test "$local_count" -lt 8; then
    printf 'PROFILE_LOCAL_FILES_MISSING variant=%s rank=%s count=%s\n' \
      "$VARIANT" "$RANK" "$local_count" >&2
    find "$PROFILE_DIR" -maxdepth 7 -print | sort || true
    exit 1
  fi
  find "$PROFILE_DIR" -type f -size +0c \
    -exec cp --parents '{}' "$OUT/profiling/xprof"/ \;
  find "$PROFILE_DIR" -type f -size +0c -print | sort \
    >"$OUT/profile-files.txt"
  printf 'KIMI_K3_SP_TP32_PROFILE_LOCAL_OK variant=%s rank=%s xplanes=%s\n' \
    "$VARIANT" "$RANK" "$local_count"
fi

barrier artifact-finished 1800
cleanup_server

if test "$RANK" = 0 && test "$MODE" = benchmark; then
  python - "$OUT/benchmark/metrics.jsonl" "$VARIANT_OUT/summary.json" \
    "$VARIANT" <<'PY'
import json
import pathlib
import statistics
import sys

metrics_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
variant = sys.argv[3]
rows = [json.loads(line) for line in metrics_path.read_text().splitlines()
        if line.strip()]
values = [row["latency_ms"] for row in rows]
summary = {
    "variant": variant,
    "measurements": rows,
    "mean_ttft_ms": statistics.mean(values),
    "median_round_ttft_ms": statistics.median(values),
    "min_ttft_ms": min(values),
    "max_ttft_ms": max(values),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print("KIMI_K3_SP_TP32_SUMMARY=" + json.dumps(summary, sort_keys=True))
PY
fi

if test "$RANK" = 0; then
  test -s "$ROOT/manifest.json"
  if test "$MODE" = benchmark; then
    test -s "$OUT/benchmark/metrics.jsonl"
    test -s "$VARIANT_OUT/summary.json"
  fi
  printf 'KIMI_K3_SP_TP32_OK variant=%s mode=%s\n' "$VARIANT" "$MODE" \
    | tee "$VARIANT_OUT/success.marker"
fi

barrier complete 600
