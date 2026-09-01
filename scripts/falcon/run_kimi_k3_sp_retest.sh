#!/usr/bin/env bash
# Paired Kimi-K3 sequence-parallel correctness and prefill benchmark runner.
# Falcon supplies the pinned sources, model, torch-tpu wheel, TPU slice, and
# ARTIFACT_LOCAL_DIR.  The same script is used by the main baseline and the
# rebased PR candidate so only the source revision and SP setting differ.

set -euo pipefail
ulimit -c 0

readonly VARIANT="${FALCON_K3_SP_VARIANT:?FALCON_K3_SP_VARIANT is required}"
readonly HAS_SP_FEATURE="${FALCON_K3_SP_HAS_FEATURE:-0}"
readonly RUN_TESTS="${FALCON_K3_SP_RUN_TESTS:-0}"
readonly E2E_SMOKE="${FALCON_K3_SP_E2E_SMOKE:-0}"
readonly ISL_CSV="${FALCON_K3_SP_ISLS:-4096}"
readonly COMPILE_SIZE_CSV="${FALCON_K3_SP_COMPILE_SIZES:-4,4096}"
readonly ROUNDS="${FALCON_K3_SP_ROUNDS:-3}"
readonly EXPECTED_SOURCE_COMMIT="${EXPECTED_TORCHTPU_COMMIT:?EXPECTED_TORCHTPU_COMMIT is required}"

readonly MODEL_NAME="mgoin/Kimi-K3-pruned75"
readonly MODEL_PATH="/models/Kimi-K3-pruned75"
readonly MODEL_REVISION="3c8e22fdac0c14409eebd48ba78c10a940f5267a"
readonly VLLM_COMMIT="4bdc8a788d2e2ce9165d552b3d4d8b72604626bf"
readonly TORCH_TPU_VERSION="0.1.1.dev20260813160135"
readonly TORCH_TPU_SHA256="e1e326fd4efbc85e965618e6a97a653eb256f00f817d2955f1fffaa59d0e525a"
readonly WHEEL_PATH="/wheels/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl"

readonly ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/kimi-k3-sp-retest-artifact}"
readonly OUT="$ROOT/rank-0"
readonly VARIANT_OUT="$OUT/$VARIANT"
readonly CACHE_ROOT="/tmp/tpu_logs_k3_sp_${VARIANT}_${EXPECTED_SOURCE_COMMIT:0:8}"
readonly VENV="$CACHE_ROOT/venv"

case "$VARIANT" in
  main)
    readonly SP_SIZE=1
    test "$HAS_SP_FEATURE" = 0
    ;;
  sp1)
    readonly SP_SIZE=1
    test "$HAS_SP_FEATURE" = 1
    ;;
  sp4)
    readonly SP_SIZE=4
    test "$HAS_SP_FEATURE" = 1
    ;;
  *)
    printf 'Unsupported FALCON_K3_SP_VARIANT=%s\n' "$VARIANT" >&2
    exit 1
    ;;
esac

case "$ROUNDS" in
  ''|*[!0-9]*) printf 'FALCON_K3_SP_ROUNDS must be an integer\n' >&2; exit 1 ;;
esac
test "$ROUNDS" -ge 1

IFS=',' read -r -a ISLS <<<"$ISL_CSV"
IFS=',' read -r -a COMPILE_SIZES <<<"$COMPILE_SIZE_CSV"
test "${#ISLS[@]}" -ge 1
test "${#COMPILE_SIZES[@]}" -ge 1

MAX_ISL=0
for isl in "${ISLS[@]}"; do
  case "$isl" in
    ''|*[!0-9]*) printf 'Invalid ISL %s\n' "$isl" >&2; exit 1 ;;
  esac
  test "$isl" -gt 0
  if test "$SP_SIZE" = 4; then
    test "$((isl % 4))" -eq 0
  fi
  if test "$isl" -gt "$MAX_ISL"; then
    MAX_ISL="$isl"
  fi
done
for size in "${COMPILE_SIZES[@]}"; do
  case "$size" in
    ''|*[!0-9]*) printf 'Invalid compile size %s\n' "$size" >&2; exit 1 ;;
  esac
  if test "$SP_SIZE" = 4; then
    test "$((size % 4))" -eq 0
  fi
done

export TMPDIR="$CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip-cache"
export UV_CACHE_DIR="$CACHE_ROOT/uv-cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export PYTHONUNBUFFERED=1
export VLLM_TARGET_DEVICE=empty
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0
# Pin the current-main default explicitly so all variants use the same KDA
# implementation.  The prefill selector is ignored while the fused path is on.
export VLLM_TORCHTPU_KDA_FUSED=1
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=chunked

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$ROOT/profiling" \
  "$OUT/benchmark" \
  "$VARIANT_OUT/correctness" \
  "$VARIANT_OUT/benchmarks"

test -f "$MODEL_PATH/config.json"
test -s "$WHEEL_PATH"
printf '%s  %s\n' "$TORCH_TPU_SHA256" "$WHEEL_PATH" | sha256sum -c -
test "$(git -C /workspace/vllm-torchtpu rev-parse HEAD)" = \
  "$EXPECTED_SOURCE_COMMIT"
test "$(git -C /workspace/vllm rev-parse HEAD)" = "$VLLM_COMMIT"

python3 -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip 'setuptools>=77,<81' wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu'
python -m pip install "$WHEEL_PATH"

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
python -m pip install pytest

python - <<'PY' | tee "$ROOT/profiling/python-packages.txt"
import importlib.metadata as md
import torch
import torch_tpu

for package in (
    "torch",
    "torchvision",
    "torch-tpu",
    "vllm",
    "vllm-torchtpu",
    "jax",
    "jaxlib",
    "libtpu",
):
    try:
        print(f"{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}=MISSING")
x = torch.ones((8, 8), device="tpu")
value = (x @ x).cpu()[0, 0].item()
print(f"tpu_matmul={value}")
assert value == 8.0
print("TORCH_TPU_DEVICE_SMOKE_OK")
PY

if test "$RUN_TESTS" = 1; then
  python -m pytest -q \
    /workspace/vllm-torchtpu/tests/models/vllm/test_kimi_k3.py \
    -k attention_residual_matches_reference \
    2>&1 | tee "$VARIANT_OUT/attention-residual-pytest.log"
  if test "$HAS_SP_FEATURE" = 1; then
    python -m pytest -q \
      /workspace/vllm-torchtpu/tests/distributed/test_sequence_parallel.py \
      /workspace/vllm-torchtpu/tests/models/vllm/test_kimi_k3_sequence_parallel.py \
      /workspace/vllm-torchtpu/tests/platforms/test_tpu_platform.py \
      -k sequence_parallel \
      2>&1 | tee "$VARIANT_OUT/sequence-parallel-pytest.log"
  fi
fi

python - "$ROOT/manifest.json" "$VARIANT" "$EXPECTED_SOURCE_COMMIT" \
  "$SP_SIZE" "$ISL_CSV" "$COMPILE_SIZE_CSV" "$ROUNDS" <<'PY'
import json
import pathlib
import sys

path, variant, commit, sp_size, isls, compile_sizes, rounds = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "workflow": "operator-optimization",
    "operator_family": "kimi_k3_attention_residual_sp",
    "operator_name": "vllm_kimi_k3_sequence_parallel",
    "model": "mgoin/Kimi-K3-pruned75",
    "model_revision": "3c8e22fdac0c14409eebd48ba78c10a940f5267a",
    "source_commit": commit,
    "vllm_commit": "4bdc8a788d2e2ce9165d552b3d4d8b72604626bf",
    "torch_tpu": "0.1.1.dev20260813160135",
    "hardware": {
        "name": "v7x-8",
        "device_type": "v7x",
        "device_count": 8,
        "device_topo": "2x2x1",
        "replica": 1,
    },
    "dimensions": {
        "variant": variant,
        "tp_size": 8,
        "sequence_parallel_size": int(sp_size),
        "isls": [int(value) for value in isls.split(",")],
        "osl": 1,
        "concurrency": 1,
        "compile_sizes": [int(value) for value in compile_sizes.split(",")],
        "rounds": int(rounds),
        "kda_backend": "fused-conv1d-gdn-v3",
    },
}
pathlib.Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
PY

env | sort | grep -E \
  '^(VLLM_TORCHTPU_|FALCON_K3_SP_|EXPECTED_TORCHTPU_COMMIT)=' \
  >"$ROOT/profiling/env.txt" || true

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

SP_ARGS=()
if test "$HAS_SP_FEATURE" = 1; then
  SP_ARGS=(--additional-config "{\"sequence_parallel_size\":$SP_SIZE}")
fi

VLLM_CACHE_ROOT="$CACHE_ROOT/vllm-cache" \
TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/tier3" \
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --max-num-batched-tokens "$MAX_ISL" \
  --max-num-scheduled-tokens "$MAX_ISL" \
  --long-prefill-token-threshold "$MAX_ISL" \
  --enable-chunked-prefill \
  --block-size 256 \
  --gpu-memory-utilization 0.81 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --safetensors-load-strategy prefetch \
  --safetensors-prefetch-num-threads 8 \
  --enable-logging-iteration-details \
  "--compilation-config={\"compile_sizes\":[${COMPILE_SIZE_CSV}]}" \
  "${SP_ARGS[@]}" \
  --host 0.0.0.0 \
  --port 8000 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 1080); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -500 "$SERVER_LOG" || true
    exit 1
  fi
  if curl -sf http://localhost:8000/health >/dev/null; then
    printf 'KIMI_K3_SP_SERVER_READY variant=%s elapsed_seconds=%s\n' \
      "$VARIANT" "$((i * 10))" | tee "$VARIANT_OUT/ready.marker"
    ready=1
    break
  fi
  sleep 10
done
test "$ready" = 1

grep -F 'using the fused conv1d + GDN v3 path' "$SERVER_LOG" \
  >"$VARIANT_OUT/kda-backend.log"
if test "$SP_SIZE" = 4; then
  test "$(grep -c 'Sequence parallelism enabled' "$SERVER_LOG")" -eq 8
elif grep -Fq 'Sequence parallelism enabled' "$SERVER_LOG"; then
  printf 'Sequence parallelism unexpectedly enabled for %s\n' "$VARIANT" >&2
  exit 1
fi

fixed_request() {
  local tokens="$1"
  local output="$2"
  python - "$VARIANT" "$MODEL_NAME" "$tokens" "$output" <<'PY'
import json
import pathlib
import sys

import requests

variant, model, token_count, output_path = sys.argv[1:]
token_count = int(token_count)
prompt = [1 + ((index * 17 + 23) % 10000) for index in range(token_count)]
response = requests.post(
    "http://127.0.0.1:8000/v1/completions",
    json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 42,
        "logprobs": 20,
        "ignore_eos": True,
    },
    timeout=1800,
)
response.raise_for_status()
choice = response.json()["choices"][0]
logprobs = choice["logprobs"]
record = {
    "variant": variant,
    "tokens": token_count,
    "sampled_text": choice["text"],
    "sampled_token": logprobs["tokens"][0],
    "sampled_logprob": logprobs["token_logprobs"][0],
    "top_logprobs": logprobs["top_logprobs"][0],
}
pathlib.Path(output_path).write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n"
)
print("KIMI_K3_SP_CORRECTNESS=" +
      json.dumps(record, ensure_ascii=False, sort_keys=True))
PY
}

run_benchmark_round() {
  local tokens="$1"
  local round="$2"
  local round_dir="$VARIANT_OUT/benchmarks/isl-${tokens}/round-${round}"
  local result="$round_dir/benchmark.json"
  mkdir -p "$round_dir"
  vllm bench serve \
    --backend vllm \
    --model "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --trust-remote-code \
    --base-url http://127.0.0.1:8000 \
    --dataset-name random \
    --random-input-len "$tokens" \
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
    --metadata variant="$VARIANT" logical_isl="$tokens" \
      logical_osl=1 round="$round" \
    2>&1 | tee "$round_dir/benchmark.log"
  python - "$VARIANT" "$tokens" "$round" "$result" \
    "$OUT/benchmark/metrics.jsonl" <<'PY'
import json
import pathlib
import sys

variant, expected_tokens, round_index, result_path, metrics_path = sys.argv[1:]
expected_tokens = int(expected_tokens)
data = json.loads(pathlib.Path(result_path).read_text())
if data.get("completed") != 8 or data.get("failed", 0) != 0:
    raise SystemExit(f"incomplete benchmark: {data}")
if set(data.get("input_lens", [])) != {expected_tokens}:
    raise SystemExit(f"unexpected input lengths: {data.get('input_lens')}")
if set(data.get("output_lens", [])) != {1}:
    raise SystemExit(f"unexpected output lengths: {data.get('output_lens')}")
row = {
    "variant": variant,
    "round": int(round_index),
    "tokens": expected_tokens,
    "seq_len": expected_tokens,
    "batch_size": 1,
    "concurrency": 1,
    "latency_ms": data["mean_ttft_ms"],
    "mean_ttft_ms": data["mean_ttft_ms"],
    "median_ttft_ms": data.get("median_ttft_ms"),
    "p90_ttft_ms": data.get("p90_ttft_ms"),
    "p99_ttft_ms": data.get("p99_ttft_ms"),
    "request_throughput": data.get("request_throughput"),
    "total_token_throughput": data.get("total_token_throughput"),
    "completed": data["completed"],
    "failed": data.get("failed", 0),
}
with pathlib.Path(metrics_path).open("a") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

for isl in "${ISLS[@]}"; do
  fixed_request "$isl" "$VARIANT_OUT/correctness/isl-${isl}.json"
  for round in $(seq 1 "$ROUNDS"); do
    run_benchmark_round "$isl" "$round"
  done
done

if test "$E2E_SMOKE" = 1; then
  python - "$VARIANT" "$MODEL_NAME" "$VARIANT_OUT/e2e-smoke.json" <<'PY'
import json
import pathlib
import sys

import requests

variant, model, output_path = sys.argv[1:]
prompt = [1 + ((index * 17 + 23) % 10000) for index in range(4096)]
response = requests.post(
    "http://127.0.0.1:8000/v1/completions",
    json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
    },
    timeout=1800,
)
response.raise_for_status()
choice = response.json()["choices"][0]
record = {
    "variant": variant,
    "isl": 4096,
    "requested_osl": 32,
    "finish_reason": choice.get("finish_reason"),
    "generated_text": choice.get("text", ""),
}
pathlib.Path(output_path).write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n"
)
print("KIMI_K3_SP_E2E_SMOKE=" +
      json.dumps(record, ensure_ascii=False, sort_keys=True))
PY
fi

cleanup_server

python - "$OUT/benchmark/metrics.jsonl" "$VARIANT_OUT/summary.json" <<'PY'
import json
import pathlib
import statistics
import sys

metrics_path, summary_path = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in metrics_path.read_text().splitlines()
        if line.strip()]
grouped = {}
for row in rows:
    grouped.setdefault(str(row["tokens"]), []).append(row["mean_ttft_ms"])
summary = {
    "variant": rows[0]["variant"],
    "measurements": rows,
    "by_isl": {
        tokens: {
            "rounds": len(values),
            "mean_ttft_ms": statistics.mean(values),
            "median_round_ttft_ms": statistics.median(values),
            "min_ttft_ms": min(values),
            "max_ttft_ms": max(values),
        }
        for tokens, values in sorted(grouped.items(), key=lambda item: int(item[0]))
    },
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print("KIMI_K3_SP_RETEST_SUMMARY=" + json.dumps(summary, sort_keys=True))
PY

test -s "$ROOT/manifest.json"
test -s "$OUT/benchmark/metrics.jsonl"
test -s "$VARIANT_OUT/summary.json"
printf 'KIMI_K3_SP_RETEST_OK variant=%s\n' "$VARIANT" \
  | tee "$VARIANT_OUT/success.marker"
