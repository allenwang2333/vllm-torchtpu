#!/usr/bin/env bash
# Clean C=1 decode timing for one Kimi-K3 routed-MoE layout on v7x-32.

set -euo pipefail
ulimit -c 0

readonly SOURCE_COMMIT="${EXPECTED_TORCHTPU_COMMIT:?missing source commit}"
readonly VARIANT="${K3_MOE_VARIANT:?missing K3_MOE_VARIANT}"
readonly HIERARCHICAL="${TPU_MOE_HIERARCHICAL_EP:?missing hierarchical flag}"
readonly MODEL_NAME="moonshotai/Kimi-K3"
readonly MODEL_PATH="/models/Kimi-K3-a590ce09"
readonly MODEL_REVISION="a590ce090cb049c93a33dfe8c208ec652aa20503"
readonly VLLM_COMMIT="4bdc8a788d2e2ce9165d552b3d4d8b72604626bf"
readonly WHEEL_PATH="/wheels/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl"
readonly WHEEL_SHA256="e1e326fd4efbc85e965618e6a97a653eb256f00f817d2955f1fffaa59d0e525a"
readonly RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
readonly WORLD="${FALCON_WORLD_SIZE:-4}"
readonly POD_HOST="$(hostname)"
readonly JOB_BASE="${POD_HOST%-*}"
readonly MASTER_ADDR="${JOB_BASE}-0.${JOB_BASE}"
readonly OUT_ROOT="${ARTIFACT_LOCAL_DIR:?missing ARTIFACT_LOCAL_DIR}"
readonly OUT="$OUT_ROOT/rank-$RANK"
readonly BARRIER_ROOT="$OUT_ROOT/barriers"
readonly CACHE_ROOT="/tmp/tpu_logs_k3_moe_decode_${VARIANT}"
readonly VENV="$CACHE_ROOT/venv"
readonly SERVER_LOG="$CACHE_ROOT/server-rank$RANK.log"

if test "$WORLD" != 4 || test "$RANK" -lt 0 || test "$RANK" -gt 3; then
  printf 'UNEXPECTED_FALCON_TOPOLOGY rank=%s world=%s\n' "$RANK" "$WORLD" >&2
  exit 1
fi
case "$VARIANT:$HIERARCHICAL" in
  flat-ep32:0|chip-ep16-tp2:1) ;;
  *)
    printf 'INVALID_VARIANT variant=%s hierarchical=%s\n' \
      "$VARIANT" "$HIERARCHICAL" >&2
    exit 1
    ;;
esac

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
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=mega
export TPU_TOKEN_BUCKET_EXTRA=4
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_MOE_SKIP_PADDED_TOKENS=1
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=0
export RAGGED_GATHER_REDUCE_VERSION=v2

SERVER_PID=0
TAIL_PID=0

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" \
  "$HF_HOME" \
  "$OUT" \
  "$BARRIER_ROOT" \
  "$OUT_ROOT/correctness" \
  "$OUT_ROOT/rank-0/benchmark"

exec > >(tee -a "$OUT/task.log") 2>&1
printf 'FALCON_MULTIHOST rank=%s world=%s pod=%s master=%s variant=%s\n' \
  "$RANK" "$WORLD" "$POD_HOST" "$MASTER_ADDR" "$VARIANT"
getent hosts "$MASTER_ADDR"

test "$(git -C /workspace/vllm-torchtpu rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(git -C /workspace/vllm rev-parse HEAD)" = "$VLLM_COMMIT"
test -s "$MODEL_PATH/config.json"
test -s "$MODEL_PATH/model.safetensors.index.json"
test -s "$WHEEL_PATH"
printf '%s  %s\n' "$WHEEL_SHA256" "$WHEEL_PATH" | sha256sum -c -

python3 -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip 'setuptools>=77,<81' wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu'
python -m pip install "$WHEEL_PATH"
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

python - <<'PY' > "$OUT/runtime-versions.txt"
import importlib.metadata as md

for package in (
    "jax", "jaxlib", "libtpu", "torch", "torch-tpu", "vllm",
    "vllm-torchtpu", "runai-model-streamer",
):
    try:
        print(f"{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}=MISSING")
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

cleanup_server() {
  set +e
  if test "$TAIL_PID" -gt 0; then
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
  fi
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
  TAIL_PID=0
  rm -f /tmp/libtpu_lockfile* 2>/dev/null || true
  set -e
}
trap cleanup_server EXIT

start_server() {
  vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 32 \
    --nnodes 4 \
    --node-rank "$RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port 29500 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 256 \
    --max-num-scheduled-tokens 256 \
    --long-prefill-token-threshold 256 \
    --enable-chunked-prefill \
    --block-size 256 \
    --gpu-memory-utilization 0.81 \
    --no-enable-prefix-caching \
    --language-model-only \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":16,"memory_limit":8589934592}' \
    --enable-logging-iteration-details \
    '--compilation-config={"compile_sizes":[4]}' \
    $([[ "$RANK" = 0 ]] && printf '%s' '--host 0.0.0.0 --port 8000' || printf '%s' '--headless') \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  tail -n +1 -F "$SERVER_LOG" &
  TAIL_PID=$!

  local ready=0
  for i in $(seq 1 1440); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      printf 'VLLM_PROCESS_DIED variant=%s rank=%s\n' "$VARIANT" "$RANK" >&2
      tail -500 "$SERVER_LOG" || true
      return 1
    fi
    if curl --connect-timeout 2 --max-time 5 -sf \
      "http://$MASTER_ADDR:8000/health" >/dev/null; then
      printf 'VLLM_CLUSTER_READY variant=%s rank=%s elapsed_seconds=%s\n' \
        "$VARIANT" "$RANK" "$((i * 10))"
      ready=1
      break
    fi
    sleep 10
  done
  test "$ready" = 1

  if test "$HIERARCHICAL" = 1; then
    grep -F 'Hierarchical MoE parallelism: EP between chips (ep_size=16' \
      "$SERVER_LOG" > "$OUT/hierarchical-split.log"
    grep -F 'EP weight filter realigned to chip-block ownership: ep_size=16' \
      "$SERVER_LOG" > "$OUT/hierarchical-weight-filter.log"
  elif grep -Fq 'Hierarchical MoE parallelism:' "$SERVER_LOG"; then
    printf 'HIERARCHICAL_UNEXPECTED rank=%s\n' "$RANK" >&2
    return 1
  fi
}

fixed_request() {
  python - "$MODEL_NAME" "$MASTER_ADDR" "$OUT_ROOT/correctness/$VARIANT.json" <<'PY'
import json
import pathlib
import sys
import requests

model, host, output = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
response = requests.post(
    f"http://{host}:8000/v1/completions",
    json={
        "model": model,
        "prompt": [24],
        "max_tokens": 128,
        "temperature": 0,
        "seed": 42,
        "logprobs": 5,
        "ignore_eos": True,
    },
    timeout=7200,
)
response.raise_for_status()
body = response.json()
choice = body["choices"][0]
logprobs = choice.get("logprobs") or {}
tokens = logprobs.get("tokens") or []
if len(tokens) != 128 or len(logprobs.get("token_logprobs") or []) != 128:
    raise SystemExit(f"incomplete deterministic decode: {len(tokens)} tokens")
output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"event": "correctness", "tokens": len(tokens)}))
PY
}

run_benchmarks() {
  for repeat in 0 1 2; do
    local result="$OUT_ROOT/rank-0/benchmark/$VARIANT-repeat-$repeat.json"
    vllm bench serve \
      --backend vllm \
      --model "$MODEL_NAME" \
      --tokenizer "$MODEL_PATH" \
      --trust-remote-code \
      --base-url "http://$MASTER_ADDR:8000" \
      --dataset-name random \
      --random-input-len 1 \
      --random-output-len 256 \
      --random-range-ratio 0 \
      --num-prompts 8 \
      --num-warmups 4 \
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
      --metadata variant="$VARIANT" repeat="$repeat" logical_isl=1 logical_osl=256 \
      2>&1 | tee "$OUT_ROOT/rank-0/benchmark/$VARIANT-repeat-$repeat.log"
    python - "$result" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if data.get("completed") != 8 or data.get("failed", 0) != 0:
    raise SystemExit(f"incomplete benchmark: {data}")
if set(data.get("input_lens", [])) != {1}:
    raise SystemExit(f"unexpected input lengths: {data.get('input_lens')}")
if set(data.get("output_lens", [])) != {256}:
    raise SystemExit(f"unexpected output lengths: {data.get('output_lens')}")
PY
  done
}

barrier setup-complete 1200
start_server
barrier server-ready 1800

if test "$RANK" = 0; then
  fixed_request
  run_benchmarks
fi
barrier benchmark-complete 14400

cp "$SERVER_LOG" "$OUT/server.log"
python - "$SERVER_LOG" "$OUT/iteration-shapes.json" <<'PY'
import collections
import json
import pathlib
import re
import sys

source, output = map(pathlib.Path, sys.argv[1:])
pattern = re.compile(
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([0-9.]+) ms"
)
histogram = collections.Counter()
decode_ms = []
for line in source.read_text(errors="replace").splitlines():
    match = pattern.search(line)
    if not match:
        continue
    requests, tokens, elapsed_ms = match.groups()
    if int(requests) > 0 and int(tokens) > 0:
        histogram[int(tokens)] += 1
        decode_ms.append(float(elapsed_ms))
record = {
    "generation_token_count_histogram": dict(sorted(histogram.items())),
    "decode_iterations": len(decode_ms),
    "iteration_elapsed_ms_min": min(decode_ms) if decode_ms else None,
    "iteration_elapsed_ms_max": max(decode_ms) if decode_ms else None,
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
if not histogram:
    raise SystemExit("no generation iteration details found")
PY
barrier logs-copied 1200
cleanup_server
barrier server-stopped 600

if test "$RANK" = 0; then
  python - "$OUT_ROOT" "$VARIANT" "$SOURCE_COMMIT" "$HIERARCHICAL" <<'PY'
import hashlib
import json
import pathlib
import statistics
import sys

root, variant, source_commit, hierarchical = sys.argv[1:]
root = pathlib.Path(root)
metrics = []
for repeat in range(3):
    path = root / "rank-0" / "benchmark" / f"{variant}-repeat-{repeat}.json"
    data = json.loads(path.read_text())
    row = {
        "variant": variant,
        "repeat": repeat,
        "physical_generation_tokens": 1,
        "median_tpot_ms": data.get("median_tpot_ms"),
        "p90_tpot_ms": data.get("p90_tpot_ms"),
        "median_itl_ms": data.get("median_itl_ms"),
        "p90_itl_ms": data.get("p90_itl_ms"),
        "output_tokens_per_sec": data.get("output_throughput"),
        "request_throughput": data.get("request_throughput"),
        "completed": data.get("completed"),
        "failed": data.get("failed", 0),
    }
    if row["median_tpot_ms"] is None or row["p90_tpot_ms"] is None:
        raise SystemExit(f"missing TPOT metrics: {data.keys()}")
    metrics.append(row)

metrics_path = root / "rank-0" / "benchmark" / "metrics.jsonl"
metrics_path.write_text("".join(json.dumps(row) + "\n" for row in metrics))
correctness_path = root / "correctness" / f"{variant}.json"
correctness = json.loads(correctness_path.read_text())
choice = correctness["choices"][0]
token_strings = choice["logprobs"]["tokens"]
summary = {
    "variant": variant,
    "source_commit": source_commit,
    "hierarchical": bool(int(hierarchical)),
    "repeats": len(metrics),
    "median_of_median_tpot_ms": statistics.median(
        row["median_tpot_ms"] for row in metrics
    ),
    "median_of_p90_tpot_ms": statistics.median(
        row["p90_tpot_ms"] for row in metrics
    ),
    "token_strings_sha256": hashlib.sha256(
        json.dumps(token_strings, ensure_ascii=False).encode()
    ).hexdigest(),
    "generated_text_sha256": hashlib.sha256(
        choice["text"].encode()
    ).hexdigest(),
    "generation_token_count_histogram": json.loads(
        (root / "rank-0" / "iteration-shapes.json").read_text()
    )["generation_token_count_histogram"],
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
(root / "manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "workflow": "kimi-k3-c1-moe-decode-clean",
    "operator_family": "kimi_k3_decode_moe",
    "variant": variant,
    "source_commit": source_commit,
    "model_revision": "a590ce090cb049c93a33dfe8c208ec652aa20503",
    "vllm_commit": "4bdc8a788d2e2ce9165d552b3d4d8b72604626bf",
    "torch_tpu": "0.1.1.dev20260813160135",
    "hardware": "v7x32",
    "dimensions": {
        "tep_size": 32,
        "ep_size": 16 if int(hierarchical) else 32,
        "expert_tp_size": 2 if int(hierarchical) else 1,
        "concurrency": 1,
        "logical_isl": 1,
        "logical_osl": 256,
        "compile_sizes": [4],
        "top_k": 16,
        "latent_size": 3584,
        "intermediate_size": 3072,
        "num_experts": 896,
        "weight_dtype": "mxfp4_group32",
    },
}, indent=2) + "\n")
print("KIMI_K3_MOE_DECODE_SUMMARY=" + json.dumps(summary, sort_keys=True))
PY
  printf 'KIMI_K3_MOE_DECODE_CLEAN_OK\n' | tee "$OUT_ROOT/success.marker"
fi

barrier variant-complete 600
