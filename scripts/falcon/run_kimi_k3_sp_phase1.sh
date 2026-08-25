#!/usr/bin/env bash
# Phase 1 correctness and profiling gate for the experimental Kimi-K3 SP4
# layout. Falcon supplies the source, model, wheel, TPU, and artifact directory.

set -euo pipefail
ulimit -c 0

readonly MODEL_NAME="mgoin/Kimi-K3-pruned75"
readonly MODEL_PATH="/models/Kimi-K3-pruned75"
readonly MODEL_REVISION="3c8e22fdac0c14409eebd48ba78c10a940f5267a"
readonly VLLM_COMMIT="4bdc8a788d2e2ce9165d552b3d4d8b72604626bf"
readonly TORCH_TPU_VERSION="0.1.1.dev20260813160135"
readonly TORCH_TPU_SHA256="e1e326fd4efbc85e965618e6a97a653eb256f00f817d2955f1fffaa59d0e525a"
readonly WHEEL_PATH="/wheels/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl"
readonly TORCHTPU_COMMIT="${EXPECTED_TORCHTPU_COMMIT:?EXPECTED_TORCHTPU_COMMIT is required}"
readonly OUT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
readonly CACHE_ROOT="/tmp/tpu_logs"
readonly VENV="${CACHE_ROOT}/venv"
readonly PHASE1_MODE="${KIMI_K3_SP_PHASE1_MODE:-full}"

if test "$PHASE1_MODE" != full && test "$PHASE1_MODE" != sp-smoke; then
  printf 'Unsupported KIMI_K3_SP_PHASE1_MODE=%s\n' "$PHASE1_MODE" >&2
  exit 1
fi

export TMPDIR="${CACHE_ROOT}/tmp"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip-cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv-cache"
export HF_HOME="${CACHE_ROOT}/huggingface"
export PYTHONUNBUFFERED=1
export VLLM_TARGET_DEVICE=empty
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=mega

SERVER_PID=0
SERVER_LOG=""

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$OUT/E0-control-4096/profile" \
  "$OUT/E1-sp4-1024" \
  "$OUT/E2-sp4-4096/profile" \
  "$OUT/E3-sp4-2048" \
  "$OUT/correctness" \
  "$OUT/rank-0/benchmark"

test -f "$MODEL_PATH/_DOWNLOAD_COMPLETE.json"
test -f "$MODEL_PATH/config.json"
test -s "$WHEEL_PATH"
printf '%s  %s\n' "$TORCH_TPU_SHA256" "$WHEEL_PATH" | sha256sum -c -
test "$(git -C /workspace/vllm-torchtpu rev-parse HEAD)" = "$TORCHTPU_COMMIT"
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

python - <<'PY' | tee "$OUT/runtime-versions.txt"
import importlib.metadata as md
import torch
import torch_tpu
import vllm
import vllm_torchtpu

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

python -m pytest -q \
  /workspace/vllm-torchtpu/tests/models/vllm/test_kimi_k3_prefill_sp.py \
  /workspace/vllm-torchtpu/tests/tracing/test_annotation.py \
  2>&1 | tee "$OUT/targeted-pytest.log"

python - "$OUT/manifest.json" "$TORCHTPU_COMMIT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source_commit = sys.argv[2]
path.write_text(json.dumps({
    "schema_version": 1,
    "workflow": "kimi-k3-prefill-sp-phase1",
    "model": "mgoin/Kimi-K3-pruned75",
    "model_revision": "3c8e22fdac0c14409eebd48ba78c10a940f5267a",
    "source_commit": source_commit,
    "vllm_commit": "4bdc8a788d2e2ce9165d552b3d4d8b72604626bf",
    "torch_tpu": "0.1.1.dev20260813160135",
    "hardware": "v7x8",
    "logical_mesh": "SP4 x lane2",
    "rank_mapping": "tp_rank = lane * 4 + sp_rank",
    "lane_groups": [[0, 4], [1, 5], [2, 6], [3, 7]],
    "sp_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "events": {
        "E0": {"sp": 1, "tokens": 4096, "profile": True},
        "E1": {"sp": 4, "tokens": 1024, "profile": False},
        "E2": {"sp": 4, "tokens": 4096, "profile": True},
        "E3": {"sp": 4, "tokens": 2048, "profile": False},
    },
}, indent=2) + "\n")
PY

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
  if test -n "$SERVER_LOG"; then
    tail -500 "$SERVER_LOG" >"$OUT/last-server-lines.log" 2>/dev/null || true
  fi
}
trap cleanup_all EXIT

compile_count() {
  grep -cF '[TpuCompilerAdaptor] Saved compiled artifact' "$SERVER_LOG" || true
}

start_server() {
  local label="$1"
  local sp_size="$2"
  local profile_dir="$3"
  local variant_dir="$OUT/$label"

  SERVER_LOG="$variant_dir/server.log"
  mkdir -p \
    "$variant_dir" \
    "$profile_dir" \
    "$CACHE_ROOT/vllm-cache-$label" \
    "$CACHE_ROOT/tier3-$label"

  VLLM_CACHE_ROOT="$CACHE_ROOT/vllm-cache-$label" \
  TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/tier3-$label" \
  VLLM_TORCHTPU_K3_PREFILL_SP_SIZE="$sp_size" \
  vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    -tp 8 \
    -ep \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --max-num-scheduled-tokens 4096 \
    --long-prefill-token-threshold 4096 \
    --enable-chunked-prefill \
    --block-size 256 \
    --gpu-memory-utilization 0.81 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --safetensors-load-strategy prefetch \
    --safetensors-prefetch-num-threads 8 \
    --enable-logging-iteration-details \
    '--compilation-config={"compile_sizes":[4,1024,2048,4096]}' \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "$profile_dir" \
    --profiler-config.ignore_frontend true \
    --host 0.0.0.0 \
    --port 8000 \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for i in $(seq 1 1080); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -500 "$SERVER_LOG" || true
      return 1
    fi
    if curl -sf http://localhost:8000/health >/dev/null; then
      printf 'VLLM_SERVER_READY label=%s elapsed_seconds=%s\n' \
        "$label" "$((i * 10))" | tee "$variant_dir/ready.marker"
      ready=1
      break
    fi
    sleep 10
  done
  if test "$ready" != 1; then
    tail -500 "$SERVER_LOG" || true
    return 1
  fi

  grep -F 'Kimi-K3 KDA prefill kernel selected: mega' "$SERVER_LOG" \
    >"$variant_dir/kda-kernel.log"
  if test "$sp_size" = 4; then
    grep -F 'Kimi-K3 prefill SP enabled' "$SERVER_LOG" \
      >"$variant_dir/sp-groups.log"
    grep -F 'lane_groups=((0, 4), (1, 5), (2, 6), (3, 7))' \
      "$variant_dir/sp-groups.log"
    grep -F 'sp_groups=((0, 1, 2, 3), (4, 5, 6, 7))' \
      "$variant_dir/sp-groups.log"
  elif grep -Fq 'Kimi-K3 prefill SP enabled' "$SERVER_LOG"; then
    printf 'SP unexpectedly enabled in control\n' >&2
    return 1
  fi

  curl --fail-with-body -sS http://localhost:8000/v1/models \
    >"$variant_dir/models.json"
  compile_count >"$variant_dir/compile-count-at-ready.txt"
}

stop_server() {
  local variant_dir="$1"
  compile_count >"$variant_dir/compile-count-after-workloads.txt"
  cmp \
    "$variant_dir/compile-count-at-ready.txt" \
    "$variant_dir/compile-count-after-workloads.txt"
  cleanup_server
}

fixed_request() {
  local label="$1"
  local tokens="$2"
  local output="$3"
  python - "$label" "$MODEL_NAME" "$tokens" "$output" <<'PY'
import json
import pathlib
import sys
import requests

label = sys.argv[1]
model = sys.argv[2]
token_count = int(sys.argv[3])
output = pathlib.Path(sys.argv[4])
prompt = [1 + ((index * 17 + 23) % 10000) for index in range(token_count)]
payload = {
    "model": model,
    "prompt": prompt,
    "max_tokens": 1,
    "temperature": 0,
    "seed": 42,
    "logprobs": 20,
    "ignore_eos": True,
}
response = requests.post(
    "http://127.0.0.1:8000/v1/completions",
    json=payload,
    timeout=1800,
)
response.raise_for_status()
body = response.json()
if not body.get("choices"):
    raise SystemExit(f"missing completion choice: {body}")
choice = body["choices"][0]
logprobs = choice.get("logprobs") or {}
if not logprobs.get("token_logprobs") or not logprobs.get("top_logprobs"):
    raise SystemExit(f"missing final logprobs: {body}")
record = {
    "label": label,
    "token_count": token_count,
    "prompt_token_ids": prompt,
    "response": body,
}
output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "output": str(output),
    "token_count": token_count,
    "sampled_token": logprobs["tokens"][0],
    "sampled_token_logprob": logprobs["token_logprobs"][0],
}, ensure_ascii=False))
PY
}

run_bench() {
  local event="$1"
  local tokens="$2"
  local result="$OUT/$event/benchmark.json"
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
    --metadata event="$event" logical_isl="$tokens" logical_osl=1 \
    2>&1 | tee "$OUT/$event/benchmark.log"
  python - "$event" "$tokens" "$result" <<'PY'
import json
import pathlib
import sys

event, expected_tokens, path = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
data = json.loads(path.read_text())
if data.get("completed") != 8 or data.get("failed", 0) != 0:
    raise SystemExit(f"{event}: incomplete benchmark: {data}")
if set(data.get("input_lens", [])) != {expected_tokens}:
    raise SystemExit(f"{event}: unexpected input lengths: {data.get('input_lens')}")
if set(data.get("output_lens", [])) != {1}:
    raise SystemExit(f"{event}: unexpected output lengths: {data.get('output_lens')}")
PY
}

capture_profile() {
  local event="$1"
  local output="$2"
  local profile_dir="$3"
  local artifact_profile_dir="$OUT/$event/profile"
  mkdir -p "$profile_dir" "$artifact_profile_dir"
  curl --fail-with-body -sS -X POST http://localhost:8000/start_profile \
    | tee "$OUT/$event/start-profile.json"
  fixed_request "$event-profile" 4096 "$output"
  curl --fail-with-body -sS -X POST http://localhost:8000/stop_profile \
    | tee "$OUT/$event/stop-profile.json"
  local ready=0
  for _ in $(seq 1 120); do
    if find "$profile_dir" -type f \
      \( -name '*.xplane.pb' -o -name '*.json' -o -name '*.json.gz' \) \
      -size +0c | grep -q .; then
      ready=1
      break
    fi
    sleep 5
  done
  if test "$ready" != 1; then
    printf 'No profile artifacts appeared for %s\n' "$event" >&2
    find "$profile_dir" -maxdepth 6 -print | sort || true
    grep -Ei 'profil|trace|xplane|kineto' "$SERVER_LOG" | tail -200 || true
    return 1
  fi
  cp -a "$profile_dir/." "$artifact_profile_dir/"
  find "$profile_dir" -type f -size +0c -print | sort \
    >"$OUT/$event/profile-files.txt"
  cat "$OUT/$event/profile-files.txt"
}

if test "$PHASE1_MODE" = sp-smoke; then
  start_server E2-sp4-4096 4 "$CACHE_ROOT/profile-E2-sp4-4096"
  fixed_request sp4-smoke-1024 1024 \
    "$OUT/correctness/sp4-smoke-1024.json"
  fixed_request sp4-smoke-4096 4096 \
    "$OUT/correctness/sp4-smoke-4096.json"
  stop_server "$OUT/E2-sp4-4096"
  printf 'KIMI_K3_PREFILL_SP_PHASE1_SMOKE_OK\n' | tee "$OUT/success.marker"
  exit 0
fi

# Control: E0 is the measured 4096-token run. The extra fixed 1024/2048
# requests provide same-source correctness references for E1 and E3.
start_server E0-control-4096 1 "$CACHE_ROOT/profile-E0-control-4096"
fixed_request control-1024 1024 "$OUT/correctness/control-1024.json"
fixed_request control-2048 2048 "$OUT/correctness/control-2048.json"
fixed_request control-4096 4096 "$OUT/correctness/control-4096.json"
run_bench E0-control-4096 4096
capture_profile \
  E0-control-4096 \
  "$OUT/correctness/control-4096-profile.json" \
  "$CACHE_ROOT/profile-E0-control-4096"
stop_server "$OUT/E0-control-4096"

start_server E2-sp4-4096 4 "$CACHE_ROOT/profile-E2-sp4-4096"
fixed_request sp4-1024 1024 "$OUT/correctness/sp4-1024.json"
run_bench E1-sp4-1024 1024
fixed_request sp4-4096 4096 "$OUT/correctness/sp4-4096.json"
run_bench E2-sp4-4096 4096
capture_profile \
  E2-sp4-4096 \
  "$OUT/correctness/sp4-4096-profile.json" \
  "$CACHE_ROOT/profile-E2-sp4-4096"
fixed_request sp4-2048 2048 "$OUT/correctness/sp4-2048.json"
run_bench E3-sp4-2048 2048
stop_server "$OUT/E2-sp4-4096"

python - "$OUT" <<'PY'
import json
import math
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
comparisons = {}
min_top20_overlap = 19
logprob_atol = 0.125
for token_count in (1024, 2048, 4096):
    control = json.loads((out / "correctness" / f"control-{token_count}.json").read_text())
    sp = json.loads((out / "correctness" / f"sp4-{token_count}.json").read_text())
    control_choice = control["response"]["choices"][0]
    sp_choice = sp["response"]["choices"][0]
    control_lp = control_choice["logprobs"]
    sp_lp = sp_choice["logprobs"]
    if control_choice["text"] != sp_choice["text"]:
        raise SystemExit(
            f"{token_count}: sampled token mismatch: "
            f"{control_choice['text']!r} != {sp_choice['text']!r}"
        )
    control_top = control_lp["top_logprobs"][0]
    sp_top = sp_lp["top_logprobs"][0]
    common_tokens = set(control_top) & set(sp_top)
    if len(common_tokens) < min_top20_overlap:
        raise SystemExit(
            f"{token_count}: final top-20 overlap={len(common_tokens)}/20; "
            f"symmetric difference={set(control_top) ^ set(sp_top)}"
        )
    deltas = {
        token: abs(float(control_top[token]) - float(sp_top[token]))
        for token in common_tokens
    }
    max_delta = max(deltas.values(), default=0.0)
    sampled_delta = abs(float(control_lp["token_logprobs"][0]) -
                        float(sp_lp["token_logprobs"][0]))
    if (not math.isfinite(max_delta) or not math.isfinite(sampled_delta)
            or max(max_delta, sampled_delta) > logprob_atol):
        raise SystemExit(f"{token_count}: final top-20 logprob delta={max_delta}")
    comparisons[str(token_count)] = {
        "sampled_token": control_choice["text"],
        "sampled_logprob_control": control_lp["token_logprobs"][0],
        "sampled_logprob_sp4": sp_lp["token_logprobs"][0],
        "sampled_logprob_delta": sampled_delta,
        "top20_overlap": len(common_tokens),
        "top20_symmetric_difference": sorted(set(control_top) ^ set(sp_top)),
        "max_abs_top20_logprob_delta": max_delta,
        "top20_logprob_deltas": deltas,
        "min_top20_overlap": min_top20_overlap,
        "logprob_atol": logprob_atol,
    }

(out / "correctness-summary.json").write_text(
    json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n"
)
print("KIMI_K3_PREFILL_SP_PHASE1_CORRECTNESS=" +
      json.dumps(comparisons, ensure_ascii=False, sort_keys=True))

metrics = []
for event in (
    "E0-control-4096",
    "E1-sp4-1024",
    "E2-sp4-4096",
    "E3-sp4-2048",
):
    data = json.loads((out / event / "benchmark.json").read_text())
    metrics.append({
        "variant": event,
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "median_ttft_ms": data.get("median_ttft_ms"),
        "p90_ttft_ms": data.get("p90_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "request_throughput": data.get("request_throughput"),
        "completed": data.get("completed"),
        "failed": data.get("failed", 0),
    })
(out / "rank-0" / "benchmark" / "metrics.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in metrics)
)
(out / "phase1-summary.json").write_text(json.dumps({
    "status": "passed",
    "correctness": comparisons,
    "metrics": metrics,
    "profiles": ["E0-control-4096", "E2-sp4-4096"],
}, ensure_ascii=False, indent=2) + "\n")
PY

test -s "$OUT/correctness-summary.json"
test -s "$OUT/phase1-summary.json"
test -s "$OUT/rank-0/benchmark/metrics.jsonl"
printf 'KIMI_K3_PREFILL_SP_PHASE1_OK\n' | tee "$OUT/success.marker"
