#!/usr/bin/env bash
# Phase 2 full-model correctness and paired profiling gate for Kimi-K3 SP4.
# Falcon launches this command on four v7x8 hosts forming one v7x32 slice.

set -euo pipefail
ulimit -c 0

readonly MODEL_NAME="moonshotai/Kimi-K3"
readonly MODEL_PATH="/models/Kimi-K3-a590ce09"
readonly MODEL_REVISION="a590ce090cb049c93a33dfe8c208ec652aa20503"
readonly VLLM_COMMIT="4bdc8a788d2e2ce9165d552b3d4d8b72604626bf"
readonly TORCH_TPU_SHA256="e1e326fd4efbc85e965618e6a97a653eb256f00f817d2955f1fffaa59d0e525a"
readonly WHEEL_PATH="/wheels/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl"
readonly TORCHTPU_COMMIT="${EXPECTED_TORCHTPU_COMMIT:?EXPECTED_TORCHTPU_COMMIT is required}"
readonly RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
readonly WORLD="${FALCON_WORLD_SIZE:-4}"
readonly POD_HOST="$(hostname)"
readonly JOB_BASE="${POD_HOST%-*}"
readonly MASTER_ADDR="${JOB_BASE}-0.${JOB_BASE}"
readonly OUT_ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
readonly OUT="$OUT_ROOT/rank-$RANK"
readonly BARRIER_ROOT="$OUT_ROOT/barriers"
readonly CACHE_ROOT="/tmp/tpu_logs"
readonly VENV="$CACHE_ROOT/venv"

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
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=mega
export TPU_TOKEN_BUCKET_EXTRA=4,8,16,32,64,128
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_MOE_SKIP_PADDED_TOKENS=1

SERVER_PID=0
TAIL_PID=0
SERVER_LOG=""

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
  "$OUT_ROOT/E4-control-4096/profile/plugins/profile/E4-control-4096" \
  "$OUT_ROOT/E6-sp4-4096/profile/plugins/profile/E6-sp4-4096" \
  "$OUT_ROOT/correctness" \
  "$OUT_ROOT/rank-0/benchmark"

exec > >(tee -a "$OUT/task.log") 2>&1
printf 'FALCON_MULTIHOST rank=%s world=%s pod=%s master=%s\n' \
  "$RANK" "$WORLD" "$POD_HOST" "$MASTER_ADDR"
getent hosts "$MASTER_ADDR"

test -s "$WHEEL_PATH"
printf '%s  %s\n' "$TORCH_TPU_SHA256" "$WHEEL_PATH" | sha256sum -c -
test "$(git -C /workspace/vllm-torchtpu rev-parse HEAD)" = "$TORCHTPU_COMMIT"
test "$(git -C /workspace/vllm rev-parse HEAD)" = "$VLLM_COMMIT"

export MODEL_PATH MODEL_REVISION
python3 - <<'PY' | tee "$OUT/model-validation.json"
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_PATH"])
expected_revision = os.environ["MODEL_REVISION"]
expected_shards = 96
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
if len(shards) != expected_shards or len(referenced) != expected_shards:
    raise SystemExit(f"incomplete model shards: {record}")
if missing_referenced:
    raise SystemExit(f"missing referenced shards: {missing_referenced}")
print("KIMI_K3_OFFICIAL_WEIGHT_VALIDATION_OK")
PY

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
python -m pip install pytest

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

python -m pytest -q \
  /workspace/vllm-torchtpu/tests/models/vllm/test_kimi_k3_prefill_sp.py \
  /workspace/vllm-torchtpu/tests/tracing/test_annotation.py \
  /workspace/vllm-torchtpu/tests/test_profiler_trace.py \
  2>&1 | tee "$OUT/targeted-pytest.log"

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

cleanup_all() {
  cleanup_server
  if test -n "$SERVER_LOG"; then
    cp "$SERVER_LOG" "$OUT/last-server.log" 2>/dev/null || true
  fi
}
trap cleanup_all EXIT

compile_count() {
  grep -cF '[TpuCompilerAdaptor] Saved compiled artifact' "$SERVER_LOG" || true
}

start_server() {
  local event="$1"
  local sp_size="$2"
  local profile_dir="$CACHE_ROOT/profile-$event"
  local event_cache="$CACHE_ROOT/$event"
  SERVER_LOG="$CACHE_ROOT/server-$event-rank$RANK.log"
  mkdir -p "$profile_dir" "$event_cache/vllm" "$event_cache/tier3"

  local serve_args=(
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
    '--compilation-config={"compile_sizes":[4,1024,2048,4096]}'
    --profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$profile_dir"
    --profiler-config.ignore_frontend true
  )

  local launch_args=()
  if test "$RANK" = 0; then
    launch_args+=(--host 0.0.0.0 --port 8000)
  else
    launch_args+=(--headless)
  fi

  VLLM_CACHE_ROOT="$event_cache/vllm" \
  TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$event_cache/tier3" \
  VLLM_TORCHTPU_K3_PREFILL_SP_SIZE="$sp_size" \
  vllm serve "${serve_args[@]}" "${launch_args[@]}" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  tail -n +1 -F "$SERVER_LOG" &
  TAIL_PID=$!

  local ready=0
  for i in $(seq 1 1440); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      printf 'VLLM_PROCESS_DIED event=%s rank=%s\n' "$event" "$RANK" >&2
      return 1
    fi
    if curl --connect-timeout 2 --max-time 5 -sf \
      "http://$MASTER_ADDR:8000/health" >/dev/null; then
      printf 'VLLM_CLUSTER_READY event=%s rank=%s elapsed_seconds=%s\n' \
        "$event" "$RANK" "$((i * 10))"
      ready=1
      break
    fi
    sleep 10
  done
  if test "$ready" != 1; then
    printf 'VLLM_CLUSTER_READY_TIMEOUT event=%s rank=%s\n' "$event" "$RANK" >&2
    return 1
  fi

  grep -F 'Kimi-K3 KDA prefill kernel selected: mega' "$SERVER_LOG" \
    >"$OUT/$event-kda-kernel.log"
  if test "$sp_size" = 4; then
    grep -F 'Kimi-K3 prefill SP enabled' "$SERVER_LOG" \
      >"$OUT/$event-sp-groups.log"
    grep -F 'lane_groups=((0, 4, 8, 12, 16, 20, 24, 28)' \
      "$OUT/$event-sp-groups.log"
    grep -F 'sp_groups=((0, 1, 2, 3), (4, 5, 6, 7)' \
      "$OUT/$event-sp-groups.log"
  elif grep -Fq 'Kimi-K3 prefill SP enabled' "$SERVER_LOG"; then
    printf 'SP unexpectedly enabled in E4 control rank=%s\n' "$RANK" >&2
    return 1
  fi
  compile_count >"$OUT/$event-compile-count-at-ready.txt"
  barrier "$event-ready" 1800
}

fixed_request() {
  local label="$1"
  local output="$2"
  python - "$label" "$MODEL_NAME" "$MASTER_ADDR" "$output" <<'PY'
import json
import pathlib
import sys
import requests

label, model, host, output = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
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
body = response.json()
choice = body["choices"][0]
logprobs = choice.get("logprobs") or {}
if not logprobs.get("top_logprobs"):
    raise SystemExit(f"missing final logprobs: {body}")
output.write_text(json.dumps({
    "label": label,
    "token_count": 4096,
    "prompt_token_ids": prompt,
    "response": body,
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "label": label,
    "sampled_token": logprobs["tokens"][0],
    "sampled_token_logprob": logprobs["token_logprobs"][0],
}, ensure_ascii=False))
PY
}

run_bench() {
  local event="$1"
  local result="$OUT_ROOT/$event/benchmark.json"
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
    --metadata event="$event" logical_isl=4096 logical_osl=1 \
    2>&1 | tee "$OUT_ROOT/$event/benchmark.log"
  python - "$event" "$result" <<'PY'
import json
import pathlib
import sys

event, path = sys.argv[1], pathlib.Path(sys.argv[2])
data = json.loads(path.read_text())
if data.get("completed") != 8 or data.get("failed", 0) != 0:
    raise SystemExit(f"{event}: incomplete benchmark: {data}")
if set(data.get("input_lens", [])) != {4096}:
    raise SystemExit(f"{event}: unexpected input lengths: {data.get('input_lens')}")
PY
}

capture_profile_head() {
  local event="$1"
  local output="$2"
  curl --fail-with-body -sS -X POST "http://$MASTER_ADDR:8000/start_profile" \
    | tee "$OUT_ROOT/$event/start-profile.json"
  fixed_request "$event-profile" "$output"
  curl --fail-with-body -sS -X POST "http://$MASTER_ADDR:8000/stop_profile" \
    | tee "$OUT_ROOT/$event/stop-profile.json"
}

collect_local_profile() {
  local event="$1"
  local profile_dir="$CACHE_ROOT/profile-$event"
  local shared_dir="$OUT_ROOT/$event/profile"
  local local_count=0
  for _ in $(seq 1 120); do
    local_count=$(find "$profile_dir" -type f -name '*.xplane.pb' -size +0c | wc -l)
    if test "$local_count" -ge 8; then
      break
    fi
    sleep 5
  done
  if test "$local_count" -lt 8; then
    printf 'PROFILE_LOCAL_FILES_MISSING event=%s rank=%s count=%s\n' \
      "$event" "$RANK" "$local_count" >&2
    find "$profile_dir" -maxdepth 7 -print | sort || true
    grep -Ei 'profil|trace|xplane|kineto' "$SERVER_LOG" | tail -300 || true
    return 1
  fi

  mkdir -p \
    "$shared_dir/plugins/profile/$event" \
    "$shared_dir/host-$RANK"
  while IFS= read -r path; do
    cp "$path" "$shared_dir/plugins/profile/$event/$(basename "$path")"
  done < <(find "$profile_dir" -type f -name '*.xplane.pb' -size +0c | sort)
  find "$profile_dir" -type f ! -name '*.xplane.pb' -size +0c \
    -exec cp --parents '{}' "$shared_dir/host-$RANK"/ \;
  find "$profile_dir" -type f -size +0c -print | sort \
    >"$OUT/$event-profile-files.txt"
  barrier "$event-profile-copied" 1200

  if test "$RANK" = 0; then
    local shared_count
    shared_count=$(find "$shared_dir/plugins/profile/$event" \
      -type f -name '*.xplane.pb' -size +0c | wc -l)
    if test "$shared_count" -lt 32; then
      printf 'PROFILE_SHARED_FILES_MISSING event=%s count=%s\n' \
        "$event" "$shared_count" >&2
      return 1
    fi
    printf 'PROFILE_SHARED_OK event=%s xplanes=%s\n' "$event" "$shared_count"
  fi
}

finish_event() {
  local event="$1"
  compile_count >"$OUT/$event-compile-count-after-workloads.txt"
  cmp \
    "$OUT/$event-compile-count-at-ready.txt" \
    "$OUT/$event-compile-count-after-workloads.txt"
  barrier "$event-finished" 1200
  cleanup_server
  barrier "$event-stopped" 600
}

barrier setup-complete 1200

start_server E4-control-4096 1
if test "$RANK" = 0; then
  curl --fail-with-body -sS "http://$MASTER_ADDR:8000/v1/models" \
    >"$OUT_ROOT/E4-control-4096/models.json"
  fixed_request E4-control-4096 "$OUT_ROOT/correctness/E4-control-4096.json"
  run_bench E4-control-4096
  capture_profile_head \
    E4-control-4096 \
    "$OUT_ROOT/correctness/E4-control-4096-profile.json"
fi
barrier E4-control-4096-profile-stopped 1200
collect_local_profile E4-control-4096
finish_event E4-control-4096

start_server E6-sp4-4096 4
if test "$RANK" = 0; then
  curl --fail-with-body -sS "http://$MASTER_ADDR:8000/v1/models" \
    >"$OUT_ROOT/E6-sp4-4096/models.json"
  fixed_request E6-sp4-4096 "$OUT_ROOT/correctness/E6-sp4-4096.json"
  run_bench E6-sp4-4096
  capture_profile_head \
    E6-sp4-4096 \
    "$OUT_ROOT/correctness/E6-sp4-4096-profile.json"
fi
barrier E6-sp4-4096-profile-stopped 1200
collect_local_profile E6-sp4-4096
finish_event E6-sp4-4096

if test "$RANK" = 0; then
  python - "$OUT_ROOT" "$TORCHTPU_COMMIT" <<'PY'
import json
import math
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
source_commit = sys.argv[2]
control = json.loads((out / "correctness" / "E4-control-4096.json").read_text())
sp = json.loads((out / "correctness" / "E6-sp4-4096.json").read_text())
control_choice = control["response"]["choices"][0]
sp_choice = sp["response"]["choices"][0]
if control_choice["text"] != sp_choice["text"]:
    raise SystemExit(
        f"sampled token mismatch: {control_choice['text']!r} != {sp_choice['text']!r}"
    )
control_lp = control_choice["logprobs"]
sp_lp = sp_choice["logprobs"]
control_top = control_lp["top_logprobs"][0]
sp_top = sp_lp["top_logprobs"][0]
if set(control_top) != set(sp_top):
    raise SystemExit(f"final top-20 token set mismatch: {set(control_top) ^ set(sp_top)}")
deltas = {
    token: abs(float(control_top[token]) - float(sp_top[token]))
    for token in control_top
}
max_delta = max(deltas.values(), default=0.0)
if not math.isfinite(max_delta) or max_delta > 0.05:
    raise SystemExit(f"final top-20 logprob delta={max_delta}")
correctness = {
    "sampled_token": control_choice["text"],
    "sampled_logprob_control": control_lp["token_logprobs"][0],
    "sampled_logprob_sp4": sp_lp["token_logprobs"][0],
    "max_abs_top20_logprob_delta": max_delta,
    "top20_logprob_deltas": deltas,
    "pass_tolerance": 0.05,
}
(out / "correctness-summary.json").write_text(
    json.dumps(correctness, ensure_ascii=False, indent=2) + "\n"
)

metrics = []
for event in ("E4-control-4096", "E6-sp4-4096"):
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
(out / "manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "workflow": "kimi-k3-prefill-sp-phase2",
    "model": "moonshotai/Kimi-K3",
    "model_revision": "a590ce090cb049c93a33dfe8c208ec652aa20503",
    "source_commit": source_commit,
    "vllm_commit": "4bdc8a788d2e2ce9165d552b3d4d8b72604626bf",
    "torch_tpu": "0.1.1.dev20260813160135",
    "hardware": "v7x32",
    "logical_mesh": "SP4 x lane8",
    "rank_mapping": "tp_rank = lane * 4 + sp_rank",
    "events": {
        "E4": {"sp": 1, "tokens": 4096, "profile": True},
        "E5": {"status": "optional-skipped"},
        "E6": {"sp": 4, "tokens": 4096, "profile": True},
    },
}, indent=2) + "\n")
(out / "phase2-summary.json").write_text(json.dumps({
    "status": "passed",
    "correctness": correctness,
    "metrics": metrics,
    "profiles": ["E4-control-4096", "E6-sp4-4096"],
    "hybrid_moe": False,
}, ensure_ascii=False, indent=2) + "\n")
PY
  printf 'KIMI_K3_PREFILL_SP_PHASE2_OK\n' | tee "$OUT_ROOT/success.marker"
fi

barrier phase2-complete 600
