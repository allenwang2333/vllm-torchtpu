#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

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
export K3_TEST_MLA_AUDIT=1
export VLLM_TARGET_DEVICE=empty
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,4
export TORCH_TPU_TOPOLOGY=2,2,4,2
export JAX_PLATFORMS=""
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_TORCHTPU_KDA_FUSED=1
export VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL=mega
export TPU_TOKEN_BUCKET_EXTRA=1,2,4,6,8,12,16,24,32,48,64,96,128,192,256,512
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_MOE_SKIP_PADDED_TOKENS=1
unset DP_SCHED_ENABLED || true
unset USE_PHASED_PROFILER || true

MODEL_ID=moonshotai/Kimi-K3
MODEL_PATH=/models/Kimi-K3-a590ce09
MODEL_REVISION=a590ce090cb049c93a33dfe8c208ec652aa20503
BLOCK5_PATH=/models/Inferact/Kimi-K3-DSpark-Block5
BLOCK7_PATH=/models/Inferact/Kimi-K3-DSpark
WHEEL_PATH=/wheels/rank-0/torch_tpu-0.1.1.dev20260813160135-cp312-cp312-manylinux_2_31_x86_64.whl
TORCHTPU_COMMIT="${EXPECTED_TORCHTPU_COMMIT:?}"
VLLM_COMMIT=4bdc8a788d2e2ce9165d552b3d4d8b72604626bf
RUNBOOK_COMMIT=de2c67b2a9ad20746d78340c881f98a87e51417c
RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
WORLD="${FALCON_WORLD_SIZE:-4}"
MASTER_ADDR="${FALCON_JAX_COORDINATOR_ADDRESS%:*}"
OUT_ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
OUT="$OUT_ROOT/rank-$RANK"
VENV=/tmp/tpu_logs/venv
SERVER_PID=0
TAIL_PID=0
SERVER_LOG=""

if test "$WORLD" != 4 || test "$RANK" -lt 0 || test "$RANK" -gt 3; then
  echo "UNEXPECTED_FALCON_TOPOLOGY rank=$RANK world=$WORLD" >&2
  exit 1
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR" "$VLLM_CACHE_ROOT" \
  "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" "$HF_HOME" "$OUT"
exec > >(tee -a "$OUT/task.log") 2>&1
echo "FIXED_TOTAL_32_DEVICES rank=$RANK world=$WORLD master=$MASTER_ADDR"
getent hosts "$MASTER_ADDR"

for i in $(seq 1 60); do
  test -s "$WHEEL_PATH" && break
  echo "WAITING_FOR_WHEEL attempt=$i"
  sleep 5
done
test -s "$WHEEL_PATH"
test "$(git -C /workspace/vllm-torchtpu rev-parse HEAD)" = "$TORCHTPU_COMMIT"
test "$(git -C /workspace/vllm rev-parse HEAD)" = "$VLLM_COMMIT"
test "$(git -C /workspace/k3-launch-scripts rev-parse HEAD)" = "$RUNBOOK_COMMIT"
test -s "$MODEL_PATH/config.json"
test -s "$BLOCK5_PATH/config.json"
test -s "$BLOCK7_PATH/config.json"

HUB_MODEL_ROOT="$HF_HUB_CACHE/models--moonshotai--Kimi-K3"
mkdir -p "$HUB_MODEL_ROOT/refs" "$HUB_MODEL_ROOT/snapshots"
ln -sfn "$MODEL_PATH" "$HUB_MODEL_ROOT/snapshots/$MODEL_REVISION"
printf '%s\n' "$MODEL_REVISION" > "$HUB_MODEL_ROOT/refs/main"

python3 -m venv "$VENV"
# shellcheck source=/dev/null
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
skip = ("vllm ", "torch==", "torchvision==", "torch-tpu==", "tpu-raiden-torch==")
deps = [dep for dep in project["project"]["dependencies"] if not dep.startswith(skip)]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--pre", *deps])
PY
VLLM_TARGET_DEVICE=empty python -m pip install -e /workspace/vllm
python -m pip install --no-deps -e /workspace/vllm-torchtpu

python - /workspace/vllm/vllm/transformers_utils/configs/k3_dspark.py <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
replacements = {
    "from transformers import DeepseekV2Config": "from transformers import DeepseekV3Config",
    "class K3DSparkConfig(DeepseekV2Config):": "class K3DSparkConfig(DeepseekV3Config):",
    "DeepseekV2Config defaults to a MoE topology": "DeepseekV3Config defaults to a MoE topology",
}
for old, new in replacements.items():
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
path.write_text(source)
print("UPSTREAM_PR_54005_K3_CONFIG_BACKPORT_APPLIED")
PY

python - <<'PY'
from transformers import DeepseekV3Config
from vllm.transformers_utils.configs.k3_dspark import K3DSparkConfig

assert issubclass(K3DSparkConfig, DeepseekV3Config)
print("UPSTREAM_K3_DSPARK_CONFIG_FIX_PRESENT")
PY

export MODEL_PATH BLOCK5_PATH BLOCK7_PATH
python - <<'PY' | tee "$OUT/model-validation.json"
import json
import os
from pathlib import Path

result = {}
for label, env, expected in (("block5", "BLOCK5_PATH", 5), ("block7", "BLOCK7_PATH", 7)):
    root = Path(os.environ[env])
    cfg = json.loads((root / "config.json").read_text())
    configured_block_size = cfg.get("block_size")
    effective_block_size = expected if configured_block_size is None else configured_block_size
    result[label] = {
        "path": str(root),
        "model_type": cfg.get("model_type"),
        "configured_block_size": configured_block_size,
        "effective_block_size": effective_block_size,
        "weight_files": len(list(root.glob("*.safetensors"))),
    }
    assert result[label]["model_type"] == "k3_dspark"
    assert result[label]["effective_block_size"] == expected
    assert result[label]["weight_files"] > 0
print(json.dumps(result, indent=2, sort_keys=True))
PY

python - <<'PY' | tee "$OUT/runtime-versions.txt"
import importlib.metadata as md
import torch
import torch_tpu

for package in ("torch", "torch-tpu", "vllm", "vllm-torchtpu", "libtpu"):
    try:
        print(f"{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}=MISSING")
assert torch.tpu.device_count() == 8
print("LOCAL_TPU_DEVICES=8")
PY

if test "$RANK" = 0; then
  mkdir -p "$OUT/reproduction"
  cp /workspace/k3-launch-scripts/scripts/k3/bench_isl1_osl1k.sh "$OUT/reproduction/"
  cp /workspace/k3-launch-scripts/scripts/k3/profile_isl1_osl1k.sh "$OUT/reproduction/"
  cp /workspace/k3-launch-scripts/scripts/k3/start_profile.sh "$OUT/reproduction/"
  cp /workspace/k3-launch-scripts/scripts/k3/stop_profile.sh "$OUT/reproduction/"
fi

stop_server() {
  set +e
  if test "$TAIL_PID" -gt 0; then
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    TAIL_PID=0
  fi
  if test "$SERVER_PID" -gt 0; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=0
  fi
  set -e
}
cleanup() {
  stop_server
  if test -n "$SERVER_LOG" && test -f "$SERVER_LOG"; then
    cp "$SERVER_LOG" "$OUT/" || true
  fi
}
trap cleanup EXIT

wait_ready() {
  local label="$1"
  local ready=0
  for i in $(seq 1 1440); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "SERVER_DIED_BEFORE_READY label=$label rank=$RANK" >&2
      return 1
    fi
    if curl --connect-timeout 2 --max-time 5 -sf "http://$MASTER_ADDR:8000/health" >/dev/null; then
      echo "SERVER_READY label=$label rank=$RANK elapsed_seconds=$((i * 10))"
      ready=1
      break
    fi
    sleep 10
  done
  test "$ready" = 1
}

wait_marker() {
  local marker="$1"
  for _ in $(seq 1 8640); do
    test -f "$marker" && return 0
    if test "$SERVER_PID" -gt 0 && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "SERVER_DIED_WAITING marker=$marker rank=$RANK" >&2
      return 1
    fi
    sleep 5
  done
  return 1
}

finish_service() {
  local label="$1"
  local stopped="/coord/${label}_STOPPED_${RANK}"
  local released="/coord/${label}_RELEASED"
  stop_server
  touch "$stopped"
  if test "$RANK" = 0; then
    for _ in $(seq 1 600); do
      local all=1
      for rank in 0 1 2 3; do
        test -f "/coord/${label}_STOPPED_${rank}" || all=0
      done
      test "$all" = 1 && break
      sleep 2
    done
    test "$all" = 1
    touch "$released"
  else
    for _ in $(seq 1 600); do
      test -f "$released" && break
      sleep 2
    done
    test -f "$released"
  fi
}


parse_point() {
  local result="$1" before="$2" after="$3" output="$4" label="$5"
  local concurrency="$6" prompts="$7" configured_acc="$8" speculative="$9"
  python - "$result" "$before" "$after" "$output" "$label" \
    "$concurrency" "$prompts" "$configured_acc" "$speculative" <<'PY'
import json
import re
import sys
from pathlib import Path

result_path, before_path, after_path, output_path = map(Path, sys.argv[1:5])
label = sys.argv[5]
concurrency, prompts = map(int, sys.argv[6:8])
configured_acc = float(sys.argv[8])
speculative = sys.argv[9] == "1"

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

raw = json.loads(result_path.read_text())
before, after = counters(before_path), counters(after_path)
steps = after.get("vllm:spec_decode_num_drafts", 0) - before.get("vllm:spec_decode_num_drafts", 0)
drafted = after.get("vllm:spec_decode_num_draft_tokens", 0) - before.get("vllm:spec_decode_num_draft_tokens", 0)
accepted = after.get("vllm:spec_decode_num_accepted_tokens", 0) - before.get("vllm:spec_decode_num_accepted_tokens", 0)
acc_len = 1.0 + accepted / steps if steps else None
if speculative:
    if acc_len is None or abs(acc_len - configured_acc) > 0.08:
        raise RuntimeError(f"acceptance length={acc_len}, expected={configured_acc}")
else:
    if steps or drafted or accepted:
        raise RuntimeError(f"unexpected speculative counters: {steps}, {drafted}, {accepted}")
if int(raw.get("completed", -1)) != prompts:
    raise RuntimeError(f"completed={raw.get('completed')}, expected={prompts}")
assert raw["total_input_tokens"] == prompts, raw
assert raw["total_output_tokens"] == 1000 * prompts, raw
assert raw.get("output_throughput", 0) > 0 and raw.get("mean_tpot_ms", 0) > 0, raw
row = {
    "label": label,
    "devices_total": 32,
    "data_parallel_size": 1,
    "tensor_parallel_size": 32,
    "expert_parallel_size": 32,
    "draft_tensor_parallel_size": 32 if speculative else None,
    "target_verify_tokens_per_sequence": 6 if speculative else 1,
    "concurrency": concurrency,
    "num_prompts": prompts,
    "num_warmups": 2 * concurrency,
    "input_tokens_per_request": 1,
    "output_tokens_per_request": 1000,
    "output_throughput_tok_s": raw.get("output_throughput"),
    "request_throughput_req_s": raw.get("request_throughput"),
    "mean_ttft_ms": raw.get("mean_ttft_ms"),
    "median_ttft_ms": raw.get("median_ttft_ms"),
    "p99_ttft_ms": raw.get("p99_ttft_ms"),
    "mean_tpot_ms": raw.get("mean_tpot_ms"),
    "median_tpot_ms": raw.get("median_tpot_ms"),
    "p99_tpot_ms": raw.get("p99_tpot_ms"),
    "mean_itl_ms": raw.get("mean_itl_ms"),
    "median_itl_ms": raw.get("median_itl_ms"),
    "p99_itl_ms": raw.get("p99_itl_ms"),
    "mean_e2el_ms": raw.get("mean_e2el_ms"),
    "median_e2el_ms": raw.get("median_e2el_ms"),
    "p99_e2el_ms": raw.get("p99_e2el_ms"),
    "draft_steps": steps,
    "draft_tokens": drafted,
    "accepted_draft_tokens": accepted,
    "measured_acceptance_length": acc_len,
    "configured_acceptance_length": configured_acc if speculative else None,
    "raw_result_file": str(result_path),
    "repetition": int(result_path.parent.name[4:]),
}
output_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
print("PERF_POINT " + json.dumps(row, sort_keys=True), flush=True)
PY
}

run_benchmark_sweep() {
  local label="$1" configured_acc="$2" speculative="$3"
  local run_out="$OUT/performance/$label"
  mkdir -p "$run_out/benchmark"
  curl --fail-with-body -sS "http://$MASTER_ADDR:8000/v1/models" > "$run_out/models.json"
  for concurrency in 1 2 4 8 16 32; do
    for repetition in 1 2 3; do
      local point="$run_out/benchmark/c_${concurrency}/rep_${repetition}"
      local prompts=$((4 * concurrency))
      local warmups=$((2 * concurrency))
      mkdir -p "$point"
      curl -fsS "http://$MASTER_ADDR:8000/metrics" > "$point/metrics-before.prom"
      local filename="kimi-k3_isl1_osl1000_c${concurrency}.json"
      "$VENV/bin/vllm" bench serve \
        --backend vllm \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_ID" \
        --tokenizer "$MODEL_PATH" \
        --trust-remote-code \
        --base-url "http://$MASTER_ADDR:8000" \
        --dataset-name random \
        --random-input-len 0 \
        --random-prefix-len 1 \
        --random-output-len 1000 \
        --random-range-ratio 0 \
        --num-prompts "$prompts" \
        --num-warmups "$warmups" \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --temperature 0 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir "$point" \
        --result-filename "$filename" \
        --seed 980406 2>&1 | tee "$point/bench.log"
      curl -fsS "http://$MASTER_ADDR:8000/metrics" > "$point/metrics-after.prom"
      parse_point "$point/$filename" "$point/metrics-before.prom" \
        "$point/metrics-after.prom" "$point/summary.json" "$label" \
        "$concurrency" "$prompts" "$configured_acc" "$speculative"
    done
  done
}

capture_profile() {
  local label="$1" concurrency="$2" profile_root="$3"
  local run_out="$OUT/performance/$label/profiling/c_${concurrency}"
  local marker="/tmp/tpu_logs/${label}-c${concurrency}-profile.marker"
  mkdir -p "$run_out"
  python - "$MASTER_ADDR" "$run_out" "$concurrency" "$marker" <<'PY'
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

base = "http://" + sys.argv[1] + ":8000"
out = Path(sys.argv[2])
concurrency = int(sys.argv[3])
marker = Path(sys.argv[4])

def post(path, body=None, timeout=14400):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else b"",
        headers={"Content-Type": "application/json", "X-data-parallel-rank": "0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()

body = {
    "model": "moonshotai/Kimi-K3",
    "prompt": [1000],
    "max_tokens": 1000,
    "min_tokens": 1000,
    "ignore_eos": True,
    "temperature": 0,
    "seed": 980406,
}

def issue_one(index):
    response = json.loads(post("/v1/completions", body))
    assert response["usage"]["prompt_tokens"] == 1, response["usage"]
    assert response["usage"]["completion_tokens"] == 1000, response["usage"]
    (out / f"response-{index}.json").write_text(json.dumps(response))

def issue_batch():
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(issue_one, range(concurrency)))

issue_batch()
marker.touch()
post("/start_profile")
try:
    issue_batch()
finally:
    post("/stop_profile")
print(f"PROFILE_REQUEST_COMPLETE concurrency={concurrency}", flush=True)
PY
  local plane=""
  for _ in $(seq 1 240); do
    plane="$(find "$profile_root/decode_only" -type f -name 'rank0_*.xplane.pb' -newer "$marker" -print 2>/dev/null | sort | tail -1 || true)"
    test -n "$plane" && break
    sleep 5
  done
  test -n "$plane" && test -s "$plane"
  local dest="$run_out/rank0-${label}-c${concurrency}-profile15.xplane.pb"
  cp "$plane" "$dest"
  sha256sum "$dest" > "$dest.sha256"
  echo "PROFILE_CAPTURED label=$label concurrency=$concurrency path=$dest"
}

summarize_sweep() {
  local label="$1" configured_acc="$2" speculative="$3"
  local run_out="$OUT/performance/$label"
  python - "$run_out" "$label" "$configured_acc" "$speculative" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]
configured_acc = float(sys.argv[3])
speculative = sys.argv[4] == "1"
rows = [json.loads(path.read_text()) for path in sorted(
    root.glob("benchmark/c_*/rep_*/summary.json"),
    key=lambda p: (int(p.parent.parent.name[2:]), int(p.parent.name[4:])),
)]
assert len(rows) == 18, len(rows)
profiles = sorted(str(p) for p in root.glob("profiling/c_*/*.xplane.pb"))
assert len(profiles) == 3, profiles
summary = {
    "label": label,
    "devices_total": 32,
    "data_parallel_size": 1,
    "tensor_parallel_size": 32,
    "expert_parallel_size": 32,
    "configured_acceptance_length": configured_acc if speculative else None,
    "profile_concurrency": [1, 8, 16],
    "profile_active_decode_steps": 15,
    "profiles": profiles,
    "results": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (root / "summary.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("PERF_SWEEP_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
PY
}

run_perf_rank0() {
  local label="$1" configured_acc="$2" speculative="$3" profile_root="$4"
  run_benchmark_sweep "$label" "$configured_acc" "$speculative"
  grep -F "No tuned parameters found for the given tuning key" "$SERVER_LOG" \
    > "$OUT/performance/$label/mla-tuning-fallbacks.log" || true
  grep -E "TuningKey\(case='(batched_decode|mixed)'" "$SERVER_LOG" \
    > "$OUT/performance/$label/mla-tuning-keys.log" || true
  grep -F "MLA_LOOKUP_AUDIT" "$SERVER_LOG" > "$OUT/performance/$label/mla-lookup-audit.log" || true
  for concurrency in 1 8 16; do
    capture_profile "$label" "$concurrency" "$profile_root"
  done
  summarize_sweep "$label" "$configured_acc" "$speculative"
}

run_service() {
  local label="$1" mode="$2" acceptance="$3" master_port="$4"
  local done="/coord/${label}_DONE"
  local profile_root="/tmp/profile-${label}"
  local speculative=0
  SERVER_LOG="/tmp/tpu_logs/server-${label}-rank${RANK}.log"
  if test "$RANK" = 0; then
    test ! -e "$done"
  fi

  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=""
  export TORCH_TPU_DP_MASTER_ADDR="$MASTER_ADDR"
  export TORCH_TPU_DP_MASTER_PORT="$master_port"
  compile_config='{"compile_sizes":[1,2,4,8,16,32]}'
  if test "$mode" = synthetic; then
    compile_config='{"compile_sizes":[8,16,32,48,64,96,128,192,256,512]}'
  fi
  export JAX_COMPILATION_CACHE_DIR="/tmp/tpu_logs/jax-cache-$label"
  export VLLM_CACHE_ROOT="/tmp/tpu_logs/vllm-cache-$label"
  export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/tmp/tpu_logs/tier3-$label"
  serve_args=(
    "$MODEL_PATH"
    --served-model-name "$MODEL_ID"
    --tensor-parallel-size 32
    --data-parallel-size 1
    --data-parallel-backend mp
    --enable-expert-parallel
    --enable-ep-weight-filter
    --nnodes 4
    --node-rank "$RANK"
    --master-addr "$MASTER_ADDR"
    --master-port "$master_port"
    --quantization compressed-tensors
    --load-format runai_streamer
    --model-loader-extra-config '{"concurrency":16,"memory_limit":8589934592}'
    --dtype bfloat16
    --kv-cache-dtype fp8
    --attention-backend CUSTOM
    --async-scheduling
    --max-model-len 32768
    --max-num-seqs 32
    --max-num-batched-tokens 512
    --block-size 16
    --gpu-memory-utilization 0.82
    --no-enable-prefix-caching
    --language-model-only
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --trust-remote-code
    --tokenizer "$MODEL_PATH"
    --tokenizer-mode kimi_k3
    --compilation-config "$compile_config"
  )
  if test "$mode" = synthetic; then
    speculative=1
    spec_json="$(python - "$BLOCK5_PATH" "$acceptance" <<'PY'
import json
import sys
print(json.dumps({
    "method": "dspark",
    "model": sys.argv[1],
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 32,
    "draft_sample_method": "greedy",
    "rejection_sample_method": "synthetic",
    "synthetic_acceptance_length": float(sys.argv[2]),
}, separators=(",", ":")))
PY
)"
    serve_args+=(--speculative-config "$spec_json")
  elif test "$mode" = standard; then
    speculative=1
    spec_json='{"method":"dspark","model":"/models/Inferact/Kimi-K3-DSpark-Block5","num_speculative_tokens":5,"draft_tensor_parallel_size":32,"draft_sample_method":"greedy","rejection_sample_method":"standard"}'
    serve_args+=(--speculative-config "$spec_json")
  fi

  if test "$mode" = standard; then
    unset USE_PHASED_PROFILER || true
  else
    export USE_PHASED_PROFILER=true
    serve_args+=(
      --profiler-config.profiler torch
      --profiler-config.torch_profiler_dir "$profile_root"
      --profiler-config.ignore_frontend true
      --profiler-config.delay_iterations 3
      --additional-config '{"phased_profiler_decode_only_kv_len_threshold":32,"phased_profiler_prefill_only_kv_len_threshold":1000000000}'
      --profiler-config.max_iterations 15
    )
  fi

  export VLLM_TARGET_DEVICE=tpu
  if test "$RANK" = 0; then
    vllm serve "${serve_args[@]}" --host 0.0.0.0 --port 8000 > "$SERVER_LOG" 2>&1 &
  else
    vllm serve "${serve_args[@]}" --headless > "$SERVER_LOG" 2>&1 &
  fi
  SERVER_PID=$!
  tail -n +1 -F "$SERVER_LOG" &
  TAIL_PID=$!
  wait_ready "$label"
  if test "$RANK" = 0; then
    run_perf_rank0 "$label" "$acceptance" "$speculative" "$profile_root"
    touch "$done"
  else
    wait_marker "$done"
  fi
  finish_service "$label"
  cp "$SERVER_LOG" "$OUT/server-${label}-rank${RANK}.log" || true
}

if test "${K3_SERVING_MODE:?}" = no_dspark; then
  export K3_TEST_EXACT_DECODE_KEY=0
  run_service "no_dspark_pr_lookup" "none" "0" "29500"
  export K3_TEST_EXACT_DECODE_KEY=1
  run_service "no_dspark_exact_lookup" "none" "0" "29510"
else
  export K3_TEST_EXACT_DECODE_KEY=0
  run_service "dspark_al4_pr_lookup" "synthetic" "4" "29500"
fi
echo "ALL_REQUESTED_WORK_COMPLETE mode=$K3_SERVING_MODE"
