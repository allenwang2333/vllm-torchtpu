#!/bin/bash

# shellcheck disable=all
set -e

DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run|--dryrun|-d)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done



wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}
# Function to print logs on exit
print_logs_on_exit() {
  echo "--- Script exiting, displaying logs ---"

  # The logs are written inside containers to /root/logs, which is mapped from $LOG_DIR on the host.
  LOG_DIR=$HOME/logs

  if [ -d "$LOG_DIR" ]; then
    echo "--- Contents of $LOG_DIR/prefill_0.txt ---"
    if [ -f "$LOG_DIR/prefill_0.txt" ]; then
      cat "$LOG_DIR/prefill_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/decode_0.txt ---"
    if [ -f "$LOG_DIR/decode_0.txt" ]; then
      cat "$LOG_DIR/decode_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/benchmark_0.txt ---"
    if [ -f "$LOG_DIR/benchmark_0.txt" ]; then
      cat "$LOG_DIR/benchmark_0.txt"
    else
      echo "File not found."
    fi

    echo "--- Contents of $LOG_DIR/proxy_0.txt ---"
    if [ -f "$LOG_DIR/proxy_0.txt" ]; then
      cat "$LOG_DIR/proxy_0.txt"
    else
      echo "File not found."
    fi
  else
    echo "Log directory '$LOG_DIR' not found."
  fi
  echo "--- End of logs ---"
}

check_failed_requests() {
  local log_file="$1"
  local failed_requests
  failed_requests=$(grep "Failed requests:" "$log_file" | awk '{print $3}' || true)

  if [ -z "$failed_requests" ]; then
    echo "Error: Could not find 'Failed requests:' in the benchmark output." >&2
    return 1
  fi

  if [ "$failed_requests" -gt 0 ]; then
    echo "Error: Benchmark reported $failed_requests failed requests." >&2
    return 1
  fi

  echo "Success: Benchmark reported $failed_requests failed requests." >&2
  return 0
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm" || true
  pkill -f "toy_proxy_server" || true
  sleep 5
  pkill -9 -f "vllm" || true
  pkill -9 -f "toy_proxy_server" || true
  fuser -k -9 /dev/vfio/* || true
  fuser -k -9 /dev/accel* || true
  rm -rf /tmp/jax_cache_* || true
  rm -f /tmp/libtpu_lockfile || true
  rm -f /dev/shm/tpu_conn_* || true
}

MODEL=${MODEL:="Qwen/Qwen3-0.6B"}
INPUT_LEN=${INPUT_LEN:=512}
OUTPUT_LEN=${OUTPUT_LEN:=128}
NUM_PROMPTS=${NUM_PROMPTS:=200}
REQUEST_RATE=${REQUEST_RATE:=4}

NUM_PREFILL_INSTANCES=1
NUM_DECODE_INSTANCES=1
PREFILLER_TP_SIZE=1
DECODER_TP_SIZE=1

PREFILL_HOSTS=()
PREFILL_PORTS=()
DECODE_HOSTS=()
DECODE_PORTS=()

# Parse allocated chip indices
if [ -n "$TPU_VISIBLE_CHIPS" ]; then
  IFS=',' read -ra ALLOCATED_CHIPS <<< "$TPU_VISIBLE_CHIPS"
  FOUND_CHIPS=${#ALLOCATED_CHIPS[@]}
else
  # Retrieve per-chip vfio device paths (e.g. /dev/vfio/1) from `tpu-info`.
  TPU_DEVICE_PATHS=($(tpu-info 2>/dev/null | awk '/^\| \/dev\/vfio/ {print $2}'))
  if [ ${#TPU_DEVICE_PATHS[@]} -eq 0 ]; then
    # Fallback to listing /dev/vfio/ entries if tpu-info is not installed or returned nothing.
    # On 2-cores-per-chip systems (v4, v5p, v7), sequential dev numbers (e.g. 0 and 1) belong to the same chip.
    # We only take even numbered device files to ensure we don't allocate different processes to the same physical chip.
    if [ -d /dev/vfio ]; then
      for f in /dev/vfio/[0-9]*; do
        if [ -e "$f" ]; then
          dev_num=$(basename "$f")
          if (( dev_num % 2 == 0 )); then
            TPU_DEVICE_PATHS+=("$f")
          fi
        fi
      done
    fi
  fi
  FOUND_CHIPS=${#TPU_DEVICE_PATHS[@]}
  ALLOCATED_CHIPS=()
  for i in $(seq 0 $((FOUND_CHIPS-1))); do
    ALLOCATED_CHIPS+=("$i")
  done
fi

REQUIRED_CHIPS=$((NUM_PREFILL_INSTANCES + NUM_DECODE_INSTANCES))

if [ "$DRY_RUN" = true ]; then
  echo "--- Dry Run: Checking TPU Device Setup ---"
fi

if [ -n "$TPU_VISIBLE_CHIPS" ]; then
  echo "TPU_VISIBLE_CHIPS: ${TPU_VISIBLE_CHIPS}"
else
  echo "TPU_DEVICE_PATHS: ${TPU_DEVICE_PATHS[*]}"
fi
echo "Number of TPU chips found: $FOUND_CHIPS"
echo "Number of TPU chips required: $REQUIRED_CHIPS"

if [ "$FOUND_CHIPS" -lt "$REQUIRED_CHIPS" ]; then
  echo "Error: Not enough TPU chips. Found $FOUND_CHIPS, but need at least $REQUIRED_CHIPS."
  if [ "$DRY_RUN" = false ]; then
    exit 1
  fi
else
  if [ "$DRY_RUN" = true ]; then
    echo "TPU device setup check PASSED."
  fi
fi

if [ "$DRY_RUN" = false ]; then
  # Register the cleanup function to be called on script exit (normal or error)
  trap print_logs_on_exit EXIT

  LOG_DIR=$HOME/logs

  if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
  else
    # Delete old log files to avoid printing stale logs at the end
    rm -f $LOG_DIR/prefill_0.txt $LOG_DIR/decode_0.txt $LOG_DIR/benchmark_0.txt $LOG_DIR/proxy_0.txt
  fi

  cleanup_instances
fi

# Start prefill instances
for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    PORT=$((8400 + i))
    KV_PORT=$((7100 + i))
    SIDE_PORT=$((6100 + i))
    CHIP_IDX=$i
    TARGET_CHIP=${ALLOCATED_CHIPS[$CHIP_IDX]}

    cmd=(
      env
      TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
      TPU_PROCESS_BOUNDS=1,1,1
      TPU_VISIBLE_CHIPS="$TARGET_CHIP"
      TPU_KV_TRANSFER_PORT="$KV_PORT"
      TPU_SIDE_CHANNEL_PORT="$SIDE_PORT"
      SKIP_JAX_PRECOMPILE=1
      vllm serve "$MODEL"
      --port "$PORT"
      --gpu-memory-utilization 0.2
      --tensor-parallel-size "$PREFILLER_TP_SIZE"
      --attention-backend CUSTOM
      --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_producer\"}"
    )

    echo "Prefill instance $i command:"
    echo "  ${cmd[*]}"

    if [ "$DRY_RUN" = false ]; then
      "${cmd[@]}" > "$LOG_DIR/prefill_$i.txt" 2>&1 &

      echo "Waiting for prefill on port $PORT to start..."
      wait_for_server "$PORT"

      echo "Scanning and unlinking locked VFIO device nodes..."
      for fd in /proc/*/fd/*; do
        link=$(readlink "$fd" 2>/dev/null || true)
        if [[ "$link" == /dev/vfio/* && "$link" != "/dev/vfio/vfio" ]]; then
          if [ -w "$link" ] 2>/dev/null; then
            echo "Unlinking busy device node: $link"
            rm -f "$link" 2>/dev/null || true
          fi
        fi
      done
    fi

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=("$PORT")
done


# Start decode instances
for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    PORT=$((9400 + i))
    KV_PORT=$((7200 + i))
    # Same as prefill SIDE_PORT
    SIDE_PORT=$((6100 + i))
    CHIP_IDX=$((NUM_PREFILL_INSTANCES + i))
    TARGET_CHIP=${ALLOCATED_CHIPS[$CHIP_IDX]}

    cmd=(
      env
      TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
      TPU_PROCESS_BOUNDS=1,1,1
      TPU_VISIBLE_CHIPS="$TARGET_CHIP"
      TPU_KV_TRANSFER_PORT="$KV_PORT"
      TPU_SIDE_CHANNEL_PORT="$SIDE_PORT"
      SKIP_JAX_PRECOMPILE=1
      vllm serve "$MODEL"
      --port "$PORT"
      --gpu-memory-utilization 0.2
      --tensor-parallel-size "$DECODER_TP_SIZE"
      --attention-backend CUSTOM
      --kv-transfer-config "{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_consumer\"}"
    )

    echo "Decode instance $i command:"
    echo "  ${cmd[*]}"

    if [ "$DRY_RUN" = false ]; then
      "${cmd[@]}" > "$LOG_DIR/decode_$i.txt" 2>&1 &

      echo "Waiting for decode on port $PORT to start..."
      wait_for_server "$PORT"

      echo "Scanning and unlinking locked VFIO device nodes..."
      for fd in /proc/*/fd/*; do
        link=$(readlink "$fd" 2>/dev/null || true)
        if [[ "$link" == /dev/vfio/* && "$link" != "/dev/vfio/vfio" ]]; then
          if [ -w "$link" ] 2>/dev/null; then
            echo "Unlinking busy device node: $link"
            rm -f "$link" 2>/dev/null || true
          fi
        fi
      done
    fi

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=("$PORT")
done


# Start proxy server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd=(
  python "${SCRIPT_DIR}/toy_proxy_server.py"
  --host localhost
  --port 8000
  --prefiller-hosts "${PREFILL_HOSTS[@]}"
  --prefiller-ports "${PREFILL_PORTS[@]}"
  --decoder-hosts "${DECODE_HOSTS[@]}"
  --decoder-ports "${DECODE_PORTS[@]}"
)

echo "Proxy server command:"
echo "  ${cmd[*]}"

if [ "$DRY_RUN" = false ]; then
  "${cmd[@]}" > "$LOG_DIR/proxy_0.txt" 2>&1 &
fi

cmd=(
  vllm bench serve
  --model="$MODEL"
  --num-warmups=3
  --dataset-name=random
  --random-input-len="${INPUT_LEN}"
  --random-output-len="${OUTPUT_LEN}"
  --num-prompts="${NUM_PROMPTS}"
  --ignore-eos
  --host=localhost
  --port 8000
  --request-rate="${REQUEST_RATE}"
)

echo "Benchmark serve command:"
echo "  ${cmd[*]}"

if [ "$DRY_RUN" = false ]; then
  # run benchmark for both disagg and non-disagg
  LOG_FILE="$LOG_DIR/benchmark_0.txt"
  echo "--- Running Disagg Benchmark ---" > "$LOG_FILE"

  # run ben for disagg
  set -x
  "${cmd[@]}" >> "$LOG_FILE" 2>&1
  set +x

  check_failed_requests "$LOG_FILE"

  cat <<'EOF'
The proxy server has been launched on: 127.0.0.1:7080

>> Send example request:

curl -X POST \
http://127.0.0.1:7080/v1/completions \
-H "Content-Type: application/json" \
-d '{"prompt": "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed,  ", "max_tokens": 10}'

>> Stop the proxy server and all prefill/decode instances:

pkill -f "vllm serve" && pkill -f "toy_proxy_server" && pkill -f "run_disagg_single_host"
EOF
fi
