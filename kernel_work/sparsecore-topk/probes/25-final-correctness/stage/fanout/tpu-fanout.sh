#!/usr/bin/env bash
# tpu-fanout.sh — run several commands concurrently on ONE simple-run-command pod,
# each pinned to its own TPU chip.
#
# The pod is always 4 chips (tpu7x, 2x2x1, 8 cores). A single-chip command leaves
# three chips idle for the life of the job. This script splits the pod.
#
# Usage:
#   tpu-fanout.sh [OPTIONS] CMD [CMD ...]
#   tpu-fanout.sh [OPTIONS] -e "TEMPLATE containing {}" ARG [ARG ...]
#
# Options:
#   -e T   template mode: each remaining ARG is substituted into every {} in T.
#   -o DIR where per-shard logs go (default $CDK_OUTPUT_DIR/fanout, else /tmp/fanout).
#   -c N   chips per shard. Only 1 works; see the refusal below for why.
#   -h     this help.
#
# One chip per shard, so at most 4 shards. A chip is 2 TPU cores, so a shard can
# still run a 2-core workload; it cannot run one that needs 4 chips.
#
# Each shard runs under `bash -c` with:
#   TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES  -> its own chip index
#   ALLOW_MULTIPLE_LIBTPU_LOAD=1
#   TPU_LOG_DIR, VLLM_CACHE_ROOT, VLLM_XLA_CACHE_PATH, TMPDIR  -> per-shard paths
#   TPU_FANOUT_SHARD / TPU_FANOUT_NUM_SHARDS / TPU_FANOUT_CHIPS
#
# Output: shards are captured to $OUTDIR/shard<i>.log, not interleaved on stdout.
# A heartbeat names the still-running shards every 60s; every log is replayed at
# the end under a banner, followed by an EXIT CODE table.
#
# Exit status: 0 if every shard exited 0, otherwise the number of failed shards
# (capped at 125). Put this script LAST in RUN_CMD — simple-run-command reports
# the exit code of the last command in the chain.

set -u

CHIPS_PER_SHARD=1
TEMPLATE=""
OUT_DIR="${CDK_OUTPUT_DIR:-/tmp}/fanout"

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }

while getopts ":c:e:o:h" opt; do
  case "$opt" in
    c) CHIPS_PER_SHARD=$OPTARG ;;
    e) TEMPLATE=$OPTARG ;;
    o) OUT_DIR=$OPTARG ;;
    h) usage; exit 0 ;;
    \?) echo "tpu-fanout: unknown option -$OPTARG" >&2; exit 2 ;;
    :)  echo "tpu-fanout: -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

case "$CHIPS_PER_SHARD" in
  ""|*[!0-9]*) echo "tpu-fanout: -c wants a positive integer, got \"$CHIPS_PER_SHARD\"" >&2; exit 2 ;;
  0) echo "tpu-fanout: -c 0 is not a thing" >&2; exit 2 ;;
esac

if [ $# -eq 0 ]; then
  echo "tpu-fanout: nothing to run" >&2; usage >&2; exit 2
fi

# Build the shard command list.
CMDS=()
if [ -n "$TEMPLATE" ]; then
  for arg in "$@"; do CMDS+=("${TEMPLATE//\{\}/$arg}"); done
else
  CMDS=("$@")
fi
NUM_SHARDS=${#CMDS[@]}

# How many chips does this pod actually have?
#
# NOT by counting /dev/vfio/<n>: the pod exposes 8 numeric entries there but only 4
# are TPU chips (measured — tpu-info lists /dev/vfio/{2,4,5,6}). Counting them
# reports 8, which silently oversubscribes every chip 2x.
#
# TPU_CHIPS_PER_HOST_BOUNDS is set by the recipe's node ("2,2,1" -> 4) and is the
# cheap, exact answer. tpu-info's device table is the fallback.
if [ -n "${TPU_FANOUT_NUM_CHIPS:-}" ]; then
  NUM_CHIPS=$TPU_FANOUT_NUM_CHIPS
elif [ -n "${TPU_CHIPS_PER_HOST_BOUNDS:-}" ]; then
  NUM_CHIPS=$(echo "$TPU_CHIPS_PER_HOST_BOUNDS" | awk -F, '{print $1*$2*$3}')
else
  NUM_CHIPS=$(tpu-info 2>/dev/null | grep -c -o "/dev/vfio/[0-9]\+")
fi
if [ -z "$NUM_CHIPS" ] || [ "$NUM_CHIPS" -lt 1 ] 2>/dev/null; then
  echo "tpu-fanout: could not determine the chip count (TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-unset})" >&2
  echo "tpu-fanout: pass it explicitly with TPU_FANOUT_NUM_CHIPS=<n>" >&2
  exit 2
fi

# Only 1 chip per shard actually works. A process pinned to a 2- or 3-chip SUBSET of
# this pod cannot bring up its ICI mesh: libtpu aborts (SIGABRT, rc=134) after ~130s
# with
#   Check failed: vf_helper_->TearDownMesh() is OK (INTERNAL: ... STOP_SESSION failed)
# It fails for a *single* 2-chip process, not just two concurrent ones, and it fails
# identically under TPU_CHIPS_PER_HOST_BOUNDS and under TPU_CHIPS_PER_PROCESS_BOUNDS.
# 1 chip works and the whole 4-chip pod works; nothing in between does. Re-test with
# TPU_FANOUT_ALLOW_MULTICHIP=1 after a libtpu bump — the pin itself is fine, it is the
# sub-slice mesh that is not.
if [ "$CHIPS_PER_SHARD" -ne 1 ] && [ -z "${TPU_FANOUT_ALLOW_MULTICHIP:-}" ]; then
  echo "tpu-fanout: -c $CHIPS_PER_SHARD refused: a multi-chip subset of this pod cannot" >&2
  echo "tpu-fanout: initialize (libtpu TearDownMesh SIGABRT after ~130s)." >&2
  echo "tpu-fanout: one chip already gives a shard 2 TPU cores. Set TPU_FANOUT_ALLOW_MULTICHIP=1 to retry anyway." >&2
  exit 2
fi

NEED=$((NUM_SHARDS * CHIPS_PER_SHARD))
if [ "$NEED" -gt "$NUM_CHIPS" ]; then
  echo "tpu-fanout: $NUM_SHARDS shards x $CHIPS_PER_SHARD chips = $NEED > $NUM_CHIPS chips available." >&2
  echo "tpu-fanout: two processes on one chip fail with 'Device or resource busy'; refusing to oversubscribe." >&2
  exit 2
fi
if [ "$NEED" -lt "$NUM_CHIPS" ]; then
  echo "tpu-fanout: note $((NUM_CHIPS - NEED)) of $NUM_CHIPS chips will sit idle."
fi

mkdir -p "$OUT_DIR"

# libtpu's lockfile is a single global path; a stale one from an earlier process in
# this container blocks every shard. vllm-torchtpu's own launchers clear it the same way.
rm -f /tmp/libtpu_lockfile* 2>/dev/null

echo "tpu-fanout: $NUM_SHARDS shards, $CHIPS_PER_SHARD chip(s) each, $NUM_CHIPS chips on this pod"
echo "tpu-fanout: logs -> $OUT_DIR"

PIDS=()
CHIPSETS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  first=$((i * CHIPS_PER_SHARD))
  chips=$(seq -s, "$first" $((first + CHIPS_PER_SHARD - 1)))
  CHIPSETS+=("$chips")
  log="$OUT_DIR/shard$i.log"
  echo "tpu-fanout: shard$i chips=$chips cmd: ${CMDS[$i]}"
  (
    export TPU_VISIBLE_CHIPS="$chips"
    export TPU_VISIBLE_DEVICES="$chips"
    export ALLOW_MULTIPLE_LIBTPU_LOAD=1
    export TPU_LOG_DIR="/tmp/tpu_logs/shard$i"
    # VLLM_XLA_CACHE_PATH does NOT derive from VLLM_CACHE_ROOT (vllm/envs.py falls
    # back to the default cache root), so both have to be set to keep shards from
    # writing the same compile cache.
    export VLLM_CACHE_ROOT="${HOME:-/root}/.cache/vllm/shard$i"
    export VLLM_XLA_CACHE_PATH="${HOME:-/root}/.cache/vllm/shard$i/xla_cache"
    export TMPDIR="/tmp/shard$i"
    mkdir -p "$TMPDIR" "$TPU_LOG_DIR" "$VLLM_CACHE_ROOT"
    export TPU_FANOUT_SHARD="$i"
    export TPU_FANOUT_NUM_SHARDS="$NUM_SHARDS"
    export TPU_FANOUT_CHIPS="$chips"
    bash -c "${CMDS[$i]}"
  ) > "$log" 2>&1 &
  PIDS+=($!)
done

# Heartbeat, so a hung shard is visible without garbling the logs.
(
  while sleep 60; do
    alive=""
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
      kill -0 "${PIDS[$i]}" 2>/dev/null && alive="$alive $i"
    done
    [ -z "$alive" ] && break
    echo "tpu-fanout: still running:$alive"
  done
) &
HEARTBEAT=$!

RCS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  wait "${PIDS[$i]}"
  RCS+=($?)
done
kill "$HEARTBEAT" 2>/dev/null
wait "$HEARTBEAT" 2>/dev/null

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo
  echo "===== tpu-fanout shard$i (chips=${CHIPSETS[$i]}, rc=${RCS[$i]}) : ${CMDS[$i]} ====="
  cat "$OUT_DIR/shard$i.log"
done

echo
echo "===== tpu-fanout EXIT CODES ====="
FAILED=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  printf "shard%-3s chips=%-8s rc=%-4s %s\n" "$i" "${CHIPSETS[$i]}" "${RCS[$i]}" "${CMDS[$i]}"
  [ "${RCS[$i]}" -ne 0 ] && FAILED=$((FAILED + 1))
done
echo "tpu-fanout: $((NUM_SHARDS - FAILED))/$NUM_SHARDS shards succeeded"

[ "$FAILED" -gt 125 ] && FAILED=125
exit "$FAILED"
