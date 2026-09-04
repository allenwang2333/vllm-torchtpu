#!/bin/bash
# `bash -n` is a SYNTAX check and cannot see an undefined function: a missing
# one is a runtime 127, and inside `if ! fn; then ... exit 0; fi` that reads as
# "the gate failed", skips the arm, and exits GREEN. That is exactly what build
# 374 did -- five arms skipped, three minutes of pod time, a passed build and no
# data. This runs the driver with the benchmark replaced by /bin/true so every
# code path is executed on CPU in a second.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=$(mktemp)
OUT="${OUT:-$(mktemp -d)}" BENCH=/bin/true CLEANUP=/bin/true REPS=1 \
  K3_LADDER_HI=0 MIN_FREE_MB=0 K3_CELLS_LO="1:1" \
  bash "${SCRIPT_DIR}/k3r5_ladder.sh" base k1 c1 e8m0 bf16 f3pp > "$LOG" 2>&1
rc=$?
fail=0
if grep -q 'command not found' "$LOG"; then
  echo "FAIL: an undefined function was called"; grep 'command not found' "$LOG" | sort -u; fail=1
fi
for arm in base k1 c1 e8m0 bf16 f3pp; do
  grep -q "ARM ${arm} " "$LOG" || { echo "FAIL: arm ${arm} never started"; fail=1; }
done
# every arm must reach the lifetime, not skip it
n=$(grep -c 'starts the server, keeps it alive' "$LOG")
[ "$n" -eq 6 ] || { echo "FAIL: ${n}/6 arms reached run_lifetime"; fail=1; }
grep -q '__UNKNOWN__' "$LOG" && { echo "FAIL: an arm name did not resolve"; fail=1; }
if [ "$fail" = 0 ]; then echo "SELFTEST PASS: 6 arms, all functions resolved, all lifetimes entered"; else
  echo "--- driver output ---"; tail -40 "$LOG"; fi
rm -f "$LOG"; exit "$fail"
