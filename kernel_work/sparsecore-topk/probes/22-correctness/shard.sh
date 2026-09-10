#!/usr/bin/env bash
# One shard = one variant on one chip. Runs the frozen 68 with the subject
# swapped in per-process (never by cp -- four shards share one filesystem),
# then the cross-module gate the frozen 68 does not implement.
#
#   shard.sh <name> <subject-md5>
set -u
NAME=$1
MD5=$2
ROOT=/root/p22
export SCTOPK_SUBJECT="$ROOT/subjects/$NAME/sparsecore_topk.py"
export SCTOPK_SUBJECT_MD5="$MD5"
OUT="${CDK_OUTPUT_DIR:-/tmp}/p22/$NAME"
mkdir -p "$OUT"

echo "===== SHARD $NAME  subject=$SCTOPK_SUBJECT  want=$MD5"
md5sum "$SCTOPK_SUBJECT" "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
       "$ROOT/subjects/incumbent/sparsecore_topk.py" || exit 1
test "$(md5sum "$SCTOPK_SUBJECT" | cut -d' ' -f1)" = "$MD5" || {
  echo "SHARD_SUBJECT_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$ROOT/suite/test_dsv4_sparsecore_topk.py" | cut -d' ' -f1)" \
     = "b3241859e471fc87fa6e7a50e4ffec4b" || {
  echo "SHARD_SUITE_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$ROOT/subjects/incumbent/sparsecore_topk.py" | cut -d' ' -f1)" \
     = "44353b347c96e0eedbd742c710d4540a" || {
  echo "SHARD_INCUMBENT_GATE_FAIL $NAME"; exit 1; }
echo "SHARD_GATE_OK $NAME"

# ---- the frozen 68, no -k ----------------------------------------------
# Collection first: the id count is verified, not assumed. 68 is the baseline
# at j-62275e52 and a different number means the suite moved under us.
python -m pytest "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --collect-only -q 2>&1 | tail -3
python -m pytest "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  -v --timeout=3600 -p no:randomly \
  --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log" | tail -110
RC_PYTEST=${PIPESTATUS[0]}
echo "SHARD_PYTEST_RC $NAME $RC_PYTEST"
# `tail` alone drops conftest.py's SUBJECT_GATE_OK banner off the TOP, which is
# the one line proving the subject swap took. Keep the head as well as the tail.
head -12 "$OUT/pytest.log"

# ---- the cross-module gate ---------------------------------------------
# Runs regardless of the pytest result: if the frozen suite went red we still
# want to know whether the variant is exact and whether it is bit-identical,
# because those are the two facts the gate table is written in terms of.
python "$ROOT/xgate.py" --name "$NAME" \
  --variant "$SCTOPK_SUBJECT" --variant-md5 "$MD5" \
  --incumbent "$ROOT/subjects/incumbent/sparsecore_topk.py" \
  --incumbent-md5 44353b347c96e0eedbd742c710d4540a \
  --suite "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --json "$OUT/xgate.json" 2>&1 | tail -60
RC_XGATE=${PIPESTATUS[0]}
echo "SHARD_XGATE_RC $NAME $RC_XGATE"

echo "SHARD_DONE $NAME pytest=$RC_PYTEST xgate=$RC_XGATE"
[ "$RC_PYTEST" -eq 0 ] && [ "$RC_XGATE" -eq 0 ]
