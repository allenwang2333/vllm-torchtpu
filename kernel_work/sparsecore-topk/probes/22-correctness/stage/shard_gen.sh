#!/usr/bin/env bash
# One shard = one single-file variant on one chip, with a scope-name arm check.
#
#   shard_gen.sh <name> <subject-file-under-subjects/> <md5> <expect-suffix>
#
# `<expect-suffix>` is the Pallas scope-name suffix that proves THIS arm
# lowered (`_v4ragged`, `_v4c2048`, …), or `NONE` for a bare
# `sc_topk_direct_b{b}_n{n}_k{k}`. It is checked before the suite runs: a green
# 68 against the wrong body is worse than a red one, and for `v4` the control
# rung's md5 is byte-identical to the incumbent, so the file hash alone cannot
# distinguish "the ladder is wired" from "everything resolved to the base".
set -u
NAME=$1
FILE=$2
MD5=$3
SUFFIX=$4
ROOT=/root/p22
export SCTOPK_SUBJECT="$ROOT/subjects/$FILE"
export SCTOPK_SUBJECT_MD5="$MD5"
OUT="${CDK_OUTPUT_DIR:-/tmp}/p22/$NAME"
mkdir -p "$OUT"

echo "===== SHARD $NAME  subject=$SCTOPK_SUBJECT  want=$MD5  suffix=$SUFFIX"
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

python "$ROOT/arm_check.py" --subject "$SCTOPK_SUBJECT" --subject-md5 "$MD5" \
  --arity 0 --expect-suffix "$SUFFIX" 2>&1 | tee "$OUT/arm_check.log" | tail -20
RC_ARM=${PIPESTATUS[0]}
echo "SHARD_ARM_RC $NAME $RC_ARM"
if [ "$RC_ARM" -ne 0 ]; then
  echo "SHARD_ABORT $NAME the expected arm is not the one that lowered"
  exit 1
fi

python -m pytest "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --collect-only -q 2>&1 | tail -3
python -m pytest "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  -v --timeout=3600 -p no:randomly \
  --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log" | tail -120
RC_PYTEST=${PIPESTATUS[0]}
echo "SHARD_PYTEST_RC $NAME $RC_PYTEST"
head -12 "$OUT/pytest.log"

python "$ROOT/xgate.py" --name "$NAME" \
  --variant "$SCTOPK_SUBJECT" --variant-md5 "$MD5" \
  --incumbent "$ROOT/subjects/incumbent/sparsecore_topk.py" \
  --incumbent-md5 44353b347c96e0eedbd742c710d4540a \
  --suite "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --json "$OUT/xgate.json" 2>&1 | tail -60
RC_XGATE=${PIPESTATUS[0]}
echo "SHARD_XGATE_RC $NAME $RC_XGATE"

echo "SHARD_DONE $NAME arm=$RC_ARM pytest=$RC_PYTEST xgate=$RC_XGATE"
[ "$RC_PYTEST" -eq 0 ] && [ "$RC_XGATE" -eq 0 ]
