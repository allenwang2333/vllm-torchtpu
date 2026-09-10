#!/usr/bin/env bash
# One shard = one subject on one chip.
#
#   shard_final.sh <name> <subject-rel-path> <md5> <mode>
#
# mode = final    the composed candidate: `arm_check.py --expect-suffix _final`
#                 AND the paired `arm_check_final.py` (every scope name over all
#                 7 suite shapes must carry `_c<chunk>_final`, and the incumbent
#                 read the same way must carry it on none).
# mode = control  the pinned incumbent driven through this identical harness, so
#                 a red on the other shard is the candidate and not the harness.
#                 `arm_check.py --expect-suffix NONE`.
#
# Both modes then run the frozen 68 UNMODIFIED (no -k) and `xgate.py`, which is
# the only thing here that compares a subject to the incumbent FILE. The shard
# is green only if pytest and xgate are both green.
set -u
NAME=$1
FILE=$2
MD5=$3
MODE=$4
ROOT=/root/p25
INCUMBENT="$ROOT/subjects/incumbent/sparsecore_topk.py"
INC_MD5=44353b347c96e0eedbd742c710d4540a
SUITE_MD5=b3241859e471fc87fa6e7a50e4ffec4b
export SCTOPK_SUBJECT="$ROOT/subjects/$FILE"
export SCTOPK_SUBJECT_MD5="$MD5"
OUT="${CDK_OUTPUT_DIR:-/tmp}/p25/$NAME"
mkdir -p "$OUT"

echo "===== SHARD $NAME  subject=$SCTOPK_SUBJECT  want=$MD5  mode=$MODE"
md5sum "$SCTOPK_SUBJECT" "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
       "$INCUMBENT" || exit 1
test "$(md5sum "$SCTOPK_SUBJECT" | cut -d' ' -f1)" = "$MD5" || {
  echo "SHARD_SUBJECT_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$ROOT/suite/test_dsv4_sparsecore_topk.py" | cut -d' ' -f1)" \
     = "$SUITE_MD5" || { echo "SHARD_SUITE_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$INCUMBENT" | cut -d' ' -f1)" = "$INC_MD5" || {
  echo "SHARD_INCUMBENT_GATE_FAIL $NAME"; exit 1; }
echo "SHARD_GATE_OK $NAME"

if [ "$MODE" = "final" ]; then EXPECT=_final; else EXPECT=NONE; fi
python "$ROOT/arm_check.py" --subject "$SCTOPK_SUBJECT" --subject-md5 "$MD5" \
  --arity 0 --expect-suffix "$EXPECT" 2>&1 | tee "$OUT/arm_check.log" | tail -20
RC_ARM=${PIPESTATUS[0]}
echo "SHARD_ARM_RC $NAME $RC_ARM"

RC_ARM2=0
if [ "$MODE" = "final" ]; then
  python "$ROOT/arm_check_final.py" \
    --variant "$SCTOPK_SUBJECT" --variant-md5 "$MD5" \
    --incumbent "$INCUMBENT" --incumbent-md5 "$INC_MD5" 2>&1 \
    | tee "$OUT/arm_check_final.log" | tail -40
  RC_ARM2=${PIPESTATUS[0]}
  echo "SHARD_ARM2_RC $NAME $RC_ARM2"
fi

if [ "$RC_ARM" -ne 0 ] || [ "$RC_ARM2" -ne 0 ]; then
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
  --incumbent "$INCUMBENT" --incumbent-md5 "$INC_MD5" \
  --suite "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --json "$OUT/xgate.json" 2>&1 | tail -60
RC_XGATE=${PIPESTATUS[0]}
echo "SHARD_XGATE_RC $NAME $RC_XGATE"

echo "SHARD_DONE $NAME arm=$RC_ARM arm2=$RC_ARM2 pytest=$RC_PYTEST xgate=$RC_XGATE"
[ "$RC_PYTEST" -eq 0 ] && [ "$RC_XGATE" -eq 0 ]
