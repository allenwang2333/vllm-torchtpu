#!/usr/bin/env bash
# One shard = `v10` at one count arity, on one chip.
#
#   shard_v10.sh <arity> [sample 0|1]
#
# `sample 0` is the CONTROL arm: the same file, the incumbent branch. It proves
# the 16 hunks of flag plumbing did not disturb the path they wrap, and it is
# what makes a red sample arm attributable to the sample body rather than to
# `v10` breaking the file in general.
#
# Three differences from `shard.sh`, all forced by `v10`:
#
#  1. **Two subject files.** `sparsecore_topk.py` imports `sparsecore_topk_sample`
#     and falls back to a bare `import` when the installed package has no such
#     submodule -- which it does not. So the subject DIRECTORY goes on
#     `PYTHONPATH`, and both md5s are gated. Mapping only the first file makes
#     the import fail, not silently succeed, but gating both is what proves the
#     pair is the pair the builder froze.
#  2. **The arm is an env default read at import.** `SCTOPK_SAMPLE_AND_COUNT=0`
#     runs the incumbent branch and passes 68/68 while testing nothing, so the
#     env var is set here AND `arm_check.py` proves the sample kernel is the one
#     that lowered, by scope name. The env var is not the evidence.
#  3. **Full pytest output is kept** under `$OUT/pytest.log`. `shard.sh` piped to
#     `tail -110` and lost `conftest.py`'s SUBJECT_GATE_OK banner off the top.
set -u
ARITY=$1
SAMPLE=${2:-1}
if [ "$SAMPLE" = "1" ]; then NAME="v10-a$ARITY"; else NAME="v10-control"; fi
ROOT=/root/p22
SUBJ_DIR="$ROOT/subjects/v10"
export SCTOPK_SUBJECT="$SUBJ_DIR/sparsecore_topk.py"
export SCTOPK_SUBJECT_MD5=f83332273f1cdc05995965c0405e8665
export SCTOPK_SAMPLE_AND_COUNT="$SAMPLE"
export SCTOPK_COUNT_ARITY="$ARITY"
export PYTHONPATH="$SUBJ_DIR:${PYTHONPATH:-}"
OUT="${CDK_OUTPUT_DIR:-/tmp}/p22/$NAME"
mkdir -p "$OUT"

echo "===== SHARD $NAME  arity=$ARITY  sample=$SAMPLE  subject=$SCTOPK_SUBJECT"
md5sum "$SUBJ_DIR/sparsecore_topk.py" "$SUBJ_DIR/sparsecore_topk_sample.py" \
       "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
       "$ROOT/subjects/incumbent/sparsecore_topk.py" || exit 1
test "$(md5sum "$SUBJ_DIR/sparsecore_topk.py" | cut -d' ' -f1)" \
     = "f83332273f1cdc05995965c0405e8665" || {
  echo "SHARD_SUBJECT_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$SUBJ_DIR/sparsecore_topk_sample.py" | cut -d' ' -f1)" \
     = "21f2d777419b3cf43ba940f004bd765a" || {
  echo "SHARD_SAMPLE_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$ROOT/suite/test_dsv4_sparsecore_topk.py" | cut -d' ' -f1)" \
     = "b3241859e471fc87fa6e7a50e4ffec4b" || {
  echo "SHARD_SUITE_GATE_FAIL $NAME"; exit 1; }
test "$(md5sum "$ROOT/subjects/incumbent/sparsecore_topk.py" | cut -d' ' -f1)" \
     = "44353b347c96e0eedbd742c710d4540a" || {
  echo "SHARD_INCUMBENT_GATE_FAIL $NAME"; exit 1; }
echo "SHARD_GATE_OK $NAME"

# ---- did the arm fire? -------------------------------------------------
python "$ROOT/arm_check.py" --subject "$SCTOPK_SUBJECT" \
  --subject-md5 f83332273f1cdc05995965c0405e8665 \
  --arity "$ARITY" --expect-sample "$SAMPLE" 2>&1 | tee "$OUT/arm_check.log" | tail -20
RC_ARM=${PIPESTATUS[0]}
echo "SHARD_ARM_RC $NAME $RC_ARM"
if [ "$RC_ARM" -ne 0 ]; then
  echo "SHARD_ABORT $NAME the expected arm is not the one that lowered -- a"
  echo "green suite below would be the wrong body passing, so nothing else runs"
  exit 1
fi

# ---- the frozen 68, no -k ----------------------------------------------
python -m pytest "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --collect-only -q 2>&1 | tail -3
python -m pytest "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  -v --timeout=3600 -p no:randomly \
  --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log" | tail -120
RC_PYTEST=${PIPESTATUS[0]}
echo "SHARD_PYTEST_RC $NAME $RC_PYTEST"
head -12 "$OUT/pytest.log"

# ---- the cross-module gate ---------------------------------------------
# `v10`'s gate is `v3`'s shape: scores as a multiset, zero tolerance, never a
# byte comparison of indices -- on a tie it may pick a different subset of tied
# indices. `xgate.py` exits on the multiset verdict and reports `bit` alongside.
python "$ROOT/xgate.py" --name "$NAME" \
  --variant "$SCTOPK_SUBJECT" --variant-md5 f83332273f1cdc05995965c0405e8665 \
  --incumbent "$ROOT/subjects/incumbent/sparsecore_topk.py" \
  --incumbent-md5 44353b347c96e0eedbd742c710d4540a \
  --suite "$ROOT/suite/test_dsv4_sparsecore_topk.py" \
  --json "$OUT/xgate.json" 2>&1 | tail -60
RC_XGATE=${PIPESTATUS[0]}
echo "SHARD_XGATE_RC $NAME $RC_XGATE"

echo "SHARD_DONE $NAME arm=$RC_ARM pytest=$RC_PYTEST xgate=$RC_XGATE"
[ "$RC_PYTEST" -eq 0 ] && [ "$RC_XGATE" -eq 0 ]
