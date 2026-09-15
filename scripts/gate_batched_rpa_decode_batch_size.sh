#!/usr/bin/env bash
# Correctness gate for VLLM_RPA_DECODE_BATCH_SIZE=4 and =2 on the RPAd decode
# kernel, at the production decode shape, against the shipping default of 8.
#
# Run 3. It closes the one coverage gap runs 1 and 2 left and fixes two real
# defects in the comparator that a review demonstrated executably.
#
#   THE GAP. num_live defaulted to min(CONCURRENCY, max_num_seqs) = 64, and 64
#   is divisible by 8, 4 AND 2. No arm ever entered a partial batch block, and
#   with num_live == max_num_seqs there were no padded slots either. The tail
#   block is exactly the path BlockSizes.batch_size controls, so the gate had
#   not touched it. In production max_num_seqs is 64 but the live count varies
#   continuously, which makes the partial block the common case. This run
#   sweeps num_live in {64, 63, 33, 7}:
#       63 -> tail of 7 at arm 8, 3 at arm 4, 1 at arm 2;  1 padded slot
#       33 -> tail of 1 at arm 8 and at arm 4, exact at arm 2; 31 padded
#        7 -> fewer live seqs than one block at arm 8;      57 padded
#   The drop_last_seqs / drop_first_seqs controls are numpy edits to the saved
#   npz: they simulate the symptom of a dropped tail block without ever running
#   the kernel's tail path, so they never covered this.
#
#   DEFECT 1. `caught = (not cross_ok) or ...` credited Check B, which fails on
#   clean data at this shape, so EVERY injection returned rc=0 regardless --
#   including scale_one_seq:1.0, a multiply by one. Sections 3a-3c and 5b of
#   runs 1 and 2 were vacuous; only the --rel-tol 1e9 sections were real. Now
#   only Check C may credit a catch, and an injection that moved nothing is
#   refused. Phase 3d runs the no-op and requires NOT CAUGHT.
#
#   DEFECT 2. The disclosed sensitivity floor was wrong in the gate's favour.
#   C2 binds at oracle_ratio_tol x the default's own distance from truth --
#   2.00 x 1.504e-02 = 3.0e-2, about a 3% single-sequence error, not the
#   "1.5-2%" previously claimed. Check C now prints the figure from the run's
#   own data, and phase 4 shows a 0.1% rescale is caught by NOTHING.
#
#   phase 0  overlay identity  -- refuse if the pod is not running d2648e80
#   phase 1  emit, 3 arms x 4 num_live, one arm per chip
#   phase 2  the gate at each num_live: all three, then arm 4 and arm 2 alone
#   phase 3  negative controls at num_live=63, the partial-block config
#   phase 4  the sensitivity floor, disclosed by running it
#   phase 5  summary
#
# Correctness fans out; nothing here is timed, so contention does not matter.
set -u

REPO=/root/cloud-devkit/vllm-torchtpu
NPZDIR=/tmp/rpa_gate
OUT="${CDK_OUTPUT_DIR:-/tmp/cdk-outputs}/rpa_batch_gate"
LOGDIR="$OUT/logs"
mkdir -p "$NPZDIR" "$LOGDIR"

NLIVE_LIST="64 63 33 7"
PARTIAL=63          # the num_live the negative controls run at

WRAPPER_REL=src/vllm_torchtpu/kernels/experimental/batched_rpa/wrapper.py
CHECK_REL=scripts/check_batched_rpa_decode_batch_size.py
SELF_REL=scripts/selftest_check_batched_rpa_decode_batch_size.py
MICRO_REL=scripts/microbench_batched_rpa_kv_layout.py

WANT_WRAPPER=90ca5c3bed49e63142bdc2bc66fa768e98b5180b1c3221abb19920ff21660463
IMAGE_WRAPPER=0d91cb90ef72c0ff6ca7465a4d193b3ec851240d2dc37d00987b980f5447f8d9
WANT_MICRO=95ef698d809c1c9abb963536d5fdf506c9808f99f750001c36707151dd57553c
WANT_CHECK=f0826104c935f3a368d80af64c796808f389e3c0665c26dc997f6624dc5df600
WANT_SELF=4b3eed49dd4b204e3747df47a4e055f81d68d1ec7c4bf2e65afe1fa22ebc2484

banner() { echo; echo "=============================================================================="; echo "== $*"; echo "=============================================================================="; }

banner "0. OVERLAY IDENTITY -- is this pod running d2648e80 + the current gate?"
cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 90; }
echo "date: $(date -u +%FT%TZ)   hostname: $(hostname)"
echo "--- git state baked into the image (NOT the overlay) ---"
git log --oneline -1 2>&1

GOT_WRAPPER=$(sha256sum "$REPO/$WRAPPER_REL" | awk '{print $1}')
GOT_MICRO=$(sha256sum "$REPO/$MICRO_REL" | awk '{print $1}')
GOT_CHECK=$(sha256sum "$REPO/$CHECK_REL" 2>/dev/null | awk '{print $1}')
GOT_SELF=$(sha256sum "$REPO/$SELF_REL" 2>/dev/null | awk '{print $1}')
echo
echo "wrapper.py  got  = $GOT_WRAPPER"
echo "wrapper.py  want = $WANT_WRAPPER   (d2648e80)"
echo "wrapper.py  image-only (d37c68cf, NO env var) = $IMAGE_WRAPPER"
echo "microbench  got  = $GOT_MICRO"
echo "microbench  want = $WANT_MICRO"
echo "checkscript got  = ${GOT_CHECK:-<MISSING>}"
echo "checkscript want = $WANT_CHECK"
echo "selftest    got  = ${GOT_SELF:-<MISSING>}"
echo "selftest    want = $WANT_SELF"
echo
echo "--- vllm_torchtpu import path (must be the mapped src/) ---"
python3 -c "import vllm_torchtpu; print(vllm_torchtpu.__file__)"
echo "--- the env var must be present in the source that will be imported ---"
grep -n "VLLM_RPA_DECODE_BATCH_SIZE" "$REPO/$WRAPPER_REL" || echo "(NOT PRESENT)"

if [ "$GOT_WRAPPER" != "$WANT_WRAPPER" ]; then
  echo
  echo "FATAL: wrapper.py is not the d2648e80 copy."
  [ "$GOT_WRAPPER" = "$IMAGE_WRAPPER" ] && \
    echo "It is the image's d37c68cf copy -- the --map-dir overlay did not land."
  echo "Every arm would silently run batch_size=8 and the gate would be vacuous."
  exit 91
fi
if [ "$GOT_MICRO" != "$WANT_MICRO" ]; then echo "FATAL: microbench mismatch"; exit 92; fi
if [ "$GOT_CHECK" != "$WANT_CHECK" ]; then echo "FATAL: check script mismatch"; exit 93; fi
if [ "$GOT_SELF" != "$WANT_SELF" ]; then echo "FATAL: selftest mismatch"; exit 95; fi
echo
echo "OVERLAY OK: wrapper.py, microbench, check script and selftest all match."

banner "0b. COMPARATOR SELF-TEST (CPU, fabricated npz, no TPU)"
echo "Nineteen branches, healthy and broken, including the no-op injection that"
echo "the old attribution bug reported as caught. If this is red the gate below"
echo "cannot be trusted whatever it says."
python3 "$SELF_REL" 2>&1 | tee "$LOGDIR/selftest.log"
SELFTEST_RC=${PIPESTATUS[0]}
echo "selftest rc=$SELFTEST_RC (want 0)"

banner "1. EMIT -- 3 arms x num_live {$NLIVE_LIST}, one arm per chip"
bash /root/fanout/tpu-fanout.sh \
    "bash /root/gate/arm.sh 8 \"$NLIVE_LIST\"" \
    "bash /root/gate/arm.sh 4 \"$NLIVE_LIST\"" \
    "bash /root/gate/arm.sh 2 \"$NLIVE_LIST\""
FANOUT_RC=$?
echo "tpu-fanout rc=$FANOUT_RC (number of failed shards)"

ls -l "$NPZDIR" || true
MISSING=0
for n in $NLIVE_LIST; do
  for b in 8 4 2; do
    [ -s "$NPZDIR/arm_b${b}_n${n}.npz" ] || { echo "MISSING arm_b${b}_n${n}.npz"; MISSING=$((MISSING+1)); }
  done
done
[ "$MISSING" = 0 ] || { echo "FATAL: $MISSING arm file(s) missing -- an emit did not run."; exit 94; }
cp -f "$NPZDIR"/*.npz "$OUT"/ 2>/dev/null || true

CHK="python3 $CHECK_REL"
A() { echo "$NPZDIR/arm_b$1_n$2.npz"; }

banner "2. THE GATE, AT EACH num_live"
echo "A failure at a partial batch block is a real result, not a run to work"
echo "around. Each num_live is reported on its own."
declare -A RC_ALL RC_B4 RC_B2
for n in $NLIVE_LIST; do
  echo
  echo "######################## num_live=$n ########################"
  $CHK --compare "$(A 8 $n)" "$(A 4 $n)" "$(A 2 $n)" 2>&1 | tee "$LOGDIR/gate_all_n$n.log"
  RC_ALL[$n]=${PIPESTATUS[0]}
  echo "rc_all[n=$n]=${RC_ALL[$n]}"

  echo
  echo "--- arm 4 vs arm 8, num_live=$n ---"
  $CHK --compare "$(A 8 $n)" "$(A 4 $n)" 2>&1 | tee "$LOGDIR/gate_arm4_n$n.log"
  RC_B4[$n]=${PIPESTATUS[0]}
  echo "rc_arm4[n=$n]=${RC_B4[$n]}"

  echo
  echo "--- arm 2 vs arm 8, num_live=$n ---"
  $CHK --compare "$(A 8 $n)" "$(A 2 $n)" 2>&1 | tee "$LOGDIR/gate_arm2_n$n.log"
  RC_B2[$n]=${PIPESTATUS[0]}
  echo "rc_arm2[n=$n]=${RC_B2[$n]}"
done

banner "3. NEGATIVE CONTROLS at num_live=$PARTIAL (the partial-block config)"
echo "Only Check C -- the float32 reference -- may credit a catch now. Check B"
echo "fails on clean data at this shape, so crediting it marked everything"
echo "caught, including a multiply by one."
P=$PARTIAL

echo
echo "--- 3a. drop_last_seqs:2 into arm 2 (a skipped final batch block) ---"
$CHK --compare "$(A 8 $P)" "$(A 4 $P)" "$(A 2 $P)" \
     --inject-fault drop_last_seqs:2 --inject-into 2 \
    2>&1 | tee "$LOGDIR/nc_drop_last.log"
NC1=${PIPESTATUS[0]}; echo "rc=$NC1 (want 0 = caught)"

echo
echo "--- 3b. drop_first_seqs:2 into arm 4 (a skipped first batch block) ---"
$CHK --compare "$(A 8 $P)" "$(A 4 $P)" "$(A 2 $P)" \
     --inject-fault drop_first_seqs:2 --inject-into 4 \
    2>&1 | tee "$LOGDIR/nc_drop_first.log"
NC2=${PIPESTATUS[0]}; echo "rc=$NC2 (want 0 = caught)"

echo
echo "--- 3c. scale_one_seq:1.05 into arm 2 (a 5% m/l rescale on one seq) ---"
$CHK --compare "$(A 8 $P)" "$(A 4 $P)" "$(A 2 $P)" \
     --inject-fault scale_one_seq:1.05 --inject-into 2 \
    2>&1 | tee "$LOGDIR/nc_scale.log"
NC3=${PIPESTATUS[0]}; echo "rc=$NC3 (want 0 = caught)"

echo
echo "--- 3d. THE FIX, PROVEN: scale_one_seq:1.0 -- a multiply by ONE ---"
echo "     Nothing is perturbed, so nothing can have been caught. This used to"
echo "     print 'was caught by check B. The check discriminates.' It must now"
echo "     report NOT CAUGHT (rc=1)."
$CHK --compare "$(A 8 $P)" "$(A 4 $P)" "$(A 2 $P)" \
     --inject-fault scale_one_seq:1.0 --inject-into 2 \
    2>&1 | tee "$LOGDIR/nc_noop.log"
NOOP=${PIPESTATUS[0]}; echo "rc=$NOOP (want 1 = correctly NOT caught)"

echo
echo "--- 3e. vacuity control: the same arm file compared with itself ---"
$CHK --compare "$(A 8 $P)" "$(A 8 $P)" 2>&1 | tee "$LOGDIR/nc_vacuous.log"
NC4=${PIPESTATUS[0]}; echo "rc=$NC4 (want 2)"

echo
echo "--- 3f. cross-num_live: arm 4 at n=$P against arm 8 at n=64 ---"
echo "     Different configs must be refused outright, not silently compared."
$CHK --compare "$(A 8 64)" "$(A 4 $P)" 2>&1 | tee "$LOGDIR/nc_crossn.log"
NC5=${PIPESTATUS[0]}; echo "rc=$NC5 (want 2)"

banner "4. SENSITIVITY FLOOR -- disclosed by running it, at num_live=$PARTIAL"
echo "C2 binds at oracle_ratio_tol x the default's own distance from truth."
echo "A 0.1% single-sequence rescale is below that and is caught by NOTHING;"
echo "the previous reports' claim that Check B caught it was the attribution"
echo "bug. rc=1 is the honest answer and is what this must print."
$CHK --compare "$(A 8 $P)" "$(A 4 $P)" "$(A 2 $P)" \
     --inject-fault scale_one_seq:1.001 --inject-into 2 \
    2>&1 | tee "$LOGDIR/floor_1e3.log"
FLOOR3=${PIPESTATUS[0]}; echo "rc=$FLOOR3 (want 1 = not caught, the floor)"

echo
echo "--- 4b. scale_one_seq:1.03, just above the floor -- must be caught ---"
$CHK --compare "$(A 8 $P)" "$(A 4 $P)" "$(A 2 $P)" \
     --inject-fault scale_one_seq:1.03 --inject-into 2 \
    2>&1 | tee "$LOGDIR/floor_3e2.log"
FLOOR2=${PIPESTATUS[0]}; echo "rc=$FLOOR2 (want 0 = caught)"

banner "5. SUMMARY"
printf "%-52s %s  %s\n" "comparator self-test (want rc 0)" "$SELFTEST_RC" "$([ "$SELFTEST_RC" = 0 ] && echo OK || echo HOLLOW)"
printf "%-52s %s\n" "emit shards failed (tpu-fanout rc)" "$FANOUT_RC"
for n in $NLIVE_LIST; do
  printf "%-52s %s  %s\n" "gate num_live=$n: all three arms" "${RC_ALL[$n]}" "$([ "${RC_ALL[$n]}" = 0 ] && echo PASS || echo NOT-PASS)"
  printf "%-52s %s  %s\n" "gate num_live=$n: arm 4 vs arm 8" "${RC_B4[$n]}" "$([ "${RC_B4[$n]}" = 0 ] && echo PASS || echo NOT-PASS)"
  printf "%-52s %s  %s\n" "gate num_live=$n: arm 2 vs arm 8" "${RC_B2[$n]}" "$([ "${RC_B2[$n]}" = 0 ] && echo PASS || echo NOT-PASS)"
done
printf "%-52s %s  %s\n" "neg ctl n=$P: drop_last_seqs:2 (want 0)" "$NC1" "$([ "$NC1" = 0 ] && echo OK || echo HOLLOW)"
printf "%-52s %s  %s\n" "neg ctl n=$P: drop_first_seqs:2 (want 0)" "$NC2" "$([ "$NC2" = 0 ] && echo OK || echo HOLLOW)"
printf "%-52s %s  %s\n" "neg ctl n=$P: scale_one_seq:1.05 (want 0)" "$NC3" "$([ "$NC3" = 0 ] && echo OK || echo HOLLOW)"
printf "%-52s %s  %s\n" "NO-OP scale_one_seq:1.0 (want 1 = NOT caught)" "$NOOP" "$([ "$NOOP" = 1 ] && echo OK || echo HOLLOW)"
printf "%-52s %s  %s\n" "neg ctl: self-compare vacuity (want 2)" "$NC4" "$([ "$NC4" = 2 ] && echo OK || echo HOLLOW)"
printf "%-52s %s  %s\n" "neg ctl: cross-num_live refusal (want 2)" "$NC5" "$([ "$NC5" = 2 ] && echo OK || echo HOLLOW)"
printf "%-52s %s  %s\n" "floor: scale_one_seq:1.001 (want 1 = not caught)" "$FLOOR3" "$([ "$FLOOR3" = 1 ] && echo OK || echo UNEXPECTED)"
printf "%-52s %s  %s\n" "floor: scale_one_seq:1.03 (want 0 = caught)" "$FLOOR2" "$([ "$FLOOR2" = 0 ] && echo OK || echo HOLLOW)"

banner "6. ARTIFACTS"
find "$OUT" -type f -printf "%s\t%p\n" 2>/dev/null | sort -k2

banner "DONE"
# Job status reflects the controls AND the gate. A hollow control is as much a
# failure of this job as a failed arm: it means the answer cannot be trusted.
FAIL=0
[ "$SELFTEST_RC" = 0 ] || FAIL=$((FAIL+1))
[ "$FANOUT_RC" = 0 ] || FAIL=$((FAIL+1))
[ "$NC1" = 0 ] || FAIL=$((FAIL+1))
[ "$NC2" = 0 ] || FAIL=$((FAIL+1))
[ "$NC3" = 0 ] || FAIL=$((FAIL+1))
[ "$NOOP" = 1 ] || FAIL=$((FAIL+1))
[ "$NC4" = 2 ] || FAIL=$((FAIL+1))
[ "$NC5" = 2 ] || FAIL=$((FAIL+1))
[ "$FLOOR2" = 0 ] || FAIL=$((FAIL+1))
SUM=""
for n in $NLIVE_LIST; do
  [ "${RC_ALL[$n]}" = 0 ] || FAIL=$((FAIL+1))
  SUM="$SUM n$n:all=${RC_ALL[$n]},arm4=${RC_B4[$n]},arm2=${RC_B2[$n]}"
done
echo "CONTROLS_AND_GATE_FAILURES=$FAIL"
echo "GATE_RESULT$SUM"
exit $((FAIL > 120 ? 120 : FAIL))
