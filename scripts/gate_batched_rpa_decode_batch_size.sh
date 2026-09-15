#!/usr/bin/env bash
# Correctness gate for VLLM_RPA_DECODE_BATCH_SIZE=4 and =2 on the RPAd decode
# kernel, at the production decode shape, against the shipping default of 8.
#
# This is the cdk driver, committed so the next session does not rebuild it.
# It ran as job j-03c539ac-411d-4422-b39d (Succeeded, tpu7x 2x2x1, image
# vllm-torchtpu-vdziuba@sha256:fe34291c..., RUN_CMD=bash /root/gate/driver.sh),
# submitted with:
#
#   cdk job create simple-run-command \
#     DOCKER_IMAGE=<image> 'RUN_CMD=bash /root/gate/driver.sh' ENABLE_XPROF=0 \
#     --map-dir <repo>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src \
#     --map-dir <repo>/scripts:tpu-job:/root/cloud-devkit/vllm-torchtpu/scripts \
#     --map-dir <skills>/run-command-in-cdk/fanout:tpu-job:/root/fanout \
#     --map-dir <payload>:tpu-job:/root/gate \
#     -t agent-jobs,preflight-check
#
# where <payload> holds this file as driver.sh and its sibling
# gate_batched_rpa_decode_batch_size_arm.sh as arm.sh. The image predates
# d2648e80, so the src/ map is what delivers the env var; phase 0 refuses the
# run by sha256 if it did not land, because without it every arm silently runs
# batch_size=8 and the gate is vacuous.
#
# Run 2. Run 1 (j-1dfedbef) passed every absolute check on all three arms and
# failed the cross-arm tolerance marginally, with only 7.9x separation from the
# over-read signature. Rather than move the constant to fit the answer, the
# comparator now judges each arm against a float32 reference computed from the
# post-step KV cache. Check C is the numerical gate; Check B is advisory.
#
#   phase 0  overlay identity  -- refuse if the pod is not running 1e67367b
#   phase 1  emit, 3 arms, one chip each (8 / 4 / 2), in parallel
#   phase 2  the gate itself: all three arms, then each arm pairwise vs 8
#   phase 3  negative controls: faults injected on purpose; the gate MUST go red
#   phase 4  Check C on its own, with Check B disabled -- proves C discriminates
#   phase 5  sensitivity floor, and Check B's own verdict, both informational
#
# Correctness fans out; nothing here is timed, so contention does not matter.
set -u

REPO=/root/cloud-devkit/vllm-torchtpu
NPZDIR=/tmp/rpa_gate
OUT="${CDK_OUTPUT_DIR:-/tmp/cdk-outputs}/rpa_batch_gate"
LOGDIR="$OUT/logs"
mkdir -p "$NPZDIR" "$LOGDIR"

WRAPPER_REL=src/vllm_torchtpu/kernels/experimental/batched_rpa/wrapper.py
CHECK_REL=scripts/check_batched_rpa_decode_batch_size.py
SELF_REL=scripts/selftest_check_batched_rpa_decode_batch_size.py
MICRO_REL=scripts/microbench_batched_rpa_kv_layout.py

WANT_WRAPPER=90ca5c3bed49e63142bdc2bc66fa768e98b5180b1c3221abb19920ff21660463
IMAGE_WRAPPER=0d91cb90ef72c0ff6ca7465a4d193b3ec851240d2dc37d00987b980f5447f8d9
WANT_MICRO=95ef698d809c1c9abb963536d5fdf506c9808f99f750001c36707151dd57553c
WANT_CHECK=1be91b451269295c204499d6a5f270e22e769a2520a71b25cd29a0e8768bb8ee
WANT_SELF=4f0e0aab1452dde75313dd323c4143b27036edf412c02562150a7d73a64588e3

banner() { echo; echo "=============================================================================="; echo "== $*"; echo "=============================================================================="; }

banner "0. OVERLAY IDENTITY -- is this pod running d2648e80 + the gate at 1e67367b?"
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
echo "Seventeen branches, healthy and broken, including an arm that reads past"
echo "kv_len, an arm that drops sequences, an arm placed 4x further from float32"
echo "truth, a reference blind to a one-page over-read, two arms sharing a build"
echo "hash, and the tolerance's sensitivity floor. If this is red the gate below"
echo "cannot be trusted whatever it says."
python3 "$SELF_REL" 2>&1 | tee "$LOGDIR/selftest.log"
SELFTEST_RC=${PIPESTATUS[0]}
echo "selftest rc=$SELFTEST_RC (want 0)"

banner "1. EMIT -- 3 arms, one chip each, in parallel"
bash /root/fanout/tpu-fanout.sh \
    "bash /root/gate/arm.sh 8" \
    "bash /root/gate/arm.sh 4" \
    "bash /root/gate/arm.sh 2"
FANOUT_RC=$?
echo "tpu-fanout rc=$FANOUT_RC (number of failed shards)"

A8="$NPZDIR/arm_b8.npz"
A4="$NPZDIR/arm_b4.npz"
A2="$NPZDIR/arm_b2.npz"
ls -l "$NPZDIR" || true
for f in "$A8" "$A4" "$A2"; do
  [ -s "$f" ] || { echo "FATAL: $f missing or empty -- an arm did not emit."; exit 94; }
done
cp -f "$NPZDIR"/*.npz "$OUT"/ 2>/dev/null || true

CHK="python3 $CHECK_REL"

banner "2. THE GATE"
echo "--- 2a. all three arms together ---"
$CHK --compare "$A8" "$A4" "$A2" 2>&1 | tee "$LOGDIR/gate_all.log"
RC_ALL=${PIPESTATUS[0]}
echo "rc_all=$RC_ALL"

echo
echo "--- 2b. arm 4 vs arm 8, on its own ---"
$CHK --compare "$A8" "$A4" 2>&1 | tee "$LOGDIR/gate_arm4.log"
RC_B4=${PIPESTATUS[0]}
echo "rc_arm4=$RC_B4"

echo
echo "--- 2c. arm 2 vs arm 8, on its own ---"
$CHK --compare "$A8" "$A2" 2>&1 | tee "$LOGDIR/gate_arm2.log"
RC_B2=${PIPESTATUS[0]}
echo "rc_arm2=$RC_B2"

banner "3. NEGATIVE CONTROLS -- the gate must go red on a planted bug"
echo "Each of these corrupts one arm's output with a bug the kernel could"
echo "plausibly have, and requires the comparison to CATCH it. rc=0 means"
echo "caught (control passed); rc=1 means the tolerance let it through and"
echo "the gate is hollow for that bug."

echo
echo "--- 3a. drop_last_seqs:2 into arm 2 (a skipped final batch block) ---"
$CHK --compare "$A8" "$A4" "$A2" --inject-fault drop_last_seqs:2 --inject-into 2 \
    2>&1 | tee "$LOGDIR/nc_drop_last.log"
NC1=${PIPESTATUS[0]}; echo "rc=$NC1 (want 0)"

echo
echo "--- 3b. drop_first_seqs:2 into arm 4 (a skipped first batch block) ---"
$CHK --compare "$A8" "$A4" "$A2" --inject-fault drop_first_seqs:2 --inject-into 4 \
    2>&1 | tee "$LOGDIR/nc_drop_first.log"
NC2=${PIPESTATUS[0]}; echo "rc=$NC2 (want 0)"

echo
echo "--- 3c. scale_one_seq:1.05 into arm 2 (a 5% m/l rescaling error on one seq) ---"
$CHK --compare "$A8" "$A4" "$A2" --inject-fault scale_one_seq:1.05 --inject-into 2 \
    2>&1 | tee "$LOGDIR/nc_scale.log"
NC3=${PIPESTATUS[0]}; echo "rc=$NC3 (want 0)"

echo
echo "--- 3d. vacuity control: the same arm file compared with itself ---"
echo "     The guard must refuse this outright (rc=2), not report a pass."
$CHK --compare "$A8" "$A8" 2>&1 | tee "$LOGDIR/nc_vacuous.log"
NC4=${PIPESTATUS[0]}; echo "rc=$NC4 (want 2)"

banner "4. CHECK C ON ITS OWN -- Check B disabled with --rel-tol 1e9"
echo "Run 1's discrimination came from Check B, which is now advisory. These"
echo "repeat the controls with the cross-arm tolerance set so wide it cannot"
echo "fire, so a catch here is Check C -- the float32 reference -- catching the"
echo "bug unaided. If any of these comes back rc=1, Check C is decorative and"
echo "the gate rests on nothing."

echo
echo "--- 4a. drop_last_seqs:2 into arm 2, Check C alone ---"
$CHK --compare "$A8" "$A4" "$A2" --rel-tol 1e9 \
     --inject-fault drop_last_seqs:2 --inject-into 2 \
    2>&1 | tee "$LOGDIR/conly_drop_last.log"
CO1=${PIPESTATUS[0]}; echo "rc=$CO1 (want 0)"

echo
echo "--- 4b. drop_first_seqs:2 into arm 4, Check C alone ---"
$CHK --compare "$A8" "$A4" "$A2" --rel-tol 1e9 \
     --inject-fault drop_first_seqs:2 --inject-into 4 \
    2>&1 | tee "$LOGDIR/conly_drop_first.log"
CO2=${PIPESTATUS[0]}; echo "rc=$CO2 (want 0)"

echo
echo "--- 4c. scale_one_seq:1.05 into arm 2, Check C alone ---"
$CHK --compare "$A8" "$A4" "$A2" --rel-tol 1e9 \
     --inject-fault scale_one_seq:1.05 --inject-into 2 \
    2>&1 | tee "$LOGDIR/conly_scale.log"
CO3=${PIPESTATUS[0]}; echo "rc=$CO3 (want 0)"

echo
echo "--- 4d. C2 tightened to 1.02x: at least one arm must go red ---"
echo "     Not a bug, a calibration probe: it shows C2 is a live predicate at"
echo "     this shape and not a bound so loose nothing could ever trip it."
$CHK --compare "$A8" "$A4" "$A2" --oracle-ratio-tol 1.02 \
    2>&1 | tee "$LOGDIR/c2_tight.log"
CO4=${PIPESTATUS[0]}; echo "rc=$CO4 (informational)"

banner "5. INFORMATIONAL"
echo "--- 5a. Check B as a gate, at the run-1 tolerance (2e-2) ---"
echo "     Recorded so run 1's marginal cross-arm failure stays on the record"
echo "     rather than being quietly demoted."
$CHK --compare "$A8" "$A4" "$A2" --strict-cross-arm \
    2>&1 | tee "$LOGDIR/strict_b.log"
STRICT=${PIPESTATUS[0]}; echo "rc=$STRICT"

echo
echo "--- 5b. sensitivity floor: a 0.1% single-sequence rescale ---"
$CHK --compare "$A8" "$A4" "$A2" --inject-fault scale_one_seq:1.001 --inject-into 2 \
    2>&1 | tee "$LOGDIR/floor_1e3.log"
FLOOR=${PIPESTATUS[0]}; echo "rc=$FLOOR"

banner "6. SUMMARY"
printf "%-46s %s  %s\n" "comparator self-test (want rc 0)" "$SELFTEST_RC" "$([ "$SELFTEST_RC" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s\n" "emit shards failed (tpu-fanout rc)" "$FANOUT_RC"
printf "%-46s %s  %s\n" "gate: all three arms" "$RC_ALL" "$([ "$RC_ALL" = 0 ] && echo PASS || echo NOT-PASS)"
printf "%-46s %s  %s\n" "gate: arm 4 vs arm 8" "$RC_B4" "$([ "$RC_B4" = 0 ] && echo PASS || echo NOT-PASS)"
printf "%-46s %s  %s\n" "gate: arm 2 vs arm 8" "$RC_B2" "$([ "$RC_B2" = 0 ] && echo PASS || echo NOT-PASS)"
printf "%-46s %s  %s\n" "neg ctl: drop_last_seqs:2 (want rc 0)" "$NC1" "$([ "$NC1" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s  %s\n" "neg ctl: drop_first_seqs:2 (want rc 0)" "$NC2" "$([ "$NC2" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s  %s\n" "neg ctl: scale_one_seq:1.05 (want rc 0)" "$NC3" "$([ "$NC3" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s  %s\n" "neg ctl: self-compare vacuity (want rc 2)" "$NC4" "$([ "$NC4" = 2 ] && echo OK || echo HOLLOW)"
printf "%-46s %s  %s\n" "C-only: drop_last_seqs:2 (want rc 0)" "$CO1" "$([ "$CO1" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s  %s\n" "C-only: drop_first_seqs:2 (want rc 0)" "$CO2" "$([ "$CO2" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s  %s\n" "C-only: scale_one_seq:1.05 (want rc 0)" "$CO3" "$([ "$CO3" = 0 ] && echo OK || echo HOLLOW)"
printf "%-46s %s\n" "C2 tightened to 1.02x (probe)" "$CO4"
printf "%-46s %s\n" "Check B as a gate at 2e-2 (informational)" "$STRICT"
printf "%-46s %s\n" "sensitivity floor probe 0.1% (informational)" "$FLOOR"

banner "7. ARTIFACTS"
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
[ "$NC4" = 2 ] || FAIL=$((FAIL+1))
[ "$CO1" = 0 ] || FAIL=$((FAIL+1))
[ "$CO2" = 0 ] || FAIL=$((FAIL+1))
[ "$CO3" = 0 ] || FAIL=$((FAIL+1))
[ "$RC_ALL" = 0 ] || FAIL=$((FAIL+1))
echo "CONTROLS_AND_GATE_FAILURES=$FAIL"
echo "GATE_RESULT all=$RC_ALL arm4=$RC_B4 arm2=$RC_B2 strictB=$STRICT"
exit $((FAIL > 120 ? 120 : FAIL))
