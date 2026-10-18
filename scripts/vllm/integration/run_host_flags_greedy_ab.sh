#!/usr/bin/env bash
# Runs on the pod: arm off, arm on, arm off again (noise / determinism check), compare.
set -uo pipefail
cd /root
O=${CDK_OUTPUT_DIR:-/tmp}/ab; mkdir -p "$O"
S=${AB_SCRIPT:-/root/cloud-devkit/vllm-torchtpu/scripts/vllm/integration/host_flags_greedy_ab.py}
export TPU_MOE_SKIP_PADDED_TOKENS=1
python3 $S run "$O/off1.json" > "$O/off1.log" 2>&1; echo "off1 rc=$?"; grep -E "AB_RUN|Error|error:" "$O/off1.log" | tail -5
env TPU_DISABLE_RUNNER_STAGE_TIMING=1 TPU_SKIP_UNCHANGED_PADDING_UPLOAD=1 TPU_DRAFT_LOOP_CACHE_UPLOADS=1 TPU_GC_STEP_ALIGNED_COLLECT=16 \
  python3 $S run "$O/on.json" > "$O/on.log" 2>&1; echo "on rc=$?"; grep -E "AB_RUN|Automatic GC off|Error|error:" "$O/on.log" | tail -5
python3 $S run "$O/off2.json" > "$O/off2.log" 2>&1; echo "off2 rc=$?"; grep -E "AB_RUN|Error|error:" "$O/off2.log" | tail -5
echo "== off1 vs off2 (determinism)"; python3 $S compare "$O/off1.json" "$O/off2.json"; D=$?
echo "== off1 vs on"; python3 $S compare "$O/off1.json" "$O/on.json"; R=$?
echo "determinism_rc=$D ab_rc=$R"
exit $R
