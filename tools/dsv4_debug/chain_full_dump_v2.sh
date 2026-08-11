#!/bin/bash
# Full-stack matched dump, ours vs reference, on the CURRENT commit.
#
# Why re-run rather than reuse what's on disk:
#   - The vllm_torchtpu dumps were overwritten 2026-08-10 23:28 by a run on a
#     DIFFERENT prompt, so the on-disk ours/ref pair is not comparable.
#   - The prior full dump (runR/runS) predates the current commit; it may have
#     been measuring an older bug.
# Both trees are re-run here, back to back, same prompt, all 43 layers.
#
# NOTE ON EAGER: all three dump paths bail on torch.compiler.is_compiling(),
# so a dumping run is necessarily EAGER. That is not a limitation of this
# comparison -- the compile investigation (doc section 9) established that our
# eager and compiled paths both ramble, so the divergence is present in eager.
set -u
SP=${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu}/scratchpad
HS=${DSV4_SCRATCH:-$HOME/workspace/scratch}/hidden_states
PROMPT=$SP/needle_250.txt

wait_for_tpu() {
  for _ in $(seq 1 400); do pgrep -f "^VLLM::" >/dev/null 2>&1 || break; sleep 15; done
  sleep 20
}

echo "########## commit under test ##########"
git -C ${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu} log --oneline -1
git -C ${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu} status --short

wait_for_tpu
rm -rf "$HS/vllm_torchtpu" "$HS/tpu_inference" \
       ${DSV4_SCRATCH:-$HOME/workspace/scratch}/moe_dumps

echo "########## OURS: all 43 layers + MoE internals ##########"
date
bash "$SP/run_ours_needle.sh" "$SP/runAB_ours_full.log" \
  DSV4_DUMP_HIDDEN_STATES=1 DSV4_DUMP_LAYERS=all DSV4_DEBUG_MOE_FULL=1 \
  DSV4_MAX_TOKENS=2 \
  DSV4_PROMPT_FILES="$PROMPT"

echo "########## REFERENCE: all 43 layers ##########"
date
wait_for_tpu
bash "$SP/run_ref_needle.sh" "$SP/runAC_ref_full.log" \
  DSV4_DUMP_HIDDEN_STATES=1 DSV4_DUMP_LAYERS=all \
  DSV4_MAX_TOKENS=2 \
  DSV4_PROMPT_FILES="$PROMPT"

echo "########## INVENTORY ##########"
date
echo "ours: $(ls $HS/vllm_torchtpu 2>/dev/null | wc -l) files"
echo "ref:  $(ls $HS/tpu_inference 2>/dev/null | wc -l) files"
echo "moe:  $(ls ${DSV4_SCRATCH:-$HOME/workspace/scratch}/moe_dumps 2>/dev/null | wc -l) files"

echo "########## FULL-STACK DIFF (cos / relL2 / norm-ratio) ##########"
${WORKSPACE:-$HOME/workspace}/uv_venv_dsv4/bin/python3 "$SP/diff_full.py"
