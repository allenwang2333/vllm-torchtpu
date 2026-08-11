#!/bin/bash
# Ground-truth check: real long-form generation, ours vs the JAX reference,
# on the identical prompt/config -- NO dumping, NO probes.
#
# Why this and not more instrumentation: the original symptom was behavioural
# (docs/dsv4_torch_compile_investigation_20260810.md section 9) -- on
# "Explain the difference between XLA and LLVM in one sentence", tpu-inference
# produced one clean correctly-terminated sentence while both our eager and
# compiled paths rambled across multiple </think> spans. Every activation
# probe this session has been distorted by at least one instrument artifact
# (warmup/prefill step offset; DSV4_DUMP_HIDDEN_STATES swapping
# DeepseekV4DecoderLayer.forward outright). Generated text is the one
# measurement no probe can corrupt.
#
# max_tokens=150 because the divergence only becomes visible over long
# generation: at 2-16 tokens both trees emit byte-identical output.
set -u
SP=${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu}/scratchpad
PROMPT_FILE=$SP/xla_llvm_prompt.txt

cat > "$PROMPT_FILE" <<'EOF'
Explain the difference between XLA and LLVM in one sentence.
EOF

wait_for_tpu() {
  for _ in $(seq 1 400); do pgrep -f "^VLLM::" >/dev/null 2>&1 || break; sleep 15; done
  sleep 20
}

echo "########## commit under test ##########"
git -C ${VLLM_TORCHTPU_ROOT:-$HOME/workspace/vllm-torchtpu} log --oneline -1

wait_for_tpu
echo "########## OURS (no dumping, no probes) ##########"
date
bash "$SP/run_ours_needle.sh" "$SP/runAI_ours_real.log" \
  DSV4_MAX_TOKENS=150 \
  DSV4_PROMPT_FILES="$PROMPT_FILE"

echo "########## REFERENCE ##########"
date
wait_for_tpu
bash "$SP/run_ref_needle.sh" "$SP/runAJ_ref_real.log" \
  DSV4_MAX_TOKENS=150 \
  DSV4_PROMPT_FILES="$PROMPT_FILE"

echo "########## OURS OUTPUT ##########"
grep -A40 "Generated text" "$SP/runAI_ours_real.log" | head -50
echo "########## REFERENCE OUTPUT ##########"
grep -A40 "Generated text" "$SP/runAJ_ref_real.log" | head -50
