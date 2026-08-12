#!/bin/bash
# Prefill-vs-decode behavioral A/B -- the zero-instrumentation discriminator
# that localized the decode-corruption bug (see
# docs/dsv4_decode_race_postmortem.md). Run this FIRST whenever decode output
# looks worse than short/prefill-scored output.
#
#   1. decode leg: free-running generation on the prompt (N tokens)
#   2. TF leg: the SAME generated tokens re-scored via growing-prefix
#      prefills (DSV4_TF_FILES)
#   3. diff: per-step logprob of the decode-picked token, decode-run vs
#      prefill-scored. A healthy stack matches within reporting quantization
#      at every step (the reference does); growing or multi-logit disparity
#      means decode-built state is corrupt.
#
# Usage: chain_prefill_decode.sh [prompt_file] [max_tokens] [out_dir]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT="${1:-$HERE/prompts/xla_llvm_prompt.txt}"
MAXTOK="${2:-150}"
OUT="${3:-${DSV4_SCRATCH:-$HOME/workspace/scratch}/prefill_decode_ab}"
mkdir -p "$OUT"

wait_for_tpu() {
  for _ in $(seq 1 400); do
    pgrep -f "examples/run_dsv4_inference.py" >/dev/null 2>&1 || break
    sleep 15
  done
  sleep 30
}

echo "== leg 1: decode generation ($PROMPT, $MAXTOK tokens)"
wait_for_tpu
bash "$HERE/run_ours.sh" "$OUT/decode.log" \
  DSV4_MAX_TOKENS="$MAXTOK" DSV4_PROMPT_FILES="$PROMPT"

echo "== extracting token ids"
python3 - "$OUT" <<'EOF'
import re, json, sys
out = sys.argv[1]
def get(path, pat):
    for line in open(path, errors="replace"):
        if pat in line:
            m = re.search(r"\[([\d, ]+)\]", line)
            return [int(x) for x in m.group(1).split(",")]
prompt = get(f"{out}/decode.log", "Prompt token IDs")
gen = get(f"{out}/decode.log", "Generated token IDs")
assert prompt and gen, "token id lines missing from decode.log"
json.dump({"label": "ab", "prompt_token_ids": prompt,
           "forced_token_ids": gen}, open(f"{out}/tf_tokens.json", "w"))
print(f"prompt={len(prompt)} generated={len(gen)}")
EOF

echo "== leg 2: teacher-forced prefill scoring of the same tokens"
wait_for_tpu
bash "$HERE/run_ours.sh" "$OUT/tf.log" DSV4_TF_FILES="$OUT/tf_tokens.json"

echo "== diff"
python3 "$HERE/diff_teacher_forcing.py" "$OUT/decode.log" "$OUT/tf.log"
