#!/bin/bash
# Run a pool of prompts through both trees and diff the output distributions.
#
# One prompt is a bad instrument. On the XLA/LLVM prompt the two trees split at
# step 13 where our own top-1/top-2 margin was 1.25 -- close enough to a coin
# flip that it measures luck as much as correctness. A pool tells you whether
# divergence is systematic, and which prompts are worth localizing with (the
# ones where the reference was confident and we still disagreed).
#
# Both harnesses take a comma-separated DSV4_PROMPT_FILES and run each prompt
# in its own generate() call against a single model load, so this costs two
# loads total rather than two per prompt.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL="$HERE/prompts/pool"
OUT="${DSV4_LOG_DIR:-$HERE}"
MAX_TOKENS="${DSV4_POOL_MAX_TOKENS:-120}"

PROMPTS=$(ls "$POOL"/*.txt | tr '\n' ',' | sed 's/,$//')
n=$(ls "$POOL"/*.txt | wc -l)
echo "pool: $n prompts, $MAX_TOKENS tokens each"

wait_for_tpu() {
  for _ in $(seq 1 400); do pgrep -f "^VLLM::" >/dev/null 2>&1 || break; sleep 15; done
  sleep 20
}

wait_for_tpu
echo "########## OURS ##########"
date
bash "$HERE/run_ours_needle.sh" "$OUT/pool_ours.log" \
  DSV4_MAX_TOKENS="$MAX_TOKENS" DSV4_PROMPT_FILES="$PROMPTS"

wait_for_tpu
echo "########## REFERENCE ##########"
date
bash "$HERE/run_ref_needle.sh" "$OUT/pool_ref.log" \
  DSV4_MAX_TOKENS="$MAX_TOKENS" DSV4_PROMPT_FILES="$PROMPTS"

echo "########## GENERATED TEXT ##########"
# Text first: it is the one readout nothing in this stack has managed to
# distort, and a glance at it usually tells you more than the table below.
for tag in ours ref; do
  echo "--- $tag"
  grep -E "^\[SWEEP\] [0-9a-z_]+\.txt:" "$OUT/pool_$tag.log" | cut -c1-200
done

echo "########## DISTRIBUTION DIFF ##########"
python3 "$HERE/diff_logprobs.py" "$OUT/pool_ours.log" "$OUT/pool_ref.log"
