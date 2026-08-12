#!/bin/bash
# Byte-level cache differential, ONE model load: decode generation followed
# by teacher-forced prefills of the same tokens in the same process
# (DSV4_TF_AFTER_PROMPTS), with block-table-indexed page dumps at the debug
# layer (DSV4_DUMP_PAGES). Prefill is bit-deterministic, so byte-equality
# between decode-written and prefill-written cache rows is a legitimate
# assertion; any mismatch beyond quantization is a decode-path defect.
# This is the instrument that pinned the decode-corruption root cause --
# docs/dsv4_decode_race_postmortem.md.
#
# Usage: chain_cache_diff.sh [prompt_file] [max_tokens] [layer] [label]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT="${1:-$HERE/prompts/xla_llvm_prompt.txt}"
MAXTOK="${2:-150}"
LAYER="${3:-2}"
LABEL="${4:-cache_diff_L${3:-2}}"
SCRATCH="${DSV4_SCRATCH:-$HOME/workspace/scratch}"
OUT="$SCRATCH/cache_diff"
mkdir -p "$OUT"

for _ in $(seq 1 400); do
  pgrep -f "examples/run_dsv4_inference.py" >/dev/null 2>&1 || break
  sleep 15
done
sleep 30

echo "== decode leg (generates), extracting ids, then TF prefills, one load"
# First a quick decode-only pass is NOT needed: run decode, then extract ids
# from the SAME process's output? The TF file must exist before launch, so
# this chain does one short decode first to harvest token ids, then the
# combined run. If you already have a tf json, pass DSV4_TF_JSON to skip.
if [ -z "${DSV4_TF_JSON:-}" ]; then
  bash "$HERE/run_ours.sh" "$OUT/harvest.log" \
    DSV4_MAX_TOKENS="$MAXTOK" DSV4_PROMPT_FILES="$PROMPT"
  python3 - "$OUT" "$MAXTOK" <<'EOF'
import re, json, sys
out, maxtok = sys.argv[1], int(sys.argv[2])
def get(path, pat):
    for line in open(path, errors="replace"):
        if pat in line:
            m = re.search(r"\[([\d, ]+)\]", line)
            return [int(x) for x in m.group(1).split(",")]
prompt = get(f"{out}/harvest.log", "Prompt token IDs")
gen = get(f"{out}/harvest.log", "Generated token IDs")
# score only the first ~14 contexts: enough to cover several compressed
# groups while keeping the TF pass fast
json.dump({"label": "cachediff", "prompt_token_ids": prompt,
           "forced_token_ids": gen[:14]},
          open(f"{out}/tf_tokens.json", "w"))
print(f"prompt={len(prompt)} forced=14 (of {len(gen)})")
EOF
  TFJSON="$OUT/tf_tokens.json"
  for _ in $(seq 1 400); do
    pgrep -f "examples/run_dsv4_inference.py" >/dev/null 2>&1 || break
    sleep 15
  done
  sleep 30
else
  TFJSON="$DSV4_TF_JSON"
fi

rm -rf "$SCRATCH/page_dumps/$LABEL"
bash "$HERE/run_ours.sh" "$OUT/combined.log" \
  DSV4_MAX_TOKENS="$MAXTOK" \
  DSV4_PROMPT_FILES="$PROMPT" \
  DSV4_TF_FILES="$TFJSON" \
  DSV4_TF_AFTER_PROMPTS=1 \
  DSV4_DEBUG_LAYER="$LAYER" \
  DSV4_DUMP_PAGES=1 \
  DSV4_DUMP_LABEL="$LABEL"

echo "== diff"
python3 "$HERE/diff_cache_pages.py" "$LABEL"
