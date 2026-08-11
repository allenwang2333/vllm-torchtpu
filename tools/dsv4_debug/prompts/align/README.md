# Alignment probe prompts

Same question, same correct answer, four different prompt lengths chosen so
`prompt_tokens % 4` covers 0, 1, 2 and 3 (the filename records which).

CSA compresses tokens in groups of 4, and `main_kv_lens = seq_len // 4`, so a
prompt whose length isn't a multiple of 4 leaves a partial group straddling the
prefill/decode boundary — started by prefill, finished by decode. An aligned
prompt never creates one.

The pool comparison couldn't separate that from prompt difficulty: the only
clean output was also by far the easiest question. These four hold difficulty
fixed so alignment is the only thing moving.

Run all four in one load:

    run_ours.sh out.log DSV4_MAX_TOKENS=120 \
      DSV4_PROMPT_FILES="$(ls prompts/align/a*.txt | tr '\n' ',' | sed 's/,$//')"

Read the generated text first. If only the mod-0 prompt is clean, alignment is
implicated. If all four are clean or all four degrade, it isn't.
