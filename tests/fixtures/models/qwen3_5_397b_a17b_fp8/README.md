# Qwen3.5-397B dummy-load fixture

`config.json` is the Hugging Face configuration for
`Qwen/Qwen3.5-397B-A17B-FP8`. No tokenizer files or model weights are stored in
this fixture.

The TPU compile-reproduction CI starts vLLM with `--load-format dummy` and
`--skip-tokenizer-init`, with Hugging Face network access disabled. Both
isolated cases use the same success condition: the server must become healthy.

- `split`: all 45 PCP GDN ops are listed in vLLM `splitting_ops`; the server
  is expected to start, so this job should pass.
- `full`: `splitting_ops` is empty; the current full graph is expected not to
  start, so this job should fail and its uploaded server log can be inspected.

The launcher does not classify the full-graph failure or require a particular
error message. It only waits for the health endpoint and returns failure when
the service exits or does not become healthy.

The launcher and exact compilation configs are in
`scripts/vllm/integration/run_qwen35_397b_pcp8_compile_repro.sh`.
