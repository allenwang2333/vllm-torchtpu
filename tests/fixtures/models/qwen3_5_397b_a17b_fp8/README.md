# Qwen3.5-397B dummy-load fixture

`config.json` is the Hugging Face configuration for
`Qwen/Qwen3.5-397B-A17B-FP8`. No tokenizer files or model weights are stored in
this fixture.

The TPU compile-reproduction CI starts vLLM with `--load-format dummy` and
`--skip-tokenizer-init`, with Hugging Face network access disabled. Its two
isolated cases intentionally have different expected outcomes:

- `split`: all 45 PCP GDN ops are listed in vLLM `splitting_ops`; the server
  must become healthy and the log must show `subgraph_45`.
- `full`: `splitting_ops` is empty; the current full graph must reproduce
  `RuntimeUnexpectedCoreHalt` during startup warmup.

The full-graph failure is treated as an expected reproduction, so the CI case
passes only when that error is observed. If full-graph compilation is fixed in
the future, update the expected outcome instead of preserving the failure.

The launcher and exact compilation configs are in
`scripts/vllm/integration/run_qwen35_397b_pcp8_compile_repro.sh`.
