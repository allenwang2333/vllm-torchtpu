# DeepSeek-V4-Flash serving + MMLU-Pro on vllm-torchtpu

```
scripts/dsv4/dsv4_env.sh      shared env + tunables (source it, don't run it)
scripts/dsv4/serve_dsv4.sh    start the server, wait for /health
scripts/dsv4/run_mmlu_pro.sh  server + lm_eval mmlu_pro + teardown
```

Smoke test first, then the full run:

```bash
LIMIT=10 scripts/dsv4/run_mmlu_pro.sh mmlu_pro_smoke
scripts/dsv4/run_mmlu_pro.sh mmlu_pro_full
```

Venv: `$WORKSPACE/uv_venv_dsv4` (vllm 0.26.0, libtpu 0.0.44.1, torch_tpu
0.1.1.dev20260728, lm_eval). Do not use `$WORKSPACE/uv_venv` — it no longer
exists, which is why the older `scratch/run_mmlu_*.sh` scripts cannot run.

## Mapping from the tpu-inference reference command

The reference run is:

```
MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 MOE_REQUANTIZE_BLOCK_SIZE=512 NEW_MODEL_DESIGN=1 \
MODEL_IMPL_TYPE=vllm vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=128 \
  --kv-cache-dtype=fp8 --no-enable-prefix-caching --gpu-memory-utilization=0.4 \
  --tensor-parallel-size=8 --download_dir=/mnt/pd --async-scheduling \
  --enable-expert-parallel \
  --additional_config='{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}'
```

Identical here: the four env vars, `--max-model-len`, `--max-num-batched-tokens`,
`--max-num-seqs`, `--no-enable-prefix-caching`, `--tensor-parallel-size`,
`--enable-expert-parallel`.

| Reference | Here | Why |
|---|---|---|
| `--kv-cache-dtype=fp8` | `fp8_e4m3` | Same dtype; vLLM's TPU path wants the explicit name. |
| `--gpu-memory-utilization=0.4` | `0.75` (`GPU_MEM_UTIL`) | Not real memory pressure: vLLM's DSv4-specific KV budget check over-estimates the per-request requirement by ~62x, and the engine refuses to start. Real need at 9216 is 0.69 GiB. See `docs/dsv4_status.md`, "Larger runs". |
| `--download_dir=/mnt/pd` | local snapshot path | Weights are already on this node under `hf_cache`; nothing is downloaded. |
| `--async-scheduling` | same (on) | vLLM 0.26 enables async scheduling by default on this path -- the server log reports "Asynchronous scheduling is enabled" regardless of the flag. Passed explicitly to match the reference. |
| `enable_dp_attention: true` | **no equivalent** | That `additional_config.sharding` block is a JAX/tpu-inference sharding-strategy option. vllm-torchtpu has no DP-attention implementation at all (no `dp_attention` anywhere in `src/`), so attention runs TP-only. This is the one genuine non-1:1 difference in the serving config, and it is a parallelism strategy, not a numerics change. |
| (block size unset) | unset | The TPU platform derives it (`update_block_size_for_backend`). Do not pin `--block-size 512` the way the old scratch scripts did — page sizing is what caused the HCA VMEM compile failures. `BLOCK_SIZE=` is available for experiments. |

The lm_eval side matches the reference exactly (`local-completions`, `mmlu_pro`,
`--num_fewshot 5`, `--batch_size auto`, `--seed 42`, `num_concurrent=128`,
`max_gen_toks=4096`, `max_length=8192`), plus a local `tokenizer=` path so no
tokenizer is fetched from the network.

## What is and isn't verified

Verified on 2026-08-08, after the zero-MoE-weight fix (`665006c8`): the needle
test passes with generated token IDs byte-identical to tpu-inference — at
`max_model_len=1024`, `max_tokens=16`, eager, TP=8, via the **offline** harness
`examples/run_dsv4_inference.py`.

Not yet exercised since that fix, all of which this benchmark turns on at once:
the online server path, `max_model_len=9216`, chunked prefill
(`max_num_batched_tokens=1024` < prompt length), `max_num_seqs=128` batching,
and non-eager compilation. Run `LIMIT=10` first and read the accuracy before
committing to a full sweep.

Known-open numerics issue: routed-MoE output matches the reference in direction
but is ~1.3x high in magnitude (cos ~0.986, relL2 ~0.28). Suspects are fp4
requantization, the swiglu clamp, or a double-applied scale. If MMLU-Pro comes
in materially below the reference, that is the first place to look.
