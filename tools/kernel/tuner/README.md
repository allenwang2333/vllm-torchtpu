# MLA kernel tuner

Finds the MLA v2 kernel's decode block sizes for one model geometry and prints
`tuned_params_mapping` entries you can paste into
`src/vllm_torchtpu/kernels/mla/v2/tuned_params.py`.

A local port of tpu-inference's `tools/kernel/tuner/v1/mla_kernel_tuner.py`.

## Running it

Always dry-run first. A TPU allocation is expensive and a geometry typo is not
worth one:

```bash
PYTHONPATH=$PWD/src python tools/kernel/tuner/mla_kernel_tuner.py \
  --mla_actual_num_q_heads 3 --mla_actual_lkv_dim 512 --mla_actual_r_dim 64 \
  --mla_kv_dtype bfloat16 --mla_q_dtype bfloat16 \
  --mla_page_size_per_kv_packing 8 --mla_kv_packing 2 \
  --mla_max_num_seqs 8 --mla_pages_per_seq 2048 --mla_total_num_pages 30311 \
  --max_num_tokens 2,16,512 --dry_run
```

Drop `--dry_run` on a machine with a TPU. The sweep is single-device.

## Getting the geometry right

Every flag corresponds to a `TuningKey` field, and a lookup matches only when
**every** field agrees. The values are properties of the deployment, not of the
model card, so read them off a running server rather than deriving them:

| flag | where it comes from | Kimi-K3, TP=32 |
|---|---|---|
| `--mla_actual_num_q_heads` | per-rank attention heads | 3 |
| `--mla_actual_lkv_dim` | `ql_nope.shape[-1]`, i.e. `kv_lora_rank` | 512 |
| `--mla_actual_r_dim` | `q_pe.shape[-1]`, i.e. `qk_rope_head_dim` | 64 |
| `--mla_kv_dtype` / `--mla_q_dtype` | cache and query dtypes | bfloat16 |
| `--mla_page_size_per_kv_packing` | KV cache `shape[1]` | 8 |
| `--mla_kv_packing` | KV cache `shape[2]` | 2 |
| `--mla_max_num_seqs` | `--max-num-seqs` | 8 |
| `--mla_pages_per_seq` | `ceil(max_model_len / block_size)` | 2048 |
| `--mla_total_num_pages` | KV cache `shape[0]` | 30311 |
| `--max_num_tokens` | the `compile_sizes` buckets | 2,16,512 |

The quickest way to read them is the KV cache layout the server logs at
startup: Kimi-K3 at TP=32 with `block_size` 16 shows
`bf16[30311, 8, 2, 640]`, which is `total_num_pages`,
`page_size_per_kv_packing`, `kv_packing` and the padded latent width in order.

**The unified KV pool changes the paging.** On the pool K3 runs `block_size`
162, so the cache is `bf16[3078, 81, 2, 640]` and the geometry becomes
`--mla_page_size_per_kv_packing 81 --mla_pages_per_seq 203
--mla_total_num_pages 3078`. That is a different key and needs its own entries.

## Differences from upstream

Input generation and the search space are carried over unchanged, so numbers
are comparable. What is not carried over is the harness -- Buildkite pipeline
generation, the Spanner backend, the multi-process executor and the optuna
Bayesian optimizer, about 3,000 lines that a local single-chip sweep does not
need (and optuna is not installed here).

Five deliberate changes:

1. **The key is built by `mla_tuning_key`**, the same function `mla_attention`
   uses at serving time. Upstream assembles a `TuningKey` from flags
   independently of the call site, which is how all 61 shipped entries came to
   be unreachable -- they carry `kv_packing` of 32, 576 or 1024, and
   `kv_packing` is the cache's packing axis, so no invocation produces them.
   Sharing the constructor makes a generated entry reachable by construction.
2. **`max_num_tokens` is a flag**, not upstream's hardcoded
   `[4, 8, ..., 2048]`. That list has no 2, and K3 compiles buckets 2/16/512.
3. **Every configuration is checked against a reference**, not just timed. A
   tiling that computes a different answer is not a speedup. Changing the KV
   block count reassociates the online-softmax accumulation and moves the
   result by about one bf16 ulp; anything larger is a bug, and it shows in the
   `maxdiff` column.
4. **The kernel call is jitted** and the cache input is not rebound between
   iterations, so the loop times the kernel rather than Python dispatch, and
   every iteration times the same problem.
5. **`--max_kv_pages_per_block` caps the KV axis.** Upstream derives the
   ceiling from `pages_per_seq`, which is fine at its default of 9 but gives 12
   values for K3's 2048, most certain to exhaust VMEM. Whatever the cap removes
   is logged.

## Known limitations

- **Decode only.** `num_queries_per_block` is pinned to 1 and every generated
  sequence has `q_len=1`, exactly as upstream. The prefill/mixed tiling is not
  addressed, and on K3 that is the larger prize: the mixed case is 54% of a
  prefill step, and moving it off the hardcoded `(16, 1)` measured 2.39x on
  end-to-end TTFT. Tuning it needs a mixed-case generator -- vary
  `num_queries_per_block`, build inputs with `q_len > 1`, and set
  `distribution=[0, 0, num_seqs]`.
- **`max_num_tokens` does not reach the inputs.** Upstream sets `total_q_len`
  to `max_num_seqs`, so all buckets benchmark the same problem. This measures
  once and emits an identical entry per bucket rather than paying for identical
  sweeps. Reproducing each bucket's real padded query buffer would be more
  faithful and needs a TPU to validate the padding convention against a live
  step.
