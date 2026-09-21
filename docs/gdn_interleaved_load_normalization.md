# GDN checkpoint layout normalization

Interleaved Qwen3Next checkpoints and contiguous Qwen3.5 checkpoints now
produce the same rank-local runtime layout:

- QKVZ: all local Q, then K, V and Z channels.
- BA: all local B, then A channels.

The interleaved loader selects complete Q/K groups, replicates Q/K when TP
exceeds the number of key heads, and partitions V/Z and B/A among replicas.
It flattens each selected segment's groups before concatenating segments.
Channel and block scales follow the same permutation as the weight rows.
Per-tensor scales remain shared; unsupported block/head alignment still
raises instead of requantizing or selecting a fallback.

Only projection construction selects a loader based on the checkpoint's
layout. Ordinary forward splits canonical projection results; pooled PCP
forward uses the same canonical BA split. The PCP projection kernel reads
the existing contiguous weight tiles. Neither the core wrapper nor the
kernel accepts an interleaved-layout argument anymore.

Conv weights retain their existing canonical Q/K/V layout and head ownership.
Conv and recurrent state layouts, communication, and GDN recurrence are unchanged.
No enable flag or alternate computation fallback is introduced.

## Validation on 2026-09-21

Logs are retained under `../runs/gdn_layout_refactor/` in the local workspace.

- Before implementation, canonical-layout loader assertions failed in six
  BF16/FP8 weight and scale cases with multiple Q/K groups per TP rank.
- After implementation, 165 loader/adapter CPU cases passed, including
  TP1/2/4/8/16, replicated and non-replicated heads, and FP8 tensor/channel/block
  scales. Both checkpoint layouts exercise the same forward interface.
- Two eight-device SPMD cases load actual interleaved tensors through the
  loader and FP8 postprocessing, then compare TP8 against TP1. BF16 and
  block-scaled FP8 passed prefill, continuation and three decode steps,
  including conv and recurrent state checks.
- Four PCP SPMD numerical cases passed: BF16/block-scaled FP8, multiple
  groups per shard and PCP8 Q/K replication, using canonical runtime weights.
- Full relevant pre-commit checks passed.

The first PCP SPMD invocation did not initialize because setting
`TPU_ACCELERATOR_TYPE` without worker hostnames caused invalid hostname
discovery. The retry explicitly supplied `TPU_WORKER_HOSTNAMES=127.0.0.1`
and `TPU_WORKER_ID=0`, alongside `TPU_SKIP_MDS_QUERY=true`; no code change was
needed for that initialization error.

## Single-host serving acceptance

Use `scripts/qwen38_c3_smoke/start.sh` and its `request_smoke.py` from this
checkout. The existing c3 smoke configuration has 12 layers, linear key/value
heads 4/32 (one quarter of the full model's 16/128), and 128 routed experts.
FA heads remain 64/4, hidden size is 8192, and expert top-k is 10.

The serving configuration is dummy load, PCP4TP2/EP8, adaptive fused MoE,
compile bucket 4096, global token budget 16384, HND FP8 KV, BF16 SSM,
manager block 384, interleave 128, 512 KV blocks, max length 32768 and
max sequences 4. The endpoint is `http://127.0.0.1:18201`.

Run `scripts/qwen38_c3_smoke/runs/20260921-111318-c3-pcp4tp2` reached ready
at 11:18:04; health returned HTTP 200. All six requests returned HTTP 200
with exactly the requested prompt and completion token counts:

| Request | Completion tokens | Concurrency | Elapsed seconds |
| --- | --- | --- | --- |
| 128 tokens | 8 | 1 | 2.877 |
| 8192 tokens | 1 | 1 | 0.254 |
| 8192 tokens, request 1 | 1 | 2 | 0.250 |
| 8192 tokens, request 2 | 1 | 2 | 0.488 |
| 8192 tokens | 8 | 1 | 2.546 |
| 53 tokens | 3 | 1 | 0.978 |

These timings include first-use work and are functional smoke measurements,
not a performance benchmark. Requests, responses, `smoke_results.json` and
`smoke_continuation_results.json` are saved in the run directory alongside
the launch command, model configuration, source diff and service log.
No additional code or parameter workaround was required during this startup.
The service is left running on port 18201.

Dummy serving validates compilation, execution and state continuation; it
does not validate checkpoint loading or semantic output. The real-tensor
loader and numerical tests above cover normalization correctness separately.
