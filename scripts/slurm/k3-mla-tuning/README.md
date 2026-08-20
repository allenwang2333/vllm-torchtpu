# Kimi-K3 MLA tuning standard

This tuner optimizes the mixed MLA stage used by the Kimi-K3 cold-prefill
workload.  A microbenchmark result is not a serving result.  Candidates move
through the gates below in order; failing a gate excludes the candidate from
all later ranking.

## Reference workload and geometry

The current reference point is TPU v7x-8, TP=8, BF16 MLA, server
`block_size=256`, compile bucket 4096, and client ISL=4000 / OSL=1 / C=1.
The benchmark geometry is taken from the serving compiler boundary, not from
the logical CLI values:

- physical/valid query tokens: `4096/4000`
- local/global query heads: `12/96`
- latent (`ql_nope`) dimension: `512`
- RoPE dimension: `64`
- cache layout: `[493, 128, 2, 640]`
- padded metadata: `max_num_seqs=8`, `pages_per_seq=128`
- flattened block table length: `1024`
- request distribution: mixed-only `[0, 0, 1]`

Any change to model revision, TP, block size, compile bucket, dtype, cache
capacity, or compiler padding creates a new tuning key and requires a new
geometry capture.

## Gates

1. **Geometry gate**
   - Record all dimensions above in the emitted manifest.
   - Assert output shape `[4096, 96, 512]`.
   - The registry key must use `q.shape[-1]` for `actual_lkv_dim`; K3's
     `qk_nope_head_dim=128` is not the latent kernel dimension.
2. **Runtime gate**
   - TPU v7x-8 (`2x2x1`) and model-axis shard-map over 8 devices.
   - JAX 0.10.2, jaxlib 0.10.2, and libtpu 0.0.44.1.
   - Import `vllm_torchtpu.env_override` before JAX so libtpu flags and JAX
     lowering patches match serving.
3. **Kernel compile gate**
   - Compile every candidate at the exact reference geometry.
   - OOM, compiler error, or wrong output shape is a hard rejection and is
     never included in latency ranking.
4. **Kernel ranking gate**
   - Screening: 3 warmups and 20 measured iterations.
   - Finalists: a fresh run with 10 warmups and 50 measured iterations.
   - Rank by median latency, then mean, then P90.  Retain the best two only
     when their median differs by less than 3%; otherwise retain the winner.
5. **Full-model compile gate**
   - Apply finalists through one automated compile matrix using the same
     vLLM, torch-tpu wheel, model revision, serve arguments, and 4096 bucket.
   - A candidate passes only if all TP ranks compile without VMEM OOM and the
     observed mixed `TuningKey` exactly matches the candidate geometry.
   - This gate is feasibility validation, not a latency benchmark.
6. **E2E acceptance gate**
   - Run only the best full-model-safe candidate.
   - Use 2 warmups + 8 measured requests at ISL=4000 / OSL=1 / C=1.
   - Confirm every context iteration has 4000 tokens and no 512/416 chunks.
   - Compare mean, median, P90, and P99 TTFT against the main/block-16
     baseline.  OSL=1 means E2EL is expected to track TTFT.

## Search policy

Tune `num_kv_pages_per_block`, `num_queries_per_block`, and `q_split`.
Start with a small, analytically safe grid.  Expand only around compile-safe
front-runners; do not spend full-model runs discovering basic kernel OOMs.
Keep server parameters (block size, chunk size, compile bucket, TP) fixed for
the entire tuning run.

The standalone tuner is stage 1 only.  No value may be added to
`tuned_params.py` or described as serving-tuned until gates 1-5 pass.
