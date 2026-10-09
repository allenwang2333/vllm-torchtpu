# Qwen3.8-2.4T on 32 tpu7x cores: PCP16xTP2 (baseline) vs PP4xPCP8 (exp)

Part 1 compares disaggregated 1P1D serving (prefill engine feeding a TP32 + DCP8 decode engine over Raiden Stage-3). Part 2 compares prefill-only performance of the same two layouts.

# Part 1: disaggregated serving, 1P1D

## Setup

- Prefill engine (32 cores, 2x2x4 slice): baseline = PCP16xTP2; exp = PP4xPCP8 (dynamic chunks + lazy push enabled).
- Decode engine (32 cores, 2x2x4 slice, identical on both legs): TP32 + DCP8, EP32, `--async-scheduling`, max 256 sequences, block size 384, fp8 KV.
- KV hand-off: Raiden Stage-3 (seq-on-lane layout), resharding 32 producer ranks into 32 decode shards x 8 DCP lanes, with per-layer pool pairing on pipeline stages (`TPU_RAIDEN_POOL_TAGS_PER_LAYER=1`).
- Δ = (exp − baseline) / baseline. Positive Δ tok/s means exp is faster; negative Δ TTFT, TPOT, ITL, or fill time means exp is faster.
- Throughput and TTFT: Cases 1-2 report steady-state values (completions K + 1 to N − K, K = max(10, c)), except c256 (n = 512 < 3c, whole-pass). Prefix caching is on across all runs, with unique prompts in Cases 1, 2, and 4, and prefix reuse in Case 3.

Code under test: branch `benliangw/pp-pcp-head` in vllm-torchtpu, which is `origin/main` at `b94bc6d4f` (2026-10-07) plus:

- `origin/yinlin/pp-pcp-stages-v2`: [PR #1345](https://github.com/vllm-project/vllm-torchtpu/pull/1345) (pipeline hand-off through `jax_op`) and [PR #1346](https://github.com/vllm-project/vllm-torchtpu/pull/1346) (a pipeline stage spans a prefill context-parallel group)
- `origin/yinlin/pp-lazy-push`: [PR #1481](https://github.com/vllm-project/vllm-torchtpu/pull/1481) (engine pushes the pipeline only when stage 0 would idle, `TPU_PP_LAZY_PUSH=1`)
- `origin/yinlin/pp-pcp-dcp-reshard`: [PR #1677](https://github.com/vllm-project/vllm-torchtpu/pull/1677) (PP x PCP prefill KV and GDN state resharded into TP x DCP decode)
- `origin/yinlin/pp-raiden-early-registration`: [PR #982](https://github.com/vllm-project/vllm-torchtpu/pull/982) (pipeline producer registers KV blocks in the prompt-completing step)
- `origin/benliangw/tpu-worker-gc-freeze`: [PR #1761](https://github.com/vllm-project/vllm-torchtpu/pull/1761) (freeze the GC heap after TPU worker warmup)
- Two cherry-picks from `origin/handsomehow/qwen38_pp4pcp8_bench` (no PR): `af1ae5ed5` (full-width Qwen GDN and FA inside PCP stages) and `9c5926f02` (dummy-load MoE routing workaround)

raiden wheel `tpu_sync_torch-0.0.1.dev20261007210147-1_pp_reshard_0923_55a23251`, torch_tpu wheel `torch_tpu-0.1.1.dev20260928182719`, image `torchtpu-vllm-dev:nightly-20261001`, vLLM v0.29.0.

## Case 1: fresh 8k prompts, OSL 1, through the proxy (prefill-bound)

| conc | tok/s baseline | tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 22.4K | 57.0K | +154% | 2.9 s | 1.2 s | −61% |
| 32 | 37.7K | 65.0K | +73% | 7.0 s | 4.1 s | −42% |
| 64 | 37.6K | 69.8K | +86% | 14.0 s | 7.5 s | −46% |
| 256 | 37.2K | 66.4K | +79% | 55.4 s | 29.9 s | −46% |

KV hand-off and 1 decode step add ~0.2 s to TTFT on both legs. Exp nears saturation at c8, whereas baseline requires c32.

## Case 2: 120k cold tokens, OSL 1, through the proxy (prefill-bound)

| conc | tok/s baseline | tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 27.5K | 40.5K | +47% | 36.1 s | 24.5 s | −32% |

## Case 3: 120k prefixes warmed, then revisited through the proxy, KV pool sized by vLLM

Free-HBM prefill pools hold 163 prefixes on PP4xPCP8 vs. 118 on PCP16xTP2 (see Part 2 KV cache pool size). Each pass warms N prefixes (122880 tokens + 1024 new tokens, OSL 1, concurrency 8), then revisits each prefix 4 times in random order (122880-token prefix + 8192 new tokens, OSL 1, concurrency 8).

| prefixes | requests | hit share baseline | hit share exp | input tok/s baseline | input tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 60 | 240 | 100% | 100% | 181.5K | 321.4K | +77% | 5.7 s | 3.3 s | −43% |
| 90 | 360 | 83% | 100% | 93.2K | 328.7K | +253% | 7.1 s | 3.2 s | −54% |
| 120 | 480 | 46% | 100% | 44.8K | 307.6K | +586% | 23.8 s | 3.3 s | −86% |
| 150 | 600 | 25% | 65% | 34.4K | 92.0K | +167% | 32.1 s | 11.1 s | −65% |

At 60 prefixes both pools hit 100% (+77% tok/s, TTFT −43%). At 90-120 prefixes PCP16xTP2 evicts (83% and 46% hit) while PP4xPCP8 holds 100%, widening the gain to +253% and +586% (TTFT −54% and −86%). At 150 prefixes both pools evict (25% vs. 65% hit, +167% tok/s, TTFT −65%).

## Case 4: decode passes at concurrency 256

| pass | metric | baseline | exp | Δ |
|:---:|---|:---:|:---:|:---:|
| 8k in / 256 out, n 512 (prefill-bound) | fill to 256 active | 59.5 s | 33.8 s | −43% |
| | whole-pass input tok/s | 33.0K | 53.2K | +61% |
| | whole-pass output tok/s | 1032 | 1664 | +61% |
| | ITL p50 | 50.1 ms | 76.6 ms | +53% |
| | TTFT p50 | 41.0 s | 11.2 s | −73% |
| 8k in / 1024 out, n 512 (decode-bound) | fill to 256 active | 57.8 s | 33.0 s | −43% |
| | whole-pass input tok/s | 19.7K | 22.0K | +12% |
| | whole-pass output tok/s | 2458 | 2750 | +12% |
| | peak 1 s output tok/s | 3555 | 3584 | +1% |
| | ITL p50 | 77.2 ms | 77.2 ms | 0% |
| | mean TPOT | 76.0 ms | 76.3 ms | 0% |
| | TTFT p50 | 5.4 s | 4.6 s | −16% |
| 32k in / 1024 out, n 256 (prefill-bound) | fill to 256 active | 231.7 s | 136.5 s | −41% |
| | whole-pass input tok/s | 30.3K | 44.3K | +46% |
| | whole-pass output tok/s | 947 | 1384 | +46% |
| | ITL p50 | 50.1 ms | 59.2 ms | +18% |
| | TTFT p50 | 118.5 s | 70.3 s | −41% |

At full batch (256 active), both legs decode at the same speed (77 ms/step: 74.8 ms fused forward + 2.5 ms view prologue + 0.2 ms sampler, 3.5K tok/s). In the two prefill-bound passes (8k/256 and 32k/1024), prefill cannot keep all 256 decode slots full (8k/256 decode utilization: 50% exp vs. 31% baseline), so baseline's lower ITL reflects smaller active decode batches rather than faster decode. In the decode-bound pass (8k/1024; crossover at ISL 8k is OSL ≈ 440 on exp vs. ≈ 720 on baseline), exp's +12% whole-pass gain comes from filling the batch 43% faster.

# Part 2: prefill only, PCP16xTP2 vs PP4xPCP8

## Setup

- baseline = PCP16xTP2, no dynamic chunks or lazy push
- exp = PP4xPCP8, dynamic chunks + lazy push enabled
- Δ = (exp − baseline) / baseline. Positive Δ tok/s means exp is faster; negative Δ TTFT means exp is faster.
- Throughput and mean TTFT are steady-state: Only completions K + 1 to N − K count (K = max(10, c)), warm-up and drain rounds are ignored.
- KV pool: Cases 1-4 cap blocks with `--num-gpu-blocks-override` and never trigger eviction. Case 5 sizes pools from free HBM (163 prefixes on PP4xPCP8 vs. 118 on PCP16xTP2) and scales active prefixes past pool capacity.

Code under test: branch `benliangw/pp-pcp-head` in vllm-torchtpu, which is `origin/main` at `b97b97eeb` (2026-10-01) plus:

- `origin/yinlin/pp-pcp-stages-v2`: [PR #1345](https://github.com/vllm-project/vllm-torchtpu/pull/1345) (pipeline hand-off through `jax_op`) and [PR #1346](https://github.com/vllm-project/vllm-torchtpu/pull/1346) (a pipeline stage spans a prefill context-parallel group, plus `VLLM_TPU_DEVICE_ORDER` stage placement)
- `origin/yinlin/pp-lazy-push`: [PR #1481](https://github.com/vllm-project/vllm-torchtpu/pull/1481) (engine pushes the pipeline only when stage 0 would idle, `TPU_PP_LAZY_PUSH=1`)
- Two cherry-picks from `origin/handsomehow/qwen38_pp4pcp8_bench` (no PR): `a0411d989` (GDN Z shard across several tiles and halved FA query tile; required for Qwen3.8 at PCP8 TP1) and `8242a6f9f` (dummy-load MoE routing workaround)

torch_tpu wheel `torch_tpu-0.1.1.dev20260928182719` (torch_tpu#4387, `get_pallas_mesh(process_group=)`), image `torchtpu-vllm-dev:nightly-20261001`, vLLM v0.29.0. Dynamic chunks and lazy push only take effect with pipeline parallelism, so they are no-ops on the baseline.

## Case 1: fresh 8k prompt, prefix caching off

| conc | tok/s baseline | tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 30.8K | 14.7K | −52% | 266 ms | 557 ms | +110% |
| 2 | 36.1K | 29.4K | −19% | 453 ms | 557 ms | +23% |
| 4 | 31.9K | 56.3K | +76% | 1026 ms | 581 ms | −43% |
| 8 | 39.1K | 62.0K | +59% | 1696 ms | 1056 ms | −38% |
| 16 | 46.1K | 73.2K | +59% | 2844 ms | 1806 ms | −36% |
| 32 | 46.1K | 77.9K | +69% | 5688 ms | 3384 ms | −41% |
| 64 | 46.1K | 77.8K | +69% | 11374 ms | 6762 ms | −41% |

## Case 2: fresh 8k prompt, prefix caching on

| conc | tok/s baseline | tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 13.5K | 10.8K | −20% | 606 ms | 756 ms | +25% |
| 2 | 20.1K | 21.6K | +7% | 814 ms | 759 ms | −7% |
| 4 | 23.1K | 36.1K | +56% | 1416 ms | 908 ms | −36% |
| 8 | 27.8K | 59.8K | +115% | 2381 ms | 1096 ms | −54% |
| 16 | 36.3K | 72.2K | +99% | 3587 ms | 1848 ms | −48% |
| 32 | 43.2K | 71.1K | +65% | 6094 ms | 3656 ms | −40% |
| 64 | 43.7K | 78.0K | +78% | 12021 ms | 6746 ms | −44% |

## Case 3: 128k cold tokens, prefix caching off

| conc | tok/s baseline | tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 27.1K | 25.2K | −7% | 4.8 s | 5.2 s | +7% |
| 4 | 29.4K | 41.0K | +39% | 17.8 s | 12.8 s | −28% |
| 8 | 30.2K | 41.7K | +38% | 34.7 s | 25.1 s | −28% |
| 16 | 28.8K | 41.6K | +44% | 72.7 s | 50.4 s | −31% |
| 32 | 30.2K | 41.9K | +39% | 139.2 s | 100.2 s | −28% |
| 64 | 29.1K | 41.5K | +43% | 286.8 s | 201.9 s | −30% |

## Case 4: 128k cache hit + 8k new, prefix caching on

Steady-state window starts after all initial cold warm-up requests complete, counting only pure cache-hit requests (8192 new tokens each).

| conc | new tok/s baseline | new tok/s exp | Δ tok/s | hit TTFT baseline | hit TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 6.0K | 5.3K | −12% | 1.37 s | 1.55 s | +13% |
| 2 | 10.6K | 10.5K | −1% | 1.57 s | 1.55 s | −1% |
| 4 | 12.6K | 17.8K | +42% | 2.61 s | 1.84 s | −29% |
| 8 | 11.4K | 19.0K | +66% | 5.75 s | 3.48 s | −40% |
| 16 | 11.9K | 22.7K | +92% | 11.05 s | 5.81 s | −47% |
| 32 | 14.3K | 24.3K | +70% | 18.43 s | 10.85 s | −41% |
| 64 | 14.6K | 24.2K | +66% | 36.00 s | 21.71 s | −40% |

## Case 5: 128k prefixes warmed, then revisited, KV pool sized by vLLM

Free-HBM pools hold 163 prefixes on PP4xPCP8 (43 of 7010 blocks each: 40 FA + 3 GDN) vs. 118 on PCP16xTP2 (23 of 2731: 20 FA + 3 GDN). Each pass first sends 1 warm-up request per prefix (122880-token prefix + 1024 new tokens, tail < 1 block) to cache the GDN state at the prefix boundary, followed by the timed pass (4 requests/prefix, 122880-token prefix + 8192 new tokens, OSL 1, concurrency 8, random order).

| prefixes | pool share baseline / exp | requests | hit share baseline | hit share exp | input tok/s baseline | input tok/s exp | Δ tok/s | TTFT baseline | TTFT exp | Δ TTFT |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 60 | 51% / 37% | 240 | 100% | 100% | 189.2K | 377.1K | +99% | 5.5 s | 2.8 s | −50% |
| 90 | 76% / 55% | 360 | 80% | 100% | 87.4K | 382.9K | +338% | 11.8 s | 2.7 s | −77% |
| 120 | 101% / 74% | 480 | 43% | 99% | 43.1K | 353.4K | +719% | 24.1 s | 3.0 s | −88% |
| 150 | 126% / 92% | 600 | 23% | 65% | 34.2K | 95.4K | +179% | 30.6 s | 10.9 s | −64% |
| 200 | 168% / 123% | 800 | 13% | 29% | 30.9K | 54.6K | +77% | 34.0 s | 19.1 s | −44% |

At 60 prefixes both pools hit 100% (+99% pure hit-compute gain). At 90-120 prefixes PCP16xTP2 evicts (80% and 43% hit) while PP4xPCP8 holds 99-100%, widening the gain to +338% and +719% (TTFT −77% and −88%). At 150+ prefixes both pools evict.

## KV cache pool size

| | baseline: PCP16xTP2 | exp: PP4xPCP8 (stage 0-2) | exp: PP4xPCP8 last stage with lm_head (stage 3) |
|---|:---:|:---:|:---:|
| HBM cap per rank (0.90 × 94.74 GiB) | 85.26 GiB | 85.26 GiB | 85.26 GiB |
| Weights + compile/activation buffers | 62.3 GiB | 50.6 GiB | 54.5 GiB |
| Available KV pool size per rank | 23.0 GiB | 34.6 GiB (+50%) | 30.8 GiB (+34%) |
| Bytes per block per rank | 9.04 MB | 4.72 MB | 4.72 MB |
| Blocks per rank | 2731 | 7880 | 7010 |
| Slice capacity formula | blocks × block_size (384) × 16 | blocks × block_size (384) × 8 | blocks × block_size (384) × 8 |
| Slice token capacity | 16.8M | 24.2M (+44%) | 21.5M (+28%) |
| Cached 122880-token prefixes (blocks/prefix) | 118 (23) | 183 (43, +55%) | 163 (43, +38%) |
