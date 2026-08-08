# E0200 SparseCore SC2 halt on the torch 2.13 / torch-tpu dev20260804 stack — evidence

## TL;DR

After #288 (torch 2.11 → 2.13, torch-tpu `0.1.1.dev20260728090214` → `0.1.1.dev20260804130134`,
vllm 0.26.0 → v0.26.1rc0; libtpu pin `0.0.44.1` unchanged), TPU v7x serving workloads
crash with `E0200 RuntimeUnexpectedCoreHalt` — a `UserFatal` raised by **SparseCore SC2**
(all 16 tile sequencers halted at PC `0x3fff`, SC sequencer PCs recur across incidents:
`0x3`, `0xa3`, `0xb3`, `0x133`). The halt reproduces on **two unrelated model families**
(gemma-4-26B and Qwen3.5-397B), in **two serving modes** (regular serving on bare-metal CI
and PD-disagg decode on GKE), on **6+ distinct physical hosts** (tpu444, tpu1330, tpu2144,
tpu757, tpu893, + build-110 hosts), and **never occurs on the pre-#288 stack** under
identical workloads. A dedicated bisect (7 TPU runs) ruled out every model-side and
SC-offload-configuration factor we can control; the regression is in the runtime/compiler.

## Crash signature (identical in every incident)

```
E0200: RuntimeUnexpectedCoreHalt: Program or fatal error occurred; computation may be invalid
Detailed error: Node N halted unexpectedly at tag:pc SparseCoreSequencer:0:0x3 ...
  <chip>_SC2 SparseCoreTileExecuteCoreSequencer:{0..15} halted at tag:pc ...:1:0x3fff
core location: <chip>_SC2, error type: UserFatal
```

Always **SC2** — never SC0/1/3 — on whichever chip/host, suggesting one specific
runtime-owned SparseCore program pinned to that core. The "HLO module" named in the
report varies (`tt_jit_uniform_`, `tt_jit_layernorm_*_rms_norm_mul`,
`tt_jit_logits_processor_*`, TensorCore RPA kernels) and includes modules with
"no HLO mapping" — these are the victims of the async D2H failure at detection time,
not the faulting program.

## Incident & control matrix

| Build | Date | Workload / model | Stack | Result | Evidence |
|---|---|---|---|---|---|
| [ci 1888](https://buildkite.com/tpu-commons/vllm-torchtpu-ci/builds/1888) | Aug 6 | gemma-4-26b-fp8-dp4-tp2-ep-text, full c512/1024/2048 sweep, sampling on (temp 1.0 / top_k 64 / top_p 0.95) | pre-#288 | **PASS**, 0×E0200 | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-ci/builds/1888/jobs/019fd572-4cb2-4210-ae18-ed08acb78805/artifacts/019fd5b4-85b4-4212-b086-d557b9e58251) |
| [ci 2006](https://buildkite.com/tpu-commons/vllm-torchtpu-ci/builds/2006) | Aug 7 | same config | post-#288 | **HALT** (SC2, PC 0xa3/0x123, host tpu444) | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-ci/builds/2006/jobs/019fda95-1ad5-4e22-9846-da348e836f28/artifacts/019fdade-9f1b-4837-b5f3-8f5447e19011) (~line 3950) |
| [ci 2024](https://buildkite.com/tpu-commons/vllm-torchtpu-ci/builds/2024) | Aug 7 | same config + `USE_MOE_SPARSE_CORE=0` | post-#288 | **HALT** (SC2, PC 0x3, host tpu1330, 2 chips) | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-ci/builds/2024/jobs/019fdafd-1676-448c-8040-b7711d44822b/artifacts/019fdb43-6b2e-4c94-951d-ae58a43c7a8a) (~line 3950) |
| [nightly 1904](https://buildkite.com/tpu-commons/vllm-torchtpu-ci/builds/1904) | Aug 6 | qwen3.5-397b tp1-dp8 / tp2-dp4 / tp8-dp1, c64/256/512, greedy | post-#288 | serving **PASS** (tp8-dp1 failed only the perf-regression gate) | [tp8-dp1 job](https://buildkite.com/tpu-commons/vllm-torchtpu-ci/builds/1904#019fd64d-2c93-475e-886d-b43fefce9d37) |
| [dev 110 A](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/110) | Aug 7 | gemma, `BENCHMARK_TEMPERATURE=0` (greedy) | post-#288 | **HALT** at c1024 (SC2) | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-dev/builds/110/jobs/019fdced-0a8b-436e-9cc0-9306602343c4/artifacts/019fdd04-a833-42f2-8b34-1ac8d22eebda) |
| [dev 110 B](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/110) | Aug 7 | gemma, `KV_CACHE_DTYPE=auto` | post-#288 | **HALT** (SC2) | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-dev/builds/110/jobs/019fdced-0a8f-4a00-98a4-9a540bf15213/artifacts/019fdd03-b50f-4a14-9209-c79336ab059d) |
| [dev 110 C](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/110) | Aug 7 | gemma, `--max-num-seqs=128`, c512 only | post-#288 | **PASS**, 0×E0200 | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-dev/builds/110/jobs/019fdced-0a8f-4d2b-9f14-473a756bcd2b/artifacts/019fdd02-6dc2-4b75-a154-e010d2c8d9f2) |
| [dev 112 D](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/112) | Aug 7 | gemma, AR/AG SC-offload min-size = 16 GiB | post-#288 | **HALT** at c1024 (SC2, host tpu2144) | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-dev/builds/112/jobs/019fdd0a-6c54-4f22-be7f-601f2576a09c/artifacts/019fdd24-bec3-4c92-a777-92581c7c617c) |
| [dev 112 E](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/112) | Aug 7 | gemma, all 8 `xla_tpu_enable_sparse_core_collective_offload_*=false` | post-#288 | **HALT** at c1024 (SC2, PC 0xb3/0x133, host tpu757) | [server.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-dev/builds/112/jobs/019fdd0a-6c57-49ed-9cae-d3ddc0569bd0/artifacts/019fdd27-13e0-4514-8ab4-b9580f909ddc) |
| [pd-disagg 58](https://buildkite.com/tpu-commons/vllm-torchtpu-pd-disagg-pipeline/builds/58) | Aug 6 | **Qwen/Qwen3.5-397B-A17B-FP8**, GKE 1P1D, decode pod TP8-EP, mns 512 | prod image `sha256:a335f817a3cc` = nightly image 23 (pre-#288) | **PASS**, 0×E0200 | [decode.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-pd-disagg-pipeline/builds/58/jobs/019fd7d8-1bdc-4aed-99ad-51ca3469486e/artifacts/019fd927-06da-4ae7-a2c5-e487aa1a72df) |
| [pd-disagg 59](https://buildkite.com/tpu-commons/vllm-torchtpu-pd-disagg-pipeline/builds/59) | Aug 7 | same workload | prod image `sha256:49562f005722` = nightly image 24 (**contains #288**) | **HALT**: 25×E0200 (SC2, PC 0x3, host tpu893, 2 chips) at 07:28, ~56 min in; pod died, build "passed" via restart and uploaded results to BigQuery | [decode.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-pd-disagg-pipeline/builds/59/jobs/019fdaea-4533-4542-97d0-153708693e1f/artifacts/019fdb35-c4fc-44d2-8b0d-c5317bed9ebd), [benchmark.log](https://buildkite.com/organizations/tpu-commons/pipelines/vllm-torchtpu-pd-disagg-pipeline/builds/59/jobs/019fdaea-4533-4542-97d0-153708693e1f/artifacts/019fdb35-c4fe-4a8b-97ba-f33758804583) |

Image bracketing: [nightly-image 23](https://buildkite.com/tpu-commons/vllm-torchtpu-nightly-image/builds/23)
(Aug 6, commit `fa6c602a`, pre-#288) vs [nightly-image 24](https://buildkite.com/tpu-commons/vllm-torchtpu-nightly-image/builds/24)
(Aug 7, commit `59d64431`, includes `a40ace87` = #288). The only relevant delta between
pd-disagg builds 58 and 59 is that image roll.

## Model diversity — this is not a gemma issue

The PD-disagg victim is **Qwen/Qwen3.5-397B-A17B-FP8** (v1 HMA connector, decode pod
TP8-EP, `max_num_seqs=512`), a completely different architecture, vocab size, and
parallelism layout from gemma-4-26B (dp4-tp2-ep, `Gemma4ForCausalLM`,
`--language-model-only`). Two independent model families × two serving modes on the same
new stack, zero occurrences on the old stack.

## Ruled out by bisect (dev builds 110 / 112)

- Sampling / RNG path (`tt_jit_uniform_`): greedy variant A halts identically.
- fp8 KV cache: `kv-cache-dtype=auto` variant B halts identically.
- Repo's SparseCore MoE token-movement kernels: `USE_MOE_SPARSE_CORE=0` (ci 2024) halts.
- SC-offloaded all-reduce/all-gather: min-size threshold at 16 GiB (variant D) halts.
- SC collective offload of any type (AR, AG incl. 2d/3d, RS incl. nd, dense/ragged
  all-to-all) force-disabled via `LIBTPU_INIT_ARGS` (variant E): halts.
- One bad machine: 6+ distinct hosts across two fleets (bare-metal CI agents + GKE).
- libtpu version: pin `0.0.44.1` unchanged across the regression boundary.

## Trigger correlation (imperfect — for the runtime team)

- Bare-metal gemma runs: deterministic once per-rank decode-batch bucket crosses 128 → 256
  (c1024 with `max-num-seqs=512`); capping at 128 seqs/rank (variant C) survives c512.
- GKE disagg: the halt fired during the **c1 / isl8192 / osl1 TTFT phase** — decode pod
  nearly idle (≤2 concurrent requests) but receiving 8k-token KV transfers via the HMA
  connector. So large decode batches are sufficient but not necessary; heavy KV/DMA
  traffic appears to be another path to the same halt.
- Timing: minutes-to-an-hour into sustained serving; never at startup/compile.

## Data-integrity note

pd-disagg build 59 uploaded benchmark results to BigQuery after the halt/restart
(the c8/c16/c32 result JSONs are visibly truncated vs build 58's). Aug-7 `GKE_DISAGG`
rows should be treated as suspect.

## RESOLVED: dependency bisect (vllm-torchtpu-dev builds 110-146)

A per-dependency isolation matrix (one pin change per dev build, all against a
c1024-only gemma reproducer, verdicts from server.log E0200 counts — job state
is unreliable because a halted DP rank leaves the other ranks serving) found:

- **torch / torch-tpu exonerated**: the halt reproduces on torch 2.11 +
  torch-tpu dev20260728 (the pre-#288 runtime that was green for weeks) and on
  every torch-2.13 nightly through dev20260807160217 (builds 124, 126-129).
- **Plugin commits exonerated**: current vllm-torchtpu main + full old dep
  stack + vllm 0.26.0 is clean (build 134); pre-#288 tree + only vllm→rc0
  halts (build 132). Perfect 2x2 with vllm as the only discriminator.
- **vllm still broken at v0.27.0rc1** (build 140) — no forward fix upstream.
- **Commit bisect over the 300 commits v0.26.0..v0.26.1rc0** (builds 136-145,
  quartiles → 60 → 30 → 7 → 1):
  clean at 60/120/180/195, halt at 198/202/210/216/240/300.

**First bad commit: vllm `a82f1b38` — "[Perf][V1] Skip LRU hash-split in
free_blocks when prefix caching is off" (#48017).** With prefix caching
disabled (all our serving configs), freed KV blocks switch from prepend
(LIFO reuse of a small hot set of block ids) to append (FIFO cycling through
the entire block-id space). Under sustained load the block tables then
reference the full physical block-id range, and an indexed
SparseCore-offloaded op on the paged-KV path traps (E0200, UserFatal, always
SC2, tile sequencers at PC 0x3fff). This explains the minutes-in timing (time
to first cycle through the pool), the load threshold, the apparent
model-specificity (block count x churn rate), and the GKE disagg decode halt
(fast block churn from KV transfers). The underlying index-handling bug is in
the TPU stack and is merely exposed by the vllm change — gather offload to
SparseCore is active in these graphs (build 133's compile error proves it),
though disabling `xla_tpu_offload_gather_to_sparsecore` alone did not avoid
the halt (build 135).

## Actions

1. **Workaround (validated in build 146 if green): plugin-side patch**
   restoring pre-#48017 LIFO free order when prefix caching is off — branch
   `mhhua/sc-halt-workaround-lifo-free`, a `BlockPool.free_blocks` monkeypatch
   in `vllm_torchtpu.platforms.register_tpu_platform`. Unblocks the gemma
   General-suite re-add (pipeline_perf.yml TODO) and protects GKE disagg,
   with no dependency rollback.
2. **torch-tpu bug**: SparseCore-side trap on paged-KV block indices under
   FIFO block reuse; repro = any TPU v7x serving load with
   `--no-enable-prefix-caching` on vllm >= a82f1b38, or directly via the
   gemma c1024 reproducer on branch `mhhua/gemma-sc-halt-bisect`.
3. **vllm upstream**: report that #48017's cache-locality heuristic is
   actively harmful on TPU (and gate it per-platform), independent of the
   TPU trap being fixed.
4. Aug-7 `GKE_DISAGG` BigQuery rows remain suspect (see data-integrity note).
