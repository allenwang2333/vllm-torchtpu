# GLM-5.2-FP8 on multihost TPU v7x-16: bring-up status

Status as of 2026-08-05. Audience: model bring-up / torch_tpu / kernel folks.
Contact: Ming Huang <mhhua@google.com>.

## TL;DR

GLM-5.2-FP8 (zai-org/GLM-5.2-FP8, 761 GB, MLA + MoE + DSA sparse-attention
indexer) now serves **end-to-end** on a 2-host tpu7x-16 slice (TP=16, Ray
executor, weights streamed from GCS). First benchmark numbers are in — and
they show serving perf is **~50x off**, with expert parallelism disabled
being the dominant suspect. EP cannot be enabled because torch_tpu crashes
on multihost process sub-group creation (details below). That crash is now
the single highest-leverage fix for GLM on this platform.

Stack: branch `glm_bringup` (sparse MLA/indexer kernels) + one infra commit
(PR #246: benchmark config, multihost harness, pipeline). No other source
changes are needed — earlier per-bug workarounds were all superseded by
`glm_bringup`.

## First numbers (dev build 67, 2026-08-05)

`vllm bench serve`, 8k input / 1k output, 640 prompts, serve flags per
`scripts/vllm/benchmarking/glm_cmds.yaml` (glm branch) except:
`--tensor-parallel-size=16`, no `--enable-expert-parallel` (crash, below),
`--max-num-batched-tokens=4096` (compile OOM, below),
`--gpu-memory-utilization=0.92`, `--max-model-len=10240`.

| Concurrency | Output tok/s (aggregate) | Mean TPOT | Median TTFT |
|---|---|---|---|
| 64  | 56.2 | 1065 ms | 13.0 s |
| 256 | 57.7 | 1562 ms | 2367 s |

Interpretation:

- **TPOT > 1 s/token** means each engine step takes over a second; healthy
  would be tens of milliseconds. Throughput is flat between concurrency 64
  and 256 — the engine is completely compute-bound, extra concurrency only
  builds queue.
- **TTFT at 256 (~39 min median) is queueing, not a hang**: 256 prompts x
  2 prefill chunks of 4096 tokens, at seconds per prefill-heavy step,
  served by the same slow pipeline that is also stepping decode. Median
  TTFT ~= queue drain time. At concurrency 64 the median (13 s) is roughly
  the pure prefill service time for one 8k prompt.
- Suspected split of the gap: EP disabled inflates MoE compute (every rank
  computes over the full expert set instead of its top-k share; ~10-20x on
  the MoE fraction), the rest is first-cut sparse-indexer / MLA kernel and
  dense-gmm cost. Needs a per-step profile to apportion (next step).

## Blocker 1: torch_tpu multihost EP sub-group crash (the big one)

- With `--enable-expert-parallel`, vLLM's `initialize_model_parallel`
  creates torch.distributed sub-groups beyond the world group. On a 2-host
  tpu7x-16 the first cross-host sub-group creation aborts the TPU runtime
  before any model code runs. Exact error
  ([build 54](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/54),
  `server.log` artifact; the failing group in that run was the
  decode-context-parallel group, the first sub-group vLLM creates):

  ```text
  File ".../vllm/distributed/parallel_state.py", line 1831, in initialize_model_parallel
    _DCP = init_model_parallel_group(
  File ".../vllm/distributed/parallel_state.py", line 436, in __init__
    device_group = torch.distributed.new_group(
  File ".../torch/distributed/distributed_c10d.py", line 2164, in _new_process_group_helper
    backend_class = creator_fn(
  RuntimeError: transfer to 'cpu' device failed with: failed to copy buffer
    from device into host buffer -- the XLA compiler failed with: the
    program continuator has halted unexpectedly
  ```

  Note the crash surfaces during a device-to-host buffer copy inside
  torch_tpu's process-group backend constructor.
- Isolation (dev builds 54-56):
  - single-host EP: works (qwen3.5 EP configs run nightly on v7x-8);
  - multihost TP-only world group: works (Qwen3-0.6B TP=16 smoke passes
    repeatably; GLM serves);
  - multihost + any EP sub-group: crashes at group creation.
  So the bug is specifically construction of **non-world communicators
  that span hosts**.
- Caveat: the crash was observed on the pre-`glm_bringup` stack (libtpu
  0.0.45); it has not yet been re-run on the current base. torch-tpu is
  the same version on both, so it is expected to reproduce — probe in
  flight.
- Repro: `vllm-torchtpu-dev` pipeline on branch `glm-perf-nightly` with
  `ENABLE_EP=true` in `scripts/vllm/benchmarking/configs/glm-5.2-fp8-tp16-ep.sh`
  (currently `false`, marked PROBE). The pipeline's Qwen smoke gives a
  ~7-minute multihost sanity gate before the GLM step.
- torch-tpu pin at time of writing: `0.1.1.dev20260728090214`.

## Blocker 2: XLA compile-temporaries OOM at 8192 batched tokens

- With `--max-num-batched-tokens=8192`, compiling the 8192-token graph
  fails: *"HLO temporaries (98.84G) exceeds available HBM (94.74G)"*
  (glm_bringup kernels; an earlier indexer implementation needed 95.10G —
  same wall). Note the temporaries requirement **alone exceeds total chip
  HBM**, so no `gpu-memory-utilization` value can fix it; the knob is the
  largest compiled graph size.
- Current workaround: `--max-num-batched-tokens=4096` (chunked prefill
  covers 8k prompts in 2 chunks). Question for kernel owners: is the
  8192-token working set reducible (rematerialization / smaller fusion
  scope in the sparse-MLA path)?

## Environment prerequisites (already handled, listed for completeness)

- **libtpu >= 0.0.44.1** (0.0.45 also verified). 0.0.43-0.0.44.0 have a
  regression: PjRt init validates per-core process layouts against VM
  metadata HOST_BOUNDS and rejects multihost tpu7x slices ("Topology
  HOST_BOUNDS does not match VM metadata"). `glm_bringup` still pins
  0.0.43 — any multihost run from that branch alone will fail PjRt init.
  Verified runbook + analysis: `docs/multihost_v7x_manual_repro.md` on
  branch `glm-multihost-manual-repro`.
- **Weights**: no disk on the slice fits 761 GB; weights stream from
  `gs://tpu-commons-ci/glm/GLM-5.2-FP8` via
  `--load-format runai_streamer` (~10 min for all 141 safetensors).
- **Cluster**: `.buildkite/scripts/run_multihost.sh` brings up Ray
  head+worker containers across both hosts (metadata IP discovery,
  supervised retries, scorched-earth pre-clean), runs the benchmark in the
  head container, archives Ray session logs as build artifacts.

## Known flakes on the slice (retry-class, not code)

- `START_SESSION failed` (GRPC, from `MaybeInitializeSliceBuilder`) when a
  step starts before the previous tenant's TPU session fully releases —
  job retry clears it.
- Single-rank `PjRtClient is not initialized` at `init_process_group` when
  a stale process still holds a chip (e.g. leftovers of a failed prior
  job) — retry clears it; a chip-holder sweep in the harness pre-clean
  would remove the class.
- The second v7x-16 slice's worker host currently drops all ssh; the dev
  pipeline is pinned to agent `tpu7x-16-ci-1` until that's repaired.

## How to iterate

1. Branch `glm-perf-nightly` (PR #246, based on `glm_bringup`). Edit, push,
   then trigger the `vllm-torchtpu-dev` Buildkite pipeline on the branch —
   it builds the image, runs the multihost Qwen smoke, then the GLM step.
   Do not use floating image tags; the pipeline pins the image it builds.
2. Benchmark config: `scripts/vllm/benchmarking/configs/glm-5.2-fp8-tp16-ep.sh`.
   For perf iteration, drop `NUM_PROMPTS` (640 -> 64) so cells take minutes
   at current speeds; restore before recording baselines.
3. Reference builds (pipeline `tpu-commons/vllm-torchtpu-dev`; each GLM
   perf job uploads `perf_eval_results/.../server.log`, `benchmark.log`,
   and Ray session logs `ray_logs_{head,worker1}.tgz` as artifacts):
   - [54](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/54) —
     EP sub-group crash (server.log has the traceback above);
   - [56](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/56) —
     TP-only isolation (passes `initialize_model_parallel`, EP is the
     delta);
   - [64](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/64) —
     sparse indexer first compiles; 8192-token OOM at util 0.95;
   - [66](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/66) —
     glm_bringup stack validated; 8192-token OOM independent of util;
   - [67](https://buildkite.com/tpu-commons/vllm-torchtpu-dev/builds/67) —
     first end-to-end serve + the numbers table above (jobs > "Perf:
     glm-5.2-fp8-tp16-ep" > Artifacts).

## Next steps

1. torch_tpu: fix multihost EP sub-group creation (owner: torch_tpu team;
   this doc + builds 54-56 logs are the hand-off).
2. Re-probe `ENABLE_EP=true` on the current stack (cheap; confirms the
   crash still reproduces and refreshes logs for the issue).
3. Per-step profile of decode + prefill on glm_bringup kernels to split
   the remaining gap (indexer vs MLA vs MoE gmm).
4. Once perf is sane: restore `NUM_PROMPTS=640`, re-check 8192
   batched-tokens, seed `baselines/perf/glm-5.2-fp8-tp16-ep.baseline.json`
   from the first healthy run, and re-verify qwen suites on libtpu 0.0.45
   before merging PR #246.
