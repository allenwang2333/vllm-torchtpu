# DP8 prefill adaptation — work in progress

Branch: `develop/dsv41-dp8-prefill`.
Base: `origin/epic/dsv41`, `ddf9b3766365ecaf00d4ac678990c55b4d4ca5cd` (#1794).
Pinned vLLM contract inspected: `e37e51dd246cf421c554e7d6f53e178e3fd29085`.

## Implementation

- Engram tables remain head-sharded over the node-local DP group. Startup rejects
  a smaller/missing Engram group instead of silently replicating host tables.
- All offloaded hash inputs move to CPU before the DP hash collective, including
  idle/decode ranks that could not hash on host.
- CPU lookup gathers local heads for all destination token slots, then Gloo
  all-to-all exchanges packed FP8/E8M0 bytes. Only local tokens with complete
  heads are uploaded. No device DP all-gather or upstream BF16 staging buffer.
- Device staging is cached by local bucket. For token slot T, heads H and width D,
  uploaded bytes are approximately T *H* (D + D/block_size), with int32 padding.
  CPU exchange temporaries are bounded by the configured token slot; table size
  does not enter the H2D allocation. Peak memory still needs device validation.
- CED synchronizes decoder bucket sizes across DP before any rank can skip the
  decoder. A rank with zero kept tokens participates with padding if peers need
  decoder work. Decoder DP metadata uses the decoder bucket, not the encoder's.
- DP support is intentionally limited to TP1 + EP + host offload on one node.
  SP/local-Q flags stay enabled; TP1 has no nontrivial TP sequence partition.
- DSpark uses the existing DFlash DP dummy-forward coordination. This is code
  inspection, not a DSpark multirank execution result. O1 alone cannot validate
  active draft/verify collectives; O>1 with uneven/idle ranks is still required.

## Scripts brought forward

The original corrected scripts and reference summaries are retained in
`../dsv41-chunk-diagnosis-20261010/`. They record the previous TP8/DP1 run and must
not be presented as results on this new branch.

New reusable entrypoints:

```bash
# Activate the pinned Falcon runtime first. Defaults: TP1 DP8, per-rank chunk2k.
bash scripts/dsv41_bench/serve_cached_prefill.sh
python scripts/dsv41_bench/bench_cached_prefill.py \
  --dataset /tmp/tpu_logs/longbench-data.json \
  --out /tmp/tpu_logs/dp8-prefill --dp-size 8 --concurrency 8
```

The benchmark pins each request via the pinned vLLM's `X-data-parallel-rank`
header. Prime and extend retain rank affinity. Every wave uses independent
LongBench prefixes; cache is reset between rounds. One warmup + three measured
rounds, 65,536 hit + 16,384 extend, output 1. Hit count must equal C * 65,536.
The SSE reader uses 4096-byte chunks and prime uses valid UTF-8 sentinel tokens.

Primary timing is the metrics-delta mean server prefill latency across C requests;
TTFT is retained per request. New-token throughput uses C * 16,384 / wave wall
time. Neither metric is pure device kernel time. No profiler is enabled during
these timed waves. Metrics require an otherwise idle service.

Defaults use `[128,256,2048]` buckets and fixed 4 GiB KV per rank; `CHUNK=8192`
selects `[128,256,8192]`. These are candidate settings, not validated capacity
claims. DP8 chunk2k has a 16k global token budget, versus 2k for TP8/DP1 chunk2k;
report that distinction with throughput comparisons. C can be changed; set
`MAX_SEQS` appropriately for C > DP and revisit KV capacity.

## Validation status

- Real CPU Gloo tests: DP2 and DP8 token/head ordering across buckets 1, 7, 2,
  including invalid-scale padding; 3 tests passed on macOS, torch 2.14.1.
- `check_staging_cpu.py`: production staging methods isolated from TPU imports;
  DP1/DP8 dequantization, local upload size, reuse and overflow checks passed.
  This is not a full imported-model integration test.
- Expanded existing Engram/CED/guard test fixtures are pending the pinned Linux
  runtime. Local existing project venv cannot import torch (missing dylib).
- Falcon `exp-rcy41v83oe` / `art-8817xsba4s` submitted, but admission reports
  insufficient unused TPU quota (4 chips needed). No TPU model run or performance
  result has been obtained.

Before calling this a baseline: run imported-model tests; validate numeric output,
Engram group/table slices, CPU/HBM peaks and H2D sizes; exercise uneven CED and
DSpark/idle ranks; then run repeated C8 prime/extend waves and compare TP8/DP1 on
this same source/runtime/bucket policy.

The pending Falcon experiment was withdrawn after repeated quota rejection; no
TPU allocation was acquired. The manifest is retained for a later validation run.

User requested continued resource waiting. Resubmitted as `exp-c5fnabwuc4` /
`art-9yg1gs5t9q` using `experiment-watch.json`. Preserve this pending allocation;
do not cancel merely for temporary quota shortage. Thread heartbeat checks every
5 minutes and continues validation once setup is ready. Upload local changes
before testing: the remote source remains the unmodified epic base.

## Latest scheduling instruction

User requested 2-prod for this task and future 4-chip tasks. Old pending
`exp-c5fnabwuc4` withdrawn by explicit instruction. Current experiment:
`exp-1mfygs8zhi`, artifact `art-6ylt7tm6ib`, cluster
`gke-tpu-train-us-central1-2-prod`, manifest `experiment-2prod.json`.
Heartbeat updated to monitor this experiment; retain its queue/allocation.

11:30 UTC heartbeat: 2-prod experiment RUNNING, setup still installing vLLM
build dependencies (no DIAGNOSTIC_SETUP_READY yet). Overlay uploaded via Falcon
cp to `/tmp/dsv41-dp8-overlay.tar.gz` (17 files), not yet extracted. File hashes
recorded in `overlay-manifest.json`. Extract into `/workspace/vllm-torchtpu`
only after setup completes, then save patch/hash evidence with test artifacts.

11:35–11:39 UTC: setup ready; overlay extracted. First remote test batch
137 passed / 1 stale assertion about removed BF16 staged_rows failed; assertion
updated locally and remotely. Second batch (corrected case + model_full/model_sp)
56 passed. Total 193 distinct passing tests. Logs: artifact dp8-validation/unit.log
and unit2.log. Launched `/tmp/dp8-run-server.sh` (local run-server.sh); server logs
under artifact `dp8-service/server.log`, memory `dp8-service/memory.jsonl`, PID
`/tmp/dp8-server.pid`. Check /tmp/dp8-service.ready before requests. Server runner
waits for /tmp/dp8-service.release; no benchmark auto-start yet.

11:47–11:50 UTC: first service launch failed at startup precompile (not OOM):
Engram token count exceeded the DP slot because ordinary `_dummy_run` supplied
zero-token DP metadata despite a nonempty padded warmup bucket. Updated warmup
metadata to report the bucket on all DP ranks whenever DP lockstep is enabled.
Uploaded runner and regression tests through Falcon cp. Pinned Linux runner
regressions: 67 passed (`dp8-validation/runner.log`); local Ruff check/format pass.
Second service attempt launched with `run-server-r2.sh`, logs/memory under
`dp8-service-r2`. Keep r1 artifacts for diagnosis. Current ready flag is still
`/tmp/dp8-service.ready`; no benchmark result yet. First attempt's partial memory
sample had minimum MemAvailable 241.3 GiB and zero cgroup OOM events; this is not
an end-to-end peak-memory validation.

11:55 UTC heartbeat: r2 still loading weights (last log ~70%, no failure).
Prepared and submitted `run-smoke-r2.sh` / `smoke_dp8.py` via Falcon cp/exec:
waits up to 30 minutes for health, then O16 uneven C8 and C1 on rank3 to exercise
active draft plus idle ranks. Evidence under `dp8-smoke-r2` (run.log/run.exit,
responses, before/after Prometheus). Inspect speculative counters to confirm
DSpark actually ran; successful O16 HTTP responses alone do not prove that.
No timed LongBench benchmark auto-started yet.

12:00 UTC heartbeat: r2 target model loaded; DSpark draft loading reached 93%
(last log 12:00:30). No new error observed. O16 smoke runner remains waiting for
health; run.log empty. Preserve current service/allocation and inspect precompile
then smoke at next follow-up. No DP8 performance measurements yet.

12:05 UTC heartbeat: r2 passed the former Engram DP slot failure point. Encoder
128/256 graph compilation progressed; currently compiling 2048 on all eight
ranks. No new error observed. Smoke runner still waiting for health. Retain
allocation and inspect decoder/draft compilation and O16 results next.

12:10 UTC heartbeat: encoder compilation advanced to decoder; decoder 256
compiled on all ranks, 2048 compilation underway. Shared-memory wait notices
coincide with active compilation; no failure diagnosed from those notices.
Smoke still waiting for health, no performance result. Allocation preserved.

12:15–12:17 UTC: r2 service reached health 200 at 12:13:53. O16 uneven C8
smoke has begun (before.prom exists), but no responses yet; workers show shared
memory wait notices with no new compile messages. Health remains 200. Need
investigate possible collective stall if 240s request timeout fires. Attempted
py-spy stack dumps on ranks0/1/4; container ptrace denied (`Failed to copy
Py_Version symbol`, permission denied), saved stack-*.txt in dp8-service-r2.
Do not signal workers without installing an explicit handler. Can add diagnostic
faulthandler registration for next service restart if required; retain experiment
and compilation cache. No benchmark started or throughput claim made.

User reaffirmed continuous monitoring during diagnosis. O16 uneven C8 smoke now
confirmed failed with HTTP ReadTimeout after 240 seconds (local evidence
/tmp/dp8-smoke-now.log, remote dp8-smoke-r2/run.log). Health success did not prove
inference works. Next action: add explicit faulthandler diagnostics before a
same-allocation service restart (reuse r2 compilation cache), reproduce and
capture all rank stacks without ptrace. Keep the current Falcon holding session;
do not release or resubmit the allocation. Existing 5-minute heartbeat remains
active. No LongBench timing is valid yet.

User clarified: NO scheduled follow-up, continue direct investigation. Automation
`dp8-prefill` PAUSED through app tool. r3 diagnostic restart uses sitecustomize
SIGUSR1 faulthandler registration, retaining the r2 compilation cache. Two stack
samples show rank0 blocked in model.py `_prepare_engram` at hash `.to("cpu")`,
ranks2/4 blocked in host_engram.py CPU `all_gather_into_tensor`. Consistent with
host-hashing peers entering Gloo before submitting prior deferred EP work needed
by the device-hashing rank's D2H. Added all-rank `synchronize_tensors()` before
D2H/CPU gather for TPU DP>1 host-offloaded Engram. r4 launched on same allocation,
artifacts dp8-service-r4 / dp8-smoke-r4. Smoke and stack capture auto-run after
health. Test batch model-r4.log/model-r4.exit under dp8-validation. This fix is
still under TPU validation; do not report resolved until requests complete.

r4 initial launch overlapped the model unit tests: those tests initialized TPU,
causing service startup 'device node already opened' failures. Test batch completed
82 passed; serial restart submitted after tests finished (cmd-j9ihty4j4q). r4
server.log is overwritten by the serial attempt; initial failed log retained
locally /tmp/dp8-r4-now.log. Do not run model test imports concurrently with a
service on this allocation. Selected r3 blocking stacks and r4 test log saved
under notes/dsv41-dp8/results. r2 memory: 344 samples, min host MemAvailable
233.5 GiB, sampled cgroup current max 895.9 GiB, final oom/oom_kill counters zero.

r4 reproduced the hang despite a device sync: rank0 blocks in the sync itself,
rank2 waits in CPU hash gather. Removed that ineffective sync. Found a concrete
DSpark idle-rank pairing bug: DFlashProposer uses `is_mixture_of_experts` for TP1
drafts, but DSparkDeepseekV41ForCausalLM lacks that full protocol despite its EP
MoE layers. Added explicit `requires_dp_collective_pairing=True` draft contract
and consumed it in DFlashProposer; regression uses the actual V4.1 DSpark class
without allocating weights and checks two idle draft phases. Draft test suite
29 passed in 14.70s using full tests/conftest (`draft-r5b.log`); initial command
with confcutdir omitted the device fixture and was corrected. r5 service starts
only after tests have exited. Current dirs dp8-service-r5, dp8-smoke-r5. All
follow-up is direct in the active turn; automation remains PAUSED by user request.

## Validated result (2026-10-10, r5)

TP1/DP8/EP8, all target features ON, LongBench independent prefixes, 65,536 hit +
16,384 extend, O1, C8, chunk2,048 per rank (global budget16,384), KV4GiB/rank.
Three measured waves after warmup: mean server extend1.9560/1.9610/1.9553s;
median1.9560s. Median wave throughput50,239.6 new tokens/s. Median wave mean
HTTP TTFT2.1841s. Each wave: cache hits524,288; new computed131,072; preemptions0.
Server prefill includes scheduling/replay and is not pure device kernel latency.
No controlled same-revision TP8 comparison was run; older TP8 data is reference.

O16 uneven C8 completed in0.760s,195 draft tokens; C1 idle-rank scenario0.270s,
20 draft tokens. Same-prompt C8 all ranks and C1 ranks0/3/7 generated identical
text. This checks rank/concurrency consistency, not full reference-model quality.
CPU Gloo tests verify row/head/scales ordering; host staging regression checks
local H2D buffer size and dequantization. Final draft tests29 passed; runner67;
prior imported-model suites193 distinct passed, additional model batch82 passed
(overlapping tests, do not sum as distinct). Ruff check/format on20 changed/new
Python files pass; git diff --check passes.

Memory: service r5 host minimum available277.2GiB through baseline, no OOM or
OOM kill. Separate LongBench memory validation completed;160 HBM readings
(20 samples x8 devices), observed peak71.54/94.74GiB per device. Sampled maximum
is not an exact allocator high-water mark. Final manifests, results and tracked
patch in this notes directory; full artifacts kept in Falcon art-6ylt7tm6ib.
Main fixes: host head-shard exchange/local H2D staging; CED DP replay pairing;
warmup DP slot metadata; DSpark TP1 explicit idle-rank collective contract.
Automation PAUSED at user's request. Work remains uncommitted and unpushed.

Final archive `results/final-falcon-results.tar.gz` downloaded and verified:
baseline, smoke, consistency, memory and service exit codes all0. Cross-rank
same-prompt token logprobs also identical (max absolute delta0.0). Service stopped;
completion marker /tmp/diagnostic.done=0 sent to end the bounded Falcon session.
Falcon final wait confirmed SUCCEEDED (job reason succeeded; expected pods
completed). Resource session closed successfully. No further automation scheduled.

## 2026-10-10 DP8/C8 profile

Completed `exp-2257y2oh7d` on 2-prod; artifact `art-nj3o7jv3a7`, TPU released. Full report and reproducible manifests/scripts: [profile/README.md](profile/README.md). Same all-on TP1/DP8/EP8, chunk 2k/rank, global budget16k, LongBench65,536hit+16,384extend,O1,C8. Control server1.87181s,TTFT2.05353s,52,537.69newtok/s;profile server1.98291s (+5.9%). All8core XLA-op unions1.260–1.298s, device spans1.853–1.898s, gaps0.592–0.599s with repeated60–63ms boundaries. EP MoE ~29% device HLOselftime;matmul~21%. CPU alltoall92–169ms/rank,copy155–255ms/rank (inclusive,nonadditive). NineModelForward calls and eightDSpark modules observed despiteO1. Priority: host-stage/gap attribution, then same-workload4k/8kchunk comparison; no model optimization was applied in the profiling turn.

## 2026-10-10 host gap attribution

Reused existing trace, no TPU resubmission or model changes. See [gap-analysis/README.md](gap-analysis/README.md). 258 XPlane/Torch module anchors agree within ~1ns offset spread. Rank0/core0 nine long gaps total565.37ms: cat72.77,clone69.03,copy123.03,index_select54.48,Gloo alltoall120.67,allgather32.24,allreduce26.72,unattributed66.43ms after interval union. Call-order inference separates~201ms host packing/repacking/staging,~63ms H2D host calls. Main target is repeated serial host data preparation; no equal-to-duration speedup claimed.

## 2026-10-11：chunk overlap 调查进行中

最新状态与可复现材料见 `overlap/README.md`。最小 CPU DP metadata snapshot 修复已在 exp-jew0a7weqi 完成测试/profile（SUCCEEDED，资源释放）：非 profile extend 1.872→1.467 s，新增 token 吞吐 52.5k→63.1k/s，device gaps 0.60→0.07–0.08 s；34 单测、O16 C8/idle C1 smoke 通过。

但历史与新结果的输出一致性未通过，历史自身也有重复输入波动，**不可宣称数值验证通过**。当前正在 exp-usduz1nwhn（artifact art-e6moty8stp，2-prod）用同一服务做 serial/overlap/fresh-staging/serial 对照，含不带 logprobs 和 top5 logprobs 请求。持续直接分析，无定时任务。具体文件、已执行的 client 覆盖、诊断模式初始化见 overlap/README.md。尚未提交或推送。

2026-10-11 收尾：同服务数值隔离 exp-usduz1nwhn 已 SUCCEEDED 并释放资源。serial-1/overlap/fresh/serial-2 的不带 logprobs extend 分别约 1.895/1.465/1.471/1.881 s。原串行与独立 staging 也有显著输出/logprob 波动，数值基线仍未通过，不能简单归因于 overlap buffer 复用。性能机制与收益已复现；完整报告及 192 个响应见 overlap/README.md、overlap/numeric/。当前无运行实验、无新定时任务，未提交或推送。

## 2026-10-11：chunk sweep 与 idle CED 优化

本轮比较2k/4k/8k并实现空闲DP rank的CED host metadata修复。报告见 [chunk-sweep/README.md](chunk-sweep/README.md)。修复后extend中位数分别1.312/1.429/1.786s，新增token吞吐67.3k/61.7k/51.5k/s；保留2k。2k相对上一overlap版本1.465s再降低10.4%，profile证明152ms冗余full decoder图消失，111 pinned Linux回归通过。三档cache命中及零preemption校验通过。长上下文数值重复性问题仍未解决，不宣称数值验收通过。exp-i42cf8kt8e和exp-hfyt24ii6g均SUCCEEDED并结束资源使用。代码未提交或推送；自动化保持暂停。
