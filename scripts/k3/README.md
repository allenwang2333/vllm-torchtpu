# Kimi K3 TP=32: bring-up, accuracy, benchmark, profile

A runbook for the measure-a-change loop on K3 at TP=32 across four hosts.
With dummy weights (`--load-format dummy`, see below) a full cycle is
roughly **45-60 min of bring-up** (compile-only, cold cache) plus **~25 min
of measurement**; with real weights from `/mnt/models` add **~110 min of
weight loading**. The order below is arranged to fail fast and to avoid
re-running the expensive part.

## 1. Stand up a server

```bash
TPU_MOE_HIERARCHICAL_EP=1 \
TPU_LATENT_PROJ_INTRA_CHIP_TP=1 \
PROFILE_DIR=$PWD/results/<name>/traces \
KIMI_K3_LOG_ROOT=$PWD/results/<name>-log \
  sbatch --job-name=<name> scripts/k3/slurm_launch_k3_tp32_tp.sbatch
```

Starts Ray on the four allocated hosts, waits for all 32 TPU resources, runs
`launch_kimi_k3_tp32_ep.sh` on the head, then holds the allocation until the
time limit or `scancel`. It deliberately runs no eval: bringing K3 up is
expensive, so any mistake in a measurement should cost the measurement and not
the load. Extra args after the script name are forwarded to `vllm serve`.

The endpoint is written to `${KIMI_K3_LOG_ROOT}/endpoint.txt` as soon as the
server answers, before any diagnostics run.

### Weights: dummy by default for perf iteration

The only model store is `/mnt/models` (shared, HF hub layout, mounted on
every host). A full K3 weight read from it measured **6559 s** (job 29) --
vs ~10 min from the node-local NVMe we used to have. For performance work,
skip it:

```bash
  sbatch ... scripts/k3/slurm_launch_k3_tp32_tp.sbatch --load-format dummy
```

TTFT/TPOT are unaffected (the ops are dense and value-independent), but the
canary prints garbage by construction and GSM8K is meaningless, so skip
both and gate numerics changes on unit tests. Use real weights only when
accuracy or real output text matters -- and then raise `READY_TIMEOUT`
(default 7200 s) past load + compile: job 29 died at 2h02m because the
timeout fired seconds after the last rank finished its 110-min load.
`READY_TIMEOUT=14400` covers it.

### Environment pitfalls seen on the current cluster (ici-2x2x4-a)

- **torch_tpu >= 0.1.1.dev20260813 renamed the tier-2/3 cache env vars**
  (drops the `INTERNAL_` infix). The sbatch exports both spellings. Symptom
  of the old names being inert: workers fall back to tier-2 namespace
  `default` and SIGABRT with `Failed to create tier-2 cache entry lock file:
  /dev/shm/torch_tpu_cache/default/... Permission denied` when another
  user's job owns that directory (jobs 620/622).
- **Do not derive HF_HOME/HF_MODULES_CACHE from the model store's parent.**
  `/mnt` is root-owned on the compute nodes, so `HF_HOME=/mnt` made
  `HF_MODULES_CACHE=/mnt/modules` and the bring-up `mkdir` died with
  Permission denied (job 28). The sbatch pins both under the repo's
  `.runtime-cache/`.
- **`/mnt/models` must be populated before first use**; the K3 snapshot
  there is verified complete (113/113 blobs, 96 shards + metadata, revision
  `9f62e4e`). The sbatch probes the fragile metadata files on every host
  before committing to a load.


### Read the banner before you trust a number

```
SERVER_READY after 1983s
torch.compile: enabled
profiling: /.../results/<name>/traces
async scheduling: async_scheduling': True
max_num_seqs: max_num_seqs': 8
compile_sizes: requested=... resolved=compile_sizes': [...]
max_num_batched_tokens: max_num_batched_tokens': 512
KDA path: VLLM_TORCHTPU_KDA_FUSED=1 (1=fused, 0=unfused)
MLA tuned-param misses (mixed): ...
Latent projection fallbacks: 0
```

These are read back from the server's **own resolved config**, not echoed from
the submitting shell, and every one of them silently changes decode speed:

| flag | wrong value costs |
|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | absent -> auto-enabled for K3 -> torch.compile off -> **~18x slower** (344 ms vs 19 ms TPOT) |
| `ASYNC_SCHEDULING` | off -> host work stops overlapping the device -> **+8.4 ms/step** |
| `MAX_NUM_SEQS` | differs between arms -> concurrency sweeps above that value are not comparable |
| `COMPILE_SIZES` | decode batches pad UP to the next bucket -> with `2,16,512` a c=2..16 sweep all runs the batch-16 graph; use `1,2,4,8,16,...` for a real scaling curve |
| `MAX_NUM_BATCHED_TOKENS` | below 8192 -> prefill chunks and never hits the PR #598 tuned MLA key (it keys on `max_num_tokens=8192`); pair with a matching compile bucket |
| `PROFILE_DIR` | unset -> `/start_profile` returns **404** and the capture silently records nothing |

All are pinned with defaults in the sbatch. They are pinned because they were
once inherited from an interactive shell, and a run that differed in them
produced a confident, entirely wrong "31% regression" against a baseline that
did not. **A benchmark must not depend on the shell that submitted it.**

The MLA line lists `get_tuned_params` misses by bucket (a hit is silent). For
K3, the tuned mixed-prefill key only matches with fp8 KV cache
(`--kv-cache-dtype fp8`) **and** `MAX_MODEL_LEN` ~= 8320 (`pages_per_seq=520`);
with the default bf16 KV / 32k context every bucket misses and the 8k prefill
runs ~4x slower than tuned (5786 ms vs 1345 ms TTFT measured).

Profiling needs three flags together, all at startup, none settable afterwards:
`USE_PHASED_PROFILER=true`, `--profiler-config.profiler=torch`, and
`--profiler-config.torch_profiler_dir=<abs path>`. Setting `PROFILE_DIR` wires
up all three.

## 2. Measure

```bash
ENDPOINT=$(cat results/<name>-log/endpoint.txt) \
ARM=<name> OUT=results/<name> GSM8K_LIMIT=100 \
  scripts/k3/arm_check.sh
```

For prefill-side work (TTFT across exact input lengths instead of a decode
sweep), run `arm_check_prefill.sh` the same way; it also verifies from the
server log which MLA tuning keys missed. With `--load-format dummy` servers,
skip `arm_check.sh` entirely -- its canary and GSM8K are meaningless on
random weights; `arm_check_prefill.sh` still applies (its canary is expected
to be garbage; read the TTFTs only).

Runs, in this order and for a reason:

1. **Canary** -- one greedy prompt, printed. A benchmark with `--ignore-eos`
   never inspects a token, so a model emitting garbage still reports a healthy
   tok/s. Check this first, always.
2. **GSM8K** (`eval_gsm8k.sh`, `LIMIT=100`) -- the accuracy gate.
3. **Concurrency sweep** (`concurrency_sweep.sh`, c=1..16) -- decode speed.
4. **Profile** at c=1 -- last, because it leaves the profiler session closed and
   writes gigabytes.

Both arms of a comparison must run this same script. A hand-assembled sequence
that drifts between arms is not a comparison.

## 3. Compare

```bash
python scripts/k3/diff_arms.py results/<control> results/<treatment>
```

Reports total device self time, the ops of interest matched by weight shape,
every collective with **per-occurrence** cost, and category totals.

### A/B discipline

Run the control yourself, on the **same nodes**, with the **same flags**,
changing **one variable**. Do not compare against an older results directory: in
practice those differ in flags nobody remembers setting. The one time this was
skipped it cost about three hours and produced a sign error in the conclusion.

A good sanity check is that device time and wall clock agree. A -10.8% device
delta against a -10.3% TPOT delta means the mechanism is understood; a large
disagreement means something off-device is dominating and the trace is not the
whole story.

## 4. Reading traces

Traces land in `results/<name>/traces/{prefill_only,decode_only}/plugins/profile/*/rank*.xplane.pb`
(64 files per capture, ~11 GB). Analysis uses `xprof`:

```python
from xprof.convert import raw_to_tool_data
data = raw_to_tool_data.xspace_to_tool_data([path], "hlo_stats", {})   # per-op self times
data = raw_to_tool_data.xspace_to_tool_data([path], "trace_viewer", {})# timeline, ~1 GB JSON
```

`hlo_stats` returns a Google DataTable (`{"cols": [...], "rows": [...]}`).
`trace_viewer` returns Chrome Trace Format, one event per line, so it streams
cheaply; device ops are `pid=3, tid=3` ("XLA Ops") for `/device:TPU:0`.

Four things that will mislead you:

- **The profiler inflates step time ~50x** (875 ms/step vs 16.6 ms unprofiled).
  Per-op *self times* are hardware counters and stay valid; *gaps between steps*
  are profiler overhead. Within a step the device is ~95-99% busy.
- **Measure a bubble to the next op that does real work**, not to the next
  event. Zero-duration `copy-start`s fragment one visible 1.3 us gap into a
  dozen 0.001 us gaps and it disappears from a naive per-pair histogram.
- **Fusion boundaries move with `compile_sizes`.** The same reduction appeared
  as `multiply_reduce_fusion` at bucket 1 and was absorbed into a generic
  `fusion` at bucket 2 -- same total op count, different partitioning. Compare
  like buckets.
- **Framework names differ from HLO names.** The trace viewer's `mean` is
  `multiply_reduce_fusion`; match on the `Framework op name` column when
  hunting for something you saw in the UI.

## 5. Tear down

```bash
scancel <jobid>
sinfo -p ici-2x2x4-a -o "%.12T %N"    # confirm the nodes are idle
```

Traces are ~11 GB per capture. Delete superseded ones.

## Files

| file | purpose |
|---|---|
| `slurm_launch_k3_tp32_tp.sbatch` | Ray + server on 4 hosts; pins the flags; prints the banner |
| `launch_kimi_k3_tp32_ep.sh` | the `vllm serve` invocation (TP=32, EP, mxfp4) |
| `arm_check.sh` | canary -> GSM8K -> sweep -> profile, one decode arm |
| `arm_check_prefill.sh` | canary -> prefill sweep -> MLA tuned-key check -> 8k profile, one prefill arm |
| `concurrency_sweep.sh` | `vllm-bench` across concurrency levels |
| `prefill_sweep.sh` | `vllm-bench` across exact input lengths (TTFT, osl=1) |
| `eval_gsm8k.sh` | GSM8K via lm-eval against a live endpoint |
| `diff_arms.py` | rank-0 decode HLO diff between two arms |
| `hlo_cache.py` | parse a phase's rank-0 hlo_stats once, cache as JSON |

`vllm-bench` (the standalone Rust client) is used rather than `vllm bench
serve`, which imports torch and dies on a login node under vLLM 0.27 with
`RuntimeError: ... device type at start of device string: tpu`. K3 ships no
`tokenizer.json`, so the random dataset borrows a small local tokenizer; at
`--random-input-len 1` that cannot affect decode speed.
