# The end-to-end A/B: what it takes to stand up a long-context GLM-5.2 run

Written as reconnaissance, and now partly executed. This file exists so the finding survives the
session — the checkout facts below are not recoverable from the exercise's other documents.

**Status.** The route is settled (see § Route) and the smoke run of step 3 has been submitted. The
shapes, traps and the confound are unchanged and still unverified against hardware. Nothing in the
A/B itself has been run.

Scope: the caller asked for an end-to-end benchmark of GLM-5.2 comparing the incumbent
`streamindex_topk` against whatever this exercise produces, **at a long-context shape** and at
**one run per arm**. The long-context choice is theirs and is recorded in `report.md` § "Why long
context"; the one-run-per-arm choice means the result is a single unreplicated measurement per arm
and must be reported as one.

Sign convention, as everywhere in this exercise: percentages are changes in **time**, negative is
faster, and signed percentages appear only in tables under a `Δ time %` column.

---

## What is checked in today

`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh` is the only GLM-5.2 benchmark config,
and **both `scripts/vllm/benchmark/**` and `scripts/vllm/benchmarking/**` are deny-listed for
edits** — deliberately, so the harness cannot change underneath a measurement. Nothing below edits
them.

| knob | value | line |
| --- | --- | --- |
| `MODEL` | `zai-org/GLM-5.2-FP8` | `:18` |
| `TENSOR_PARALLELISM` / `DATA_PARALLELISM` / `ENABLE_EP` | 1 / 16 / true | `:22-24` |
| `ISL_OSL_CONFIGS` | `8192:1024` | `:27` |
| `CONCURRENCY_OPTIONS` | `1024` | `:33` |
| `NUM_PROMPTS` | `2048` | `:34` |
| `RANDOM_RANGE_RATIO` | `0.0` — fixed-length prompts | `:37` |
| `MAX_MODEL_LEN` | **9216** | `:43` |
| `MAX_NUM_BATCHED_TOKENS` | **1024** | `:49` |
| `MAX_NUM_SEQS` | 64 *per DP engine*, so 1024 across 16 | `:54` |
| `GPU_MEMORY_UTILIZATION` / `KV_CACHE_DTYPE` | 0.90 / fp8 | `:57-58` |

Topology is tpu7x v7x-16 — 2 hosts × 8 chips (`.buildkite/pipeline_perf.yml:146-156`). The client
is **`vllm bench serve`**, not the InferenceX client, and it takes
`--dataset-name random --random-input-len N --random-range-ratio 0.0`
(`scripts/vllm/benchmarking/run_benchmarks.sh:410-431`). The InferenceX client under
`scripts/vllm/benchmark/inferencex/qwen3.5/` is Qwen-only and 8-core-only and cannot serve this.

**There is no checked-in long-context config for GLM-5.2, or for any DSA model in this repo.**
Every one is `MAX_MODEL_LEN=9216`.

## The model will go much further than 9216

From the checkpoint's own `config.json` (`gs://tpu-commons-ci/glm/GLM-5.2-FP8/config.json`):
`max_position_embeddings` is **1048576**, and `rope_parameters` is `{"rope_theta": 8000000,
"rope_type": "default"}` — unscaled, so no YaRN to enable and no rope-scaling ceiling. The same
file confirms `index_topk: 2048`, `index_n_heads: 32`, `index_head_dim: 128`,
`num_hidden_layers: 78`, matching `instruments/shapes.py:42-45`.

Neither derived cap in the runner binds at 64k: the min-page-size check
(`platforms/tpu_block_size_utils.py:367-373`) stays slack until `max_model_len > 1,048,576` at
`max_num_seqs=64`, and `runner/tpu_runner.py:731-734` yields 2016 against a `max_num_seqs` of 64.
Page size for MLA/DSA is hardcoded to 1024 (`layers/vllm/attention.py:1323-1324`) and does **not**
move with `max_model_len`.

**The real ceiling is HBM, and this repo does not state it.** There is no checked-in per-token KV
figure for GLM-5.2 and no v7x per-chip HBM budget. So concurrency at 64k has to be found
empirically, not derived. Do not put a guessed number in a recipe.

## Three traps, all of which fail quietly

1. **`ISL + OSL > MAX_MODEL_LEN` silently skips the cell.** `run_benchmarks.sh:578-581` prints
   `Skipping isl…_osl…` and exits 0. A job that skipped its only cell is green and has no numbers
   in it. This is precisely the case `CLAUDE.md` means by "a green `JobStatus` says nothing about
   which shape was benchmarked" — read the `bench` log, do not trust the status.
2. **`MAX_NUM_BATCHED_TOKENS` must stay at 1024.** `glm-5.2-fp8-dp16-ep.sh:44-49` records that
   2048 per engine faults the SparseCore at DP=16. Longer inputs therefore chunk into more prefill
   steps rather than bigger ones. Raising it to "make long context go faster" is the obvious wrong
   move and it is written down as such.
3. **Result filenames and the regression key are hardcoded from the config, not from the actual
   flags.** `run_benchmarks.sh:588` and `:512-521` name the cell `isl8192_osl1024_c1024` regardless,
   and that is the key `check_regression.py` looks up in
   `baselines/perf/glm-5.2-fp8-dp16-ep.baseline.json`. A long-context run pushed through
   `run_eval_flow.sh` would be **scored against the 8k baseline**. This A/B must not go through the
   regression path at all; it is a direct comparison of two arms measured in this session.

## Route — **resolved, use the generic 2-node recipe**

This section originally recommended copying the hardcoded JobSet, because the local `~/cloud-devkit`
had only the single-node generic recipe. **The `git pull` this section told the next session to try
was run, and it settles the question in favour of the better route.** `~/cloud-devkit` moved
`c4e1d4e` → `4c6383f`, and **`vllm-torchtpu-generic-bench-2node` is now registered**
(`recipes.yml:484-487`, `k8s_file: recipes/experimental/torchtpu-vllm-generic/generic-bench-2node.yaml`).

Use it. It takes `VLLM_SERVER_CMD`, `BENCH_CMD`, `DOCKER_IMAGE`, `MODEL_ID`, `MODEL_GCS_PREFIX`,
`SERVER_PORT` (default 8000), `SERVER_WAIT_TIMEOUT` (default 7200) and `ENABLE_XPROF` as **job
arguments**, so `cdk job recipe` reads both commands back verbatim and **no YAML is edited at all** —
neither in the repo nor in a copy. That removes the whole class of failure where a misquoted edit
silently benchmarks a different shape. It appends `--host` itself (`generic-bench-2node.yaml:337`),
so `BENCH_CMD` must not carry one.

Two things to carry over rather than rediscover:

- The known-good GLM-5.2 serve line is in the *old* route's YAML at
  `vllm-torchtpu-glm-5.2-benchmark.yml:126-142`. Take it verbatim and change only `--max-model-len`
  and the port. Flags that look irrelevant to a random-dataset benchmark — the tool-call and
  reasoning parsers — are cheaper to keep than to justify dropping.
- The generic recipe resolves weights through `MODEL_ID` / `MODEL_GCS_PREFIX` (default
  `gs://cloud-devkit/ckpt/hf`) and exports `MODEL_PATH`; the old GLM recipe resolves from HF by
  name. These are **different mechanisms**, and whether the generic one finds `zai-org/GLM-5.2-FP8`
  with `MODEL_ID` unset must be checked by reading both YAMLs before submitting, not discovered by
  losing a slice. A checkpoint exists at `gs://tpu-commons-ci/glm/GLM-5.2-FP8/`.

The third route — `EXTRA_SERVE_ARGS` / `EXTRA_BENCH_ARGS` on a direct `run_benchmarks.sh`
invocation — remains rejected, for the reasons below.

`EXTRA_SERVE_ARGS` / `EXTRA_BENCH_ARGS` appended to a direct
`run_benchmarks.sh` invocation do win on argparse last-occurrence, but
`.buildkite/scripts/run_multihost.sh:185-205` does not forward them into the containers, and
trap 1 still fires because the skip check reads the config's 9216 rather than the override.

Whichever route: **read `BENCH_CMD` and `VLLM_SERVER_CMD` back from `cdk job recipe` and quote the
read-back**, not the command as sent.

## The smoke run — measured, and it sizes the A/B

**Job `j-7a8ba86c-dca3-4141-82e7`**, `JobStatus: Succeeded`, recipe `vllm-torchtpu-generic-bench-2node`,
image `…/vllm-torchtpu-annyan_google_com:13617694` (= `origin/main` `13617694`, the baseline arm's
image — it is servable, and no rebuild is needed for the A/B). `~/cloud-devkit` at `4c6383f`, dirty
only with a pre-existing untracked file that was not touched.

**It boots at 66560.** No KV-capacity refusal, all 16 engines up, 8/8 requests succeeded. That was
the single finding that could have invalidated the whole plan, and it is now closed.

### Model resolution — the open question in § Route, answered

**`MODEL_ID` must be set.** Left unset the generic recipe sets `MODEL_PATH=""` and skips the GCS
copy entirely (`generic-bench-2node.yaml:151-165`), leaving `vllm serve zai-org/GLM-5.2-FP8` to pull
**703.74 GiB from HF Hub on both hosts**. The settled form, taken from a prior *Succeeded* GLM-5.2
job on this exact recipe (`j-8d43bb89-8bac-4197-883e`) rather than from reading the YAML:

- `MODEL_ID=zai-org/GLM-5.2-FP8`, `MODEL_GCS_PREFIX` left at its default `gs://cloud-devkit/ckpt/hf`
  — the checkpoint is there and complete (149 objects, 755,663,674,511 B, within 1.6 KB of the
  `gs://tpu-commons-ci/glm/` copy this file previously named, which is therefore not needed)
- serve `${MODEL_PATH} --served-model-name=${MODEL_ID}`, which keeps the served id as
  `zai-org/GLM-5.2-FP8` so `BENCH_CMD` needs no change
- prefix the server command with `export VLLM_ENGINE_READY_TIMEOUT_S=14400 &&`, and pass
  `SERVER_WAIT_TIMEOUT=14400`, `RAY_WAIT_TIMEOUT=7200`. **These are load-bearing, not padding** —
  see the wall-clock below. The generic recipe does not set `VLLM_ENGINE_READY_TIMEOUT_S` at all
  and its default would very likely not have held.

### KV cache, and what it permits — this is the deliverable

Identical on all 16 `EngineCore_DP*`:

```
GPU KV cache size: 424,960 tokens
Maximum concurrency for 66,560 tokens per request: 6.38x
Memory statistics | total_hbm_limit_gb=94.74GiB | total_hbm_limit_cap_gb=85.26GiB | total_hbm_used_gb=63.35GiB | total_hbm_avail_gb=21.91GiB
Using KV cache block size: 1024
```

This build prints no `# GPU blocks` line; `GPU KV cache size` plus `Maximum concurrency` is all
there is. The arithmetic:

- per engine `424,960 / 1024 = 415` KV blocks; one request at ISL 65536 + OSL 1024 peaks at exactly
  `66,560 / 1024 = 65` blocks, so `415 / 65 = 6.38` — matching vLLM's own figure
- **6 whole sequences per engine**, 390 blocks used, 25 spare
- **ceiling 6 × 16 = 96 concurrent** across DP=16. `6.38 × 16 = 102` is *not* the answer: a sequence
  cannot span engines, so the per-engine floor is taken first
- cross-check: `21.91 GiB / 424,960 tokens = 55,360 B/token`, ≈ 96 layers × 576 B/token/layer of
  fp8 MLA latent+rope. The two independent numbers agree.

**`--max-num-seqs=64` per engine is far above the KV limit of 6, so KV is the binding constraint,
not `max-num-seqs`.** Leave `--max-num-seqs` alone; it is not the knob.

**Settled for the A/B: `--max-concurrency=64`, `--num-prompts=192`.** 64 is 4 per engine at 62% KV
occupancy. The ceiling of 96 is 100% occupancy with zero headroom, and with
`--no-enable-prefix-caching` any DP imbalance there turns into preemption and a full recompute of a
65k prefill — a risk not worth taking on a run with **no replication to absorb a bad arm**.
`num_prompts` is a multiple of 16 so the DP engines finish level, and 192 gives three waves of 64
rather than the two that 128 would, so start-up and tail effects are a smaller share. `modelled`,
from the smoke's throughput; it is a planning figure, not a measurement.

### Wall-clock — budget ~45 minutes of startup per arm, every time

| phase | duration |
| --- | --- |
| checkpoint copy from GCS (both nodes, ~2 s skew) | **~783 s** (13m03s) |
| vLLM weight load off local NVMe | 37.7–57.8 s per worker |
| Torch/Mosaic compile, cold cache | **~28 min — the dominant term** |
| job create → `/health` green | **2,675 s = 44m35s** |
| 8 prompts at concurrency 4 | 187.3 s |
| total job | 49m22s |

**The compile cache does not persist**, so every arm pays the ~28 minutes again:
`compile_prefix isolation: DeepseekV2Model has no self.prefix; using non-deterministic fallback
_tpu_instance_0 (its on-disk compile cache will not persist across restarts)`. Budget each arm at
roughly 45 min fixed plus its benchmark, and **run the two arms concurrently on separate slices** —
four 16-chip slices were free, and the caller explicitly asked for parallelism.

That parallelism has a cost worth stating in the report: the two arms then run on **different
physical slices**, and with one run per arm there is no replication to separate a slice-to-slice
difference from a kernel difference. Running them sequentially would not fix it either — different
pods, and a ~45 min gap. It is a limitation of the one-run-per-arm design, not of the parallelism.

### The smoke's own numbers, for the record and nothing else

`ENABLE_XPROF=0`, so a single pass, the block after `bench-start-event: name=benchmark`.
**8 prompts at concurrency 4 is not a performance result and must not be quoted as one** — only 4
of the 16 DP engines were ever busy.

```
Successful requests: 8      Failed requests: 0      Benchmark duration (s): 187.30
Total input tokens: 524288  Total generated tokens: 8192
Median TTFT (ms): 49928.76  Median TPOT (ms): 42.59  Median ITL (ms): 42.52  Median E2EL (ms): 93649.65
```

**The bench really ran at 65536**, confirmed four ways, the last being arithmetic on the output
rather than an echo of the input: the recipe read-back; the bench container's parsed
`Namespace(random_input_len=65536, …)`; `Sampling input_len from [65536, 65536]`; and
`Total input tokens: 524288 = 8 × 65,536` exactly with `Total generated tokens: 8192 = 8 × 1,024`
exactly. Trap 1 did not fire. Zero `preempt` matches in the server log.

One warning that will recur on every run and is benign: `safetensors_load_strategy='prefetch' …
checkpoint total size (703.74 GiB) exceeds 90% of available RAM (124.73 GiB)`. It did not OOM —
prefetch degrades to a partial page-cache warm and weights still loaded in 38–58 s. Also
`/root/.cache` is 70% full at 704 G of a 1024 Gi PVC: room for this checkpoint, not for a second.

## The shape, and the one thing that is not settled

Take the shape from `instruments/shapes.py:229-236`, which already carries it as the arm this A/B
targets: **ISL 65536, OSL 1024, `max_model_len` 66560**. That is exact — 65536 + 1024 = 66560 — and
it satisfies the harness's `ISL + OSL ≤ max_model_len` rule with nothing to spare, which is what
trap 1 punishes.

**Concurrency and `num_prompts` were open and have now been measured** — see § The smoke run.
`--max-concurrency=64`, `--num-prompts=192`. The reasoning this section originally carried, that a
64k arm might not be longer in wall-clock than the nightly, has been replaced by the smoke and
should not be quoted: the fixed startup alone is ~45 minutes.

One prediction the smoke did *not* contradict, kept because it still bounds expectations: DSA holds
attention at `k = 2048` regardless of context, so prefill scales closer to linearly than a dense
model's would. Still `modelled`, and the A/B measures it.

## A confound this exercise created and must not hide

`report.md` records that `num_sublanes_total` (`streamindex_topk.py:603-605`) derives from
**`max_model_len`, not from the batch's actual KV length**, so the f32 scores buffer is sized by the
advertised context for every call — 2.6 MB at `max_model_len=9216`, 276.8 MB at 66560, including for
short decodes that will never use it.

Two consequences, and they pull in opposite directions:

- It is part of why the indexer's share grows at long context, which is what makes the A/B
  measurable at all.
- It means **some of any e2e win measured here is a win against padding that only exists because
  the config advertises 64k.** A kernel that sizes the buffer to the real KV length would show a
  large win at `max_model_len=66560` and little at 9216.

That is a legitimate result for a deployment that actually serves 64k, and a misleading one if
quoted as a property of the kernel. Whatever the A/B returns, report it as **measured at
`max_model_len=66560`, ISL 65536** and say that the nightly 9216 configuration is a different
question that this run does not answer. `report.md` also carries the counterweight for the other
phase: decode indexer cost is close to invariant to advertised context length, because resident
tokens per rank are capped by HBM rather than by `max_model_len`, so the growth is in prefill and
in the mixed steps. Lead the report on **P99 ITL**, which is where indexer work lands.

## Sequence, once the kernel exists

1. `/preflight-check` on step 4's `final_report.md` candidate, dispatched as a general-purpose
   subagent with the change, the target shape, the image and the checkout path quoted into it.
2. `/critical-review` on the preflight evidence — **before** the run it gates, with the raw
   `cdk job recipe` read-back pasted verbatim into the dispatch.
3. Smoke run at ISL 65536, few prompts, to size concurrency.
4. The A/B: baseline arm and candidate arm, one run each. **Re-measure the baseline**; do not
   compare against any logged number, since none was taken at this configuration.
5. `/log-experiment` after every one of those that produced a number, including the smoke if it
   did. A **new** experiment file — no existing one will match this `## Environment`.
6. `/critical-review` again on the finished A/B.
