# The end-to-end A/B: what it takes to stand up a long-context GLM-5.2 run

Reconnaissance only. **Nothing here has been run.** This file exists so the finding survives the
session — the kernel work is hours away from needing it, and the checkout facts below are not
recoverable from the exercise's other documents.

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

## Route

**Copy the CDK JobSet, do not touch the repo.**
`~/cloud-devkit/recipes/experimental/multihost/torchtpu/vllm-torchtpu-glm-5.2-v7x-16/vllm-torchtpu-glm-5.2-benchmark.yml`
is fully hardcoded with no templating — `--max-model-len=9216` at `:128`, `--random-input-len=8192`
at `:230` — and it lives **outside** the `vllm-torchtpu` repo, so copying and editing it touches
neither deny-listed directory.

The alternative, the `vllm-torchtpu-generic-bench` recipe that takes `VLLM_SERVER_CMD` and
`BENCH_CMD` as job arguments, is cleaner but the local `~/cloud-devkit` checkout has only the
**single-node 4-chip** variant registered (`recipes.yml:422-425`, topology `2x2x1`). GLM-5.2 at
DP=16 needs the 2-node one. Try `git -C ~/cloud-devkit pull` first; if the 2-node generic recipe
appears, prefer it, because passing the commands as arguments means `cdk job recipe` reads them
back verbatim and there is no edited YAML to misquote.

Do **not** use the third route — `EXTRA_SERVE_ARGS` / `EXTRA_BENCH_ARGS` appended to a direct
`run_benchmarks.sh` invocation. They do win on argparse last-occurrence, but
`.buildkite/scripts/run_multihost.sh:185-205` does not forward them into the containers, and
trap 1 still fires because the skip check reads the config's 9216 rather than the override.

Whichever route: **read `BENCH_CMD` and `VLLM_SERVER_CMD` back from `cdk job recipe` and quote the
read-back**, not the command as sent.

## The shape, and the one thing that is not settled

Take the shape from `instruments/shapes.py:229-236`, which already carries it as the arm this A/B
targets: **ISL 65536, OSL 1024, `max_model_len` 66560**. That is exact — 65536 + 1024 = 66560 — and
it satisfies the harness's `ISL + OSL ≤ max_model_len` rule with nothing to spare, which is what
trap 1 punishes.

**Concurrency and `num_prompts` are open and must be measured, not chosen.** The microbenchmark's
own ladder (64 sequences at 9216, 12 at 33792, 6 at 66560) is a bound on the *instrument's* host
memory and says nothing about what the server will hold. Run a short smoke — a handful of prompts
at a low concurrency — read what vLLM reports for KV cache blocks, and size the real run from that.
With one run per arm there is no replication to absorb a failed arm, so the smoke is not optional.

Order of magnitude, for planning only: the default run is 2048 × 9216 ≈ 18.9 M tokens. Sixty-four
prompts at 66560 is ≈ 4.3 M, so a long-context arm of that size is not obviously longer in
wall-clock than the nightly. DSA keeps attention at `k = 2048` regardless of context, so prefill
should stay closer to linear than a dense model's would. **All of that is reasoning, not a
measurement**, and the smoke replaces it.

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
