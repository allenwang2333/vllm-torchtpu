# Verifying DSv4 P/D disaggregation (PR #684)

## What actually needs proving

PR #684 lets Raiden carry DeepSeek-V4's KV. Three of its claims can fail
silently, which is what makes them worth a plan rather than a smoke test.

The first is block-ID disjointness. Four DSv4 caches share one array at byte
offset 0, and the only thing keeping them apart is that the allocator never
gives one block ID to two cache groups. If a tag reads the wrong group's block
table, the transfer moves real bytes to real pages and nothing raises. The
decode replica then attends over another cache's KV and produces fluent,
wrong text.

The second is sliding-window trimming. `_stage3_trim_window_pages` drops the
dead prefix of a windowed group's block table, and producer and consumer
compute that count independently. If the two disagree by one page, their page
lists shift relative to each other and every subsequent page lands in the
wrong place.

The third is per-group page geometry. DSv4's SWA group pages at an eighth of
its CSA group, so any check that uses one connector-wide page size is wrong
for five of the six groups.

None of these announce themselves. A run can be healthy, fast, and wrong.

## What the PR's own verification did and did not cover

The PR served DP4/DP4 on one tpu7x-8, sent two requests, and got back
`{"text": "4", "prompt_tokens": 16}` plus a coherent completion. That proves
admission works, nine tags plan across six groups, and the transfer completes
without raising. It is real evidence and it rules out the loud failures.

It does not touch the quiet ones. DSv4-Flash has `sliding_window: 128`, so at
16 prompt tokens `_stage3_skipped_window_pages` computes
`max(0, 16 - 128 + 1) = 0` and the trimming returns its input unchanged. The
most delicate arithmetic in the PR never ran. Two short requests also cannot
distinguish correct KV from plausible-looking KV, because a 2-token completion
of "what is 2+2" is recoverable from almost any cache state.

So the gap is not the topology. It is prompt length and the absence of an
accuracy oracle.

## Tier 0: offline, no TPU

Already green in the PR: `tests/distributed/` at 289 passed, 2 skipped,
including `test_dsv4_block_id_disjointness.py`. Re-run on the merge commit
before anything else, because it is thirty seconds and it catches rebase
damage.

```bash
python -m pytest tests/distributed/ -q
```

## Tier 1: single host, DP4/DP4, on alynie-v7x-8

This is the tier that closes the real gap, and it needs no new hardware. The
existing `launch_dsv4_dp4_dp4.sh` brings the pair up; what changes is what we
send through it.

**T1.1 Exercise the window trim.** Send prompts well past 128 tokens. At
`MAX_MODEL_LEN=9216` a 4096-token prompt makes `skipped_tokens = 3969`, so the
trim drops a large majority of the SWA block table and any off-by-one in the
shared arithmetic misaligns the rest. Sweep several lengths rather than one,
because the interesting cases are the boundaries: 127, 128, 129, then 512,
4096, 8192. Lengths straddling a page boundary matter more than large ones.

**T1.2 Needle-in-a-haystack accuracy oracle.** Embed a random code deep in a
long prompt and ask for it back through the proxy. This is the check that
catches a cross-group read, because recall of a specific token from the middle
of the context requires the transferred KV to be correct, not merely present.
`run_qwen35_raiden_pcp8_dp8_live_pair.sh:890` (`run_meaningful_oracle`) already
implements exactly this against a P/D pair and can be pointed at the DSv4
proxy unchanged.

Pass criterion: the needle comes back verbatim on every request. Not "mostly",
not "semantically close". A single miss means stop and investigate.

**T1.3 Disagg output must match non-disagg output.** Serve DSv4 on the same
host without the connector, greedy decode with a fixed seed, and diff the
token IDs against the disagg run on the same prompts. Exact equality is the
right bar for greedy decoding, and it is the strongest cheap signal available:
if any tag reads the wrong pages, the sequences diverge, usually within a few
tokens of where the window starts biting.

**T1.4 Assert the plan in the logs.** The admission summary should show nine
tags across six groups. Grep prefill and decode for the tag list and fail if
it differs:

```
dsv4.csa.nope.g0  dsv4.csa.rope.g0  dsv4.hca.g0  dsv4.idx.g0
dsv4.swa.g1       dsv4.swa.g2
dsv4.state.idx.g3 dsv4.state.csa.g4 dsv4.state.hca.g5
```

Also assert that `raiden_stage3_dsv4_request_blocks_registered` reports
`group_pages` with per-group counts in the expected 8:1 ratio between the CSA
and SWA groups, and that no `ManifestError` appears. A run where every group
reports the same page count means the per-group page size is not taking
effect.

**T1.5 Concurrency.** Everything above with `--max-num-seqs 8` and several
requests in flight, because block-ID collisions are far likelier when the
allocator is under pressure and reusing freed pages.

## Tier 2: two hosts, DP8/DP8, via cdk

This is the configuration the PR calls step 1 and marks itself draft against.
It cannot run on `alynie-v7x-8`, which is a tpu7x-8: 8 chips, `HOST_BOUNDS
1,1,1`. DP8 prefill plus DP8 decode needs 16 chips across two nodes.

Route: `cdk` on `alynie-cdk-vm` (10.128.0.35). There is no DSv4 recipe yet, so
one has to be written; the template is
`recipes/benchmark/vllm-torchtpu/qwen3.5-397b/vllm-qwen35-397b-pd-pcp8-dp8-raiden.yml`,
which already runs two `2x2x1` tpu7x nodes with prefill on one and decode on
the other over `TPURaidenConnector`. Note that `2x2x1` on tpu7x is four chips
and eight devices, so one node is a DP8 replica.

What must change from the Qwen template:

- `--prefill-context-parallel-size 8` becomes `--data-parallel-size 8` on the
  prefill side; both replicas run DP attention with TP1.
- `TPU_RAIDEN_QWEN35_ADMISSION=1` becomes `TPU_RAIDEN_DSV4_ADMISSION=1`.
- `TPU_RAIDEN_TRANSFER_PARALLELISM` drops from 8 to 1. DSv4 admission raises
  if it is anything else, because DP attention gives every rank whole requests.
- `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL` flips from 1 to 0. The unified pool
  gives an `aliased_raw` binding and `build_dsv4_pool_manifest` requires
  private typed tensors, so it would be rejected at admission.
- The DSv4 model env from `launch_dsv4_dp4_dp4.sh` comes across whole:
  `VLLM_PLUGINS=torchtpu,torchtpu_layers`, `USE_MOE_SPARSE_CORE=0`,
  `MOE_REQUANTIZE_WEIGHT_DTYPE=fp4`, `ENABLE_QUANTIZED_MATMUL_KERNEL=0`,
  `--kv-cache-dtype fp8_e4m3`, `--no-disable-hybrid-kv-cache-manager`.
- The model becomes DeepSeek-V4-Flash and the image a recent nightly
  (`torchtpu-vllm-dev:nightly-20260903`), with the PR checkout mapped in via
  `--map-dir` and placed ahead of the image's copy on `PYTHONPATH`, since #684
  is not merged.

Then Tier 1's oracles run again unchanged against the two-host proxy. The
point of Tier 2 is not to re-prove correctness from scratch; it is to show
that nothing in the transfer depends on producer and consumer sharing a host,
which is precisely what the single-host run cannot rule out.

**Known risk to check first.** The image pins `tpu_sync`, and the PR calls
`facade.register_request_blocks(..., pool_spans=...)` per tag. If the pinned
`tpu_sync` predates multi-tag span registration, admission fails at startup
rather than silently, so this shows up in the first two minutes of the first
job. Check it before burning a two-hour slot.

## Results, run of 2026-09-03 on alynie-v7x-8

Tier 1 ran against `launch_dsv4_dp4_dp4.sh` on this host, DP4 prefill on local
ranks 0-3 and DP4 decode on 4-7. Admission is healthy and the transfer is
correct over a wide range of prompt lengths, but a short prompt kills the
prefill replica, so Tier 1 is red and Tier 2 is not worth booking yet.

**T1.4 passed, and the page geometry is wider than this document assumed.**
Both replicas log `pool admission complete ... pools=188 storages=84 tags=9`,
with all nine tags across the six groups. The per-group page sizes are 1024
tokens for the four group-0 tags, 128 for the two SWA groups, and 32, 16 and
32 for the indexer, CSA and HCA state groups. That is a 64:1 spread, not the
8:1 written above, which makes the case for per-group geometry stronger than
the PR's own comment claims.

**T1.1 and T1.2 passed.** The needle oracle returned all 22 codes verbatim at
prompt lengths from 95 to 8775 tokens, through the proxy, twice over. The
transfer really carried them: the decode replica logged 23 submitted and 23
completed Stage-3 transfers with an external prefix cache hit rate of 100%, so
nothing was re-prefilled locally. At 8774 tokens the SWA trim drops 63 of 64
pages and producer and consumer still agreed page for page. This is the part
the PR's 16-token probe could not have shown.

**T1.3 crashed the prefill replica, and T1.5 never got to run.** The first
exact-equality request, a 114-token prompt, took the whole prefill engine down
with an `EngineDeadError`. The root cause is this PR's own guard:

```
Stage-3 sliding-window block table still repeats a page after trimming:
group=dsv4.state.csa.g4, kept=[0, 0], num_tokens=113, page_tokens=16, window=8
```

The guard is right to fire; what it caught is real. The whole eight-entry
block table was the null block, because the companion warning about a
non-uniform dropped prefix did not fire even once. So the trim's stated
premise, a run of null pages followed by live ones, does not hold here.

The two arithmetics are identical, which is what makes this subtle.
`SlidingWindowManager.get_num_skipped_tokens` is
`max(0, num_computed_tokens - sliding_window + 1)`, exactly what
`_stage3_skipped_window_pages` computes. They differ in the argument. vLLM
passes `processed_computed_tokens`, the scheduler's committed prefix, and the
connector passes the request's `num_tokens`. For all eight blocks to be null,
vLLM's skip must have covered at least 128 tokens, so its count had reached
135 or more while the connector still said 113. The disagreement is not
between producer and consumer; it is between the connector and the scheduler.

It reproduces deterministically from a fresh pair, twice out of two attempts,
but only after history: run the oracle sweep, then send a ~114-token prompt.
The same prompt sent first, to an idle replica, completes normally. So the
trigger depends on allocator state, not on length alone, which is why both the
PR's smoke test and a long-prompt sweep miss it from opposite sides. `113` is
also small enough that ordinary chat traffic will reach it.

Two things follow for the PR. The connector should derive the skip from the
same committed-token count the scheduler used rather than from the request
length. Separately, a mis-trim should fail the request, not raise out of
`execute_model` and kill every other request on the replica.

## Order and cost

Tier 0 is seconds. Tier 1 is roughly 20 minutes of startup, most of it MoE fp4
requantization, plus the oracle runs, and it uses hardware already allocated.
Tier 2 needs two shared TPU nodes for about two hours and should not be
attempted until Tier 1 is green, because every failure Tier 1 catches would
cost far more to diagnose across two hosts.

If Tier 1 fails on T1.2 or T1.3, that is the PR's central claim breaking and
Tier 2 is pointless until it is fixed.
