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

**Staging, which is done.** The recipe is
`examples/disagg/cdk-dsv4-pd-dp8-dp8-raiden.yml`. Both of its out-of-band
inputs now exist. Unmerged PR code ships to cdk as a GCS tarball rather than
over NFS: `gs://cloud-devkit/repos/vllm-torchtpu/prs/` holds fifteen of them,
and each consuming recipe copies one down, untars it to
`/workspace/vllm-torchtpu`, and puts `src` at the front of `PYTHONPATH`. This
recipe does the same with `pr684.tar.gz`. The checkpoint had to be uploaded,
because nothing in the cluster had ever run a DeepSeek model: the recipes draw
models from `gs://cloud-devkit/ckpt/hf` and `gs://junyanxu-benchmark-models`,
and between them those hold only Qwen, nvidia/GLM, gpt-oss and llama. The
148.7 GiB snapshot now sits at
`gs://alynie-k3-scratch/ckpt/hf/deepseek-ai/DeepSeek-V4-Flash` and is handed to
`vllm serve --model` as a `gs://` URI, which is how every other recipe loads
its weights.

**Launching it without the cdk CLI.** `alynie-v7x-8` has no `cdk` binary and no
`cloud-devkit` checkout, but a cdk recipe is a JobSet with `{{ }}` placeholders
and without the metadata the CLI injects, so `kubectl` can submit it directly.
The cluster is `cloud-devkit-gke` in `us-central1`, the namespace is
`cloud-devkit-jobs`, and jobs are gated by Kueue through `cdk-local-queue`.
`gcloud container clusters get-credentials` writes a kubeconfig that wants
`gke-gcloud-auth-plugin`, which the snap `gcloud` will not install; setting the
user credential to `gcloud auth print-access-token` avoids the plugin entirely.
`examples/disagg/render_cdk_recipe.py` fills in the placeholder defaults, the
namespace, the queue label and the priority class.

Two things the recipe needed before it would work this way. It addresses peers
by pod DNS name, which the JobSet only publishes if `spec.network` sets
`enableDNSHostnames` and `publishNotReadyAddresses`, and the producer resolves
the consumer while the consumer is still loading weights, so not-ready
addresses have to be published too. The benchmark job also ran
`dsv4_pd_trim_repro.py`, which cannot reproduce anything through the proxy for
the reason given under T1.3, so it is replaced by `dsv4_pd_guard_probe.py`
against the producer's own port, followed by a second oracle pass to catch a
producer that died just after answering.

**Known risk to check first.** The image pins `tpu_sync`, and the PR calls
`facade.register_request_blocks(..., pool_spans=...)` per tag. If the pinned
`tpu_sync` predates multi-tag span registration, admission fails at startup
rather than silently, so this shows up in the first two minutes of the first
job. Check it before burning a two-hour slot.

## Results, run of 2026-09-03 on alynie-v7x-8

Tier 1 ran against `launch_dsv4_dp4_dp4.sh` on this host, DP4 prefill on local
ranks 0-3 and DP4 decode on 4-7. Admission is healthy and the transfer is
correct over a wide range of prompt lengths. A short prompt killed the prefill
replica, which is T1.3 below; with the guard added in this branch Tier 1 is
green and Tier 2 is worth booking. The underlying disagreement is contained
rather than repaired, so read T1.3 before treating the green as a clean bill of
health.

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

**T1.3 crashed the prefill replica.** A 114-token prompt took the whole prefill
engine down with an `EngineDeadError`, out of this PR's own guard:

```
Stage-3 sliding-window block table still repeats a page after trimming:
group=dsv4.state.csa.g4, kept=[0, 0], ids=[0, 0, 0, 0, 0, 0, 0, 0], skipped=6,
num_tokens=113, page_tokens=16, window=8
```

The guard is right to fire, and what it caught is worse than a crash. Dumping
every producer block table under `TPU_RAIDEN_DEBUG_BLOCK_TABLES=1` shows the
same defect firing quietly on the ordinary path. Across 178 requests through
the proxy, 14 shipped a page the allocator had already taken back: six in
`state.csa.g4`, four in `state.hca.g5`, three in `state.idx.g3`, one in each
SWA group. The crash is the same bug with the disagreement large enough that
two dead pages collide.

The connector and the allocator disagree about which pages are dead because
they measure from different positions. The connector skips
`_stage3_skipped_window_pages(num_tokens)`, where `num_tokens` is the transfer
prefix, `prompt - 1`. The allocator freed pages against `num_computed_tokens`,
wherever the producer actually stopped. On a clean captured example the
producer computed 375 tokens and the prefix was 374: the allocator freed
`(375 - 8 + 1) // 16 = 23` pages, which is exactly the 23 null entries the
table shows, while the connector skipped 22. Page 22 went over the wire as the
null block. One page of drift is a hole in the decode replica's window that
nothing reports; the accuracy oracle still passes, because a single dead page
of an 8-token compressor-state window rarely changes which token comes out.

The crash needs the same drift, just more of it. It is not allocator history,
as the first pass at this suggested; it is the producer generating past the
prompt. The proxy pins the prefill leg to `max_tokens=1`
(`toy_proxy_server.py:272`), so through the intended path the producer stops at
`computed == prompt` and the drift stays under one page. A request sent
straight at the producer skips that clamp. Of 26 distinct request IDs in the
crashed run, 25 were proxy-generated UUIDs and exactly one was not, and that
one was the request that died. Reproducing it is a single request to port 8400
with `max_tokens=48`: the producer ran to 161 computed tokens, the allocator
freed 9 of its 11 pages, `_stage3_normalize_producer_blocks` kept the first 8
because the transfer prefix is 8 pages long, and all 8 were the null block.

The fix here counts the null blocks left in the transfer range, which reads
what the block table says rather than re-deriving where the allocator got to,
and refuses the send when any group has two of them. That request prefills on
the decode side instead, costing one prefill rather than an engine.

It deliberately lets the single dead page through, with a warning. Refusing
those too was the first attempt and it was worse: the decode replica is
provisioned to decode, with `compile_sizes:[256]`, and asking it to prefill the
oracle's 2174-token prompt walked off the end of its padding buckets and killed
it in `_get_padded_token_len`. Trading a quiet hole in one window for a dead
decode engine is not a trade worth making. So the crash is gone and the latent
corruption is now visible in the logs, but it is still there.

**Re-run with the guard in place.** The request that killed the engine now
returns its 48 tokens and both replicas stay at HTTP 200. The guard logged the
skip once, for exactly that request: the producer ran to 161 computed tokens
against a 113-token prefix, with dead pages `[(3, 1), (4, 2), (5, 1)]`, and
group 4's two dead pages are the duplicate destination Raiden would have
refused. Over the rest of the run, 148 Stage-3 sends, the oracle returned 18 of
18 codes through the proxy at prompt lengths from 95 to 4372 tokens, 128
concurrent requests at 16 in flight finished with no failures, and no send
raised the "still repeats a page" error. Nine sends carried a single dead page
and said so, which is the latent hole described above rather than anything the
guard introduced.

This is containment, not repair. The design still has both peers deriving the
skip from a token count that matches the allocator's only while the producer
halts at the handoff token. The repair is for the producer to publish the
number of pages it actually dropped, per group, rather than having the consumer
re-derive it. That is a wire-format change and it belongs to whoever owns the
PR. Worth noting alongside it: the decode-side prefill fallback, which the
connector reaches whenever it declines a transfer, is not survivable for long
prompts in this launcher's configuration. That is independent of this bug and
worth fixing on its own.

### Why no other model hits this

Qwen3.5 and GLM cannot reach the trim, and would not be affected if they
could. `_stage3_trim_window_pages` has exactly one caller,
`_stage3_dsv4_pool_pages`, and that has two: `_register_stage3_request_blocks_dsv4`
on the producer and the DSv4 branch of `_submit_stage3_loads` on the consumer.
Every other model registers its blocks through a different path. Even if one
did call it, `window_tokens` is filled in only by `_measure_dsv4_pool_plan`,
and it comes from the group spec's `sliding_window`. Qwen's Stage-3 pool tags
are `("fa",)` and GLM's are `("fa", "dsa.idx")`, both full attention, so
`_stage3_skipped_window_pages` returns on its first branch and hands the block
table back untouched.

The structural reason is worth stating separately, because it says which
future models are at risk. A full-attention group never frees a page while its
request is alive; its block table only grows, and every entry stays live. So it
does not matter that the connector and the allocator count computed tokens
differently, because the page list is the same under either count. DSv4 is the
first model to put groups whose pages are freed mid-request into Stage 3. Once
pages can be freed, what the block table holds depends on which token count
drove the freeing, and the connector has to use that same count rather than the
transfer prefix. That is the bug, and it is a property of windowed groups
rather than of DeepSeek.

The geometry makes DSv4 the worst case rather than a marginal one. Its
compressor state groups run a window of 8 tokens against a 16-token page, so
one page covers the whole live window and everything before it is dead prefix.
The trim is not shaving an edge; on any prompt past a hundred tokens it is
discarding the entire table but the tail. The two SWA groups, window 128
against a 128-token page, are the mild case. Qwen has neither.

## Order and cost

Tier 0 is seconds. Tier 1 is roughly 20 minutes of startup, most of it MoE fp4
requantization, plus the oracle runs, and it uses hardware already allocated.
Tier 2 needs two shared TPU nodes for about two hours and should not be
attempted until Tier 1 is green, because every failure Tier 1 catches would
cost far more to diagnose across two hosts.

If Tier 1 fails on T1.2 or T1.3, that is the PR's central claim breaking and
Tier 2 is pointless until it is fixed.
