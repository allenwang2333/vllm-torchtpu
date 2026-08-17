# DeepSeek-V4: remaining structural work

> **Last updated 2026-08-17.** Companion to `dsv4_tracker_20260816.md`
> (evidence, run log, refuted claims) and `dsv4_next_session_20260816.md`
> (pickup prompt). Where any older doc disagrees, this and the tracker win.
>
> An earlier revision named these items G1–G5. The codes are gone; the mapping
> is at the end of the tracker.

## Objective

Accuracy parity with tpu-inference's DeepSeek-V4, expressed in the structure
`kimi_k3` already uses in this repo. Accuracy parity is **met** — MMLU-Pro
0.7821 ± 0.0243 against the reference's 0.7786 ± 0.0154. What remains is
structural, plus one memory question.

Not in scope: byte-level parity with tpu-inference, its memory characteristics,
data-parallel attention, or the performance gap against it.

## Status

| Item | State |
|---|---|
| KV cache over-allocation | **Done** (`36abf661`). Utilization 0.55 → 0.75 |
| Restructure the model package like `kimi_k3` | **Done** (`421698f0`), gate byte-identical |
| Submit the native model refactor | **Open, ungated.** Needs Pate |
| Memory utilization above 0.75 | Open. Cause is compile scratch, unmeasured |
| MHC op registration | **Dropped** — misattributed to the reviewer |

## Done: KV cache over-allocation

Root cause was not the allocation but the overlay plan. Host selection used one
global set of already-used hosts, so with 21 compressed-latent layers only 21 of
43 sliding-window caches could overlay and the other 22 took standalone
full-size arrays — 1.3223x the memory vLLM had budgeted. Cache groups have
disjoint block-id namespaces, so one array can host one layer from each group,
which is what the reference does.

Result: overlay 42 of 43, allocation 0.9839x of budget, gate byte-identical,
long-context probe passes, utilization 0.55 → 0.75 (KV 309,223 → 485,612 tokens
at ctx 9216).

The earlier plan — "fold the DeepSeek-V4 branch into the shared allocation tail"
— turned out to be inapplicable: that path computes
`num_blocks = tensor_size // total_group_page_size` per tensor, but under vLLM's
packed layout `tensor_size` is the whole slab. The TODO at `tpu_runner.py:5364`
stands as tidiness only.

**Before submitting anything:** strip the `[KV_PROBE]` instrumentation
(`_probe_ds_v4_kv_budget` in `tpu_runner.py`).

Full measurement in the tracker.

## Done: the model package mirrors `kimi_k3`

Split in `421698f0`: `layers.py` (164), `moe.py` (142), `attention.py` (628),
`model.py` (491), and `custom_ops/.../deepseek_v4_attention_op.py` (207) for the
weightless half of attention. `DeepseekV4Head` was inlined into `Model.forward`
in the same change. Compiled gate passed with generations byte-identical to
`gate_09f6c694c.sub`; imports re-verified on the real TPU backend.

The rule applied, and worth reusing: **the model package holds what holds
weights.** Kernel dispatch, backend subclasses and cache subclasses stay in
`custom_ops/`. That is what `kimi_k3` does — its `attention.py` imports
`custom_ops/kda_attention_op.py` rather than absorbing it.

Details and the one deliberate wart (`_attn_op_cache` crossing the new module
boundary) are in the tracker.

## Open: submit the native model refactor

Nine commits, `ff1033a2..HEAD`, absent from `origin/main`. This is what gxd3
asked for on PR #462 and it has never been sent.

**Nothing gates it.** Accuracy is settled, compiled serving works at 9216, and
the earlier advice to hold it until the MHC cleanup landed rested on a
misattribution — the MHC item was never something gxd3 raised.

Ordering is settled: the package was restructured first (`421698f0`), so what
gets submitted is already the final layout rather than a 795-line file a
reviewer would be asked to read and then re-review after a split.

The agent never opens PRs. It can draft the split and the PR bodies.

## Open: memory utilization above 0.75

0.92 fails in compilation, not in cache allocation: *HLO temporaries (106.4G)
exceeds available HBM (94.74G)*. Utilization is the driver, proven by holding
the passing configuration fixed and changing only that; batch shape and context
length are both refuted.

There is no knob that raises the 94.74 GB — it is genuine physical memory
(`xla_tpu_user_reserved_hbm_bytes = -1`, `xla_tpu_max_hbm_size_mib = 0`). Demand
has to come down, and rematerialization is the untried lever: host offload is
off and nothing under 10 MB is a candidate.

**Measure before turning anything.** Re-run the diagnostic with
`--xla_dump_buffer_assignment_analysis` and **no** `--xla_dump_max_hlo_modules`
— that cap counts modules in creation order and discarded the one that failed. A
shortfall made of one huge buffer and one made of thousands of 8 MB buffers
point at different flags.

0.75 works today, so this is a capacity improvement rather than a blocker.

## Dropped: MHC op registration

`layers.py:116` assigns `HCHeadOp.forward_native` onto a vLLM class at import
time, where the reference registers four subclasses through vLLM's out-of-tree
hook. The item claimed this was "the pattern gxd3 asked us to leave on PR #462".
Verified against the threads on 2026-08-16: it is not. His #462 comment asks for
a native `model.py` plus a registry entry — that is the refactor above, already
implemented. No comment on #459, #462 or #464 mentions MHC.

The difference is real but cosmetic and gates nothing. See the tracker for the
one finding worth keeping: our HC-head body is higher precision than the
reference's, so a future port must move ours rather than copy theirs.

## Verification standard

Every code change above needs the compiled gate:

```bash
setsid nohup scratchpad/gate_dsv4_compile.sh . <tag> > scratchpad/<tag>_wrapper.log 2>&1 &
```

Pass is `tracebacks=0 sweeps=13 raises=0`, needle exact (`7391-QUARTZ-4408`),
and generations diffed against `scratchpad/gate_09f6c694c.sub` — **not**
`gate_cache_compile.sub`, which predates the requantizer removal.

Commit before gating and tag the run with the short SHA. A dirty tree makes the
SHA meaningless.

The gate runs at 1024 tokens and proves nothing about long context. For anything
touching compilation or memory, also run:

```bash
setsid nohup scratchpad/probe_compile_ceiling.sh 9216 > scratchpad/<tag>_wrapper.log 2>&1 &
```

Accuracy only needs re-checking if a change moves weights. MMLU-Pro at
`--limit 20` is ~30 min compiled; the bar is 0.78 ± 0.03.
