> **SUPERSEDED 2026-08-17 by `dsv4_next_session_20260817.md`.** Kept only
> because it records the state at that handoff. Its item codes (G1-G5, D1)
> no longer exist; the mapping is at the end of the tracker.

# Start here — DeepSeek-V4, next session

> **Written 2026-08-16 19:41 UTC.** Valid until superseded by a later-dated
> `dsv4_next_session_*.md`. Read `dsv4_tracker_20260816.md` alongside it.

Paste the block at the bottom as the opening prompt. Read
`docs/dsv4_tracker_20260816.md` before doing anything else; it is the source of truth
and supersedes anything remembered from a previous conversation.

## State as of 2026-08-16 evening

Branch `patemotter/dsv4-native-model`, eight commits ahead of the last handoff,
tree clean. Everything below is verified on hardware unless marked otherwise.

- **Accuracy parity is met.** MMLU-Pro 0.7821 ± 0.0243 on `09f6c694`, against
  0.7893 the same morning and 0.7786 on 2026-08-12.
- **Compiled serves at 9216.** The earlier "compile ceiling" was an artifact of
  a recursion-limit removal that has been reverted. Do not use eager: it costs
  ~3× on generation.
- **The requantizer is gone**, replaced by the standard FP8 block path, proven
  bit-identical to tpu-inference by `scratchpad/compare_scale_equivalence.py`.
- **Design parity is the current objective** — the G rows in the tracker.

## What to pick up, in order

1. **G5** — DSv4 allocates KV outside the profiler's envelope. It fails to start
   at 0.92, 0.85 *and* 0.83 `gpu-memory-utilization` while every other model runs
   0.92-0.95, and the shortfall is **non-monotonic** (0.83 left less free than
   0.85), so there is no threshold to find by nudging. **Start with the
   measurement, not an edit:** log `num_blocks × actual page bytes` against
   `kv_cache_tensor.size` at `tpu_runner.py:5239`. Then fold DSv4 into the shared
   allocation tail — Pate's own TODO at `tpu_runner.py:5364` says the same.
   Highest value: it costs real serving capacity.
2. **G1** — replace the `HCHeadOp.forward_native` monkey-patch with
   `register_oot` subclasses, copying `tpu_inference/layers/vllm/custom_ops/mhc.py`.
   Five-step plan in the tracker. This is the pattern gxd3 asked us to drop.
3. **G3/G2** — split `model.py` into `layers.py`/`moe.py`/`model.py` as
   `kimi_k3` does, and inline the `Head` class as the reference does.
4. **D1** — the nine-commit native-model refactor is still unsubmitted. No
   longer blocked: A8 was refuted, and PR11 was dropped as a consideration.
   Whether to send it before or after G1 is Pate's call; see the plan.

Full plan with steps, verification and risk per item:
`docs/dsv4_design_parity_plan_20260816.md`.

**No DSv4 throughput number exists.** A benchmark config was added but never
completed a run — every attempt died on the G5 allocation failure. The only
throughput figures are the serving-log medians in A13 (eager 6.9 tok/s vs
compiled 20.4 tok/s generation).

## Rules that cost time when ignored

- Commit before gating. Today's runs shared a commit but differed in working
  tree, which made the SHA meaningless.
- The only way to tell if the TPU is free is scanning `/proc/*/fd` for vfio.
  `lsof /dev/vfio/*` and `pgrep -f entrypoints.openai.api_server` both lie.
- Free the chips with `scripts/vllm/benchmarking/cleanup_server.sh`.
- One run at a time. Never `pytest` while serving.
- A passing gate at `DSV4_MAX_MODEL_LEN=1024` proves nothing about long context;
  two separate failures this session lived entirely above that size.

## Opening prompt

```
Read docs/dsv4_tracker_20260816.md and docs/dsv4_next_session_20260816.md first — they are the
source of truth and supersede any memory of earlier sessions, several of which
recorded claims that were later disproved (see the Refuted table).

We are on branch patemotter/dsv4-native-model. Accuracy parity with
tpu-inference is met and compiled serving works at 9216; the current objective
is design parity with tpu-inference's DSv4 expressed in the structure kimi_k3
uses in this repo — the G rows.

Start with G5, the KV allocation divergence. Confirm the cause with the one log
line described in that row before changing anything, and tell me what you find
before you start editing.
```

## Opening prompt — unattended variant

Use this instead when Pate is away and work should proceed without check-ins.

```
Read docs/dsv4_tracker_20260816.md, docs/dsv4_design_parity_plan_20260816.md and
this file first. They are the source of truth and supersede any memory of
earlier sessions — several recorded claims that were later disproved, see the
Refuted table. Branch is patemotter/dsv4-native-model, tree clean, TPU idle.

I am asleep. Work unattended until I'm back. You have standing authorization to
edit, commit, and run TPU jobs. Do NOT push, open PRs, post review comments, or
merge.

Work the design-parity plan in order, starting with G5:

1. G5 step 1 is a MEASUREMENT, not an edit. Log num_blocks × actual page bytes
   against kv_cache_tensor.size at tpu_runner.py:5239, run once, and read it.
   Most of the previous session's waste came from acting on inferences that were
   never checked — do not skip this.
2. If the measurement shows the accounting disagreement, fix it and fold the
   _is_ds_v4 branch into the shared tail per the TODO at tpu_runner.py:5364.
   Then find the highest gpu-memory-utilization that starts, target 0.92.
3. Then G1: the register_oot port for MHC, five steps in the plan.
4. Then G2+G3 together, as one change sharing one gate.

Discipline, because TPU time is cheap and round-trips are not:
- Batch edits, then one validation chain. Never edit-gate-edit-gate.
- Commit BEFORE gating and tag the run with the short SHA. A dirty tree makes
  the SHA meaningless.
- Gate pass: tracebacks=0 sweeps=13 raises=0, needle exact 7391-QUARTZ-4408,
  generations diffed against scratchpad/gate_09f6c694c.sub — NOT
  gate_cache_compile.sub.
- The gate runs at 1024 tokens and proves nothing about long context. For
  anything touching compile or memory, also run probe_compile_ceiling.sh 9216.
- One run at a time. Check the TPU is free by scanning /proc/*/fd for vfio;
  lsof and pgrep on the server name both lie. Free it with
  scripts/vllm/benchmarking/cleanup_server.sh.
- Never run pytest while serving.

Record every run in the tracker's run log with its config dimensions, and add
anything disproved to the Refuted table. If something is inferred rather than
measured, label it inferred in the same sentence.

Stop and wait for me if: a gate fails twice for the same reason, accuracy moves
outside 0.78 ± 0.03 on MMLU-Pro, or a change would touch kernels or the
attention numerics path. Leave the TPU idle and the tree clean whenever you
pause.
```
