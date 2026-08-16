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
