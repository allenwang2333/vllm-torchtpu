# DeepSeek-V4 vllm-torchtpu vs. tpu-inference Comparison: Commands + Design

**Date:** 2026-07-30. Written at the end of the `patemotter/dsv4-rebuild-on-main` rebuild
session (see `docs/dsv4_running_and_testing.md` and `docs/dsv4_architecture_and_rationale.md`
for the investigation history this continues). This doc has two parts: commands that work
today with zero new code, and a design for the not-yet-built per-layer comparison harness
that's been the standing next step since session 6.

**Status of the underlying bug this is chasing:** DSV4 runs clean end-to-end (verified this
session on `patemotter/dsv4-rebuild-on-main`: 43 layers, 48 decode tokens, zero NaN, zero
crash, clean shutdown) but collapses into a short deterministic greedy-decode repetition loop
almost immediately. This session's exact repro: prompt `"Compare XLA and LLVM in two bullet
points."`, output `,Irästyle,Irästyle,...` (token IDs `[14, 42498, 1419, 8618]` repeating from
step 1 onward, never varying). `tpu-inference` is confirmed (user-verified, see
`docs/dsv4_architecture_and_rationale.md`) to produce coherent output on the same checkpoint,
so this is a real vllm-torchtpu bug, not a checkpoint/quantization quality ceiling.

---

## Part 1: Commands that work today (no new code needed)

### 1.1 Baseline e2e run (already established)

```bash
bash scratch/run_dsv4_docker.sh
```

Or directly:

```bash
MODEL_PATH="/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"

docker run --init --privileged --net=host --ipc=host --rm \
  -v /home/patemotter_google_com/workspace/vllm-torchtpu/src/vllm_torchtpu:/usr/local/lib/python3.12/site-packages/vllm_torchtpu \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e HF_HOME=/home/patemotter_google_com/workspace/hf_cache \
  -e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so \
  -e LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=1000000000 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1000000000" \
  -e TORCH_DIST_TIMEOUT=1200 \
  -e VLLM_RPC_TIMEOUT=1200000 \
  -e VLLM_TP_SIZE=8 -e VLLM_EP_SIZE=8 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e VLLM_FP4_CACHE_DIR=/dev/shm/dsv4-flash-real-fp4 \
  -e JAX_COMPILATION_CACHE_DIR=/home/patemotter_google_com/workspace/jax_cache \
  -e VLLM_ENFORCE_EAGER=1 \
  -e DSV4_MODEL_PATH="$MODEL_PATH" \
  -e DSV4_PROMPT="Compare XLA and LLVM in two bullet points." \
  -v /dev/shm:/dev/shm \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  vllm-torchtpu-local:latest \
  bash -c 'python3 -u /home/patemotter_google_com/workspace/vllm-torchtpu/examples/run_dsv4_inference.py'
```

Takes ~90s for a 48-token greedy decode (mostly eager-mode per-layer retracing). Run
detached (`setsid nohup ... & disown`) for anything longer -- see
`docs/dsv4_running_and_testing.md` §3 for the durability pattern.

**Gotcha, hit repeatedly this session:** if a previous `docker run` didn't clean up (even
with `--rm`), it can hold the TPU device and the next run fails immediately with
`FAILED_PRECONDITION: TPU initialization failed: open(/dev/vfio/N): Device or resource busy`.
Check `docker ps -a --filter ancestor=vllm-torchtpu-local:latest` for stragglers, `docker stop`
them, and verify the device is free with:

```bash
docker run --rm --privileged --net=host \
  -e TPU_LIBRARY_PATH=/home/patemotter_google_com/workspace/vllm-torchtpu/docker/libtpu.so \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  vllm-torchtpu-local:latest python3 -c "import jax; print(jax.devices())"
```

before re-running -- should print all 8 `TpuDevice`s.

### 1.2 Cheap single-repo activation trace (vllm-torchtpu only, no tpu-inference needed)

Add `VLLM_CHECK_NAN=1 DSV4_TRACE_ACTIVATIONS=1` to the same command above (or to
`run_dsv4_docker.sh`):

```bash
-e VLLM_CHECK_NAN=1 \
-e DSV4_TRACE_ACTIVATIONS=1 \
```

This exercises `check_nan()` in `layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py`
(fixed this session -- was previously silently dead code due to a mis-indentation bug), called
at ~16 named points per attention layer (`"q (after rope)"`, `"kv (after rope)"`,
`"o (input to o_proj)"`, weight checks, etc.).

**Not yet run this session** -- this session's baseline e2e run (Part 1.1) did not set
`VLLM_CHECK_NAN=1`, so `check_nan()`'s first line (`if os.getenv("VLLM_CHECK_NAN") != "1":
return`) made it a no-op the entire time; it produced zero evidence either way about NaN
presence in *this* run. The "the model is NaN-free" claim in the investigation history
(e.g. `docs/dsv4_running_and_testing.md`'s session 3 "zero NaN detections" result) is from
past sessions' runs, not verified fresh here -- per this repo's own convention
(`docs/dsv4_running_and_testing.md`'s "never trust an old log's numbers" note), don't treat
that as settled without a fresh check if it matters for a decision.

If run with `VLLM_CHECK_NAN=1 DSV4_TRACE_ACTIVATIONS=1` and the model is in fact still
NaN-free, the *expected* behavior (from reading `check_nan()`'s code, not from having run
it) is that the NaN branch won't fire and every call site instead prints:

```
[ACT_TRACE] [<layer_prefix>] <name>: shape=[...] mean=... std=... min=... max=...
```

~16 lines × 43 layers ≈ 700 lines total, if that prediction holds. Grep/scan for a layer
where `std` spikes, collapses toward zero, or `mean` shifts abruptly -- a cheap way to get a
*rough* candidate layer before committing to the full cross-repo comparison in Part 2. This
does **not** tell you what's wrong, only "something looks statistically off starting around
layer N."

```bash
docker run ... vllm-torchtpu-local:latest \
  bash -c 'python3 -u .../run_dsv4_inference.py' 2>&1 | grep "ACT_TRACE" > /tmp/act_trace.log
```

Then eyeball `/tmp/act_trace.log` for the discontinuity, e.g.:

```bash
awk -F'std=' '{print $2}' /tmp/act_trace.log | awk '{print $1}' | \
  paste -d' ' <(grep -oE '\[model\.layers\.[0-9]+' /tmp/act_trace.log) -
```

(adjust to taste -- the point is eyeballing std/mean trend per layer, not a polished script).

### 1.3 The NaN auto-dump workflow (already exists, for if/when NaN reappears)

Documented in full in `docs/dsv4_running_and_testing.md` §4. `check_nan()` dumps all
attention inputs for a layer to `scratch/nan_dump_model.layers.<N>.attn_rank<R>.pt` the moment
it detects NaN. Not useful for the *current* garbled-but-clean bug, but keep in mind if a
future change reintroduces NaN.

---

## Part 2: Design for the per-layer hidden-state comparison harness

### 2.1 Goal

Localize the first layer (and ideally the first *operation within* that layer) where
vllm-torchtpu's computed hidden state diverges meaningfully from tpu-inference's, for the
identical prompt + checkpoint + greedy decoding. This turns "the whole model is somehow wrong"
into "layer N's <specific op> is wrong," which is the same bisection strategy that found every
real regression in this investigation so far (sessions 5-9), just applied at the
whole-forward-pass level instead of kernel-unit-test level.

### 2.2 Why this hasn't been done yet

Every session since 6 has recommended it as the next step and none has started it, because
the two codebases are architecturally different enough that "just add a print statement in
both" doesn't quite work:

- **vllm-torchtpu**: PyTorch/XLA, eager-mode per-layer retracing (`VLLM_ENFORCE_EAGER=1`),
  vLLM's `nn.Module`-based decoder layers, multiprocess TP=8/EP=8 workers.
  `models/vllm/deepseek_v4_patch.py`'s `patched_layer_forward` already wraps every decoder
  layer's `forward()` (see `deepseek_v4_patch.py:1191` from this session's stack traces) --
  this is the natural hook point.
- **tpu-inference**: JAX-native, `jax.jit`-compiled, functional (no persistent `nn.Module`
  instances holding state the way PyTorch does), likely SPMD-sharded across the mesh
  differently than vllm-torchtpu's TP/EP split.

A hidden-state dump that works for one won't drop into the other unchanged; each side needs
its own instrumentation, correlated by layer index and decode step.

### 2.3 Proposed architecture

**Capture format (shared, so the comparison script is symmetric):** one file per
`(side, layer_idx, step_idx)`, e.g. `scratch/hidden_states/{vllm_torchtpu,tpu_inference}/
layer{N:02d}_step{S:03d}.npy`, containing the residual-stream hidden state
`[num_tokens, hidden_size]` as float32 (cast up from bf16/fp8 before saving, so the comparison
metric isn't confounded by storage dtype). Also save `positions`/`seq_lens` alongside for
sanity-checking token alignment between the two runs (prefill token count must match
exactly, or the comparison is meaningless).

**vllm-torchtpu side:** extend `patched_layer_forward` in `deepseek_v4_patch.py` (already
wraps every layer) with an env-gated hook (matching the `VLLM_CHECK_NAN` pattern -- add e.g.
`DSV4_DUMP_HIDDEN_STATES=1`) that saves the layer's output residual stream after
`_orig_layer_forward` returns, tagged with layer index (from `self.layer_idx` /
`self.custom_prefix`, both already used elsewhere in this file) and a monotonically
incrementing step counter (rank 0 only, matching `check_nan`'s existing `rank == 0` gating
to avoid 8x redundant TP-replica dumps).

**tpu-inference side:** needs new instrumentation from scratch (checked this session: no
existing `TRACE`/`dump`/`activation`-related hooks in
`tpu_inference/models/vllm/experimental/deepseek_v4.py` or its custom_ops files). The
equivalent hook point is whatever function tpu-inference's decoder-layer loop calls per
layer -- start by reading `tpu_inference/models/vllm/experimental/deepseek_v4.py`'s top-level
forward loop (mirroring how vllm-torchtpu's monkey-patches `patched_layer_forward` around
vLLM's own per-layer loop) to find the equivalent per-layer boundary. Since tpu-inference is
`jax.jit`-compiled, the dump call needs `jax.debug.callback` (or similar host-callback
mechanism) rather than a bare Python `print`/tensor `.numpy()` call, or the dump needs to
happen on a code path that runs *outside* the jitted region (e.g. between decode steps, not
inside the per-token loop) -- this is the main new-code risk in this design, worth a small
spike before committing to the full harness.

**Test harness requirements (both sides):**
- Same checkpoint path, same prompt, same tokenizer.
- Greedy decoding (`temperature=0`) on both -- already vllm-torchtpu's default in
  `run_dsv4_inference.py`; confirm tpu-inference's `offline_inference.py` invocation matches
  (its `SamplingParams` construction, not yet checked this session).
- Disable KV-cache/scheduling nondeterminism: `VLLM_ENFORCE_EAGER=1` on the vllm-torchtpu
  side (already used); check whether tpu-inference needs an equivalent eager/no-cudagraph
  flag for bit-reproducible per-layer values.
- Match `max_model_len`/prompt length exactly, and pick a prompt long enough to cross the
  HCA `compress_ratio=128` compression boundary (position 128) if the compression path is
  still a suspect -- per `docs/dsv4_running_and_testing.md` session 8, no run in this
  investigation has ever shown a distinct signature change at that boundary, which weakly
  deprioritizes it, but a real per-layer comparison crossing it would be more conclusive than
  the qualitative check done so far.
- Same quantization path (both should be loading the checkpoint's native FP8 dense +
  native-packed-FP4 MoE experts; confirm tpu-inference's quant config resolves the same way).

### 2.4 Comparison metric

**Not exact equality** -- PyTorch/XLA and JAX numerics will never match bit-for-bit even in
fully correct code (different reduction orders, different fused-op decompositions). Use, per
layer, per token position:
- Cosine similarity between the two hidden-state vectors.
- Relative L2 error: `||a - b|| / (||a|| + eps)`.

Plot (or just tabulate) both metrics vs. layer index. A real bug shows up as a sharp
discontinuity -- cosine similarity near 1.0 for layers 0..N-1, then a sudden drop at layer N
that persists (doesn't recover) for all later layers. Gradual drift across all layers instead
of a discontinuity would suggest the two implementations are just accumulating ordinary
floating-point divergence, not evidence of a discrete bug -- a real possibility worth
considering, since it would mean this comparison approach doesn't cleanly localize the issue
and a different strategy is needed.

### 2.5 Execution plan (bisection, not exhaustive)

Once both sides can dump a given layer's hidden state on demand:

1. Dump layer ~21 (middle of 43) on both sides. Compare.
2. If diverged: bisect into layers 0-21. If matched: bisect into layers 21-43.
3. Repeat until the exact first-diverging layer is found (~6 iterations for 43 layers,
   `log2(43) ≈ 5.4`).
4. Once localized, the actual bug-finding within that layer is comparatively cheap --
   diff that layer's vllm-torchtpu file against its tpu-inference counterpart directly (the
   same methodology that found every real regression in sessions 5-9), now scoped to one
   layer's worth of code instead of the whole model.

This is meaningfully cheaper than dumping and comparing all 43 layers up front, and doesn't
require the full harness to support arbitrary-layer dumping simultaneously -- a single
"dump layer K" env var (`DSV4_DUMP_LAYER=K`) on each side is sufficient; no need to dump
every layer in a single run.

### 2.6 Known risks / things that could invalidate a naive comparison

- **Prefill vs. decode kernel paths differ** (already verified consistent with each other
  within vllm-torchtpu alone, per `docs/dsv4_running_and_testing.md` session 4's
  `verify_prefill_vs_decode.py` finding) -- but this hasn't been cross-checked against
  tpu-inference. Compare at a prefill position first (simpler, no KV-cache-history
  dependency) before attempting a decode-step comparison.
- **EP/TP sharding differences**: vllm-torchtpu runs TP=8/EP=8; confirm tpu-inference's
  reference run uses a comparable or reconcilable sharding config, since expert routing
  could assign different (but equally valid) experts under different sharding, making a
  naive numeric comparison noisy even when both are "correct."
- **fp4 cache**: vllm-torchtpu's run depends on the pre-quantized FP4 expert cache
  (`VLLM_FP4_CACHE_DIR`); confirm tpu-inference's dequant/requant path produces numerically
  equivalent (not just functionally equivalent) expert weights, or the very first MoE layer
  will show a spurious divergence unrelated to any real bug.

### 2.7 Estimated effort

Rough, not a commitment: a few hours for a careful implementer who reads both codebases
first -- most of the time goes to (a) finding tpu-inference's correct per-layer hook point
under `jax.jit`, and (b) the sharding/quantization-equivalence checks in §2.6, not the
comparison script itself (which is a small, mechanical numpy diff once both sides can
produce comparable dumps).
