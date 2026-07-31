# DeepSeek-V4 on TPU: How to Run, Test, and Debug (Session Playbook)

**Date:** 2026-07-27. Companion to `docs/dsv4_architecture_and_rationale.md` (the *why*/*what was found*); this doc is the *how* — exact commands, so anyone can pick this up cold.

We're on `alyssa` (the TPU v7x-8 host) directly — no SSH/rsync needed, unlike the older commands in `docs/handoff_notes.md`/`docs/tpu_execution_and_testing.md` which assume a remote workstation.

---

## 1. Environment

**For full-model runs (the garbled-text investigation specifically): use the Docker image, not a fresh `uv` venv.** This is a harder requirement than it used to be — see session 7 in §5: a fresh dependency resolution of `vllm==0.23.0` can silently be a *different, never-validated* build of vLLM's DeepSeek-V4 code than whatever's baked into this image, and at least one such difference (a flattened attention-module structure in newer builds, no `DeepseekV4MultiHeadLatentAttentionWrapper`) is a hard crash, not a subtle numeric difference. Six sessions of findings are anchored to this exact image's vLLM build; don't move off it for full-model runs without deliberately deciding to.

```bash
docker images | grep vllm-torchtpu-local
# vllm-torchtpu-local   latest   <id>   ...   21.1GB
```

**If this image is missing** (it disappeared mid-session-7, cause unknown — not this session's doing, and the `vllm-torchtpu-local:latest` tag was gone with no obvious replacement): first try re-pulling/re-authenticating rather than rebuilding from scratch:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
docker pull us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-patemotter_google_com:latest
docker build -f docker/Dockerfile.local -t vllm-torchtpu-local:latest docker/
```

**Correction (session 8): `docker/Dockerfile.local` is NOT a no-op FROM-only file** — it has two real build steps (`RUN rm -rf .../tpu_inference*` and `COPY libtpu.so ...`). A previous session's doc claimed pulling+retagging the raw registry image was sufficient; this is wrong and was verified wrong empirically (session 8: a raw pull+retag left `tpu_inference` present — `find_spec('tpu_inference')` returned non-`None`, i.e. `CONFLICT`, not `clean`). **Always run the `docker build` step**, not a plain `docker tag`, when restoring this image. (In session 8 the build's two custom layers hit the Docker layer cache and completed instantly, reproducing the exact prior image — but don't assume that cache will always be warm.) Verify if in doubt: same command below (must print `clean`, not `CONFLICT`).

```bash
docker run --rm vllm-torchtpu-local:latest python3 -c \
  "import importlib.util; print('clean' if not importlib.util.find_spec('tpu_inference') else 'CONFLICT')"
```

Source is mounted live (`-v .../src/vllm_torchtpu:/usr/local/lib/python3.12/site-packages/vllm_torchtpu`), so editing files in this checkout takes effect on the **next** `docker run` — no rebuild needed for source changes.

### 1.1. Native `uv` venv (kernel-level tests only, session 7)

For iterating on the kernel-level pytest suite (§2), a native venv is faster than Docker and now proven equivalent (30/30 tests pass, confirmed identical to Docker's results before the image disappeared). **Do not use this for full-model runs** — see the warning above.

```bash
uv venv --python 3.12 ~/uv_venv
source ~/uv_venv/bin/activate
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"
uv tool install keyring --with keyrings.google-artifactregistry-auth --force
export UV_TORCH_BACKEND=cpu
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git ~/workspace/vllm
sed -i '/tpu-inference/d' ~/workspace/vllm/requirements/tpu.txt
SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 VLLM_TARGET_DEVICE="tpu" uv pip install -e ~/workspace/vllm
cd /path/to/vllm-torchtpu && uv pip install --pre -e ".[test]"
```

No `TPU_LIBRARY_PATH`/`LIBTPU_INIT_ARGS` override needed — `pyproject.toml` already pins `libtpu==0.0.42.1` as a real pip dependency; the Docker command's `docker/libtpu.so` override is Docker-specific residue, not something a native venv needs.

**Two permission gotchas**, both from root-owned leftovers of prior Docker-based work blocking a non-root native build/run — fix with `sudo chown -R $(whoami) <path>` (they're just build/compilation caches, safe to reclaim):
- `src/vllm_torchtpu.egg-info` — blocks `uv pip install -e .` with `error: Cannot update time stamp of directory`.
- `/dev/shm/torch_tpu_cache` — blocks TPU compilation with a **native SIGABRT across every worker process**, not a Python exception: `Check failed: lock_fd_.valid() ... Permission denied` deep inside `torch_tpu::CompilationCache::GetFromTier2OrCompile`. Easy to misdiagnose as a code bug; it's pure permissions. If you still see this after chowning, also check `JAX_COMPILATION_CACHE_DIR` isn't pointed at another root-owned path.

---

## 2. Running the kernel-level pytest suite

This is the fast, cheap correctness gate — run it after *any* kernel change, before touching the full model:

```bash
docker run --rm --net=host --ipc=host --privileged \
  -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
  -w /home/patemotter_google_com/workspace/vllm-torchtpu \
  -e PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
  -e VLLM_PLUGINS=torchtpu,torchtpu_general \
  -e PJRT_DEVICE=TPU \
  -e JAX_PLATFORMS=tpu,cpu \
  vllm-torchtpu-local:latest \
  pytest -sv --use-tpu \
    tests/kernels/test_dsv4_mla.py \
    tests/kernels/test_dsv4_mla_swa.py \
    tests/kernels/test_dsv4_compressor.py \
    tests/kernels/test_dsv4_moe_routing.py \
    tests/kernels/test_dsv4_streamindex_topk.py \
    tests/layers/vllm/quantization/test_mxfp4_dsv4_correctness.py \
    tests/layers/vllm/quantization/test_mxfp4_tpu.py
```

**`--use-tpu` is required.** Without it, `src/sitecustomize.py` auto-detects a bare `pytest` invocation and silently switches JAX to CPU interpret mode (`PALLAS_INTERPRET=1`), which will pass/fail differently than real hardware and can mask real bugs.

**Required Flags:**
- **`-w /home/patemotter_google_com/workspace/vllm-torchtpu`**: Sets container working directory so `pytest` loads `tests/conftest.py` (which defines `--use-tpu`) and locates repo-relative paths.
- **`-e PJRT_DEVICE=TPU -e JAX_PLATFORMS=tpu,cpu`**: Initializes both TPU and CPU backends in JAX. TPU executes the Pallas kernels while CPU performs initial reference tensor quantization and correctness checks.

Current status: **all 30 tests pass** on real hardware (`tpu7x-8`, 8 chips) in ~48s.

---

## 3. Running the full 43-layer model end-to-end

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
  -e PYTHONWARNINGS=ignore \
  -e TRANSFORMERS_VERBOSITY=error \
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

Run this in the background (`run_in_background: true` if using Claude Code's Bash tool) — it takes **80s** just for engine init (weight loading + KV cache allocation), then several more minutes grinding through eager-mode per-layer kernel tracing. Tail the output file rather than waiting synchronously.

### Why each non-obvious env var is there
- **`TPU_LIBRARY_PATH` / `LIBTPU_INIT_ARGS`**: pins the exact `libtpu.so` build and sparse-core collective offload thresholds this branch was validated against (from `docs/handoff_notes.md`). Omitting these still runs, but on whatever `libtpu` ships in the image instead of the pinned one.
- **`VLLM_RPC_TIMEOUT=1200000` (20 min) / `TORCH_DIST_TIMEOUT=1200`**: the vLLM default `VLLM_RPC_TIMEOUT` is **10 seconds**, which is nowhere near enough for eager mode's first-token per-layer kernel tracing across 43 layers. Without this, you get a misleading `TimeoutError: RPC call to sample_tokens timed out` that looks like a hang but is really just normal (slow) eager-mode compilation — wasted real debugging time before this was caught. If you switch off eager mode, you can likely drop these back down.
- **`VLLM_ENFORCE_EAGER=1`**: needed historically to avoid host-memory OOM compiling the full 43-layer graph (`docs/old_attempts.md`). Worth revisiting once the model produces correct output — eager mode's per-kernel retracing is a plausible source of its own artifacts (flagged by the user mid-session), separate from the timeout issue above.
- **`VLLM_FP4_CACHE_DIR`**: points at the pre-quantized FP4 expert-weight cache in `/dev/shm` (already populated, ~169GB) — without it, startup does host-side requantization which takes ~1 hour instead of ~80s.

Model checkpoint path note: `$MODEL_PATH` above is the *full* 43-layer checkpoint. Do **not** use `dsv4_flash_1layer` (its first 2 layers are `compress_ratio=0`, dense-only — clamped to 1 by the vendor's `max(1, ...)`, so it never exercises SWA/CSA/HCA merge logic) as a substitute — it's not a valid smoke test for anything this branch has been fixing.

**`max_model_len`/`max_num_batched_tokens` are hardcoded to 64 in `examples/run_dsv4_inference.py`** (session 8 finding) — this is *why* every session's test scenario has topped out at 64 total positions (15ish prompt + 49 decode), never reaching the HCA `compress_ratio=128` compression boundary flagged as untested since session 5. Not a prompt-length coincidence; a hard `LLM(...)` config cap that silently rejects (`VLLMValidationError`, not a crash) any request whose prompt+max_tokens exceeds it. Session 8 added env var overrides: `DSV4_MAX_TOKENS` (sampling `max_tokens`, default `50`) and `DSV4_MAX_MODEL_LEN` (default `64`, also drives `max_num_batched_tokens` and `compilation_config.compile_sizes`, which needed to include the new length too — check this file's `main()` if bumping much further).

---

## 4. The NaN auto-dump debug workflow

`custom_ops/deepseek_v4.py`'s attention forward has debug instrumentation (search `NAN_DETECTION`) that, on detecting NaN in its output, prints `[PYTORCH_DEBUG_NAN]`/`[NAN_DETECTION]` and dumps **all** attention inputs for that layer/rank to:

```
scratch/nan_dump_model.layers.<N>.attn_rank<R>.pt
```

The dumped dict includes `q`, `kv`, `sw_cache`, `main_cache_kv`, `main_kv_lens`, `extra` (topk_indices for CSA layers / kv_lens_to_attend otherwise), `main_page_indices`, `main_cu_q_lens`, `main_distribution`, `attn_sink`, `output` — plus **`positions`** and **`compress_ratio`**, which were added mid-session (not in the original instrumentation) specifically to diagnose the layer-2 CSA issue in §3.7 of the rationale doc. Keep these two fields; they're needed to reconstruct `max_valid_kv` and correlate a token's absolute position with which compressed slots it should be attending to.

This is left armed intentionally (see rationale doc §4) — it's the single most useful tool for this investigation. Workflow:

1. **Before a run you care about, clear stale dumps**: `rm -f scratch/nan_dump_*.pt` — otherwise you can't tell which dumps are fresh (this tripped me up once this session; always check `ls -la --time-style=full-iso` against the run's actual timeframe if you forget).
2. **After a run, find the true origin layer**, not just any layer that has a dump — NaN propagates through the residual stream, so every layer *after* the origin will also show a dump. Compare `q`/`kv` (the attention op's own inputs) for NaN: if they're already corrupted, the bug is upstream of that layer; if they're clean but `output` isn't, that layer's own kernel is the origin.
3. **Inspect a dump** (must run inside the docker image — no local `torch`):

   ```bash
docker run --rm -v /home/patemotter_google_com/workspace:/home/patemotter_google_com/workspace \
     vllm-torchtpu-local:latest python3 -c "
   import torch
   data = torch.load('scratch/nan_dump_model.layers.<N>.attn_rank0.pt', map_location='cpu')
   for k, v in data.items():
       if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
           print(k, list(v.shape), 'NaN=', torch.isnan(v).sum().item(), '/', v.numel())
   "

   ```

4. **For byte-level cache inspection** (e.g. checking if a specific cache position is a raw uninitialized/poison byte pattern vs. a computed value): load `main_cache_kv`/`sw_cache` (shape `[num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]`), index by `main_page_indices[0]` to get the physical page, reshape to `[page_size, lkv_dim]`, and inspect raw bytes at specific `(row, col)` positions directly — see the session's actual analysis commands for the exact pattern (they're reconstructable from `kernels/deepseek_v4/mla.py`'s `load_bkv` function, which documents the exact byte layout: 448 fp8 nope + 128 bf16 rope (packed as 2×64 bytes) + 7 UE8M0 scale bytes + padding).
5. **`scratch/inspect_dump.py`** is a pre-existing (though `layers.0`-hardcoded) script doing a version of this — useful as a reference/template, edit the layer number before running.

Existing `float8_e4m3fn` NaN bit patterns (confirmed this session, `ml_dtypes`): raw bytes `0x7F` and `0xFF` decode to NaN. If you find these exact bytes in a cache dump, don't assume it's a quantization-rounding bug — **test that assumption first** (see below), it's very likely genuinely uninitialized memory instead.

---

## 5. Current investigation state (as of this doc)

**Confirmed and fixed** (see rationale doc for full details): the original `mla.py` output-write bug, a JAX Ref-unpacking incompatibility, a stale test reference, the `self.mla_attn` aliasing regression, profiling-pass `kv_cache` guards, and the `attention_sinks` unsliced-broadcast regression. All 12 kernel tests pass with these applied.

**Active investigation**: a reproducible NaN still occurs on the full 43-layer run, originating at an early layer (moved from layer 3 to layer 2 between two runs — both are compressed-attention layer types, HCA `compress_ratio=128` and CSA `compress_ratio=4` respectively). Byte-level inspection of the cache at the NaN'd positions shows raw `0x7F`/`0xFF` bytes.

**Ruled out**: a fp8-quantization rounding-overflow bug (hypothesis: a legitimately-computed value near `fp8_max` rounds up into the NaN-reserved bit pattern during the `.astype(jnp.float8_e4m3fn)` cast). Verified directly on CPU (`JAX_PLATFORMS=cpu`) that JAX's fp8 cast correctly saturates values even 1% over `fp8_max` — only truly astronomical inputs (`1e30`) produce NaN. An explicit `jnp.clip(..., -fp8_max, fp8_max)` was still added before both fp8 casts (`kernels/deepseek_v4/mla_swa.py`, `kernels/deepseek_v4/compress_norm_rope.py`'s `quantize_fp8_ue8m0`) as a harmless defensive improvement, but it is **not** the fix for this bug.

**Ruled out (round 2)**: an isolated-kernel write bug in `mla_swa.py`'s `_pack_new_kv`/`_update_kv_cache` alone. Built `scratch/poison_fill_repro.py` (pre-fills the SWA cache with `0xFF` poison, calls the real `mla_sliding_window_ragged_paged_attention` on real TPU hardware with a scenario matching the real dump exactly — 15-token fresh prefill padded to 16, matching page size/page-index-1 start) — every in-range row was correctly, fully overwritten. The SWA write kernel itself is fine in isolation.

**Confirmed architectural root cause**: `custom_ops/deepseek_v4.py`'s `_deepseek_v4_compressor_jax` (~lines 179-199) computes *both* the compressor's state slot mapping and its compressed-KV slot mapping by indexing into the **same** `state_block_tables` array:

```python
state_cache_block_num = positions // state_block_size + token_to_req_indices * max_num_blocks_per_req
slot_mapping = state_block_tables[state_cache_block_num] * state_block_size + state_cache_block_offset
k_cache_block_num = (positions // compress_ratio) // k_block_size + token_to_req_indices * max_num_blocks_per_req
kv_slot_mapping = state_block_tables[k_cache_block_num] * k_block_size + k_cache_block_offset
```

For any token before the first compression boundary (`compress_ratio=128`, positions 0-14: `k_cache_block_num` evaluates to `0` for every one of them), `kv_slot_mapping` resolves to `state_block_tables[0]` — the *same* physical page number the state mapping also uses for early positions. `tests/kernels/test_dsv4_compressor.py`'s own test harness (`_make_inputs`) explicitly avoids this by placing state and KV pages in disjoint ranges (`kv_base = num_state_pages * PAGE_SIZE`, with a comment calling out exactly this hazard) — the real model's indexing has no equivalent separation. Since HCA layers share their cache buffer with the main/SWA attention cache (`two_caches_same_buffer`), and the compressor's (unconditional, every-token) state write runs before the attention forward's own SWA write in the same layer's `attn_gemm` → `forward_mqa` call order, the state write's physical target can coincide with rows the SWA write is also responsible for — and the SWA write's own dense sub-word packing logic (`_pack_new_kv`) evidently doesn't cleanly re-cover every byte lane on top of it, leaving the residual corruption (columns 10/34 in specific rows) that was observed.

**Fixed and confirmed**: `custom_ops/deepseek_v4.py`'s `state_cache_block_num`/`k_cache_block_num` (in `_deepseek_v4_compressor_jax`) were missing the modulo wraparound against `max_num_blocks_per_req` that a circular/sliding-window buffer requires — real runtime values confirmed `state_block_size=1`, `max_num_blocks_per_req=8`, meaning any position beyond 8 aliased into the *next* request's block-table slice. Added `% max_num_blocks_per_req` to both the write side (`custom_ops/deepseek_v4.py`) and the read side (`gather_state_windows` in `compress_norm_rope.py`, which had the identical gap — fixing only one side would have created a new write/read mismatch). **Confirmed via a full model run**: layers 0-1 are now clean (previously the very first layer, an HCA type, already showed corruption). All 12 kernel tests still pass.

**New, separate issue found past this fix, not yet resolved — and likely a bigger question than a simple kernel bug**: the NaN origin has moved to layer 2 (a CSA layer, `compress_ratio=4`), specifically on a decode step at `position=15` where `extra` (topk_indices for the CSA "context-sparse attention" path) correctly selects compressed slots `[0,1,2,3]` (verified: `max_valid_kv=(15+1)//4=4`, matching this sequence having reached 4 valid compressed slots — the masking logic itself checks out). Scanned the physical cache page for the tell-tale `0x7F`/`0xFF` poison bytes at the raw page rows corresponding to those slot indices and found none.

Tracing further: the compressor's write-side page table (`state_block_tables`, passed into `_deepseek_v4_compressor_jax`) comes from `state_metadata.block_tables`, which is looked up via the compressor's *own* `state_cache.prefix` — i.e., it's allocated against `VllmCompressorStateCache.get_kv_cache_spec`'s `SlidingWindowMLASpec`. The attention read-side page table (`main_page_indices`, captured in the NaN dump) is allocated against the *outer attention layer's own* `MLAAttentionSpec`. These are **two independent vLLM KV-cache-spec allocations** — even though, per `custom_ops/deepseek_v4.py`'s `__init__` (`compressor.k_cache = self` for `compress_ratio > 1`), they're both meant to reference the *same* underlying physical cache tensor (`self.kv_cache`).

**UPDATE (session 2): the above paragraph's theory is refuted — do not re-investigate this path.** Traced vLLM's actual cache-group code (`tpu_runner.py`'s `get_kv_cache_spec`/`initialize_kv_cache`, vLLM's `kv_cache_utils.py`'s `_get_kv_cache_config_deepseek_v4`) and confirmed `num_blocks` is deliberately kept uniform across groups, and separately confirmed via a live diagnostic at the real forward pass that `attn_ctx[compressor_state_prefix] is attn_ctx[main_layer_name]` — literally the same Python object, for both layer 2 and layer 3. The block tables are synchronized by construction; this was a red herring (see rationale doc §3.8 for the full trace).

**Real root cause (session 2, fixed and verified)**: `mla.py` and `mla_swa.py`'s `flash_attention` functions zero the attention weight `p` at masked positions but never sanitize `kv` itself before `pv = einsum(p, kv)`. `0.0 * NaN == NaN` in IEEE-754, so poison bytes in an unwritten cache row still corrupt every query's output at that column despite the zero weight. Fixed in both files by zeroing `kv` rows that are masked for *every* query in the block, before the `pv` einsum (see rationale doc §3.8 for the exact diff and the Mosaic i1-reshape gotcha hit along the way: **never reshape/broadcast a boolean array directly on TPU — `jnp.where(bool_array[:, None], ...)` fails to compile with `unsupported shape cast`; cast to a float dtype first, then reshape that**). Verified: all 12 kernel tests pass with both fixes.

**Still open, next investigation**: even with both fixes, a full-model decode-step dump shows `q`/`kv` already NaN at layer entry (upstream corruption via residual stream) and `main_cache_kv` page 0 fully corrupted (1024 poison bytes across all 64 rows, vs. 15 stray bytes pre-decode) — consistent with a real *write* corrupting the page, not unwritten memory. This matches the "Known remaining gaps" item in `dsv4_correctness_fixes.md`: the compressor's per-token state write (unconditional, every token, shares the buffer with real KV data) may be clobbering real KV bytes for small/tiny-prompt positions. Next step: trace `_deepseek_v4_compressor_jax`'s `_compressor_local` state `slot_mapping` write against wherever the SWA cache's real K/V write lands, to confirm the overlap and find where genuine row separation is needed.

**Untracked stray files found this session, not yet cleaned up**: `mla.py` (repo root) and `src/vllm_torchtpu/layers/vllm/custom_ops/mla_swa.py` are both untracked, unimported duplicate/abandoned copies — safe to delete once the investigation concludes (kept for now in case anything in them is still a useful reference). Also new this session: `scratch/inspect_dump_layer3.py` (a reusable, non-hardcoded-layer version of `inspect_dump.py` — prints NaN column locations, `positions`/`compress_ratio`/`main_page_indices`, and scans `main_cache_kv` pages for poison bytes) and `scratch/nan_dump_*.pt` files (safe to `rm` before each fresh run, per the existing workflow in §4).

**Docker gotcha (session 2)**: after a run that hit real NaN/errors and needed killing, `docker stop`/`docker kill` on the container occasionally hang indefinitely (the docker CLI itself stays responsive — `docker ps` returns fine — but commands targeting that specific container ID don't return). This looked like a wedged container process, but a quick `docker run ... python3 -c "import jax; print(jax.devices())"` check confirmed the TPU device was actually free the whole time (all 8 cores visible) — the hang was purely in docker/containerd's bookkeeping for that one container, not a real device lock. If this happens again: don't block on the stuck stop/kill call, just try launching the next run directly (or a JAX-device-list smoke check) — it will likely work despite `docker ps` still listing the old container as "Up".

**Docker gotcha (session 3, more severe variant)**: a *previous run's container can outlive its own crashed Python process* and keep holding the TPU device (`/dev/vfio/*`), causing the *next* run/test invocation to fail immediately with `jax.errors.JaxRuntimeError: FAILED_PRECONDITION: TPU initialization failed: ... Device or resource busy`. Unlike the session-2 gotcha (a hung `docker stop` with the device actually free), this time the device really was busy — `docker ps` showed a stray `vllm-torchtpu-local` container still `Up`, from a test/run invocation whose host-visible pytest process had already exited. Fix: `docker ps` to spot any stray `vllm-torchtpu-local` container older than your current command, `docker stop <name>` it (worked fine this time, no hang), then retry. If a run/test fails with this exact `vfio`/`Device or resource busy` error, check for a stray container *first* before assuming a real code regression.

**Session 3 resolution — compressor state/real-KV collision (the §3.7/§3.8 lead from sessions 1-2) is fixed**: root cause was two compounding bugs in `VllmCompressorStateCache` (`custom_ops/deepseek_v4.py`) — see rationale doc §3.9 for the full trace and fix. In short: (1) the class recomputed `block_size` with a broken formula that produced `1` instead of the vLLM base class's already-correct `4`/`8`, making one compressor-state token clobber an entire physical page; (2) compressor state was writing into the *same physical tensor* as real/compressed KV (`compressor.k_cache.kv_cache`) even though vLLM's cache manager already allocates the compressor's `state_cache.kv_cache` as a genuinely separate, disjoint tensor — the code just never used it. Fixed by pointing state reads/writes at that dedicated buffer (with a dtype fix too — it was binding as `float8_e4m3fn` instead of `float32`/uint8-packed, per `kv_cache_spec_normalizer.py`'s dtype-override behavior) and fetching a separate block table for the shared main-buffer compressed-KV addressing, mirroring `tpu-inference`'s (the sibling JAX-native repo) reference implementation. **Verified: full 43-layer real-checkpoint run now produces zero NaN detections and no crash** (previously NaN by layer 3 on every run). Text generation is still garbled after the first token — a new, separate, more ordinary-looking issue (see rationale doc §5), not yet investigated.

**Session 4 — decode-path investigation (see rationale doc §5.1 for full detail).** Started from the §5 "garbled after first token" open item, in the order that doc's candidate list suggested:

- **EP combine**: code-reading only initially. Confirmed DSV4's actual MoE path (`fused_moe_gmm.py`) uses vLLM's own generic `allgather_reducescatter` combine, not the hand-written collectives kernel in `kernels/fused_moe/v1/kernel.py` (that kernel isn't even called for DSV4).
- **KV/position bookkeeping**: added temporary `DEBUG_DECODE_STEP` instrumentation to `forward_mqa` (gated to rank 0, layers 0 and 2) printing positions/seq-lens/kv-lens every call, then ran the full 43-layer real-checkpoint prompt again. **Verified correct** for all 49 decode steps on both an SWA-only layer and a CSA layer — position and KV length both advance exactly 1-for-1 with decode step count, and the CSA compressed length matches `(position+1)//4` at every step. This rules out a frozen/stale-cache bug, which was this session's leading hypothesis going in.
- **Per-step logprobs**: added `[STEP_LOGPROBS]` prints (all steps, not just step 1) to `examples/run_dsv4_inference.py`. The greedy collapse into `" the"` sharpens smoothly over 5 steps rather than jumping discontinuously — the shape of a real (if degenerate) language-model repetition attractor, not an obvious sign of numeric corruption.
- **Sampling test**: added `DSV4_TEMPERATURE` env var to the same script (default `0.0`, unchanged). `DSV4_SEED`/`SamplingParams(seed=...)` was also added but **does not work on this platform** — fails immediately with `ValueError: JAX does not support per-request seed.` (`tpu_platform.py:913`); left the plumbing in the script but it's non-functional, don't rely on it for reproducibility. At `temperature=0.8` (no seed), the literal token-repeat loop disappears, but the output is still incoherent and looks disconnected from the prompt (unrelated Chinese text, code fragments) rather than merely "a different plausible completion" — complicating the "just greedy degeneracy" read.
- **EP combine, round 2 — re-opened, then conclusively ruled out.** The existing `[DEBUG_PATCH]` prints (in `patched_apply_quant_method`) showed 8 different `fused_expert_output` values across ranks after `MoERunner._apply_quant_method` returns — looked exactly like a broken all-reduce. Turned out to be a misread of vLLM's call structure: that hook captures the *pre-reduce* partial (a different method, `_apply_quant_method`, returns before the reduce happens); the real reduce lives in `MoERunner.forward()` → `_maybe_reduce_final_output` → `tensor_model_parallel_all_reduce`. Patched `_maybe_reduce_final_output` directly (temporary `DEBUG_AR` prints in `deepseek_v4_patch.py`, before/after the reduce) and confirmed it fires and produces byte-identical post-reduce values across all 8 ranks, every layer, every step. **EP combine is not the bug.**
- **Prefill-vs-decode kernel consistency test (decisive, new).** `scratch/verify_prefill_vs_decode.py`: run A does 15-prompt + 2 real decode steps; run B does a *fresh single prefill* of the 16-token sequence (prompt + A's real first token) in the same process. If the decode kernel path (`bq_sz=1`) differed numerically from the prefill path (`bq_sz=15/16`) for equivalent context, these would diverge. They matched: both picked token 2572 (`" understand"`) as top choice with consistent, overlapping top-5 logprobs. **Rules out a decode-kernel-specific numerical bug.**
- **Stale FP4 weight cache — investigated and refuted, cost ~30 min.** `/dev/shm/dsv4-flash-real-fp4`'s files predate commit `4d6b9450`, which changed `mxfp4.py`'s per-rank weight-loader offset (`shard_size * self.tp_rank` → `shard_size * 0`). This looked like a serious bug: under the old code, `tp_rank > 0` would produce a negative/zero `narrow_size` and **skip the weight copy entirely** for 7 of 8 ranks — plausibly matching the rank-divergent outputs seen in the EP-combine false lead above. Tested by regenerating the cache into `/dev/shm/dsv4-flash-real-fp4-v2` with current code (run fully detached via `setsid nohup ... & disown`, survives client/session disconnects, ~30 min for all 344 files). **Result: `cmp` confirms the new cache is byte-identical to the old one; generated output is byte-identical too.** Why: `self.tp_rank` in that loader is `self.moe_parallel_config.tp_rank`, which `deepseek_v4_patch.py`'s own `FusedMoEParallelConfig.make` patch already hardcodes to `0` for every rank under EP — the "buggy" old code was already a no-op. **Lesson: verify what a suspect variable resolves to at runtime before paying for an hour-long regeneration to test a hypothesis about it.** The `-v2` cache directory is a confirmed-redundant 162GB copy in `/dev/shm` — safe to delete, not yet cleaned up.
- **Not tested this session**: the non-CSA `max_valid_kv` clamp (§2.6) — moot for this specific test prompt/length, since HCA's `compress_ratio=128` boundary is never reached within 64 total positions (15 prompt + 49 decode) — and eager-mode retracing artifacts (`VLLM_ENFORCE_EAGER=1`).
- **Still no reference/golden output** for this exact prompt+checkpoint+quantization combination, so "still garbled" cannot yet be distinguished from "this is what this checkpoint produces here" with full confidence — though the checkpoint's scale (43 layers, 4096 hidden, 256 routed experts — not a toy/distilled model) makes "inherent quality ceiling" a less likely explanation for text this incoherent (unrelated Chinese web-spam-like text, code fragments) than a real bug.

**Session 4 continued — see rationale doc §5.2 for the full trace of each item below:**

- **`enable_dp_attention` confirmed dead/unreachable**: `additional_config['sharding']` is now hard-blocked by an assertion in `platforms/tpu_platform.py` (added by a self-authored commit, `60da50f8`, during this same extended effort — not upstream/settled). Passing it crashes immediately rather than being silently ignored.
- **`deepseek_v4_patch.py`'s custom EP-rank-remapping (`patched_make`) is also dead code, traced through in full and found harmless**: its branch never fires (zero `[DEBUG_MAKE]` prints all session), but stock vLLM's real `FusedMoEParallelConfig.make()` — given the still-active, unconditional `get_tensor_model_parallel_world_size`/`rank` overrides `deepseek_v4_patch.py` also installs — independently derives the identical `ep_size=8, ep_rank=rank` the dead branch was written to produce. Confirmed by reading vLLM's actual source, not assumed.
- **Finding (superseded, see session 5 below)**: `patched_decoder_load_weights` — the function carrying every TPU-specific weight-loading fix this branch has (`attn_sink`, hash-MoE `hash_indices_table`, `e_score_correction_bias`, `wq_b`/`wo_a`/`wo_b`, `fused_wqa_wkv` stacked-param remapping) — **has never executed on any run this entire session**. It's registered on `DeepseekV4DecoderLayer`, but `AutoWeightsLoader` never reaches that class — confirmed via a debug probe that it calls native `DeepseekV4Model.load_weights` (a different class) instead, which swallows the whole weight stream itself. Re-targeting the registration to `DeepseekV4Model` does make it run, but crashes (`KeyError: 'attn.wo_a.weight'`) because the function's body uses hardcoded, layer-unprefixed keys only valid at per-layer scope. **Reverted to the original (dead, non-crashing) registration.** At the time, whether hash-MoE/`e_score_correction_bias`/TPU-fused-weight formats load correctly through native's path was unverified and flagged as the most promising open lead — **session 5 closed this, see below: they do load correctly, this was a dead end.**

**Session 5 — `patched_decoder_load_weights` dead-code lead closed (see rationale doc §3.10 for the full trace).** Compared native `vllm/models/deepseek_v4/amd/model.py`'s `DeepseekV4Model.load_weights` (the class actually invoked) against `patched_decoder_load_weights` directly — they're near-identical, including byte-identical `attn_sink` narrow-loading logic and the same `stacked_params_mapping` entries for `fused_wqa_wkv`/`compressor.fused_wkv_wgate`. This makes sense: `VllmDeepseekV4MLAAttention` subclasses AMD's native attention module rather than defining its own, so `fused_wqa_wkv`/`wq_b`/`wo_a`/`wo_b`/`attn_sink` are all native vendor objects vLLM's own loader was already written for — not a TPU-invented format needing custom handling. `hash_indices_table` wiring and `e_score_correction_bias` were never actually in `patched_decoder_load_weights`'s real responsibility either: the former is wired directly by a separate, confirmed-running function (`patched_load_weights` on `DeepseekV4ForCausalLM`), the latter falls through to a plain default `weight_loader` in both the dead function and native, identically. Verified empirically too, not just by code-reading: added temporary post-load stat prints (mean/std/nan) for `wq_b`/`wo_a`/`wo_b`/`attn_sink`/`fused_wqa_wkv`/`compressor.fused_wkv_wgate`/`e_score_correction_bias`/`hash_indices_table`, ran the full 43-layer checkpoint launch detached, killed it right after the probe printed (no need for full inference). All 8 ranks: every one of these loads with finite, sane, non-NaN, per-rank-varying values — `attn_sink`'s `mean=-inf` is the *correct*, by-design padding value (only first `n_local_heads` of its 64 padded slots are real). **`patched_decoder_load_weights` is confirmed genuinely dead/redundant, not a missing fix — not the cause of the garbled-text bug.**

**Session 5 continued — reference implementation confirmed working, three regressions found and fixed (see rationale doc §3.11 for the full trace).** The user confirmed directly that `~/workspace/tpu-inference` produces coherent output on this exact checkpoint — this is a real bug, not a checkpoint/quantization quality ceiling; that theory is now closed. `tpu-inference` has a complete, directly-diffable DSV4 stack (same file names under `tpu_inference/kernels/experimental/deepseek_v4/` and `tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/`) — diffing against it directly found three real regressions, each verified via git archaeology (traced to a specific commit whose stated rationale doesn't cover the actual change):
1. `custom_ops/deepseek_v4.py` `forward_mqa`: HCA layers' `main_kv_lens` wasn't divided by `compress_ratio` (commit `64855fed`'s stated SWA-only rationale didn't apply to this already-`not swa_only`-gated code). Fixed: unconditional division, matching the reference.
2. `custom_ops/deepseek_v4.py` `_deepseek_v4_attention_jax`: `two_caches_same_buffer` substitution was gated `and not is_csa`, added by commit `3cd0d62f` in the same change that flipped `two_caches_same_buffer`'s definition — net effect, an inversion of which layer type got the substitution. Fixed: removed the gate, matching the reference (unconditional).
3. `kernels/deepseek_v4/streamindex_topk.py` `_fetch_bkv`: missing page-index bounds clamp present in the reference, absent in vllm_torchtpu — an unclamped `bkv_idx` overrun could read `page_indices_ref` out of range and DMA from a garbage physical page into CSA's top-k token selection. Fixed: added the same clamp.

All three verified against the standard 12-test kernel suite (still passes) and a newly-discovered 15-test suite, `tests/kernels/test_dsv4_streamindex_topk.py`, that exists but isn't part of the standard 4-file suite this doc documents in §2 — **now added there** (see session 6 below).

**Still garbled after all three fixes** — byte-identical `Generated text` output before and after, for the standard test prompt. Either these three bugs aren't exercised by this specific short/greedy/single-request scenario, or at least one more bug remains. All three fixes are kept regardless (independently justified, reference-verified, no regressions). Not yet audited against the reference: the indexer's `fused_indexer_q_rope_quant` quantization path, and the MoE/FusedMoE routing/dispatch logic beyond weight-loading stats.

**Session 6 — MXFP4-quantized + hash-routing MoE test coverage gap closed (see rationale doc §3.13 for the full trace).** Session 5's own recommended next step (§3.12) was to write a quantized-weight, hash-routing correctness test for `VllmMxfp4MoEMethod`, since `test_dsv4_moe_routing.py` only exercises unquantized bf16 weights and uniform-random routing. Added `tests/layers/vllm/quantization/test_mxfp4_dsv4_correctness.py` (2 tests, added to the standard suite in §2): real checkpoint-format MXFP4-quantized weights (block-32 e2m1 + e8m0 scale, ported from `tpu-inference`'s own trusted `quantize_to_mxfp4` helper) run through the *actual* `process_weights_after_loading` dequantize-then-requantize-then-pack path (not `test_mxfp4_tpu.py`'s cache-hit bypass, which skips that logic entirely), for both grouped-topk and DSV4's hash-based routing. Both tests pass on real TPU hardware, verified against a plain-torch reference. All 30 tests (12 original + 15 streamindex_topk + 2 new + 1 existing mxfp4_tpu) pass together — no regressions.

**Investigation status: this closes the coverage gap, but not the bug.** As with session 5's four fixes, this was a real, previously-unverified gap (confirmed no numerical-correctness test exercised the real quantized/hash-routing path at all), but the MoE dequant/GMM/routing path it tests is now confirmed correct against a reference — so it is **ruled out**, not implicated, as the garbled-text root cause. Combined with session 5's `mla_swa.py` 32/50-step stress test and the four already-fixed regressions (none of which changed the full-model output), the list of verified-correct components is now large: MLA/SWA/compressor kernels, cache indexing, indexer scale, streamindex_topk clamp, and now MoE dequant+routing+GMM. **The root cause remains unfound.** Recommended next step is still §3.12's item (3): compare per-layer hidden-state statistics against `tpu-inference` running the identical prompt+checkpoint, to localize which layer's output first diverges — a systematic bisection is likely to be more productive at this point than further component-level unit tests, given how much of the component surface area is now verified clean.

**Session 7 — wiring/dtype audit (see rationale doc §3.14 for the full trace). One real fix, several hypotheses ruled out via direct tracing, and a major environment-fragility discovery.**

Prompted by a different angle than sessions 5-6's numeric auditing: is something silently misconfigured, defaulting to the wrong class, or a whole step being skipped? Checked and ruled out (each via direct tracing/empirical verification, not assumption): `custom_routing_function` bypass (always `None` for DSV4), the `-1` expert-ID sentinel round-trip (self-consistent, safe), a stale code comment about `RoutedExperts` vs `FusedMoE` object identity (traced the real vLLM call chain — the code is correct despite the comment being wrong), the `e8m0_to_fp32`/`ldexp` TPU-overflow FIXME (checked real checkpoint scale bytes directly — nowhere near the danger range), and the missing `--attention-backend` flag (DSV4 hardcodes its own backend regardless). Also found a second existing-but-broken, never-run test (`tests/layers/vllm/test_dsv4_moe_weight_loading.py`, stale interleaved-`w13` assumption) — not yet fixed, not currently a live coverage gap since session 6's new test independently covers the same ground.

**Real fix**: `VllmDeepseekV4MLAAttention`'s base-class MRO conflict (`AttentionLayerBase` listed before `ds_v4_amd_model.DeepseekV4Attention` fails to resolve on vLLM builds where the latter is `DeepseekV4ROCMAiterMLAAttention`, which already inherits `AttentionLayerBase` itself) — reordered the bases, verified safe (`AttentionLayerBase` is a pure ABC, zero concrete methods).

**Major discovery, not fixed, decided with the user to defer**: the Docker image this whole investigation runs against (`vllm-torchtpu-local:latest`) disappeared mid-session, prompting a native `uv` venv setup (§1.1). That surfaced a second, larger incompatibility: on the actual private-registry `vllm==0.23.0` build (confirmed genuinely different from the older commit baked into the missing Docker image), `DeepseekV4Attention.__init__` no longer builds the wrapper-composition chain (`outer.mla_attn = DeepseekV4MultiHeadLatentAttentionWrapper(...)`) that architecture-doc §1.1-§1.2 describes and that `VllmDeepseekV4MLAAttention.__init__`'s `mla_attn`-collapsing logic depends on — it's been flattened upstream, with `self.swa_cache_layer`/`self.compressor` etc. set directly on the outer object. This is a hard crash (`AttributeError: ... no attribute 'mla_attn'`), not a subtle bug, confirming `vllm==0.23.0` is not a reproducible pin: the exact same version string resolves to meaningfully different, incompatible code depending on when/how it's installed. **Decision: this is out of scope for the garbled-text investigation** (fixing it would mean investigating a vLLM build nobody has ever validated against, making findings incomparable to six sessions of prior work) — restore Docker access instead (see §1) and keep the native venv scoped to kernel-level test iteration, where it's already proven equivalent.

**Not yet done**: the actual "does the garbled text still happen" check in a restored Docker environment, incorporating this session's MRO fix and the session-5/6 fixes it was committed alongside. This is the concrete next step once Docker access is back.

**Session 8 — Docker restored; baseline confirmed unchanged (byte-for-byte, not just qualitatively).** `gcloud auth configure-docker` + registry pull succeeded on the first try (the underlying registry image was never actually gone — only the local `vllm-torchtpu-local:latest` tag had vanished). However, a plain pull+retag left `tpu_inference` present (`CONFLICT`, not `clean`) — see the correction above §1's "if this image is missing" block: the fix is `docker build -f docker/Dockerfile.local`, not `docker tag`. After the correct build (cache-hit, instant, 21.1GB, `clean`), ran the full 43-layer real-checkpoint prompt end-to-end: **no crash, no NaN dump, clean shutdown** — same as session 3 onward.

Full literal output this session: `'The correct correctpackage main\n\npackage main\n\nimport#https://githubgithubgithubgithubgithub#1\n[ 0.000 0.000000000000000000000000000000000000000000000000000000000000'` (49 tokens, greedy). **This matches, character-for-character as far as the truncated record goes, the literal `Generated text` string logged in §3.11-3.12 of the rationale doc from session 5** (after the HCA-kv_lens/two_caches_same_buffer/streamindex-clamp fixes) — i.e. the digit-repetition-loop failure mode has now been confirmed stable across session 5 → session 7's 6 commits → the Docker rebuild in this session. **Caution for future sessions comparing against "the baseline": the literal output text changed once already, between session 3/4 (`'the the the the the Irregularity = ...'`, documented in rationale doc §3.9-§3.10) and session 5 (the digit-loop text above) — those are two different eras of the investigation, not two runs of the same code. Always diff against the specific session's recorded literal text, not a qualitative description, and record the full untruncated `Generated text` string (not `...`-truncated) when documenting a run so this comparison stays unambiguous.** Confirms last session's 6 commits (4 regression fixes + MRO fix + MXFP4 test + docs) did not move the needle, consistent with sessions 5-7's own conclusion that their fixes were real but not root-cause.

**Session 8 continued — long-prompt/long-decode test, crossing the HCA `compress_ratio=128` boundary for the first time this entire investigation.** With `max_model_len` unstuck (see the `DSV4_MAX_MODEL_LEN` note above), reran with an 89-token prompt + `DSV4_MAX_TOKENS=250` (`max_model_len=512`) — 339 total positions, well past the 128-position mark where HCA's first compressed-cache window becomes non-empty (`extra=(pos+1)//128` goes from `0` to `1`+). Run fully detached (`setsid nohup ... & disown`, confirmed independent process-group/session via `ps -o pid,ppid,sid`) so it survived a deliberate client/laptop disconnect — same durability pattern as session 4's FP4 cache regen.

**Result: all 250/250 tokens generated, zero NaN, clean shutdown.** Output is garbled throughout (oscillates between a `.`-repetition loop and `package`/`Irregular verbs`/`regular` fragments — qualitatively different specific tokens than the 64-cap runs' `000`-loop, as expected given a different prompt and much longer generation), but **no distinct new corruption signature appears at or after position ~128** — no NaN onset, no qualitative shift in the failure pattern coinciding with the compressed-cache window activating. This is a real (if negative) data point: if the HCA/CSA compression machinery itself were the root cause, crossing its first-activation boundary would be a natural place to see *something* change; it didn't. Weakly suggests the root cause corrupts hidden states early (well before position ~128) such that everything downstream — compressed-path or not — is already operating on already-garbled state, rather than the compression path being where the corruption is introduced. Does not rule the compression path out entirely (a bug could still be there and simply invisible against an already-garbled residual stream), but lowers its priority relative to earlier-layer/earlier-step causes.

**Next step (unchanged from session 6/7's recommendation, still not started): per-layer hidden-state comparison against `tpu-inference`** on the identical prompt+checkpoint, using cosine similarity / relative-error metrics per layer (not raw tensor equality — PyTorch vs. JAX numerics will never match bit-for-bit), looking for a discontinuous jump in the error-vs-layer curve to localize the first-diverging layer. This is now the most promising unexplored angle: MLA/SWA/compressor/indexer kernels, MoE dequant+GMM+routing, cache indexing, and multiple wiring/dtype hypotheses are all verified correct or ruled out; a systematic bisection by layer is likely more productive than further component-level unit tests. Given session 8's long-prompt result, prioritize early layers/early decode steps in that bisection over the compression-boundary-adjacent layers.

**Session 8 continued — checkpoint quantization structure verified directly, and a new candidate lead identified: the MoE FP4 dequant+repack pipeline itself.** Prompted by a manager question about FP8→FP4 support, inspected the real checkpoint's safetensors byte-for-byte (not just code behavior):
- **Attention/dense weights are genuinely FP8** (`F8_E4M3`, unpacked, e.g. `layers.0.attn.wkv.weight` shape `[512,4096]`), matching `config.json`'s top-level `quantization_config` (`quant_method: fp8, fmt: e4m3, weight_block_size: [128,128]`).
- **Routed MoE expert weights are genuinely native packed FP4** (`I8` dtype with a halved contracting-dimension shape — e.g. `layers.0.ffn.experts.0.w1.weight` is `[2048,2048]` where unpacked would be `[2048,4096]`, consistent with 2 e2m1 nibbles packed per byte — plus an `F8_E8M0` block scale). **This is not declared anywhere in `config.json`'s `quantization_config` block**, which only describes the FP8 portion — a real trap for anyone (including this investigation) reading just the declared config.
- Confirmed the checkpoint at `$MODEL_PATH` is genuinely unmodified: standard `huggingface_hub` blob-cache layout, all blobs written in one 22-minute window, no later-timestamped writes. `scratch/prequantize_4layer_to_fp4.py` (the "offline quantization" this session's user recalled) is read-only against the checkpoint — it reads native-FP4 shards from `model_path` and writes only to a separate `cache_dir` (the `VLLM_FP4_CACHE_DIR` cache), never back into the checkpoint. What it does is dequant+repack already-native-FP4 weights into the TPU-optimized layout — not an FP8→FP4 conversion.
- **Confirmed gap**: vllm-torchtpu has no working FP8→FP4 requantization path (`_MOE_REQUANT_WEIGHT_DTYPES` in `layers/vllm/quantization/fp8.py` only supports `float8_e4m3fn`/`float8_e5m2`/`int8`, no fp4 entry), whereas tpu-inference genuinely does this for DeepSeek-R1/V3 (`MOE_REQUANTIZE_WEIGHT_DTYPE=fp4` + `MOE_REQUANTIZE_BLOCK_SIZE=512`, exercised in `.buildkite/benchmark/cases/daily/deepseek.json`). Not directly relevant to *this* checkpoint (whose experts are already FP4), but confirms the env var plumbing was copied from tpu-inference without the FP4 branch being finished.
- **New candidate lead for the garbled-text root cause, not yet investigated**: the MoE dequant+repack pipeline (`VllmMxfp4MoEMethod.process_weights_after_loading` in `mxfp4.py`) is a *full numerical round-trip* over every routed expert in ~41 of 43 layers — dequantize native packed-FP4 (real e8m0 scale, real block layout) → BF16 → requantize to FP4 (block-512) or FP8 (block-32, v6e only) → pack. Git history shows this exact pipeline has had many real bugs fixed over time (scale layout for w13/w2, Case B expert layout transpose, axis alignment, cross-model cache collisions). Session 6's `test_mxfp4_dsv4_correctness.py` already verifies this path against a plain-torch reference — but worth double-checking whether that test's synthetic fixture (block-32, small expert count) faithfully reproduces the *real* checkpoint's exact scale shape/block arrangement (e.g. the real `w1.scale` shape `[2048,128]` against packed weight `[2048,2048]` — confirm the effective block size this implies matches what the test exercises), since a subtle mismatch between test fixture and real checkpoint geometry could hide a bug that only manifests on the real weights.

---

## Session 9 — all-zero-attention root cause found and fixed; a new, narrower numerical bug isolated. Full technical detail in `docs/bisection_progress_report.md` §2 "Iteration 4" — this entry is the condensed pointer + exact repro.

**Started from**: sessions 5-8's own recommended next step, finally executed — a real, working sub-layer hidden-state bisection harness against `tpu-inference`, comparing 6 checkpoints inside `_forward_unfused_post_pre` (hc_pre_attn, post_attn, hc_post_attn, hc_pre_ffn, post_ffn, layer-out) for Layer 0, prefill step, all 15 real prompt tokens.

**Root cause of the garbled-text bug found and fixed (not yet fully — see below):**
`_get_active_sw_cache`/`_get_active_main_cache` in `layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py` gated every candidate cache tensor on `cache.dtype == torch.uint8`. The real cache's actual dtype is whatever the checkpoint's `kv_cache_dtype` resolves to (`torch.float8_e4m3fn` here) — never `torch.uint8` — so this check *never* matched, both functions *always* returned `None`, on every layer, every step, since this code was introduced in commit `2ac94b211` (session-4-era refactor). `forward_mqa` then silently fell back to a zero-page dummy cache, and `_deepseek_v4_attention_jax`'s early-return shape check skipped the real kernel entirely, returning all-zero attention output — for every single layer, every single forward pass, this whole investigation. This directly explains sessions 5-8's "garbled but structurally normal-looking" output and why so many verified-correct components (MoE, streamindex_topk, weight loading) never moved the needle: attention itself was contributing nothing.

Fixing just the dtype gate was not enough to get a working run — it unmasked four more real, previously-dormant bugs in the SWA-only (`compress_ratio<=1`) real kernel call path, which had *never executed with real data* before (every earlier session's "attention outputs zero" symptom meant this whole code path was untested territory, not verified-safe):
1. A `numel()==0` placeholder cache during `profile_run` (before `initialize_kv_cache()` runs) got treated as real once the dtype gate stopped filtering it, and XLA silently dropped it as a compiled operand (empty arrays need no data read) — Mosaic `number of refinements ... 10 vs 9`.
2. `forward_mqa` hardcoded 4 kernel args to Python `None` for `swa_only` layers; `None` isn't a JAX pytree leaf and never becomes a real operand — same symptom, `14 vs 13` after fix #1.
3. `main_cache_kv = sw_cache` (plain alias) and the fix-#2 substitutions (also plain aliases) each made two formal kernel params resolve to one physical buffer; XLA's input dedup merged them — same symptom again.
4. Even with a real, distinct, correctly-shaped cache, `mla_swa.py`'s kernel hard-asserts `cache_kv.dtype == jnp.uint8` (confirmed identical, and therefore not a bug, in `tpu-inference`'s reference kernel) — but vLLM's own KV-cache-spec/allocation pipeline does not preserve this class's `dtype=torch.uint8` spec declaration through to the actually-allocated tensor (verified: `torch.uint8` at construction via a temporary print, `torch.float8_e4m3fn` at allocation via the existing `[KV_SIZE_DEBUG]` log — the exact vLLM-core mutation site was not found; `unify_hybrid_kv_cache_specs`, `_get_kv_cache_config_packed`, and `copy_with_new_block_size` were all read and ruled out). Fixed by bitcasting instead: `cache.view(torch.uint8)` right before returning a real cache (both dtypes are 1 byte/element, so this is free and lossless — same trick as `mxfp4.py`'s `loaded_weight.view(torch.uint8)`).

All five fixes are in `_get_active_sw_cache`/`_get_active_main_cache`/`forward_mqa` in `deepseek_v4_attention.py` (helper: new `_as_kernel_cache_view` static method). **Result: the pipeline now runs end-to-end with zero crashes/exceptions for the first time in this entire investigation** — full prefill + decode to `max_tokens`, real (non-dummy) cache flowing through the SWA-only path throughout.

**But the output is still garbled** — confirmed a *different* bug from all nine sessions' prior findings, via the same sub-layer bisection, rerun with matching prompt/tokenization against a fresh `tpu-inference` reference:

```
sub_01_hc_pre_attn   mean_cos=1.000000  OK    (attention's *input* — matches exactly)
sub_02_post_attn     mean_cos=0.133476  DIVERGE  (attention's *output* — no longer zero, but numerically wrong)
```

Generated text (matching prompt `"Compare XLA and LLVM in two bullet points."`, tokenization verified identical): `'I understand the nature of thepackage com.ibpackage, "spackage",\n  "package.json\n<package> <package 1package",\npackage main\n\nimport\n- * "Ir)package":package com.azurep\n>'` — incoherent, but a different failure signature than any of sessions 4-8's recorded outputs (not the `000`-loop, not the `.`-loop, not `Irästyle` repetition) — consistent with this being new territory, not a recurrence of an old bug.

**Important scope note for whoever picks this up**: `sub_02_post_attn` is dumped at the *entire attention module's* output (`x = self.attn(positions, x, None)` in `_forward_unfused_post_pre`) — this brackets `attn_gemm` (QKV projection) → `qnorm_rope`/`kv_rope` → `forward_mqa`/kernel call → `_o_proj` (dequant + grouped einsum + `wo_b`) *all together* as one black box. The divergence could be introduced anywhere in that range, not necessarily in the cache/kernel plumbing this session's fixes touched. **Recommended next step**: add 1-2 finer-grained dump points *inside* `attention_impl`/`forward_mqa` — e.g. right after `q = self.qnorm_rope(...)` / `kv = self.kv_rope(...)` (before the kernel call) and right after `self.attn_op(...)` returns (before `_o_proj`) — mirrored in `tpu-inference`'s equivalent (`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py`, same function names) to bisect within this range: projections/RoPE vs. the kernel/cache read-write cycle vs. `_o_proj`.

**Also not yet done**: exercising the HCA/CSA (`compress_ratio>1`) real kernel path the same way — this session's fixes were verified specifically against Layer 0 (`compress_ratio=1`, `swa_only`). Fixes #2/#3 (the `swa_only`-branch `None`/alias substitutions) are gated to the `swa_only` branch specifically and shouldn't affect HCA/CSA layers, but fix #4 (`_as_kernel_cache_view`) applies to *every* layer via `_get_active_main_cache`, and hasn't been separately confirmed correct for a real HCA/CSA layer's `mla_ragged_paged_attention` call — worth a quick check if the sub_02 bisection above turns out clean for other reasons.

**Environment note**: this session used the native `uv` venv (`~/uv_venv`), *not* Docker — contrary to §1's standing "use Docker for full-model runs" policy, which was written around session 7's finding that a *floating* `pip install vllm==0.23.0` can silently resolve to an incompatible, unpinned build. This session instead followed the repo's actual pinned-install instructions (`git clone --depth 1 --branch v0.25.1`, exact tag, not a floating version) and never hit session 7's wrapper-collapse crash. Worth revisiting whether §1's Docker requirement is still necessary given an exact-tag-pinned venv, but not re-litigated this session — the next session should just continue in this same venv (already set up, already produced a real fix) rather than switching to Docker mid-investigation.

### Exact repro command (copy-paste, all gotchas included)

Three real gotchas cost real time this session — all baked into the command below:
1. **`PYTHONPATH` must be set explicitly**, even though `vllm_torchtpu` is `pip install -e`'d (its `.pth` file adds `src/` to `sys.path`, but *after* the stdlib). Without it, the system's `/usr/lib/python3.12/sitecustomize.py` shadows this repo's own `src/sitecustomize.py` (which carries a needed `DeepseekV4Attention` compat alias) — causes an `AttributeError` at import time that looks unrelated to anything above.
2. **`DSV4_MODEL_PATH` must point at the local snapshot**, not the default `deepseek-ai/DeepSeek-V4-Flash` HF Hub ID — the unset default tries to *download* the full checkpoint, which filled the disk to 100% mid-session (root partition is only 145GB; the checkpoint is much larger than what's left after everything else on this box).
3. **`DSV4_PROMPT` must match** whatever `tpu-inference` reference run you're comparing against, exactly — different prompts tokenize to different lengths/IDs, silently invalidating any per-token hidden-state comparison.

```bash
cd /home/patemotter_google_com/workspace/vllm-torchtpu
source /home/patemotter_google_com/uv_venv/bin/activate
PYTHONPATH=/home/patemotter_google_com/workspace/vllm-torchtpu/src \
DSV4_MODEL_PATH=/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1 \
DSV4_PROMPT="Compare XLA and LLVM in two bullet points." \
VLLM_ENFORCE_EAGER=1 \
GPU_MEMORY_UTILIZATION=0.40 \
DSV4_MAX_NUM_SEQS=8 \
DSV4_NUM_GPU_BLOCKS=512 \
DSV4_DUMP_HIDDEN_STATES=1 \
nohup python examples/run_dsv4_inference.py > /home/patemotter_google_com/workspace/scratch/run_<name>.log 2>&1 &
```

Takes ~10-13 min (mostly MoE weight loading from the `VLLM_FP4_CACHE_DIR` cache, `/dev/shm/dsv4-flash-real-fp4` — already warm, don't delete it). Tail the log or `pgrep -f run_dsv4_inference.py` to check progress; the background process does **not** exit when the launching shell command returns (it's `nohup`'d + backgrounded) — wait for `Generated text:` or a traceback in the log, not for the shell command itself to report done.

The `tpu-inference` reference side needs its own separate venv/run — see §1.2-1.3 above for that environment; the existing reference dump at `scratch/hidden_states/tpu_inference/` (prompt `"Compare XLA and LLVM in two bullet points."`, 15 tokens, matches the command above) is still valid and doesn't need re-running unless the reference implementation itself has changed.

After a run, compare with:

```bash
python /home/patemotter_google_com/workspace/scratch/compare_sublayer.py
```

(This script now lives in `scratch/` — it previously only existed in a background-job temp dir that gets cleaned up; copied here this session specifically so it survives into future sessions.)

### Files modified this session (uncommitted at session end — review before committing)
- `src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention.py` — all 5 fixes above (`_get_active_sw_cache`, `_get_active_main_cache`, `forward_mqa`, new `_as_kernel_cache_view`).
- `src/vllm_torchtpu/runner/tpu_runner.py` — the KV-cache-sizing bug fix (see `docs/bisection_progress_report.md` §2 "Also fixed today") plus `[KV_SIZE_DEBUG]`/`[HBM_CHECKPOINT]` logging added while chasing it; unrelated to the attention fixes above but in the same working tree.
- `src/vllm_torchtpu/layers/vllm/quantization/mxfp4.py`, `src/vllm_torchtpu/layers/vllm/quantization/deepseek_v4_fp8.py` — vLLM 0.25.1 API-compat fixes (`FusedMoE` class → `RoutedExperts` module rename, `moe_parallel_config`/`moe_config.ep_size` path changes). Needed just to get any run loading at all on this pinned vLLM version; unrelated to the attention/cache bugs.
- `src/vllm_torchtpu/models/vllm/deepseek_v4_patch.py` — same 0.25.1 API-compat category (`MoERunner._apply_quant_method` signature), plus removal of a duplicate `process_weights_after_loading` call that was double-processing every MoE layer's weights and inflating worker RSS (~90GB/worker) enough to OOM the host when the FP4 weight cache was enabled.
- `examples/run_dsv4_inference.py` — `DSV4_MODEL_PATH` env var support (was hardcoded to the HF Hub ID).
- `docs/bisection_progress_report.md` — full Iteration 4 write-up (more detail than this entry).
- `scratch/compare_sublayer.py` — copied here this session (see above) for persistence.

None of this has been committed yet. Recommend splitting into at least two commits before pushing: (1) the vLLM-0.25.1-compat + double-processing fixes (mxfp4.py, deepseek_v4_fp8.py, deepseek_v4_patch.py's compat parts), which are straightforward and independently justified; (2) the KV-cache-sizing fix (tpu_runner.py) and the five attention/cache fixes (deepseek_v4_attention.py), which are the substance of this investigation and worth their own message(s) referencing this doc.

### Session 9 continued — `sub_02_post_attn` divergence localized inside the kernel call (2026-07-30, later same day)

Executed this session's own recommended next step: added `sub_02a_pre_kernel_q`/`sub_02a_pre_kernel_kv` (right after `qnorm_rope`/`kv_rope`, before `self.attn_op(...)`) and `sub_02b_post_kernel_output` (right after `self.attn_op(...)` returns, before `_o_proj`) to `attention_impl`/`forward_mqa` in `deepseek_v4_attention.py` — mirrored identically in `tpu-inference`'s equivalent file so `scratch/compare_sublayer.py` can diff them directly. Full detail, exact numbers, and the shape/head-sharding gotcha hit along the way: `docs/bisection_progress_report.md` §2 "Iteration 5" — condensed pointer here.

**Result: q/kv going into the kernel match the reference almost exactly (cos_sim 0.9994-0.9999); the kernel's own output does not (cos_sim 0.17 mean, 0.0 min).** This rules out `wq_b`/`qnorm_rope`/`kv_rope`/indexer/compressor as the bug's location — the divergence is specifically inside `self.attn_op(...)`, Layer 0's `swa_only` call into `mla_sliding_window_ragged_paged_attention` (`kernels/deepseek_v4/mla_swa.py`).

**Sharper clue, not yet chased into the kernel source**: the kernel's output isn't uniformly wrong — for this 15-token prompt (page size 4, so 4 pages: rows 0-3/4-7/8-11/12-14), tokens 0-11 (the first 3 pages) come out **exactly zero**, and only tokens 12-14 (the last page) are non-zero. Shape of bug: looks like the kernel's per-page block loop only keeps the last page's contribution instead of accumulating/masking correctly across all pages a query should attend to. **Next step for whoever picks this up**: read `mla_swa.py`'s block-loop/accumulator logic (`num_kv_pages_per_block=1`) and diff it directly against `tpu-inference`'s kernel invocation (same file structure, same kwargs) to determine whether the *kernel* itself differs or only its *inputs* (block tables, page indices, or the `.view(torch.uint8)` cache bitcast's byte layout) do.

Environment/repro: same native `uv` venv as before (§ above), same exact repro command for the vllm-torchtpu side; the tpu-inference reference side needs `source ~/workspace/tpu_inference_venv_setup/vllm_env/bin/activate` + `DSV4_DUMP_HIDDEN_STATES=1 python3 ~/workspace/scratch/run_dsv4_reference.py` (see `docs/bisection_progress_report.md` §5.2). Both runs this session completed cleanly in ~5 min each on a fully free TPU device (no stray containers) — much faster than the multi-hour overnight run in the timeline log, since this session's runs didn't hit that session's setup/import issues. `tpu-inference`'s reference `Generated text` this session: coherent English (`'Here are two key comparisons between XLA and LLVM: ...'`) — confirms the reference is still healthy and the comparison basis is valid.

Files modified this session, uncommitted: `deepseek_v4_attention.py` (both repos, the two new dump points), `scratch/compare_sublayer.py` (added the 3 new checkpoint names to `CHECKPOINTS` + the head-slicing fix for the 3D q/kv/kernel-output shape mismatch), `docs/bisection_progress_report.md` (Iteration 5 write-up).

### Session 9 continued (Iteration 7) — two more real bugs fixed; garbled text persists; full write-up + next-session handoff in the bisection doc

Same day, continuing directly from the entry above. **Condensed pointer — full detail, exact repro commands, and a ready-to-use next-session prompt are in `docs/bisection_progress_report.md` §6 "Iteration 7" and its "HANDOFF FOR NEXT SESSION" section. Read that, not just this summary, before doing anything.**

**Found and fixed, both verified real**:
1. `mla_swa.py`'s flash-attention softmax accumulator (`exp_cond`) — the bug this doc's Session 9 entry above left as "not yet found." Fixed: `exp_cond = jnp.isneginf(m_curr)`. Verified: 30/30 kernel tests pass, and the first generated token now exactly matches the reference (`'Here'`, `logprob=0.0`).
2. `deepseek_v4_attention.py`'s `main_attn_metadata` lookup was trying the compressor's dedicated state-cache prefix *before* the main layer's own prefix, silently pointing CSA/HCA layers' attention reads at the wrong KV-cache-group's block table. Traced to the exact introducing commit (`3cd0d62f`) via `git log -p -L`; harmless when introduced, wrong since a later fix gave the state cache its own disjoint buffer. Fixed by restoring `main_layer_name` as first priority (matches `tpu-inference`'s reference exactly). Verified as a real, live bug via a debug print showing genuinely different block tables were being selected.

**Still garbled after both fixes.** A third, distinct, not-yet-found bug remains in CSA/HCA layers' decode-step attention (`kernels/deepseek_v4/mla.py`) — confirmed via the dump-comparison methodology that layer 2's attention *output* blows up to ~1e27-1e29 magnitude during decode even though its input and the KV cache slot it reads (independently dequantized and checked) are both confirmed clean. Leading, unproven suspect: an unguarded-vs-guarded `exp()` discrepancy in the attention-sinks epilogue, found via the same line-diff-against-reference technique that found fix #1 above — but a `pl.debug_print`-based attempt to get value-level confirmation produced no output across 4 tries, for reasons not understood. Full numbered next-steps list is in the bisection doc.

**Also worth knowing before touching anything**: this session found and documents (bisection doc §6.2) three separate cases where its own diagnostic tooling gave a wrong answer before being corrected — a JAX tracing-closure bug, host-callback/export-serialization incompatibility, and torch/XLA buffer-donation making naive clone-based diffing unreliable. Also: the `/dev/shm/torch_tpu_cache` compile cache needs clearing (`rm -rf /dev/shm/torch_tpu_cache/default/*`) before any run that changes *kernel-body* code (not plain host-side Python), or stale compiled code silently runs instead.
