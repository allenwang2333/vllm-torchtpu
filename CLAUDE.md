# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vllm_torchtpu` is a **vLLM platform plugin** that integrates **TorchTPU** (a custom TPU-backed PyTorch build) with vLLM's engine, so vLLM can serve models on Google TPUs. It is *not* a fork of vLLM — it registers itself with vLLM's out-of-tree plugin system (`vllm.platform_plugins` / `vllm.general_plugins` entry points in `pyproject.toml`) and layers TPU-specific runners, kernels, and monkey-patches on top of an installed `vllm` package.

A sibling repo, `tpu-inference` (JAX-native, not vLLM-based), is treated as the trusted reference implementation for numerical correctness when debugging model-specific kernels (e.g. DeepSeek-V4) — diff against it directly rather than trusting this repo's own tests when investigating correctness bugs.

## Installation

Requires a private Google Artifact Registry (`torch-tpu-virtual-registry`) for `torch`, `torch-tpu`, and other pinned deps — see `README.md` for full `gcloud auth` + `uv`/`pip` setup. Key points if bootstrapping an environment:

```bash
# vLLM itself must be installed from source at the pinned tag (patched to drop
# the upstream tpu-inference plugin, which would conflict with this one):
git clone --depth 1 --branch v0.25.1 https://github.com/vllm-project/vllm.git ../vllm
sed -i '/tpu-inference/d' ../vllm/requirements/tpu.txt
SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1 VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm
uv pip install --pre -e ".[test]"
```

The exact `vllm`/`torch`/`torch-tpu`/`jax`/`libtpu` versions in `pyproject.toml` are load-bearing — several are pinned specifically because a monkey-patch in `src/vllm_torchtpu/env_override.py` depends on that exact version's behavior (read the comments next to each dependency before bumping any of them).

## Running tests

Tests run against either CPU (JAX interpret mode) or real TPU hardware, selected by the `--use-tpu` pytest flag (defined in `tests/conftest.py`).

```bash
# CPU (JAX/Pallas interpret mode) — default, no TPU hardware needed:
pytest tests/

# Real TPU hardware — required for kernel numerics to mean anything:
pytest -sv --use-tpu tests/kernels/test_dsv4_mla.py

# Single test:
pytest -sv --use-tpu tests/kernels/test_dsv4_mla_swa.py::TestName::test_method
```

**`--use-tpu` matters even when a TPU is physically available.** `src/sitecustomize.py` auto-detects a bare `pytest` invocation (no `--use-tpu`) and forces `JAX_PLATFORMS=cpu` + `PALLAS_INTERPRET=1` process-wide, *before* `vllm_torchtpu` or `jax` is imported. Forgetting the flag silently runs Pallas kernels in interpret mode instead of on hardware, which can pass/fail differently and mask real bugs.

Tests are typically run inside the project's Docker image (`docker/Dockerfile` / `docker/Dockerfile.local`), which pins a validated `libtpu`/JAX build and mounts the TPU device. `docker/Dockerfile.local` has real build steps (removing a conflicting `tpu_inference` package, copying `libtpu.so`) — restore it with `docker build`, never a plain `docker pull`+`tag`. See `docs/dsv4_running_and_testing.md` for concrete `docker run` invocations, required env vars, and gotchas (stray containers holding the TPU device, permission issues on shared caches, etc.) — treat the exact commands there as more current than anything summarized here.

CI (Buildkite, `.buildkite/pipeline_tests.yml`) runs two suites per PR: a speculative-decoding suite (`tpu_v7x_2_queue`) and a general suite excluding it (`tpu_v7x_8_queue`), both `pytest -m 'not nightly'`; nightly runs drop that marker. Mark TPU-hardware-heavy/slow tests `@pytest.mark.nightly` (registered in `pyproject.toml`) if they shouldn't run on every PR.

Linting/formatting: `pre-commit install --hook-type pre-commit --hook-type commit-msg`, then `pre-commit run --all-files` (isort, yapf, ruff, clang-format for C++/CUDA, pymarkdown, actionlint, shellcheck). Commits must be signed off (`git commit -s`) — a local hook auto-appends `Signed-off-by:` if missing.

## Architecture

### Plugin entry point and import-time patching

`register_tpu_platform()` (`src/vllm_torchtpu/platforms/__init__.py`) is vLLM's discovery hook; it returns the dotted path to `TpuPlatform` (`platforms/tpu_platform.py`), vLLM's `Platform` subclass that tells the engine how to build workers/executors/attention backends for this device.

Two modules apply **monkey-patches to vLLM internals at import time**, before other imports happen:
- `src/vllm_torchtpu/env_override.py` — imported first (from `vllm_torchtpu/__init__.py` and `platforms/tpu_platform.py`) specifically so environment variables land before `torch_tpu` reads them. Also patches vLLM's compile-cache hashing, collective-op dispatch, MoE runner paths, compile-range handling (TPU needs static shapes, not `SymInt` dynamic ranges), and worker rank/env binding for multiproc spawn. Every patch function has a docstring explaining *why* vLLM's default breaks on TPU — read it before assuming a patch is dead code.
- `src/sitecustomize.py` — process-wide, runs before any package import (including `vllm_torchtpu`). Detects bare `pytest` (see above) and patches around device-registration conflicts (`torch.utils.rename_privateuse1_backend`, `torch._register_device_module`) that fire when TorchTPU's device is registered more than once in-process.

If something behaves differently than "stock vLLM would," check both of these files before assuming it's a bug in the TPU-specific code you're editing.

### Request path

`worker/tpu_worker.py` (`TPUWorker`) and `runner/tpu_runner.py` (`TPUModelRunner`, ~4700 lines — the largest file in the repo) implement vLLM's worker/runner interfaces for TPU: KV cache spec computation and allocation, input batching/padding to fixed compile shapes, and driving the actual forward pass. `core/tpu_scheduler.py` customizes vLLM's scheduler for TPU-specific batching constraints. `executors/` provides TPU-flavored `MultiprocExecutor`/`RayDistributedExecutor` variants for multi-host/multi-chip topologies.

`compilation/tpu_compiler.py` computes an additional hash folded into vLLM's AOT compile cache key (via the `env_override.py` patch) so that TPU-compiler-relevant config also invalidates the cache correctly. `compilation/kernel_reload.py` and the `TPU_KERNEL_ITER_MODE`/`TPU_KERNEL_RELOAD_MODULES` env vars support hot-swapping a single Pallas kernel without a full graph recompile — useful when iterating on kernel code.

**TPU requires static shapes.** Unlike GPU, the runner pads every batch to one of a fixed set of `compile_sizes`; `env_override.py` patches `CompilationConfig.get_compile_ranges()` to return `[]` specifically to stop vLLM from building dynamic `SymInt`-shaped compile ranges, which the TorchTPU backend rejects outright.

### Layers, models, kernels split

Mirrors vLLM's own layout (from `CONTRIBUTING.md`), split three ways under both `layers/` and `models/`:
- `common/` — shared between vLLM-served and JAX-native model paths.
- `vllm/` — vLLM-specific (e.g. `layers/vllm/custom_ops/`, `layers/vllm/quantization/`, `models/vllm/*_patch.py` per-model monkey-patches like `deepseek_v4_patch.py`).
- `jax/` — JAX-native model paths (where present).

`kernels/` holds the actual Pallas/TPU kernel implementations (flash attention, MLA, ragged paged attention, fused MoE, quantized matmul, GDN, sparse core collectives, `deepseek_v4/` for DSV4-specific fused kernels). These are the lowest-level, hardware-facing code — bugs here often show up only on real TPU hardware, not in CPU interpret mode, and are usually the last thing to suspect after ruling out wiring/config issues higher up the stack.

`patches/pallas_patched.py` holds patches to upstream Pallas/Mosaic itself (as opposed to vLLM), for TPU-specific lowering workarounds.

### Quantization

FP4/FP8/MXFP4 quantized-weight support lives under `layers/vllm/quantization/`. Weight loading paths dequantize-then-requantize-then-repack into TPU-optimized block layouts (see `mxfp4.py`); this round-trip has historically been a real source of bugs (scale layout, expert-axis transposes, cross-model cache collisions) — don't assume a synthetic test fixture's block size/shape matches a real checkpoint's without checking.

### DeepSeek-V4 (active work area)

DSV4 is the most actively-developed and most fragile model integration in this repo right now (hybrid attention with compressed-KV variants — HCA/CSA — sliding-window attention, and hash-routed MoE). Its code spans `kernels/deepseek_v4/`, `layers/vllm/custom_ops/deepseek_v4.py`, `models/vllm/deepseek_v4_patch.py`. `docs/dsv4_architecture_and_rationale.md` (the *why*) and `docs/dsv4_running_and_testing.md` (the *how* — exact docker/env-var invocations, current investigation state) are living session logs for an ongoing correctness investigation; read the latest entries there before re-deriving something already ruled in/out. Treat `tpu-inference`'s DSV4 implementation as ground truth when these two disagree.

## Non-obvious conventions

- Never trust an old log's numbers when re-verifying a fix — re-run fresh and read the actual current formula/values in code, don't reverse-engineer conclusions from stale output.
- `docs/` accumulates dated investigation/session logs (`dsv4_*`, `debug_findings_*`) rather than being pure reference documentation — check dates and "confirmed/refuted/superseded" language before relying on an older entry.
- `scratch/` is a working scratch space (debug scripts, one-off logs, dumped tensors) — expect it to contain stale/abandoned artifacts from prior sessions; don't assume everything there is current or wired up.
