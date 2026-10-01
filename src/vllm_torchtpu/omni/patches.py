# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni diffusion models."""

import functools
from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _unwrap_tpu_patch(fn: Any) -> Any:
    """Return the underlying unpatched callable if ``fn`` was wrapped by TPU patches."""
    if getattr(fn, "_is_tpu_patch_wrapper", False) is True:
        return fn._orig_tpu_fn
    return fn


def _mark_tpu_patch(wrapper: Any, orig: Any) -> Any:
    """Mark ``wrapper`` with a reference to ``orig`` so re-patching is idempotent.

    ``functools.wraps`` keeps ``__name__``/``__qualname__``/``inspect.signature``
    of the original for callers that introspect patched methods. It is skipped
    when ``orig`` has no string ``__name__`` (e.g. the ``MagicMock`` originals the
    hermetic unit tests install), where ``wraps`` would raise.
    """
    if isinstance(getattr(orig, "__name__", None), str):
        functools.wraps(orig)(wrapper)
    wrapper._is_tpu_patch_wrapper = True
    wrapper._orig_tpu_fn = orig
    return wrapper


@run_once
def apply_omni_tpu_patches() -> None:
    """Extend async output timeout and patch TPU unsupported ops."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Lazy execution attributes failure to sync point: In PyTorch/XLA
    #    and torch_tpu, forward passes are queued lazily in the computation
    #    graph. diffusion_worker emits COMPUTE_DONE prematurely before the TPU
    #    hardware finishes execution. When the async output background loop
    #    triggers host synchronization (.cpu()), it blocks on multi-step diffusion
    #    graph execution and initial compilation.
    # 2. Recompilation storms & JIT compile latency: Initial JIT/AOT
    #    compilation and weight distribution for large diffusion models (e.g.
    #    Wan 14B) take >600s, exceeding vLLM-Omni's default 600s timeout.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed/reduced once:
    # 1. OmniTpuPlatform.record_device_event() implements hardware synchronization
    #    (torch.tpu.synchronize()) to ensure execution completes before COMPUTE_DONE.
    # 2. AOT compilation caching stabilizes startup compile times.
    # -------------------------------------------------------------------------
    import os

    os.environ.setdefault("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", "1800.0")
    logger.info("Applied TPU patch: extended VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT to 1800s.")

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Missing aten::upsample_linear1d.out operator in torch_tpu.
    # Calling torch.nn.functional.interpolate(..., mode="linear") on 3D tensors
    # fails with:
    #   RuntimeError: operator 'aten::upsample_linear1d.out' is not implemented for TPU
    # Used in temporal/audio diffusion alignment (Wan 2.2 S2V, LongCat, audio vocoders).
    #
    # UPSTREAM STATUS & REMOVAL CONDITION:
    # Fixed upstream in torch_tpu commit a0b184ed27d
    # ("Implement aten::upsample_linear1d.out", Sep 14, 2026). This CPU fallback patch
    # can be deleted once vllm-torchtpu bumps to a torch_tpu development wheel
    # >= 0.1.1.dev20260914 containing that commit.
    # -------------------------------------------------------------------------
    orig_interpolate = _unwrap_tpu_patch(torch.nn.functional.interpolate)

    def _safe_interpolate(
        input: torch.Tensor,
        size: Any = None,
        scale_factor: Any = None,
        mode: str = "nearest",
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input.device.type == "tpu" and mode == "linear":
            return orig_interpolate(
                input.cpu(), size, scale_factor, mode, *args, **kwargs
            ).to(input.device)
        return orig_interpolate(input, size, scale_factor, mode, *args, **kwargs)

    torch.nn.functional.interpolate = _mark_tpu_patch(
        _safe_interpolate, orig_interpolate
    )
    logger.info(
        "Applied TPU patch: safe torch.nn.functional.interpolate for 1D linear TPU ops."
    )


@run_once
def apply_omni_model_specific_patches() -> None:
    """Apply model-specific runtime patches."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Pickling a device tensor aborts process: In torch_tpu, calling
    #    pickle.dumps on a device tensor hits
    #    ABSL_CHECK(tensor.storage().allocator() == nullptr) in
    #    csrc/eager/tensor_to_buffer.cc:260-261, terminating the worker process
    #    with SIGABRT (exit code 134) with no Python traceback.
    # 2. libtpu per-process lockfile prevents cross-process init:
    #    Holding TPU tensors across process boundaries causes the CPU-only
    #    orchestrator process to trigger PJRT backend initialization during
    #    unpickling, colliding with the worker on /tmp/libtpu_lockfile.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed once torch_tpu supports safe inter-process tensor
    # serialization / proxy unpickling without triggering PJRT initialization
    # in host processes that do not own TPU hardware.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion import ipc

    orig_pack_tensor = _unwrap_tpu_patch(ipc._pack_tensor_if_large)

    def _safe_pack_tensor_if_large(
        val: torch.Tensor,
        d2h_stream: Any = None,
    ) -> Any:
        return orig_pack_tensor(val.cpu(), d2h_stream=d2h_stream)

    ipc._pack_tensor_if_large = _mark_tpu_patch(
        _safe_pack_tensor_if_large, orig_pack_tensor
    )
    logger.info("Applied TPU patch: safe CPU serialization for diffusion IPC tensors.")

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # In LTX-2, LTXRuntime._decode_output produces a (video, audio) tuple. Audio
    # tensors are sub-threshold (<1MB) and returned directly in the output object,
    # bypassing SHM packing. If audio remains on TPU, unpickling in the orchestrator
    # triggers PJRT re-initialization and /tmp/libtpu_lockfile collision, while
    # pickling in the worker risks tensor_to_buffer.cc ABSL_CHECK aborts.
    # Moving video and audio to CPU prevents device tensors entering the IPC stream.
    #
    # REMOVAL / FIX CONDITION:
    # Can be removed once torch_tpu provides safe cross-process device tensor
    # serialization, or once vllm_omni universally ensures all diffusion output
    # payloads are moved to host CPU before IPC serialization.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    ltx_cls = ltx2_runtime.LTXRuntime
    orig_decode_output = _unwrap_tpu_patch(ltx_cls._decode_output)

    def _safe_decode_output(self: Any, *args: Any, **kwargs: Any):
        res = orig_decode_output(self, *args, **kwargs)
        video, audio = res.output
        res.output = (
            video.cpu() if isinstance(video, torch.Tensor) else video,
            audio.cpu() if isinstance(audio, torch.Tensor) else audio,
        )
        return res

    ltx_cls._decode_output = _mark_tpu_patch(_safe_decode_output, orig_decode_output)
    logger.info("Applied TPU patch: safe LTX-2 CPU decode output.")


@run_once
def apply_omni_worker_patches() -> None:
    """Apply patches that must be live before ``WorkerProc`` starts its threads.

    Called from ``OmniTpuPlatform.set_device``. ``DiffusionWorker.__init__``
    calls ``init_device()`` (-> ``set_device``) first, before ``load_model()``
    and before ``WorkerProc.__init__`` starts the ``DiffusionAsyncOutput``
    thread, on both the default path and the ``custom_pipeline_args`` path
    (where ``skip_load_model=True`` defers ``load_model`` until after the
    thread exists). Installing from the attention-backend hook would be too
    late on the custom-pipeline path. ``vllm_omni.diffusion.worker.
    diffusion_worker`` is already imported in the worker process by then, so
    the import below adds no cost.
    """

    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni / torch_tpu Root Cause Tracking:
    # In vllm_omni.diffusion.worker.diffusion_worker, WorkerProc._async_output_loop
    # constructs a device stream using `self.gpu_id`:
    #   device = torch.device(torch.accelerator.current_accelerator().type, self.gpu_id)
    #   d2h_stream = torch.Stream(device=device)
    # Because each TPU worker process isolates a single chip via TPU_VISIBLE_CHIPS,
    # its only addressable local device index is 0. On worker ranks > 0 (where
    # self.gpu_id > 0), torch.Stream(device=torch.device("tpu", self.gpu_id))
    # crashes the background DiffusionAsyncOutput thread with:
    #   RuntimeError: Device index N is out of bounds of addressable devices 1
    # Wrapping `self` in a proxy that exposes `gpu_id = 0` inside `_async_output_loop`
    # fixes the stream device index without mutating `self.gpu_id` used by the
    # main worker loop for rank/output routing.
    #
    # REMOVAL / FIX CONDITION:
    # Can be removed once vLLM-Omni's WorkerProc._async_output_loop uses
    # `current_omni_platform.get_torch_device(self.gpu_id)` instead of raw
    # `torch.device(..., self.gpu_id)`.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.worker import diffusion_worker

    worker_proc_cls = diffusion_worker.WorkerProc
    orig_async_output_loop = _unwrap_tpu_patch(worker_proc_cls._async_output_loop)

    class _ZeroGpuIdProxy:
        """``self`` stand-in for the body of ``_async_output_loop`` only.

        Every attribute read/write is forwarded to the real WorkerProc except
        ``gpu_id``, which reads as the local device index 0. Nothing outside
        that one method body ever sees the proxy, so ``self.gpu_id`` keeps its
        rank meaning everywhere else. Any new ``self.gpu_id`` use *inside* the
        loop (e.g. a log line) would also read 0.
        """

        def __init__(self, target: Any) -> None:
            object.__setattr__(self, "_target", target)

        @property
        def gpu_id(self) -> int:
            return 0

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_target"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(object.__getattribute__(self, "_target"), name, value)

    def _safe_async_output_loop(self: Any, *args: Any, **kwargs: Any) -> Any:
        return orig_async_output_loop(_ZeroGpuIdProxy(self), *args, **kwargs)

    worker_proc_cls._async_output_loop = _mark_tpu_patch(
        _safe_async_output_loop, orig_async_output_loop
    )
    logger.info(
        "Applied TPU patch: local device index 0 for WorkerProc._async_output_loop."
    )
