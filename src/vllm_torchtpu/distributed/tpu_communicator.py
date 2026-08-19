# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU device communicator for EP across data-parallel engines.

Dispatch gathers tokens across DP ranks that share the same TP rank. Combine
sums those matching-TP partials across DP and returns this DP rank's token
slice. vLLM's existing ``reduce_results`` TP all-reduce remains responsible for
the final TP-side reduction. Callers must pass equal token counts on each DP
rank.
"""

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase, DeviceCommunicatorBase)
from vllm.distributed.parallel_state import get_dp_group

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import fp8_wire
from vllm_torchtpu.env_override import (SC_ALL_GATHER_OFFLOAD_FLAG,
                                        sc_all_gather_offload_threshold_bytes)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _offload_gate_is_traceable() -> bool:
    """Whether the runner can compile a wire choice that varies per bucket.

    It cannot, here. ``dispatch_hidden_states`` runs inside the compiled
    backbone, whose token dim vLLM marks dynamic; a Python branch on that dim
    specializes it and ``torch.compile`` refuses:

        ConstraintViolationError: You marked L['positions'].size()[1] as
        dynamic but your code specialized it to be a constant (4096)

    Measured on device: every configuration with the gate on died at engine
    init, every configuration without it served. The capability is a runner
    property, not a communicator one -- ``capture_model`` has to notice that a
    bucket needs its own trace, which #474 (``6ecb5c99``, one graph per
    token-bucket group) has since landed. That is necessary and not sufficient:
    ``test_a_wire_chosen_per_bucket_cannot_be_traced`` still raises on this
    tree, because the boundary coincides with the dim's upper bound and the
    range collapses to a point before any retrace can help. So this stays an
    explicit opt-in that no runner sets today; a runner that has solved both
    halves says so by defining ``RETRACES_PER_BUCKET_GROUP``, and that test is
    the tripwire that says when it is safe.
    """
    try:
        from vllm_torchtpu.runner import tpu_runner
    except Exception:  # a communicator must not depend on importing the runner
        return False
    return getattr(tpu_runner, "RETRACES_PER_BUCKET_GROUP", False)


class TpuNullAll2AllManager(All2AllManagerBase):
    """Probe-only stub; TPU EP dispatch/combine never goes through it."""


class TpuDeviceCommunicator(DeviceCommunicatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TPU performs EP dispatch/combine directly (see dispatch/combine
        # below), not through an all2all manager. The stub only satisfies
        # vLLM's DP>1 + MoE fault-tolerance probe
        # (support_fault_tolerance=False); it is never used for compute.
        if self.use_all2all:
            self.all2all_manager = TpuNullAll2AllManager(self.cpu_group)
        # Read once: one communicator is built per process at init.
        self.dispatch_fp8 = envs.TPU_MOE_DISPATCH_FP8
        self.packed_wire = fp8_wire.packed_wire()
        self.offload_threshold = (sc_all_gather_offload_threshold_bytes()
                                  if envs.TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD
                                  and _offload_gate_is_traceable() else None)
        self._log_engagement()

    def _log_engagement(self) -> None:
        """One greppable ``[fp8wire]`` line per knob, so a server log says
        which wire configuration booted.

        All of it lands at init and none of it at dispatch time on purpose:
        torch_tpu's tracer rejects a logger call inside the compiled MoE
        forward ("logging.Logger method not supported for non-export cases"),
        which is the same reason ``moe_routing`` resolves its simulation
        strategy at import. The per-bucket consequence of the gate is
        therefore reported as the row count where it flips, computed by the
        arithmetic the predicate uses.
        """
        if not self.dispatch_fp8:
            return  # with the flag off, no [fp8wire] line is printed at all
        block = fp8_wire.wire_block()
        grid = ("gmm_v2's own grid, so its lhs requant is a bitwise no-op"
                if block else "one scale per row, so gmm_v2 rounds again")
        logger.info_once(
            "[fp8wire] MoE dispatch all-gather runs in fp8: "
            f"wire={'uint8' if self.packed_wire else 'fp8+scale'} "
            f"block={block or 'row'} ({grid})")
        if not envs.TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD:
            logger.info_once(
                "[fp8wire] offload gate=off: fp8 at every token bucket")
        elif not _offload_gate_is_traceable():
            logger.warning_once(
                "[fp8wire] offload gate=REFUSED: choosing the wire per bucket "
                "is a Python branch on the token count, which specializes the "
                "dynamic token dim and dies at engine init with "
                "ConstraintViolationError on this runner. It needs the "
                "per-bucket retrace series (d4a603cb, e682de33a); until then "
                "this arm runs as plain fp8 dispatch.")
        elif self.offload_threshold is None:
            logger.info_once(
                "[fp8wire] offload gate=on but INERT: no "
                f"{SC_ALL_GATHER_OFFLOAD_FLAG} in LIBTPU_INIT_ARGS, so the "
                "threshold is libtpu's own and unknown here")
        else:
            logger.info_once(
                f"[fp8wire] offload gate=on threshold="
                f"{self.offload_threshold}B{self._engagement_bucket()}")

    def _engagement_bucket(self) -> str:
        """``T_local`` at which the gated wire starts taking the fp8 path."""
        config = get_current_vllm_config_or_none()
        if config is None:
            return ""
        try:
            hidden = config.model_config.get_hidden_size()
            dp = config.parallel_config.data_parallel_size
            rows = -(-self.offload_threshold // (dp * hidden))
        except Exception:  # a log line must never be the thing that fails
            return ""
        return (f": with dp={dp} hidden={hidden}, fp8 engages at "
                f"T_local>={rows} rows ({rows * dp} gathered); every smaller "
                "bucket takes the bf16 all-gather")

    def _dp_gather(self, x: torch.Tensor) -> torch.Tensor:
        return get_dp_group().all_gather(x, dim=0)

    def _fp8_keeps_the_offloaded_gather(self, x: torch.Tensor) -> bool:
        """Whether the fp8 payload still earns the async all-gather.

        XLA offloads an all-gather to the SparseCore -- the asynchronous form
        that overlaps the compute it sits between -- at ``output_bytes >=``
        the threshold ``env_override`` sets (64 MiB on tpu7x). Halving the
        payload can cross back under it, turning an overlapped bf16 gather
        into an exposed synchronous fp8 one; and below the threshold both
        wires are exposed, where the extra collective's fixed cost (~9 us) is worth
        more than the bytes saved. One byte per element: this is the fp8
        buffer.

        Unreachable today: ``self.offload_threshold`` is ``None`` unless
        ``_offload_gate_is_traceable()``, which no runner satisfies, because
        this predicate is a Python branch on the token dim and the compiled
        backbone marks that dim dynamic.

        Deliberately silent: this runs inside the compiled MoE forward, where
        a logger call is rejected by the tracer. ``_log_engagement`` reports
        the flip point at init instead.
        """
        if self.offload_threshold is None:
            return True
        return x.numel() * get_dp_group().world_size >= self.offload_threshold

    def dispatch_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """DP all-gather of MoE token rows, in fp8 when the flag is set.

        The single owner of the dispatch wire format: anything that gathers
        activations for the MoE -- including a patched upstream
        ``MoERunner._maybe_dispatch`` -- must come through here, or
        ``TPU_MOE_DISPATCH_FP8`` silently stops applying. Safe to call at
        ``dp_size == 1``.

        The rows are rounded to fp8 one step later regardless -- gmm_v2
        quantizes its lhs before the matmul -- so the wire mostly costs a
        rounding the MoE was going to do anyway. Not exactly, and the default
        is the expensive side: the wire scales per row where the kernel scales
        per 512-wide k block, so the payload is rounded twice on two grids and
        the lhs the kernel finally sees sits 3.66e-2 from the fp32 rows where
        the bf16 dispatch puts it at 2.65e-2 -- 1.38x the error, at hidden
        4096, measured on this code. ``TPU_MOE_DISPATCH_FP8_BLOCK=512`` makes
        the two grids the same and the kernel's requantization a bitwise no-op.

        The rows and their scales cross as two collectives, not one packed
        uint8 buffer: packing saves ~9 us of fixed cost and loses the async
        collective form. See ``fp8_wire.packed_wire`` for the two device runs
        that measured that and what they do and do not attribute.
        """
        if get_dp_group().world_size == 1:
            return x
        if not self.dispatch_fp8 or not x.is_floating_point():
            return self._dp_gather(x)
        if not self._fp8_keeps_the_offloaded_gather(x):
            return self._dp_gather(x)
        if self.packed_wire:
            payload = self._dp_gather(fp8_wire.encode(x))
            return fp8_wire.decode(payload, x.shape[-1], x.dtype)
        # Two collectives: the fp8 rows, which keep the async offloaded form
        # the bf16 gather had, and the fp32 scale that reconstructs them. The
        # scale is [T, H/block], laid out dense -- 4 to 32 B/token, six orders
        # of magnitude below the offload threshold, so that leg is always
        # synchronous and costs a fixed ~9 us.
        q, scale = fp8_wire.quantize_rows(x)
        return fp8_wire.dequantize_rows(self._dp_gather(q),
                                        self._dp_gather(scale), x.dtype)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, router_logits
        hidden_states = self.dispatch_hidden_states(hidden_states)
        # Router logits stay bf16: they are 1/8 of the hidden payload and a
        # rounding there moves top-k membership, not just magnitudes.
        router_logits = self._dp_gather(router_logits)
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported by TpuDeviceCommunicator")
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states, topk_weights, topk_ids
        hidden_states = self.dispatch_hidden_states(hidden_states)
        topk_weights = self._dp_gather(topk_weights)
        topk_ids = self._dp_gather(topk_ids)
        return hidden_states, topk_weights, topk_ids

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        dp = get_dp_group()
        if dp.world_size == 1:
            return hidden_states
        return dp.reduce_scatter(hidden_states, dim=0)
