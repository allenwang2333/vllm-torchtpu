# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for arming the fused TP MoE + all-reduce path and the reduce flag."""

from types import SimpleNamespace
from unittest.mock import patch

import vllm_torchtpu.envs as envs


def _runner(armed, skip_final=False):
    return SimpleNamespace(
        _quant_method=SimpleNamespace(moe_kernel=None),
        moe_config=SimpleNamespace(skip_final_all_reduce=skip_final),
        _fused_moe_tp_ar_armed=armed,
    )


def test_reduce_flag_true_only_for_armed_runner():
    from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as oot

    prop = oot.VllmMoERunner._fused_output_is_reduced
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", False),
        patch.object(oot, "_upstream_fused_output_is_reduced", lambda self: False),
    ):
        assert prop.fget(_runner(True)) is True
        assert not prop.fget(_runner(False))
        # skip_final_all_reduce keeps the upstream answer even for an armed runner.
        assert not prop.fget(_runner(True, skip_final=True))


def test_arming_is_a_noop_without_the_flag():
    from vllm_torchtpu.layers.adapter import fused_moe_tp_ar as bridge

    with patch.object(envs, "USE_MOE_FUSED_TP_AR", False):
        assert bridge.arm_fused_tp_ar(SimpleNamespace(modules=lambda: [])) == 0


def test_shared_slice_rejects_blockwise_and_biased_layouts():
    import torch

    from vllm_torchtpu.layers.adapter import fused_moe_tp_ar as bridge

    def runner(scale_dim=1, gate_bias=None):
        lin = lambda w, s: SimpleNamespace(weight=w, weight_scale=s)  # noqa: E731
        f8 = torch.zeros(1024, 256, dtype=torch.float8_e4m3fn)
        f8d = torch.zeros(128, 1024, dtype=torch.float8_e4m3fn)
        s1 = torch.ones(256) if scale_dim == 1 else torch.ones(1, 8, 1, 256)
        mlp = SimpleNamespace(
            gate_up_proj=lin(f8, s1),
            down_proj=lin(f8d, torch.ones(1024)),
            expert_gate=SimpleNamespace(weight=torch.zeros(1, 1024), bias=gate_bias),
        )
        return SimpleNamespace(_shared_experts=SimpleNamespace(_layer=mlp))

    ok = bridge._shared_slice(runner())
    assert ok is not None and len(ok) == 5
    # The parameters themselves (no views): 1-D scales and the [1, H] gate weight.
    assert (
        tuple(ok[1].shape) == (256,)
        and tuple(ok[3].shape) == (1024,)
        and tuple(ok[4].shape) == (1, 1024)
    )
    assert ok[4]._base is None and ok[1]._base is None
    assert bridge._shared_slice(runner(scale_dim=4)) is None
    assert bridge._shared_slice(runner(gate_bias=torch.zeros(1))) is None
    assert bridge._shared_slice(SimpleNamespace(_shared_experts=None)) is None
