# SPDX-License-Identifier: Apache-2.0
"""MoERunner patches for layers armed by fused_moe_tp_ar.

The base has no custom_ops/moe_runner.py: `_patch_moe_runner_fused_output_is_reduced`
patches upstream MoERunner in place. An armed TP layer owns its own token
exchange (replicated below the SP bucket gate, gathered/scattered in-kernel
above it), so vLLM's naive EP dispatch/combine must be off for it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

import vllm_torchtpu
import vllm_torchtpu.envs as envs

_ATTRS = (
    "_fused_output_is_reduced",
    "do_naive_dispatch_combine",
    "_tpu_fused_output_reduced_patch",
)


@pytest.fixture
def runner_cls():
    cls = mr.MoERunner
    saved = {name: cls.__dict__.get(name) for name in _ATTRS}
    if "_tpu_fused_output_reduced_patch" in cls.__dict__:
        del cls._tpu_fused_output_reduced_patch
    yield cls
    for name, value in saved.items():
        if value is not None:
            setattr(cls, name, value)
        elif name in cls.__dict__:
            delattr(cls, name)


def _runner(*, armed, sequence_parallel, dp_size=1):
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            dp_size=dp_size,
            is_sequence_parallel=sequence_parallel,
            skip_final_all_reduce=False,
        ),
        _quant_method=SimpleNamespace(supports_internal_mk=False, moe_kernel=None),
    )
    if armed:
        runner._fused_moe_tp_ar_armed = True
    return runner


def _apply(**flags):
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", flags.get("ep", False)),
        patch.object(envs, "USE_MOE_FUSED_TP_AR", flags.get("tp_ar", False)),
    ):
        vllm_torchtpu._patch_moe_runner_fused_output_is_reduced()


def test_not_installed_without_a_fused_kernel_flag(runner_cls):
    before = runner_cls.__dict__.get("do_naive_dispatch_combine")
    _apply()
    assert runner_cls.__dict__.get("do_naive_dispatch_combine") is before


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_armed_tp_layer_skips_naive_dispatch(runner_cls, sequence_parallel):
    _apply(tp_ar=True)
    prop = runner_cls.do_naive_dispatch_combine
    armed = _runner(armed=True, sequence_parallel=sequence_parallel, dp_size=2)
    assert prop.fget(armed) is False
    assert runner_cls._fused_output_is_reduced.fget(armed) is True


@pytest.mark.parametrize(
    ("sequence_parallel", "dp_size", "expected"),
    [(False, 1, False), (True, 1, True), (False, 2, True)],
)
def test_unarmed_layer_keeps_upstream_naive_dispatch(
    runner_cls, sequence_parallel, dp_size, expected
):
    _apply(tp_ar=True)
    runner = _runner(armed=False, sequence_parallel=sequence_parallel, dp_size=dp_size)
    assert runner_cls.do_naive_dispatch_combine.fget(runner) is expected
