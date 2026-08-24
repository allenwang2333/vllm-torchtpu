# SPDX-License-Identifier: Apache-2.0
"""Dispatch logic of the latent-MoE concat runner.

Three routes leave this runner and only one of them uses the kernel, so what is
under test is which route a given layer and token bucket takes. The concat
itself -- cat, reduce, split -- needs a real MoE layer and a TPU and is covered
end to end.

Instances are built with ``object.__new__`` and given only the attributes
``_use_concat_path`` reads. A real ``MoERunner.__init__`` wants a quant method,
a router and a device, none of which exist off-TPU, and none of which this
decision consults.
"""

from types import SimpleNamespace

import pytest
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu.distributed import low_latency_all_reduce as _llar
from vllm_torchtpu.models.vllm.kimi_k3.latent_moe_runner import \
    KimiK3LatentMoERunner

SENTINEL = object()


def make_runner(monkeypatch,
                *,
                shared_experts=True,
                output_transform=True,
                fused_is_reduced=False,
                sequence_parallel=False,
                tp_size=8,
                ep_size=1):
    runner = object.__new__(KimiK3LatentMoERunner)
    runner._shared_experts = object() if shared_experts else None
    runner.routed_output_transform = object() if output_transform else None
    runner.moe_config = SimpleNamespace(is_sequence_parallel=sequence_parallel,
                                        tp_size=tp_size,
                                        ep_size=ep_size)
    # A property on the class, so it is patched there rather than on the
    # instance -- assigning through an instance would raise AttributeError.
    monkeypatch.setattr(MoERunner,
                        "_fused_output_is_reduced",
                        property(lambda self: fused_is_reduced),
                        raising=False)
    return runner


@pytest.fixture
def super_forward(monkeypatch):
    """Make MoERunner.forward identifiable so a fallback is visible."""
    monkeypatch.setattr(MoERunner, "forward", lambda self, *a, **kw: SENTINEL)


@pytest.fixture
def wired(monkeypatch):
    """Wire a concat entry for token bucket 2."""
    entries = {2: (None, ())}
    monkeypatch.setattr(_llar, "_MOE_CONCAT_ENTRIES", entries, raising=False)
    monkeypatch.setattr(_llar, "select_entry", lambda e: e.get(2))
    return entries


@pytest.mark.parametrize(
    "kwargs",
    [
        # Nothing to concatenate.
        {
            "shared_experts": False
        },
        # No latent transform, so there is no second reduction at latent width.
        {
            "output_transform": False
        },
        # The combine kernel already reduced; upstream reduces only shared.
        {
            "fused_is_reduced": True
        },
        # SP handles its own reduction in the model.
        {
            "sequence_parallel": True
        },
        # No parallelism, so no collective at all.
        {
            "tp_size": 1,
            "ep_size": 1
        },
    ],
)
def test_structural_non_cases_keep_upstream(monkeypatch, super_forward, wired,
                                            kwargs):
    runner = make_runner(monkeypatch, **kwargs)
    assert runner._use_concat_path() is False
    assert runner.forward(None, None) is SENTINEL


def test_the_k3_shape_takes_the_concat_path(monkeypatch, wired):
    runner = make_runner(monkeypatch)
    assert runner._use_concat_path() is True


def test_ep_alone_is_enough(monkeypatch, wired):
    # tp_size == 1 with ep_size > 1 still reduces, matching the predicate
    # upstream's own _maybe_reduce_routed_output_before_transform uses.
    runner = make_runner(monkeypatch, tp_size=1, ep_size=8)
    assert runner._use_concat_path() is True


def test_buckets_above_the_threshold_fall_back_quietly(monkeypatch,
                                                       super_forward, wired):
    # warmup wires the buckets at or below the batch threshold; the rest must
    # keep upstream's pair so XLA's combiner can still fuse it. Silent, not an
    # error -- unlike a wired-but-empty table, this is the expected steady
    # state at any batch size the kernel was not asked to cover.
    runner = make_runner(monkeypatch)
    monkeypatch.setattr(_llar, "select_entry", lambda entries: None)
    assert runner.forward(None, None) is SENTINEL


def test_no_plan_for_any_bucket_is_loud(monkeypatch, super_forward):
    # This class is only built when the feature was requested, so reaching a
    # forward with nothing wired at all means warmup declined and the run would
    # otherwise look wired while quietly doing the old thing. Distinct from the
    # test above: there the table exists and this bucket is simply not in it.
    runner = make_runner(monkeypatch)
    monkeypatch.setattr(_llar, "_MOE_CONCAT_ENTRIES", {}, raising=False)
    with pytest.raises(RuntimeError, match="wired no"):
        runner.forward(None, None)
