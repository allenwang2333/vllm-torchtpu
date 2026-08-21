# SPDX-License-Identifier: Apache-2.0
"""The token-bucket guard on the low-latency all-reduce entries.

Warmup can wire exactly one shape -- the workspaces have to be arguments of a
custom op whose schema is fixed at registration -- so on every other token
bucket the reduction has to go somewhere else. The op's own shape check already
falls back to XLA and is correct, but from inside an opaque custom call XLA can
neither fuse nor reorder it, and the caller has by then done whatever work it
does to prepare the payload. On K3's latent-MoE tail that preparation is a
``torch.cat`` plus two slices per layer, ~300 us per decode step, spent on a
kernel that never runs: job 139 gained 3.3% TPOT at the wired bucket and lost
5.2% / 4.0% at the two that were not.

So callers ask first, and this is what they ask. It has to be right without a
TPU present, and it has to read a Python int rather than a tensor dim -- reading
a traced dim pins vLLM's dynamic token dimension and fails compilation outright.

The context is installed by assigning ``forward_context._forward_context``
directly rather than through ``set_forward_context``, because that helper needs a
real ``VllmConfig`` and one cannot be built off-device: ``DeviceConfig`` calls
``torch.device("tpu")``, which no CPU-only torch accepts. Assigning the global is
what ``set_forward_context`` does anyway, and the field this reads is the same
one either way.
"""

import pytest
import torch
from vllm import forward_context as fc
from vllm.forward_context import BatchDescriptor

from vllm_torchtpu.distributed.low_latency_all_reduce import bucket_matches

# (rows, op, buffers) -- only element 0 is under test here.
ENTRY = (2, object(), ())


@pytest.fixture
def context(monkeypatch):
    """Install a ForwardContext with the given padded token count, or none."""

    def install(num_tokens: int | None):
        if num_tokens is None:
            monkeypatch.setattr(fc, "_forward_context", None, raising=False)
            return None
        ctx = fc.ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        )
        monkeypatch.setattr(fc, "_forward_context", ctx, raising=False)
        return ctx

    return install


def test_matches_the_wired_bucket(context):
    context(2)
    assert bucket_matches(ENTRY) is True


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 512])
def test_rejects_every_other_bucket(context, num_tokens):
    # A decode step is padded up to a compile bucket, so this is the c4/c8 case:
    # with compile_sizes [2,16,512] both pad to 16 while the kernel was built
    # for 2, and job 139 regressed at both.
    context(num_tokens)
    assert bucket_matches(ENTRY) is False


def test_no_forward_context_means_no_kernel(context):
    # Unit tests and any pre-warmup call site have no context. "Cannot tell"
    # must resolve to the path the caller would have had anyway, not to a kernel
    # invoked on an unknown shape.
    context(None)
    assert bucket_matches(ENTRY) is False


def test_missing_descriptor_means_no_kernel(monkeypatch):
    # cudagraph dispatch is what populates batch_descriptor upstream, so a
    # context without one is reachable; it must not raise.
    ctx = fc.ForwardContext(no_compile_layers={},
                            attn_metadata={},
                            slot_mapping={},
                            batch_descriptor=None)
    monkeypatch.setattr(fc, "_forward_context", ctx, raising=False)
    assert bucket_matches(ENTRY) is False


def test_reads_an_int_not_a_tensor(context):
    # The whole point of going through the forward context: the TPU runner
    # passes num_tokens=num_tokens_padded, an ordinary int set outside anything
    # traced. If this ever became a tensor the guard would stop folding at trace
    # time and start pinning the dynamic token dimension -- the failure this
    # module is arranged to avoid.
    ctx = context(2)
    assert type(ctx.batch_descriptor.num_tokens) is int


# --- shape-guard mode (TPU_LOW_LATENCY_ALLREDUCE_SHAPE_GUARD=1) --------------
#
# The context path above folds to a constant, which leaves no ShapeEnv guard for
# compilation/shape_variants (#474) to partition on, so on a multi-bucket config
# two traces claim the same bucket with different graph inputs and startup dies
# with ShapeSpecializationError. These cover the alternative: read the dim, so
# the branch is visible to the guard machinery.


def test_shape_guard_uses_the_payload_not_the_context(context, monkeypatch):
    # The context says bucket 512, the payload says 2 rows. Under the flag the
    # payload wins -- that is what makes the branch legible to #474.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_SHAPE_GUARD", "1")
    context(512)
    assert bucket_matches(ENTRY, torch.empty(2, 3584)) is True


@pytest.mark.parametrize("rows", [16, 512])
def test_shape_guard_rejects_larger_payloads(context, monkeypatch, rows):
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_SHAPE_GUARD", "1")
    context(2)
    assert bucket_matches(ENTRY, torch.empty(rows, 3584)) is False


def test_shape_guard_needs_the_payload(context, monkeypatch):
    # No payload means the caller could not supply one, so fall back to the
    # context path rather than silently answering from nothing.
    monkeypatch.setenv("TPU_LOW_LATENCY_ALLREDUCE_SHAPE_GUARD", "1")
    context(2)
    assert bucket_matches(ENTRY) is True
    context(512)
    assert bucket_matches(ENTRY) is False


def test_payload_is_ignored_while_the_flag_is_off(context, monkeypatch):
    # Default path must be untouched: every measurement on this branch was taken
    # with the context deciding, so passing a payload must not change it.
    monkeypatch.delenv("TPU_LOW_LATENCY_ALLREDUCE_SHAPE_GUARD", raising=False)
    context(2)
    assert bucket_matches(ENTRY, torch.empty(512, 3584)) is True
    context(512)
    assert bucket_matches(ENTRY, torch.empty(2, 3584)) is False
