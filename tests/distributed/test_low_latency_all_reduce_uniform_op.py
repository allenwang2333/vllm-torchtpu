# SPDX-License-Identifier: Apache-2.0
"""The TP reduce op is emitted uniformly and chooses inside itself.

This is the property that makes the kernel safe under vLLM's
one-trace-many-buckets compilation, so it is worth pinning down without a TPU.
``compile_all_ranges`` feeds a *single* FX graph concrete inputs per compile
size, so a graph's structure is fixed at trace time and every bucket inherits
it; ``compilation/shape_variants.py`` can re-trace per group, but only from
ShapeEnv guards, and a token-count branch cannot leave one (reading that dim
pins vLLM's marked-dynamic input_ids dimension). The trace that stays live must
then serve every bucket, and ``BucketExecutables.get`` checks each handed-over
executable against the subgraph's placeholder list. An op whose workspaces are
graph inputs on some buckets and absent on others produces two placeholder
lists for one live graph, which is the ShapeSpecializationError this design
exists to avoid.

So: the op is called for every payload, and the kernel-or-psum choice happens
below the custom-op boundary where the shape is a concrete int. What is tested
here is the torch half of that -- that no bucket is diverted around the op, and
that the workspace advances for exactly the payload the kernel serves. The JAX
half needs real devices.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import low_latency_all_reduce as llar

WIRED = (2, 8)


@pytest.fixture
def probe():
    """A plan whose op records its calls and fakes both branches.

    ``x * 2`` stands in for the kernel and ``x * 3`` for ``jax.lax.psum``, so
    which branch ran is readable from the output. Only the kernel branch
    advances the workspace, matching the real body.
    """
    calls = []

    class FakePlan:
        shape = WIRED
        dtype = torch.float32
        kernel = "two_shot"
        buffers = ()

        def op(self, x, rs, ag, step):
            calls.append(tuple(x.shape))
            if x.numel() == WIRED[0] * WIRED[1]:
                return x * 2, rs + 1, ag + 1, step + 1
            return x * 3, rs, ag, step

    dispatcher = SimpleNamespace(xla_all_reduce=lambda x: x)
    return FakePlan(), dispatcher, calls


@pytest.fixture
def workspace():
    return (torch.zeros(2), torch.zeros(2), torch.zeros(1, dtype=torch.int32))


def _op(probe, name):
    plan, dispatcher, _ = probe
    return llar.build_reduce_op(dispatcher, plan, f"vllm_torchtpu::{name}")


def test_the_wired_payload_runs_the_kernel_and_advances_the_workspace(
        probe, workspace):
    # Slot parity is the kernel's only defence against reusing a buffer a peer
    # is still reading, so the writeback is not an optimisation detail.
    op = _op(probe, "_t_wired")
    rs, ag, step = workspace
    out = op(torch.ones(*WIRED), rs, ag, step)
    assert torch.equal(out, torch.full(WIRED, 2.0))
    assert step.item() == 1


def test_other_payloads_still_go_through_the_same_op(probe, workspace):
    # The point of the design: no bucket is diverted around the op, so every
    # trace of the model has the same call node with the same operands.
    op = _op(probe, "_t_uniform")
    _, _, calls = probe
    rs, ag, step = workspace
    out = op(torch.ones(16, 8), rs, ag, step)
    assert torch.equal(out, torch.full((16, 8), 3.0))
    assert calls == [(16, 8)]


def test_the_workspace_is_frozen_on_the_fallback_branch(probe, workspace):
    # The fallback reduces without touching the workspace, so writing it back
    # would cost three copies on every bucket the kernel does not serve -- and
    # would also desynchronise the round counter from the slots it indexes.
    op = _op(probe, "_t_frozen")
    rs, ag, step = workspace
    op(torch.ones(*WIRED), rs, ag, step)
    op(torch.ones(16, 8), rs, ag, step)
    assert step.item() == 1


def test_element_count_decides_not_the_row_count(probe, workspace):
    # Both sides of the boundary test numel, not shape: _build_two_shot
    # flattens to (num_devices, numel/num_devices), so a payload with the same
    # element count is the same payload as far as the workspace is concerned.
    # If these two predicates ever diverge, torch would skip a writeback that
    # JAX's kernel branch needed and the slot parity would go stale silently.
    op = _op(probe, "_t_numel")
    rs, ag, step = workspace
    out = op(torch.ones(4, 4), rs, ag, step)
    assert torch.equal(out, torch.full((4, 4), 2.0))
    assert step.item() == 1
