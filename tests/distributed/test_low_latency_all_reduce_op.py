# SPDX-License-Identifier: Apache-2.0
"""The out-proj reduce op is one opaque op that switches inside itself.

This is the property that makes the kernel safe under this stack's
one-trace-many-buckets compilation, so it is worth pinning down without a
TPU. The backbone is traced once with the token dim dynamic and the guards
dropped, so a Python branch on token count in the layer's forward raises
ConstraintViolationError at profile_run, and wiring the kernel's workspaces
per bucket would give the single FX graph a different interface per compile
size (ShapeSpecializationError). So the layer calls ONE custom op whose
signature -- ``(x, rs, ag, step)`` -- never varies, and the implementation
reads the concrete token count at execution time.

What is tested here is the torch half: every bucket goes through the same op,
the kernel branch runs for exactly the payloads at or below the wired
ceiling, the workspace advances only for those payloads, and everything else
takes the XLA fallback unchanged. The JAX half needs real devices and lives
in ``scripts/k3/llar_out_proj_check.sbatch``.
"""

import itertools

import pytest
import torch

from vllm_torchtpu.distributed import low_latency_all_reduce as llar

# The defaults K3's out-proj runs with: 32 devices, bf16, width 7168, and the
# batch knob at its default of 8.
NUM_DEVICES = 32
WIDTH = 7168
WIRED_MAX_NUMEL = 8 * WIDTH

_NAME_COUNTER = itertools.count()


class FakeInner:
    """Stands in for the sharded JAX op; records calls and tags its branch.

    ``x * 2`` marks the kernel path and ``x * 3`` the XLA fallback, so which
    branch ran is readable from the output. Only the kernel branch advances
    the workspace, matching the real body.
    """

    def __init__(self):
        self.kernel_calls = []
        self.fallback_calls = []

    def __call__(self, x, rs, ag, step):
        self.kernel_calls.append(tuple(x.shape))
        return x * 2, rs + 1, ag + 1, step + 1

    def xla_fallback(self, x):
        self.fallback_calls.append(tuple(x.shape))
        return x * 3


@pytest.fixture
def op_and_fake():
    fake = FakeInner()
    name = f"vllm_torchtpu::test_llar_switch_{next(_NAME_COUNTER)}"
    op = llar._build_outer_op(name,
                              fake,
                              max_numel=WIRED_MAX_NUMEL,
                              num_devices=NUM_DEVICES,
                              xla_fallback=fake.xla_fallback)
    return op, fake


def _workspace():
    rs = torch.zeros((2, NUM_DEVICES - 1, 1, WIRED_MAX_NUMEL // NUM_DEVICES))
    ag = torch.zeros((2, NUM_DEVICES, 1, WIRED_MAX_NUMEL // NUM_DEVICES))
    step = torch.zeros(1, dtype=torch.int32)
    return rs, ag, step


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_wired_buckets_take_the_kernel(op_and_fake, tokens):
    op, fake = op_and_fake
    rs, ag, step = _workspace()
    x = torch.full((tokens, WIDTH), 1.0)
    out = op(x, rs, ag, step)
    assert torch.equal(out, x * 2)
    assert fake.kernel_calls == [(tokens, WIDTH)]
    assert not fake.fallback_calls
    # The workspace advanced exactly once.
    assert step.item() == 1


@pytest.mark.parametrize("tokens", [16, 32, 8192])
def test_unwired_buckets_take_the_xla_fallback(op_and_fake, tokens):
    op, fake = op_and_fake
    rs, ag, step = _workspace()
    x = torch.full((tokens, WIDTH), 1.0)
    out = op(x, rs, ag, step)
    assert torch.equal(out, x * 3)
    assert not fake.kernel_calls
    assert fake.fallback_calls == [(tokens, WIDTH)]
    # The workspace is untouched, so a later kernel call's slot parity is
    # exactly where the last kernel call left it.
    assert step.item() == 0
    assert rs.sum().item() == 0 and ag.sum().item() == 0


def test_the_switch_at_the_measured_boundary(op_and_fake):
    """Both sides of the threshold, interleaved, through one op.

    T = 4, 8 take the kernel; T = 16, 32 take the fallback. Interleaving them
    also pins the workspace-parity contract: fallback calls must not advance
    the step counter the kernel's slot rotation depends on.
    """
    op, fake = op_and_fake
    rs, ag, step = _workspace()
    expected_step = 0
    for tokens in (4, 16, 8, 32, 4, 8):
        x = torch.full((tokens, WIDTH), 1.0)
        out = op(x, rs, ag, step)
        if llar._takes_kernel(x.numel(), WIRED_MAX_NUMEL, NUM_DEVICES):
            expected_step += 1
            assert torch.equal(out, x * 2)
        else:
            assert torch.equal(out, x * 3)
        assert step.item() == expected_step
    assert fake.kernel_calls == [(4, WIDTH), (8, WIDTH), (4, WIDTH),
                                 (8, WIDTH)]
    assert fake.fallback_calls == [(16, WIDTH), (32, WIDTH)]


def test_indivisible_payload_falls_back_even_below_the_ceiling(op_and_fake):
    op, fake = op_and_fake
    rs, ag, step = _workspace()
    # Below the ceiling in bytes, but 3000 elements do not divide 32 devices,
    # so the kernel cannot scatter the payload.
    x = torch.full((3, 1000), 1.0)
    assert x.numel() % NUM_DEVICES != 0
    out = op(x, rs, ag, step)
    assert torch.equal(out, x * 3)
    assert not fake.kernel_calls


def test_default_ceiling_matches_the_gate():
    """The test constant and the production ceiling must not drift apart."""
    assert llar.wired_max_numel(WIDTH, torch.bfloat16,
                                NUM_DEVICES) == WIRED_MAX_NUMEL
