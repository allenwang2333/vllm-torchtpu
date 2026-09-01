# SPDX-License-Identifier: Apache-2.0
"""The out-proj reduce op is one opaque op that switches inside itself.

This is the property that makes the kernel safe under this stack's
one-trace-many-buckets compilation, so it is worth pinning down without a
TPU. The backbone is traced once with the token dim dynamic and the guards
dropped, so a Python branch on token count in the layer's forward raises
ConstraintViolationError at profile_run, and wiring the kernel's workspaces
per bucket would give the single FX graph a different interface per compile
size (ShapeSpecializationError). So the layer calls ONE custom op whose
signature -- ``(x, rs_list, ag_list, slot, barrier_first)`` -- never varies:
the lists hold one exactly-sized workspace pair per wired bucket, the
implementation reads the concrete token count at execution time, and the two
statics come from the layer's ordinal (alternating slot; barrier-first on the
first layer).

What is tested here is the torch half: every bucket goes through the same op,
the kernel branch runs for exactly the wired buckets, and everything else
takes the XLA fallback unchanged. The kernel itself is stateless (full
barrier per call, single workspace slot), so there is no cross-call counter
to track. The JAX half needs real devices and lives in
``scripts/k3/llar_out_proj_check.sbatch``.
"""

import itertools

import pytest
import torch

from vllm_torchtpu.distributed import low_latency_all_reduce as llar

# The defaults K3's out-proj runs with: 32 devices, bf16, width 7168, and the
# batch knob at its default of 8. At 32 devices the lane-alignment rule leaves
# the multiples of 4 wired: buckets 4 and 8.
NUM_DEVICES = 32
WIDTH = 7168
WIRED = [4, 8]
WIRED_MAX_NUMEL = max(WIRED) * WIDTH

_NAME_COUNTER = itertools.count()


class FakeInner:
    """Stands in for the sharded JAX op; records calls and tags its branch.

    ``x * 2`` marks the kernel path and ``x * 3`` the XLA fallback, so which
    branch ran is readable from the output.
    """

    def __init__(self):
        self.kernel_calls = []
        self.fallback_calls = []

    def __call__(self, x, rs, ag):
        self.kernel_calls.append(tuple(x.shape))
        return x * 2

    def xla_fallback(self, x):
        self.fallback_calls.append(tuple(x.shape))
        return x * 3


def _workspaces():
    """One workspace pair per wired bucket, shaped like the real ones."""
    rs_list, ag_list = [], []
    for tokens in WIRED:
        w = tokens * WIDTH // NUM_DEVICES
        rs_list.append(torch.zeros((2, NUM_DEVICES - 1, 1, w)))
        ag_list.append(torch.zeros((2, tokens, WIDTH // w, 1, w)))
    return rs_list, ag_list


@pytest.fixture
def op_and_fake():
    fake = FakeInner()
    name = f"vllm_torchtpu::test_llar_switch_{next(_NAME_COUNTER)}"
    # The outer op takes one inner op per (slot, barrier_first) variant; the
    # statics are baked into each variant's closure in production.
    inner_map = {(0, 1): fake, (0, 0): fake, (1, 0): fake}
    op = llar._build_outer_op(name,
                              inner_map,
                              width=WIDTH,
                              bucket_index={
                                  t: i
                                  for i, t in enumerate(WIRED)
                              },
                              xla_fallback=fake.xla_fallback)
    return op, fake


@pytest.mark.parametrize("tokens", WIRED)
def test_wired_buckets_take_the_kernel(op_and_fake, tokens):
    op, fake = op_and_fake
    rs, ag = _workspaces()
    x = torch.full((tokens, WIDTH), 1.0)
    out = op(x, rs, ag, 0, 1)
    assert torch.equal(out, x * 2)
    assert fake.kernel_calls == [(tokens, WIDTH)]
    assert not fake.fallback_calls


@pytest.mark.parametrize("tokens", [1, 2, 16, 32, 8192])
def test_unwired_buckets_take_the_xla_fallback(op_and_fake, tokens):
    """T=1,2 are not lane-aligned at 32 devices; 16+ is past the gate."""
    op, fake = op_and_fake
    rs, ag = _workspaces()
    x = torch.full((tokens, WIDTH), 1.0)
    out = op(x, rs, ag, 0, 0)
    assert torch.equal(out, x * 3)
    assert not fake.kernel_calls
    assert fake.fallback_calls == [(tokens, WIDTH)]


def test_the_switch_across_buckets(op_and_fake):
    """Both sides of the wiring, interleaved, through one op."""
    op, fake = op_and_fake
    rs, ag = _workspaces()
    for i, tokens in enumerate((4, 16, 8, 32, 4, 8)):
        x = torch.full((tokens, WIDTH), 1.0)
        out = op(x, rs, ag, i % 2, 1 if i == 0 else 0)
        if tokens in WIRED:
            assert torch.equal(out, x * 2)
        else:
            assert torch.equal(out, x * 3)
    assert fake.kernel_calls == [(4, WIDTH), (8, WIDTH), (4, WIDTH),
                                 (8, WIDTH)]
    assert fake.fallback_calls == [(16, WIDTH), (32, WIDTH)]


def test_odd_payload_falls_back(op_and_fake):
    op, fake = op_and_fake
    rs, ag = _workspaces()
    # 3000 elements is not a wired bucket of width 7168, so the kernel never
    # sees it.
    x = torch.full((3, 1000), 1.0)
    out = op(x, rs, ag, 0, 1)
    assert torch.equal(out, x * 3)
    assert not fake.kernel_calls


def test_default_wiring_matches_the_gate():
    """The test constant and the production wiring must not drift apart."""
    assert llar.wired_buckets(WIDTH, torch.bfloat16, NUM_DEVICES) == WIRED
    assert llar.wired_max_numel(WIDTH, torch.bfloat16,
                                NUM_DEVICES) == WIRED_MAX_NUMEL
