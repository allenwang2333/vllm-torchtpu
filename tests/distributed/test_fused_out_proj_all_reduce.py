# SPDX-License-Identifier: Apache-2.0
"""The fused out-proj+AR op is one opaque op that switches inside itself.

Same property as the two-shot low-latency all-reduce (see
test_low_latency_all_reduce_op.py for the full rationale): the backbone is
traced once with the token dim dynamic, so the T=1 kernel-vs-stock choice
must live behind a custom-op boundary. What is tested here is the torch
half: T=1 routes to the kernel branch, everything else routes to the
fallback, and the (slot, barrier_first) statics select the right inner op.
The kernel itself needs real devices and lives in
``scripts/k3/fused_out_proj_ar_check.sbatch``.

Since the tail-epilogue fusion (docs/llar_out_proj_op.md, jobs 787/790) the
op also takes the residual stream and the f32 folded AttentionResidual
scorer weight, and returns (hidden, inv_rms, score).
"""

import itertools

import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.distributed import fused_out_proj_all_reduce as foar
from vllm_torchtpu.kernels.collectives.fused_out_proj_all_reduce import (
    fused_out_proj_all_reduce, make_fused_out_proj_workspace)

NUM_DEVICES = 32
WIDTH = 7168
K_SHARD = 224
TILES = WIDTH // 256
_NAME_COUNTER = itertools.count()


class FakeInner:
    """Stands in for the sharded JAX op; records calls and tags its branch.

    ``x * 2`` marks the kernel path and ``x * 3`` the fallback, so which
    branch ran is readable from the output.
    """

    def __init__(self):
        self.kernel_calls = []

    def __call__(self, x, w, resid, fw, rs, ag):
        self.kernel_calls.append(tuple(x.shape))
        return (x * 2, torch.full((x.shape[0], ),
                                  2.0), torch.full((x.shape[0], ), 2.0))


@pytest.fixture()
def op_and_fakes():
    inner = {(0, 1): FakeInner(), (0, 0): FakeInner(), (1, 0): FakeInner()}
    fallback_calls = []

    def fallback(x, w, resid, fw):
        fallback_calls.append(tuple(x.shape))
        return (x * 3, torch.full((x.shape[0], ),
                                  3.0), torch.full((x.shape[0], ), 3.0))

    name = f"vllm_torchtpu::test_fused_oproj_{next(_NAME_COUNTER)}"
    op = foar._build_outer_op(name, inner, fallback, width=WIDTH)
    return op, inner, fallback_calls


def _inputs(tokens):
    x = torch.ones((tokens, K_SHARD))
    w = torch.ones((K_SHARD, WIDTH))
    resid = torch.ones((tokens, WIDTH))
    fw = torch.ones((WIDTH, ), dtype=torch.float32)
    rs = torch.zeros((2, NUM_DEVICES, 2, 128))
    ag = torch.zeros((2, TILES + 1, 2, 128))
    return x, w, resid, fw, rs, ag


def test_t1_takes_the_kernel(op_and_fakes):
    op, inner, fallback_calls = op_and_fakes
    out = op(*_inputs(1), 0, 1)
    assert torch.all(out[0] == 2)
    assert inner[(0, 1)].kernel_calls == [(1, K_SHARD)]
    assert not fallback_calls


def test_larger_tokens_take_the_fallback(op_and_fakes):
    op, inner, fallback_calls = op_and_fakes
    for tokens in (0, 2, 4, 8):
        out = op(*_inputs(tokens), 0, 0)
        assert torch.all(out[0] == 3), f"tokens={tokens}"
    assert len(fallback_calls) == 4
    assert not any(f.kernel_calls for f in inner.values())


def test_slot_and_barrier_select_the_inner_op(op_and_fakes):
    op, inner, _ = op_and_fakes
    op(*_inputs(1), 0, 1)
    op(*_inputs(1), 1, 0)
    op(*_inputs(1), 0, 0)
    assert inner[(0, 1)].kernel_calls == [(1, K_SHARD)]
    assert inner[(1, 0)].kernel_calls == [(1, K_SHARD)]
    assert inner[(0, 0)].kernel_calls == [(1, K_SHARD)]


def test_fake_registration_shapes(op_and_fakes):
    from torch._subclasses.fake_tensor import FakeTensorMode

    op, _, _ = op_and_fakes
    with FakeTensorMode() as m:
        x, w, resid, fw, rs, ag = (m.from_tensor(t) for t in _inputs(1))
        hid, inv, score = op(x, w, resid, fw, rs, ag, 0, 1)
        assert tuple(hid.shape) == (1, WIDTH)
        assert tuple(inv.shape) == (1, )
        assert inv.dtype == torch.float32
        assert tuple(score.shape) == (1, )
        assert score.dtype == torch.float32
        x4 = m.from_tensor(torch.ones((4, K_SHARD)))
        resid4 = m.from_tensor(torch.ones((4, WIDTH)))
        hid, inv, score = op(x4, w, resid4, fw, rs, ag, 0, 0)
        assert tuple(hid.shape) == (4, WIDTH)
        assert tuple(inv.shape) == (4, )


def test_flag_off_builds_nothing(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_FUSED_OUT_PROJ_ALLREDUCE", False)
    assert foar.get_fused_out_proj_entry(WIDTH, K_SHARD,
                                         torch.bfloat16) is None


def test_workspace_shapes():
    rs, ag = make_fused_out_proj_workspace(WIDTH, jnp.bfloat16, NUM_DEVICES)
    assert tuple(rs.shape) == (2, NUM_DEVICES, 2, 128)
    assert tuple(ag.shape) == (2, TILES + 1, 2, 128)
    assert rs.dtype == jnp.bfloat16


def test_kernel_rejects_bad_shapes():
    x = jnp.zeros((1, K_SHARD), jnp.bfloat16)
    w = jnp.zeros((K_SHARD, WIDTH), jnp.bfloat16)
    resid = jnp.zeros((TILES, 2, 128), jnp.bfloat16)
    fw = jnp.zeros((TILES, 2, 128), jnp.float32)
    rs, ag = make_fused_out_proj_workspace(WIDTH, jnp.bfloat16, NUM_DEVICES)
    good = dict(slot=0,
                barrier_first=0,
                axis_name="tp",
                num_devices=NUM_DEVICES)

    with pytest.raises(ValueError, match="T=1 only"):
        fused_out_proj_all_reduce(jnp.zeros((4, K_SHARD), jnp.bfloat16), w,
                                  resid, fw, rs, ag, **good)
    with pytest.raises(ValueError, match="k_shard"):
        fused_out_proj_all_reduce(x, jnp.zeros((112, WIDTH), jnp.bfloat16),
                                  resid, fw, rs, ag, **good)
    with pytest.raises(ValueError, match="multiple"):
        fused_out_proj_all_reduce(x, jnp.zeros((K_SHARD, 7100), jnp.bfloat16),
                                  resid, fw, rs, ag, **good)
    with pytest.raises(ValueError, match="slot"):
        fused_out_proj_all_reduce(x, w, resid, fw, rs, ag, **{
            **good, "slot": 2
        })
    with pytest.raises(ValueError, match="resid must be"):
        fused_out_proj_all_reduce(x, w, jnp.zeros((2, 2, 128), jnp.bfloat16),
                                  fw, rs, ag, **good)
    with pytest.raises(ValueError, match="fw must be"):
        fused_out_proj_all_reduce(x, w, resid,
                                  jnp.zeros((TILES, 2, 128), jnp.bfloat16), rs,
                                  ag, **good)
    with pytest.raises(ValueError, match="rs must be"):
        fused_out_proj_all_reduce(x, w, resid, fw, ag, ag, **good)
    with pytest.raises(ValueError, match="tiles"):
        fused_out_proj_all_reduce(x, w, resid, fw, rs, ag, **{
            **good, "num_devices": 16
        })
