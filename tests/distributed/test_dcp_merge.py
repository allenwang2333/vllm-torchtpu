# SPDX-License-Identifier: Apache-2.0
"""The torch-level DCP log-sum-exp merge.

`attention_interface.merge_dcp_lse_partials` is the JAX statement of the same
arithmetic and is validated on hardware against a numpy MLA reference
(`SparseMlaDcpTest`). These tests hold the torch path to that already-checked
implementation, so the two cannot drift, and cover the collective plumbing the
JAX version has no equivalent of.

The process group is faked: every rank's partials are held in one process and
the group calls are served from that stack. That is a real test of the merge --
the arithmetic, the rank indexing, and the `-inf` masking are all exercised --
and it needs no distributed launcher.
"""

import multiprocessing

import numpy as np
import pytest
import torch

from vllm_torchtpu.distributed import dcp


class _FakeGroup:
    """Serves `all_gather`/`all_reduce` from a full stack of rank partials.

    `all_reduce` is where the fake has to be careful: the real collective sums
    contributions from every rank, but only one rank's tensor is passed in. So
    the summands come from the stack, and the argument is ignored beyond its
    shape -- which means `all_reduce` here is only faithful when every rank
    would have called it with its own weighted output, exactly as
    `merge_lse_partials` does.
    """

    def __init__(self, outputs, lses, rank):
        self.world_size = len(lses)
        self.rank_in_group = rank
        self._outputs = outputs  # [D, T, N, V] f32
        self._lses = lses  # [D, T, N] f32

    def all_gather(self, tensor, dim=0):
        assert dim == 0 and tensor.shape[0] == 1
        return self._lses.clone()

    def all_reduce(self, tensor):
        shift = torch.where(torch.isfinite(self._lses), self._lses,
                            torch.full_like(self._lses,
                                            float("-inf"))).amax(dim=0)
        empty = ~torch.isfinite(shift)
        shift = torch.where(empty, torch.zeros_like(shift), shift)
        w = torch.where(torch.isfinite(self._lses),
                        torch.exp(self._lses - shift.unsqueeze(0)),
                        torch.zeros_like(self._lses))
        denom = w.sum(0)
        w = w / torch.where(denom > 0, denom, torch.ones_like(denom))
        return (self._outputs * w.unsqueeze(-1)).sum(0).to(tensor.dtype)

    def reduce_scatter(self, tensor, dim=-1):
        total = self.all_reduce(tensor)
        width = total.shape[dim] // self.world_size
        return total.narrow(dim, self.rank_in_group * width, width)


def _merge_as_rank(outputs, lses, rank, monkeypatch):
    group = _FakeGroup(outputs, lses, rank)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(dcp, "get_dcp_rank", lambda: rank)
    return dcp.merge_lse_partials(outputs[rank], lses[rank])


def _jax_merge_worker(payload, queue):
    """Body of `_jax_merge`, run in its own process. Top-level, so it pickles."""
    outputs, lses = payload
    try:
        import jax.numpy as jnp

        from vllm_torchtpu.layers.common import attention_interface
        queue.put(("OK",
                   np.asarray(
                       attention_interface.merge_dcp_lse_partials(
                           jnp.asarray(outputs), jnp.asarray(lses)))))
    except Exception:  # noqa: BLE001 - re-raised in the parent
        import traceback
        queue.put(("ERROR", traceback.format_exc()))


def _jax_merge(outputs, lses):
    """`merge_dcp_lse_partials` (the JAX twin), evaluated out of process.

    Spawned rather than called inline because importing the JAX side here
    initializes a PJRT client and takes the libtpu lockfile for the whole
    pytest process. `test_kv_scatter.py` sorts after this module and runs its
    own cases in spawned subprocesses, which then cannot get a client at all --
    so an inline call turns three unrelated tests red. Pinning JAX to the CPU
    backend does not avoid it; only keeping the parent clean does.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_jax_merge_worker,
                          args=((outputs.numpy(), lses.numpy()), queue))
    process.start()
    result = queue.get()
    process.join()
    status, payload = result
    if status == "ERROR":
        raise AssertionError(f"JAX reference merge failed:\n{payload}")
    return payload


def _partials(rng, dcp_size, tokens=8, heads=4, vdim=16, empty=()):
    outputs = torch.from_numpy(
        rng.standard_normal(
            (dcp_size, tokens, heads, vdim)).astype(np.float32))
    lses = torch.from_numpy((rng.standard_normal(
        (dcp_size, tokens, heads)) * 4).astype(np.float32))
    for rank, token in empty:
        lses[rank, token] = float("-inf")
    return outputs, lses


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
def test_torch_merge_matches_the_jax_merge(dcp_size, monkeypatch):
    outputs, lses = _partials(np.random.default_rng(20260907), dcp_size)
    got = _merge_as_rank(outputs, lses, 0, monkeypatch).numpy()
    np.testing.assert_allclose(got,
                               _jax_merge(outputs, lses),
                               rtol=1e-5,
                               atol=1e-5)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_every_rank_computes_the_same_answer(rank, monkeypatch):
    """All-reduce means the result is replicated; the rank index must only
    select which weight this rank contributes, never change the total."""
    outputs, lses = _partials(np.random.default_rng(11), 4)
    got = _merge_as_rank(outputs, lses, rank, monkeypatch).numpy()
    ref = _merge_as_rank(outputs, lses, 0, monkeypatch).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_ranks_owning_nothing_are_excluded(monkeypatch):
    """A `-inf` rank must contribute nothing -- not a small weight."""
    rng = np.random.default_rng(5)
    outputs, lses = _partials(rng, 4, empty=[(1, 3), (2, 3), (3, 3)])
    got = _merge_as_rank(outputs, lses, 0, monkeypatch).numpy()
    # Token 3 is owned by rank 0 alone, so the merge must return rank 0's
    # output there verbatim.
    np.testing.assert_allclose(got[3],
                               outputs[0, 3].numpy(),
                               rtol=1e-6,
                               atol=1e-6)


def test_a_token_no_rank_owns_is_zero_not_nan(monkeypatch):
    """Padding rows are `-inf` everywhere; `inf - inf` must not leak out."""
    rng = np.random.default_rng(6)
    outputs, lses = _partials(rng, 4, empty=[(r, 7) for r in range(4)])
    got = _merge_as_rank(outputs, lses, 0, monkeypatch).numpy()
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got[7], np.zeros_like(got[7]))


def test_merge_is_a_noop_without_a_group(monkeypatch):
    """Single-rank and pre-init paths must not pay for, or require, DCP."""
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    out = torch.ones(2, 3, 4)
    assert dcp.merge_lse_partials(out, torch.zeros(2, 3)) is out


class _HeadGatherGroup:
    """Serves only `all_gather(dim=1)`, from a stack of per-rank head slices."""

    def __init__(self, per_rank, rank):
        self.world_size = len(per_rank)
        self.rank_in_group = rank
        self._per_rank = per_rank

    def all_gather(self, tensor, dim=0):
        assert dim == 1
        assert tensor.shape == self._per_rank[self.rank_in_group].shape
        return torch.cat(list(self._per_rank), dim=1)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_head_gather_puts_this_ranks_heads_in_its_own_block(rank, monkeypatch):
    """The gather/scatter pair only lines up if block `r` belongs to rank `r`.

    `merge_lse_partials_scatter_heads` hands block `r` back to rank `r`, and
    `W_UV` downstream is that rank's TP head slice, so anything else silently
    applies the wrong up-projection.
    """
    rng = np.random.default_rng(31)
    per_rank = torch.from_numpy(
        rng.standard_normal((4, 6, 3, 8)).astype(np.float32))
    monkeypatch.setattr(dcp, "get_dcp_group",
                        lambda: _HeadGatherGroup(per_rank, rank))

    got = dcp.all_gather_heads(per_rank[rank])
    assert got.shape == (6, 12, 8)
    np.testing.assert_array_equal(got[:, rank * 3:(rank + 1) * 3].numpy(),
                                  per_rank[rank].numpy())


def test_head_gather_is_a_noop_without_a_group(monkeypatch):
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    q = torch.ones(2, 3, 4)
    assert dcp.all_gather_heads(q) is q


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_scatter_merge_is_the_all_reduce_merge_narrowed(rank, monkeypatch):
    """The two merges must agree; only the head axis they leave behind differs.

    Every rank attended with all `dcp_size * heads` heads, so the whole-context
    answer is the same tensor either way -- the scatter just drops the
    `dcp_size - 1` head blocks whose `W_UV` lives on another rank.
    """
    dcp_size, local_heads = 4, 3
    outputs, lses = _partials(np.random.default_rng(77),
                              dcp_size,
                              heads=dcp_size * local_heads)
    group = _FakeGroup(outputs, lses, rank)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)

    whole = dcp.merge_lse_partials(outputs[rank], lses[rank])
    mine = dcp.merge_lse_partials_scatter_heads(outputs[rank], lses[rank])

    assert mine.shape == (outputs.shape[1], local_heads, outputs.shape[3])
    np.testing.assert_allclose(mine.numpy(),
                               whole[:, rank * local_heads:(rank + 1) *
                                     local_heads].numpy(),
                               rtol=1e-6,
                               atol=1e-6)


def test_scatter_merge_is_a_noop_without_a_group(monkeypatch):
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: None)
    out = torch.ones(2, 3, 4)
    assert dcp.merge_lse_partials_scatter_heads(out, torch.zeros(2, 3)) is out


def test_merge_reproduces_an_unsplit_softmax(monkeypatch):
    """Ground truth, independent of both implementations.

    Split one attention problem's keys across ranks, attend per shard, merge:
    the result must equal the softmax over all the keys at once.
    """
    rng = np.random.default_rng(99)
    dcp_size, tokens, heads, vdim, keys = 4, 6, 3, 8, 40
    q = rng.standard_normal((tokens, heads, vdim))
    k = rng.standard_normal((keys, vdim))
    v = rng.standard_normal((keys, vdim))
    scores = np.einsum("tnd,kd->tnk", q, k)

    whole = np.exp(scores - scores.max(-1, keepdims=True))
    whole /= whole.sum(-1, keepdims=True)
    expected = np.einsum("tnk,kd->tnd", whole, v)

    outs = np.zeros((dcp_size, tokens, heads, vdim))
    lses = np.zeros((dcp_size, tokens, heads))
    for r in range(dcp_size):
        s = scores[..., r::dcp_size]
        m = s.max(-1)
        e = np.exp(s - m[..., None])
        lses[r] = m + np.log(e.sum(-1))
        outs[r] = np.einsum("tnk,kd->tnd", e / e.sum(-1, keepdims=True),
                            v[r::dcp_size])

    got = _merge_as_rank(
        torch.from_numpy(outs).float(),
        torch.from_numpy(lses).float(), 0, monkeypatch).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
