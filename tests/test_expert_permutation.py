# SPDX-License-Identifier: Apache-2.0
"""Static MoE expert relabelling (expert_permutation.py) at weight load.

CPU only. A relabelled checkpoint must give a model computing the identical
function, each EP rank must fetch exactly the experts its contiguous block
holds after relabelling, and the identity permutation must change nothing.
"""

import json

import numpy as np
import pytest
import torch

from vllm_torchtpu import expert_permutation as ep
from vllm_torchtpu import model_loader_patches

E, H, INTER, TOPK, EP_SIZE = 16, 8, 4, 3, 4


def _checkpoint(layers=(0, 1), seed=0):
    g = torch.Generator().manual_seed(seed)
    ckpt = []
    for layer in layers:
        p = f"layers.{layer}.ffn"
        ckpt.append((f"{p}.gate.weight", torch.randn(E, H, generator=g)))
        ckpt.append((f"{p}.gate.bias", torch.randn(E, generator=g)))
        ckpt.append((f"{p}.gate.bias_vl", torch.randn(E, generator=g)))
        for e in range(E):
            ckpt.append(
                (f"{p}.experts.{e}.w1.weight", torch.randn(INTER, H, generator=g))
            )
            ckpt.append(
                (f"{p}.experts.{e}.w2.weight", torch.randn(H, INTER, generator=g))
            )
            ckpt.append(
                (f"{p}.experts.{e}.w1.scale", torch.randn(INTER, 1, generator=g))
            )
    ckpt.append(("mtp.0.ffn.experts.3.w1.weight", torch.randn(INTER, H, generator=g)))
    ckpt.append(("mtp.0.ffn.gate.weight", torch.randn(E, H, generator=g)))
    return ckpt


def _moe(state, layer, x):
    """sqrt-softplus + bias top-k MoE over a {name: tensor} state, by new id."""
    p = f"layers.{layer}.ffn"
    scores = torch.sqrt(torch.nn.functional.softplus(x @ state[f"{p}.gate.weight"].T))
    _, ids = torch.topk(scores + state[f"{p}.gate.bias"], TOPK, dim=-1)
    w = torch.gather(scores, 1, ids)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for k in range(TOPK):
        for t in range(x.shape[0]):
            e = int(ids[t, k])
            h = torch.relu(x[t] @ state[f"{p}.experts.{e}.w1.weight"].T)
            out[t] += w[t, k] * (h @ state[f"{p}.experts.{e}.w2.weight"].T)
    return out, ids


def _perm(seed):
    return np.random.default_rng(seed).permutation(E)


def test_rejects_a_non_permutation(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"perm": {"0": [0, 1, 1, 3]}}))
    with pytest.raises(ValueError, match="not a permutation"):
        ep.load_permutations(str(path))


def test_identity_permutation_changes_nothing():
    ckpt = _checkpoint()
    out = list(ep.relabel_checkpoint(ckpt, perms={0: np.arange(E), 1: np.arange(E)}))
    assert [n for n, _ in out] == [n for n, _ in ckpt]
    for (_, a), (_, b) in zip(out, ckpt, strict=True):
        assert torch.equal(a, b)


def test_off_passes_the_stream_through():
    ckpt = _checkpoint()
    out = list(ep.relabel_checkpoint(iter(ckpt), perms={}))
    assert all(a is b for (_, a), (_, b) in zip(out, ckpt, strict=True))


def test_relabelled_model_computes_the_same_function():
    perm = _perm(1)
    ckpt = _checkpoint()
    before = dict(ckpt)
    after = dict(ep.relabel_checkpoint(ckpt, perms={0: perm}))
    x = torch.randn(32, H, generator=torch.Generator().manual_seed(7))
    y0, ids0 = _moe(before, 0, x)
    y1, ids1 = _moe(after, 0, x)
    assert torch.equal(y0, y1)
    assert torch.equal(torch.from_numpy(perm)[ids0], ids1)  # same experts, new ids
    # Row perm[g] of every gate tensor holds checkpoint row g.
    for leaf in ("weight", "bias", "bias_vl"):
        new = after[f"layers.0.ffn.gate.{leaf}"]
        old = before[f"layers.0.ffn.gate.{leaf}"]
        assert torch.equal(new[torch.from_numpy(perm)], old)
    # Unlisted layers and the draft (mtp) layers keep their numbering.
    for name, tensor in ckpt:
        if not name.startswith("layers.0."):
            assert torch.equal(after[name], tensor), name


def test_hash_routed_layer_is_refused():
    ckpt = [("layers.0.ffn.gate.tid2eid", torch.zeros(10, TOPK, dtype=torch.int32))]
    with pytest.raises(NotImplementedError):
        list(ep.relabel_checkpoint(ckpt, perms={0: _perm(2)}))
    assert len(list(ep.relabel_checkpoint(ckpt, perms={1: _perm(2)}))) == 1


def test_gate_with_the_wrong_expert_count_is_refused():
    ckpt = [("layers.0.ffn.gate.weight", torch.zeros(E + 1, H))]
    with pytest.raises(ValueError, match="experts"):
        list(ep.relabel_checkpoint(ckpt, perms={0: _perm(3)}))


def test_each_rank_fetches_its_relabelled_block(tmp_path, monkeypatch):
    perm = _perm(4)
    path = tmp_path / "perm.json"
    path.write_text(json.dumps({"perm": {"0": perm.tolist()}}))
    monkeypatch.setenv("TPU_MOE_EXPERT_PERM_PATH", str(path))
    ep._configured.cache_clear()
    try:
        local = E // EP_SIZE
        loaded_by_rank = []
        for rank in range(EP_SIZE):
            ids = set(range(rank * local, (rank + 1) * local))
            kept = [
                e
                for e in range(E)
                if not model_loader_patches._should_skip(
                    f"layers.0.ffn.experts.{e}.w1.weight", ids
                )
            ]
            assert sorted(kept) == sorted(
                np.flatnonzero(perm // local == rank).tolist()
            )
            # The relabelled stream names those experts with ids in the block.
            renamed = dict(
                ep.relabel_checkpoint(
                    [
                        (f"layers.0.ffn.experts.{e}.w1.weight", torch.zeros(1))
                        for e in kept
                    ]
                )
            )
            got = sorted(int(n.split(".")[4]) for n in renamed)
            assert got == sorted(ids)
            loaded_by_rank.append(kept)
            # Unlisted layers keep the linear placement.
            assert not model_loader_patches._should_skip(
                f"layers.1.ffn.experts.{rank * local}.w1.weight", ids
            )
        assert sorted(sum(loaded_by_rank, [])) == list(range(E))
    finally:
        ep._configured.cache_clear()
