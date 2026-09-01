# SPDX-License-Identifier: Apache-2.0
"""Coverage for the static MoE expert permutation (`expert_permutation.py`).

The module shipped with no tests at all, which is awkward because it is the
one merged path that rewrites weights: it moves which rank owns which expert
(``permuted_expert_map``) and reorders the router gate rows to match
(``permute_router_gates``). Those two moves are supposed to cancel exactly, so
the model computes the identical function. If they do not cancel -- the
classic failure is reordering the gate by ``pi`` where the algebra calls for
``argsort(pi)`` -- the server still starts, the kernel still runs, every shape
still matches, and the model quietly routes every token to the wrong expert.
``vllm bench serve`` runs with ``--ignore-eos``, so that failure is invisible
in a throughput number.

The load-bearing test is therefore :func:`test_placement_and_gate_reorder_
compose_to_identity`, which checks the composition directly, plus its negative
control immediately below it, which applies the wrong direction and requires
the same assertion to fail.

The remaining tests pin the inertness the merge's attribution rests on: with
``TPU_MOE_EXPERT_PERM_URI`` unset, both public entry points must not touch
anything.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_torchtpu import expert_permutation

NUM_EXPERTS = 32
EP_SIZE = 4
LOCAL = NUM_EXPERTS // EP_SIZE


@pytest.fixture(autouse=True)
def _clear_perm_cache():
    """`_perms()` memoizes on first read; every test needs a clean slate."""
    expert_permutation._CACHE.update({
        "loaded": False,
        "perm": None,
        "meta": None
    })
    yield
    expert_permutation._CACHE.update({
        "loaded": False,
        "perm": None,
        "meta": None
    })


def _perm_uri(tmp_path, perm_by_layer, **meta):
    path = tmp_path / "perm.json"
    doc = dict(meta)
    doc["perm"] = {str(k): list(map(int, v)) for k, v in perm_by_layer.items()}
    path.write_text(json.dumps(doc))
    return patch.dict(os.environ, {"TPU_MOE_EXPERT_PERM_URI": str(path)})


def _a_real_permutation(num_experts=NUM_EXPERTS, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_experts)
    assert not np.array_equal(perm, np.arange(num_experts))
    return perm


class _Gate(torch.nn.Module):

    def __init__(self, num_experts, hidden=8):
        super().__init__()
        # Row g is filled with the value g, so a reordered gate can be read
        # back as "which physical expert's row now sits here".
        w = torch.arange(num_experts, dtype=torch.float32)
        self.weight = torch.nn.Parameter(
            w[:, None].repeat(1, hidden).contiguous())
        self.bias = torch.nn.Parameter(w.clone())


class _Model(torch.nn.Module):
    """Minimal stand-in exposing `model.layers.<i>.mlp.gate`."""

    def __init__(self, num_layers, num_experts=NUM_EXPERTS):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.Module()
            mlp = torch.nn.Module()
            mlp.gate = _Gate(num_experts)
            layer.mlp = mlp
            self.layers.append(layer)


def _gate_rows(model, layer_idx):
    """Physical expert id sitting at each gate row, as an int array."""
    gate = model.layers[layer_idx].mlp.gate
    return gate.weight.detach()[:, 0].to(torch.int64).numpy()


# --------------------------------------------------------------------------
# permuted_expert_map
# --------------------------------------------------------------------------


def test_identity_permutation_reproduces_the_contiguous_block_map():
    """pi = identity must give back the stock contiguous-block placement."""
    identity = np.arange(NUM_EXPERTS)
    for ep_rank in range(EP_SIZE):
        got = expert_permutation.permuted_expert_map(identity, LOCAL,
                                                     ep_rank).numpy()
        expected = np.full(NUM_EXPERTS, -1, dtype=np.int32)
        lo = ep_rank * LOCAL
        expected[lo:lo + LOCAL] = np.arange(LOCAL)
        np.testing.assert_array_equal(got, expected)


def test_every_rank_owns_exactly_local_num_experts():
    perm = _a_real_permutation()
    owned = np.zeros(NUM_EXPERTS, dtype=int)
    for ep_rank in range(EP_SIZE):
        m = expert_permutation.permuted_expert_map(perm, LOCAL,
                                                   ep_rank).numpy()
        assert int((m >= 0).sum()) == LOCAL
        owned += (m >= 0)
    # Each physical expert is owned by exactly one rank: a partition, not a
    # replication and not a drop.
    np.testing.assert_array_equal(owned, np.ones(NUM_EXPERTS, dtype=int))


def test_non_partitioning_map_is_rejected():
    """A 'permutation' that overloads a rank must raise, not silently pad."""
    bad = np.zeros(NUM_EXPERTS, dtype=np.int64)  # everything onto rank 0 slot 0
    with pytest.raises(ValueError, match="expected"):
        expert_permutation.permuted_expert_map(bad, LOCAL, 0)


# --------------------------------------------------------------------------
# The invariant the whole optimization rests on
# --------------------------------------------------------------------------


def _composition_holds(perm, gate_rows):
    """For every logit index n, does the kernel reach the gate's expert?

    The kernel sends logit index ``n`` to rank ``n // LOCAL``, local slot
    ``n % LOCAL``. Whichever physical expert the permuted map placed there
    must be the physical expert whose gate row now sits at index ``n``.
    """
    mismatches = []
    for n in range(NUM_EXPERTS):
        rank, slot = n // LOCAL, n % LOCAL
        m = expert_permutation.permuted_expert_map(perm, LOCAL, rank).numpy()
        (physical, ) = np.nonzero(m == slot)
        if int(physical[0]) != int(gate_rows[n]):
            mismatches.append((n, int(physical[0]), int(gate_rows[n])))
    return mismatches


def test_placement_and_gate_reorder_compose_to_identity(tmp_path):
    """The two moves must cancel: same function, different placement."""
    perm = _a_real_permutation()
    model = _Model(num_layers=1)
    with _perm_uri(tmp_path, {0: perm}):
        expert_permutation.permute_router_gates(model)

    rows = _gate_rows(model, 0)
    # The gate really was reordered -- otherwise this test would pass for the
    # trivial reason that nothing happened.
    assert not np.array_equal(rows, np.arange(NUM_EXPERTS))

    mismatches = _composition_holds(perm, rows)
    assert not mismatches, (
        f"{len(mismatches)} logit indices reach the wrong expert; "
        f"first few (n, placed_physical, gate_row_physical): {mismatches[:5]}")


def test_composition_check_fails_for_the_wrong_reorder_direction():
    """Negative control: the check above must be able to go red.

    Reordering by ``pi`` instead of ``argsort(pi)`` is the realistic bug. If
    this test ever passes, `test_placement_and_gate_reorder_compose_to_
    identity` is vacuous and proves nothing.
    """
    perm = _a_real_permutation()
    wrong_rows = np.arange(NUM_EXPERTS)[perm]  # forward, not inverse
    mismatches = _composition_holds(perm, wrong_rows)
    assert mismatches, (
        "applying the permutation in the wrong direction still satisfied the "
        "composition check -- the check does not discriminate")


def test_identity_permutation_is_a_bit_exact_noop_on_the_gate(tmp_path):
    """The module's own documented smoke test, as an assertion."""
    identity = np.arange(NUM_EXPERTS)
    model = _Model(num_layers=2)
    before = [
        model.layers[i].mlp.gate.weight.detach().clone() for i in range(2)
    ]
    with _perm_uri(tmp_path, {0: identity, 1: identity}, identity=True):
        expert_permutation.permute_router_gates(model)
    for i in range(2):
        assert torch.equal(model.layers[i].mlp.gate.weight.detach(), before[i])


def test_gate_bias_is_reordered_with_the_weight(tmp_path):
    perm = _a_real_permutation()
    model = _Model(num_layers=1)
    with _perm_uri(tmp_path, {0: perm}):
        expert_permutation.permute_router_gates(model)
    gate = model.layers[0].mlp.gate
    np.testing.assert_array_equal(
        gate.bias.detach().to(torch.int64).numpy(), _gate_rows(model, 0))


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_layer_count_mismatch_refuses_to_run_half_permuted(tmp_path):
    """A perm file covering more layers than the model has must raise.

    This is the guard that protects against pointing a 60-layer permutation at
    a model whose first block is dense.
    """
    perm = _a_real_permutation()
    model = _Model(num_layers=2)
    with _perm_uri(tmp_path, {0: perm, 1: perm, 2: perm}):
        with pytest.raises(ValueError, match="half-permuted"):
            expert_permutation.permute_router_gates(model)


def test_expert_count_mismatch_raises(tmp_path):
    model = _Model(num_layers=1)
    with _perm_uri(tmp_path, {0: _a_real_permutation(num_experts=8)}):
        with pytest.raises(ValueError, match="rows but perm"):
            expert_permutation.permute_router_gates(model)


def test_a_non_permutation_is_rejected_at_load(tmp_path):
    model = _Model(num_layers=1)
    duplicated = [0] * NUM_EXPERTS
    with _perm_uri(tmp_path, {0: duplicated}):
        with pytest.raises(ValueError, match="not a permutation"):
            expert_permutation.permute_router_gates(model)


# --------------------------------------------------------------------------
# Q1: inertness with the flag unset
# --------------------------------------------------------------------------


def test_unset_uri_leaves_the_gate_untouched():
    model = _Model(num_layers=2)
    before = [
        model.layers[i].mlp.gate.weight.detach().clone() for i in range(2)
    ]
    with patch.dict(os.environ, {"TPU_MOE_EXPERT_PERM_URI": ""}):
        assert expert_permutation.enabled() is False
        expert_permutation.permute_router_gates(model)
    for i in range(2):
        assert torch.equal(model.layers[i].mlp.gate.weight.detach(), before[i])
    # And it must not even have read the (nonexistent) file.
    assert expert_permutation._CACHE["perm"] is None


def test_unset_uri_does_not_patch_routed_experts():
    """`patch_routed_experts` is called unconditionally from tpu_platform."""
    from vllm.model_executor.layers.fused_moe.routed_experts import \
        RoutedExperts
    original = RoutedExperts.update_expert_map_info
    with patch.dict(os.environ, {"TPU_MOE_EXPERT_PERM_URI": ""}):
        expert_permutation.patch_routed_experts()
    assert RoutedExperts.update_expert_map_info is original
    assert not getattr(RoutedExperts, "_tpu_expert_perm_patch", False)


# --------------------------------------------------------------------------
# The artifacts this campaign will actually run with
# --------------------------------------------------------------------------

_ARTIFACTS = {
    "real":
    "gs://vllm-opt-experiments/_artifacts/perm_real_20260828_074300.json",
    "identity":
    "gs://vllm-opt-experiments/_artifacts/perm_identity_20260828_094500.json",
}


@pytest.mark.parametrize("which", sorted(_ARTIFACTS))
def test_campaign_artifact_is_structurally_valid(which):
    """Validate the real GCS artifacts against the loader's own rules.

    Skips rather than fails when GCS is unreachable: an offline pod should not
    turn this file red.
    """
    uri = _ARTIFACTS[which]
    with patch.dict(os.environ, {"TPU_MOE_EXPERT_PERM_URI": uri}):
        try:
            perms = expert_permutation._perms()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"cannot read {uri}: {exc!r}")

    assert perms is not None and len(perms) == 60
    assert sorted(perms) == list(range(60))
    meta = expert_permutation._CACHE["meta"]
    assert meta["num_experts"] == 512
    ep_size = meta["ep_size"]
    local = 512 // ep_size
    for idx, perm in perms.items():
        assert perm.shape[0] == 512, idx
        # `_perms` already checks bijectivity; this checks the placement
        # constraint `permuted_expert_map` will enforce at load time, for
        # every rank, so a bad file fails here and not 40 minutes into a job.
        for ep_rank in range(ep_size):
            expert_permutation.permuted_expert_map(perm, local, ep_rank)

    is_identity = all(
        np.array_equal(p, np.arange(512)) for p in perms.values())
    assert is_identity == bool(meta.get("identity", False)), (
        f"{which}: meta says identity={meta.get('identity')} but the arrays "
        f"say {is_identity}")
