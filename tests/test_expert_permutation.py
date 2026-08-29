# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the static expert-relabelling path.

``expert_permutation`` claims that relabelling expert ids is a *placement*
change and not a numerics change: physical expert ``g`` moves to id ``pi[g]``
and gate row ``pi[g]`` takes over expert ``g``'s row, so the routing decision
is unchanged. These tests hold that claim to the routing code that actually
consumes the gate -- ``moe_routing.select_experts`` -- rather than restating
the argument.

Everything here runs on CPU. The permutation is applied at weight load, so no
kernel, no TPU and no expert weights are involved; the only tensors that move
are the replicated router gate row and the per-expert routing bias.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from vllm_torchtpu import expert_permutation
from vllm_torchtpu.layers.vllm import moe_routing

NUM_EXPERTS = 256
HIDDEN = 128
TOPK = 8
FIRST_MOE_LAYER = 3
NUM_LAYERS = 8
EP_SIZE = 16
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE


class _Gate(nn.Module):
    """Stand-in for vLLM's ``GateLinear`` under ``noaux_tc`` routing.

    ``deepseek_v2.py`` attaches ``e_score_correction_bias`` to the gate module
    and hands *the same Parameter object* to the experts layer, which is why
    the permutation has to mutate ``.data`` rather than rebind. The tests below
    rely on that aliasing, so it is reproduced faithfully.
    """

    def __init__(self, generator: torch.Generator, with_score_bias: bool):
        super().__init__()
        # 1/sqrt(HIDDEN) keeps the logits at unit scale. Unscaled randn rows
        # give logits with sigma ~ sqrt(HIDDEN), whose sigmoid saturates to
        # exactly 1.0 for most experts -- the top-k would then be decided
        # entirely by tie-breaking on expert id, which is the one thing a
        # relabelling is *not* invariant to, and the test would be measuring
        # the fixture rather than the code.
        self.weight = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, generator=generator) /
            HIDDEN**0.5)
        self.e_score_correction_bias = nn.Parameter(
            torch.randn(NUM_EXPERTS, generator=generator).float()
        ) if with_score_bias else None


class _MoE(nn.Module):

    def __init__(self, generator: torch.Generator, with_score_bias: bool):
        super().__init__()
        self.gate = _Gate(generator, with_score_bias)


class _Decoder(nn.Module):

    def __init__(self, generator: torch.Generator, is_moe: bool,
                 with_score_bias: bool):
        super().__init__()
        self.mlp = (_MoE(generator, with_score_bias)
                    if is_moe else nn.Linear(HIDDEN, HIDDEN))


class _Inner(nn.Module):

    def __init__(self, generator: torch.Generator, with_score_bias: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            _Decoder(generator, i >= FIRST_MOE_LAYER, with_score_bias)
            for i in range(NUM_LAYERS)
        ])


class _Model(nn.Module):
    """Wrapped one level deep, exactly as the real model is.

    ``_LAYER_RE`` matches ``\\.layers\\.(\\d+)\\.``, with a *leading* dot, and
    ``named_modules()`` on the root emits no leading dot. In production the
    names read ``model.layers.3.mlp.gate`` and the dot comes from this wrapper;
    a flatter fixture would silently match nothing and the test would pass for
    the wrong reason.
    """

    def __init__(self, seed: int = 0, with_score_bias: bool = True):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.model = _Inner(generator, with_score_bias)


class _RoutingLayer:
    """The attribute surface ``select_experts`` reads off the experts layer."""

    def __init__(self, e_score_correction_bias):
        self.e_score_correction_bias = e_score_correction_bias
        self.hash_indices_table = None
        # GLM-5.2 has n_group = topk_group = 1, so grouped selection is off and
        # the ungrouped noaux_tc branch is the live one.
        self.use_grouped_topk = True
        self.num_expert_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 2.5


def _moe_layer_indices() -> list[int]:
    return list(range(FIRST_MOE_LAYER, NUM_LAYERS))


def _write_perm(tmp_path, perms: dict[int, np.ndarray]) -> str:
    path = tmp_path / "perm.json"
    path.write_text(
        json.dumps({
            "perm": {str(k): v.tolist()
                     for k, v in perms.items()},
            "note": "unit test",
        }))
    return str(path)


def _install(monkeypatch, uri: str) -> None:
    """Point the module at ``uri`` and clear its process-wide perm cache."""
    monkeypatch.setattr(expert_permutation, "uri", lambda: uri)
    monkeypatch.setattr(expert_permutation, "_CACHE", {
        "loaded": False,
        "perm": None,
        "meta": None
    })


def _identity_perms() -> dict[int, np.ndarray]:
    return {
        i: np.arange(NUM_EXPERTS, dtype=np.int64)
        for i in _moe_layer_indices()
    }


def _random_perms(seed: int = 7) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        i: rng.permutation(NUM_EXPERTS).astype(np.int64)
        for i in _moe_layer_indices()
    }


def _route(gate: _Gate, hidden: torch.Tensor):
    router_logits = hidden @ gate.weight.t()
    layer = _RoutingLayer(gate.e_score_correction_bias)
    return moe_routing.select_experts(
        hidden,
        router_logits,
        topk=TOPK,
        renormalize=True,
        scoring_fn="sigmoid",
        layer=layer,
    )


def _canonical(topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """Sort each row's (id, weight) pairs by id.

    The top-k *set* is what routing decides; the order within a row is an
    implementation detail of the top-k kernel and a relabelling can permute it
    without changing which expert gets which weight.
    """
    order = torch.argsort(topk_ids.long(), dim=-1, stable=True)
    return (torch.gather(topk_ids.long(), -1, order),
            torch.gather(topk_weights, -1, order))


@pytest.fixture
def hidden() -> torch.Tensor:
    return torch.randn(64,
                       HIDDEN,
                       generator=torch.Generator().manual_seed(1234))


@pytest.mark.parametrize("with_score_bias", [True, False])
def test_fixture_is_not_tie_dominated(hidden, with_score_bias):
    """Guard the guards: ties make every test below equivariant for free.

    ``rowmax_topk`` breaks ties towards the lowest expert id, so a fixture
    whose scores collapse onto one value would select ids 0..k-1 before and
    after any permutation and every assertion here would pass vacuously.
    """
    gate = _Model(with_score_bias=with_score_bias).model.layers[
        FIRST_MOE_LAYER].mlp.gate
    weights, _ = _route(gate, hidden)
    distinct = weights.unique().numel()
    assert distinct > weights.shape[0], (
        f"only {distinct} distinct routed weights across {weights.shape[0]} "
        "rows -- the scores are saturated and the top-k is decided by "
        "tie-breaking on expert id")


def test_identity_permutation_is_a_literal_noop(tmp_path, monkeypatch, hidden):
    """pi = identity must not move a single byte."""
    model = _Model()
    gate = model.model.layers[FIRST_MOE_LAYER].mlp.gate
    before_w = gate.weight.detach().clone()
    before_b = gate.e_score_correction_bias.detach().clone()
    ref_weights, ref_ids = _route(gate, hidden)

    _install(monkeypatch, _write_perm(tmp_path, _identity_perms()))
    expert_permutation.permute_router_gates(model)

    assert torch.equal(gate.weight, before_w)
    assert torch.equal(gate.e_score_correction_bias, before_b)
    got_weights, got_ids = _route(gate, hidden)
    assert torch.equal(got_ids, ref_ids)
    assert torch.equal(got_weights, ref_weights)


def test_permutation_preserves_the_routing_decision(tmp_path, monkeypatch,
                                                    hidden):
    """The relabelled model must pick the same experts with the same weights.

    Physical expert ``g`` answers to id ``pi[g]`` afterwards, so the ids are
    compared through ``pi`` while the weights are compared directly.
    """
    model = _Model()
    perms = _random_perms()

    refs = {}
    for layer_idx in _moe_layer_indices():
        ref_weights, ref_ids = _route(model.model.layers[layer_idx].mlp.gate,
                                      hidden)
        refs[layer_idx] = (ref_weights.clone(), ref_ids.clone())

    _install(monkeypatch, _write_perm(tmp_path, perms))
    expert_permutation.permute_router_gates(model)

    for layer_idx in _moe_layer_indices():
        ref_weights, ref_ids = refs[layer_idx]
        pi = torch.from_numpy(perms[layer_idx])
        expected_ids = pi[ref_ids.long()]
        got_weights, got_ids = _route(
            model.model.layers[layer_idx].mlp.gate, hidden)

        exp_ids_c, exp_w_c = _canonical(ref_weights, expected_ids)
        got_ids_c, got_w_c = _canonical(got_weights, got_ids)
        assert torch.equal(got_ids_c, exp_ids_c), f"layer {layer_idx}"
        assert torch.equal(got_w_c, exp_w_c), f"layer {layer_idx}"


def test_score_bias_must_be_permuted_too(tmp_path, monkeypatch, hidden):
    """Regression guard: permuting the gate weight alone changes routing.

    ``noaux_tc`` adds ``e_score_correction_bias`` to the scores *before* top-k.
    Leaving it behind is not a cosmetic omission -- it selects different
    experts -- so this test fails loudly if the bias permute is ever dropped.
    """
    model = _Model()
    gate = model.model.layers[FIRST_MOE_LAYER].mlp.gate
    perms = _random_perms()
    pi = torch.from_numpy(perms[FIRST_MOE_LAYER])
    inverse = torch.from_numpy(np.argsort(perms[FIRST_MOE_LAYER]))

    ref_weights, ref_ids = _route(gate, hidden)

    # The pre-fix behaviour: weight only.
    with torch.no_grad():
        gate.weight.data = gate.weight.data.index_select(0,
                                                         inverse).contiguous()
    bad_weights, bad_ids = _route(gate, hidden)
    exp_ids_c, exp_w_c = _canonical(ref_weights, pi[ref_ids.long()])
    bad_ids_c, bad_w_c = _canonical(bad_weights, bad_ids)
    assert not (torch.equal(bad_ids_c, exp_ids_c)
                and torch.equal(bad_w_c, exp_w_c)), (
                    "permuting the gate weight without e_score_correction_bias "
                    "happened to be a no-op; the test fixture is degenerate")

    # Adding the bias permute restores the identity.
    with torch.no_grad():
        gate.e_score_correction_bias.data = (
            gate.e_score_correction_bias.data.index_select(
                0, inverse).contiguous())
    fixed_weights, fixed_ids = _route(gate, hidden)
    fixed_ids_c, fixed_w_c = _canonical(fixed_weights, fixed_ids)
    assert torch.equal(fixed_ids_c, exp_ids_c)
    assert torch.equal(fixed_w_c, exp_w_c)


def test_score_bias_alias_survives_the_permute(tmp_path, monkeypatch):
    """The experts layer's captured reference must see the permuted values.

    ``routed_experts.py`` stores the gate's Parameter object itself, so a
    rebind on the gate would leave the experts layer reading stale data. Only
    a ``.data`` mutation is visible through the alias.
    """
    model = _Model()
    gate = model.model.layers[FIRST_MOE_LAYER].mlp.gate
    alias = gate.e_score_correction_bias  # what FusedMoEFactory captures
    before = alias.detach().clone()

    _install(monkeypatch, _write_perm(tmp_path, _random_perms()))
    expert_permutation.permute_router_gates(model)

    assert alias is gate.e_score_correction_bias
    assert not torch.equal(alias.detach(), before)
    assert torch.equal(alias.detach(), gate.e_score_correction_bias.detach())


def test_half_permuted_model_is_refused(tmp_path, monkeypatch):
    """A perm file covering layers that do not exist must not run silently."""
    perms = _random_perms()
    perms[NUM_LAYERS + 5] = np.arange(NUM_EXPERTS, dtype=np.int64)
    _install(monkeypatch, _write_perm(tmp_path, perms))
    with pytest.raises(ValueError, match="half-permuted"):
        expert_permutation.permute_router_gates(_Model())


def test_mismatched_score_bias_width_is_refused(tmp_path, monkeypatch):
    model = _Model()
    gate = model.model.layers[FIRST_MOE_LAYER].mlp.gate
    gate.e_score_correction_bias = nn.Parameter(torch.zeros(NUM_EXPERTS - 1))
    _install(monkeypatch, _write_perm(tmp_path, _random_perms()))
    with pytest.raises(ValueError, match="e_score_correction_bias"):
        expert_permutation.permute_router_gates(model)


def test_gate_without_score_bias_still_permutes(tmp_path, monkeypatch, hidden):
    """Models with no ``e_score_correction_bias`` must be unaffected."""
    model = _Model(with_score_bias=False)
    gate = model.model.layers[FIRST_MOE_LAYER].mlp.gate
    ref_weights, ref_ids = _route(gate, hidden)
    perms = _random_perms()
    pi = torch.from_numpy(perms[FIRST_MOE_LAYER])

    _install(monkeypatch, _write_perm(tmp_path, perms))
    expert_permutation.permute_router_gates(model)

    got_weights, got_ids = _route(gate, hidden)
    exp_ids_c, exp_w_c = _canonical(ref_weights, pi[ref_ids.long()])
    got_ids_c, got_w_c = _canonical(got_weights, got_ids)
    assert torch.equal(got_ids_c, exp_ids_c)
    assert torch.equal(got_w_c, exp_w_c)


def test_expert_map_and_gate_relabel_agree():
    """The two halves of the relabelling must cancel.

    ``permuted_expert_map`` puts physical ``g`` in slot ``pi[g] % local`` on
    rank ``pi[g] // local``; the kernel reaches logit index ``n`` by taking
    slot ``n - local * rank`` on rank ``n // local``. Walking both directions
    is what makes "the two moves cancel" checkable rather than asserted.
    """
    perm = _random_perms()[FIRST_MOE_LAYER]
    maps = [
        expert_permutation.permuted_expert_map(perm, LOCAL_EXPERTS, r).numpy()
        for r in range(EP_SIZE)
    ]
    for rank, m in enumerate(maps):
        assert int((m >= 0).sum()) == LOCAL_EXPERTS

    for n in range(NUM_EXPERTS):
        rank = n // LOCAL_EXPERTS
        slot = n - LOCAL_EXPERTS * rank
        owners = np.flatnonzero(maps[rank] == slot)
        assert owners.shape[0] == 1, f"logit {n} resolves to {owners.tolist()}"
        # The expert now serving logit index n is pi^-1[n], which is exactly
        # the expert whose gate row permute_router_gates moved to row n.
        assert owners[0] == int(np.argsort(perm)[n])
