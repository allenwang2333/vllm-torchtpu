# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for vllm_torchtpu.layers.vllm.moe_routing helpers."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.vllm import moe_routing


def _make_layer(*, use_ep, ep_size=1, ep_rank=0, global_num_experts=0):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
            ep_size=ep_size,
            ep_rank=ep_rank),
        global_num_experts=global_num_experts,
    )


# (ep_size, global_num_experts) -- mix of divisible and non-divisible cases,
# plus the production sizes we actually ship (Qwen3-30B: 128, Qwen3-480B: 160).
@pytest.mark.parametrize("ep_size,global_num_experts", [
    (2, 8),
    (2, 9),
    (4, 18),
    (4, 33),
    (8, 128),
    (8, 160),
])
def test_get_experts_start_matches_determine_expert_map(
        ep_size, global_num_experts):
    """Parity against vLLM's linear-placement contract for every rank."""
    determine_expert_map = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.expert_map_manager"
    ).determine_expert_map
    for ep_rank in range(ep_size):
        _, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy="linear",
        )
        owned = (expert_map >= 0).nonzero(as_tuple=True)[0]
        expected = int(owned[0].item()) if owned.numel() else 0
        actual = moe_routing.get_experts_start(
            _make_layer(use_ep=True,
                        ep_size=ep_size,
                        ep_rank=ep_rank,
                        global_num_experts=global_num_experts))
        assert actual == expected, (
            f"ep_size={ep_size} ep_rank={ep_rank} "
            f"global_num_experts={global_num_experts}: "
            f"expected experts_start={expected}, got {actual}")


def test_get_experts_start_returns_none_without_ep():
    assert moe_routing.get_experts_start(_make_layer(use_ep=False)) is None


def test_validate_linear_ep_placement_accepts_linear():
    moe_routing.validate_linear_ep_placement(
        SimpleNamespace(expert_placement_strategy="linear"))


def test_validate_linear_ep_placement_requires_declared_strategy():
    # An object that never declares its placement must fail loudly, not be
    # assumed linear.
    with pytest.raises(AttributeError):
        moe_routing.validate_linear_ep_placement(SimpleNamespace())


def test_validate_linear_ep_placement_rejects_round_robin():
    with pytest.raises(NotImplementedError, match="linear EP placement"):
        moe_routing.validate_linear_ep_placement(
            SimpleNamespace(expert_placement_strategy="round_robin"))


class TestRenormalizeDenominator:
    """The fold reassociates, so pin the tolerance it actually holds to.

    It exists to remove a `reduce` HLO whose consumer fusion stalls ~2.8 us on
    TPU (254 us/step across K3's 92 MoE layers, the largest single source of
    within-graph device idle). Halving changes the summation order, and float
    addition is not associative, so this is *not* bit-equal to `.sum()` -- it
    differs by about one ulp.

    That is not a regression against a stricter status quo: XLA reassociates
    reductions freely, so the `.sum()` this replaces never matched CPU torch
    bit-for-bit either. What matters is that the difference stays at ulp scale,
    which these tests pin, and that it vanishes under the bf16 cast the routing
    weights get before use -- the 92-layer probe's end-to-end output in bf16 is
    unchanged.
    """

    @pytest.mark.parametrize("topk", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 31, 32])
    def test_shape_and_ulp_agreement_float32(self, topk):
        import torch
        w = torch.rand(64, topk, dtype=torch.float32) + 0.5
        got = moe_routing._renormalize_denominator(w)
        assert got.shape == (64, 1)
        ref = w.sum(dim=-1, keepdim=True)
        # One float32 ulp is ~1.2e-7 relative; allow a small multiple.
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize("topk", [1, 2, 3, 8, 16])
    def test_exact_for_small_integers(self, topk):
        """Reassociation is exact when every partial sum is representable."""
        import torch
        w = torch.arange(1, 64 * topk + 1,
                         dtype=torch.float32).reshape(64, topk)
        assert torch.equal(moe_routing._renormalize_denominator(w),
                           w.sum(dim=-1, keepdim=True))

    def test_bf16_cast_hides_the_difference(self):
        import torch
        torch.manual_seed(0)
        w = torch.rand(4096, 16, dtype=torch.float32) + 0.5
        fold = w / torch.clamp(moe_routing._renormalize_denominator(w),
                               min=1e-20)
        ref = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-20)
        assert torch.equal(fold.to(torch.bfloat16), ref.to(torch.bfloat16))

    def test_renormalized_weights_sum_to_one(self):
        import torch
        w = torch.rand(3, 16, dtype=torch.float32) + 0.5
        torch.testing.assert_close(
            moe_routing._renormalize(w).sum(dim=-1),
            torch.ones(3, dtype=torch.float32))

    def test_clamp_guards_zero_denominator(self):
        import torch
        out = moe_routing._renormalize(torch.zeros(2, 8, dtype=torch.float32))
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("tokens", [1, 8, 64, 65, 512, 4096])
    def test_token_gate_picks_a_path_and_both_agree(self, monkeypatch, tokens):
        """Above the threshold the fold is bypassed; results agree regardless."""
        import torch
        w = torch.rand(tokens, 16, dtype=torch.float32) + 0.5
        got = moe_routing._renormalize(w)
        ref = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-20)
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=0.0)
        # Above the threshold it must be the plain reduce, bit-for-bit.
        if tokens > moe_routing._RENORMALIZE_FOLD_MAX_TOKENS:
            assert torch.equal(got, ref)

    def test_token_gate_threshold_is_respected(self, monkeypatch):
        """The fold is only reached at or below the configured token count."""
        import torch
        calls = []
        real = moe_routing._renormalize_denominator
        monkeypatch.setattr(moe_routing, "_renormalize_denominator",
                            lambda w: calls.append(w.shape[0]) or real(w))
        monkeypatch.setattr(moe_routing, "_RENORMALIZE_FOLD_MAX_TOKENS", 8)
        for n in (1, 8, 9, 64):
            moe_routing._renormalize(torch.rand(n, 16) + 0.5)
        assert calls == [1, 8]

    @pytest.mark.parametrize("fold", [True, False])
    def test_gate_selects_equivalent_paths(self, monkeypatch, fold):
        import torch
        monkeypatch.setattr(moe_routing, "_RENORMALIZE_FOLD", fold)
        w = torch.rand(5, 16, dtype=torch.float32) + 0.5
        torch.testing.assert_close(
            moe_routing._renormalize(w),
            w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-20),
            rtol=1e-6,
            atol=0.0)
