# SPDX-License-Identifier: Apache-2.0
"""Load-time selection and forward wiring of the small-batch MoE scores entry."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores as scores

pytestmark = pytest.mark.cpu_test


def make_layer(experts=64, hidden=512, inter=256, use_ep=True):
    def meta(*shape, dtype=torch.float8_e4m3fn):
        return torch.empty(shape, dtype=dtype, device="meta")

    return SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        hidden_size=hidden,
        global_num_experts=512,
        renormalize=True,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        apply_router_weight_on_input=False,
        moe_config=SimpleNamespace(
            experts_per_token=10,
            in_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ),
        w13_weight=meta(experts, hidden, 2 * inter),
        w2_weight=meta(experts, inter, hidden),
        w13_weight_scale_inv=meta(experts, 1, 1, 2 * inter, dtype=torch.float32),
        w2_weight_scale_inv=meta(experts, 1, 1, hidden, dtype=torch.float32),
        _experts_start=torch.tensor(192, dtype=torch.int32),
    )


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK", True, raising=False)
    monkeypatch.setattr(
        scores, "enable_pipelined_collective_and_compute", lambda: False
    )
    op = Mock(name="op")
    monkeypatch.setattr(scores, "_build_op", Mock(return_value=op))
    return op


def test_disabled_builds_nothing(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK", False, raising=False)
    assert scores.prebuild(make_layer(use_ep=False), "gelu") is None


def test_plain_softmax_router_moves_into_the_kernel(enabled):
    assert scores.prebuild(make_layer(), "silu") is enabled
    scores._build_op.assert_called_once_with(10)


@pytest.mark.parametrize(
    "layer, activation, match",
    [
        (make_layer(use_ep=False), "silu", "expert parallelism"),
        (make_layer(experts=129), "silu", "envelope"),
        (make_layer(inter=128), "silu", "envelope"),
        (make_layer(), "gelu", "envelope"),
    ],
)
def test_layer_the_kernel_cannot_serve_is_refused(enabled, layer, activation, match):
    with pytest.raises(NotImplementedError, match=f"TPU_MOE_LOWTOK.*{match}"):
        scores.prebuild(layer, activation)


@pytest.mark.parametrize(
    "option, value",
    [
        ("custom_routing_function", lambda **kw: None),
        ("scoring_func", "sigmoid"),
        ("renormalize", False),
        ("e_score_correction_bias", torch.zeros(512)),
        ("use_grouped_topk", True),
        ("hash_indices_table", torch.zeros(8, 10)),
        ("routed_scaling_factor", 2.5),
        ("apply_router_weight_on_input", True),
        ("global_num_experts", 256),
    ],
)
def test_routing_the_kernel_does_not_reproduce_keeps_the_ids_entry(
    enabled, option, value
):
    layer = make_layer()
    setattr(layer, option, value)
    if option == "use_grouped_topk":
        layer.num_expert_group, layer.topk_group = 8, 4
    assert scores.prebuild(layer, "silu") is None


def test_chunk_pipelining_keeps_the_ids_entry(enabled, monkeypatch):
    monkeypatch.setattr(scores, "enable_pipelined_collective_and_compute", lambda: True)
    assert scores.prebuild(make_layer(), "silu") is None


@pytest.mark.parametrize("rows, served", [(1, True), (256, True), (257, False)])
def test_row_cap_counts_every_row(rows, served):
    """Under MTP the rows include the draft tokens of every request."""
    x = torch.empty(rows, 8)
    assert scores.serves(Mock(), x) is served
    assert not scores.serves(None, x)


@pytest.mark.parametrize("skip_padded", [False, True])
def test_apply_passes_softmax_scores_and_the_grouped_operand_order(
    monkeypatch, skip_padded
):
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", skip_padded, raising=False)
    padded = torch.tensor([False, True, False])
    monkeypatch.setattr(
        scores.token_padding,
        "zero_routing_weights_for_padding",
        lambda ids, weights: (ids, torch.where(padded[:, None], 0.0, weights)),
    )
    layer = make_layer()
    op = Mock(return_value="out")
    x = torch.randn(3, 512, dtype=torch.bfloat16)
    logits = torch.randn(3, 512, dtype=torch.bfloat16)
    assert scores.apply(op, layer, x, logits) == "out"
    args = op.call_args.args
    want = torch.softmax(logits.float(), dim=-1)
    if skip_padded:
        want[1] = 0
    assert args[0] is x
    torch.testing.assert_close(args[1], want)
    assert args[2:] == (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        layer._experts_start,
    )


@pytest.mark.parametrize("rows, lowtok", [(256, True), (257, False)])
def test_fp8_forward_takes_the_scores_entry_below_the_cap(monkeypatch, rows, lowtok):
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False, raising=False)
    monkeypatch.setattr(fp8, "enable_pipelined_collective_and_compute", lambda: False)
    monkeypatch.setattr(scores, "apply", Mock(return_value="lowtok"))
    monkeypatch.setattr(fp8.moe_routing, "route", Mock(return_value=(None, None)))
    monkeypatch.setattr(fp8, "fused_moe_gmm", Mock(return_value="gmm"))
    method = SimpleNamespace(
        _tpu_activation_str="silu",
        _use_adaptive_fused_moe=False,
        _moe_lowtok_scores_op=Mock(),
    )
    layer = make_layer()
    layer.w13_bias = layer.w2_bias = None
    x = torch.empty(rows, 512)
    out = fp8.VllmFp8MoEMethodTPU.apply_monolithic(
        method, layer, x, torch.empty(rows, 512)
    )
    assert out == ("lowtok" if lowtok else "gmm")


def test_op_is_built_once_per_topk_and_fakes_the_row_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = scores._build_op(10)
    assert scores._build_op(10) is op
    layer = make_layer()
    with FakeTensorMode():
        x = torch.empty(4, 512, dtype=torch.bfloat16)
        w = [
            torch.empty(t.shape, dtype=t.dtype)
            for t in (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
            )
        ]
        out = op(x, torch.empty(4, 512), *w, torch.empty((), dtype=torch.int32))
    assert tuple(out.shape) == (4, 512) and out.dtype == torch.bfloat16
