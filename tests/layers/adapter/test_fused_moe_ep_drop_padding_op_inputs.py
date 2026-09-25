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
"""MOE_FUSED_EP_V2_DROP_PADDING adds one operand; build and call must agree.

The fused EP op's operand count is decided at BUILD (`_build_op(...,
drop_padding=)` picks a registered function with a trailing `valid` operand
and one more partition spec) and read back at CALL (`fused_moe_ep` appends
`~padding_mask` only for an op recorded in `_DROP_PADDING_OPS`). A mismatch
is a torch wrong-number-of-arguments error at the first trace, after a server start of most of
an hour, so the contract is pinned here: the drop-padding op takes EXACTLY
ONE operand more than the op without it, for either score-bias setting, and
the call site passes exactly the count the op was built with. Both counts
are read out of the code under test -- the registered function's signature
and the partition-spec tuple -- so no 7/8/9 is written down here.

NO TPU. Everything under test is decided before device work: `sharded_jax_op`
is replaced by a recorder, the kernel by a mock, and the call site is driven
with CPU torch tensors.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter import token_padding

_STUB_EP = 2
_TOKENS = 8
_REAL = 3


class _RecordedOp:
    """Stands in for the torch custom op `sharded_jax_op` registers."""

    def __init__(
        self, name, fn, *, mesh, input_partition_specs, output_partition_specs
    ):
        self.name = name
        self.fn = fn
        self.input_partition_specs = tuple(input_partition_specs)
        self.calls = []

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0]

    @property
    def built_operands(self) -> int:
        return len(inspect.signature(self.fn).parameters)

    @property
    def called_operands(self) -> int:
        args, kwargs = self.calls[-1]
        assert not kwargs, f"keyword operands {sorted(kwargs)}"
        return len(args)


@pytest.fixture
def recorder(monkeypatch):
    """Fresh op caches (ids in `_DROP_PADDING_OPS` must not outlive their op)
    and a recording `sharded_jax_op`."""
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(bridge, "_DROP_PADDING_OPS", set())
    built = []

    def sharded_jax_op(name, fn, **kwargs):
        op = _RecordedOp(name, fn, **kwargs)
        built.append(op)
        return op

    monkeypatch.setattr(bridge, "sharded_jax_op", sharded_jax_op)
    return built


@pytest.fixture
def padding_state():
    state = token_padding.TokenPaddingState.create(_TOKENS * 2, torch.device("cpu"))
    state.update(_REAL, _TOKENS)
    token_padding.set_padding_state(state)
    yield state
    token_padding.set_padding_state(None)


def _build(has_score_bias, drop_padding):
    return bridge._build_op(
        None,
        2,
        True,
        "silu",
        None,
        True,
        scoring_fn="sigmoid" if has_score_bias else "softmax",
        has_score_bias=has_score_bias,
        routed_scaling_factor=2.5 if has_score_bias else 1.0,
        token_replica_groups=((0,), (1,)),
        drop_padding=drop_padding,
    )


_SCORE_BIAS = torch.zeros(2 * _STUB_EP)


def _call_site(op, has_score_bias):
    """Drive `fused_moe_ep` exactly as the fp8 quant method does."""
    owner = SimpleNamespace(**{bridge.FUSED_MOE_EP_OP_ATTR: op})
    hidden = torch.ones(_TOKENS, 16, dtype=torch.bfloat16)
    w1 = torch.zeros(2, 16, 32, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(2, 16, 16, dtype=torch.float8_e4m3fn)
    s1 = torch.ones(2, 32)
    s2 = torch.ones(2, 16)
    logits = torch.zeros(_TOKENS, 2 * _STUB_EP)
    bias = _SCORE_BIAS if has_score_bias else None
    bridge.fused_moe_ep(owner, hidden, w1, w2, s1, s2, logits, bias)
    return op.calls[-1][0]


@pytest.mark.parametrize("has_score_bias", [False, True])
def test_drop_padding_adds_exactly_one_operand(recorder, has_score_bias):
    plain = _build(has_score_bias, drop_padding=False)
    dropping = _build(has_score_bias, drop_padding=True)
    assert plain is not dropping, "drop_padding is not part of the op cache key"
    assert len(recorder) == 2
    for op in (plain, dropping):
        assert op.built_operands == len(op.input_partition_specs), (
            f"{op.fn.__name__} registers {op.built_operands} parameters but "
            f"was given {len(op.input_partition_specs)} partition specs"
        )
    assert dropping.built_operands == plain.built_operands + 1, (
        f"drop_padding should add exactly one operand: "
        f"{plain.fn.__name__}={plain.built_operands}, "
        f"{dropping.fn.__name__}={dropping.built_operands}"
    )
    params = list(inspect.signature(dropping.fn).parameters)
    assert params[-1] == "valid", params
    assert list(inspect.signature(plain.fn).parameters) == params[:-1]
    # The mask rides the token axis like x, not replicated like the bias.
    assert dropping.input_partition_specs[-1] == dropping.input_partition_specs[0]
    assert dropping.input_partition_specs[:-1] == plain.input_partition_specs
    assert ("score_bias" in params) is has_score_bias


@pytest.mark.parametrize("has_score_bias", [False, True])
def test_call_site_passes_the_built_count(recorder, padding_state, has_score_bias):
    plain = _build(has_score_bias, drop_padding=False)
    dropping = _build(has_score_bias, drop_padding=True)

    plain_args = _call_site(plain, has_score_bias)
    assert plain.called_operands == plain.built_operands, (
        f"the op without drop_padding took {plain.built_operands} operands and "
        f"was passed {plain.called_operands} even though a padding state exists"
    )
    drop_args = _call_site(dropping, has_score_bias)
    assert dropping.called_operands == dropping.built_operands, (
        f"the drop-padding op was built with {dropping.built_operands} "
        f"operands and the call site passed {dropping.called_operands}"
    )
    assert len(drop_args) == len(plain_args) + 1
    valid = drop_args[-1]
    want = torch.zeros(_TOKENS, dtype=torch.int32)
    want[:_REAL] = 1
    assert valid.dtype == torch.int32
    torch.testing.assert_close(valid, want, rtol=0, atol=0)
    if has_score_bias:
        # The mask goes AFTER the bias: the bias keeps its position.
        assert plain_args[-1] is _SCORE_BIAS
        assert drop_args[-2] is _SCORE_BIAS


def test_call_without_padding_state_refuses(recorder):
    token_padding.set_padding_state(None)
    dropping = _build(False, drop_padding=True)
    with pytest.raises(RuntimeError, match="token padding state"):
        _call_site(dropping, False)


@pytest.mark.parametrize("has_score_bias", [False, True])
def test_valid_operand_reaches_the_kernel(recorder, monkeypatch, has_score_bias):
    """The trailing operand is forwarded as `valid_rows`, and only there."""
    from vllm_torchtpu.kernels.fused_moe import v2

    kernel = Mock(side_effect=lambda x, *a, **kw: x)
    monkeypatch.setattr(v2, "fused_ep_moe_v2", kernel)
    plain = _build(has_score_bias, drop_padding=False)
    dropping = _build(has_score_bias, drop_padding=True)
    x = np.ones((_TOKENS, 8), np.float32)
    base = (x, None, None, None, None, np.zeros((_TOKENS, 4), np.float32), None)
    bias = (np.arange(4, dtype=np.float32),) if has_score_bias else ()
    valid = np.arange(_TOKENS, dtype=np.int32) < _REAL

    assert plain.fn(*base, *bias) is x
    assert kernel.call_args.kwargs["valid_rows"] is None
    assert dropping.fn(*base, *bias, valid) is x
    forwarded = kernel.call_args.kwargs
    assert forwarded["valid_rows"] is valid
    assert forwarded["sharded_plan"] is True
    if has_score_bias:
        np.testing.assert_array_equal(forwarded["score_bias"], bias[0])


def _layer(experts=4, hidden=8, inter=16):
    return SimpleNamespace(
        w13_weight=torch.zeros(experts, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(experts, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(experts, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(experts, 1, 1, hidden),
        global_num_experts=experts * _STUB_EP,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )


@pytest.mark.parametrize(
    "sharded_plan,drop_env,want",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_prebuild_builds_drop_padding_only_under_the_sharded_plan(
    sharded_plan, drop_env, want
):
    """The env reaches `_build_op` as drop_padding, and without the sharded
    plan -- which is the only plan that can drop a selection -- it is off."""
    seen = []
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP})
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True),
        patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024),
        patch.object(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", sharded_plan),
        patch.object(envs, "MOE_FUSED_EP_V2_DROP_PADDING", drop_env),
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=None),
        patch.object(bridge, "ep_token_replica_groups", return_value=((0,), (1,))),
        patch.object(bridge, "_max_node_tokens", return_value=4096),
        patch.object(bridge, "_smem_capacity_bytes", return_value=1024 * 1024),
        patch.object(
            bridge, "_build_op", side_effect=lambda *a, **kw: seen.append(kw) or "op"
        ),
    ):
        op = bridge.prebuild_fused_moe_ep(
            _layer(), topk=2, renormalize=True, activation="silu"
        )
    assert op == "op", "the layer did not arm, so this asserts nothing"
    assert seen[0]["drop_padding"] is want
