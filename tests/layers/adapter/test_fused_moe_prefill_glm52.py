# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 on the fused-MoE op: a shared expert without a gate and a routed
scaling factor of 2.5. Admission and arming of a GLM-5.2-shaped layer, the
per-rank body's dispatch for such a layer (the prefill kernel from
FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS, GMM MoE below, never the decode kernel)
and the GMM fallback's sum (shared + 2.5 x routed, all-reduced). No TPU
needed: kernels, collectives and the mesh are stubs, and the weights are meta
tensors (admission reads only shapes and dtypes)."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.runner import moe_runner

from vllm_torchtpu.layers.adapter import fused_moe_decode as bridge
from vllm_torchtpu.layers.adapter import fused_moe_prefill, moe_routing
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

# GLM-5.2 at TP16, per rank: 256 experts, hidden 6144, routed and shared
# intermediate 128, top-8, FP8 weights with per-channel scales.
N_RANKS, E, H, INTER, INTER_S, TOPK = 16, 256, 6144, 128, 128, 8
ROUTED_SCALE = 2.5
F8 = torch.float8_e4m3fn


def _meta(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _gate(shape=(1, H), bias=None):
    """A Qwen-style expert_gate linear: weight [1, H], no bias."""
    return types.SimpleNamespace(weight=_meta(*shape, dtype=torch.bfloat16), bias=bias)


class _Runner:
    """The MoERunner attributes admission and arming use, GLM-5.2-shaped:
    TP MoE, a gate-less FP8 shared expert, routed_scaling_factor 2.5."""

    def __init__(self, gate=None, routed_scaling_factor=ROUTED_SCALE):
        layer = torch.nn.Module()
        layer.w13_weight = _meta(E, H, 2 * INTER, dtype=F8)
        layer.w2_weight = _meta(E, INTER, H, dtype=F8)
        layer.w13_weight_scale_inv = _meta(E, 1, 1, 2 * INTER)
        layer.w2_weight_scale_inv = _meta(E, 1, 1, H)
        layer.moe_config = types.SimpleNamespace(experts_per_token=TOPK)
        # DeepseekV2MLP: SiluAndMul, K-major FP8 linears with 1-D scales and
        # no biases, no expert_gate.
        mlp = types.SimpleNamespace(
            act_fn=object.__new__(SiluAndMul),
            gate_up_proj=types.SimpleNamespace(
                weight=_meta(H, 2 * INTER_S, dtype=F8), weight_scale=_meta(2 * INTER_S)
            ),
            down_proj=types.SimpleNamespace(
                weight=_meta(INTER_S, H, dtype=F8), weight_scale=_meta(H)
            ),
        )
        if gate is not None:
            mlp.expert_gate = gate
        quant_method = object.__new__(VllmFp8MoEMethodTPU)
        quant_method._tpu_activation_str = "silu"
        self.moe_config = types.SimpleNamespace(
            moe_parallel_config=types.SimpleNamespace(
                use_ep=False, tp_size=N_RANKS, ep_size=1, dp_size=1
            ),
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            in_dtype=torch.bfloat16,
        )
        self._quant_method = quant_method
        self.routed_experts = layer
        self.routed_input_transform = None
        self.routed_output_transform = None
        self.routed_scaling_factor = routed_scaling_factor
        self.shared_expert_gate = None
        self._shared_experts = types.SimpleNamespace(_layer=mlp)

    def _select_forward(self):
        return "forward"


def test_admits_glm52_layer():
    runner = _Runner()
    admitted = bridge._admit(runner, N_RANKS)
    assert admitted is not None
    shared, expert_parallel = admitted
    assert not expert_parallel
    # None in the gate weight's place; the shared linears as they are.
    assert [None if t is None else tuple(t.shape) for t in shared] == [
        (H, 2 * INTER_S),
        (2 * INTER_S,),
        (INTER_S, H),
        (H,),
        None,
    ]
    # The same layer with a Qwen-style gate and no routed scale: admitted with
    # its gate weight.
    shared, _ = bridge._admit(_Runner(gate=_gate(), routed_scaling_factor=1.0), N_RANKS)
    assert tuple(shared[4].shape) == (1, H)


def _shared_mlp(runner):
    return runner._shared_experts._layer


def _set(obj, **attrs):
    for name, value in attrs.items():
        setattr(obj, name, value)


# Each case breaks one thing about the GLM-5.2 layer that admission refused
# before the gate-less / scaled additions, and still refuses.
_REFUSED = {
    "gate with a bias": lambda r: _set(
        _shared_mlp(r), expert_gate=_gate(bias=_meta(1))
    ),
    "gate weight not [1, H]": lambda r: _set(_shared_mlp(r), expert_gate=_gate((H, 1))),
    "gate without a weight": lambda r: _set(
        _shared_mlp(r), expert_gate=types.SimpleNamespace(weight=None)
    ),
    "runner-level shared expert gate": lambda r: _set(r, shared_expert_gate=object()),
    "gated shared expert with a routed scale": lambda r: _set(
        _shared_mlp(r), expert_gate=_gate()
    ),
    "scale NaN": lambda r: _set(r, routed_scaling_factor=float("nan")),
    "scale inf": lambda r: _set(r, routed_scaling_factor=float("inf")),
    "scale not exact in bf16": lambda r: _set(r, routed_scaling_factor=1.8),
    "scale under FP16": lambda r: _set(r.moe_config, in_dtype=torch.float16),
    "ungated shared activation not SiluAndMul": lambda r: _set(
        _shared_mlp(r), act_fn=object()
    ),
    "ungated shared gate_up with a bias": lambda r: _set(
        _shared_mlp(r).gate_up_proj, bias=_meta(2 * INTER_S)
    ),
    "ungated shared down with a bias": lambda r: _set(
        _shared_mlp(r).down_proj, bias=_meta(H)
    ),
    "no shared expert": lambda r: _set(r, _shared_experts=None),
    "shared scale 2-D": lambda r: _set(
        _shared_mlp(r).down_proj, weight_scale=_meta(1, H)
    ),
    "shared intermediate above the routed one": lambda r: _set(
        _shared_mlp(r).down_proj, weight=_meta(2 * INTER, H, dtype=F8)
    ),
    "activation not silu": lambda r: _set(r._quant_method, _tpu_activation_str="gelu"),
    "routed input transform": lambda r: _set(r, routed_input_transform=object()),
    "TP size is not the mesh size": lambda r: _set(
        r.moe_config.moe_parallel_config, tp_size=N_RANKS // 2
    ),
}


@pytest.mark.parametrize("case", list(_REFUSED))
def test_still_refused(case):
    runner = _Runner()
    _REFUSED[case](runner)
    assert bridge._admit(runner, N_RANKS) is None


@pytest.fixture
def arming(monkeypatch):
    """arm_fused_moe_decode with the mesh, the rank and the op builder stubbed;
    returns the recorded _build_op and _check_vmem calls."""
    built, vmem = [], []

    def build_op(mesh, n_ranks, topk, expert_parallel, max_tokens, min_tokens, *a):
        built.append((n_ranks, topk, expert_parallel, max_tokens, min_tokens, *a))
        return object()

    mesh = types.SimpleNamespace(shape={bridge.EP_AXIS_NAME: N_RANKS})
    monkeypatch.setattr(moe_runner, "MoERunner", _Runner)
    monkeypatch.setattr(bridge, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 3)
    monkeypatch.setattr(moe_routing, "get_experts_start", lambda layer: None)
    monkeypatch.setattr(bridge, "_RANK_BUFFER", None)
    monkeypatch.setattr(bridge, "_RANK_BUFFER_START", 0)
    monkeypatch.setattr(bridge, "_check_vmem", lambda *a: vmem.append(a))
    monkeypatch.setattr(bridge, "_build_op", build_op)
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", "256")
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS", "256")
    return built, vmem


def _model(*runners):
    return types.SimpleNamespace(modules=lambda: list(runners))


def _armed(runner):
    return getattr(runner, bridge._RUNNER_ARMED_ATTR, False)


@pytest.mark.parametrize("decode", [False, True])
def test_arms_glm52_layer_with_decode_cap_0(monkeypatch, arming, decode):
    built, vmem = arming
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL", "1")
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "1" if decode else "0")
    glm = _Runner()
    qwen = _Runner(gate=_gate(), routed_scaling_factor=1.0)
    assert bridge.arm_fused_moe_decode(_model(glm, qwen)) == 2
    cap = 256 if decode else 0
    # (n_ranks, topk, expert_parallel, decode cap, prefill from, gated, scale):
    # the GLM-5.2 layer gets cap 0 whatever FUSED_MOE_DECODE_KERNEL says.
    assert built == [
        (N_RANKS, TOPK, False, 0, 256, False, ROUTED_SCALE),
        (N_RANKS, TOPK, False, cap, 256, True, 1.0),
    ]
    assert [v[0] for v in vmem] == ([256] if decode else [])  # the gated layer's
    for runner in (glm, qwen):
        assert _armed(runner)
        assert runner._shared_experts is None
        assert runner._quant_method._fused_moe_decode_op is not None
    # The op applies the scale: the runner must not scale the reduced sum.
    assert glm.routed_scaling_factor == 1.0
    # A tensor stands in for the missing gate (the op takes tensors only).
    assert tuple(glm.routed_experts._fmd_wg.shape) == (1, 1)
    assert tuple(qwen.routed_experts._fmd_wg.shape) == (1, H)


def test_glm52_layer_not_armed_by_decode_kernel_alone(monkeypatch, arming):
    built, vmem = arming
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "1")
    monkeypatch.delenv("FUSED_MOE_PREFILL_KERNEL", raising=False)
    glm = _Runner()
    qwen = _Runner(gate=_gate(), routed_scaling_factor=1.0)
    assert bridge.arm_fused_moe_decode(_model(glm, qwen)) == 1
    assert built == [(N_RANKS, TOPK, False, 256, None, True, 1.0)]
    assert not _armed(glm) and _armed(qwen)
    # Untouched: the runner keeps its shared expert and its scale.
    assert glm._shared_experts is not None
    assert glm.routed_scaling_factor == ROUTED_SCALE


def test_glm52_layer_not_armed_with_ep_experts(monkeypatch, arming):
    built, vmem = arming
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL", "1")
    monkeypatch.delenv("FUSED_MOE_DECODE_KERNEL", raising=False)
    glm = _Runner()
    qwen = _Runner(gate=_gate(), routed_scaling_factor=1.0)
    for runner in (glm, qwen):
        runner.moe_config.moe_parallel_config = types.SimpleNamespace(
            use_ep=True, tp_size=1, ep_size=N_RANKS, dp_size=1
        )
    # Only TP experts are tested for GLM-5.2 layers: under EP the gated layer
    # is still armed, the GLM-5.2 layer keeps the runner's path.
    assert bridge.arm_fused_moe_decode(_model(glm, qwen)) == 1
    assert built == [(N_RANKS, TOPK, True, 0, 256, True, 1.0)]
    assert not _armed(glm) and _armed(qwen)
    assert glm._shared_experts is not None
    assert glm.routed_scaling_factor == ROUTED_SCALE


def test_prefill_layer_passes_gate_and_scale_to_kernel(monkeypatch):
    seen = {}

    def kernel(*args, axis_name, expert_start, routed_scale):
        seen.update(gate=args[9], expert_start=expert_start, routed_scale=routed_scale)
        return args[0]

    monkeypatch.setattr(fused_moe_prefill, "_kernel_body", kernel)
    w = np.zeros((1, 1), np.float32)
    fused_moe_prefill.prefill_layer(
        np.zeros((256, H), np.float32),
        *([w] * 8),
        None,
        np.zeros((256, TOPK), np.float32),
        np.zeros((256, TOPK), np.int32),
        None,
        ROUTED_SCALE,
    )
    assert seen == dict(gate=None, expert_start=None, routed_scale=ROUTED_SCALE)


def test_build_op_refuses_decode_cap_for_glm52_layer():
    for gated, scale in ((False, 1.0), (True, ROUTED_SCALE), (False, ROUTED_SCALE)):
        with pytest.raises(AssertionError):
            bridge._build_op(None, N_RANKS, TOPK, False, 256, 256, gated, scale)


def test_build_op_binds_statics(monkeypatch):
    """The exported program runs the per-rank body with the layer's statics,
    and a layer with other statics gets its own op. jax_op and shard_map are
    stubs: the op is the traced function itself."""
    seen = []

    def layer(*operands, **statics):
        # An unbound static is what _layer would see: its default.
        seen.append((statics.get("gated", True), statics.get("routed_scale", 1.0)))
        return operands[1]

    def jax_op(name, fn, **kwargs):
        return types.SimpleNamespace(fn=fn, register_fake=lambda fake: None)

    monkeypatch.setattr(bridge, "_layer", layer)
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(bridge, "jax_op", jax_op)
    monkeypatch.setattr(jax, "shard_map", lambda f, **kwargs: f)
    mesh = object()
    glm = bridge._build_op(mesh, N_RANKS, TOPK, False, 0, 256, False, ROUTED_SCALE)
    qwen = bridge._build_op(mesh, N_RANKS, TOPK, False, 0, 256, True, 1.0)
    assert qwen is not glm
    assert (
        bridge._build_op(mesh, N_RANKS, TOPK, False, 0, 256, False, ROUTED_SCALE) is glm
    )
    for op in (glm, qwen):
        op.fn(*[np.zeros((1, 1), np.float32)] * 13)
    assert seen == [(False, ROUTED_SCALE), (True, 1.0)]


def _dispatch(monkeypatch, tokens, **statics):
    """Run the op's per-rank body with every path stubbed; returns the calls:
    (path, gate operand, expert start, routed scale)."""
    calls = []

    def decode(rank, x, *args, expert_start, **kwargs):
        calls.append(("decode", args[8], expert_start, None))
        return x

    def gmm(*args, topk):
        calls.append(("gmm", args[9], args[12], args[13]))
        return args[0]

    def prefill(*args):
        calls.append(("prefill", args[9], args[12], args[13]))
        return args[0]

    monkeypatch.setattr(bridge, "_kernel_body", decode)
    monkeypatch.setattr(bridge, "_gmm_layer", gmm)
    monkeypatch.setattr(bridge, "prefill_layer", prefill)
    rank = np.array([[3, 192]], np.int32)  # [[mesh index, first expert]]
    w = np.zeros((1, 1), np.float32)
    bridge._layer(
        rank,
        np.zeros((tokens, 8), np.float32),
        *([w] * 9),
        np.zeros((tokens, TOPK), np.float32),
        np.zeros((tokens, TOPK), np.int32),
        n_ranks=N_RANKS,
        topk=TOPK,
        **statics,
    )
    return calls, w


@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "tokens,path", [(1, "gmm"), (64, "gmm"), (255, "gmm"), (256, "prefill")]
)
def test_dispatch_with_scale(monkeypatch, tokens, path, gated, expert_parallel):
    calls, w = _dispatch(
        monkeypatch,
        tokens,
        expert_parallel=expert_parallel,
        max_tokens=0,
        prefill_min_tokens=256,
        gated=gated,
        routed_scale=ROUTED_SCALE,
    )
    start = 192 if expert_parallel else None
    assert len(calls) == 1
    got_path, gate, got_start, scale = calls[0]
    assert (got_path, got_start, scale) == (path, start, ROUTED_SCALE)
    # Gate-less: None reaches the path, never the placeholder operand.
    assert gate is (w if gated else None)


def test_decode_kernel_never_serves_armed_glm52_layer(monkeypatch, arming):
    built, _ = arming
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "1")
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL", "1")
    assert bridge.arm_fused_moe_decode(_model(_Runner())) == 1
    _, topk, expert_parallel, cap, min_tokens, gated, scale = built[0]
    for tokens in (1, 16, 64, 128, 255, 256, 1024):
        calls, _ = _dispatch(
            monkeypatch,
            tokens,
            expert_parallel=expert_parallel,
            max_tokens=cap,
            prefill_min_tokens=min_tokens,
            gated=gated,
            routed_scale=scale,
        )
        assert [c[0] for c in calls] == ["prefill" if tokens >= 256 else "gmm"]


def _gmm_fallback(monkeypatch, gate, scale):
    """_gmm_layer on tiny operands with GMM MoE, the quantized linears and the
    all-reduce stubbed. Every value is a small dyadic number, so each bf16
    rounding is exact and the result can be compared exactly."""
    t, h, i_s = 4, 8, 2
    rng = np.random.default_rng(0)
    routed = jnp.asarray(rng.integers(-16, 16, (t, h)) / 8, jnp.bfloat16)
    shared = jnp.asarray(rng.integers(-16, 16, (t, h)) / 16, jnp.bfloat16)
    gate_up = jnp.asarray(rng.integers(-4, 4, (t, 2 * i_s)) / 4, jnp.bfloat16)
    tw = jnp.full((t, TOPK), 1 / TOPK, jnp.bfloat16)
    seen = {}

    def fused_moe_func(x, w1, w2, w1_scale, w2_scale, b1, b2, topk_w, topk_i, *a, **k):
        seen["topk_weights"] = topk_w
        return routed

    outs = iter((gate_up, shared))

    def quantized_matmul(x, w, scale):
        return next(outs)

    def psum(value, axis_name):
        seen["psum"] = (value, axis_name)
        return value

    monkeypatch.setattr(bridge, "fused_moe_func", fused_moe_func)
    monkeypatch.setattr(bridge, "_quantized_matmul_jax", quantized_matmul)
    monkeypatch.setattr(jax.lax, "psum", psum)
    z = jnp.zeros
    out = bridge._gmm_layer(
        z((t, h), jnp.bfloat16),
        z((E, h, 2)),
        z((E, 1, h)),
        z((E, 1, 1, 2)),
        z((E, 1, 1, h)),
        z((h, 2 * i_s)),
        z((2 * i_s,)),
        z((i_s, h)),
        z((h,)),
        gate,
        tw,
        z((t, TOPK), jnp.int32),
        None,
        scale,
        topk=TOPK,
    )
    # The scale is applied after GMM MoE, not folded into the router weights.
    assert seen["topk_weights"] is tw
    assert out is seen["psum"][0] and seen["psum"][1] == bridge.EP_AXIS_NAME
    assert out.dtype == jnp.bfloat16
    return np.asarray(out, np.float32), routed, shared


def test_gmm_fallback_glm52(monkeypatch):
    out, routed, shared = _gmm_fallback(monkeypatch, None, ROUTED_SCALE)
    expected = np.asarray(shared, np.float32) + ROUTED_SCALE * np.asarray(
        routed, np.float32
    )
    np.testing.assert_array_equal(out, expected)


def test_gmm_fallback_defaults_unchanged(monkeypatch):
    # A zero gate weight: sigmoid(0) = 1/2 on the shared expert; no scale.
    out, routed, shared = _gmm_fallback(monkeypatch, jnp.zeros((1, 8)), 1.0)
    expected = 0.5 * np.asarray(shared, np.float32) + np.asarray(routed, np.float32)
    np.testing.assert_array_equal(out, expected)
