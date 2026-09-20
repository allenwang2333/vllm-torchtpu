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
"""The fused EP MoE op's operand count is decided twice; they must agree.

WHAT BROKE. `fused_moe/v2`'s operand count is a function of the layer's
`e_score_correction_bias`, and TWO places compute it independently:

* BUILD -- `_routing_config(layer)` reads `layer.e_score_correction_bias is
  not None`, and `_build_op` then registers `moe_with_score_bias` (eight
  operands) instead of `moe` (seven).
* CALL -- the quant method's forward reads the STAGED buffer,
  `score_bias_operand(layer)` (attribute `_tpu_fused_score_bias`, written by
  `register_score_bias_buffer` at weight load), and `fused_moe_ep` passes
  seven operands when that is None.

On the fp8 path `register_score_bias_buffer` was never called -- only
`nvfp4.py` and `mxfp4.py` called it -- so GLM-5.2-FP8, the first fp8
checkpoint with a routing bias, built an eight-operand op and called it with
seven. Dynamo refused the trace with `pallas::fused_moe_ep() is missing value
for argument 'score_bias'`, twice, each time after a ~50-minute server start.
Fixed by `f83ea46ae` (pass it at the call site) and `326816249` (stage it at
weight load); NEITHER HALF IS SUFFICIENT ALONE, which is why the assertion
below is the two counts against each other and not "the call site passes
something".

WHAT IS ASSERTED. Not the symptom -- a Dynamo error message is a property of
one torch version -- but the contract: the operand count the op was BUILT
with equals the operand count the call site PASSES. Both numbers are read out
of the code under test (the built count from the registered function's
signature and from the partition-spec tuple `_build_op` hands `shard_map`;
the called count from the arguments the op object actually receives), so
neither 7 nor 8 is written down here and a future ninth operand needs no edit.

The bias-free layer is the discriminator: it must arm with the SEVEN-operand
form and stage nothing. Without it, "the counts agree" is satisfiable by a
buffer that is always present.

NO TPU. Every decision under test is made at weight load and at trace time,
before any device work: the mesh is stubbed, and the op is recorded rather
than executed. `test_a_missing_staged_buffer_is_caught` is the red control --
it re-runs the same check with `register_score_bias_buffer` neutered, which
is precisely the pre-`326816249` tree, and requires the check to FAIL.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge

# The stubbed mesh width. Layer factories size the global expert count against
# it, because the kernel requires one equal contiguous expert block per shard.
_STUB_EP = 2

# 512 is the smallest width every one of the three loaders accepts: the fp4
# paths requantize at a block that must divide both contractions, and mxfp4's
# `REQUANTIZED_BLOCK_SIZE` is 512 unless the environment overrides it.
_EXPERTS = 4
_HIDDEN = 512
_INTER = 512
_GLOBAL_EXPERTS = _EXPERTS * _STUB_EP


class _RecordedOp:
    """Stands in for the torch custom op `sharded_jax_op` would register.

    Holds both halves of the contract: what the op was BUILT with (the
    function whose signature torch_tpu turns into the op's schema, and the
    input partition specs `shard_map` is given, one per operand) and what the
    call site PASSED (the positional arguments this object was called with).
    """

    def __init__(self, name, fn, *, mesh, input_partition_specs,
                 output_partition_specs):
        self.name = name
        self.fn = fn
        self.input_partition_specs = tuple(input_partition_specs)
        self.calls = []

    # `_build_op` replaces the shard-aware fake with a cheap one.
    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0]

    # --- the two operand counts, each read off the code under test ---------
    @property
    def built_operands(self) -> int:
        """How many operands the op was built to take.

        Two independent reads, asserted equal by `_arm_and_forward`: the
        registered function's parameter list is what torch_tpu turns into the
        op schema Dynamo checks a call against, and the partition-spec tuple
        is what `shard_map` is handed.
        """
        return len(inspect.signature(self.fn).parameters)

    @property
    def called_operands(self) -> int:
        args, kwargs = self.calls[0]
        assert not kwargs, (
            f"the call site passed keyword operands {sorted(kwargs)}; the "
            "counts below compare positional arity only")
        return len(args)


class _Armed(SimpleNamespace):
    """What one armed layer's load-and-forward produced."""


def _moe_config(use_ep=True):
    return SimpleNamespace(
        has_bias=False,
        is_act_and_mul=True,
        experts_per_token=2,
        intermediate_size_per_partition=_INTER,
        intermediate_size_per_partition_unpadded=_INTER,
        tp_size=1,
        tp_rank=0,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep, pcp_size=1),
    )


def _routing_attrs(layer, bias: bool):
    """The routing fields `RoutedExperts.__init__` always sets.

    `scoring_func`/`routed_scaling_factor` follow GLM-5.2's `noaux_tc`
    router when a bias is present -- a sigmoid score with a per-global-expert
    correction added for selection -- because that is the configuration the
    arity bug was found on.
    """
    layer.custom_routing_function = None
    layer.use_grouped_topk = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.global_num_experts = _GLOBAL_EXPERTS
    layer.renormalize = True
    layer.activation = "silu"
    layer.moe_config = _moe_config()
    if bias:
        layer.scoring_func = "sigmoid"
        layer.routed_scaling_factor = 2.5
        layer.register_parameter(
            "e_score_correction_bias",
            torch.nn.Parameter(torch.arange(_GLOBAL_EXPERTS,
                                            dtype=torch.float32),
                               requires_grad=False))
    else:
        layer.scoring_func = "softmax"
        layer.routed_scaling_factor = 1.0
        layer.e_score_correction_bias = None
    return layer


# ---------------------------------------------------------------------------
# one checkpoint-shaped layer and one loaded quant method per weight format
# ---------------------------------------------------------------------------
def _fp8_case(monkeypatch, bias):
    """A block-free fp8 MoE checkpoint, loaded by the real fp8 method.

    This is the path the bug was on, so as little as possible is stubbed:
    `process_weights_after_loading` runs its real dequantize/requantize and
    its real arming sequence. Only the GMM-side registrations, which want a
    device and a distributed group, are neutered.
    """
    from vllm_torchtpu.layers.adapter import fused_moe_rs
    from vllm_torchtpu.layers.adapter.quantization import fp8

    layer = _routing_attrs(torch.nn.Module(), bias)
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(torch.ones(_EXPERTS,
                                      2 * _INTER,
                                      _HIDDEN,
                                      dtype=torch.float8_e4m3fn),
                           requires_grad=False))
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(torch.ones(_EXPERTS,
                                      _HIDDEN,
                                      _INTER,
                                      dtype=torch.float8_e4m3fn),
                           requires_grad=False))
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(torch.ones(_EXPERTS, 2), requires_grad=False))
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(torch.ones(_EXPERTS), requires_grad=False))

    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method.moe = layer.moe_config
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_block_size = None
    method.block_quant = False
    method.weight_scale_name = "weight_scale"

    monkeypatch.setattr(fp8, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(fp8, "prebuild_fused_moe_kernel", lambda **kw: None)
    monkeypatch.setattr(fp8.moe_routing, "register_experts_start_buffer",
                        lambda *a, **kw: None)
    monkeypatch.setattr(fp8.moe_routing, "validate_linear_ep_placement",
                        lambda *a: None)
    monkeypatch.setattr(fused_moe_rs, "prebuild_fused_moe_rs",
                        lambda *a, **kw: None)
    return layer, method, method.process_weights_after_loading, "apply_monolithic"


def _nvfp4_case(monkeypatch, bias):
    """An NVFP4 checkpoint, loaded by the real nvfp4 method.

    The requantizer is a Pallas op, so it is replaced by a shape-correct
    stub: what this file gates happens after it, and the fp4 numerics are
    covered by `test_fused_moe_ep.py`.
    """
    from vllm_torchtpu.layers.adapter.quantization import nvfp4

    layer = _routing_attrs(torch.nn.Module(), bias)
    for name, shape, dtype in (
        ("w13_weight", (_EXPERTS, 2 * _INTER, _HIDDEN // 2), torch.uint8),
        ("w2_weight", (_EXPERTS, _HIDDEN, _INTER // 2), torch.uint8),
        ("w13_weight_scale", (_EXPERTS, 2 * _INTER, _HIDDEN // 16),
         torch.float8_e4m3fn),
        ("w2_weight_scale", (_EXPERTS, _HIDDEN, _INTER // 16),
         torch.float8_e4m3fn),
        ("w13_weight_scale_2", (_EXPERTS, 2), torch.float32),
        ("w2_weight_scale_2", (_EXPERTS, ), torch.float32),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.ones(shape, dtype=dtype),
                               requires_grad=False))

    method = object.__new__(nvfp4.VllmNvfp4MoEMethod)
    method.moe = layer.moe_config
    method.group_size = 16

    def requant(w, scales, block):
        e, n, half_k = w.shape
        return (torch.empty((e, half_k * 2, n // 2),
                            dtype=torch.float4_e2m1fn_x2),
                torch.ones(e, half_k * 2 // block, 1, n))

    monkeypatch.setattr(nvfp4, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(nvfp4, "requant_load_kmajor_fp4", requant)
    monkeypatch.setattr(
        nvfp4, "load_kmajor_fp4", lambda w: torch.empty(
            (w.shape[0], w.shape[2] * 2, w.shape[1] // 2),
            dtype=torch.float4_e2m1fn_x2))
    monkeypatch.setattr(nvfp4, "prebuild_fused_moe_kernel", lambda **kw: None)
    monkeypatch.setattr(nvfp4.moe_routing, "register_experts_start_buffer",
                        lambda *a, **kw: None)
    monkeypatch.setattr(nvfp4.moe_routing, "validate_linear_ep_placement",
                        lambda *a: None)
    monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", 128)
    return layer, method, method.process_weights_after_loading, "apply_monolithic"


def _mxfp4_case(monkeypatch, bias):
    """A DeepSeek-V4 MXFP4 checkpoint, loaded by the real mxfp4 method.

    Same treatment as nvfp4: the native-fp4 quantizer is a device op and is
    stubbed to the layout the real one produces, and the forward entry point
    is `_forward_monolithic_tpu` rather than `apply_monolithic`.
    """
    from vllm_torchtpu.layers.adapter.quantization import mxfp4

    layer = _routing_attrs(torch.nn.Module(), bias)
    for name, shape, dtype in (
        ("w13_weight", (_EXPERTS, 2 * _INTER, _HIDDEN // 2), torch.uint8),
        ("w2_weight", (_EXPERTS, _HIDDEN, _INTER // 2), torch.uint8),
        ("w13_weight_scale", (_EXPERTS, 2 * _INTER, _HIDDEN // 32),
         torch.uint8),
        ("w2_weight_scale", (_EXPERTS, _HIDDEN, _INTER // 32), torch.uint8),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.full(shape, 127, dtype=dtype),
                               requires_grad=False))

    method = object.__new__(mxfp4.VllmDeepseekV4Mxfp4MoEMethod)
    method.moe = layer.moe_config
    method.intermediate_size = _INTER

    def quantize(w, block, pack):
        e, k, n = w.shape
        packed = torch.empty(
            (e, k, n // 2),
            dtype=torch.uint8 if pack else torch.float4_e2m1fn_x2)
        return packed, torch.ones(e, k // block, 1, n)

    monkeypatch.setattr(mxfp4, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(mxfp4, "quantize_native_fp4_kmajor", quantize)
    monkeypatch.setattr(mxfp4, "synchronize_tensors", lambda *a, **kw: None)
    monkeypatch.setattr(mxfp4, "prebuild_fused_moe_kernel", lambda **kw: None)
    monkeypatch.setattr(mxfp4.moe_routing, "register_experts_start_buffer",
                        lambda *a, **kw: None)
    monkeypatch.setattr(mxfp4.moe_routing, "validate_linear_ep_placement",
                        lambda *a: None)
    return (layer, method, method.process_weights_after_loading,
            "_forward_monolithic_tpu")


_CASES = {"fp8": _fp8_case, "nvfp4": _nvfp4_case, "mxfp4": _mxfp4_case}


def _arm_and_forward(monkeypatch, fmt, bias, stage_bias=True) -> _Armed:
    """Load one layer through its real quant method, then drive one forward.

    Returns the recorded op plus what the layer staged. `stage_bias=False`
    neuters `register_score_bias_buffer`, which is the pre-`326816249` fp8
    path exactly, and is what the red control uses.
    """
    # `_build_op` memoizes on its closure; a shared cache would let one
    # parametrization answer another's build.
    monkeypatch.setattr(bridge, "_OPS", {})
    built = []

    def sharded_jax_op(name, fn, **kwargs):
        op = _RecordedOp(name, fn, **kwargs)
        built.append(op)
        return op

    # Prebuild's refusals are logged and return None, and `*_once` dedupes
    # across parametrizations, so an unarmed layer would otherwise fail
    # without saying which gate refused it.
    refusals = []
    monkeypatch.setattr(bridge.logger, "warning_once",
                        lambda msg, *a: refusals.append(msg % a if a else msg))
    monkeypatch.setattr(bridge.logger, "info_once",
                        lambda msg, *a: refusals.append(msg % a if a else msg))

    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: _STUB_EP})
    monkeypatch.setattr(bridge, "sharded_jax_op", sharded_jax_op)
    monkeypatch.setattr(bridge, "build_ep_mesh", lambda *a, **kw: mesh)
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(bridge, "ep_rank_order", lambda: None)
    monkeypatch.setattr(bridge, "_max_node_tokens", lambda *a: 4096)
    monkeypatch.setattr(bridge, "_smem_capacity_bytes", lambda: 1024 * 1024)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", False)
    if not stage_bias:
        monkeypatch.setattr(bridge, "register_score_bias_buffer",
                            lambda layer: None)

    layer, method, load, forward_name = _CASES[fmt](monkeypatch, bias)
    load(layer)

    assert bridge.fused_moe_ep_supported(method), (
        f"the {fmt} layer did not arm the fused EP path, so this test would "
        "assert nothing; prebuild said: " + " | ".join(refusals))
    assert len(built) == 1, f"expected one built op, got {len(built)}"
    op = built[0]

    tokens = 8
    getattr(method, forward_name)(layer,
                                  torch.ones(tokens,
                                             _HIDDEN,
                                             dtype=torch.bfloat16),
                                  torch.ones(tokens, _GLOBAL_EXPERTS))
    assert len(op.calls) == 1, "the forward did not reach the fused op"
    # The two independent reads of the BUILT count must agree before either
    # is compared with the call site's.
    assert op.built_operands == len(op.input_partition_specs), (
        f"the op registered {op.built_operands} parameters but was given "
        f"{len(op.input_partition_specs)} partition specs")
    return _Armed(op=op,
                  layer=layer,
                  method=method,
                  staged=bridge.score_bias_operand(layer))


@pytest.mark.parametrize("fmt", sorted(_CASES))
@pytest.mark.parametrize("bias", [True, False])
def test_built_and_called_operand_counts_agree(monkeypatch, fmt, bias):
    """THE regression assertion. Neither count is written down here."""
    armed = _arm_and_forward(monkeypatch, fmt, bias)
    assert armed.op.built_operands == armed.op.called_operands, (
        f"{fmt} layer with e_score_correction_bias="
        f"{armed.layer.e_score_correction_bias is not None} built "
        f"{armed.op.built_operands} operands ({armed.op.fn.__name__}) but the "
        f"call site passed {armed.op.called_operands}; torch.compile refuses "
        "this with `pallas::fused_moe_ep() is missing value for argument "
        "'score_bias'`")


@pytest.mark.parametrize("fmt", sorted(_CASES))
def test_a_routing_bias_is_staged_and_adds_exactly_one_operand(
        monkeypatch, fmt):
    """The positive and its discriminator, in one test so they are compared.

    A bias-free layer must arm with the SEVEN-operand form and stage nothing;
    a biased one must stage a buffer and take exactly one operand more.
    """
    with_bias = _arm_and_forward(monkeypatch, fmt, bias=True)
    without = _arm_and_forward(monkeypatch, fmt, bias=False)

    assert without.staged is None, (
        f"{fmt}: a layer with no e_score_correction_bias staged "
        f"{without.staged!r}; the call site would then pass an operand the "
        "op was not built to take")
    assert with_bias.staged is not None, (
        f"{fmt}: process_weights_after_loading left "
        f"{bridge.FUSED_MOE_EP_SCORE_BIAS_ATTR} unset on a layer whose "
        "e_score_correction_bias is not None -- the op is built with the "
        "bias operand and the call site cannot supply it")
    torch.testing.assert_close(
        with_bias.staged,
        with_bias.layer.e_score_correction_bias.detach().to(torch.float32))
    assert with_bias.op.built_operands == without.op.built_operands + 1, (
        f"{fmt}: the bias is supposed to be exactly one extra operand, but "
        f"the counts were {with_bias.op.built_operands} and "
        f"{without.op.built_operands}")
    assert "score_bias" in inspect.signature(with_bias.op.fn).parameters, (
        f"{fmt}: the built op's operands are "
        f"{list(inspect.signature(with_bias.op.fn).parameters)}")


@pytest.mark.parametrize("fmt", sorted(_CASES))
def test_a_missing_staged_buffer_is_caught(monkeypatch, fmt):
    """The red control: the tree before `326816249`, and it must FAIL.

    Without this, a check that never fires reads exactly like a check that
    passes. `register_score_bias_buffer` is neutered, which is what the fp8
    path did by simply never calling it, and the arity comparison above is
    required to break.
    """
    armed = _arm_and_forward(monkeypatch, fmt, bias=True, stage_bias=False)
    assert armed.staged is None  # the bug, reproduced
    assert armed.op.built_operands != armed.op.called_operands, (
        f"{fmt}: the operand counts still agreed with the staging step "
        "removed, so the assertion in "
        "test_built_and_called_operand_counts_agree cannot detect the bug "
        "it exists for")


def test_the_call_site_reads_the_staged_buffer_not_the_parameter(monkeypatch):
    """Why staging is a separate step, and what reading the wrong one costs.

    `fused_moe_ep`'s eighth operand must be the STAGED f32 buffer: the
    parameter itself may be any dtype the checkpoint carried, and the op
    closes over `score_bias.astype(jnp.float32)` only for the operand it was
    built with. A call site that passed `layer.e_score_correction_bias`
    directly would agree on arity and still be a different tensor, so this
    pins the identity rather than the count.
    """
    armed = _arm_and_forward(monkeypatch, "fp8", bias=True)
    passed = armed.op.calls[0][0][-1]
    assert passed is armed.staged
    assert passed is not armed.layer.e_score_correction_bias
    assert passed.dtype is torch.float32
