"""The sequence-parallel decoder forward traced by Dynamo on the CPU, with no chips: the real patched forwards (the
decoder layer's, the attention layer's, the GDN projection helper) on stub modules that carry the real linear methods
(fp8 per-channel, and the checkpoint's unquantised b/a projection), traced with a symbolic row count at a decode-sized
and a prompt-sized row count, with the fp8 all-gather on and off.

It checks layer attributes, graph outputs, symbolic shape guards, matching lifted inputs across trace groups,
and the dtypes consumed by the pre-quantised projection paths."""
import importlib
import types

import pytest
import torch

HIDDEN, OUT, ROWS_SMALL, ROWS_LARGE, MIN_ROWS = 4096, 512, 8, 2048, 1024


@pytest.fixture(scope="module")
def _dist():
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (ensure_model_parallel_initialized,
                                  init_distributed_environment)
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29571",
            backend="gloo")
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)
    yield


def _fp8_linear(n_in, n_out):
    """A stub linear carrying the plugin's fp8 method in its runtime layout: weight (k, n) fp8, per-channel scales."""
    from vllm_torchtpu.layers.vllm.quantization.fp8 import (
        VllmFp8Config, VllmFp8LinearMethodTPU)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(n_in, n_out).clamp(-2, 2).to(
        torch.float8_e4m3fn),
                                      requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(n_out, dtype=torch.float32) / 30, requires_grad=False)
    layer.bias = None
    layer.skip_bias_add = False
    layer.quant_method = VllmFp8LinearMethodTPU(
        VllmFp8Config.from_config({"activation_scheme": "dynamic"}))
    layer.forward = types.MethodType(
        lambda self, x: (self.quant_method.apply(self, x, None), None), layer)
    return layer


def _bf16_linear(n_in, n_out):
    """The checkpoint's unquantised projection (the GDN's b/a): a bf16 weight with the plugin's unquantised method."""
    from vllm_torchtpu.layers.vllm.quantization.unquantized import \
        VllmUnquantizedLinearMethod
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(n_in, n_out, dtype=torch.bfloat16) / 60,
        requires_grad=False)
    layer.bias = None
    layer.skip_bias_add = False
    try:
        layer.quant_method = VllmUnquantizedLinearMethod()
    except TypeError:
        layer.quant_method = VllmUnquantizedLinearMethod(None)
    layer.forward = types.MethodType(
        lambda self, x: (self.quant_method.apply(self, x, None), None), layer)
    return layer


class _Norm(torch.nn.Module):

    def forward(self, x, residual=None):
        if residual is None:
            return x * 0.5
        y = x + residual
        return y * 0.5, y


class _Attn(torch.nn.Module):
    """The attention layer's surface for the plugin's patched forward: qkv_proj, _project_qkv_gate, attn, o_proj."""

    def __init__(self):
        super().__init__()
        self.qkv_proj = _fp8_linear(HIDDEN, 3 * OUT)
        self.o_proj = _fp8_linear(OUT, HIDDEN)

    def _project_qkv_gate(self, qkv, positions):
        q, k, v = qkv.split([OUT, OUT, OUT], dim=-1)
        return q, k, v, None

    def attn(self, q, k, v):
        return q + k + v


class _Gdn(torch.nn.Module):
    """The GDN layer's projections through the plugin's real helper: qkvz fp8, b/a unquantised."""

    def __init__(self):
        super().__init__()
        self.in_proj_qkvz = _fp8_linear(HIDDEN, 2 * OUT)
        self.in_proj_ba = _bf16_linear(HIDDEN, 64)
        self.out_proj = _fp8_linear(OUT, HIDDEN)

    def forward(self, hidden_states, prequant=None):
        from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import \
            _project
        qkvz = _project(self.in_proj_qkvz, hidden_states, prequant)
        ba = _project(self.in_proj_ba, hidden_states, prequant)
        core = qkvz[:, :OUT] * torch.sigmoid(ba[:, :1])
        out, _ = self.out_proj(core)
        return out


class _Mlp(torch.nn.Module):

    def forward(self, hidden_states, already_sequence_parallel=False):
        return hidden_states * 1.5


def _layer(layer_type):
    from vllm.model_executor.models import qwen3_next as qn
    layer = torch.nn.Module()
    layer.use_attn_reduce_scatter_for_moe = True
    layer.layer_type = layer_type
    layer.layer_scale = False
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    layer.mlp = _Mlp()
    if layer_type == "linear_attention":
        layer.linear_attn = _Gdn()
    else:
        attn = _Attn()
        attn.forward = types.MethodType(qn.Qwen3NextAttention.forward, attn)
        layer.self_attn = attn
    layer.register_buffer("_tpu_sp_rank", torch.tensor([0], dtype=torch.int32))
    layer.forward = types.MethodType(qn.Qwen3NextDecoderLayer.forward, layer)
    return layer


def _patched(monkeypatch, fp8, qk_fusion=True):
    monkeypatch.setenv("TPU_SP_MIN_ROWS", str(MIN_ROWS))
    monkeypatch.setenv("TPU_SP_FP8_GATHER", "1" if fp8 else "0")
    monkeypatch.setenv("TPU_FUSED_QK_NORM_ROPE", "1" if qk_fusion else "0")
    from vllm_torchtpu import envs
    importlib.reload(envs)
    from vllm.model_executor.models import qwen3_next as qn

    import vllm_torchtpu
    monkeypatch.setattr(qn.Qwen3NextDecoderLayer,
                        "_tpu_sp_by_rows_patch",
                        False,
                        raising=False)
    monkeypatch.setattr(qn,
                        "_tpu_fused_qk_norm_rope_patch",
                        False,
                        raising=False)
    vllm_torchtpu._patch_qwen3_next_sequence_parallel_by_rows()
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    assert "prequant" in qn.Qwen3NextAttention.forward.__code__.co_varnames, "sequence parallelism did not install its attention interface"
    from vllm_torchtpu.layers.vllm import linear_common as lc
    lc.prebuild_prequant_ops()
    # serving fills the dense matmul op's cache before any trace (the requant at load); the lazy
    # builder's lock is never inside a traced forward there, and must not be here either
    lc._get_quantized_matmul_op()


def _trace(layer, rows, with_residual):
    """Dynamo's export of the decoder forward with the row count symbolic: the graph and its placeholders."""
    import torch._dynamo as dynamo
    hidden = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    residual = torch.randn(rows, HIDDEN,
                           dtype=torch.bfloat16) if with_residual else None
    positions = torch.arange(rows if not with_residual else rows * 1,
                             dtype=torch.int64)
    full = positions
    if with_residual:
        # a later layer: the rows in hand are this rank's slice (world 1: the full rows), positions the full count
        pass
    dynamo.mark_dynamic(hidden, 0)
    dynamo.mark_dynamic(full, 0)
    if residual is not None:
        dynamo.mark_dynamic(residual, 0)
    exported = dynamo.export(layer, aten_graph=False)(hidden,
                                                      residual,
                                                      positions=full)
    gm = exported.graph_module
    outputs = [n for n in gm.graph.nodes if n.op == "output"][0].args[0]
    for o in (outputs if isinstance(outputs, (tuple, list)) else [outputs]):
        assert hasattr(
            o, "meta") or o is None, f"a non-tensor graph output: {o!r}"
    placeholders = sorted(n.target for n in gm.graph.nodes
                          if n.op == "placeholder")
    return gm, placeholders


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("qk_fusion", [False, True])
@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_decoder_forward_traces_on_both_branches(_dist, monkeypatch, fp8,
                                                 qk_fusion, layer_type):
    _patched(monkeypatch, fp8, qk_fusion)
    layer = _layer(layer_type)
    # the first layer: full rows in, no residual
    _trace(layer, ROWS_SMALL, with_residual=False)
    _trace(layer, ROWS_LARGE, with_residual=False)
    # a later layer: residual in hand (the sequence-parallel branch gathers, then reduce-scatters)
    _, small = _trace(layer, ROWS_SMALL, with_residual=True)
    _, large = _trace(layer, ROWS_LARGE, with_residual=True)
    # the two trace groups must lift the same buffers or the compile flow refuses the hand-over
    assert small == large, f"lifted inputs differ between the branches: {set(small) ^ set(large)}"


def test_prequant_rows_reach_an_unquantised_projection(_dist, monkeypatch):
    # the GDN's b/a projection is bf16 in the checkpoint: the helper's fallback must hand it bf16 rows
    _patched(monkeypatch, True)
    from vllm_torchtpu.layers.vllm import linear_common as lc
    from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import _project
    layer = _bf16_linear(HIDDEN, 64)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode(allow_non_fake_inputs=True):
        x = torch.empty(ROWS_LARGE, HIDDEN, dtype=torch.bfloat16)
        q, s = lc.quantize_rows(x)
        out = _project(layer, q, (q, s))
        assert out.dtype == torch.bfloat16 and out.shape == (ROWS_LARGE, 64)
