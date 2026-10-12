"""v2 model wiring for the D-sharded mHC residual."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model

MULT = 4


def _vllm_config(tp=8, layers=4, pp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=True,
            data_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=layers)
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )


@pytest.mark.parametrize(
    "flag,tp,layer,sp,fused_ep,pipelined,expected",
    [
        (True, 8, 0, False, False, False, True),
        (True, 8, 3, False, False, False, True),
        (False, 8, 0, False, False, False, False),
        (True, 1, 0, False, False, False, False),
        (True, 8, 4, False, False, False, False),  # DSpark draft layer
        (True, 8, 0, True, False, False, False),
        (True, 8, 0, False, True, False, False),
        (True, 8, 0, False, False, True, False),
    ],
)
def test_shards_mhc_residual(
    monkeypatch, flag, tp, layer, sp, fused_ep, pipelined, expected
):
    monkeypatch.setattr(v2_model.envs, "TPU_DEEPSEEK_V41_SHARDED_MHC", flag)
    monkeypatch.setattr(v2_model.envs, "USE_MOE_FUSED_EP_KERNEL", fused_ep)
    monkeypatch.setattr(v2_model, "_use_sequence_parallel", lambda _: sp)
    monkeypatch.setattr(
        v2_model, "enable_pipelined_collective_and_compute", lambda: pipelined
    )
    assert v2_model._shards_mhc_residual(_vllm_config(tp), layer) is expected
    assert v2_model._shards_mhc_residual(_vllm_config(tp, pp=2), layer) is False


@pytest.mark.parametrize("sharded", [True, False])
def test_decoder_layer_defers_all_reduce(monkeypatch, sharded):
    def fake_init(self, vllm_config, prefix="", **kwargs):
        torch.nn.Module.__init__(self)
        self.attn = SimpleNamespace(wo_b=SimpleNamespace(reduce_results=True))
        self.ffn = SimpleNamespace(
            experts=SimpleNamespace(
                moe_config=SimpleNamespace(skip_final_all_reduce=False)
            )
        )

    monkeypatch.setattr(v2_model.upstream.DeepseekV4DecoderLayer, "__init__", fake_init)
    monkeypatch.setattr(v2_model.envs, "TPU_DEEPSEEK_V41_CED", False)
    monkeypatch.setattr(v2_model, "_shards_mhc_residual", lambda cfg, idx: sharded)
    calls = []
    monkeypatch.setattr(
        v2_model.DeepseekV41DecoderLayer,
        "_sharded_forward",
        lambda self, *a, **kw: calls.append("sharded"),
    )
    monkeypatch.setattr(
        v2_model.DeepseekV41DecoderLayer,
        "_replicated_forward",
        lambda self, *a, **kw: calls.append("replicated"),
    )
    layer = v2_model.DeepseekV41DecoderLayer(_vllm_config(), prefix="model.layers.1")
    assert layer.defers_seam_all_reduce is sharded
    assert layer.attn.wo_b.reduce_results is not sharded
    assert layer.ffn.experts.moe_config.skip_final_all_reduce is sharded
    layer(torch.zeros(1))
    assert calls == ["sharded" if sharded else "replicated"]


def test_layer_forward_bindings_have_distinct_namespaces():
    cls = v2_model.DeepseekV41DecoderLayer
    replicated, sharded = cls._replicated_forward, cls._sharded_forward
    assert replicated.__code__ is sharded.__code__
    assert replicated.__globals__["__name__"] != sharded.__globals__["__name__"]
    assert sharded.__globals__["mhc_shifted_post_pre"] is v2_model._sharded_seam
    assert sharded.__globals__["mhc_post_tilelang"] is v2_model._reduce_then_post
    assert replicated.__globals__["mhc_shifted_post_pre"] is v2_model.shifted_post_pre


def test_sharded_seam_and_reduce_then_post(monkeypatch):
    seen = []
    monkeypatch.setattr(
        v2_model, "shifted_post_pre", lambda *a, **kw: seen.append((a, kw)) or "seam"
    )
    assert v2_model._sharded_seam(1, 2, pre_mix=3, reduce_results=False) == "seam"
    assert seen == [((1, 2), {"pre_mix": 3, "reduce_results": True})]

    monkeypatch.setattr(v2_model, "tensor_model_parallel_all_reduce", lambda t: t + 1)
    monkeypatch.setattr(v2_model, "post", lambda x, r, p, c: (x, r, p, c))
    x = torch.zeros(2)
    out = v2_model._reduce_then_post(x, "r", "p", "c")
    torch.testing.assert_close(out[0], x + 1)
    assert out[1:] == ("r", "p", "c")


def _final_inputs(tokens=3, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, dim, generator=g).to(torch.bfloat16)
    residual = torch.randn(tokens, MULT, dim, generator=g).to(torch.bfloat16)
    post = torch.rand(tokens, MULT, 1, generator=g)
    comb = torch.rand(tokens, MULT, MULT, generator=g)
    pre = torch.rand(tokens, MULT, generator=g)
    return x, residual, post, comb, pre


@pytest.mark.parametrize("capture_aux", [True, False])
def test_reduce_post_and_collapse_paths(monkeypatch, capture_aux):
    x, residual, post, comb, pre = _final_inputs()
    want, want_aux = v2_model._post_and_collapse(
        x, residual, post, comb, pre, capture_aux=capture_aux
    )
    calls = []
    monkeypatch.setattr(
        v2_model,
        "tensor_model_parallel_all_reduce",
        lambda t: calls.append("ar") or t,
    )
    got, aux = v2_model._reduce_post_and_collapse(
        x, residual, post, comb, pre, deferred_reduce=False, capture_aux=capture_aux
    )
    torch.testing.assert_close(got, want)
    assert calls == []
    got, aux = v2_model._reduce_post_and_collapse(
        x, residual, post, comb, pre, deferred_reduce=True, capture_aux=capture_aux
    )
    torch.testing.assert_close(got, want)
    assert calls == ["ar"]
    assert (aux is None) is (not capture_aux)


@pytest.mark.parametrize("capture_aux", [True, False])
def test_reduce_post_and_collapse_sharded_is_column_local(monkeypatch, capture_aux):
    """Rank 1 of 2 collapses its own columns; the gather reassembles them."""
    dim, tp, rank = 8, 2, 1
    width = dim // tp
    x, residual, post, comb, pre = _final_inputs(dim=dim)
    want, want_aux = v2_model._post_and_collapse(
        x, residual, post, comb, pre, capture_aux=capture_aux
    )
    cols = slice(rank * width, (rank + 1) * width)
    events = []

    def reduce_scatter(t, dim=-1):
        events.append(("rs", tuple(t.shape)))
        return t[..., cols]

    def all_gather(t, dim=-1):
        events.append(("ag", tuple(t.shape)))
        return torch.cat([t] * tp, dim=dim)

    monkeypatch.setattr(
        v2_model, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )
    monkeypatch.setattr(v2_model, "tensor_model_parallel_all_gather", all_gather)
    got, aux = v2_model._reduce_post_and_collapse(
        x,
        residual[..., cols].contiguous(),
        post,
        comb,
        pre,
        deferred_reduce=True,
        capture_aux=capture_aux,
    )
    assert got.shape == want.shape
    torch.testing.assert_close(got[:, cols], want[:, cols])
    expected = [("rs", (3, dim)), ("ag", (3, width))]
    if capture_aux:
        torch.testing.assert_close(aux[:, cols], want_aux[:, cols])
        expected.append(("ag", (3, width)))
    assert events == expected


class _FusedLayer:
    defers_seam_all_reduce = True

    def __call__(self, hidden, pos, ids, pre, post, res_mix, res, *a, **kw):
        tokens = hidden.shape[0]
        out_pre = pre if pre is not None else torch.ones(tokens, MULT)
        residual = torch.zeros(tokens, MULT, hidden.shape[-1])
        return (
            hidden,
            residual,
            torch.ones(tokens, MULT, 1),
            torch.eye(MULT).expand(tokens, MULT, MULT),
            out_pre,
            None,
        )


def _stage_model(written):
    class _Decoder(_FusedLayer):
        def attn_norm(self, x):
            return x

        attn = SimpleNamespace(
            write_encoder_global_kv=lambda n, p, sp_sharded=False: written.append(n)
        )

    return SimpleNamespace(
        _ced_boundary=1,
        start_layer=0,
        end_layer=2,
        _stage_uses_sequence_parallel=lambda stage: False,
        use_native_mega_moe=False,
        _prepare_engram=lambda *a, **kw: (None, None),
        aux_hidden_state_layers=set(),
        layers=[_FusedLayer(), _Decoder()],
        norm=lambda x: x,
        _mtp_hidden_buffer=None,
    )


def test_forward_stage_reduces_deferred_last_output(monkeypatch):
    written: list[torch.Tensor] = []
    model = _stage_model(written)
    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    events = []
    monkeypatch.setattr(
        v2_model,
        "tensor_model_parallel_all_reduce",
        lambda t: events.append("ar") or t,
    )
    monkeypatch.setattr(
        v2_model,
        "post",
        lambda h, r, p, c: events.append("post") or r,
    )

    def fake_reduce_collapse(x, residual, post, comb, pre, *, deferred_reduce, **kw):
        events.append(("collapse", deferred_reduce))
        return x, None

    monkeypatch.setattr(v2_model, "_reduce_post_and_collapse", fake_reduce_collapse)
    ids = torch.arange(4, dtype=torch.int32)
    embeds = torch.ones(4, 16)
    out = v2_model.DeepseekV41Model._forward_stage(
        model, ids, ids, None, inputs_embeds=embeds, stage="full"
    )
    assert out.shape == (4, 16)
    assert events == [("collapse", True)]

    # The encoder stage ends on `post`, which needs the reduced output.
    events.clear()
    monkeypatch.setattr(v2_model, "collapse", lambda h, pm: h[:, 0])
    hidden, _ = v2_model.DeepseekV41Model._forward_stage(
        model, ids, ids, None, inputs_embeds=embeds, stage="encoder"
    )
    assert events == ["ar", "post"]
    assert hidden.shape == (4, MULT, 16) and len(written) == 1
