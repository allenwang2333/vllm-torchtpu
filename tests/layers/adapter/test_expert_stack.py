# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Load-time expert stacks (layers.adapter.expert_stack) on CPU.

The TPU path (donated dynamic_update_slice row writes, never-materialized placeholders, HBM) is checked on
hardware; here the bookkeeping and the numerics: rows written from the staged checkpoint bytes, requantized
in place by the real fp8 processing, must hold exactly what the per-layer path produces.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import model_loader_patches as mlp
from vllm_torchtpu.layers.adapter import expert_stack
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

E, INTER, H, BLOCK = 6, 128, 256, 128
F8 = torch.float8_e4m3fn


def _param(t):
    return torch.nn.Parameter(t, requires_grad=False)


def _raw(seed):
    g = torch.Generator().manual_seed(seed)
    return dict(
        w13=(torch.randn(E, 2 * INTER, H, generator=g) * 0.5).to(F8),
        w2=(torch.randn(E, H, INTER, generator=g) * 0.5).to(F8),
        s13=torch.rand(E, 2 * INTER // BLOCK, H // BLOCK, generator=g) + 0.5,
        s2=torch.rand(E, H // BLOCK, INTER // BLOCK, generator=g) + 0.5,
    )


def _layer(raw, *, empty=False, name="model.layers.3.mlp.experts"):
    def w(t):
        return _param(torch.empty_like(t) if empty else t.clone())

    return SimpleNamespace(
        layer_name=name,
        w13_weight=w(raw["w13"]),
        w2_weight=w(raw["w2"]),
        w13_weight_scale_inv=_param(raw["s13"].clone()),
        w2_weight_scale_inv=_param(raw["s2"].clone()),
        moe_config=SimpleNamespace(
            intermediate_size_per_partition=INTER,
            intermediate_size_per_partition_unpadded=INTER,
            tp_size=1,
            tp_rank=0,
        ),
    )


@pytest.fixture
def plan(monkeypatch):
    raws = [_raw(10 + r) for r in range(3)]
    layers = [_layer(raw, empty=True, name=f"model.layers.{3 + r}.mlp.experts") for r, raw in enumerate(raws)]
    p = expert_stack.LoadStacks(layers)
    p.allocate()
    monkeypatch.setattr(expert_stack, "_PLAN", p)
    return p, layers, raws


def _bits(t):
    return t.contiguous().view(torch.uint8) if t.dtype == F8 else t


def test_stack_shapes_are_the_served_layout(plan):
    p, layers, _ = plan
    assert {k: tuple(t.shape) for k, t in p.stacks.items()} == {
        "w13": (3, E, H, 2 * INTER), "w2": (3, E, INTER, H), "s13": (3, E, 1, 1, 2 * INTER), "s2": (3, E, 1, 1, H)}
    assert [expert_stack.row_of(layer) for layer in layers] == [0, 1, 2]
    assert expert_stack.owns(layers[1].w2_weight) and not expert_stack.owns(layers[1].w2_weight_scale_inv)
    assert p.incomplete() is not None


@pytest.mark.parametrize("chunked", [False, True])
def test_rows_hold_exactly_the_per_layer_result(plan, monkeypatch, chunked):
    """Checkpoint bytes into the rows, then the real fp8 processing requantizes them in place: every row equals
    the per-layer path's tensors on the same checkpoint, chunked or not (the chunk overwrites its own experts)."""
    p, layers, raws = plan
    if chunked:  # two experts per chunk: chunk k overwrites only the experts it read
        monkeypatch.setattr(fp8_mod, "_REQUANT_CHUNK_BYTES", 2 * 2 * INTER * H * 4)
    kwargs = dict(weight_block_size=(BLOCK, BLOCK), activation="silu")
    refs = []
    for raw in raws:
        refs.append(fp8_mod._process_fp8_moe_weights(_layer(raw), **kwargs))
    for layer, raw in zip(layers, raws):
        p.write_raw(layer.w13_weight, raw["w13"])
        p.write_raw(layer.w2_weight, raw["w2"])
    for r, layer in enumerate(layers):  # the checkpoint bytes sit in the row until the layer is processed
        assert torch.equal(_bits(p.read_raw(layer, "w13", 0, E)), _bits(raws[r]["w13"]))
        assert torch.equal(_bits(p.read_raw(layer, "w2", 1, 4)), _bits(raws[r]["w2"][1:4]))
    for layer in layers:
        out = fp8_mod._process_fp8_moe_weights(layer, **kwargs)
        assert [tuple(t.shape) for t in out[:4]] == [(E, H, 2 * INTER), (E, 1, 1, 2 * INTER), (E, INTER, H),
                                                      (E, 1, 1, H)]
    assert p.incomplete() is None
    for r, ref in enumerate(refs):
        w13, s13, w2, s2 = ref[:4]
        for kind, want in (("w13", w13), ("s13", s13), ("w2", w2), ("s2", s2)):
            assert torch.equal(_bits(p.stacks[kind][r]), _bits(want)), (r, kind)
    rep = p.report()
    assert rep["stacked_bytes"] == rep["per_layer_bytes"] == sum(
        t.numel() * t.element_size() for ref in refs for t in ref[:4])


def test_unloaded_row_cannot_be_processed(plan):
    p, layers, raws = plan
    p.write_raw(layers[0].w13_weight, raws[0]["w13"])
    with pytest.raises(RuntimeError, match="never loaded"):
        fp8_mod._process_fp8_moe_weights(layers[0], weight_block_size=(BLOCK, BLOCK), activation="silu")


def test_stager_writes_owned_parameters_into_rows(plan):
    """The expert staging patch uploads a completed stacked-row parameter into its row and leaves the parameter
    alone (no device tensor of its own); other parameters keep the per-layer upload."""
    p, layers, raws = plan
    stager = mlp.ExpertParamStager()
    param = layers[2].w2_weight
    before = param.data_ptr()
    host = stager.host_param(param)
    for e in range(E):
        host[e].copy_(raws[2]["w2"][e])
        done = stager.written(param, e, "w2")
    assert done and stager.stacked == 1 and stager.flushed == 1
    assert param.data_ptr() == before  # still the loader target create_weights made
    assert (2, "w2") in p.raw_written
    assert torch.equal(_bits(p.read_raw(layers[2], "w2", 0, E)), _bits(raws[2]["w2"]))


def test_stager_never_writes_a_stacked_row_parameter_early(plan, monkeypatch):
    p, layers, raws = plan
    monkeypatch.setattr(mlp, "_MAX_STAGED", 2)
    stager = mlp.ExpertParamStager()
    other = _param(torch.zeros(E, 4, dtype=torch.float32))
    flushed = []
    real_flush = stager.flush
    monkeypatch.setattr(stager, "flush", lambda prm: (flushed.append(prm), real_flush(prm)))
    stager.host_param(layers[0].w13_weight)
    stager.host_param(other)
    stager.host_param(layers[1].w13_weight)   # over the cap: the non-stacked one goes early
    assert flushed == [other] and id(other) in stager.written_early
    stager.host_param(layers[2].w13_weight)   # over the cap, only stacked ones staged: none is written early
    assert flushed == [other] and len(stager) == 3


def test_not_planned_without_the_megakernel_env(monkeypatch):
    monkeypatch.setenv("GLM52_DECODE_MK", "0")
    assert "GLM52_DECODE_MK" in expert_stack.wanted()
    monkeypatch.setenv("GLM52_DECODE_MK", "1")
    monkeypatch.setenv("GLM52_DECODE_MK_STACK_E", "0")
    assert "STACK_E" in expert_stack.wanted()
    monkeypatch.delenv("GLM52_DECODE_MK_STACK_E")
    assert expert_stack.wanted() is None


def test_dummy_loader_is_refused(monkeypatch):
    """Rows are written by the staged upload of real weights; any other loader keeps the per-layer path (the
    megakernel then stacks after load, as before)."""
    monkeypatch.setenv("GLM52_DECODE_MK", "1")
    monkeypatch.setattr(expert_stack, "_PLAN", None)
    monkeypatch.setattr(expert_stack, "_REFUSED", None)
    expert_stack.maybe_plan(torch.nn.Module(), SimpleNamespace(), SimpleNamespace())
    assert expert_stack.current() is None and "loader" in expert_stack.refused()
