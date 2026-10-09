# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for TD8.58 Qwen2.5-VL patch,
Qwen3-VL patch, and XPlane trace merger tool modules."""

from __future__ import annotations

import inspect
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest  # type: ignore[missing-import]
import torch

from vllm_torchtpu.models.vllm import qwen2_5_vl_patch, qwen3_vl_patch
from vllm_torchtpu.tools import merge_xplane, xplane_pb2

xpb: Any = xplane_pb2

pytestmark = pytest.mark.cpu_test


class _FakeTpuDevice:
    type = "tpu"
    index = 0

    def __str__(self):
        return "tpu:0"

    def __repr__(self):
        return "device(type='tpu', index=0)"

    def __eq__(self, other):
        return getattr(other, "type", None) == "tpu"


class _FakeTpuTensor(torch.Tensor):
    """CPU tensor subclass reporting device.type == 'tpu' for TPU branch tests."""

    @property
    def device(self):  # type: ignore[override]
        return _FakeTpuDevice()

    def cpu(self, *args, **kwargs):
        res = super().cpu(*args, **kwargs)
        return res.as_subclass(torch.Tensor)


def _make_fake_tpu_tensor(t: torch.Tensor) -> _FakeTpuTensor:
    return torch.Tensor._make_subclass(_FakeTpuTensor, t)


_orig_tensor_to = torch.Tensor.to


def _patched_tensor_to(self, *args, **kwargs):
    device = args[0] if args else kwargs.get("device")
    if device is not None and getattr(device, "type", None) == "tpu":
        rest = args[1:]
        kw = {k: v for k, v in kwargs.items() if k != "device"}
        res = _orig_tensor_to(self, *rest, **kw)
        return _make_fake_tpu_tensor(res)
    if device == "cpu" or (
        device is not None and getattr(device, "type", None) == "cpu"
    ):
        rest = args[1:]
        kw = {k: v for k, v in kwargs.items() if k != "device"}
        res = _orig_tensor_to(self, *rest, **kw)
        return res.as_subclass(torch.Tensor)
    return _orig_tensor_to(self, *args, **kwargs)


@pytest.fixture(autouse=True)
def _fake_tpu_tensor_to(monkeypatch):
    """Route `.to(tpu)` to fake TPU tensors for this module's tests only.

    Assigned at import time this patch outlived the module and broke every
    later test that calls `nn.Module.to(device=None, dtype=...)` (e.g. the
    Kimi K3 vision tower), depending on collection order.
    """
    monkeypatch.setattr(torch.Tensor, "to", _patched_tensor_to)


# ---------------------------------------------------------------------------
# 1. qwen2_5_vl_patch.py tests
# ---------------------------------------------------------------------------


def test_qwen2_5_vl_patch_all_branches(monkeypatch):
    # 1. None model_config returns False
    assert qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(None) is False

    # 2. Non-qwen2_5_vl model_type returns False
    cfg_other = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen2_vl"))
    assert qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(cfg_other) is False  # type: ignore[arg-type]

    # 3. Valid qwen2_5_vl config applies patch
    class _FakeQwen25VL:
        def __init__(self):
            self.visual = SimpleNamespace(dtype=torch.bfloat16)

        def input_norm(self, pv, dt):
            return pv * 2.0

        def get_input_modality(self, mm_kwargs):
            return "image" if "pixel_values" in mm_kwargs else "video"

        def encoder_cudagraph_forward(self, values, path="default"):
            return values["pixel_values"]

        def encoder_eager_forward(self, mm_kwargs, path="default"):
            key = (
                "pixel_values" if "pixel_values" in mm_kwargs else "pixel_values_videos"
            )
            return mm_kwargs[key]

    fake_mod = ModuleType("vllm.model_executor.models.qwen2_5_vl")
    fake_mod.Qwen2_5_VLForConditionalGeneration = _FakeQwen25VL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen2_5_vl", fake_mod)

    cfg_valid = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen2_5_vl"))
    res = qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(cfg_valid)  # type: ignore[arg-type]
    assert res is None
    assert getattr(
        _FakeQwen25VL.encoder_cudagraph_forward,
        "_tpu_qwen2_5_vl_norm_patch",
        False,
    )

    # Test normalized_graph_forward
    inst = _FakeQwen25VL()
    t_in = torch.ones(2, 2)
    out_graph = inst.encoder_cudagraph_forward({"pixel_values": t_in})
    assert torch.equal(out_graph, t_in * 2.0)

    # Test normalized_eager_forward for image
    out_eager_img = inst.encoder_eager_forward({"pixel_values": t_in})
    assert torch.equal(out_eager_img, t_in * 2.0)

    # Test normalized_eager_forward for video
    out_eager_vid = inst.encoder_eager_forward({"pixel_values_videos": t_in})
    assert torch.equal(out_eager_vid, t_in * 2.0)

    # 4. Already patched: returns None early without modifying forward again
    res2 = qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(cfg_valid)  # type: ignore[arg-type]
    assert res2 is None


# ---------------------------------------------------------------------------
# 2. qwen3_vl_patch.py tests
# ---------------------------------------------------------------------------


def test_qwen3_vl_model_detection():
    assert qwen3_vl_patch._is_qwen3_vl_model(None) is False

    # hf_config.model_type match
    c1 = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_vl_chat"), model="foo"
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(c1) is True  # type: ignore[arg-type]

    # hf_config.architectures match
    c2 = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="other", architectures=["Qwen3VLForConditionalGeneration"]
        ),
        model="foo",
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(c2) is True  # type: ignore[arg-type]

    # model name match
    c3 = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="other", architectures=[]),
        model="path/qwen3-vl-8b",
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(c3) is True  # type: ignore[arg-type]

    c4 = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="", architectures=[]),
        model="path/qwen3vl-8b",
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(c4) is True  # type: ignore[arg-type]

    # hf_config is None with model name match
    c5 = SimpleNamespace(hf_config=None, model="path/qwen3-vl-base")
    assert qwen3_vl_patch._is_qwen3_vl_model(c5) is True  # type: ignore[arg-type]

    # completely unrelated model
    c6 = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="llama", architectures=["LlamaModel"]),
        model="llama-3",
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(c6) is False  # type: ignore[arg-type]


def test_qwen3_vl_patched_masked_scatter():
    # 1. _patched_masked_scatter_ non-TPU fallback
    t_cpu = torch.zeros(4)
    mask_cpu = torch.tensor([True, False, True, False])
    src_cpu = torch.tensor([10.0, 20.0])
    qwen3_vl_patch._patched_masked_scatter_(t_cpu, mask_cpu, src_cpu)
    assert torch.equal(t_cpu, torch.tensor([10.0, 0.0, 20.0, 0.0]))

    # 2. _patched_masked_scatter_ TPU tensor: length check error
    t_tpu = _make_fake_tpu_tensor(torch.zeros(4))
    mask_tpu = torch.tensor([True, True, True, False])
    src_short = _make_fake_tpu_tensor(torch.tensor([10.0]))
    with pytest.raises(RuntimeError, match="Number of source elements"):
        qwen3_vl_patch._patched_masked_scatter_(t_tpu, mask_tpu, src_short)

    # 3. _patched_masked_scatter_ TPU tensor: dtype & device match (skip .to)
    src_same = _make_fake_tpu_tensor(torch.tensor([1.0, 2.0, 3.0]))
    res_tpu = qwen3_vl_patch._patched_masked_scatter_(t_tpu, mask_tpu, src_same)
    assert res_tpu is t_tpu
    assert torch.equal(t_tpu, torch.tensor([1.0, 2.0, 3.0, 0.0]))

    # 4. _patched_masked_scatter_ TPU tensor: dtype mismatch
    t_tpu2 = _make_fake_tpu_tensor(torch.zeros(4, dtype=torch.bfloat16))
    src_diff = _make_fake_tpu_tensor(torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32))
    qwen3_vl_patch._patched_masked_scatter_(t_tpu2, mask_tpu, src_diff)
    assert torch.equal(t_tpu2, torch.tensor([5.0, 6.0, 7.0, 0.0], dtype=torch.bfloat16))

    # 5. _patched_masked_scatter non-TPU fallback
    out_cpu = qwen3_vl_patch._patched_masked_scatter(t_cpu, mask_cpu, src_cpu)
    assert torch.equal(out_cpu, torch.tensor([10.0, 0.0, 20.0, 0.0]))

    # 6. _patched_masked_scatter TPU tensor: length check error
    with pytest.raises(RuntimeError, match="Number of source elements"):
        qwen3_vl_patch._patched_masked_scatter(t_tpu, mask_tpu, src_short)

    # 7. _patched_masked_scatter TPU tensor: dtype match
    out_tpu = qwen3_vl_patch._patched_masked_scatter(t_tpu, mask_tpu, src_same)
    assert torch.equal(out_tpu, torch.tensor([1.0, 2.0, 3.0, 0.0]))

    # 8. _patched_masked_scatter TPU tensor: dtype mismatch
    out_tpu2 = qwen3_vl_patch._patched_masked_scatter(t_tpu2, mask_tpu, src_diff)
    assert torch.equal(
        out_tpu2, torch.tensor([5.0, 6.0, 7.0, 0.0], dtype=torch.bfloat16)
    )


def test_qwen3_vl_patched_repeat_interleave_and_cumsum():
    # 1. _patched_repeat_interleave non-TPU
    x_cpu = torch.tensor([1, 2, 3])
    rep_cpu = qwen3_vl_patch._patched_repeat_interleave(x_cpu, 2)
    assert torch.equal(rep_cpu, torch.tensor([1, 1, 2, 2, 3, 3]))

    # 2. _patched_repeat_interleave TPU non-int/bool dtype (e.g. float)
    x_tpu_flt = _make_fake_tpu_tensor(torch.tensor([1.0, 2.0]))
    rep_flt = qwen3_vl_patch._patched_repeat_interleave(x_tpu_flt, 2)
    assert torch.equal(rep_flt, torch.tensor([1.0, 1.0, 2.0, 2.0]))

    # 3. _patched_repeat_interleave TPU int with tensor repeats
    x_tpu_int = _make_fake_tpu_tensor(torch.tensor([1, 2], dtype=torch.int64))
    r_tpu = _make_fake_tpu_tensor(torch.tensor([2, 1]))
    rep_tpu = qwen3_vl_patch._patched_repeat_interleave(x_tpu_int, r_tpu)
    assert torch.equal(rep_tpu, torch.tensor([1, 1, 2], dtype=torch.int64))

    # 4. _patched_cumsum non-TPU vs TPU float vs TPU int
    assert torch.equal(
        qwen3_vl_patch._patched_cumsum(x_cpu, 0), torch.tensor([1, 3, 6])
    )
    assert torch.equal(
        qwen3_vl_patch._patched_cumsum(x_tpu_flt, 0), torch.tensor([1.0, 3.0])
    )
    cs_tpu = qwen3_vl_patch._patched_cumsum(x_tpu_int, 0)
    assert torch.equal(cs_tpu, torch.tensor([1, 3], dtype=torch.int64))

    # 5. _patched_tensor_cumsum non-TPU vs TPU float vs TPU int
    assert torch.equal(
        qwen3_vl_patch._patched_tensor_cumsum(x_cpu, 0), torch.tensor([1, 3, 6])
    )
    assert torch.equal(
        qwen3_vl_patch._patched_tensor_cumsum(x_tpu_flt, 0),
        torch.tensor([1.0, 3.0]),
    )
    cs_tens = qwen3_vl_patch._patched_tensor_cumsum(x_tpu_int, 0)
    assert torch.equal(cs_tens, torch.tensor([1, 3], dtype=torch.int64))


def test_qwen3_vl_scoped_torch_ops(monkeypatch):
    orig_ms = torch.masked_scatter
    orig_ri = torch.repeat_interleave

    # Test re-entrant scoping and has_tb branches
    class _FakeTensorBase:
        masked_scatter = None
        masked_scatter_ = None

    monkeypatch.setattr(
        torch,
        "_C",
        SimpleNamespace(_TensorBase=_FakeTensorBase),
        raising=False,
    )

    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert torch.masked_scatter is qwen3_vl_patch._patched_masked_scatter
        assert torch.repeat_interleave is qwen3_vl_patch._patched_repeat_interleave
        # Re-entrant level 2
        with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
            assert torch.masked_scatter is qwen3_vl_patch._patched_masked_scatter
        assert torch.masked_scatter is qwen3_vl_patch._patched_masked_scatter

    assert torch.masked_scatter is orig_ms
    assert torch.repeat_interleave is orig_ri

    # Branch where _TensorBase has non-None attributes
    _FakeTensorBase.masked_scatter = orig_ms  # type: ignore[assignment]
    _FakeTensorBase.masked_scatter_ = torch.Tensor.masked_scatter_  # type: ignore[assignment]
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert _FakeTensorBase.masked_scatter is qwen3_vl_patch._patched_masked_scatter
    assert _FakeTensorBase.masked_scatter is orig_ms

    # Branch where has_tb is False
    class _FakeTorchCNoTB:
        pass

    monkeypatch.setattr(torch, "_C", _FakeTorchCNoTB())
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert torch.masked_scatter is qwen3_vl_patch._patched_masked_scatter
    assert torch.masked_scatter is orig_ms

    # Branch where _TensorBase attribute setter raises TypeError / AttributeError
    class _FailingTensorBase:
        def __init__(self):
            self._m = orig_ms
            self._m_ = torch.Tensor.masked_scatter_

        @property
        def masked_scatter(self):
            return self._m

        @masked_scatter.setter
        def masked_scatter(self, v):
            raise TypeError("read-only property")

        @property
        def masked_scatter_(self):
            return self._m_

        @masked_scatter_.setter
        def masked_scatter_(self, v):
            raise AttributeError("read-only property")

    class _FakeTorchCWithFailingTB:
        _TensorBase = _FailingTensorBase()

    monkeypatch.setattr(torch, "_C", _FakeTorchCWithFailingTB())
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert torch.masked_scatter is qwen3_vl_patch._patched_masked_scatter
    assert torch.masked_scatter is orig_ms


def test_qwen3_vl_patch_get_rope_index():
    # 1. target_cls is None
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(SimpleNamespace())

    # 2. get_rope_index is None
    mod_no_rope = SimpleNamespace(Qwen3VLModel=SimpleNamespace(get_rope_index=None))
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(mod_no_rope)

    # 3. Valid get_rope_index with signature binding and execution
    called_args: list[Any] = []

    class _FakeQwen3VLModel:
        def get_rope_index(
            self,
            input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=None,
            **kwargs,
        ):
            called_args.append(
                (
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                    kwargs,
                )
            )
            return input_ids * 2

    mod_valid = SimpleNamespace(Qwen3VLModel=_FakeQwen3VLModel)
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(mod_valid)
    assert getattr(
        _FakeQwen3VLModel.get_rope_index,
        "_tpu_qwen3_vl_rope_index_patch",
        False,
    )

    # Verify already-patched guard
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(mod_valid)

    # Call patched_get_rope_index:
    # A. 1D image_grid_thw (triggers unsqueeze), empty video_grid_thw (triggers None),
    # TPU tensor (triggers to_cpu and to_device), tuple and dict recursive args
    inst = _FakeQwen3VLModel()
    tpu_ids = _make_fake_tpu_tensor(torch.tensor([1, 2]))
    img_grid_1d = torch.tensor([1, 2, 3])
    empty_vid_grid = torch.tensor([])
    res_tpu = inst.get_rope_index(
        tpu_ids,
        image_grid_thw=img_grid_1d,
        video_grid_thw=empty_vid_grid,
        second_per_grid_ts=(torch.tensor([1]),),
        attention_mask={"mask": torch.tensor([1])},
        extra_opt="extra",
    )
    assert torch.equal(res_tpu, torch.tensor([2, 4]))
    assert called_args[-1][1].shape == (1, 3)
    assert called_args[-1][2] is None

    # B. 2D image_grid_thw (no unsqueeze), cpu tensor (no target device)
    res_cpu = inst.get_rope_index(
        torch.tensor([3, 4]),
        image_grid_thw=torch.ones(1, 3),
        video_grid_thw=None,
    )
    assert torch.equal(res_cpu, torch.tensor([6, 8]))

    # C. Non-tensor image_grid_thw (branches to fix_grid return g)
    # and nested container return from get_rope_index (tuple, list, dict, scalar)
    class _FakeQwen3VLModelNested:
        def get_rope_index(
            self,
            input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=None,
            **kwargs,
        ):
            return (
                input_ids,
                [input_ids],
                {"out": input_ids, "scalar": 42},
            )

    mod_nested = SimpleNamespace(Qwen3VLModel=_FakeQwen3VLModelNested)
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(mod_nested)
    inst_nested = _FakeQwen3VLModelNested()
    nested_res = inst_nested.get_rope_index(
        tpu_ids,
        image_grid_thw="non_tensor_grid",
        video_grid_thw=None,
        second_per_grid_ts=[torch.tensor([1])],
        attention_mask={"dict_k": torch.tensor([2])},
    )
    assert isinstance(nested_res, tuple)
    assert isinstance(nested_res[1], list)
    assert isinstance(nested_res[2], dict)
    assert nested_res[2]["scalar"] == 42
    assert nested_res[0].device.type == "tpu"

    # D. TypeError during sig.bind fallback to sig.bind_partial
    def _faulty_bind(*args, **kwargs):
        raise TypeError("partial fallback test")

    orig_bind = inspect.Signature.bind
    inspect.Signature.bind = _faulty_bind  # type: ignore[assignment]
    try:
        res_partial = inst.get_rope_index(torch.tensor([5, 6]))
        assert torch.equal(res_partial, torch.tensor([10, 12]))
    finally:
        inspect.Signature.bind = orig_bind


def test_qwen3_vl_patch_vision_attention_and_transformer():
    # 1. _patch_qwen3_vl_vision_attention early exits
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(SimpleNamespace())
    mod_no_fwd = SimpleNamespace(Qwen3VLVisionAttention=SimpleNamespace(forward=None))
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(mod_no_fwd)

    # 2. _patch_qwen3_vl_vision_attention patch & call
    attn_called: list[Any] = []

    class _FakeVisionAttention:
        def forward(self, hidden_states, cu_seqlens=None, **kwargs):
            attn_called.append((hidden_states, cu_seqlens, kwargs))
            return hidden_states

    mod_attn = SimpleNamespace(Qwen3VLVisionAttention=_FakeVisionAttention)
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(mod_attn)
    assert getattr(
        _FakeVisionAttention.forward, "_tpu_qwen3_vl_vision_attn_patch", False
    )

    # Re-patch exit
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(mod_attn)

    inst_attn = _FakeVisionAttention()
    # Call with TPU cu_seqlens
    tpu_cu = _make_fake_tpu_tensor(torch.tensor([0, 4]))
    inst_attn.forward(torch.ones(4, 8), cu_seqlens=tpu_cu, custom="foo")
    assert attn_called[-1][1].device.type == "cpu"

    # Call with CPU cu_seqlens
    inst_attn.forward(torch.ones(4, 8), cu_seqlens=torch.tensor([0, 4]))
    assert attn_called[-1][1].device.type == "cpu"

    # Call without cu_seqlens with TypeError fallback
    orig_bind = inspect.Signature.bind
    inspect.Signature.bind = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[assignment]
        TypeError("bind_partial fallback")
    )
    try:
        inst_attn.forward(torch.ones(4, 8))
    finally:
        inspect.Signature.bind = orig_bind

    # Call where forward signature does NOT contain cu_seqlens
    class _FakeVisionAttentionNoCu:
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    mod_attn_nocu = SimpleNamespace(Qwen3VLVisionAttention=_FakeVisionAttentionNoCu)
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(mod_attn_nocu)
    inst_attn_nocu = _FakeVisionAttentionNoCu()
    res_nocu = inst_attn_nocu.forward(torch.ones(2, 4))
    assert res_nocu.shape == (2, 4)

    # 3. _patch_qwen3_vl_vision_transformer
    tf_called = False

    class _FakeVisionTransformer:
        def forward(self, *args, **kwargs):
            nonlocal tf_called
            tf_called = True
            return args[0]

    mod_tf = SimpleNamespace(
        Qwen3VLVisionModel=_FakeVisionTransformer,
        Qwen3VisionTransformerPretrainedModel=SimpleNamespace(forward=None),
    )
    qwen3_vl_patch._patch_qwen3_vl_vision_transformer(mod_tf)
    assert getattr(
        _FakeVisionTransformer.forward,
        "_tpu_qwen3_vl_vision_transformer_patch",
        False,
    )
    # Call patched vision transformer forward
    inst_tf = _FakeVisionTransformer()
    inst_tf.forward(torch.ones(2, 2))
    assert tf_called is True
    # Re-patch exit
    qwen3_vl_patch._patch_qwen3_vl_vision_transformer(mod_tf)


def test_qwen3_vl_maybe_patch_top_level(monkeypatch):
    # 1. Non Qwen3-VL config
    cfg_non = SimpleNamespace(hf_config=None, model="other-model")
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(cfg_non) is False  # type: ignore[arg-type]

    cfg_q3 = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_vl"), model="qwen3_vl"
    )

    # 2. ImportError and no sys.modules match
    class _FakeImportFail:
        pass

    monkeypatch.setitem(
        sys.modules, "transformers.models.qwen3_vl.modeling_qwen3_vl", None
    )
    # Ensure no qwen3_vl in sys.modules has Qwen3VLModel
    monkeypatch.setattr(
        qwen3_vl_patch,
        "_patch_qwen3_vl_get_rope_index",
        lambda m: None,
    )
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(cfg_q3) is None  # type: ignore[arg-type]

    # 3. ImportError fallback finding module in sys.modules
    fake_q3_mod = ModuleType("my_custom_qwen3_vl_module")
    fake_q3_mod.Qwen3VLModel = SimpleNamespace()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "my_custom_qwen3_vl_module", fake_q3_mod)
    patched_mods: list[Any] = []
    monkeypatch.setattr(
        qwen3_vl_patch,
        "_patch_qwen3_vl_get_rope_index",
        lambda m: patched_mods.append(m),
    )
    monkeypatch.setattr(
        qwen3_vl_patch,
        "_patch_qwen3_vl_vision_attention",
        lambda m: None,
    )
    monkeypatch.setattr(
        qwen3_vl_patch,
        "_patch_qwen3_vl_vision_transformer",
        lambda m: None,
    )
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(cfg_q3) is None  # type: ignore[arg-type]
    assert len(patched_mods) == 1
    assert patched_mods[0] is fake_q3_mod

    # 4. Standard import success
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        fake_q3_mod,
    )
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(cfg_q3) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. merge_xplane.py tests (including all uncovered branches)
# ---------------------------------------------------------------------------


def _write_sample_xplane(
    path, rank=None, planes=("/device:TPU:0", "/host:metadata", "/host:CPU")
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    space = xpb.XSpace()
    space.hostnames.append(f"host-{rank or 0}")
    space.errors.append("sample error")
    space.warnings.append("sample warning")
    for p_name in planes:
        p = space.planes.add(id=1, name=p_name)
        if p_name == "/host:CPU":
            p.stat_metadata[1].id = 1
            p.stat_metadata[1].name = "process_id"
            p.stats.add(metadata_id=1, int64_value=rank or 0)
        # Line with events
        l1 = p.lines.add(id=10, name="pjrt-tpu-tasks/worker_0")
        l1.events.add(metadata_id=1, offset_ps=10, duration_ps=20)
        # Line with slash subsystem
        l2 = p.lines.add(id=20, name="subsys/thrd")
        l2.events.add(metadata_id=1, offset_ps=30, duration_ps=10)
        # Line without slash
        l3 = p.lines.add(id=30, name="thread_plain")
        l3.events.add(metadata_id=1, offset_ps=40, duration_ps=10)
        # Line without events
        p.lines.add(id=40, name="empty_line")
    with open(path, "wb") as f:
        f.write(space.SerializeToString())


def test_merge_xplane_data_classes_and_helpers(tmp_path):
    # Capture & Session slots
    cap = merge_xplane.Capture(path="/tmp/f.pb", rank=None, slot=0)
    assert cap.label == "file 0"
    assert hasattr(cap, "__slots__")

    cap1 = merge_xplane.Capture(path="/tmp/f1.pb", rank=1, slot=1)
    assert cap1.label == "rank 1"

    sess = merge_xplane.Session(
        label="test_phase",
        timestamp="2026_01_01",
        directory=str(tmp_path),
        captures=(cap, cap1),
    )
    assert sess.output_name == "test_phase_2026_01_01.xplane.pb"
    assert hasattr(sess, "__slots__")

    sess_no_ts = merge_xplane.Session(
        label="phase", timestamp="", directory=str(tmp_path), captures=()
    )
    assert sess_no_ts.output_name == "phase.xplane.pb"

    # missing_ranks: empty vs contiguous vs missing
    assert sess_no_ts.missing_ranks == []
    sess_missing = merge_xplane.Session(
        label="p",
        timestamp="",
        directory=str(tmp_path),
        captures=(
            merge_xplane.Capture(path="a", rank=0, slot=0),
            merge_xplane.Capture(path="b", rank=3, slot=3),
        ),
    )
    assert sess_missing.missing_ranks == [1, 2]

    # _varint multi-byte & single-byte
    assert merge_xplane._varint(1) == b"\x01"
    assert merge_xplane._varint(300) == b"\xac\x02"

    # _delimited with field_number < 16 and >= 16
    deliv_small = merge_xplane._delimited(1, b"payload")
    deliv_large = merge_xplane._delimited(20, b"payload")
    assert deliv_small.startswith(bytes([(1 << 3) | 2]))
    assert len(deliv_large) > len(b"payload")

    # _rank_of
    assert merge_xplane._rank_of("path/rank42_trace.xplane.pb") == 42
    assert merge_xplane._rank_of("path/unranked.xplane.pb") is None

    # _phase_dir & _session_identity
    p_dir = tmp_path / "my_phase" / "plugins" / "profile" / "2026_10_09_12_00"
    p_dir.mkdir(parents=True)
    assert merge_xplane._phase_dir(str(p_dir)) == str(tmp_path / "my_phase")
    assert merge_xplane._phase_dir(str(tmp_path)) is None

    assert merge_xplane._session_identity(str(p_dir)) == (
        "my_phase",
        "2026_10_09_12_00",
    )
    assert merge_xplane._session_identity(str(tmp_path)) == (tmp_path.name, "")

    # Root phase dir coverage: os.path.basename(phase_dir) == ""
    monkeypatch_phase = pytest.MonkeyPatch()
    monkeypatch_phase.setattr(merge_xplane, "_phase_dir", lambda d: "/")
    assert merge_xplane._session_identity("/plugins/profile/ts") == ("ts", "ts")
    monkeypatch_phase.undo()

    # _is_inside_session_dir
    assert merge_xplane._is_inside_session_dir(str(p_dir)) is True
    assert merge_xplane._is_inside_session_dir(str(tmp_path)) is False

    # _format_size
    assert merge_xplane._format_size(500) == "500 B"
    assert merge_xplane._format_size(50_000) == "49 KB"
    assert merge_xplane._format_size(5_000_000) == "5 MB"

    # _is_within
    assert merge_xplane._is_within(str(p_dir), str(tmp_path)) is True
    assert merge_xplane._is_within(str(tmp_path), str(tmp_path)) is True
    assert merge_xplane._is_within(str(tmp_path), str(p_dir)) is False


def test_merge_xplane_discovery_and_tagging(tmp_path):
    sess_dir = tmp_path / "phase" / "plugins" / "profile" / "2026_01_01_00_00"
    p0 = sess_dir / "rank0_host.xplane.pb"
    p1 = sess_dir / "unranked_host.xplane.pb"
    _write_sample_xplane(str(p0), rank=0)
    _write_sample_xplane(str(p1), rank=None)

    # 1. FileNotFoundError when path does not exist
    with pytest.raises(FileNotFoundError):
        merge_xplane.discover_sessions([str(tmp_path / "non_existent")])

    # 2. discover_sessions with explicit files and skip
    sessions = merge_xplane.discover_sessions([str(p0), str(p1)])
    assert len(sessions) == 1
    assert len(sessions[0].captures) == 2

    # discover_sessions with explicit file within skip
    sess_explicit_skip = merge_xplane.discover_sessions(
        [str(p0)], skip=(str(sess_dir),)
    )
    assert len(sess_explicit_skip) == 0

    # discover_sessions with directory walk and skip
    sess_skipped = merge_xplane.discover_sessions(
        [str(tmp_path)], skip=(str(sess_dir),)
    )
    assert len(sess_skipped) == 0

    # 3. _make_session with unranked collision handling
    s2 = merge_xplane._make_session(str(sess_dir), [str(p0), str(p1)])
    slots = {c.slot for c in s2.captures}
    assert len(slots) == 2

    # 4. _tag_host_cpu_plane and _rewrite_plane_name branches
    plane_cpu = xpb.XPlane()
    plane_cpu.name = "/host:CPU"
    cap_slot0 = merge_xplane.Capture(path="a", rank=0, slot=0)
    local_to_global: dict[tuple[str, int], int] = {}
    next_id: dict[str, int] = {}

    # /host:CPU with slot 0 (no slot added to name) and existing process_id stat
    merge_xplane._rewrite_plane_name(plane_cpu, cap_slot0, local_to_global, next_id)
    assert plane_cpu.name == "/host:CPU"

    # /device:TPU:0 first time and second time (key in local_to_global branch)
    plane_tpu = xpb.XPlane()
    plane_tpu.name = "/device:TPU:0"
    merge_xplane._rewrite_plane_name(plane_tpu, cap_slot0, local_to_global, next_id)
    assert plane_tpu.name == "/device:TPU:0"
    merge_xplane._rewrite_plane_name(plane_tpu, cap_slot0, local_to_global, next_id)
    assert plane_tpu.name == "/device:TPU:0"

    # Non-device, non-CPU plane: tagged with capture label
    plane_other = xpb.XPlane()
    plane_other.name = "Custom Timeline"
    merge_xplane._rewrite_plane_name(plane_other, cap_slot0, local_to_global, next_id)
    assert plane_other.name == "Custom Timeline [rank 0]"

    # _tag_host_cpu_plane when process_id is not in stat_metadata
    plane_cpu2 = xpb.XPlane()
    plane_cpu2.name = "/host:CPU"
    cap_slot2 = merge_xplane.Capture(path="b", rank=2, slot=2)
    merge_xplane._tag_host_cpu_plane(plane_cpu2, cap_slot2)
    assert plane_cpu2.name == "/host:CPU [2]"
    assert any(m.name == "process_id" for m in plane_cpu2.stat_metadata.values())


def test_merge_xplane_merge_session_and_main(tmp_path, capsys):
    sess_dir = tmp_path / "phase" / "plugins" / "profile" / "2026_01_01_00_00"
    p0 = sess_dir / "rank0_trace.xplane.pb"
    p2 = sess_dir / "rank2_trace.xplane.pb"  # missing rank 1
    _write_sample_xplane(str(p0), rank=0)
    _write_sample_xplane(str(p2), rank=2)

    sessions = merge_xplane.discover_sessions([str(sess_dir)])
    assert len(sessions) == 1
    session = sessions[0]

    out_file = tmp_path / "merged_out.xplane.pb"
    res = merge_xplane.merge_session(session, str(out_file))
    assert res.planes > 0
    assert res.deduplicated == 1  # 1 duplicate metadata dropped

    # Duplicate hostname in space and between spaces
    sess_dup_dir = tmp_path / "dup_sess" / "plugins" / "profile" / "ts"
    sess_dup_dir.mkdir(parents=True)
    p_dup0 = sess_dup_dir / "r0.xplane.pb"
    p_dup1 = sess_dup_dir / "r1.xplane.pb"
    sp_dup0 = xpb.XSpace()
    sp_dup0.hostnames.extend(["host-common", "host-common"])  # duplicate in same space
    p_dup0.write_bytes(sp_dup0.SerializeToString())
    sp_dup1 = xpb.XSpace()
    sp_dup1.hostnames.append("host-common")  # duplicate across spaces
    p_dup1.write_bytes(sp_dup1.SerializeToString())
    s_dup = merge_xplane.discover_sessions([str(sess_dup_dir)])[0]
    out_dup = tmp_path / "dup_out.xplane.pb"
    merge_xplane.merge_session(s_dup, str(out_dup))

    # Partial file cleanup on error in merge_session
    p_bad = sess_dup_dir / "bad.xplane.pb"
    p_bad.write_bytes(b"invalid protobuf data")
    s_bad = merge_xplane.Session(
        label="bad",
        timestamp="ts",
        directory=str(tmp_path),
        captures=(merge_xplane.Capture(path=str(p_bad), rank=0, slot=0),),
    )
    dest_bad = tmp_path / "bad_out.xplane.pb"
    with pytest.raises(Exception, match=".*"):
        merge_xplane.merge_session(s_bad, str(dest_bad))
    assert not os.path.exists(f"{dest_bad}.partial")

    # main: refusing to write inside session directory
    rc_refuse = merge_xplane.main([str(sess_dir), "-d", str(sess_dir)])
    assert rc_refuse == 2
    err = capsys.readouterr().err
    assert "Refusing to write into" in err

    # main: no xplane files found
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    rc_none = merge_xplane.main([str(empty_dir)])
    assert rc_none == 2
    assert "No .xplane.pb files found" in capsys.readouterr().err

    # main: --output with multiple sessions
    sess_dir2 = tmp_path / "phase2" / "plugins" / "profile" / "2026_01_02_00_00"
    p_other = sess_dir2 / "rank0.xplane.pb"
    _write_sample_xplane(str(p_other), rank=0)
    rc_multi = merge_xplane.main([str(sess_dir), str(sess_dir2), "-o", str(out_file)])
    assert rc_multi == 2
    assert "--output takes a single file" in capsys.readouterr().err

    # main: --dry-run
    rc_dry = merge_xplane.main([str(sess_dir), "--dry-run"])
    assert rc_dry == 0
    out_dry = capsys.readouterr().out
    assert "would write" in out_dry
    assert "WARNING: no trace for rank(s) 1" in out_dry

    # main: successful execution with -o
    rc_ok = merge_xplane.main([str(sess_dir), "-o", str(out_file)])
    assert rc_ok == 0
    out_ok = capsys.readouterr().out
    assert "Found 1 capture session" in out_ok
    assert "planes" in out_ok

    # main: successful execution with -d
    out_dir = tmp_path / "custom_out"
    rc_ok2 = merge_xplane.main([str(sess_dir), "-d", str(out_dir)])
    assert rc_ok2 == 0

    # main: successful execution without output flags (default phase dir)
    rc_ok3 = merge_xplane.main([str(sess_dir)])
    assert rc_ok3 == 0
    capsys.readouterr()  # clear buffer

    # Session without missing ranks: covers 611->618 (if session.missing_ranks is False)
    sess_contig_dir = tmp_path / "contig" / "plugins" / "profile" / "2026_01_03_00_00"
    p_c0 = sess_contig_dir / "rank0.xplane.pb"
    _write_sample_xplane(str(p_c0), rank=0)
    rc_contig = merge_xplane.main(
        [str(sess_contig_dir), "-o", str(tmp_path / "c_out.xplane.pb")]
    )
    assert rc_contig == 0
    out_contig = capsys.readouterr().out
    assert "WARNING: no trace for rank" not in out_contig

    # main where total_out == total_in: covers 653->660
    # (if total_in and total_out != total_in is False)
    in_size = os.path.getsize(str(p_c0))
    monkeypatch_equal = pytest.MonkeyPatch()
    monkeypatch_equal.setattr(
        merge_xplane,
        "merge_session",
        lambda sess, dest: merge_xplane.MergeResult(
            path=dest,
            size=in_size,
            planes=1,
            planes_read=1,
            deduplicated=0,
        ),
    )
    rc_equal = merge_xplane.main(
        [str(sess_contig_dir), "-o", str(tmp_path / "eq_out.xplane.pb")]
    )
    assert rc_equal == 0
    out_eq = capsys.readouterr().out
    assert "smaller" not in out_eq and "larger" not in out_eq
    monkeypatch_equal.undo()

    # main where total_out > total_in ("larger"):
    monkeypatch_larger = pytest.MonkeyPatch()
    monkeypatch_larger.setattr(
        merge_xplane,
        "merge_session",
        lambda sess, dest: merge_xplane.MergeResult(
            path=dest,
            size=in_size * 2,
            planes=1,
            planes_read=1,
            deduplicated=0,
        ),
    )
    rc_larger = merge_xplane.main(
        [str(sess_contig_dir), "-o", str(tmp_path / "lg_out.xplane.pb")]
    )
    assert rc_larger == 0
    assert "larger" in capsys.readouterr().out
    monkeypatch_larger.undo()

    # Coverage for line 670: sys.exit(main()) when run as __main__
    import runpy

    monkeypatch_main = pytest.MonkeyPatch()
    monkeypatch_main.setattr(sys, "argv", ["merge_xplane", "-h"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("vllm_torchtpu.tools.merge_xplane", run_name="__main__")
    assert exc_info.value.code == 0
    monkeypatch_main.undo()
