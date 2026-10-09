# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Track TD7.65: vision_attention and Qwen-VL patches."""

import builtins
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu.layers.adapter.vision_attention as va_mod
from vllm_torchtpu.layers.adapter.vision_attention import (
    TpuMMEncoderAttention,
    _vision_flash_core,
)
from vllm_torchtpu.models.vllm import qwen2_5_vl_patch, qwen3_vl_patch


class _FakeCPUFromTPU(torch.Tensor):
    """CPU tensor from _FakeTPUTensor.cpu() accepting fake TPU device in .to()."""

    def to(self, *args, **kwargs):
        new_args = tuple(
            torch.device("cpu")
            if getattr(a, "type", None) == "tpu" and not isinstance(a, torch.device)
            else a
            for a in args
        )
        if getattr(kwargs.get("device"), "type", None) == "tpu" and not isinstance(
            kwargs.get("device"), torch.device
        ):
            kwargs = {**kwargs, "device": torch.device("cpu")}
        out = super().to(*new_args, **kwargs)
        return out.as_subclass(_FakeTPUTensor)


class _FakeTPUTensor(torch.Tensor):
    """CPU-backed tensor reporting device.type == 'tpu' for unit testing."""

    _fake_device = SimpleNamespace(type="tpu")

    @property
    def device(self):
        return self._fake_device

    def cpu(self):
        return self.as_subclass(_FakeCPUFromTPU)

    def to(self, *args, **kwargs):
        new_args = tuple(
            torch.device("cpu")
            if getattr(a, "type", None) == "tpu" and not isinstance(a, torch.device)
            else a
            for a in args
        )
        if getattr(kwargs.get("device"), "type", None) == "tpu" and not isinstance(
            kwargs.get("device"), torch.device
        ):
            kwargs = {**kwargs, "device": torch.device("cpu")}
        out = super().to(*new_args, **kwargs)
        return out.as_subclass(_FakeTPUTensor)


def _make_tpu_tensor(data, dtype=torch.float32) -> _FakeTPUTensor:
    return torch.tensor(data, dtype=dtype).as_subclass(_FakeTPUTensor)


def test_vision_flash_core_and_init_registration(monkeypatch):
    captured = {}

    def fake_flash_attention(
        q, k, v, *, segment_ids, causal, sm_scale, vmem_limit_bytes
    ):
        captured["segment_ids"] = segment_ids
        captured["causal"] = causal
        captured["sm_scale"] = sm_scale
        captured["vmem_limit_bytes"] = vmem_limit_bytes
        return q

    monkeypatch.setattr(va_mod, "flash_attention", fake_flash_attention)
    q = torch.ones(1, 2, 128, 4)
    seg = torch.ones(1, 128, dtype=torch.int32)
    res = _vision_flash_core(q, q, q, seg, seg)
    assert res is q
    assert captured["causal"] is False
    assert captured["sm_scale"] == 1.0

    # Test TpuMMEncoderAttention.__init__ when _vision_op is None vs already populated.
    fake_fns = []

    class _FakeJaxOp:
        def register_fake(self, fn):
            fake_fns.append(fn)

        def __call__(self, q_in, k_in, v_in, q_s, kv_s):
            return q_in

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", None)
    monkeypatch.setattr(va_mod.pallas, "jax_op", lambda name, fn: _FakeJaxOp())
    monkeypatch.setattr(
        va_mod.MMEncoderAttention, "__init__", lambda self, *a, **kw: None
    )

    attn1 = TpuMMEncoderAttention(2, 4)
    assert TpuMMEncoderAttention._vision_op is not None
    assert len(fake_fns) == 1
    fake_out = fake_fns[0](q, q, q, seg, seg)
    assert fake_out.shape == q.shape

    # Second initialization reuses _vision_op without re-registering.
    attn2 = TpuMMEncoderAttention(2, 4)
    assert attn2._vision_op is attn1._vision_op
    assert len(fake_fns) == 1


def test_vision_attention_forward_oot_and_flash_branches(monkeypatch):
    recorded_segs = []

    def fake_op(q, k, v, q_seg, kv_seg):
        recorded_segs.append(q_seg.clone())
        return q

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", fake_op)

    attn = TpuMMEncoderAttention.__new__(TpuMMEncoderAttention)
    attn.num_heads = 2
    attn.num_kv_heads = 1
    attn.head_size = 4
    attn.scale = 1.0

    # 1. Head mismatch -> fallback to forward_native.
    monkeypatch.setattr(
        attn,
        "forward_native",
        lambda q, k, v, cu, max_s, seq_l: torch.full_like(q, 3.0),
    )
    q3 = torch.zeros(1, 4, 8)
    assert torch.equal(attn.forward_oot(q3, q3, q3), torch.full_like(q3, 3.0))

    # 2. Equal heads, 3D input (is_reshaped=True) and 4D input (is_reshaped=False).
    attn.num_kv_heads = 2
    monkeypatch.setattr(
        attn,
        "view_qkv_to_4d",
        lambda q, k, v, bsz, q_len, kv_len: (
            q.reshape(bsz, q_len, 2, 4),
            k.reshape(bsz, kv_len, 2, 4),
            v.reshape(bsz, kv_len, 2, 4),
        ),
    )
    out_3d = attn.forward_oot(
        torch.ones(1, 6, 8),
        torch.ones(1, 6, 8),
        torch.ones(1, 6, 8),
        cu_seqlens=torch.tensor([0, 6], dtype=torch.int64),
    )
    assert out_3d.shape == (1, 6, 8)

    # 4D input with exact multiple of 128 (pad == 0) and cu_seqlens is None.
    q4_exact = torch.ones(1, 128, 2, 4)
    out_4d_exact = attn.forward_oot(q4_exact, q4_exact, q4_exact, cu_seqlens=None)
    assert out_4d_exact.shape == (1, 128, 2, 4)
    assert (recorded_segs[-1] == 1).all()

    # 4D input with pad > 0 (seq_len=5) and cu_seqlens is None.
    q4_padded = torch.ones(1, 5, 2, 4)
    out_4d_padded = attn.forward_oot(q4_padded, q4_padded, q4_padded, cu_seqlens=None)
    assert out_4d_padded.shape == (1, 5, 2, 4)
    assert recorded_segs[-1][0, :5].tolist() == [1] * 5
    assert recorded_segs[-1][0, 5:].tolist() == [0] * (128 - 5)


def test_qwen2_5_vl_patch_full_coverage(monkeypatch):
    assert qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(None) is False
    assert (
        qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(
            SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
        )
        is False
    )

    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as Model,
    )

    monkeypatch.setattr(
        Model, "encoder_cudagraph_forward", lambda self, values, path="default": values
    )
    monkeypatch.setattr(
        Model,
        "encoder_eager_forward",
        lambda self, mm_kwargs, path="default": mm_kwargs,
    )

    assert (
        qwen2_5_vl_patch.maybe_patch_qwen2_5_vl(
            SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen2_5_vl"))
        )
        is None
    )

    dummy_self = SimpleNamespace(
        visual=SimpleNamespace(dtype=torch.float32),
        input_norm=lambda x, dtype: x * 2.0,
        get_input_modality=lambda kw: "image" if "pixel_values" in kw else "video",
    )
    graph_out = Model.encoder_cudagraph_forward(
        dummy_self, {"pixel_values": torch.tensor([1.0, 2.0])}
    )
    assert torch.equal(graph_out["pixel_values"], torch.tensor([2.0, 4.0]))

    img_out = Model.encoder_eager_forward(
        dummy_self, {"pixel_values": torch.tensor([3.0])}
    )
    assert torch.equal(img_out["pixel_values"], torch.tensor([6.0]))

    vid_out = Model.encoder_eager_forward(
        dummy_self, {"pixel_values_videos": torch.tensor([4.0])}
    )
    assert torch.equal(vid_out["pixel_values_videos"], torch.tensor([8.0]))


def test_qwen3_vl_is_model_and_patched_ops_full_coverage():
    assert not qwen3_vl_patch._is_qwen3_vl_model(None)
    assert qwen3_vl_patch._is_qwen3_vl_model(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
            model="x",
        )
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="other", architectures=["Qwen3VLForConditionalGeneration"]
            ),
            model="x",
        )
    )
    assert qwen3_vl_patch._is_qwen3_vl_model(
        SimpleNamespace(hf_config=None, model="org/Qwen3-VL-8B")
    )
    assert not qwen3_vl_patch._is_qwen3_vl_model(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="other", architectures=["Other"]),
            model="other",
        )
    )

    # _patched_masked_scatter_ (in-place) on TPU and CPU
    tpu_dst = _make_tpu_tensor([10.0, 20.0, 30.0])
    mask = torch.tensor([True, False, True])
    with pytest.raises(RuntimeError, match="Number of source elements"):
        qwen3_vl_patch._patched_masked_scatter_(tpu_dst, mask, torch.tensor([1.0]))
    out_inplace = qwen3_vl_patch._patched_masked_scatter_(
        tpu_dst, mask, _make_tpu_tensor([1.0, 2.0, 3.0])
    )
    assert out_inplace is tpu_dst
    assert torch.equal(
        tpu_dst.as_subclass(torch.Tensor), torch.tensor([1.0, 20.0, 2.0])
    )

    cpu_dst = torch.tensor([10.0, 20.0, 30.0])
    qwen3_vl_patch._patched_masked_scatter_(cpu_dst, mask, torch.tensor([5.0, 6.0]))
    assert torch.equal(cpu_dst, torch.tensor([5.0, 20.0, 6.0]))

    # _patched_masked_scatter (out-of-place) on TPU and CPU
    tpu_in = _make_tpu_tensor([10.0, 20.0, 30.0])
    with pytest.raises(RuntimeError, match="Number of source elements"):
        qwen3_vl_patch._patched_masked_scatter(tpu_in, mask, torch.tensor([1.0]))
    out_oop = qwen3_vl_patch._patched_masked_scatter(
        tpu_in, mask, _make_tpu_tensor([7.0, 8.0])
    )
    assert torch.equal(
        out_oop.as_subclass(torch.Tensor), torch.tensor([7.0, 20.0, 8.0])
    )
    cpu_oop = qwen3_vl_patch._patched_masked_scatter(
        torch.tensor([10.0, 20.0, 30.0]), mask, torch.tensor([7.0, 8.0])
    )
    assert torch.equal(cpu_oop, torch.tensor([7.0, 20.0, 8.0]))

    # _patched_repeat_interleave on TPU (int tensor & int repeats) and CPU
    tpu_int = _make_tpu_tensor([1, 2], dtype=torch.int32)
    rep_tensor = _make_tpu_tensor([2, 1], dtype=torch.int32)
    res_rep1 = qwen3_vl_patch._patched_repeat_interleave(tpu_int, rep_tensor, dim=0)
    assert res_rep1.as_subclass(torch.Tensor).tolist() == [1, 1, 2]
    res_rep2 = qwen3_vl_patch._patched_repeat_interleave(tpu_int, 2, dim=0)
    assert res_rep2.as_subclass(torch.Tensor).tolist() == [1, 1, 2, 2]
    res_rep_cpu = qwen3_vl_patch._patched_repeat_interleave(
        torch.tensor([1, 2]), 2, dim=0
    )
    assert res_rep_cpu.tolist() == [1, 1, 2, 2]

    # _patched_cumsum and _patched_tensor_cumsum on TPU and CPU
    res_cs_tpu = qwen3_vl_patch._patched_cumsum(tpu_int, dim=0)
    assert res_cs_tpu.as_subclass(torch.Tensor).tolist() == [1, 3]
    res_cs_cpu = qwen3_vl_patch._patched_cumsum(torch.tensor([1, 2]), dim=0)
    assert res_cs_cpu.tolist() == [1, 3]

    res_tcs_tpu = qwen3_vl_patch._patched_tensor_cumsum(tpu_int, dim=0)
    assert res_tcs_tpu.as_subclass(torch.Tensor).tolist() == [1, 3]
    res_tcs_cpu = qwen3_vl_patch._patched_tensor_cumsum(torch.tensor([1, 2]), dim=0)
    assert res_tcs_cpu.tolist() == [1, 3]


def test_scoped_qwen3_vl_torch_ops_tensorbase_variants(monkeypatch):
    # 1. Default C-extension _TensorBase (raises TypeError on setattr) + nested scope
    with (
        qwen3_vl_patch._scoped_qwen3_vl_torch_ops(),
        qwen3_vl_patch._scoped_qwen3_vl_torch_ops(),
    ):
        assert torch.cumsum is qwen3_vl_patch._patched_cumsum

    # 2. Mutable _TensorBase with non-None masked_scatter and masked_scatter_
    fake_tb = SimpleNamespace(masked_scatter="orig_ms", masked_scatter_="orig_ms_")
    monkeypatch.setattr(torch, "_C", SimpleNamespace(_TensorBase=fake_tb))
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert fake_tb.masked_scatter is qwen3_vl_patch._patched_masked_scatter
        assert fake_tb.masked_scatter_ is qwen3_vl_patch._patched_masked_scatter_
    assert fake_tb.masked_scatter == "orig_ms"
    assert fake_tb.masked_scatter_ == "orig_ms_"

    # 3. Mutable _TensorBase with None attributes
    fake_tb_none = SimpleNamespace(masked_scatter=None, masked_scatter_=None)
    monkeypatch.setattr(torch, "_C", SimpleNamespace(_TensorBase=fake_tb_none))
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert fake_tb_none.masked_scatter is None
        assert fake_tb_none.masked_scatter_ is None

    # 4. _TensorBase that raises TypeError on restore in finally block
    class _RestoreFailingTB:
        masked_scatter = "orig"
        masked_scatter_ = "orig_"
        fail_now = False

        def __setattr__(self, name, value):
            if self.fail_now:
                raise TypeError("immutable on restore")
            super().__setattr__(name, value)

    failing_tb = _RestoreFailingTB()
    monkeypatch.setattr(torch, "_C", SimpleNamespace(_TensorBase=failing_tb))
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        failing_tb.fail_now = True

    # 5. has_tb = False (no _TensorBase on torch._C)
    monkeypatch.setattr(torch, "_C", SimpleNamespace())
    with qwen3_vl_patch._scoped_qwen3_vl_torch_ops():
        assert torch.masked_scatter is qwen3_vl_patch._patched_masked_scatter


def test_patch_qwen3_vl_rope_attention_transformer_and_entrypoint(monkeypatch):
    # Guard branches for _patch_qwen3_vl_get_rope_index
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(SimpleNamespace())
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(
        SimpleNamespace(Qwen3VLModel=SimpleNamespace())
    )

    class _DummyModel:
        def get_rope_index(
            self,
            input_ids,
            image_grid_thw=None,
            video_grid_thw=None,
            extra=None,
            **kwargs,
        ):
            return (
                input_ids,
                [image_grid_thw, video_grid_thw],
                {"extra": extra, "kw": kwargs.get("kw_tensor"), "scalar": 5},
                99,
            )

    modeling = SimpleNamespace(Qwen3VLModel=_DummyModel)
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(modeling)
    # Second call hits already-patched guard.
    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(modeling)

    inst = _DummyModel()
    tpu_ids = _make_tpu_tensor([1, 2], dtype=torch.int64)
    # Call with 1D image_grid_thw, empty video_grid_thw, nested containers, and kwargs
    out_tpu = inst.get_rope_index(
        tpu_ids,
        image_grid_thw=_make_tpu_tensor([1, 2, 2], dtype=torch.int64),
        video_grid_thw=torch.empty(0),
        extra=(_make_tpu_tensor([3]), {"nested": [_make_tpu_tensor([4]), "str"]}),
        kw_tensor=_make_tpu_tensor([5]),
    )
    assert out_tpu[1][0].shape == (1, 3)
    assert out_tpu[1][1] is None
    assert out_tpu[3] == 99

    # Call with CPU-only inputs, None grid, 2D tensor grid, and list grid
    out_cpu = inst.get_rope_index(
        torch.tensor([1, 2]),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
    )
    assert out_cpu[1][1] is None
    out_cpu_list = inst.get_rope_index(
        torch.tensor([1, 2]),
        image_grid_thw=None,
        video_grid_thw=[1, 2, 2],
    )
    assert out_cpu_list[1][1] == [1, 2, 2]

    # Also trigger sig.bind TypeError fallback (bind_partial) by omitting a required arg
    def _strict_rope(self, req_arg, image_grid_thw=None, video_grid_thw=None):
        return (req_arg, image_grid_thw, video_grid_thw)

    class _StrictModel:
        get_rope_index = _strict_rope

    qwen3_vl_patch._patch_qwen3_vl_get_rope_index(
        SimpleNamespace(Qwen3VLModel=_StrictModel)
    )
    strict_inst = _StrictModel()
    with pytest.raises(TypeError):
        # Exercises except TypeError -> bind_partial when required arg is omitted
        strict_inst.get_rope_index(
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            video_grid_thw=[1, 2, 2],
        )

    # _patch_qwen3_vl_vision_attention guards and branches
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(SimpleNamespace())
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(
        SimpleNamespace(Qwen3VLVisionAttention=SimpleNamespace())
    )

    class _DummyVisionAttn:
        def forward(self, hidden_states, cu_seqlens=None, **kwargs):
            return hidden_states, cu_seqlens, kwargs

    class _DummyVisionAttnNoCu:
        def forward(self, hidden_states):
            return hidden_states

    attn_mod = SimpleNamespace(Qwen3VLVisionAttention=_DummyVisionAttn)
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(attn_mod)
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(attn_mod)

    attn_inst = _DummyVisionAttn()
    tpu_cu = _make_tpu_tensor([0, 4], dtype=torch.int32)
    _, out_cu, _ = attn_inst.forward(torch.ones(4, 2), cu_seqlens=tpu_cu, flag=True)
    assert out_cu.device.type == "cpu"
    # CPU cu_seqlens stays untouched
    cpu_cu = torch.tensor([0, 4], dtype=torch.int32)
    _, out_cu_cpu, _ = attn_inst.forward(torch.ones(4, 2), cu_seqlens=cpu_cu)
    assert out_cu_cpu is cpu_cu
    # Trigger bind_partial on _DummyVisionAttn by omitting hidden_states
    with pytest.raises(TypeError):
        attn_inst.forward()

    # Attention without cu_seqlens parameter in signature
    qwen3_vl_patch._patch_qwen3_vl_vision_attention(
        SimpleNamespace(Qwen3VLVisionAttention=_DummyVisionAttnNoCu)
    )
    assert torch.equal(
        _DummyVisionAttnNoCu().forward(torch.ones(2, 2)), torch.ones(2, 2)
    )

    # _patch_qwen3_vl_vision_transformer with missing forward and valid forward
    class _NoForwardTransformer:
        forward = None

    class _ValidTransformer:
        def forward(self, x):
            return x + 1

    vt_mod = SimpleNamespace(
        Qwen3VLVisionModel=_NoForwardTransformer,
        Qwen3_VisionTransformer=_ValidTransformer,
        Qwen3VisionTransformer=_ValidTransformer,
    )
    qwen3_vl_patch._patch_qwen3_vl_vision_transformer(vt_mod)
    qwen3_vl_patch._patch_qwen3_vl_vision_transformer(vt_mod)
    assert torch.equal(_ValidTransformer().forward(torch.zeros(2)), torch.ones(2))

    # maybe_patch_qwen3_vl entrypoint: non-qwen3 returns False
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(None) is False

    q3_cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
        model="Qwen/Qwen3-VL",
    )
    orig_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if name == "transformers.models.qwen3_vl.modeling_qwen3_vl":
            raise ImportError("no qwen3_vl")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    # Remove any cached qwen3_vl module in sys.modules -> hits warning & returns None
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if "qwen3_vl" in k}
    try:
        assert qwen3_vl_patch.maybe_patch_qwen3_vl(q3_cfg) is None
        # Register fake qwen3_vl modules in sys.modules (empty and full)
        sys.modules["fake_qwen3_vl_empty"] = SimpleNamespace()
        sys.modules["fake_qwen3_vl_full"] = SimpleNamespace(
            Qwen3VLModel=_DummyModel,
            Qwen3VLVisionAttention=_DummyVisionAttn,
            Qwen3VLVisionModel=_ValidTransformer,
        )
        assert qwen3_vl_patch.maybe_patch_qwen3_vl(q3_cfg) is None
    finally:
        sys.modules.pop("fake_qwen3_vl_empty", None)
        sys.modules.pop("fake_qwen3_vl_full", None)
        sys.modules.update(saved_mods)

    # Also test normal import path when module is importable
    fake_pkg = SimpleNamespace(
        Qwen3VLModel=_DummyModel,
        Qwen3VLVisionAttention=_DummyVisionAttn,
        Qwen3VLVisionModel=_ValidTransformer,
    )
    monkeypatch.setattr(builtins, "__import__", orig_import)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        fake_pkg,
    )
    import transformers.models.qwen3_vl as q3_parent

    monkeypatch.setattr(q3_parent, "modeling_qwen3_vl", fake_pkg, raising=False)
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(q3_cfg) is None
