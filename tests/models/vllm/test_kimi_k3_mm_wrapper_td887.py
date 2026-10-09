# SPDX-License-Identifier: Apache-2.0
"""TD8.87: CPU coverage for KimiK3ForConditionalGeneration's host methods.

The wrapper is built with ``__new__`` and given fake vision/language parts,
so the tests exercise the wrapper's own input validation, encoder-graph
plumbing and multimodal embedding merge without loading a model.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers

from vllm_torchtpu.models.vllm.kimi_k3 import model as kimi_model
from vllm_torchtpu.models.vllm.kimi_k3.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
)

HID = 4


class _VisionTower(torch.nn.Module):
    """Holds one bf16 parameter so the wrapper can read the tower's dtype."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))


def _wrapper(**attrs):
    wrapper = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    for name, value in attrs.items():
        setattr(wrapper, name, value)
    return wrapper


def test_set_aux_hidden_state_layers():
    seen = []
    lm = SimpleNamespace(set_aux_hidden_state_layers=seen.append)
    _wrapper(language_model=lm).set_aux_hidden_state_layers((1, 2))
    assert seen == [(1, 2)]
    with pytest.raises(RuntimeError, match="does not expose aux"):
        _wrapper(language_model=SimpleNamespace()).set_aux_hidden_state_layers((1,))


def _media_wrapper():
    return _wrapper(vision_tower=_VisionTower())


def test_parse_media_input_variants():
    wrapper = _media_wrapper()
    assert wrapper._parse_and_validate_media_input() is None

    grid = torch.tensor([[1, 2, 2]])
    # Pixel patches are (num_patches, 3, patch, patch).
    parsed = wrapper._parse_and_validate_media_input(
        pixel_values=[torch.ones(2, 3, 2, 2), torch.ones(1, 3, 2, 2)], grid_thws=grid
    )
    assert parsed["pixel_values"].shape == (3, 3, 2, 2)
    assert parsed["pixel_values"].dtype == torch.bfloat16
    assert parsed["grid_thws"].tolist() == [[1, 2, 2]]

    # A rank-5 batch folds its first two dims together.
    parsed = wrapper._parse_and_validate_media_input(
        pixel_values=torch.ones(2, 2, 3, 2, 2), grid_thws=grid.view(1, 1, 3)
    )
    assert parsed["pixel_values"].shape == (4, 3, 2, 2)
    assert parsed["grid_thws"].shape == (1, 3)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"pixel_values": "x"}, TypeError, "pixel_values must be a tensor"),
        (
            {"pixel_values": torch.ones(1, 3, 2, 2), "grid_thws": [1]},
            TypeError,
            "grid_thws",
        ),
        (
            {"pixel_values": torch.ones(1, 3, 2, 2), "grid_thws": torch.ones(2, 4)},
            ValueError,
            "unexpected grid_thws shape",
        ),
    ],
)
def test_parse_media_input_rejects(kwargs, error, match):
    with pytest.raises(error, match=match):
        _media_wrapper()._parse_and_validate_media_input(**kwargs)


@pytest.mark.parametrize("data_parallel", [False, True])
def test_embed_multimodal(monkeypatch, data_parallel):
    wrapper = _media_wrapper()
    wrapper.use_data_parallel = data_parallel
    wrapper.mm_projector = "projector"
    calls = []

    def tower_forward(tower, pixels, grids, *, mm_projector, use_data_parallel):
        calls.append(
            (
                tower,
                pixels.shape,
                mm_projector,
                use_data_parallel,
                kimi_model.vision_utils.tensor_model_parallel_all_gather,
            )
        )
        return ["embeds"]

    monkeypatch.setattr(kimi_model, "vision_tower_forward", tower_forward)
    assert wrapper.embed_multimodal() is None
    out = wrapper.embed_multimodal(
        pixel_values=torch.ones(2, 3, 2, 2), grid_thws=torch.tensor([[1, 1, 2]])
    )
    assert out == ["embeds"]
    ((tower, shape, projector, dp, gather),) = calls
    assert tower is wrapper.vision_tower and shape == (2, 3, 2, 2)
    assert projector == "projector" and dp is data_parallel
    # The DP path swaps in the TPU gather only for the duration of the call.
    assert (gather is kimi_model._tpu_tp_all_gather) is data_parallel
    assert (
        kimi_model.vision_utils.tensor_model_parallel_all_gather
        is not kimi_model._tpu_tp_all_gather
    )


def test_encoder_graph_static_hooks():
    wrapper = _wrapper(hidden_size=HID)
    config = wrapper.get_encoder_cudagraph_config()
    assert config.modalities == ["image"]
    assert config.out_hidden_size == HID
    assert "merge_gather_idx" in config.buffer_keys
    assert wrapper.get_input_modality({"anything": 1}) == "image"
    assert wrapper.get_max_frames_per_video() == 1
    assert KimiK3ForConditionalGeneration._grid_thw_list([[1, 2, 3]]) == [[1, 2, 3]]
    assert KimiK3ForConditionalGeneration._grid_thw_list(torch.tensor([[1, 2, 3]])) == [
        [1, 2, 3]
    ]


def test_prepare_encoder_cudagraph_replay_buffers():
    seen = {}

    def metadata(grid, *, max_batch_size, device):
        seen.update(grid=grid, max_batch_size=max_batch_size, device=device)
        return {"cu_seqlens": "cu"}

    wrapper = _wrapper(
        vision_tower=SimpleNamespace(prepare_encoder_cudagraph_metadata=metadata)
    )
    pixels = torch.ones(4, 3)
    buffers = wrapper.prepare_encoder_cudagraph_replay_buffers(
        {"pixel_values": pixels, "grid_thws": torch.tensor([[1, 2, 2]])}, 8, 1
    )
    assert isinstance(buffers, EncoderCudaGraphReplayBuffers)
    assert buffers.values["cu_seqlens"] == "cu"
    assert buffers.values["pixel_values"] is pixels
    assert seen == {"grid": [[1, 2, 2]], "max_batch_size": 8, "device": pixels.device}


def test_encoder_cudagraph_forward(monkeypatch):
    order = []

    def patch_embed(pixels, grid, *, pos_embeds):
        order.append(("embed", grid, pos_embeds))
        return pixels + 1

    def encoder(hidden, grid, *, encoder_metadata):
        order.append(("encoder", grid, dict(encoder_metadata)))
        return hidden * 2

    monkeypatch.setattr(
        kimi_model,
        "tpool_patch_merger_packed",
        lambda hidden, idx: (order.append(("merge", idx)), hidden.sum(0))[1],
    )
    wrapper = _wrapper(
        vision_tower=SimpleNamespace(patch_embed=patch_embed, encoder=encoder),
        mm_projector=lambda merged: merged - 3,
    )
    inputs = {
        "pixel_values": torch.ones(2, 2),
        "merge_gather_idx": "idx",
        "pos_embeds": "pos",
        "cu_seqlens": "cu",
    }
    out = wrapper.encoder_cudagraph_forward(inputs)
    # ((1 + 1) * 2) summed over 2 rows = 8, minus 3.
    torch.testing.assert_close(out, torch.full((2,), 5.0))
    assert order == [
        ("embed", None, "pos"),
        ("encoder", None, {"cu_seqlens": "cu"}),
        ("merge", "idx"),
    ]
    assert inputs == {"cu_seqlens": "cu"}  # consumed keys are popped


def test_encoder_eager_forward(monkeypatch):
    seen = {}

    class Tower:
        encoder = SimpleNamespace(
            prepare_encoder_metadata=lambda grid, device: (
                "meta",
                tuple(map(tuple, grid)),
            )
        )

        def __call__(self, pixels, grid, *, encoder_metadata):
            seen.update(grid=grid, meta=encoder_metadata)
            return [pixels[:1], pixels[1:]]

    monkeypatch.setattr(
        kimi_model,
        "mm_projector_forward",
        lambda projector, outs: [projector(o) for o in outs],
    )
    wrapper = _wrapper(vision_tower=Tower(), mm_projector=lambda t: t * 10)
    out = wrapper.encoder_eager_forward(
        {"pixel_values": torch.arange(3.0).view(3, 1), "grid_thws": [[1, 1, 3]]}
    )
    torch.testing.assert_close(out, torch.tensor([[0.0], [10.0], [20.0]]))
    assert seen == {"grid": [[1, 1, 3]], "meta": ("meta", ((1, 1, 3),))}


class _TpuTensor(torch.Tensor):
    """A CPU tensor that reports a TPU device, to drive the TPU branch."""

    @property
    def device(self):
        return SimpleNamespace(type="tpu")


def _embed_wrapper():
    wrapper = _wrapper()
    wrapper._embed_text_input_ids = lambda ids, embed, is_multimodal: ids.float()[
        :, None
    ].expand(-1, 2)
    wrapper.get_language_model = lambda: SimpleNamespace(embed_input_ids=None)
    return wrapper


def test_embed_input_ids_merges_media_rows():
    wrapper = _embed_wrapper()
    ids = torch.tensor([1, 2, 3, 4])
    text_only = wrapper.embed_input_ids(ids)
    assert text_only.tolist() == [[1, 1], [2, 2], [3, 3], [4, 4]]
    assert wrapper.embed_input_ids(ids, []).tolist() == text_only.tolist()

    mask = torch.tensor([False, True, True, False])
    media = [torch.tensor([[10.0, 11.0]]), torch.tensor([[20.0, 21.0]])]
    merged = wrapper.embed_input_ids(ids, media, is_multimodal=mask)
    assert merged.tolist() == [[1, 1], [10, 11], [20, 21], [4, 4]]

    # Zero placeholders and zero media rows: the text embedding is returned.
    none = torch.zeros(4, dtype=torch.bool)
    out = wrapper.embed_input_ids(ids, [torch.zeros(0, 2)], is_multimodal=none)
    assert out.tolist() == text_only.tolist()


def test_embed_input_ids_rejects_bad_media():
    wrapper = _embed_wrapper()
    ids = torch.tensor([1, 2])
    media = [torch.ones(1, 2)]
    with pytest.raises(ValueError, match="is_multimodal is required"):
        wrapper.embed_input_ids(ids, media)
    with pytest.raises(
        ValueError, match="Attempted to assign 1 multimodal tokens to 2"
    ):
        wrapper.embed_input_ids(ids, media, is_multimodal=torch.tensor([True, True]))


class _TorchWithTpuWhere:
    """`torch` whose `where` result reports a TPU device."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def where(*args):
        return torch.where(*args).as_subclass(_TpuTensor)


def test_embed_input_ids_materializes_on_tpu(monkeypatch):
    synced = []
    monkeypatch.setattr(kimi_model, "synchronize_tensors", synced.append)
    monkeypatch.setattr(kimi_model, "torch", _TorchWithTpuWhere())
    wrapper = _embed_wrapper()
    media = [torch.tensor([[7.0, 8.0]])]
    out = wrapper.embed_input_ids(
        torch.tensor([1, 2]), media, is_multimodal=torch.tensor([False, True])
    )
    (materialized,) = synced
    assert materialized is out
    assert out.as_subclass(torch.Tensor).tolist() == [[1.0, 1.0], [7.0, 8.0]]


def test_forward_logits_and_weight_hooks(monkeypatch):
    calls = []
    lm = SimpleNamespace(
        compute_logits=lambda h: ("logits", h),
        process_weights_after_loading=lambda: calls.append("pwal"),
    )

    class LM:
        compute_logits = staticmethod(lm.compute_logits)
        process_weights_after_loading = staticmethod(lm.process_weights_after_loading)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return "hidden"

    wrapper = _wrapper(language_model=LM())
    assert wrapper.forward(None, "pos", inputs_embeds="emb", extra=1) == "hidden"
    assert calls[0] == {
        "input_ids": None,
        "positions": "pos",
        "intermediate_tensors": None,
        "inputs_embeds": "emb",
    }
    assert wrapper.compute_logits("h", extra=1) == ("logits", "h")
    wrapper.process_weights_after_loading()
    assert calls[-1] == "pwal"

    loaded = []

    class Loader:
        def __init__(self, module):
            loaded.append(module)

        def load_weights(self, weights, *, mapper):
            loaded.append((list(weights), mapper))
            return {"w"}

    monkeypatch.setattr(kimi_model, "AutoWeightsLoader", Loader)
    assert wrapper.load_weights([("a", 1)]) == {"w"}
    assert loaded == [wrapper, ([("a", 1)], wrapper.hf_to_vllm_mapper)]


def test_mamba_state_classmethods_use_the_text_config(monkeypatch):
    text_config = object()
    seen = []

    class Config:
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config)
        )

        def with_hf_config(self, hf_config):
            seen.append(hf_config)
            return ("text-vllm-config", hf_config)

    monkeypatch.setattr(
        KimiLinearForCausalLM,
        "get_mamba_state_dtype_from_config",
        classmethod(lambda cls, cfg: ("dtype", cfg)),
    )
    monkeypatch.setattr(
        KimiLinearForCausalLM,
        "get_mamba_state_shape_from_config",
        classmethod(lambda cls, cfg: ("shape", cfg)),
    )
    monkeypatch.setattr(
        KimiLinearForCausalLM,
        "get_mamba_state_copy_func",
        classmethod(lambda cls: "copy"),
    )
    cls = KimiK3ForConditionalGeneration
    assert cls.get_mamba_state_dtype_from_config(Config()) == (
        "dtype",
        ("text-vllm-config", text_config),
    )
    assert cls.get_mamba_state_shape_from_config(Config()) == (
        "shape",
        ("text-vllm-config", text_config),
    )
    assert cls.get_mamba_state_copy_func() == "copy"
    assert seen == [text_config, text_config]
