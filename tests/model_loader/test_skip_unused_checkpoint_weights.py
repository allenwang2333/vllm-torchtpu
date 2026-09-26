# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for skipping checkpoint weights the model drops.

An MTP draft over the target checkpoint used to read every shard to keep one
layer, and the target read the MTP head only to drop it. Names the model's
WeightsMapper drops are skipped before ``get_tensor``, and shard files holding
none of the model's tensors are left out. No TPU, no network.
"""

import json
import sys

import torch

from vllm_torchtpu import model_loader_patches as mlp


class _FakeMapper:
    def __init__(self, drop_prefix: str = "mtp."):
        self.drop_prefix = drop_prefix

    def _map_name(self, name: str) -> str | None:
        return None if name.startswith(self.drop_prefix) else name


def _activate(monkeypatch, mapper) -> mlp._SkipState:
    state = mlp._SkipState(mapper=mapper)
    monkeypatch.setattr(mlp, "_skip_state", state)
    return state


def _stream():
    yield "mtp.fc.weight", torch.zeros(1)


def test_qwen3_5_target_mapper_discards_mtp():
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )

    for cls in (Qwen3_5MoeForConditionalGeneration, Qwen3_5ForCausalLM):
        mapper = getattr(cls, "hf_to_vllm_mapper", None)
        assert mapper is not None, cls
        assert mapper._map_name("mtp.layers.0.mlp.experts.3.down_proj.weight") is None
        assert (
            mapper._map_name("model.language_model.layers.0.mlp.gate.weight")
            is not None
        )
        assert mapper._map_name("lm_head.weight") is not None


# Name kinds of the Qwen3.5-397B-A17B-FP8 checkpoint.
_QWEN3_5_NAMES = (
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.experts.3.down_proj.weight",
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
    "model.language_model.norm.weight",
    "model.language_model.layers.0.mlp.experts.3.down_proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
)


def test_qwen3_5_mtp_mirror_drops_what_load_weights_drops(monkeypatch):
    from vllm.model_executor.models import qwen3_5_mtp

    kept = []

    class _RecordingLoader:
        def __init__(self, module, **kwargs):
            pass

        def load_weights(self, weights, *, mapper=None):
            kept.extend(checkpoint_name for _, checkpoint_name in weights)
            return set()

    monkeypatch.setattr(qwen3_5_mtp, "AutoWeightsLoader", _RecordingLoader)
    # Each tensor is its checkpoint name, so load_weights' renames keep it.
    qwen3_5_mtp.Qwen3_5MTP.load_weights(None, [(n, n) for n in _QWEN3_5_NAMES])

    mirror = mlp._qwen3_5_mtp_mapper()
    assert kept == [n for n in _QWEN3_5_NAMES if mirror._map_name(n) is not None]
    assert "mtp.fc.weight" in kept
    assert "model.language_model.norm.weight" not in kept


def test_mirror_covers_upstream_qwen3_5_mtp_only(monkeypatch):
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP

    mirror = mlp._qwen3_5_mtp_mapper()
    assert mlp._mirrored_mapper(Qwen3_5MTP) is mirror
    assert mlp._mirrored_mapper(Qwen3_5MoeMTP) is mirror
    assert mlp._mirrored_mapper(Qwen3_5MoeForConditionalGeneration) is None

    # Upstream giving Qwen3_5MTP its own mapper retires the mirror.
    monkeypatch.setattr(Qwen3_5MTP, "hf_to_vllm_mapper", mirror, raising=False)
    assert mlp._mirrored_mapper(Qwen3_5MoeMTP) is None


def test_mirror_lookup_never_imports_model_modules(monkeypatch):
    name = "vllm.model_executor.models.qwen3_5_mtp"
    monkeypatch.delitem(sys.modules, name, raising=False)

    class _Model:
        def load_weights(self, weights):
            return set()

    assert mlp._mirrored_mapper(_Model) is None
    assert name not in sys.modules


def test_mapper_given_the_unread_stream_decides_what_is_skipped():
    mapper = _FakeMapper(drop_prefix="mtp.")
    state = mlp._SkipState()
    stream = _stream()
    state.record_weights(stream)
    state.record_mapper(stream, mapper)
    assert state.mapper is mapper
    assert state.drops("mtp.fc.weight")
    assert not state.drops("lm_head.weight")


def test_mapper_that_misses_stored_names_decides_nothing():
    mapper = _FakeMapper(drop_prefix="mtp.")
    state = mlp._SkipState()
    stream = _stream()
    state.record_weights(stream)
    # load_weights renames or filters the stream before the mapper sees it.
    state.record_mapper(((n, w) for n, w in stream), mapper)
    assert state.mapper is None
    # Something already read from the stream.
    next(stream)
    state.record_mapper(stream, mapper)
    assert state.mapper is None


def test_mirror_set_up_front_is_kept():
    mirror = _FakeMapper(drop_prefix="model.")
    state = mlp._SkipState(mapper=mirror)
    stream = _stream()
    state.record_weights(stream)
    state.record_mapper(stream, _FakeMapper(drop_prefix="mtp."))
    assert state.mapper is mirror


def test_should_skip_weight_consults_the_skip_state(monkeypatch):
    state = _activate(monkeypatch, _FakeMapper(drop_prefix="mtp."))
    assert mlp._should_skip_weight_tpu("mtp.fc.weight", None)
    assert not mlp._should_skip_weight_tpu("lm_head.weight", None)
    assert state.skipped == 1

    monkeypatch.setattr(mlp, "_skip_state", None)
    assert not mlp._should_skip_weight_tpu("mtp.fc.weight", None)


def _write_index(folder, weight_map):
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def test_files_holding_only_discarded_tensors_are_dropped(tmp_path, monkeypatch):
    _write_index(
        tmp_path,
        {
            "model.layers.0.weight": "a.safetensors",
            "mtp.fc.weight": "a.safetensors",
            "model.layers.1.weight": "b.safetensors",
            "mtp.layers.0.weight": "c.safetensors",
        },
    )
    files = [
        str(tmp_path / f)
        for f in (
            "a.safetensors",
            "b.safetensors",
            "c.safetensors",
            "unlisted.safetensors",
        )
    ]

    monkeypatch.setattr(
        mlp, "_orig_filter_duplicate_safetensors_files", lambda fl, fd, idx: fl
    )

    # Draft mapper drops non-mtp: shard b is discarded, a and c kept.
    class _DraftMapper:
        def _map_name(self, name: str):
            return name if name.startswith("mtp.") else None

    _activate(monkeypatch, _DraftMapper())
    kept_draft = mlp._filter_duplicate_safetensors_files_tpu(
        files, str(tmp_path), "model.safetensors.index.json"
    )
    assert kept_draft == [files[0], files[2], files[3]]

    # Target mapper drops mtp: shard c is discarded, a and b kept.
    _activate(monkeypatch, _FakeMapper(drop_prefix="mtp."))
    kept_target = mlp._filter_duplicate_safetensors_files_tpu(
        files, str(tmp_path), "model.safetensors.index.json"
    )
    assert kept_target == [files[0], files[1], files[3]]


def test_file_list_kept_without_index_or_when_no_indexed_file_is_left(
    tmp_path, monkeypatch
):
    files = [str(tmp_path / "a.safetensors"), str(tmp_path / "unlisted.safetensors")]
    monkeypatch.setattr(
        mlp, "_orig_filter_duplicate_safetensors_files", lambda fl, fd, idx: fl
    )

    class _DropAllMapper:
        def _map_name(self, name: str):
            return None

    _activate(monkeypatch, _DropAllMapper())
    # No index file present -> keeps all files
    assert (
        mlp._filter_duplicate_safetensors_files_tpu(
            files, str(tmp_path), "model.safetensors.index.json"
        )
        == files
    )

    # Index present but all indexed files discarded -> safe fallback keeps all files
    _write_index(tmp_path, {"mtp.fc.weight": "a.safetensors"})
    assert (
        mlp._filter_duplicate_safetensors_files_tpu(
            files, str(tmp_path), "model.safetensors.index.json"
        )
        == files
    )


def test_lazy_iterator_never_reads_discarded_tensors(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader import weight_utils

    shard = tmp_path / "model-00001.safetensors"
    save_file(
        {
            "mtp.fc.weight": torch.zeros(2, 2),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(2, 2),
        },
        str(shard),
    )
    read_names = []
    real_safe_open = weight_utils.safe_open

    class _TrackingSafeOpen:
        def __init__(self, *args, **kwargs):
            self._inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def keys(self):
            return self._inner.keys()

        def get_tensor(self, name):
            read_names.append(name)
            return self._inner.get_tensor(name)

    monkeypatch.setattr(weight_utils, "should_skip_weight", mlp._should_skip_weight_tpu)
    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)
    _activate(monkeypatch, _FakeMapper(drop_prefix="mtp."))

    yielded = [
        name
        for name, _ in weight_utils.safetensors_weights_iterator(
            [str(shard)], use_tqdm_on_load=False
        )
    ]

    assert yielded == ["model.layers.0.self_attn.o_proj.weight"]
    assert read_names == ["model.layers.0.self_attn.o_proj.weight"]
