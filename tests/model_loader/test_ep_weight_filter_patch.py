# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DefaultModelLoader EP weight-filter patch.

vLLM 0.27.0's ``should_skip_weight`` only matches names ending in
``.weight``, so a compressed-tensors MXFP4 checkpoint (``.weight_packed`` /
``.weight_scale``) reads every expert's bytes on every rank. The TPU patch
widens the suffix allowlist. No TPU, no network.
"""

import pytest
import torch

from vllm_torchtpu import envs, model_loader_patches
from vllm_torchtpu.model_loader_patches import (
    _local_tensor_ranges,
    _should_skip_weight_tpu,
    _start_ep_range_prefetch,
    patch_default_loader_ep_weight_filter,
)

_LOCAL = {0, 1, 27}


@pytest.fixture(autouse=True)
def _contiguous_placement(monkeypatch):
    """TPU_EXPERT_PLACEMENT defaults to the Qwen3.5 table, which would place
    these layer names by table instead of by ``local_expert_ids``."""
    monkeypatch.setattr(envs, "TPU_EXPERT_PLACEMENT", "")


def _expert_name(expert_id: int, suffix: str) -> str:
    return (
        f"language_model.model.layers.3.block_sparse_moe.experts.{expert_id}.w1{suffix}"
    )


# ---------------------------------------------------------------------------
# _should_skip_weight_tpu
# ---------------------------------------------------------------------------


def test_remote_expert_packed_and_scale_skipped():
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_packed"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_scale"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight"), _LOCAL)


def test_remote_expert_dsv4_scale_skipped():
    # DeepSeek-V4 stores the per-expert MXFP4 block scales as ``.scale``.
    name = "layers.3.ffn.experts.{}.w1.scale"
    assert _should_skip_weight_tpu(name.format(500), _LOCAL)
    assert not _should_skip_weight_tpu(name.format(27), _LOCAL)


def test_local_expert_packed_and_scale_kept():
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_packed"), _LOCAL)
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_scale"), _LOCAL)


def test_remote_expert_input_scale_kept():
    # NVFP4 per-expert activation scales feed a global-max reduction in
    # FlashInfer backends; they must stay unfiltered on every rank.
    assert not _should_skip_weight_tpu(_expert_name(500, ".input_scale"), _LOCAL)


def test_no_local_ids_keeps_everything():
    # Filter off (EP weight filter disabled): the upstream contract is that
    # local_expert_ids=None keeps every tensor.
    assert not _should_skip_weight_tpu(_expert_name(500, ".weight_packed"), None)


def test_dense_and_fused_names_kept():
    for name in (
        "language_model.model.layers.3.self_attn.o_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.gate.weight",
        "language_model.model.embed_tokens.weight",
        # 3D fused-expert layout: no numeric id, full tensor must load.
        "model.layers.3.mlp.experts.gate_proj.weight",
    ):
        assert not _should_skip_weight_tpu(name, _LOCAL), name


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def _restore_iterators(monkeypatch):
    from vllm.model_executor.model_loader import default_loader, weight_utils

    monkeypatch.setattr(
        weight_utils,
        "safetensors_weights_iterator",
        weight_utils.safetensors_weights_iterator,
    )
    monkeypatch.setattr(
        default_loader,
        "safetensors_weights_iterator",
        default_loader.safetensors_weights_iterator,
    )
    monkeypatch.setattr(
        default_loader.DefaultModelLoader,
        "load_weights",
        default_loader.DefaultModelLoader.load_weights,
    )


def test_patch_installs_predicate_and_is_idempotent(monkeypatch):
    from vllm.model_executor.model_loader import (
        default_loader,
        ep_weight_filter,
        weight_utils,
    )

    # Register teardown restoration of the attributes the patch overwrites.
    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    # delattr(raising=False) records nothing when the flag is absent, so it
    # would not undo the stamp the patch sets. setattr registers a teardown
    # that deletes (or restores) it.
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)

    patch_default_loader_ep_weight_filter()
    assert weight_utils.should_skip_weight is _should_skip_weight_tpu
    assert ep_weight_filter.should_skip_weight is _should_skip_weight_tpu
    assert weight_utils._tpu_ep_filter_patch is True
    assert default_loader.safetensors_weights_iterator is (
        weight_utils.safetensors_weights_iterator
    )

    # A second call must not re-apply over a foreign value.
    foreign = object()
    weight_utils.should_skip_weight = foreign
    patch_default_loader_ep_weight_filter()
    assert weight_utils.should_skip_weight is foreign


# ---------------------------------------------------------------------------
# Iterator integration: real safetensors keys, lazy load path
# ---------------------------------------------------------------------------


def test_lazy_iterator_filters_remote_packed_and_scale(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader import weight_utils

    e = "model.layers.0.mlp.experts"
    tensors = {
        f"{e}.0.w1.weight_packed": torch.zeros(8, dtype=torch.uint8),
        f"{e}.0.w1.weight_scale": torch.zeros(1, dtype=torch.uint8),
        f"{e}.7.w1.weight_packed": torch.zeros(8, dtype=torch.uint8),
        f"{e}.7.w1.weight_scale": torch.zeros(1, dtype=torch.uint8),
        f"{e}.7.w1.input_scale": torch.zeros(1, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(2, 2),
    }
    shard = tmp_path / "model-00001.safetensors"
    save_file(tensors, str(shard))

    # Track which tensors are actually read from disk.
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

    monkeypatch.setattr(weight_utils, "should_skip_weight", _should_skip_weight_tpu)
    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)

    yielded = {
        name
        for name, _ in weight_utils.safetensors_weights_iterator(
            [str(shard)], use_tqdm_on_load=False, local_expert_ids={0}
        )
    }

    assert f"{e}.0.w1.weight_packed" in yielded
    assert f"{e}.0.w1.weight_scale" in yielded
    assert f"{e}.7.w1.input_scale" in yielded
    assert "model.layers.0.self_attn.o_proj.weight" in yielded
    assert f"{e}.7.w1.weight_packed" not in yielded
    assert f"{e}.7.w1.weight_scale" not in yielded
    # Skipped tensors must never be read from disk, not just not yielded.
    assert f"{e}.7.w1.weight_packed" not in read_names
    assert f"{e}.7.w1.weight_scale" not in read_names


# ---------------------------------------------------------------------------
# EP-local range prefetch
# ---------------------------------------------------------------------------

_E = "model.layers.0.mlp.experts"


def _save_shard(tmp_path):
    from safetensors.torch import save_file

    # Two remote experts sit between the local ones and the dense tensor, so
    # the kept tensors form more than one byte range.
    tensors = {
        f"{_E}.0.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
        f"{_E}.0.w1.scale": torch.ones(4, dtype=torch.uint8),
        f"{_E}.5.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
        f"{_E}.6.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
        f"{_E}.7.w1.scale": torch.ones(4, dtype=torch.uint8),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
    }
    shard = tmp_path / "model-00001.safetensors"
    save_file(tensors, str(shard))
    return str(shard)


def _header(path):
    import json
    import struct

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


def test_local_tensor_ranges_cover_exactly_the_kept_tensors(tmp_path):
    shard = _save_shard(tmp_path)
    base, header = _header(shard)

    ranges = _local_tensor_ranges(shard, {0})

    covered = set()
    for lo, hi in ranges:
        covered.update(range(lo, hi))
    for name, info in header.items():
        start, end = (base + o for o in info["data_offsets"])
        span = set(range(start, end))
        if _should_skip_weight_tpu(name, {0}):
            assert not span & covered, name
        else:
            assert span <= covered, name
    # Ranges are sorted, disjoint and non-adjacent (adjacent spans merge).
    assert all(a[1] < b[0] for a, b in zip(ranges, ranges[1:]))


def test_local_tensor_ranges_all_local_is_one_range(tmp_path):
    shard = _save_shard(tmp_path)
    base, header = _header(shard)
    end = base + max(info["data_offsets"][1] for info in header.values())

    assert _local_tensor_ranges(shard, {0, 5, 6, 7}) == [(base, end)]


def test_range_prefetch_reads_only_local_ranges(tmp_path, monkeypatch):
    import os

    shard = _save_shard(tmp_path)

    import threading

    started = []

    class _RecordedThread(threading.Thread):
        def start(self):
            started.append(self)
            super().start()

    reads = []
    real_pread = os.pread

    def tracking_pread(fd, n, off):
        reads.append((off, off + n))
        return real_pread(fd, n, off)

    monkeypatch.setattr(model_loader_patches.threading, "Thread", _RecordedThread)
    monkeypatch.setattr(model_loader_patches.os, "pread", tracking_pread)

    _start_ep_range_prefetch([shard], {0}, num_threads=2, block_size=16)
    for thread in started:
        thread.join(timeout=30)

    expected = set()
    for lo, hi in _local_tensor_ranges(shard, {0}):
        expected.update(range(lo, hi))
    read = set()
    for lo, hi in reads:
        assert hi - lo <= 16
        read.update(range(lo, hi))
    assert read == expected


def test_prefetch_strategy_under_filter_switches_to_lazy(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader import ep_weight_filter, weight_utils

    shard = _save_shard(tmp_path)
    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)

    original = weight_utils.safetensors_weights_iterator
    strategies = []

    def recording_iterator(files, use_tqdm, strategy=None, local_ids=None, **kw):
        strategies.append(strategy)
        return original(files, use_tqdm, strategy, local_ids, **kw)

    prefetches = []
    monkeypatch.setattr(
        weight_utils, "safetensors_weights_iterator", recording_iterator
    )
    monkeypatch.setattr(
        model_loader_patches,
        "_start_ep_range_prefetch",
        lambda files, ids, threads, block, is_unused=None: prefetches.append(
            (files, ids)
        ),
    )
    patch_default_loader_ep_weight_filter()
    iterator = weight_utils.safetensors_weights_iterator

    names = {
        name for name, _ in iterator([shard], False, "prefetch", local_expert_ids={0})
    }
    assert strategies == ["lazy"]
    assert prefetches == [([shard], {0})]
    assert f"{_E}.0.w1.scale" in names
    assert f"{_E}.7.w1.scale" not in names

    # Filter off: upstream prefetch runs unchanged.
    list(iterator([shard], False, "prefetch", local_expert_ids=None))
    assert strategies == ["lazy", "prefetch"]
    assert len(prefetches) == 1


# ---------------------------------------------------------------------------
# Tensors the loading model discards (DefaultModelLoader.load_weights)
# ---------------------------------------------------------------------------

_STAGE_CHECKPOINT = {
    "model.language_model.layers.0.weight": 1.0,
    "model.language_model.layers.1.weight": 2.0,
    "model.language_model.layers.2.weight": 3.0,
    "model.language_model.embed_tokens.weight": 4.0,
    "lm_head.weight": 5.0,
    "mtp.fc.weight": 6.0,
}
# What the stage below loads: layer 1 and the head belong to other stages,
# and its mapper drops the MTP tensors.
_STAGE_READS = {
    "model.language_model.layers.0.weight",
    "model.language_model.layers.2.weight",
    "model.language_model.embed_tokens.weight",
}


def _stage_model():
    """A pipeline stage without layer 1 or the head, loading like vLLM's models."""
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        PPMissingLayer,
        WeightsMapper,
    )

    class Stage(torch.nn.Module):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None}
        )

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [
                    PPMissingLayer() if i == 1 else torch.nn.Linear(2, 2, bias=False)
                    for i in range(3)
                ]
            )
            self.model.embed_tokens = torch.nn.Embedding(2, 2)
            self.lm_head = PPMissingLayer()
            for param in self.parameters():
                torch.nn.init.zeros_(param)

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(
                weights, mapper=self.hf_to_vllm_mapper
            )

    return Stage()


@pytest.fixture
def default_loader_filter(tmp_path, monkeypatch):
    """Installs the patch over a stand-in for vLLM's ``load_weights`` body.

    The stand-in feeds the model from the patched safetensors iterator, as
    ``DefaultModelLoader.load_weights`` does. Returns the shard, the
    installed ``load_weights`` and the stand-in, and the names whose
    ``get_tensor`` ran.
    """
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader import (
        default_loader,
        ep_weight_filter,
        weight_utils,
    )

    shard = tmp_path / "model-00001.safetensors"
    save_file(
        {name: torch.full((2, 2), v) for name, v in _STAGE_CHECKPOINT.items()},
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

    def stand_in(self, model, model_config):
        model.load_weights(
            default_loader.safetensors_weights_iterator([str(shard)], False)
        )

    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)
    monkeypatch.setattr(default_loader.DefaultModelLoader, "load_weights", stand_in)
    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)
    monkeypatch.setattr(model_loader_patches, "_compute_local_expert_ids", lambda: None)
    patch_default_loader_ep_weight_filter()
    return (
        str(shard),
        default_loader.DefaultModelLoader.load_weights,
        stand_in,
        read_names,
    )


def test_load_weights_leaves_other_stages_tensors_unread(default_loader_filter):
    _, load_weights, stand_in, read_names = default_loader_filter

    stage = _stage_model()
    load_weights(None, stage, None)
    assert set(read_names) == _STAGE_READS
    assert model_loader_patches._default_loader_unused.get() is None

    # The same load without the filter reads every tensor and leaves the
    # model with the same weights.
    read_names.clear()
    reference = _stage_model()
    stand_in(None, reference, None)
    assert set(read_names) == set(_STAGE_CHECKPOINT)
    loaded = dict(stage.named_parameters())
    for name, param in reference.named_parameters():
        assert torch.equal(loaded[name], param), name
    assert torch.equal(loaded["model.layers.2.weight"], torch.full((2, 2), 3.0))


def test_load_weights_reads_only_what_a_draft_uses(default_loader_filter):
    _, load_weights, _, read_names = default_loader_filter

    class Draft(torch.nn.Module):
        shared_checkpoint_keys = ("embed_tokens",)

        def is_checkpoint_tensor_used(self, name: str) -> bool:
            return name.startswith("mtp.") or "embed_tokens" in name

        def load_weights(self, weights):
            return {name for name, _ in weights}

    load_weights(None, Draft(), None)
    # The embedding is borrowed from the target, so it is not read either.
    assert read_names == ["mtp.fc.weight"]


def test_load_weights_without_droppable_tensors_reads_everything(
    default_loader_filter,
):
    _, load_weights, _, read_names = default_loader_filter

    class Plain(torch.nn.Module):
        def load_weights(self, weights):
            return {name for name, _ in weights}

    load_weights(None, Plain(), None)
    assert set(read_names) == set(_STAGE_CHECKPOINT)


def test_prefetch_under_the_unused_tensor_filter_reads_only_used_ranges(
    default_loader_filter, monkeypatch
):
    from vllm.model_executor.model_loader import weight_utils

    shard, _, _, _ = default_loader_filter
    is_unused = model_loader_patches._unused_checkpoint_tensor_predicate(_stage_model())
    assert is_unused is not None

    base, header = _header(shard)
    covered = set()
    for lo, hi in _local_tensor_ranges(shard, None, is_unused):
        covered.update(range(lo, hi))
    for name, info in header.items():
        span = set(range(*(base + o for o in info["data_offsets"])))
        assert (span <= covered) is (name in _STAGE_READS), name
        assert span <= covered or not span & covered, name

    prefetches = []
    monkeypatch.setattr(
        model_loader_patches,
        "_start_ep_range_prefetch",
        lambda files, ids, threads, block, is_unused=None: prefetches.append(
            (files, ids, is_unused)
        ),
    )
    token = model_loader_patches._default_loader_unused.set(is_unused)
    try:
        names = {
            name
            for name, _ in weight_utils.safetensors_weights_iterator(
                [shard], False, "prefetch"
            )
        }
    finally:
        model_loader_patches._default_loader_unused.reset(token)
    assert prefetches == [([shard], None, is_unused)]
    assert names == _STAGE_READS


def test_checkpoint_read_plan_counts_tensors_and_bytes(tmp_path):
    shard = _save_shard(tmp_path)

    plan = model_loader_patches._checkpoint_read_plan(
        [shard], {0}, lambda name: "self_attn" in name
    )

    # Expert 0's packed weight and scale are read; experts 5 and 6 and
    # expert 7's scale belong to other ranks; the model drops the attention.
    assert plan == {"read": (2, 68), "non_local_expert": (3, 132), "unused": (1, 64)}


def test_qwen35_pipeline_stage_reads_its_layers_and_the_embedding():
    """Stage 0 of Qwen3.5-397B under VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4.

    The module tree mirrors ``Qwen3_5MoeForConditionalGeneration`` with
    ``--language-model-only`` (the vision tower is a ``StageMissingLayer``)
    and the model's own ``hf_to_vllm_mapper``; names are the checkpoint's.
    """
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5MoeForConditionalGeneration,
    )
    from vllm.model_executor.models.utils import PPMissingLayer, StageMissingLayer

    class StageZero(torch.nn.Module):
        hf_to_vllm_mapper = Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper

        def __init__(self):
            super().__init__()
            self.visual = StageMissingLayer("image", torch.nn.Linear(1, 1))
            self.language_model = torch.nn.Module()
            text = torch.nn.Module()
            text.embed_tokens = torch.nn.Embedding(1, 1)
            text.layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(1, 1) if i < 8 else PPMissingLayer()
                    for i in range(60)
                ]
            )
            text.norm = PPMissingLayer()
            self.language_model.model = text
            self.language_model.lm_head = PPMissingLayer()

    is_unused = model_loader_patches._unused_checkpoint_tensor_predicate(StageZero())
    assert is_unused is not None
    for name in (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.mlp.experts.511.down_proj.weight_scale_inv",
        "model.language_model.layers.7.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
    ):
        assert not is_unused(name), name
    for name in (
        "model.language_model.layers.8.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.43.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.59.self_attn.o_proj.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "mtp.layers.0.mlp.experts.0.gate_proj.weight",
        "mtp.fc.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    ):
        assert is_unused(name), name
