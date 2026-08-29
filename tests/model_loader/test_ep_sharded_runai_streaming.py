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
"""Unit tests for EP-sharded Run:AI weight streaming (model_loader_patches).

Covers the pieces that decide WHAT gets fetched: the per-tensor skip
predicate, and the construction of the byte-range request plan from
safetensors metadata. No TPU, no network: the Run:AI streamer is faked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.model_loader_patches import (
    _sharded_runai_weights_iterator, _should_skip, _skip_reason, layer_index)

# ---------------------------------------------------------------------------
# _should_skip
# ---------------------------------------------------------------------------

_LOCAL = {0, 1, 27}


def test_skips_nonlocal_expert_packed_and_scale():
    base = "language_model.model.layers.3.block_sparse_moe.experts.500"
    assert _should_skip(f"{base}.w1.weight_packed", _LOCAL)
    assert _should_skip(f"{base}.w2.weight_scale", _LOCAL)
    assert _should_skip(f"{base}.w3.weight", _LOCAL)


def test_keeps_local_expert_weights():
    base = "language_model.model.layers.3.block_sparse_moe.experts.27"
    assert not _should_skip(f"{base}.w1.weight_packed", _LOCAL)
    assert not _should_skip(f"{base}.w2.weight_scale", _LOCAL)


def test_keeps_dense_shared_and_gate_weights():
    prefix = "language_model.model.layers.3"
    for name in (
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.block_sparse_moe.shared_experts.up_proj.weight",
            f"{prefix}.block_sparse_moe.gate.weight",
            f"{prefix}.block_sparse_moe.gate.e_score_correction_bias",
            "language_model.model.embed_tokens.weight",
    ):
        assert not _should_skip(name, _LOCAL), name


# ---------------------------------------------------------------------------
# truncation filter (--hf-overrides num_hidden_layers below the checkpoint's)
# ---------------------------------------------------------------------------

# The real Kimi-K3 checkpoint's name shapes, in the proportions the index
# json has them: 92 MoE layers x 896 experts x {w1,w2,w3} x
# {weight_packed, weight_scale}, plus the per-layer dense tensors and the
# non-layer ones. Generating them is cheaper than shipping a 59 MB index.
_K3_LAYER_DENSE = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.o_proj.weight",
    "self_attn.g_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
)
_K3_MOE_DENSE = (
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
)


def _k3_names(num_layers: int = 93,
              num_experts: int = 896,
              first_k_dense: int = 1):
    """Checkpoint tensor names for a K3-shaped model."""
    yield "language_model.model.embed_tokens.weight"
    yield "language_model.model.norm.weight"
    yield "language_model.lm_head.weight"
    for layer in range(num_layers):
        prefix = f"language_model.model.layers.{layer}"
        for suffix in _K3_LAYER_DENSE:
            yield f"{prefix}.{suffix}"
        if layer < first_k_dense:
            yield f"{prefix}.mlp.up_proj.weight"
            continue
        for suffix in _K3_MOE_DENSE:
            yield f"{prefix}.{suffix}"
        for expert in range(num_experts):
            for proj in ("w1", "w2", "w3"):
                base = f"{prefix}.block_sparse_moe.experts.{expert}.{proj}"
                yield f"{base}.weight_packed"
                yield f"{base}.weight_scale"


@pytest.mark.parametrize("name,expected", [
    ("language_model.model.layers.15.self_attn.o_proj.weight", 15),
    ("model.layers.0.block_sparse_moe.experts.7.w2.weight_scale", 0),
    ("language_model.model.embed_tokens.weight", None),
    ("vision_tower.encoder.layers.4.attn.qkv.weight", 4),
])
def test_layer_index_parses_the_decoder_index(name, expected):
    assert layer_index(name) == expected


def test_no_truncation_filter_without_a_bound():
    """num_layers=None must behave exactly like the pre-truncation filter."""
    name = "language_model.model.layers.92.self_attn.o_proj.weight"
    assert _skip_reason(name, _LOCAL, None) is None


def test_full_model_skips_zero_tensors_for_truncation():
    """The 93-layer path must be untouched: no tensor is dropped as truncated."""
    reasons = [_skip_reason(n, _LOCAL, 93) for n in _k3_names()]
    assert reasons.count("truncated_layer") == 0


def test_truncated_model_skips_exactly_the_out_of_range_layers():
    names = list(_k3_names())
    reasons = [_skip_reason(n, _LOCAL, 12) for n in names]
    # 81 dropped layers: 81 x (8 dense + 5 moe-dense + 896 x 6 expert).
    expected = 81 * (len(_K3_LAYER_DENSE) + len(_K3_MOE_DENSE) + 896 * 6)
    assert reasons.count("truncated_layer") == expected
    # and nothing inside the kept range is dropped for truncation
    kept = [n for n, r in zip(names, reasons) if r != "truncated_layer"]
    assert all((layer_index(n) is None or layer_index(n) < 12) for n in kept)


def test_truncation_and_expert_filter_compose():
    """A foreign expert inside the kept range is still reported as such."""
    kept_range = "language_model.model.layers.3.block_sparse_moe.experts.500"
    gone = "language_model.model.layers.50.block_sparse_moe.experts.0"
    assert _skip_reason(f"{kept_range}.w1.weight_packed", _LOCAL,
                        12) == ("foreign_expert")
    # Truncation wins for an out-of-range layer even when the expert is local.
    assert _skip_reason(f"{gone}.w1.weight_packed", _LOCAL,
                        12) == ("truncated_layer")
    assert _should_skip(f"{gone}.w1.weight_packed", _LOCAL, 12)


def test_keeps_expert_tensors_with_unknown_suffix():
    # Only known heavy suffixes are skippable; anything else is kept even
    # for a non-local expert, mirroring upstream's conservatism.
    name = ("language_model.model.layers.3.block_sparse_moe.experts.500"
            ".w1.some_metadata")
    assert not _should_skip(name, _LOCAL)


def test_keeps_fused_expert_tensors_without_numeric_id():
    # 3D fused-expert layouts have no numeric id; the full tensor must load
    # so the weight_loader can slice it.
    name = "model.layers.3.mlp.experts.gate_proj.weight"
    assert not _should_skip(name, _LOCAL)


# ---------------------------------------------------------------------------
# Request-plan construction
# ---------------------------------------------------------------------------


def _tensor_meta(name):
    return SimpleNamespace(name=name)


def _fake_file_meta(offset, names_and_sizes):
    return SimpleNamespace(
        offset=offset,
        tensors_metadata=[_tensor_meta(n) for n, _ in names_and_sizes],
        read_sizes=[s for _, s in names_and_sizes],
    )


def test_request_plan_groups_contiguous_kept_runs():
    """Kept tensors separated by a skipped one become separate byte-range
    requests, offsets account for the skipped bytes, and skipped tensors are
    never part of any request."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    e = "m.layers.0.experts"
    file_meta = _fake_file_meta(
        offset=1000,
        names_and_sizes=[
            (f"{e}.0.w1.weight_packed", 10),  # kept   @1000
            (f"{e}.0.w1.weight_scale", 2),  # kept   @1010
            (f"{e}.5.w1.weight_packed", 10),  # skip   @1012
            (f"{e}.5.w1.weight_scale", 2),  # skip   @1022
            ("m.layers.0.dense.weight", 7),  # kept   @1024
        ],
    )

    captured = {}

    class FakeStreamer:

        def __init__(self):
            self.file_streamer = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, requests, credentials, device, is_distributed):
            captured["requests"] = requests

        def get_chunks(self):
            return iter(())  # plan construction is the unit under test

    # The iterator imports the Run:AI modules lazily and reads these names
    # as attributes at call time, so patching the real modules' attributes
    # intercepts them deterministically.
    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    from runai_model_streamer.safetensors_streamer import safetensors_streamer

    fake_metadata = MagicMock()
    fake_metadata.from_files.return_value = [file_meta]

    with patch.object(safetensors_streamer, "SafetensorsStreamer",
                      FakeStreamer), \
         patch.object(sp, "SafetensorsMetadata", fake_metadata):
        list(
            _sharded_runai_weights_iterator(["gs://bucket/file0.safetensors"],
                                            local_expert_ids={0},
                                            use_tqdm_on_load=False))

    requests = captured["requests"]
    assert [(r.offset, r.chunks) for r in requests] == [
        (1000, [10, 2]),  # expert 0's contiguous run
        (1024, [7]),  # dense weight after the skipped gap
    ]
    total_requested = sum(sum(r.chunks) for r in requests)
    assert total_requested == 19  # 31 bytes in file, 12 skipped


def test_zero_size_kept_tensor_bypasses_fetch_plan():
    """A kept zero-byte tensor must never enter the fetch plan (the
    streamer's request iterator silently drops a zero-byte chunk group, and
    one at the head of its queue aborts the remaining stream). It is
    yielded as an empty tensor instead, and it must not break the
    contiguity of the surrounding run."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    file_meta = _fake_file_meta(
        offset=100,
        names_and_sizes=[
            ("m.layers.0.dense_a.weight", 8),  # kept @100
            ("m.layers.0.empty.weight", 0),  # kept, zero bytes
            ("m.layers.0.dense_b.weight", 4),  # kept @108 -- same run
        ],
    )
    # Give the zero-size tensor real metadata so create_torch_tensor takes
    # its empty-tensor path.
    file_meta.tensors_metadata[1].get_item_count = lambda: 0
    file_meta.tensors_metadata[1].shape = [0]
    file_meta.tensors_metadata[1].get_torch_dtype = lambda: None

    captured = {}

    class FakeStreamer:

        def __init__(self):
            self.file_streamer = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, requests, credentials, device, is_distributed):
            captured["requests"] = requests

        def get_chunks(self):
            return iter(())

    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    from runai_model_streamer.safetensors_streamer import safetensors_streamer

    fake_metadata = MagicMock()
    fake_metadata.from_files.return_value = [file_meta]
    fake_create = MagicMock(return_value=MagicMock())

    with patch.object(safetensors_streamer, "SafetensorsStreamer",
                      FakeStreamer), \
         patch.object(sp, "SafetensorsMetadata", fake_metadata), \
         patch.object(sp, "create_torch_tensor", fake_create):
        yielded = [
            name for name, _ in _sharded_runai_weights_iterator(
                ["gs://bucket/file0.safetensors"],
                local_expert_ids=set(),
                use_tqdm_on_load=False)
        ]

    requests = captured["requests"]
    # One contiguous run through the zero-byte tensor; no 0-length chunk.
    assert [(r.offset, r.chunks) for r in requests] == [(100, [8, 4])]
    # The zero-size tensor is still yielded (as an empty tensor).
    assert "m.layers.0.empty.weight" in yielded


def test_streamer_split_preserves_chunk_index_mapping():
    """Pin the library contract this loader depends on: when
    RUNAI_STREAMER_MEMORY_LIMIT forces FilesRequestsIterator to split one
    FileChunks across several sub-requests, get_global_file_and_chunk must
    keep reporting (our request id, index within OUR chunk list)."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    from runai_model_streamer.file_streamer.requests_iterator import (
        FileChunks, FilesRequestsIterator)

    # One request of 5 chunks, memory limit 6 -> split into [3,3],[3,3],[3]
    chunks = [3, 3, 3, 3, 3]
    it = FilesRequestsIterator(6, [FileChunks(7, "/f", 100, list(chunks))])

    seen = []
    while True:
        req = it.next_request()
        if req is None:
            break
        for local_file_idx, fc in enumerate(req.files):
            for local_chunk_idx in range(len(fc.chunks)):
                file_id, global_idx = it.get_global_file_and_chunk(
                    local_file_idx, local_chunk_idx)
                seen.append((file_id, global_idx))
    assert seen == [(7, i) for i in range(len(chunks))]


def test_patch_falls_back_when_filter_inactive():
    """With no local ids (flag off / old vLLM), the patched method must
    delegate to the original iterator; the patch must not double-apply."""
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    from vllm_torchtpu import model_loader_patches as mlp

    original = MagicMock(name="original_get_weights_iterator")

    class FakeLoader:
        _get_weights_iterator = original

    with patch.object(rsl, "RunaiModelStreamerLoader", FakeLoader):
        mlp.patch_runai_sharded_expert_streaming()
        patched_once = FakeLoader._get_weights_iterator
        mlp.patch_runai_sharded_expert_streaming()  # idempotent
        assert FakeLoader._get_weights_iterator is patched_once

        with patch.object(mlp, "_compute_local_expert_ids", return_value=None):
            loader = FakeLoader()
            FakeLoader._get_weights_iterator(loader, "gs://b/m", None)
        original.assert_called_once_with(loader, "gs://b/m", None)
