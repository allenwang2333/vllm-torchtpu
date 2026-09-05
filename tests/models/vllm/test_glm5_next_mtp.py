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
"""GLM-5.3-Flash MTP draft head: the parts that fail silently.

Every check here pins something whose wrong answer still produces correctly
shaped tensors and a model that loads: the ``eh_proj`` operand order, the
``.mtp_block`` name splice, and which layer-45 modules the FP8 ignore list
resolves. A draft head with any of these wrong does not crash -- it just
never gets a token accepted, which is indistinguishable from the draft
being a bad predictor.
"""

import types

import pytest
import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM
from vllm_torchtpu.models.vllm.glm5_next.mtp import (
    Glm5NextMTP, Glm5NextMultiTokenPredictorLayer)

_MTP_LAYER = 45

#: The 18 layer-45 entries of the real ``modules_to_not_convert``, verbatim
#: from zai-org/GLM-5.3-Flash's config.json, plus ``lm_head``. Note these are
#: ``model.layers.45.*`` while the tensors are
#: ``model.language_model.layers.45.*`` -- the prefix rewrite is what makes
#: them line up.
_IGNORED = [
    "model.layers.45.eh_proj",
    "model.layers.45.enorm",
    "model.layers.45.hnorm",
    "model.layers.45.input_layernorm",
    "model.layers.45.post_attention_layernorm",
    "model.layers.45.shared_head.norm",
    "model.layers.45.mlp.gate",
    "model.layers.45.mlp.gate.e_score_correction_bias",
    "model.layers.45.self_attn.indexer.index_kpool_compress_ape",
    "model.layers.45.self_attn.indexer.index_kpool_compress_gate",
    "model.layers.45.self_attn.indexer.k_norm",
    "model.layers.45.self_attn.indexer.k_norm.bias",
    "model.layers.45.self_attn.indexer.weights_proj",
    "model.layers.45.self_attn.indexer.wk",
    "model.layers.45.self_attn.indexer.wq_b",
    "model.layers.45.self_attn.kv_a_layernorm",
    "model.layers.45.self_attn.kv_b_proj",
    "model.layers.45.self_attn.q_a_layernorm",
    "lm_head",
]


def _skipped(prefix: str) -> bool:
    return is_layer_skipped(
        prefix=prefix,
        ignored_layers=_IGNORED,
        fused_mapping=Glm5NextMTP.packed_modules_mapping,
    )


@pytest.mark.parametrize(
    "prefix,expected",
    [
        # bf16 in the checkpoint.
        ("model.layers.45.eh_proj", True),
        ("model.layers.45.self_attn.kv_b_proj", True),
        ("model.layers.45.mlp.gate", True),
        ("lm_head", True),
        # FP8 block-quantized, with a weight_scale_inv sibling.
        ("model.layers.45.self_attn.fused_qkv_a_proj", False),
        ("model.layers.45.self_attn.q_b_proj", False),
        ("model.layers.45.self_attn.o_proj", False),
        ("model.layers.45.mlp.shared_experts.gate_up_proj", False),
    ])
def test_layer_45_precision_split(prefix, expected):
    assert _skipped(prefix) is expected


def test_lm_head_prefix_is_what_the_ignore_list_names():
    """The draft head must be built as ``lm_head``, not under ``shared_head``.

    ``lm_head.weight`` is bf16 and the ignore list says so under that exact
    name. Upstream's ``glm4_moe_mtp`` parks the head at
    ``model.layers.N.shared_head.head``; here that prefix resolves to *not
    skipped*, i.e. the head would be handed to the FP8 method with no scales.
    """
    assert _skipped("lm_head")
    assert not _skipped("model.layers.45.shared_head.head")


def test_mtp_reuses_the_target_fusion_mapping():
    """One mapping for both models, or layer 45 classifies differently.

    Layers 3..43 and layer 45 have byte-identical self_attn signatures, so a
    fusion declared for one and not the other would make ``is_layer_skipped``
    disagree about the same tensors.
    """
    assert (Glm5NextMTP.packed_modules_mapping
            is Glm5NextForCausalLM.packed_modules_mapping)


# --- weight name mapping ----------------------------------------------


def _rewrite(name: str) -> str | None:
    """Call the real mapper against a stand-in with just the layer index.

    Constructing a ``Glm5NextMTP`` needs a distributed group and a full
    ``VllmConfig``; the mapping under test reads one integer.
    """
    stub = types.SimpleNamespace(model=types.SimpleNamespace(
        mtp_start_layer_idx=_MTP_LAYER))
    return Glm5NextMTP._rewrite_layer_name(stub, name)


@pytest.mark.parametrize(
    "ckpt,expected",
    [
        # Block state gets `.mtp_block` spliced in; the wrapper's own does not.
        ("model.language_model.layers.45.self_attn.q_a_proj.weight",
         "model.layers.45.mtp_block.self_attn.q_a_proj.weight"),
        ("model.language_model.layers.45.input_layernorm.weight",
         "model.layers.45.mtp_block.input_layernorm.weight"),
        ("model.language_model.layers.45.mlp.experts.287.down_proj.weight",
         "model.layers.45.mtp_block.mlp.experts.287.down_proj.weight"),
        ("model.language_model.layers.45.eh_proj.weight",
         "model.layers.45.eh_proj.weight"),
        ("model.language_model.layers.45.enorm.weight",
         "model.layers.45.enorm.weight"),
        ("model.language_model.layers.45.hnorm.weight",
         "model.layers.45.hnorm.weight"),
        ("model.language_model.layers.45.shared_head.norm.weight",
         "model.layers.45.shared_head.norm.weight"),
        # Shared with the target.
        ("model.language_model.embed_tokens.weight",
         "model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ])
def test_checkpoint_names_map_onto_the_draft(ckpt, expected):
    assert _rewrite(ckpt) == expected


@pytest.mark.parametrize(
    "ckpt",
    [
        # Somebody else's weights: the draft is handed the whole checkpoint.
        "model.language_model.layers.44.self_attn.q_proj.weight",
        "model.language_model.layers.43.self_attn.q_a_proj.weight",
        "model.language_model.norm.weight",
        "model.visual.merger.proj.weight",
        # Not built, exactly as on layers 3..43.
        "model.language_model.layers.45.self_attn.indexer.wk.weight",
    ])
def test_foreign_weights_are_dropped(ckpt):
    assert _rewrite(ckpt) is None


def test_layer_44_is_not_swept_up_by_a_prefix_match():
    """``layers.4`` must not match ``layers.45``.

    A `startswith` on the un-dotted prefix would pull layer 4 (KDA) into the
    draft, where its names find no parameter and are dropped -- leaving the
    real layer-45 tensors to load correctly and hiding the bug entirely.
    """
    assert _rewrite("model.language_model.layers.4.self_attn.q_proj.weight") \
        is None


# --- the eh_proj merge -------------------------------------------------


class _RecordingProj(torch.nn.Module):
    """Stands in for ``eh_proj`` and keeps what it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        self.seen = x
        return (x[..., :x.shape[-1] // 2], None)


def _layer_stub(hidden: int = 4):
    """A ``Glm5NextMultiTokenPredictorLayer`` with only the merge wired up.

    ``__new__`` rather than ``__init__``: the real constructor builds MLA and
    a 288-expert MoE, and none of that participates in the operand order.
    """
    layer = Glm5NextMultiTokenPredictorLayer.__new__(
        Glm5NextMultiTokenPredictorLayer)
    torch.nn.Module.__init__(layer)
    # Scale the two norms differently so the halves are distinguishable.
    layer.enorm = lambda x: x * 2.0
    layer.hnorm = lambda x: x * 3.0
    layer.eh_proj = _RecordingProj()
    layer.mtp_block = lambda positions, hidden_states: hidden_states
    return layer, hidden


def test_eh_proj_takes_embeds_first():
    """Embeds in the first half, previous hidden state in the second.

    Confirmed against the real checkpoint: ``eh_proj.weight[:, 4096:]`` is
    ~1.78*I -- a residual pass-through, which can only be the hidden state --
    while the first half has no diagonal structure (cos with the identity
    0.018 vs 0.796). Swapping the two keeps every shape valid.
    """
    layer, hidden = _layer_stub()
    embeds = torch.full((2, hidden), 1.0)
    prev = torch.full((2, hidden), 10.0)
    positions = torch.tensor([5, 6])

    layer.forward(positions, prev, embeds)

    seen = layer.eh_proj.seen
    assert seen.shape == (2, 2 * hidden)
    torch.testing.assert_close(seen[:, :hidden], embeds * 2.0)
    torch.testing.assert_close(seen[:, hidden:], prev * 3.0)


def test_position_zero_embedding_is_masked():
    """Position 0 has no predecessor, so its embedding contributes nothing.

    Every GLM MTP in vLLM does this; without it the draft conditions on a
    token it is supposed to be predicting from.
    """
    layer, hidden = _layer_stub()
    embeds = torch.full((3, hidden), 1.0)
    prev = torch.full((3, hidden), 10.0)
    positions = torch.tensor([0, 1, 2])

    layer.forward(positions, prev, embeds)

    seen = layer.eh_proj.seen
    assert torch.count_nonzero(seen[0, :hidden]) == 0
    assert torch.count_nonzero(seen[1, :hidden]) == hidden
    # The hidden half is never masked -- only the embedding is.
    assert torch.count_nonzero(seen[0, hidden:]) == hidden


# --- the layer type, which the config cannot answer --------------------


def test_config_layer_type_lists_do_not_cover_the_mtp_layer():
    """Indexing them at 45 is the bug this module exists to avoid.

    ``layer_types`` and ``kda_layers`` both stop at 44. Python's negative
    indexing means a naive ``layer_types[45]`` raises, but a modulo or a
    ``get(idx, layer_types[-1])`` fallback silently returns
    ``"linear_attention"`` and builds a KDA draft head -- for which the
    checkpoint has no q_proj, k_proj, v_proj, conv or A_log tensors at all.
    """
    layer_types = ["linear_attention"] * 45
    for full in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43):
        layer_types[full] = "deepseek_sparse_attention"

    assert len(layer_types) == _MTP_LAYER
    assert layer_types[-1] == "linear_attention"
    with pytest.raises(IndexError):
        layer_types[_MTP_LAYER]

    # `linear_attn_config` is no better: neither list mentions 45.
    kda_layers = [
        i for i in range(45)
        if i not in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
    ]
    full_attn_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    assert _MTP_LAYER not in kda_layers
    assert _MTP_LAYER not in full_attn_layers
