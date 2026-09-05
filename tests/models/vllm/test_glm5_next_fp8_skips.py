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
"""GLM-5.3-Flash's FP8 ignore list must resolve against our module names.

The real checkpoint is FP8 block-quantized but leaves a lot in bf16: all 34
KDA layers entirely, plus ``kv_b_proj`` and the MoE router on the DSA layers.
It says so with 1509 entries in ``quantization_config.modules_to_not_convert``,
and those are *checkpoint* paths, which vLLM matches **exactly** against
*module* prefixes.

That makes submodule naming load-bearing rather than cosmetic: calling the
attention submodule ``attn`` instead of ``self_attn`` stops every KDA layer
from being recognised as unquantized, and they get handed to the FP8 method
with no scales to load. These tests pin the naming and the fusion mapping so
that cannot regress.
"""

import pytest
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM

# Faithful excerpt of the real `modules_to_not_convert`, one KDA layer (0) and
# one DSA layer (3). Verified against zai-org/GLM-5.3-Flash's config.json.
_IGNORED = [
    # KDA layer -- the whole thing stays bf16.
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.b_proj",
    "model.layers.0.self_attn.f_a_proj",
    "model.layers.0.self_attn.f_b_proj",
    "model.layers.0.self_attn.g_a_proj",
    "model.layers.0.self_attn.g_b_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn.q_conv1d",
    "model.layers.0.self_attn.k_conv1d",
    "model.layers.0.self_attn.v_conv1d",
    "model.layers.0.self_attn.A_log",
    "model.layers.0.self_attn.dt_bias",
    "model.layers.0.self_attn.o_norm",
    "model.layers.0.input_layernorm",
    "model.layers.0.post_attention_layernorm",
    # DSA layer -- these plus the indexer block are exempt; q_a/kv_a/q_b/
    # o_proj are FP8.
    "model.layers.3.self_attn.kv_b_proj",
    "model.layers.3.self_attn.q_a_layernorm",
    "model.layers.3.self_attn.kv_a_layernorm",
    "model.layers.3.mlp.gate",
    "model.layers.3.mlp.gate.e_score_correction_bias",
    "model.layers.3.input_layernorm",
    "model.layers.3.post_attention_layernorm",
    # The kpool indexer: all seven tensors, all BF16, all exempt. Verbatim
    # from the real config.json -- note the spelling, which is what these
    # tests exist to pin. The three projections are listed *bare*, with no
    # `.weight`; `k_norm` appears twice, once bare and once as `.bias`; and
    # there is no fused entry of any kind.
    "model.layers.3.self_attn.indexer.index_kpool_compress_ape",
    "model.layers.3.self_attn.indexer.index_kpool_compress_gate",
    "model.layers.3.self_attn.indexer.k_norm",
    "model.layers.3.self_attn.indexer.k_norm.bias",
    "model.layers.3.self_attn.indexer.weights_proj",
    "model.layers.3.self_attn.indexer.wk",
    "model.layers.3.self_attn.indexer.wq_b",
    "lm_head",
]

#: The 12 layers that carry an indexer block: the 11 DSA layers plus MTP 45.
#: 7 x 12 = the 84 `.indexer.` entries in the real `modules_to_not_convert`,
#: and the 84 `.indexer.` tensors in `model.safetensors.index.json`.
_INDEXER_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45)
_INDEXER_TENSORS_PER_LAYER = 7


def _skipped(prefix: str) -> bool:
    return is_layer_skipped(
        prefix=prefix,
        ignored_layers=_IGNORED,
        fused_mapping=Glm5NextForCausalLM.packed_modules_mapping,
    )


@pytest.mark.parametrize("prefix", [
    "model.layers.0.self_attn.fused_qkvb_proj",
    "model.layers.0.self_attn.fused_fa_ga_proj",
    "model.layers.0.self_attn.f_b_proj",
    "model.layers.0.self_attn.g_b_proj",
    "model.layers.0.self_attn.o_proj",
])
def test_kda_layers_stay_unquantized(prefix):
    """Every KDA projection, including the two we fuse, must be skipped."""
    assert _skipped(prefix), f"{prefix} would be FP8-quantized"


@pytest.mark.parametrize("prefix,expected", [
    ("model.layers.3.self_attn.fused_qkv_a_proj", False),
    ("model.layers.3.self_attn.q_b_proj", False),
    ("model.layers.3.self_attn.o_proj", False),
    ("model.layers.3.self_attn.kv_b_proj", True),
    ("model.layers.3.mlp.gate", True),
])
def test_dsa_layer_precision_split(prefix, expected):
    assert _skipped(prefix) is expected


def test_fused_qkv_a_shards_agree_on_precision():
    """Both shards of the MLA down-projection are FP8, so the fusion is legal.

    vLLM raises if a fusion mixes precisions. `q_a_proj` and
    `kv_a_proj_with_mqa` are both quantized in this checkpoint, so fusing them
    is safe -- but it is an assumption worth pinning, because the fused layer
    can only carry one quantization method.
    """
    assert _skipped("model.layers.3.self_attn.fused_qkv_a_proj") is False
    for shard in Glm5NextForCausalLM.packed_modules_mapping[
            "fused_qkv_a_proj"]:
        assert f"model.layers.3.self_attn.{shard}" not in _IGNORED


def test_renaming_self_attn_would_break_kda():
    """Pin the hazard: `attn.` instead of `self_attn.` silently quantizes KDA.

    Nothing raises in that case -- the layers are simply routed to the FP8
    method, which then has no scales to load.
    """
    assert not _skipped("model.layers.0.attn.fused_qkvb_proj")
    assert not _skipped("model.layers.0.attn.o_proj")


# --- the sparse indexer block ------------------------------------------


@pytest.mark.parametrize("prefix", [
    "model.layers.3.self_attn.indexer.wq_b",
    "model.layers.3.self_attn.indexer.wk",
    "model.layers.3.self_attn.indexer.weights_proj",
    "model.layers.3.self_attn.indexer.k_norm",
])
def test_indexer_projections_stay_unquantized(prefix):
    """All 84 indexer tensors are BF16 with no `weight_scale_inv` sibling.

    Read from the checkpoint, not assumed: shards 2 and 5 of
    zai-org/GLM-5.3-Flash carry `wq_b.weight [4096,1536]`,
    `wk.weight [128,4096]`, `k_norm.{weight,bias} [128]`,
    `weights_proj.weight [32,4096]`, `index_kpool_compress_ape [4,128]` and
    `index_kpool_compress_gate [128,4096]`, all BF16.
    """
    assert _skipped(prefix), f"{prefix} would be FP8-quantized"


def test_fusing_wk_with_weights_proj_would_reproduce_the_vision_bug():
    """The negative witness. This is the whole reason they stay separate.

    `VllmTPUIndexer` fuses DeepSeek's `wk` and `weights_proj` into one GEMM
    (`mla_attention_op.py:246-248`), and copying that here is the obvious
    optimization -- one 4096->160 matmul instead of two. But GLM-5.3-Flash's
    ignore list names `wk` and `weights_proj` separately and nothing fused, so
    the fused prefix resolves to *not skipped* and all 11 DSA layers build an
    `Fp8LinearMethod` over bf16 weights that ship no scales.

    Nothing about that fails loudly: `indexer.wk.weight` and
    `indexer.weights_proj.weight` would still be loaded into the fused
    parameter's shards and every loader check would report success. It is the
    shipped vision-tower bug, transplanted -- there, 24 blocks computed on
    unscaled FP8 while 298/298 tensors reported filled.
    """
    assert not _skipped("model.layers.3.self_attn.indexer.wk_weights_proj")
    # And a `packed_modules_mapping` entry would not rescue it either: the
    # shards disagree with nothing here, they are simply not the names the
    # checkpoint uses for a fused layer.
    assert "wk_weights_proj" not in (
        Glm5NextForCausalLM.packed_modules_mapping)


@pytest.mark.parametrize(
    "prefix",
    [
        # Abbreviations a reader might reach for, none of which the list names.
        "model.layers.3.self_attn.indexer.wq",
        "model.layers.3.self_attn.indexer.k_proj",
        "model.layers.3.self_attn.indexer.weight_proj",
        # The submodule renamed: `index` instead of `indexer`.
        "model.layers.3.self_attn.index.wk",
        # The indexer hung off the wrong parent.
        "model.layers.3.indexer.wk",
    ])
def test_a_renamed_indexer_submodule_is_not_skipped(prefix):
    """Pin the hazard rather than the fix.

    Every one of these resolves to "quantize me", silently. The construction-
    time assertion in `glm5_next/indexer.py` is what actually catches it --
    these cases document why a name match alone is not evidence.
    """
    assert not _skipped(prefix)


def test_the_indexer_ignore_entries_are_seven_per_layer():
    """7 x 12 = 84, matching the checkpoint's tensor count exactly."""
    entries = [e for e in _IGNORED if ".indexer." in e]
    assert len(entries) == _INDEXER_TENSORS_PER_LAYER
    assert len(_INDEXER_LAYERS) * _INDEXER_TENSORS_PER_LAYER == 84


def test_packed_modules_mapping_matches_the_fusions_we_build():
    """The mapping must describe exactly the fusions the loader performs."""
    fused_in_loader = {f for f, _, _ in Glm5NextForCausalLM._STACKED}
    for fused in fused_in_loader:
        leaf = fused.split(".")[-1]
        assert leaf in Glm5NextForCausalLM.packed_modules_mapping, (
            f"{leaf} is fused by the weight loader but absent from "
            "packed_modules_mapping, so is_layer_skipped cannot expand it")
