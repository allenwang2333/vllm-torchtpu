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
"""Layer classification for KDA hybrids, across two clashing conventions.

Kimi-Linear numbers ``linear_attn_config["kda_layers"]`` from 1;
GLM-5.3-Flash ships an identically-named key numbered from 0, *plus* a
``layer_types`` list. Reading GLM's list with Kimi's rule mis-sizes the KV
cache without raising, so these tests pin which branch wins.
"""

import types

import pytest


@pytest.fixture
def patched_counter(monkeypatch):
    """Install the TPU layer-count patch over a stub that always defers.

    Upstream's own implementation raises ``ValueError`` for the configs this
    patch exists to handle, so the stub reproduces that and the patch's
    fallback is what gets exercised.
    """
    from vllm.config.model import ModelConfig

    from vllm_torchtpu import _patch_vllm_kimi_kda_layer_counts

    def always_raises(self, parallel_config, block_type="attention"):
        raise ValueError("upstream cannot classify this hybrid")

    monkeypatch.setattr(ModelConfig,
                        "get_num_layers_by_block_type",
                        always_raises,
                        raising=False)
    monkeypatch.setattr(ModelConfig,
                        "_tpu_kimi_kda_layer_counts_patched",
                        False,
                        raising=False)

    _patch_vllm_kimi_kda_layer_counts()
    patched = ModelConfig.get_num_layers_by_block_type

    def count(hf_text_config, num_layers, block_type="attention"):
        stub = types.SimpleNamespace(
            hf_text_config=hf_text_config,
            get_layers_start_end_indices=lambda _pc: (0, num_layers),
        )
        return patched(stub, None, block_type)

    return count


def _glm_text_config(num_layers=8):
    """GLM-5.3-Flash: `layer_types` plus a 0-indexed `kda_layers`."""
    cycle = ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    layer_types = (cycle * ((num_layers // 4) + 1))[:num_layers]
    kda = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    return types.SimpleNamespace(
        layer_types=layer_types,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "kda_layers": kda,  # 0-indexed, unlike Kimi
        },
    )


def _kimi_text_config(num_layers=8):
    """Kimi-Linear: no `layer_types`, and `kda_layers` is 1-indexed."""
    cycle = ["linear_attention"] * 3 + ["full_attention"]
    layer_types = (cycle * ((num_layers // 4) + 1))[:num_layers]
    kda_one_indexed = [
        i + 1 for i, t in enumerate(layer_types) if t == "linear_attention"
    ]
    return types.SimpleNamespace(linear_attn_config={
        "num_heads": 64,
        "head_dim": 128,
        "kda_layers": kda_one_indexed,
    })


def test_glm_layer_types_win_over_kda_layers(patched_counter):
    """GLM's 8 layers are 6 KDA + 2 full, counted from `layer_types`."""
    assert patched_counter(_glm_text_config(8), 8) == 2
    # And at GLM-5.3-Flash's real depth: 45 layers, 11 full-attention.
    assert patched_counter(_glm_text_config(45), 45) == 11


def test_kimi_one_indexed_kda_layers_still_work(patched_counter):
    """The Kimi path is unchanged: no `layer_types`, 1-indexed `kda_layers`."""
    assert patched_counter(_kimi_text_config(8), 8) == 2


def test_glm_would_be_miscounted_by_the_kimi_rule(patched_counter):
    """Pin the bug this ordering avoids.

    Applying Kimi's `idx + 1` rule to GLM's 0-indexed list gives the wrong
    answer, so a config carrying both keys must not fall through to it.
    """
    glm = _glm_text_config(8)
    kda_zero_indexed = set(glm.linear_attn_config["kda_layers"])
    kimi_rule = sum((idx + 1) not in kda_zero_indexed for idx in range(8))

    assert kimi_rule != 2, "test no longer demonstrates the hazard"
    assert patched_counter(glm, 8) == 2


def test_config_without_either_key_defers_upstream(patched_counter):
    """Nothing to classify: the original's ValueError must surface."""
    with pytest.raises(ValueError):
        patched_counter(types.SimpleNamespace(), 8)


def test_non_attention_block_types_defer_upstream(patched_counter):
    """Only `attention` is reclassified; everything else is upstream's."""
    with pytest.raises(ValueError):
        patched_counter(_glm_text_config(8), 8, block_type="mamba")


# --- the layer_types allowlist -------------------------------------------


def test_an_unrecognised_layer_type_raises_rather_than_counting(
        patched_counter):
    """The whole reason this is an allowlist.

    A denylist -- anything that is not the string `linear_attention` is
    attention -- is silent in the direction that costs memory and accuracy:
    every recurrent layer is counted as full attention, a KV cache is sized
    for layers that have none, and the model still builds and still runs.
    """
    config = types.SimpleNamespace(layer_types=["full_attention", "sorcery"])
    with pytest.raises(ValueError, match="sorcery"):
        patched_counter(config, 2)


@pytest.mark.parametrize("spelling", ["mamba", "gated_delta_net", "kda"])
def test_other_linear_spellings_are_classified_not_miscounted(
        patched_counter, spelling):
    """`linear_attention` is GLM's word for it; it is not the only one."""
    layer_types = [spelling] * 3 + ["full_attention"]
    config = types.SimpleNamespace(layer_types=layer_types)
    assert patched_counter(config, 4) == 1


@pytest.mark.parametrize("spelling", [
    "attention", "full_attention", "sliding_attention", "chunked_attention",
    "deepseek_sparse_attention"
])
def test_attention_spellings_all_count(patched_counter, spelling):
    """Sliding and chunked windows still have a KV cache to size."""
    config = types.SimpleNamespace(layer_types=[spelling, "linear_attention"])
    assert patched_counter(config, 2) == 1
