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
"""The kpool indexer's seven tensors, its config contract, and its loaders.

Everything here
pins something whose wrong answer produces correctly shaped tensors and a model
that loads clean:

* a submodule named anything other than what the checkpoint names loads
  *nothing* and keeps its ``torch.empty`` contents, while the checkpoint tensor
  it wanted is dropped by ``params_dict.get(name) is None``;
* an indexer projection that misses ``modules_to_not_convert`` computes on
  unscaled bf16 through an FP8 method -- the shipped vision-tower bug;
* an ``indexer_types`` entry of ``"shared"``, or
  ``index_kpool_always_select_tail`` False, changes *which keys the model
  attends to* and nothing else.

The recorded checkpoint numbers below were read out of
``zai-org/GLM-5.3-Flash`` snapshot ``03eb5366``: names and shard assignment
from ``model.safetensors.index.json``, dtypes and shapes from the safetensors
headers of shards 2 and 5, and the config values from ``config.json``.
"""

import types

import pytest
import torch

from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    K_NORM_EPS, REAL_CHECKPOINT_CONTRACT, Glm5NextKpoolIndexer,
    assert_indexer_weights_loaded, assert_kpool_config_contract,
    dsa_layer_indices, indexer_parameter_names, kpool_out_width)

#: The seven per-layer tensors, verbatim from the checkpoint index. 84 in
#: total = 7 x 12 layers (DSA 3, 7, ..., 43 plus MTP 45). Every one BF16, none
#: with a ``weight_scale_inv`` sibling.
REAL_INDEXER_TENSORS = {
    "wq_b.weight": (4096, 1536),
    "wk.weight": (128, 4096),
    "k_norm.weight": (128, ),
    "k_norm.bias": (128, ),
    "weights_proj.weight": (32, 4096),
    "index_kpool_compress_ape": (4, 128),
    "index_kpool_compress_gate": (128, 4096),
}

REAL_DSA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
REAL_MTP_LAYER = 45
REAL_INDEXER_TENSOR_COUNT = 84


@pytest.fixture(autouse=True)
def _fake_tp_group(monkeypatch):
    """A single-rank TP group, so ``BasevLLMParameter`` can be constructed.

    ``ReplicatedLinear(disable_tp=True)`` never reads the group itself, but
    ``BasevLLMParameter.__init__`` calls ``get_tensor_model_parallel_rank()``
    unconditionally (``vllm/model_executor/parameter.py:65``) and
    ``update_param_tp_status`` then overwrites it with the layer's 0. Standing
    up a real gloo process group for that would cost seconds per test.
    """
    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state,
                        "_TP",
                        types.SimpleNamespace(rank_in_group=0, world_size=1),
                        raising=False)


def tiny_config(**overrides):
    """A GLM-5.3-shaped text config small enough to build in milliseconds.

    ``index_n_heads`` is 4, not the checkpoint's 32, on purpose: that is what
    ``scripts/glm5_next/build_tiny_fixture.py`` uses, and it is the reason the
    checkpoint's ``index_n_heads == 32`` cannot be a construction-time
    assertion.
    """
    cfg = dict(
        hidden_size=64,
        q_lora_rank=32,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=32,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        qk_rope_head_dim=0,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        indexer_types=["full"] * 4,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def production_config(**overrides):
    """The real checkpoint's geometry, for the shape assertions."""
    layer_types = ["linear_attention"] * 45
    for i in REAL_DSA_LAYERS:
        layer_types[i] = "deepseek_sparse_attention"
    base = dict(hidden_size=4096,
                q_lora_rank=1536,
                index_n_heads=32,
                index_topk=2048,
                num_hidden_layers=45,
                layer_types=layer_types,
                indexer_types=["full"] * 45)
    base.update(overrides)
    return tiny_config(**base)


def build(config=None,
          quant_config=None,
          prefix="model.layers.3.self_attn."
          "indexer",
          layer_idx=3,
          **kwargs):
    return Glm5NextKpoolIndexer(config or tiny_config(),
                                quant_config,
                                prefix=prefix,
                                layer_idx=layer_idx,
                                **kwargs)


# --- the seven tensors --------------------------------------------------


def test_parameter_names_are_exactly_the_checkpoint_names():
    """The module<->checkpoint name bijection, in both directions.

    This is the assertion that catches a mis-named ``Linear``. Neither
    ``default_weight_loader`` nor vLLM's missing-weight net can: a module built
    at ``indexer.wk_proj`` loads nothing and stays at ``torch.empty``, while
    ``...indexer.wk.weight`` finds no parameter and is dropped by ``continue``.
    Both halves are silent.
    """
    indexer = build(production_config())
    got = {name for name, _ in indexer.named_parameters()}
    assert got == set(REAL_INDEXER_TENSORS), (
        f"missing {sorted(set(REAL_INDEXER_TENSORS) - got)}, "
        f"extra {sorted(got - set(REAL_INDEXER_TENSORS))}")
    assert len(got) * (len(REAL_DSA_LAYERS) + 1) == REAL_INDEXER_TENSOR_COUNT


def test_parameter_shapes_match_the_real_checkpoint():
    indexer = build(production_config())
    params = dict(indexer.named_parameters())
    for name, shape in REAL_INDEXER_TENSORS.items():
        assert tuple(params[name].shape) == shape, name


def test_k_norm_is_a_layernorm_with_bias_at_1e_6():
    """Not RMSNorm, not ``rms_norm_eps``.

    ``modeling_glm5_next.py:763`` hard-codes ``nn.LayerNorm(head_dim,
    eps=1e-6)``. GLM's ``rms_norm_eps`` is 1e-5 and every other norm in the
    model is an RMSNorm, so both substitutions are the kind a reader makes for
    consistency. Neither raises; both shift every indexer key slightly, which
    reads as "the port is a bit off" rather than as a bug.
    """
    indexer = build()
    assert isinstance(indexer.k_norm, torch.nn.LayerNorm)
    assert indexer.k_norm.bias is not None
    assert indexer.k_norm.eps == K_NORM_EPS == 1e-6


def test_gate_and_ape_are_bare_parameters_not_linears():
    """``F.linear`` against a raw parameter, as the reference does.

    They are also the only two tensors here that vLLM's weights-not-loaded net
    can see -- ``UnquantizedLinearMethod`` defines
    ``process_weights_after_loading`` and whitelists every ``Linear`` param --
    which is why ``assert_indexer_weights_loaded`` exists.
    """
    indexer = build()
    assert isinstance(indexer.index_kpool_compress_ape, torch.nn.Parameter)
    assert isinstance(indexer.index_kpool_compress_gate, torch.nn.Parameter)
    assert not isinstance(indexer.index_kpool_compress_gate, torch.nn.Module)


def test_projections_are_replicated_not_head_sharded():
    """Head-sharding ``weights_proj`` would be wrong, not merely wasteful.

    ``streamindex_topk`` sums the per-head ReLU'd scores inside the Pallas body
    (``streamindex_topk.py:335``) and takes the top-k of the sum, so at TP=8 a
    rank holding 4 of 32 heads would select a different key set than its peers
    -- over a pooled cache they share.
    """
    indexer = build(production_config())
    for name in ("wq_b", "wk", "weights_proj"):
        layer = getattr(indexer, name)
        assert layer.tp_size == 1 and layer.tp_rank == 0, name
        assert layer.weight.shape[0] == layer.output_size, name


# --- out_width ----------------------------------------------------------


def test_out_width_is_derived_and_covers_the_tail():
    """2176, and the two properties that make trimming it a bug.

    Trimming to ``index_topk`` = 2048 drops the last three columns, which are
    exactly the ``(p+1) % 4`` most recent tokens -- invisible at
    ``L % 4 == 0`` and wrong at the other three residues.
    """
    assert kpool_out_width(2048, 4) == 2176
    assert kpool_out_width(2048, 4) >= 2048 + 4 - 1
    assert kpool_out_width(2048, 4) % 4 == 0
    # The tiny fixture's 64 lands on 128, not on 67.
    assert kpool_out_width(64, 4) == 128
    # 2048 would clip: it is *below* the reference's own emitted width.
    assert 2048 < 2048 + 4 - 1


def test_topk_tokens_is_the_output_width_not_index_topk():
    """The MLA wrapper copies this onto itself (``mla_attention_op.py:508``).

    Sizing ``topk_indices_buffer`` from ``config.index_topk`` instead is short
    by ``index_kpool - 1`` columns and silently clips the tail.
    """
    indexer = build(production_config())
    assert indexer.topk_tokens == 2176
    assert indexer.topk_tokens != indexer.index_topk
    assert indexer.select_k == 2048 // 4 == 512


# --- the config contract ------------------------------------------------


def test_the_real_checkpoint_contract_is_accepted():
    assert_kpool_config_contract(production_config())


def test_recorded_checkpoint_values():
    """Pinned as constants; read from config.json, not derived."""
    assert REAL_CHECKPOINT_CONTRACT["index_topk"] == 2048
    assert REAL_CHECKPOINT_CONTRACT["index_head_dim"] == 128
    assert REAL_CHECKPOINT_CONTRACT["index_n_heads"] == 32
    assert REAL_CHECKPOINT_CONTRACT["index_kpool"] == 4
    assert REAL_CHECKPOINT_CONTRACT["index_kpool_always_select_tail"] is True
    assert REAL_CHECKPOINT_CONTRACT["qk_rope_head_dim"] == 0
    assert REAL_CHECKPOINT_CONTRACT["num_hidden_layers"] == 45
    assert REAL_CHECKPOINT_CONTRACT["index_share_for_mtp_iteration"] is True


def test_dsa_layer_indices_match_the_real_checkpoint():
    assert dsa_layer_indices(production_config()) == list(REAL_DSA_LAYERS)
    assert len(REAL_DSA_LAYERS) == 11


def test_a_shared_indexer_layer_is_refused():
    """``skip_topk`` reuses an earlier layer's buffer; we build a full indexer.

    Both index sets are legal and neither raises -- the wrong one is simply a
    different sparsity pattern on those layers. Trap 17.
    """
    cfg = production_config()
    cfg.indexer_types = list(cfg.indexer_types)
    cfg.indexer_types[7] = "shared"
    with pytest.raises(ValueError, match="indexer_types is not 'full'"):
        assert_kpool_config_contract(cfg)


def test_indexer_types_indexed_by_dsa_layer_is_refused():
    """The shape of the assertion, which is easy to get wrong.

    ``indexer_types`` is indexed by *layer* over all 45 layers. A list of
    length 11 -- one entry per DSA layer -- is a different object entirely, and
    reading ``indexer_types[43]`` out of it would be an IndexError at best and
    somebody else's entry at worst.
    """
    cfg = production_config(indexer_types=["full"] * len(REAL_DSA_LAYERS))
    with pytest.raises(ValueError, match="indexed by \\*layer\\*"):
        assert_kpool_config_contract(cfg)


@pytest.mark.parametrize("overrides,match", [
    ({
        "index_kpool_always_select_tail": False
    }, "always_select_tail"),
    ({
        "index_kpool_always_select_tail": None
    }, "always_select_tail"),
    ({
        "index_head_dim": 64
    }, "index_head_dim is 64"),
    ({
        "index_kpool": 8,
        "index_topk": 32
    }, "index_kpool is 8"),
    ({
        "qk_rope_head_dim": 64
    }, "qk_rope_head_dim is 64"),
    ({
        "index_topk": 30
    }, "must be a multiple of index_kpool"),
    ({
        "indexer_types": None
    }, "declares no `indexer_types`"),
])
def test_config_contract_refusals(overrides, match):
    with pytest.raises(ValueError, match=match):
        assert_kpool_config_contract(production_config(**overrides))


def test_the_contract_is_checked_at_construction_not_at_forward():
    """An assertion that fires at construction, not a comment."""
    cfg = production_config()
    cfg.indexer_types = list(cfg.indexer_types)
    cfg.indexer_types[3] = "shared"
    with pytest.raises(ValueError):
        build(cfg)


def test_index_n_heads_is_not_a_construction_time_assertion():
    """32 in the checkpoint, 4 in the tiny fixture -- so it cannot be pinned.

    It is tempting to list ``index_n_heads == 32`` among the construction-time
    contract items. It cannot be: ``build_tiny_fixture.py`` sets
    ``index_n_heads = attn_heads = 4``, so asserting it would make the rung
    between the CPU oracle and the 328 GB checkpoint unbuildable. It is pinned
    as a recorded constant instead (``test_recorded_checkpoint_values``).
    """
    indexer = build(tiny_config(index_n_heads=4))
    assert indexer.n_head == 4
    assert REAL_CHECKPOINT_CONTRACT["index_n_heads"] == 32


# --- the quantization skip, checked by effect ---------------------------


def _real_ignore_list(prefix):
    """The seven entries the real ``modules_to_not_convert`` carries.

    Note the spelling: ``k_norm`` and ``k_norm.bias`` are both listed, while
    the three projections are listed bare (no ``.weight``).
    """
    return [
        f"{prefix}.index_kpool_compress_ape",
        f"{prefix}.index_kpool_compress_gate",
        f"{prefix}.k_norm",
        f"{prefix}.k_norm.bias",
        f"{prefix}.weights_proj",
        f"{prefix}.wk",
        f"{prefix}.wq_b",
    ]


def test_a_real_fp8_config_leaves_the_indexer_unquantized():
    """End to end through vLLM's own ``Fp8Config.get_quant_method``."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    prefix = "model.layers.3.self_attn.indexer"
    quant_config = Fp8Config(is_checkpoint_fp8_serialized=True,
                             ignored_layers=_real_ignore_list(prefix),
                             weight_block_size=[128, 128])
    indexer = build(quant_config=quant_config, prefix=prefix)
    for name in ("wq_b", "wk", "weights_proj"):
        assert isinstance(
            getattr(indexer, name).quant_method, UnquantizedLinearMethod)


class _QuantizesEverything:
    """A quant config that hands every linear a quantized method.

    Stands in for "the ignore list does not name this module", whatever the
    reason -- a fused name, a renamed submodule, a ``match_mode`` the
    checkpoint did not expect. The assertion under test looks at the *effect*,
    so it does not care which.
    """

    class _Method:

        def create_weights(self, layer, input_size_per_partition,
                           output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            shape = (sum(output_partition_sizes), input_size_per_partition)
            weight = torch.nn.Parameter(torch.empty(shape, dtype=params_dtype),
                                        requires_grad=False)
            layer.register_parameter("weight", weight)

        def apply(self, layer, x, bias=None):  # pragma: no cover - unused
            raise AssertionError("not reached")

    def get_quant_method(self, layer, prefix):
        return self._Method()


def test_construction_refuses_a_quantized_indexer_projection():
    """The name match is not the check; the resulting method is.

    This is the vision-tower failure transplanted: ``is_layer_skipped``
    reported a hit against ``.qkv_proj`` while the checkpoint spelled it
    ``.qkv``, so 24 blocks ran ``Fp8LinearMethod`` over bf16 weights that ship
    no scales -- and every loader check reported 298/298 tensors filled.
    Checking the constructed ``quant_method`` is immune to how the ignore list
    is spelled.
    """
    with pytest.raises(ValueError, match="quantized linear method"):
        build(quant_config=_QuantizesEverything())


def test_the_refusal_names_every_offending_projection():
    with pytest.raises(ValueError) as excinfo:
        build(quant_config=_QuantizesEverything())
    message = str(excinfo.value)
    for name in ("wq_b", "wk", "weights_proj"):
        assert name in message


# --- loader plumbing ----------------------------------------------------


def test_assert_indexer_weights_loaded_is_vacuous_without_an_indexer():
    """Costs nothing before an indexer is built, and self-arms afterwards."""
    module = torch.nn.Linear(2, 2)
    assert indexer_parameter_names(module) == []
    assert_indexer_weights_loaded(module, set())


def _tree_with_indexer():
    """``self_attn.indexer.*``, so the paths look like the real model's.

    ``indexer_parameter_names`` matches ``".indexer."`` with the leading dot,
    which is what distinguishes a nested indexer from a module that merely
    starts with the word.
    """
    root = torch.nn.Module()
    root.self_attn = torch.nn.Module()
    root.self_attn.indexer = build()
    return root


def test_assert_indexer_weights_loaded_accepts_a_complete_load():
    root = _tree_with_indexer()
    names = indexer_parameter_names(root)
    assert len(names) == len(REAL_INDEXER_TENSORS)
    assert all(n.startswith("self_attn.indexer.") for n in names)
    assert_indexer_weights_loaded(root, set(names))


def test_assert_indexer_weights_loaded_rejects_a_dropped_tensor():
    root = _tree_with_indexer()
    names = indexer_parameter_names(root)
    partial = set(names) - {"self_attn.indexer.wk.weight"}
    with pytest.raises(ValueError, match="received no checkpoint tensor"):
        assert_indexer_weights_loaded(root, partial)


def test_no_fusion_rule_captures_an_indexer_tensor_on_the_way_past():
    """The seven names must reach ``params_dict.get(name)`` unmolested.

    ``load_weights`` runs every checkpoint name through the ``_STACKED``
    substring loop and then the expert mapping before falling through to a
    direct lookup. A substring collision there would rewrite an indexer name
    into a fused parameter that does not exist, ``params_dict.get`` would
    return ``None``, and the tensor would be dropped by ``continue`` --
    silently, and only on the layers that have an indexer.
    """
    from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM

    ckpt_names = [
        f"model.layers.3.self_attn.indexer.{leaf}"
        for leaf in REAL_INDEXER_TENSORS
    ]
    fusion_substrings = [ckpt for _, ckpt, _ in Glm5NextForCausalLM._STACKED]
    expert_substrings = ["gate_proj", "up_proj", "down_proj"]
    for name in ckpt_names:
        for needle in fusion_substrings + expert_substrings:
            assert needle not in name, f"{needle!r} captures {name!r}"
