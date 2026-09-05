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
"""The parts of GLM-5.3-Flash's multimodal plumbing that fail silently.

Three of them, and none produces an exception when it is wrong:

* **Where the language model is mounted.** The checkpoint's FP8 ignore list
  spells the text stack ``model.layers.*`` and the tower ``visual.*``, and
  ``is_layer_skipped`` matches those against construction prefixes *exactly*.
  Mounting the text model under ``language_model.`` the way GLM-4V does would
  stop all 1509 entries matching, and 34 bf16 KDA layers would be handed to
  the FP8 method.
* **The placeholder strings.** Inherited from GLM-4V; they have to encode to
  GLM-5.3-Flash's own ids or the prompt updates target nothing.
* **M-RoPE.** GLM-5.3-Flash deletes ``rope_deltas``; declaring
  ``SupportsMRoPE`` would feed the text stack a ``(3, seq_len)`` position
  tensor.

The model class itself is imported lazily throughout, because reaching it
drags vLLM's ``glm4_1v`` and therefore ``torchvision``, whose compiled ops do
not load in the CPU dev venv.
"""

import ast
import os
from pathlib import Path

import pytest
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextForCausalLM

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MULTIMODAL_SRC = (_REPO_ROOT / "src" / "vllm_torchtpu" / "models" / "vllm" /
                   "glm5_next" / "multimodal.py")

#: Path to a GLM-5.3-Flash ``tokenizer.json``. The placeholder-string test
#: below checks the token ids too when one is available; without it the test
#: still pins the strings, which is the half that lives in this repo.
_TOKENIZER_ENV = "GLM5_NEXT_TOKENIZER"

# Faithful excerpt of the real `modules_to_not_convert`: one vision block plus
# the adapter. Note the tower is listed *without* a `model.` prefix even
# though the tensors are `model.visual.*`, which is why the tower is built at
# prefix "visual".
_IGNORED = [
    "model.visual",
    "visual",
    "visual.blocks.0.attn.k_norm",
    "visual.blocks.0.attn.proj",
    "visual.blocks.0.attn.proj.bias",
    "visual.blocks.0.attn.q_norm",
    "visual.blocks.0.attn.qkv",
    "visual.blocks.0.attn.qkv.bias",
    "visual.blocks.0.mlp.down_proj",
    "visual.blocks.0.mlp.down_proj.bias",
    "visual.blocks.0.mlp.gate_proj",
    "visual.blocks.0.mlp.gate_proj.bias",
    "visual.blocks.0.mlp.up_proj",
    "visual.blocks.0.mlp.up_proj.bias",
    "visual.blocks.0.norm1",
    "visual.blocks.0.norm2",
    "visual.downsample",
    "visual.downsample.bias",
    "visual.merger.down_proj",
    "visual.merger.gate_proj",
    "visual.merger.post_projection_norm",
    "visual.merger.post_projection_norm.bias",
    "visual.merger.proj",
    "visual.merger.up_proj",
    "visual.patch_embed.proj",
    "visual.patch_embed.proj.bias",
    "visual.post_layernorm",
    # ...and one text entry, to show the two coexist in one list.
    "model.layers.0.self_attn.q_proj",
]


def _skipped(prefix: str) -> bool:
    return is_layer_skipped(
        prefix=prefix,
        ignored_layers=_IGNORED,
        fused_mapping=Glm5NextForCausalLM.packed_modules_mapping,
    )


@pytest.mark.parametrize("prefix", [
    "visual.patch_embed.proj",
    "visual.blocks.0.attn.qkv",
    "visual.blocks.0.attn.proj",
    "visual.blocks.0.mlp.gate_up_proj",
    "visual.blocks.0.mlp.down_proj",
    "visual.downsample",
    "visual.merger.proj",
    "visual.merger.gate_up_proj",
    "visual.merger.down_proj",
])
def test_whole_tower_stays_bf16(prefix):
    """No part of the tower may be handed to the FP8 method."""
    assert _skipped(prefix), f"{prefix} would be FP8-quantized"


def test_fused_vision_projections_resolve_through_the_packed_mapping():
    """``gate_up_proj`` only matches by being expanded into its two shards."""
    assert "visual.blocks.0.mlp.gate_up_proj" not in _IGNORED
    assert _skipped("visual.blocks.0.mlp.gate_up_proj")
    assert "gate_up_proj" in Glm5NextForCausalLM.packed_modules_mapping


def test_qkv_is_matched_directly_not_expanded():
    """The checkpoint ships ``attn.qkv`` already fused and lists it by name.

    Adding a ``qkv`` entry to ``packed_modules_mapping`` would make
    ``is_layer_skipped`` look for ``attn.q_proj``/``k_proj``/``v_proj``
    instead, none of which the ignore list contains.
    """
    assert "qkv" not in Glm5NextForCausalLM.packed_modules_mapping
    assert _skipped("visual.blocks.0.attn.qkv")


@pytest.mark.parametrize("prefix", [
    "language_model.model.layers.0.self_attn.q_proj",
    "language_model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.qkv",
])
def test_a_nested_mount_would_break_the_ignore_list(prefix):
    """The regression this module's prefix choices exist to prevent."""
    assert not _skipped(prefix)


def _model_classdef() -> ast.ClassDef:
    tree = ast.parse(_MULTIMODAL_SRC.read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "Glm5NextForConditionalGeneration")


def _construction_prefix(attr: str) -> ast.expr:
    """The ``prefix=`` argument of the call assigned to ``self.<attr>``."""
    for node in ast.walk(_model_classdef()):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Attribute) and target.attr == attr
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            continue
        assert isinstance(node.value, ast.Call)
        return next(kw.value for kw in node.value.keywords
                    if kw.arg == "prefix")
    raise AssertionError(f"no `self.{attr} = ...` assignment found")


def test_language_model_is_mounted_at_the_top_level():
    """``prefix=prefix``, i.e. "", so the stack is ``model.layers.*``."""
    prefix = _construction_prefix("language_model")
    assert isinstance(prefix, ast.Name) and prefix.id == "prefix", (
        "the text stack must keep its `model.layers.*` prefixes; see "
        "test_a_nested_mount_would_break_the_ignore_list")


def test_tower_is_mounted_at_visual():
    prefix = _construction_prefix("visual")
    assert isinstance(prefix, ast.Call)
    assert prefix.func.id == "maybe_prefix"
    assert [a.value for a in prefix.args
            if isinstance(a, ast.Constant)] == ["visual"]


def test_mrope_is_not_declared():
    """GLM-5.3-Flash uses plain 1-D positions."""
    bases = [b.id for b in _model_classdef().bases if isinstance(b, ast.Name)]
    assert "SupportsMRoPE" not in bases
    assert "SupportsMultiModal" in bases


@pytest.mark.parametrize("modality,text,expected", [
    ("image", "<|begin_of_image|><|image|><|end_of_image|>",
     [154830, 154854, 154831]),
    ("video", "<|begin_of_video|><|video|><|end_of_video|>",
     [154832, 154855, 154833]),
])
def test_placeholder_strings_encode_to_the_real_ids(modality, text, expected):
    src = _MULTIMODAL_SRC.read_text()
    assert f'"{text}"' in src

    tokenizer_path = os.environ.get(_TOKENIZER_ENV)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        pytest.skip(f"set ${_TOKENIZER_ENV} to a GLM-5.3-Flash "
                    "tokenizer.json to check the ids as well as the strings")
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    assert tokenizer.encode(text, add_special_tokens=False).ids == expected


def test_model_class_shape():
    """Everything above is static. This is the part that needs the real class.

    Skipped where torchvision cannot load; ``importorskip`` is not enough,
    because torchvision imports fine while its compiled ops fail to register
    and the failure surfaces later as ``operator torchvision::nms does not
    exist``.
    """
    try:
        from vllm_torchtpu.models.vllm.glm5_next.multimodal import \
            Glm5NextForConditionalGeneration as model_cls
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"multimodal model not importable here: {exc}")

    from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                       SupportsMultiModal)

    assert issubclass(model_cls, SupportsMultiModal)
    # The KDA layers still need their recurrent state pool, and vLLM asks the
    # registered architecture -- not the inner text model -- for its shape.
    assert issubclass(model_cls, (HasInnerState, IsHybrid))
    for hook in ("get_mamba_state_dtype_from_config",
                 "get_mamba_state_shape_from_config",
                 "get_mamba_state_copy_func"):
        assert hasattr(model_cls, hook)

    assert (model_cls.get_placeholder_str(
        "image", 0) == "<|begin_of_image|><|image|><|end_of_image|>")
    assert (model_cls.get_placeholder_str(
        "video", 0) == "<|begin_of_video|><|video|><|end_of_video|>")
