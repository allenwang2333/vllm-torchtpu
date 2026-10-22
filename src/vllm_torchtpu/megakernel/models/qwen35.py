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
"""Declarative `ModelDef` for Qwen3.5-397B-A17B."""

from __future__ import annotations  # noqa: I001

import dataclasses
from dataclasses import dataclass
from typing import Any

from .. import cache, kernel
from ..blocks import GDN, GQA, Embedding, GDNBlock, GQABlock, MoE, MoEBlock, Norm  # noqa: F401
from ..exec_spec import ExecSpec
from ..geometry import FromHF, ModelGeometry
from ..model_def import LMHead, ModelDef, register_model
from ..schedule import FromHFLayerTypes, Schedule
from ..spec import Block, ModelSpec
from . import qwen35_candidate30 as _c30, qwen35_pool  # noqa: F401
from .qwen35_pack import (  # noqa: F401
    accept_states, pack_inputs, random_inputs, random_logical_inputs,
)  # fmt: skip
from .qwen35_profile import PROFILE, QWEN35_MODEL_NAME
from .qwen35_weights import (  # noqa: F401
    default_eplb_cores, encode_eplb_ecopy, random_hf_weights, split_hf_weights,
)  # fmt: skip

DEFAULT_HF_CONFIG: dict[str, Any] = {
    "model_type": "qwen3_5_moe_text",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "vocab_size": 248320, "hidden_size": 4096, "num_hidden_layers": 60,
    "rms_norm_eps": 1e-6, "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25,
    "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 64,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "num_experts": 512, "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024, "shared_expert_intermediate_size": 1024,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
}  # fmt: skip
_QWEN35_GEOM_DEFAULTS: dict[str, Any] = {
    "L": 60, "att_every": 4, "H": 4096, "NH": 32, "NKV": 2, "HD": 256, "ROT": 64,
    "KH": 16, "VH": 64, "KD": 128, "VD": 128, "E": 512, "TOPK": 10, "I": 1024,
    "IS": 1024, "moe_tp": True, "e_slots": 0, "eps": 1e-6, "gdn_l2_eps": 1e-6,
    "theta": 10_000_000.0, "zero_centered_norm": True,
}  # fmt: skip


@dataclass(frozen=True, init=False, eq=False)
class Qwen35Geometry(ModelGeometry):
    """Architecture geometry for Qwen3.5-397B-A17B."""

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**{**_QWEN35_GEOM_DEFAULTS, **overrides})

    def to_model_spec(self) -> ModelSpec:
        """Convert this geometry into its declarative `ModelSpec`."""
        return qwen35_spec(self)


@register_model(
    name=QWEN35_MODEL_NAME,
    hf_architectures=(
        "Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM", "Qwen3_5ForCausalLM",
    ),
    hf_model_type=("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text"),
)  # fmt: skip
class Qwen35(ModelDef):
    """Declarative model definition for Qwen3.5-397B-A17B."""

    geometry_cls = Qwen35Geometry
    layer_type_name = "qwen35_layer"
    default_hf_config = DEFAULT_HF_CONFIG
    geometry = FromHF(
        hidden="hidden_size", layers="num_hidden_layers", vocab="vocab_size",
        eps="rms_norm_eps", rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.25),
    )  # fmt: skip
    attn = GQA.bind(
        prefix="self_attn", heads="num_attention_heads",
        kv_heads="num_key_value_heads", head_dim="head_dim",
        qk_norm=True, output_gate=True,
    )  # fmt: skip
    linear = GDN.bind(
        prefix="linear_attn", k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads", k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim", conv_kernel="linear_conv_kernel_dim",
    )  # fmt: skip
    moe = MoE.bind(
        prefix="mlp", experts="num_experts", top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size", shared_gate=True,
    )  # fmt: skip
    schedule = Schedule(
        mixer=FromHFLayerTypes(
            "layer_types", {"full_attention": attn, "linear_attention": linear}
        ),
        ffn=moe,
        pre_norm=Norm.bind("input_layernorm", zero_centered=True),
        post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm", zero_centered=True)
    lm_head = LMHead("lm_head", tied_to=None)
    profiles = {
        "default": PROFILE, "candidate30": _c30.PROFILE,
        "candidate30-fused": _c30.FUSED_PROFILE,
    }  # fmt: skip


# Alias for callers using MKConfig name
MKConfig = Qwen35Geometry
FULL = Qwen35.derive_geometry(DEFAULT_HF_CONFIG, moe_tp=True)
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
TINY = dataclasses.replace(Qwen35.shrink(DEFAULT_HF_CONFIG, periods=1), moe_tp=True)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)
reference_step = Qwen35.reference_step


def qwen35_spec(g: Qwen35Geometry = FULL, gdn_block: Block | None = None) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.5."""
    return Qwen35.to_model_spec(g, gdn_block=gdn_block)


def build(g: Qwen35Geometry, ex: ExecSpec, *, gdn_block: Block | None = None):
    """Build Qwen3.5 megakernel `(call, info)` via the generic `kernel.build` engine."""
    return kernel.build(qwen35_spec(g, gdn_block=gdn_block), ex)


def make_step(
    mesh, g: Qwen35Geometry, ex: ExecSpec, *,
    gdn_block: Block | None = None, donate_states: bool = False,
):  # fmt: skip
    """Build Qwen3.5 `jax.jit` + `shard_map` step `(step, info)`."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return kernel.make_step(mesh, spec, ex, donate_states=donate_states)


def make_pooled_step(
    mesh, g: Qwen35Geometry, ex: ExecSpec, *,
    gdn_block: Block | None = None, donate_pools: bool = False,
    fused_pool_io: bool | str | None = None, max_b_low: int = 24, min_b_hi: int = 56,
):  # fmt: skip
    """Build Qwen3.5 paged-pool `shard_map` step (`PallasWriter`)."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return cache.make_pooled_step(
        mesh, spec, ex, donate_pools=donate_pools, fused_pool_io=fused_pool_io,
        max_b_low=max_b_low, min_b_hi=min_b_hi,
    )  # fmt: skip
