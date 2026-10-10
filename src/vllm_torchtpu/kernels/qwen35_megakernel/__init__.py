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
"""Qwen3.5-397B-A17B-FP8 whole-backbone decode megakernel for TPU v7x-8."""

from . import megakernel, pool_alias, pool_io
from .megakernel import (
    DEFAULT_TILES,
    FULL,
    FULL_EP,
    TINY,
    TINY_EP,
    TINY_TILES,
    VERIFY_TILES,
    WIDE_TILES,
    MKConfig,
    Tiles,
    build,
    default_tiles,
)
from .pool_io import (
    gather_gdn_states,
    make_pooled_step,
    scatter_pool_states,
)
from .vllm_adapter import (
    Qwen35Megakernel,
    apply_tp_moe_flags,
    extract_kn_fp8_weight,
    pack_att_in_layer,
    pack_att_out_layer,
    pack_expert_views_layer,
    pack_lin_in_layer,
    pack_lin_out_layer,
    pack_router_and_shared_layer,
    pack_smalls_layer,
    pack_weights,
    parse_mk_buckets,
    prepare_step_metadata,
    quantize_ba_weight,
    reshard_experts_ep_to_tp_local,
    reshard_layer_experts_ep_to_tp,
    select_bucket,
)

__all__ = [
    "DEFAULT_TILES",
    "FULL",
    "FULL_EP",
    "MKConfig",
    "Qwen35Megakernel",
    "TINY",
    "TINY_EP",
    "TINY_TILES",
    "Tiles",
    "VERIFY_TILES",
    "WIDE_TILES",
    "apply_tp_moe_flags",
    "build",
    "default_tiles",
    "extract_kn_fp8_weight",
    "gather_gdn_states",
    "make_pooled_step",
    "megakernel",
    "pack_att_in_layer",
    "pack_att_out_layer",
    "pack_expert_views_layer",
    "pack_lin_in_layer",
    "pack_lin_out_layer",
    "pack_router_and_shared_layer",
    "pack_smalls_layer",
    "pack_weights",
    "parse_mk_buckets",
    "pool_alias",
    "pool_io",
    "prepare_step_metadata",
    "quantize_ba_weight",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
    "scatter_pool_states",
    "select_bucket",
]
