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
"""vLLM-torchtpu serving integration for `vllm_torchtpu.megakernel`."""

from __future__ import annotations

from .exported_bridge import (
    PooledStepExport,
    build_pooled_step_op,
    execute_pooled_step_inplace,
    export_pooled_step,
    make_typed_step_wrapper,
    verify_pooled_step_hlo,
)
from .weight_collector import (
    apply_tp_moe_flags,
    extract_ecopy_layer,
    extract_kn_fp8_weight,
    find_backbone,
    get_routed_experts,
    pack_att_in_layer,
    pack_att_out_layer,
    pack_expert_views_layer,
    pack_lin_in_layer,
    pack_lin_out_layer,
    pack_router_and_shared_layer,
    pack_smalls_layer,
    pack_weights,
    quantize_ba_weight,
    reshard_experts_ep_to_tp_local,
    reshard_layer_experts_ep_to_tp,
)

__all__ = [
    "PooledStepExport",
    "apply_tp_moe_flags",
    "build_pooled_step_op",
    "execute_pooled_step_inplace",
    "export_pooled_step",
    "extract_ecopy_layer",
    "extract_kn_fp8_weight",
    "find_backbone",
    "get_routed_experts",
    "make_typed_step_wrapper",
    "pack_att_in_layer",
    "pack_att_out_layer",
    "pack_expert_views_layer",
    "pack_lin_in_layer",
    "pack_lin_out_layer",
    "pack_router_and_shared_layer",
    "pack_smalls_layer",
    "pack_weights",
    "quantize_ba_weight",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
    "verify_pooled_step_hlo",
]
