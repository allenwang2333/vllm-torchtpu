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
"""Config-driven composable TPU Pallas megakernel library."""

from . import (
    blocks,
    comm,
    kernel,
    models,
    ops,
    planner,
    pool_alias,
    spec,
    stream,
    vmem,
)
from .blocks import (
    GDNBlock,
    GDNLinInLayout,
    GDNParams,
    GQABlock,
    GQAParams,
    MoEBlock,
    MoEParams,
)
from .comm import AllReduceRing
from .kernel import (
    KernelContext,
    build,
    fold,
    make_step,
    unfold,
)
from .models.qwen35_pool import (
    conv_to_raw_page,
    default_pool_layout,
    gather_gdn_states,
    kv_to_raw_page,
    pool_to_states,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    ssm_to_raw_pages,
    states_to_pool,
)
from .spec import (
    Block,
    DebugOptions,
    ExecSpec,
    LayerType,
    ModelSpec,
    ScratchReq,
    SmallField,
    Spec,
    StateDecl,
    SwitchSlot,
    Tiles,
    WeightDecl,
    default_tiles,
)
from .stream import TwoRing

__all__ = [
    "AllReduceRing",
    "Block",
    "DebugOptions",
    "ExecSpec",
    "GDNBlock",
    "GDNLinInLayout",
    "GDNParams",
    "GQABlock",
    "GQAParams",
    "KernelContext",
    "LayerType",
    "MoEBlock",
    "MoEParams",
    "ModelSpec",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "Tiles",
    "TwoRing",
    "WeightDecl",
    "blocks",
    "build",
    "comm",
    "conv_to_raw_page",
    "default_pool_layout",
    "default_tiles",
    "fold",
    "gather_gdn_states",
    "kernel",
    "kv_to_raw_page",
    "make_step",
    "models",
    "ops",
    "planner",
    "pool_alias",
    "pool_to_states",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "spec",
    "ssm_to_raw_pages",
    "states_to_pool",
    "stream",
    "unfold",
    "vmem",
]
