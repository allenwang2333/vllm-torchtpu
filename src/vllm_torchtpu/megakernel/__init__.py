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
    cache,
    cache_experimental,
    comm,
    kernel,
    models,
    ops,
    planner,
    pool_alias,
    profile,
    profiles,
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
from .cache import (
    CachePipe,
    DirectHBM,
    PallasWriter,
    check_pooled_step_hlo,
    make_flat_pooled_step,
    make_pooled_step,
    should_fuse_pool_io,
)
from .cache_experimental import (
    check_fused_pooled_step_hlo,
    make_flat_fused_pooled_step,
    make_fused_pooled_step,
)
from .comm import AllReduceRing
from .errors import MegakernelConfigError, MegakernelLayoutError
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
    qwen35_pool_backend,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    scatter_pool_states,
    ssm_to_raw_pages,
    states_to_pool,
)
from .ops import rup
from .planner import encode_ecopy
from .profile import (
    DEFAULT_PROFILE,
    Baseline,
    Profile,
    get_profile,
    list_profiles,
    register_profile,
)
from .spec import (
    Block,
    DebugOptions,
    ExecSpec,
    LayerType,
    ModelSpec,
    PoolBackend,
    ScratchReq,
    SmallField,
    Spec,
    StateDecl,
    SwitchSlot,
    Tiles,
    WeightDecl,
    default_tiles,
    frozen_parity_tiles,
)
from .stream import TwoRing

__all__ = [
    "DEFAULT_PROFILE",
    "AllReduceRing",
    "Baseline",
    "Block",
    "CachePipe",
    "DebugOptions",
    "DirectHBM",
    "ExecSpec",
    "GDNBlock",
    "GDNLinInLayout",
    "GDNParams",
    "GQABlock",
    "GQAParams",
    "KernelContext",
    "LayerType",
    "MegakernelConfigError",
    "MegakernelLayoutError",
    "MoEBlock",
    "MoEParams",
    "ModelSpec",
    "PallasWriter",
    "PoolBackend",
    "Profile",
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
    "cache",
    "cache_experimental",
    "check_fused_pooled_step_hlo",
    "check_pooled_step_hlo",
    "comm",
    "conv_to_raw_page",
    "default_pool_layout",
    "default_tiles",
    "encode_ecopy",
    "fold",
    "frozen_parity_tiles",
    "gather_gdn_states",
    "get_profile",
    "kernel",
    "kv_to_raw_page",
    "list_profiles",
    "make_flat_fused_pooled_step",
    "make_flat_pooled_step",
    "make_fused_pooled_step",
    "make_pooled_step",
    "make_step",
    "models",
    "ops",
    "planner",
    "pool_alias",
    "pool_to_states",
    "profile",
    "profiles",
    "qwen35_pool_backend",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "register_profile",
    "rup",
    "scatter_pool_states",
    "should_fuse_pool_io",
    "spec",
    "ssm_to_raw_pages",
    "states_to_pool",
    "stream",
    "unfold",
    "vmem",
]
