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

from vllm_torchtpu.megakernel import (
    comm,
    ops,
    planner,
    pool_alias,
    spec,
    stream,
    vmem,
)
from vllm_torchtpu.megakernel.comm import AllReduceRing
from vllm_torchtpu.megakernel.spec import (
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
from vllm_torchtpu.megakernel.stream import TwoRing

__all__ = [
    "AllReduceRing",
    "Block",
    "DebugOptions",
    "ExecSpec",
    "LayerType",
    "ModelSpec",
    "ScratchReq",
    "SmallField",
    "Spec",
    "StateDecl",
    "SwitchSlot",
    "Tiles",
    "TwoRing",
    "WeightDecl",
    "comm",
    "default_tiles",
    "ops",
    "planner",
    "pool_alias",
    "spec",
    "stream",
    "vmem",
]
