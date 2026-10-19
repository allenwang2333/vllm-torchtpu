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
"""An ordering edge for the chunked DCP merge (TPU_DCP_MERGE_RS_LAG): `tie`
hands back its first operand once the second is available."""

import jax
import torch
from jax import lax
from torch_tpu._internal import pallas


def _tie_fn(a: jax.Array, b: jax.Array) -> jax.Array:
    out, _ = lax.optimization_barrier((a, b))
    return out


def _tie_fake(a: torch.Tensor, b: torch.Tensor):
    return torch.empty_like(a)


_tie_op = pallas.jax_op("pallas::dcp_tie", _tie_fn)
_tie_op.register_fake(_tie_fake)


def tie(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a`, unchanged, available only once `b` is: an optimization barrier."""
    return _tie_op(a.contiguous(), b.contiguous())
