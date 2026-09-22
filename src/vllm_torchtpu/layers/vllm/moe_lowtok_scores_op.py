# Copyright 2025 Google LLC
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
"""The few-token MoE kernel's scores entry as a torch custom op: the router's
top-k and the holder assignment run inside the kernel (kernels/moe_lowtok,
moe_lowtok_scores). Per layer the op takes this rank's id, the slot -> global
expert table and the primary / copy holder tables; without a copies
placement they are the contiguous ones. The shard's partial output comes
back; the layer's all-reduce follows as for every other path.
"""
import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.moe_lowtok import moe_lowtok_scores

_ops: dict = {}


def _jax_fn(hidden: jax.Array,
            scores: jax.Array,
            rank: jax.Array,
            gid: jax.Array,
            prim: jax.Array,
            copy: jax.Array,
            hot: jax.Array,
            hprim: jax.Array,
            hcopy: jax.Array,
            hot1h: jax.Array,
            slot: jax.Array,
            w13: jax.Array,
            s13: jax.Array,
            w2: jax.Array,
            s2: jax.Array,
            *,
            topk: int,
            ranks: int,
            owned: int = 1) -> jax.Array:
    return moe_lowtok_scores(hidden,
                             scores,
                             rank,
                             gid,
                             prim,
                             copy,
                             w13,
                             s13,
                             w2,
                             s2,
                             topk=topk,
                             ranks=ranks,
                             owned=owned,
                             hot=hot,
                             hprim=hprim,
                             hcopy=hcopy,
                             hot1h=hot1h,
                             slot=slot)


def _get_op(topk: int, ranks: int, owned: int = 1):
    """Built eagerly by prebuild() when the MoE weights are loaded; the traced forward only looks it up."""
    key = (topk, ranks, owned)
    op = _ops.get(key)
    if op is not None:
        return op
    op = pallas.jax_op(
        f"pallas::moe_lowtok_scores_k{topk}_r{ranks}_o{owned}",
        functools.partial(_jax_fn, topk=topk, ranks=ranks, owned=owned))

    def _fake(hidden, scores, rank, gid, prim, copy, hot, hprim, hcopy, hot1h,
              slot, w13, s13, w2, s2):
        return torch.empty_like(hidden)

    op.register_fake(_fake)
    _ops[key] = op
    return op


def prebuild(topk: int, ranks: int, owned: int = 1):
    _get_op(topk, ranks, owned)


def moe_lowtok_scores_op(hidden,
                         scores,
                         rank,
                         gid,
                         prim,
                         copy,
                         hot,
                         hprim,
                         hcopy,
                         hot1h,
                         slot,
                         w13,
                         s13,
                         w2,
                         s2,
                         *,
                         topk: int,
                         ranks: int,
                         owned: int = 1):
    return _get_op(topk, ranks,
                   owned)(hidden, scores, rank, gid, prim, copy, hot, hprim,
                          hcopy, hot1h, slot, w13, s13, w2, s2)
