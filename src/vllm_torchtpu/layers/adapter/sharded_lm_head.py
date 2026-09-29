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
"""The LM head split by vocabulary rows over the expert-parallel group.

Each rank keeps ``vocab / ep`` rows of the head. Per step the selected hidden
rows of every rank are all-gathered, multiplied against the local slice, and an
all-to-all returns each rank its own rows over the full vocabulary, so a rank
reads ``1 / ep`` of the head's weight instead of all of it.
"""

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _refuse(reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"TPU_SHARDED_LM_HEAD: {reason}. Unset TPU_SHARDED_LM_HEAD to serve "
        "this configuration."
    )


def request_buckets(vllm_config: VllmConfig, max_num_reqs: int) -> list[int]:
    """The runner's request buckets while the head is sharded: exactly one.

    The op's exchange is a collective every EP rank joins with one shape, so
    it runs at a fixed row count. With a second, smaller bucket the compiled
    graph pads rows before the op and trims them after, and above one row
    tile the compiler reads the op's output at the wrong vocabulary positions.
    """
    if vllm_config.speculative_config is not None:
        # TODO: size the op for max_num_reqs * (1 + k) verify rows and give
        # the draft the full head.
        raise _refuse(
            "speculative decoding is not supported; verify steps project "
            "max_num_reqs * (1 + num_speculative_tokens) rows and the draft "
            "copies the target head, which keeps only a vocabulary slice"
        )
    return [max_num_reqs]


def find_head(model: torch.nn.Module) -> ParallelLMHead:
    """The model's untied vocabulary projection."""
    heads = [m for m in model.modules() if isinstance(m, ParallelLMHead)]
    if not heads:
        raise _refuse("the model has no ParallelLMHead on this rank")
    head = heads[0]
    for module in model.modules():
        if (
            module is not head
            and isinstance(module, VocabParallelEmbedding)
            and module.weight is head.weight
        ):
            raise _refuse("the LM head is tied to the token embedding")
    return head


def head_fn(mesh: jax.sharding.Mesh):
    """Gather the rows, multiply the local vocabulary slice, exchange."""

    def local(x: jax.Array, w: jax.Array) -> jax.Array:
        rows = jax.lax.all_gather(x, EP_AXIS_NAME, axis=0, tiled=True)
        logits = jnp.dot(rows, w.T, preferred_element_type=jnp.float32)
        return jax.lax.all_to_all(
            logits.astype(x.dtype),
            EP_AXIS_NAME,
            split_axis=0,
            concat_axis=1,
            tiled=True,
        )

    def fn(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(EP_AXIS_NAME), P(EP_AXIS_NAME)),
            out_specs=P(EP_AXIS_NAME),
            check_vma=False,
        )(x, w)

    return fn


def build_op(mesh: jax.sharding.Mesh):
    ep_size = mesh.shape[EP_AXIS_NAME]
    op = sharded_jax_op(
        "pallas::sharded_lm_head",
        head_fn(mesh),
        mesh=mesh,
        input_partition_specs=(P(EP_AXIS_NAME), P(EP_AXIS_NAME)),
        output_partition_specs=P(EP_AXIS_NAME),
    )

    def _fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0], w.shape[0] * ep_size))

    op.register_fake(_fake)
    return op


class ShardedLMHeadMethod:
    """`quant_method` for the head: pad to the fixed row count, project, trim."""

    def __init__(self, op, rows: int):
        self.op = op
        self.rows = rows

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_rows = x.shape[0]
        if num_rows > self.rows:
            raise ValueError(
                f"sharded LM head: {num_rows} rows exceed the fixed {self.rows}"
            )
        x = torch.nn.functional.pad(x, (0, 0, 0, self.rows - num_rows))
        out = self.op(x, layer.weight_slice)[:num_rows]
        if bias is not None:
            out = out + bias
        return out

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor):
        raise NotImplementedError("The sharded LM head is not an embedding.")


def install(model: torch.nn.Module, buckets: list[int]) -> None:
    """Replace the model's LM head with the sharded projection."""
    assert len(buckets) == 1, f"{buckets=}: take them from request_buckets()"
    head = find_head(model)
    if head.tp_size > 1:
        raise _refuse(
            f"the LM head is already split over {head.tp_size} tensor-parallel "
            "ranks; run with tensor_parallel_size=1 and data parallelism"
        )
    mesh = build_ep_mesh()
    if mesh is None:
        raise _refuse("no expert-parallel group; pass --enable-expert-parallel")
    ep_size = mesh.shape[EP_AXIS_NAME]
    weight = head.weight.data
    vocab, hidden = weight.shape
    if vocab % ep_size:
        raise _refuse(f"the padded vocabulary {vocab} does not split {ep_size} ways")
    part = vocab // ep_size
    index = ep_mesh_index()
    # Sliced on the host: a device-side slice is a lazy strided view whose
    # compiled program is cached without the slice bounds, so ranks slicing
    # different windows would load one another's executables.
    weight_cpu = weight.cpu()
    local = weight_cpu[index * part : (index + 1) * part].contiguous()
    head.weight_slice = torch.nn.Parameter(local.to(weight.device), requires_grad=False)
    # A one-row stand-in keeps `head.weight` for shape reads without the
    # full vocabulary.
    head.weight = torch.nn.Parameter(
        weight_cpu[:1].clone().to(weight.device), requires_grad=False
    )
    rows = buckets[0]
    head.quant_method = ShardedLMHeadMethod(build_op(mesh), rows)
    logger.info(
        "sharded LM head: %d x %d split %d ways, rows %d-%d here, %d rows/call",
        vocab,
        hidden,
        ep_size,
        index * part,
        (index + 1) * part,
        rows,
    )
