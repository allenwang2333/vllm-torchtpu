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
"""The LM head sharded over the data-parallel engines (TPU_SHARDED_LM_HEAD=1).

Each engine keeps its share of the vocabulary rows of the head (the rows split
over the expert-parallel ranks). Per step: the selected hidden rows of all
engines are all-gathered, each engine multiplies them against its slice (f32
accumulate, bf16 out), and an all-to-all hands every engine its own rows with
the full vocabulary, so each engine reads a fraction of the head's weight per
step instead of all of it. The op is a JAX shard-map op wrapped as a torch
custom op with static shapes: the rows are padded to the largest request
bucket (`install(model, rows)`).

Known limit: the compiled serving graph reads the op's output with the wrong
layout in some program families once the row count passes 16 (a bf16
row-packing mismatch at the torch-to-JAX boundary; the eager op is exact at
every row count). It is right at up to 16 rows per engine with any prompt
chunk, and at up to 20 rows with prompt chunks of at most 768 rows; `install`
refuses every other combination (`unsupported_reason`) until the op's output
layout is pinned at the custom-call boundary."""

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME, build_ep_mesh,
                                               ep_mesh_index)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

OP_NAME = "pallas::sharded_lm_head"
_OP = None


def find_head(model):
    """The vocabulary projection module, wherever the model keeps it (Qwen3.5's conditional-generation
    wrapper holds it at language_model.lm_head). A head that is the token embedding itself (tied
    weights) is left alone: patching it would also change the embedding lookup."""
    modules = list(model.named_modules(remove_duplicate=False))
    heads = [(name, mod) for name, mod in modules
             if name.split(".")[-1] == "lm_head"
             and isinstance(getattr(mod, "weight", None), torch.Tensor)]
    if not heads:
        return None
    name, head = heads[0]
    tied = any(m is head and n.split(".")[-1] != "lm_head" for n, m in modules)
    if tied:
        logger.warning(
            "sharded LM head: %s shares its module with the token embedding; "
            "left as is", name)
        return None
    logger.info("sharded LM head: found %s", name)
    return head


def head_fn(mesh):
    """The op's JAX body over `mesh`: rows gathered, the slice multiplied, the slices exchanged."""

    def one(
        xs, ws
    ):  # this engine's rows [rows, hidden] and vocabulary slice [vocab / n, hidden]
        xa = jax.lax.all_gather(xs, EP_AXIS_NAME, axis=0, tiled=True)
        lg = jnp.dot(xa, ws.T,
                     preferred_element_type=jnp.float32).astype(xs.dtype)
        return jax.lax.all_to_all(lg,
                                  EP_AXIS_NAME,
                                  split_axis=0,
                                  concat_axis=1,
                                  tiled=True)

    def fn(x: jax.Array,
           w: jax.Array) -> jax.Array:  # the op wrapper reads the annotations
        return jax.shard_map(one,
                             mesh=mesh,
                             in_specs=(P(EP_AXIS_NAME), P(EP_AXIS_NAME)),
                             out_specs=P(EP_AXIS_NAME),
                             check_vma=False)(x, w)

    return fn


def build_op(mesh):
    """The sharded projection op on the expert-parallel mesh."""
    n = mesh.shape[EP_AXIS_NAME]
    op = sharded_jax_op(OP_NAME,
                        head_fn(mesh),
                        mesh=mesh,
                        input_partition_specs=(P(EP_AXIS_NAME),
                                               P(EP_AXIS_NAME)),
                        output_partition_specs=P(EP_AXIS_NAME))

    def _fake(x, w):
        return x.new_empty((x.shape[0], w.shape[0] * n))

    op.register_fake(_fake)
    return op


class ShardedHeadMethod:
    """`quant_method` for the head: pad the rows to the fixed count, run the op, trim."""

    def __init__(self, rows: int):
        self.rows = rows

    def apply(self,
              layer,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        """Project x (at most the fixed row count) through the sharded head."""
        r = x.shape[0]
        if r > self.rows:
            raise ValueError(
                f"sharded LM head: {r} rows exceed the fixed {self.rows}")
        if r < self.rows:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.rows - r))
        out = _OP(x, layer.weight_slice)[:r]
        if bias is not None:
            out = out + bias
        return out

    def embedding(self, layer, ids):
        """Not supported: the sharded head is the projection only."""
        raise NotImplementedError(
            "the sharded head is the projection only, never the embedding")


ROWS_ANY_PROMPT = 16
ROWS_SHORT_PROMPTS = 20
PROMPT_ROWS_SHORT = 768


def unsupported_reason(rows: int, prompt_rows: int) -> str | None:
    """Why the head must stay off at this many rows per engine and prompt
    rows per step (the known limit above), or None when it may go on."""
    if rows <= ROWS_ANY_PROMPT:
        return None
    if rows <= ROWS_SHORT_PROMPTS and prompt_rows <= PROMPT_ROWS_SHORT:
        return None
    return (f"{rows} rows per engine with prompt chunks of {prompt_rows} "
            f"rows: the compiled graph reads the op's output with the wrong "
            f"layout above {ROWS_ANY_PROMPT} rows, except at up to "
            f"{ROWS_SHORT_PROMPTS} rows with chunks of at most "
            f"{PROMPT_ROWS_SHORT}")


def install(model, rows: int, prompt_rows: int) -> bool:
    """Replace the model's LM head with the sharded projection: `rows` is
    the largest request count per engine, `prompt_rows` the engine's
    largest prompt chunk (its `--max-num-batched-tokens`)."""
    global _OP
    head = find_head(model)
    if head is None:
        logger.warning("sharded LM head: no head on the model; left as is")
        return False
    reason = unsupported_reason(rows, prompt_rows)
    if reason is not None:
        logger.warning("sharded LM head: %s; left as is", reason)
        return False
    mesh = build_ep_mesh()
    idx = ep_mesh_index()
    if mesh is None or idx is None:
        logger.warning("sharded LM head: no expert-parallel mesh; left as is")
        return False
    n = mesh.shape[EP_AXIS_NAME]
    w = head.weight.data
    vocab, hidden = w.shape
    if vocab % n:
        logger.warning(
            "sharded LM head: vocabulary %d does not split over %d engines; "
            "left as is", vocab, n)
        return False
    part = vocab // n
    # Slice on the host, not on the device: a device-side slice is a lazily materialized strided view whose
    # compiled program is cached under a key that does not carry the slice bounds, so eight engines slicing
    # eight different windows load one another's executables (RuntimeProgramInputMismatch on the second boot).
    w_cpu = w.cpu()
    ws = w_cpu[idx * part:(idx + 1) * part].contiguous().to(w.device)
    head.weight_slice = torch.nn.Parameter(ws, requires_grad=False)
    # The whole head is not needed any more on this engine: keep a one-row stand-in so the module still has
    # a `weight` (shape checks elsewhere read it) without the 2 GB.
    head.weight = torch.nn.Parameter(w_cpu[:1].clone().to(w.device),
                                     requires_grad=False)
    del w
    _OP = build_op(mesh)
    head.quant_method = ShardedHeadMethod(rows)
    logger.info(
        "sharded LM head: %d x %d split %d ways (this engine: mesh index %d, rows %d to %d), "
        "rows padded to %d per call", vocab, hidden, n, idx, idx * part,
        (idx + 1) * part, rows)
    del w_cpu
    return True


def install_if_enabled(model, rows: int, prompt_rows: int) -> bool:
    """Install the sharded head when TPU_SHARDED_LM_HEAD is set."""
    if not envs.TPU_SHARDED_LM_HEAD:
        return False
    return install(model, rows, prompt_rows)
