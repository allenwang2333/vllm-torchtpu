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
"""Load the GitHub baseline's native attention route for a CSA2 comparison.

AST selection avoids importing the full serving application. Only imports
are rebound to snapshot definitions; compute function bodies are unchanged.
The staged-KV entrypoint preserves the baseline's fused SWA and full-stage
SparseMLA pipeline, including its measured tile tables.

Packing here is outside timing. This is a compute-path comparison, not a
drop-in serving benchmark: the candidate consumes a different cache format.
"""

import ast
import functools
import logging
import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def select_functions(path: pathlib.Path, names, namespace):
    """Execute snapshot definitions and constants, rebinding package imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definitions[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            definitions[node.targets[0].id] = node
    if not set(names) <= definitions.keys():
        raise ValueError(f"Missing baseline functions in {path}")

    class SnapshotImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in (
                "vllm_torchtpu.kernels.deepseek_v4.core_attention",
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla",
                "vllm_torchtpu.kernels.mla",
            ):
                return None
            return node

    body = [SnapshotImports().visit(definitions[name]) for name in names]
    code = compile(
        ast.Module(body=body, type_ignores=[]), str(path), "exec", flags=0x1000000
    )
    exec(code, namespace)


def load(path: pathlib.Path):
    """Return a wrapper around the measured SWA and SparseMLA implementations."""
    env = dict(
        jax=jax,
        jnp=jnp,
        lax=lax,
        pl=pl,
        pltpu=pltpu,
        functools=functools,
        ROW_WORDS=128,
        NOPE_DIM=448,
        NOPE_SCALE_PERIOD=64,
        logger=types.SimpleNamespace(warning_once=logging.getLogger(__name__).warning),
    )
    kernel_functions = (
        "DEFAULT_VMEM_LIMIT_BYTES",
        "cdiv",
        "align_to",
        "get_dtype_bitwidth",
        "get_dtype_packing",
        "_largest_divisor",
        "_dequant_dsv4_fp8",
        "_attention_kernel",
        "prepare_q_inputs",
        "prepare_swa_inputs",
        "prepare_outputs",
    )
    select_functions(path / "sparse_mla.py", kernel_functions, env)
    env["sparse_mla"] = types.SimpleNamespace(**{n: env[n] for n in kernel_functions})
    select_functions(
        path / "tuned_block_sizes.py", ("PREFILL_BATCHES", "prefill_batch_size"), env
    )
    env["sparse_tuning"] = types.SimpleNamespace(
        prefill_batch_size=env["prefill_batch_size"]
    )
    select_functions(path / "tuned_window_attention.py", ("QUERY_BLOCKS",), env)
    env["tuned_window_attention"] = types.SimpleNamespace(
        QUERY_BLOCKS=env["QUERY_BLOCKS"]
    )
    window_functions = ("supports_window_attention", "_kernel", "attend_windows")
    select_functions(path / "window_attention.py", window_functions, env)
    env["window_attention"] = types.SimpleNamespace(
        **{n: env[n] for n in window_functions}
    )
    select_functions(
        path / "cache.py",
        (
            "_cache_num_rows",
            "_gather_raw_rows",
            "_gather_words",
            "piece_order",
            "split_pieces",
            "merge_pieces",
            "_valid_records",
            "_gather_records",
            "_split_words",
            "gather_cache_pieces",
        ),
        env,
    )
    select_functions(
        path / "attention_kernels.py",
        (
            "_attention_state",
            "_attend",
            "_window_union",
            "_attend_window",
            "repack_chunk_rope_words",
            "_pad_compacted_records",
            "_paged_pool_shape",
            "_reshape_paged_pool",
            "sparse_ragged_paged_attention_union_gather",
            "sparse_fp8_attention_jax",
        ),
        env,
    )

    def baseline(
        q, nope, rope, swa_pool, swa_records, starts, positions, sink, rope_cache
    ):
        del positions
        tokens = q.shape[0]
        valid = jnp.arange(tokens) < starts[-1]
        records = jnp.where(valid[:, None], jnp.arange(512)[None], -1)
        # Pool contents/page addresses are unused when staged buffers are
        # supplied. Keep native shapes without allocating a second cache.
        output = env["sparse_fp8_attention_jax"](
            q,
            swa_pool,
            nope.reshape(-1, 128),
            rope.reshape(-1, 128),
            swa_records,
            records,
            jnp.zeros((starts.shape[0] - 1, 1), jnp.int32),
            starts,
            sink,
            scale=512**-0.5,
            page_size=4,
            max_requests=1 if starts.shape[0] == 2 else None,
            precompacted=True,
            gathered_nope=nope,
            gathered_rope=rope,
        )
        # Same mathematical inverse RoPE and BF16 boundary as the candidate.
        # This is an adapter, not the serving wo_a projection fusion.
        prefix = output[:, :, :448]
        tail = output[:, :, 448:].astype(jnp.float32).reshape(tokens, 64, 32, 2)
        c = rope_cache[:, 0, 64::2, None][:, None]
        s = rope_cache[:, 1, 64::2, None][:, None]
        rotated = jnp.stack((tail[..., 1], -tail[..., 0]), axis=-1)
        tail = (tail * c + rotated * s).reshape(tokens, 64, 64).astype(jnp.bfloat16)
        out = jnp.concatenate((prefix, tail), axis=-1)
        return jnp.where((jnp.arange(tokens) < starts[-1])[:, None, None], out, 0)

    return baseline, env["repack_chunk_rope_words"]


def prepare(inputs, repack):
    """Encode the same values into the original native packed operands."""
    q = inputs["q"]
    tokens = q.shape[0]
    raw = np.asarray(inputs["main_kv"]).reshape(tokens, 512, 512)
    scales = np.asarray(inputs["main_kv_scale"])
    # Power-of-two scales let both formats represent exactly the same KV.
    exponent = np.rint(np.log2(scales)).astype(np.int32)
    scales = np.exp2(exponent).astype(np.float32)
    inputs["main_kv_scale"] = jnp.asarray(scales)
    packed = np.empty((tokens, 512, 512), np.uint8)
    packed[..., :448] = raw.view(np.uint8)[..., :448]
    packed[..., 448:] = (exponent + 127).astype(np.uint8)[..., None]
    # Native cache words pack the four 128-byte slab rows into byte planes,
    # not four adjacent elements (csa_cache_layout.slabs_to_words).
    slabs = packed.reshape(tokens, 512, 4, 128).astype(np.uint32)
    words = (
        slabs[..., 0, :]
        | (slabs[..., 1, :] << 8)
        | (slabs[..., 2, :] << 16)
        | (slabs[..., 3, :] << 24)
    )
    nope = jnp.asarray(words.view(np.int32))
    rope_values = (raw[..., 448:].astype(np.float32) * scales[..., None]).astype(
        jnp.bfloat16
    )
    # RoPE word k contains BF16 elements k and k + 32, as rope_words does.
    rope_bits = rope_values.view(np.uint16).astype(np.uint32)
    rope_words = jnp.asarray(
        (rope_bits[..., :32] | (rope_bits[..., 32:] << 16)).view(np.int32)
    )
    rope = repack(rope_words)
    bounds = np.asarray(inputs["starts"])
    swa = np.asarray(inputs["swa_kv"])
    records = np.full((tokens, 128), -1, np.int32)
    unique = []
    base = 0
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == end:
            continue
        rows = np.concatenate((swa[start], swa[start + 1 : end, -1]), axis=0)
        unique.append(rows)
        records[start:end] = base + np.arange(end - start)[:, None] + np.arange(128)
        base += len(rows)
    pool = jnp.asarray(np.concatenate(unique).view(np.int32).reshape(-1, 128))
    return dict(
        q=q,
        nope=nope,
        rope=rope,
        swa_pool=pool,
        swa_records=jnp.asarray(records),
        starts=inputs["starts"],
        positions=inputs["positions"],
        sink=inputs["sink"],
        rope_cache=inputs["rope_cache"],
    )
