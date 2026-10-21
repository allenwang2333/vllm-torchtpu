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
"""Static QSA backend selection and native schema compatibility."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)
from vllm_torchtpu.kernels.qwen4_exp import decode_indexer, pooled_qsa, qsa_rpa


@dataclass(frozen=True)
class ReferenceTopK:
    """Exact test selector; uses actual StreamIndex scores in TPU tests."""

    reverse: bool = False
    overlaps_scoring: ClassVar[bool] = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        assert scheduling_group_id is None
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None] < row_lengths[:, None], scores, -jnp.inf
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(jnp.isfinite(values), ids, -1)
        if self.reverse:
            count = jnp.sum(ids >= 0, axis=1)
            order = jnp.maximum(count[:, None] - 1 - jnp.arange(k)[None], 0)
            ids = jnp.where(
                jnp.arange(k)[None] < count[:, None],
                jnp.take_along_axis(ids, order, axis=1),
                -1,
            )
            values = jnp.where(
                jnp.arange(k)[None] < count[:, None],
                jnp.take_along_axis(values, order, axis=1),
                -jnp.inf,
            )
        return (
            (ids, jax.lax.bitcast_convert_type(values, jnp.int32))
            if return_scores
            else ids
        )


def test_default_is_original_sparsecore_and_jit_backend_identity(monkeypatch):
    traces = []

    def record(q, weights, cache, lengths, pages, cu, distribution, **kwargs):
        backend = kwargs["topk_backend"]
        traces.append(backend)
        return jnp.full(
            (len(q), kwargs["k"]),
            1 if isinstance(backend, SparseCoreTopK) else 2,
            jnp.int32,
        )

    monkeypatch.setattr(decode_indexer, "streamindex_topk", record)
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)
    fn = decode_indexer.select_decode_blocks
    fn.clear_cache()
    args = (
        jnp.zeros((1, 1, 128), jnp.bfloat16),
        jnp.zeros((1, 32, 1, 128), jnp.bfloat16),
        jnp.array([4], jnp.int32),
        jnp.zeros((1, 1), jnp.int32),
        jnp.zeros((1,), jnp.int32),
    )
    np.testing.assert_array_equal(fn(*args), 1)
    np.testing.assert_array_equal(fn(*args, topk_backend=ReferenceTopK()), 2)
    np.testing.assert_array_equal(fn(*args), 1)
    assert traces == [SparseCoreTopK(), ReferenceTopK()]
    fn.clear_cache()


def test_native_factories_accept_default_and_bound_backend_without_backend_init():
    import functools

    import torch
    from jax._src import xla_bridge
    from torch_tpu._internal import pallas

    initialized = torch.tpu.is_initialized()
    clients = tuple(xla_bridge._backends)
    for name, fn in [
        ("pooled", pooled_qsa.pooled_qsa_step),
        ("nonpooled", qsa_rpa.qsa_step),
    ]:
        for label, core in [
            ("default", fn),
            ("alternate", functools.partial(fn, topk_backend=ReferenceTopK())),
        ]:
            op = pallas.jax_op(f"qsa_routing_schema::{name}_{label}", core)
            assert "topk_backend" not in str(op._schema)
    assert torch.tpu.is_initialized() == initialized
    assert tuple(xla_bridge._backends) == clients
