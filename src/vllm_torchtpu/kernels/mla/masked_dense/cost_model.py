# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integer cost model for GLM-5.2 masked-dense MLA dispatch."""

from collections.abc import Iterator
from dataclasses import dataclass

_COST_MODEL_FIELDS = (
    "analytic_fixed_ns",
    "analytic_token_ns",
    "bitmap_fixed_ns",
    "bitmap_token_ns",
    "q_kv_block_ns",
    "sparse_fixed_ns",
    "sparse_token_ns",
)


@dataclass(frozen=True)
class MaskedDensePrefillCostModel:
    analytic_fixed_ns: int
    analytic_token_ns: int
    bitmap_fixed_ns: int
    bitmap_token_ns: int
    q_kv_block_ns: int
    sparse_fixed_ns: int
    sparse_token_ns: int

    _fields = _COST_MODEL_FIELDS

    def __post_init__(self) -> None:
        af = self.analytic_fixed_ns
        at = self.analytic_token_ns
        bf = self.bitmap_fixed_ns
        bt = self.bitmap_token_ns
        qkv = self.q_kv_block_ns
        sf = self.sparse_fixed_ns
        st = self.sparse_token_ns
        object.__setattr__(
            self,
            "estimate_analytic_cost_ns",
            lambda num_tokens, num_kv_blocks, _f=af, _t=at, _b=qkv: (
                _f + num_tokens * _t + num_kv_blocks * _b
            ),
        )
        object.__setattr__(
            self,
            "estimate_bitmap_cost_ns",
            lambda num_tokens, num_kv_blocks, _f=bf, _t=bt, _b=qkv: (
                _f + num_tokens * _t + num_kv_blocks * _b
            ),
        )
        object.__setattr__(
            self,
            "estimate_sparse_cost_ns",
            lambda num_tokens, _f=sf, _t=st: _f + num_tokens * _t,
        )

    def __iter__(self) -> Iterator[int]:
        return iter(
            (
                self.analytic_fixed_ns,
                self.analytic_token_ns,
                self.bitmap_fixed_ns,
                self.bitmap_token_ns,
                self.q_kv_block_ns,
                self.sparse_fixed_ns,
                self.sparse_token_ns,
            )
        )

    def estimate_analytic_cost_ns(self, num_tokens: int, num_kv_blocks: int) -> int:
        return (
            self.analytic_fixed_ns
            + num_tokens * self.analytic_token_ns
            + num_kv_blocks * self.q_kv_block_ns
        )

    def estimate_bitmap_cost_ns(self, num_tokens: int, num_kv_blocks: int) -> int:
        return (
            self.bitmap_fixed_ns
            + num_tokens * self.bitmap_token_ns
            + num_kv_blocks * self.q_kv_block_ns
        )

    def estimate_sparse_cost_ns(self, num_tokens: int) -> int:
        return self.sparse_fixed_ns + num_tokens * self.sparse_token_ns


# Geometry of the tuned GLM-5.2 kernel.
KV_BLOCK_SIZE = 1024
QUERY_BLOCK_SIZE = 32

# Conservative integer fits to the TPU7x medians produced by
# scripts/benchmark_masked_dense_vs_sparse_mla.py. The query/KV-grid term also
# includes the grouped mixed measurements at 5,120 and 6,144 KV tokens: a
# 63-decode/961-prefill call wins at 5,120 and loses at 6,144. The
# masked-dense estimate grows with the actual query-block x KV-block grid; the
# sparse gather estimate grows with query tokens and is otherwise nearly flat
# in KV length.
GLM52_TPU7X_PREFILL_COST_MODEL = MaskedDensePrefillCostModel(
    analytic_fixed_ns=215_000,
    analytic_token_ns=245,
    bitmap_fixed_ns=227_000,
    bitmap_token_ns=451,
    q_kv_block_ns=8_700,
    sparse_fixed_ns=259_000,
    sparse_token_ns=2_156,
)
