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
"""TPU implementation of DeepSeek-V4.1 Engram conditional memory."""

import ctypes
import functools
import hashlib
import os
import subprocess
import weakref
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)

from vllm_torchtpu.distributed.tp_collectives import tp_all_gather, tp_shard_slice
from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)

logger = init_logger(__name__)

_pallas_op_cache: dict[str, Any] = {}
_R = P()


def _prod_limbs_jax(tok: jax.Array, m_limbs: jax.Array) -> list[jax.Array]:
    """Multiply 32-bit token IDs [T] by 9x7-bit multiplier limbs [L, 9] in int32."""
    p = tok[:, None, None] * m_limbs[None, :, :]
    out = []
    carry = jnp.zeros_like(p[:, :, 0])
    for k in range(9):
        v = p[:, :, k] + carry
        out.append(v & jnp.int32(0x7F))
        carry = v >> 7
    return out


def _engram_hash_jax(
    input_ids: jax.Array,  # i32[T]
    pos: jax.Array,  # i32[T]
    query_start_loc: jax.Array,  # i32[max_num_seqs + 1]
    swa_block_table: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    distribution: jax.Array,  # i32[3]
    hash_cache: jax.Array,  # i32[num_slots] (donated)
    token_map: jax.Array,  # i32[vocab_size]
    mult_limbs: jax.Array,  # i32[n_layers, max_ngram, 9]
    primes: jax.Array,  # i32[n_layers, max_ngram - 1, n_heads]
    offsets: jax.Array,  # i32[n_layers, (max_ngram - 1) * n_heads]
    *,
    swa_block_size: int,
    pad_id: int,
    max_ngram: int,
) -> tuple[jax.Array, jax.Array]:
    num_tokens = input_ids.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    pages_per_seq = swa_block_table.shape[0] // max_num_seqs
    max_pos = pages_per_seq * swa_block_size - 1
    num_slots = hash_cache.shape[0]
    num_pages = num_slots // swa_block_size

    req = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        jnp.diff(query_start_loc),
        total_repeat_length=num_tokens,
    )
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    req_start_tok = query_start_loc[req]
    compressed = token_map[jnp.clip(input_ids, 0, token_map.shape[0] - 1)]

    def _slot_of(p: jax.Array) -> jax.Array:
        c = jnp.clip(p, 0, max_pos)
        page = swa_block_table[req * pages_per_seq + c // swa_block_size]
        if num_pages > 1:
            page = jnp.where(page > 0, ((page - 1) % (num_pages - 1)) + 1, 0)
        return jnp.clip(page * swa_block_size + (c % swa_block_size), 0, num_slots - 1)

    tokens = []
    blocked = jnp.zeros((num_tokens,), dtype=jnp.bool_)
    for shift in range(max_ngram):
        lookback = pos - shift
        in_batch = (token_idx - shift) >= req_start_tok
        batch_val = compressed[jnp.clip(token_idx - shift, 0, num_tokens - 1)]
        src = jnp.where(in_batch, batch_val, hash_cache[_slot_of(lookback)])
        blocked = blocked | (lookback < 0) | (src == DEAD_ID) | (~valid)
        tokens.append(jnp.where(blocked, jnp.int32(pad_id), src))

    hash_cache = hash_cache.at[jnp.where(valid, _slot_of(pos), num_slots)].set(
        compressed, mode="drop"
    )

    n_layers, _, n_heads = primes.shape
    roll_limbs = _prod_limbs_jax(tokens[0], mult_limbs[:, 0, :])
    hashes_list = []
    for i in range(1, max_ngram):
        cur_limbs = _prod_limbs_jax(tokens[i], mult_limbs[:, i, :])
        roll_limbs = [jnp.bitwise_xor(r, c) for r, c in zip(roll_limbs, cur_limbs)]
        rem = jnp.zeros((num_tokens, n_layers, n_heads), dtype=jnp.int32)
        p_i = primes[None, :, i - 1, :]
        for k in range(8, -1, -1):
            rem = ((rem << 7) + roll_limbs[k][:, :, None]) % p_i
        hashes_list.append(rem)

    out_hashes = jnp.concatenate(hashes_list, axis=-1) + offsets[None, :, :]
    out_hashes = jnp.where(valid[:, None, None], out_hashes, jnp.int32(-1))
    return out_hashes, hash_cache


class TpuNgramHashState(nn.Module):
    """TPU n-gram hash state using 9x7-bit int32 Horner arithmetic."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layout: EngramLayout,
        swa_cache_module: nn.Module,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

        model_config = vllm_config.model_config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size {vocab_size} != {layout.compressed_vocab_size}"
            )
        self.pad_id = int(token_map[layout.pad_token_id])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        mult_limbs = torch.stack(
            [(multipliers >> (7 * k)) & 0x7F for k in range(9)], dim=-1
        ).to(torch.int32)

        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int32), persistent=False
        )
        self.register_buffer("mult_limbs", mult_limbs, persistent=False)
        self.register_buffer(
            "primes", torch.tensor(layout.primes, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "offsets", layout.offsets.to(torch.int32), persistent=False
        )
        logger.info(
            "Built TPU engram token map (%d -> %d ids) for layers %s",
            len(token_map),
            vocab_size,
            layout.layer_ids,
        )
        self._token_map_np = np.ascontiguousarray(token_map, dtype=np.int64)
        self._mult_np = np.ascontiguousarray(multipliers.cpu().numpy(), dtype=np.int64)
        self._primes_np = np.ascontiguousarray(layout.primes, dtype=np.int64)
        self._offsets_np = np.ascontiguousarray(
            layout.offsets.cpu().numpy(), dtype=np.int64
        )

    def cpu_hashes(
        self, token_ids: np.ndarray, req_idx: np.ndarray, pos: np.ndarray
    ) -> np.ndarray:
        """Compute _engram_hash_jax hashes on CPU from host prompt token history."""
        tm = self._token_map_np
        blocked = np.zeros(pos.shape[0], dtype=np.bool_)
        toks = []
        for shift in range(self.layout.max_ngram_size):
            lookback = pos - shift
            blocked |= lookback < 0
            ids = np.clip(
                token_ids[req_idx, np.maximum(lookback, 0)], 0, tm.shape[0] - 1
            )
            toks.append(np.where(blocked, self.pad_id, tm[ids]))
        roll = toks[0][:, None] * self._mult_np[None, :, 0]
        out = []
        for i in range(1, self.layout.max_ngram_size):
            roll = roll ^ (toks[i][:, None] * self._mult_np[None, :, i])
            out.append(roll[:, :, None] % self._primes_np[None, :, i - 1, :])
        return (np.concatenate(out, axis=-1) + self._offsets_np[None]).astype(np.int32)

    @property
    def block_size(self) -> int:
        return self.swa_cache_module.block_size

    def ensure_cache(self) -> bool:
        kv_cache = getattr(self.swa_cache_module, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        self._cache = torch.full(
            (kv_cache.shape[0] * self.block_size,),
            self.pad_id,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True

    def dummy_hashes(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (
            (input_ids.to(torch.int32) * 0)[:, None, None]
            .expand(-1, self.primes.shape[0], self.offsets.shape[1])
            .contiguous()
        )

    @property
    def compressor_op(self):
        # Runner prebuilds ops through this attribute before tracing.
        return self._build_compressor_op()

    def _build_compressor_op(self):
        op_name = (
            f"pallas::deepseek_v41_engram_hash_b{self.block_size}"
            f"_p{self.pad_id}_n{self.layout.max_ngram_size}"
        )
        op = _pallas_op_cache.get(op_name)
        if op is None:
            num_layers = len(self.layout.layer_ids)
            n_hash_cols = (self.layout.max_ngram_size - 1) * self.layout.n_heads
            op = pallas.jax_op(
                op_name,
                functools.partial(
                    _engram_hash_jax,
                    swa_block_size=self.block_size,
                    pad_id=self.pad_id,
                    max_ngram=self.layout.max_ngram_size,
                ),
                mesh=get_vllm_model_wrapper_context().mesh,
                donate_argnums=(5,),
                input_partition_specs=(_R,) * 10,
            )
            op.register_fake(
                lambda input_ids, *args: (
                    input_ids.new_empty(
                        (input_ids.shape[0], num_layers, n_hash_cols), dtype=torch.int32
                    ),
                    torch.empty_like(args[4]),
                )
            )
            _pallas_op_cache[op_name] = op
        return op

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        swa_block_table: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        assert self._cache is not None
        hashes, new_cache = self._build_compressor_op()(
            input_ids if input_ids.dtype == torch.int32 else input_ids.to(torch.int32),
            positions if positions.dtype == torch.int32 else positions.to(torch.int32),
            query_start_loc,
            swa_block_table.reshape(-1),
            distribution,
            self._cache,
            self.token_map,
            self.mult_limbs,
            self.primes,
            self.offsets,
        )
        self._cache.copy_(new_cache)
        return hashes


_CPU_ENGRAM_WEIGHTS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_NEXT_ENGRAM_TABLE_ID = 0
# Double-buffered pinned CPU output pre-sized for max_num_batched_tokens=4096.
_OUT_CPU_BUFS: list[torch.Tensor | None] = [None, None]
_OUT_CPU_SLOT: int = 0
_OUT_CPU_MIN_TOKENS = 4096
_PIN_HOST_BUFS: bool = True
_ENGRAM_PTRS_CACHE: dict[tuple[int, ...], tuple[Any, ...]] = {}
_ENGRAM_PREFETCH: dict[str, Any] | None = None
MADV_HUGEPAGE = 14

# OpenMP row gather from the CPU tables into the packed [value | scale] layout.
_ENGRAM_GATHER_CPP = r"""
#include <cstdint>
#include <cstring>
#include <omp.h>

extern "C" void engram_gather_omp(
    const int32_t* __restrict__ hashes, const uint8_t* const* __restrict__ w_ptrs,
    const uint8_t* const* __restrict__ s_ptrs, const int64_t* __restrict__ max_rows,
    uint8_t* __restrict__ out, int64_t num_tokens, int64_t n_tables,
    int64_t n_hash_cols, int64_t val_dim, int64_t scl_dim, int64_t packed_dim
) {
    const int64_t total_work = num_tokens * n_tables * n_hash_cols;
    if (total_work <= 0) return;
    const int max_t = omp_get_max_threads();
    const int n_threads = (max_t > 16) ? 16 : ((max_t > 0) ? max_t : 1);
    #pragma omp parallel num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        int64_t work = (total_work * tid) / nt;
        const int64_t w_end = (total_work * (tid + 1)) / nt;
        int64_t item = work / n_hash_cols;
        int64_t c = work % n_hash_cols;
        int64_t idx = item % n_tables;
        while (work < w_end) {
            const int64_t rem = w_end - work;
            const int64_t c_end = (n_hash_cols - c < rem) ? n_hash_cols : (c + rem);
            const int64_t span = c_end - c;
            const uint8_t* w_ptr = w_ptrs[idx];
            const uint8_t* s_ptr = s_ptrs[idx];
            const int32_t* h_base = hashes + item * n_hash_cols;
            uint8_t* dst = out + work * packed_dim;
            if (!w_ptr || !s_ptr || h_base[0] < 0) {
                std::memset(dst, 0, span * packed_dim);
            } else {
                const int64_t max_row = max_rows[idx];
                const int64_t next_idx = (idx + 1 == n_tables) ? 0 : (idx + 1);
                const uint8_t* w_next = w_ptrs[next_idx];
                const uint8_t* s_next = s_ptrs[next_idx];
                const int64_t mr_next = max_rows[next_idx];
                const int32_t* h_next_base = h_base + n_hash_cols;
                for (int64_t col = c; col < c_end; ++col, dst += packed_dim) {
                    const int64_t pf_work = work + (col - c) + 4;
                    if (pf_work < w_end) {
                        if (col + 4 < n_hash_cols) {
                            int64_t h_pf = h_base[col + 4];
                            if (h_pf >= 0) {
                                if (h_pf > max_row) h_pf = max_row;
                                const uint8_t* wp = w_ptr + h_pf * val_dim;
                                for (int64_t o = 0; o < val_dim; o += 64)
                                    __builtin_prefetch(wp + o, 0, 1);
                                __builtin_prefetch(s_ptr + h_pf * scl_dim, 0, 1);
                            }
                        } else if (
                            col + 4 - n_hash_cols < n_hash_cols &&
                            w_next && s_next && h_next_base[0] >= 0
                        ) {
                            int64_t h_pf = h_next_base[col + 4 - n_hash_cols];
                            if (h_pf >= 0) {
                                if (h_pf > mr_next) h_pf = mr_next;
                                const uint8_t* wp = w_next + h_pf * val_dim;
                                for (int64_t o = 0; o < val_dim; o += 64)
                                    __builtin_prefetch(wp + o, 0, 1);
                                __builtin_prefetch(s_next + h_pf * scl_dim, 0, 1);
                            }
                        }
                    }
                    int64_t h = h_base[col];
                    h = h < 0 ? 0 : (h > max_row ? max_row : h);
                    std::memcpy(dst, w_ptr + h * val_dim, val_dim);
                    std::memcpy(dst + val_dim, s_ptr + h * scl_dim, scl_dim);
                }
            }
            work += span;
            c = 0;
            ++item;
            if (++idx == n_tables) idx = 0;
        }
    }
}

extern "C" void engram_hash_and_gather_omp(
    const int32_t* __restrict__ token_ids, int64_t token_ids_stride,
    const int32_t* __restrict__ req_idx, const int32_t* __restrict__ pos,
    const int64_t* __restrict__ tm, int64_t tm_len, int64_t pad_id,
    const int64_t* __restrict__ mult, const int64_t* __restrict__ primes,
    const int64_t* __restrict__ offsets,
    const uint8_t* const* __restrict__ w_ptrs,
    const uint8_t* const* __restrict__ s_ptrs,
    const int64_t* __restrict__ max_rows,
    uint8_t* __restrict__ out,
    int64_t num_tokens, int64_t n_tables, int64_t max_ngram,
    int64_t n_heads, int64_t n_hash_cols,
    int64_t val_dim, int64_t scl_dim, int64_t packed_dim
) {
    if (num_tokens <= 0 || n_tables <= 0 || n_hash_cols <= 0) return;
    const int max_t = omp_get_max_threads();
    const int n_threads = (max_t > 16) ? 16 : ((max_t > 0) ? max_t : 1);
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int64_t t = 0; t < num_tokens; ++t) {
        const int64_t r = req_idx[t];
        const int64_t p = pos[t];
        const int32_t* req_tokens = token_ids + r * token_ids_stride;
        int64_t tok[16];
        bool blk = false;
        for (int64_t s = 0; s < max_ngram; ++s) {
            const int64_t lb = p - s;
            if (lb < 0) blk = true;
            if (blk) {
                tok[s] = pad_id;
            } else {
                int64_t raw = req_tokens[lb];
                if (raw < 0) raw = 0;
                else if (raw >= tm_len) raw = tm_len - 1;
                tok[s] = tm[raw];
            }
        }
        for (int64_t tbl = 0; tbl < n_tables; ++tbl) {
            const uint8_t* w_ptr = w_ptrs[tbl];
            const uint8_t* s_ptr = s_ptrs[tbl];
            uint8_t* dst = out + (t * n_tables + tbl) * n_hash_cols * packed_dim;
            if (!w_ptr || !s_ptr) {
                std::memset(dst, 0, n_hash_cols * packed_dim);
                continue;
            }
            const int64_t max_row = max_rows[tbl];
            const int64_t* m_row = mult + tbl * max_ngram;
            const int64_t* p_tbl = primes + tbl * (max_ngram - 1) * n_heads;
            const int64_t* off_tbl = offsets + tbl * n_hash_cols;
            int64_t roll = static_cast<int64_t>(
                static_cast<uint64_t>(tok[0]) * static_cast<uint64_t>(m_row[0])
            );
            int32_t h_arr[64];
            int64_t col = 0;
            for (int64_t i = 1; i < max_ngram; ++i) {
                const int64_t prod = static_cast<int64_t>(
                    static_cast<uint64_t>(tok[i]) * static_cast<uint64_t>(m_row[i])
                );
                roll ^= prod;
                const int64_t* p_row = p_tbl + (i - 1) * n_heads;
                for (int64_t hd = 0; hd < n_heads; ++hd, ++col) {
                    const int64_t pv = p_row[hd];
                    int64_t rem = roll % pv;
                    if (rem < 0) rem += pv;
                    h_arr[col] = static_cast<int32_t>(rem + off_tbl[col]);
                }
            }
            if (h_arr[0] < 0) {
                std::memset(dst, 0, n_hash_cols * packed_dim);
                continue;
            }
            const int64_t init_pf = (n_hash_cols < 4) ? n_hash_cols : 4;
            for (int64_t c0 = 0; c0 < init_pf; ++c0) {
                int64_t h_pf = h_arr[c0];
                if (h_pf >= 0) {
                    if (h_pf > max_row) h_pf = max_row;
                    const uint8_t* wp = w_ptr + h_pf * val_dim;
                    for (int64_t o = 0; o < val_dim; o += 64)
                        __builtin_prefetch(wp + o, 0, 1);
                    __builtin_prefetch(s_ptr + h_pf * scl_dim, 0, 1);
                }
            }
            for (int64_t c0 = 0; c0 < n_hash_cols; ++c0, dst += packed_dim) {
                if (c0 + 4 < n_hash_cols) {
                    int64_t h_pf = h_arr[c0 + 4];
                    if (h_pf >= 0) {
                        if (h_pf > max_row) h_pf = max_row;
                        const uint8_t* wp = w_ptr + h_pf * val_dim;
                        for (int64_t o = 0; o < val_dim; o += 64)
                            __builtin_prefetch(wp + o, 0, 1);
                        __builtin_prefetch(s_ptr + h_pf * scl_dim, 0, 1);
                    }
                }
                int64_t h = h_arr[c0];
                h = h < 0 ? 0 : (h > max_row ? max_row : h);
                std::memcpy(dst, w_ptr + h * val_dim, val_dim);
                std::memcpy(dst + val_dim, s_ptr + h * scl_dim, scl_dim);
            }
        }
    }
}
"""

_ENGRAM_GATHER_FLAGS = ["-O3", "-march=native", "-fopenmp", "-shared", "-fPIC"]


@functools.cache
def _get_engram_gather_lib() -> ctypes.CDLL | None:
    """Build (once per host) and load the OpenMP Engram gather kernel."""
    key = " ".join([_ENGRAM_GATHER_CPP, *_ENGRAM_GATHER_FLAGS])
    so_path = (
        f"/tmp/libengram_gather_omp_{hashlib.sha256(key.encode()).hexdigest()[:12]}.so"
    )
    try:
        if not os.path.exists(so_path):
            tmp_so = f"{so_path}.{os.getpid()}"
            proc = subprocess.run(
                ["g++", *_ENGRAM_GATHER_FLAGS, "-x", "c++", "-", "-o", tmp_so],
                input=_ENGRAM_GATHER_CPP.encode(),
                capture_output=True,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Engram C++ gather build failed:\n%s",
                    proc.stderr.decode(errors="replace"),
                )
                return None
            os.replace(tmp_so, so_path)
        lib = ctypes.CDLL(so_path)
    except OSError as e:
        logger.warning("Engram C++ gather unavailable (%s)", e)
        return None
    lib.engram_gather_omp.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 6
    lib.engram_gather_omp.restype = None
    lib.engram_hash_and_gather_omp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.engram_hash_and_gather_omp.restype = None
    return lib


def _next_host_buf(nbytes: int, row_span: int) -> torch.Tensor:
    global _OUT_CPU_SLOT, _PIN_HOST_BUFS
    slot = _OUT_CPU_SLOT
    _OUT_CPU_SLOT = (slot + 1) & 1
    buf = _OUT_CPU_BUFS[slot]
    if buf is None or buf.numel() < nbytes:
        alloc_bytes = max(nbytes, _OUT_CPU_MIN_TOKENS * row_span)
        if _PIN_HOST_BUFS:
            try:
                buf = torch.empty((alloc_bytes,), dtype=torch.uint8, pin_memory=True)
            except Exception as e:
                logger.warning("Engram: pinned host buffer unavailable (%s)", e)
                _PIN_HOST_BUFS = False
        if not _PIN_HOST_BUFS:
            buf = torch.empty((alloc_bytes,), dtype=torch.uint8)
        _OUT_CPU_BUFS[slot] = buf
    return buf


def _get_table_ptrs(table_ids: list[int], n_tables: int):
    key = tuple(table_ids)
    cached = _ENGRAM_PTRS_CACHE.get(key)
    if cached is None:
        w_ptrs = (ctypes.c_void_p * n_tables)()
        s_ptrs = (ctypes.c_void_p * n_tables)()
        max_rows = (ctypes.c_int64 * n_tables)()
        val_dim = scl_dim = 0
        pairs = [_CPU_ENGRAM_WEIGHTS.get(tid) for tid in table_ids]
        for idx, pair in enumerate(pairs):
            if pair is not None:
                w_u8, s_u8 = pair
                w_ptrs[idx], s_ptrs[idx] = w_u8.data_ptr(), s_u8.data_ptr()
                max_rows[idx] = w_u8.shape[0] - 1
                val_dim, scl_dim = w_u8.shape[1], s_u8.shape[1]
        cached = (w_ptrs, s_ptrs, max_rows, val_dim, scl_dim)
        if all(p is not None for p in pairs):
            _ENGRAM_PTRS_CACHE[key] = cached
    return cached


def _gather_into(
    lib: ctypes.CDLL,
    hashes: np.ndarray,
    table_ids: list[int],
    packed_dim: int,
    out: torch.Tensor,
) -> None:
    num_tokens, n_tables, n_hash_cols = hashes.shape
    w_ptrs, s_ptrs, max_rows, val_dim, scl_dim = _get_table_ptrs(table_ids, n_tables)
    lib.engram_gather_omp(
        hashes.ctypes.data,
        w_ptrs,
        s_ptrs,
        max_rows,
        out.data_ptr(),
        num_tokens,
        n_tables,
        n_hash_cols,
        val_dim,
        scl_dim,
        packed_dim,
    )


def _hash_and_gather_into(
    lib: ctypes.CDLL,
    hash_state: TpuNgramHashState,
    token_ids: np.ndarray,
    req_idx: np.ndarray,
    pos: np.ndarray,
    table_ids: list[int],
    packed_dim: int,
    out: torch.Tensor,
) -> None:
    num_tokens = int(pos.shape[0])
    n_tables = len(table_ids)
    max_ngram = int(hash_state.layout.max_ngram_size)
    n_heads = int(hash_state.layout.n_heads)
    n_hash_cols = int(hash_state.offsets.shape[1])
    w_ptrs, s_ptrs, max_rows, val_dim, scl_dim = _get_table_ptrs(table_ids, n_tables)
    req_c = (
        req_idx
        if (req_idx.dtype == np.int32 and req_idx.flags.c_contiguous)
        else np.ascontiguousarray(req_idx, dtype=np.int32)
    )
    pos_c = (
        pos
        if (pos.dtype == np.int32 and pos.flags.c_contiguous)
        else np.ascontiguousarray(pos, dtype=np.int32)
    )
    lib.engram_hash_and_gather_omp(
        token_ids.ctypes.data,
        int(token_ids.shape[1]),
        req_c.ctypes.data,
        pos_c.ctypes.data,
        hash_state._token_map_np.ctypes.data,
        int(hash_state._token_map_np.shape[0]),
        int(hash_state.pad_id),
        hash_state._mult_np.ctypes.data,
        hash_state._primes_np.ctypes.data,
        hash_state._offsets_np.ctypes.data,
        w_ptrs,
        s_ptrs,
        max_rows,
        out.data_ptr(),
        num_tokens,
        n_tables,
        max_ngram,
        n_heads,
        n_hash_cols,
        val_dim,
        scl_dim,
        packed_dim,
    )


def engram_prefetch_prefill(
    hash_state: TpuNgramHashState,
    table_ids: list[int],
    packed_dim: int,
    token_ids: np.ndarray,
    req_idx: np.ndarray,
    pos: np.ndarray,
    known: np.ndarray,
    padded_num_tokens: int,
) -> None:
    """Gather Engram rows for prompt tokens on CPU during input preparation."""
    global _ENGRAM_PREFETCH
    _ENGRAM_PREFETCH = None
    lib = _get_engram_gather_lib()
    if lib is None:
        return
    all_known_global = bool(known.all())
    if not all_known_global and not known.any():
        return
    num_tokens = int(pos.shape[0])
    n_tables = len(table_ids)
    n_hash_cols = int(hash_state.offsets.shape[1])
    row_bytes = n_tables * n_hash_cols * packed_dim

    padded_tokens = int(padded_num_tokens)

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1 and padded_tokens % tp_size == 0:
        tp_rank = get_tensor_model_parallel_rank()
        t_per_rank = padded_tokens // tp_size
        t_start = tp_rank * t_per_rank
        t_end = t_start + t_per_rank
        n_real_local = max(0, min(num_tokens, t_end) - t_start)
        known_local = known[t_start : t_start + n_real_local]
        all_known_local = bool(known_local.all())
        if n_real_local > 0 and not all_known_local and not known_local.any():
            return
        buf = _next_host_buf(t_per_rank * row_bytes, row_bytes)
        if n_real_local > 0:
            req_local = req_idx[t_start : t_start + n_real_local]
            pos_local = pos[t_start : t_start + n_real_local]
            if (
                all_known_local
                and token_ids.dtype == np.int32
                and token_ids.flags.c_contiguous
            ):
                _hash_and_gather_into(
                    lib,
                    hash_state,
                    token_ids,
                    req_local,
                    pos_local,
                    table_ids,
                    packed_dim,
                    buf,
                )
            else:
                hashes_local = np.full(
                    (n_real_local, n_tables, n_hash_cols), -1, dtype=np.int32
                )
                hashes_local[known_local] = hash_state.cpu_hashes(
                    token_ids, req_local[known_local], pos_local[known_local]
                )
                _gather_into(lib, hashes_local, table_ids, packed_dim, buf)
        _ENGRAM_PREFETCH = {
            "key": tuple(table_ids),
            "sharded": True,
            "t_per_rank": t_per_rank,
            "num_tokens": n_real_local,
            "known": None if all_known_local else known_local.copy(),
            "all_known": all_known_local,
            "buf": buf,
            "row_bytes": row_bytes,
        }
        return

    alloc_tokens = max(num_tokens, padded_tokens)
    buf = _next_host_buf(alloc_tokens * row_bytes, row_bytes)
    if (
        all_known_global
        and token_ids.dtype == np.int32
        and token_ids.flags.c_contiguous
    ):
        _hash_and_gather_into(
            lib, hash_state, token_ids, req_idx, pos, table_ids, packed_dim, buf
        )
    else:
        hashes = np.full((num_tokens, n_tables, n_hash_cols), -1, dtype=np.int32)
        hashes[known] = hash_state.cpu_hashes(token_ids, req_idx[known], pos[known])
        _gather_into(lib, hashes, table_ids, packed_dim, buf)
    _ENGRAM_PREFETCH = {
        "key": tuple(table_ids),
        "num_tokens": num_tokens,
        "known": None if all_known_global else known.copy(),
        "all_known": all_known_global,
        "buf": buf,
        "row_bytes": row_bytes,
    }


def take_prefetched_engram_hashes(
    num_tokens: int,
    n_tables: int,
    n_hash_cols: int,
    packed_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Upload prefetched host Engram rows directly to TPU before compiled stage."""
    global _ENGRAM_PREFETCH
    pf = _ENGRAM_PREFETCH
    if pf is None or not pf.get("all_known", False):
        return None
    _ENGRAM_PREFETCH = None
    row_bytes = n_tables * n_hash_cols * packed_dim
    target_tokens = int(pf["t_per_rank"]) if pf.get("sharded") else num_tokens
    n_real = min(int(pf["num_tokens"]), target_tokens)
    rows = pf["buf"][: target_tokens * row_bytes].view(target_tokens, row_bytes)
    if n_real < target_tokens:
        rows[n_real:].zero_()
    return rows.view(target_tokens, n_tables, n_hash_cols, packed_dim).to(
        device, non_blocking=True
    )


def _lookup_with_prefetch(
    lib: ctypes.CDLL,
    pf: dict[str, Any],
    engram_hashes: torch.Tensor,
    table_ids: list[int],
    packed_dim: int,
    hashes_cpu_ready: torch.Tensor | None = None,
) -> torch.Tensor | None:
    num_tokens = engram_hashes.shape[0]
    row_bytes = pf["row_bytes"]
    buf = pf["buf"]
    if buf.numel() < num_tokens * row_bytes:
        return None
    n_real = pf["num_tokens"]
    rows = buf[: num_tokens * row_bytes].view(num_tokens, row_bytes)
    rows[n_real:].zero_()
    if not pf["all_known"]:
        if hashes_cpu_ready is not None:
            hashes_dev = hashes_cpu_ready[:n_real].to(torch.int32).numpy()
        else:
            hashes_dev = (
                engram_hashes[:n_real]
                .to("cpu", non_blocking=False, dtype=torch.int32)
                .numpy()
            )
        unknown = np.flatnonzero(~pf["known"])
        if unknown.size:
            sub = np.ascontiguousarray(hashes_dev[unknown])
            tmp = torch.empty((unknown.size * row_bytes,), dtype=torch.uint8)
            _gather_into(lib, sub, table_ids, packed_dim, tmp)
            rows[torch.from_numpy(unknown)] = tmp.view(unknown.size, row_bytes)
    return buf[: num_tokens * row_bytes].to(engram_hashes.device, non_blocking=True)


_ENGRAM_D2H_STATE: tuple[torch.Tensor, torch.Tensor, Any] | None = None


@torch.library.custom_op("tpu::engram_hash_ready", mutates_args=())
def engram_hash_ready(engram_hashes: torch.Tensor) -> None:
    global _ENGRAM_D2H_STATE
    _ENGRAM_D2H_STATE = None
    if engram_hashes.device.type != "tpu":
        return None
    pf = _ENGRAM_PREFETCH
    if pf is not None and pf.get("all_known", False):
        return None
    try:
        d2h_cpu = engram_hashes.to("cpu", non_blocking=True)
        d2h_event = torch.tpu.Event()
        d2h_event.record()
        _ENGRAM_D2H_STATE = (engram_hashes, d2h_cpu, d2h_event)
    except Exception:
        _ENGRAM_D2H_STATE = None
    return None


@engram_hash_ready.register_fake
def _engram_hash_ready_fake(engram_hashes: torch.Tensor) -> None:
    return None


@torch.library.custom_op("tpu::engram_cpu_lookup_all", mutates_args=())
def engram_cpu_lookup_all(
    engram_hashes: torch.Tensor,
    table_ids: list[int],
    packed_dim: int,
) -> torch.Tensor:
    global _ENGRAM_PREFETCH, _ENGRAM_D2H_STATE
    d2h_state, _ENGRAM_D2H_STATE = _ENGRAM_D2H_STATE, None
    hashes_cpu_ready: torch.Tensor | None = None
    if (
        d2h_state is not None
        and d2h_state[0].data_ptr() == engram_hashes.data_ptr()
        and d2h_state[0].shape == engram_hashes.shape
    ):
        d2h_state[2].synchronize()
        hashes_cpu_ready = d2h_state[1]
    d2h_state = None
    lib = _get_engram_gather_lib()
    if lib is not None:
        pf, _ENGRAM_PREFETCH = _ENGRAM_PREFETCH, None
        if (
            pf is not None
            and pf["key"] == tuple(table_ids)
            and (
                pf.get("t_per_rank") == engram_hashes.shape[0]
                if pf.get("sharded")
                else pf["num_tokens"] <= engram_hashes.shape[0]
            )
        ):
            out = _lookup_with_prefetch(
                lib,
                pf,
                engram_hashes,
                table_ids,
                packed_dim,
                hashes_cpu_ready=hashes_cpu_ready,
            )
            if out is not None:
                return out
        if hashes_cpu_ready is not None:
            hashes_cpu = hashes_cpu_ready.to(torch.int32).contiguous()
        else:
            hashes_cpu = engram_hashes.to(
                "cpu", non_blocking=False, dtype=torch.int32
            ).contiguous()
        num_tokens, n_tables, n_hash_cols = hashes_cpu.shape
        row_span = n_tables * n_hash_cols * packed_dim
        total_bytes = num_tokens * row_span
        out_cpu = _next_host_buf(total_bytes, row_span)[:total_bytes]
        _gather_into(lib, hashes_cpu.numpy(), table_ids, packed_dim, out_cpu)
        return out_cpu.to(engram_hashes.device, non_blocking=True)

    if hashes_cpu_ready is not None:
        hashes_cpu = hashes_cpu_ready.to(torch.int64)
    else:
        hashes_cpu = engram_hashes.to("cpu", non_blocking=False).to(torch.int64)
    num_tokens, _, n_hash_cols = hashes_cpu.shape
    packed_layers: list[torch.Tensor] = []
    for idx, tid in enumerate(table_ids):
        pair = _CPU_ENGRAM_WEIGHTS.get(tid)
        if pair is None:
            packed_layers.append(
                torch.zeros((num_tokens, n_hash_cols, packed_dim), dtype=torch.uint8)
            )
            continue
        w_u8, s_u8 = pair
        h = hashes_cpu[:, idx, :].clamp(0, w_u8.shape[0] - 1)
        packed_layers.append(torch.cat([w_u8[h], s_u8[h]], dim=-1))
    return (
        torch.stack(packed_layers, dim=1)
        .contiguous()
        .view(-1)
        .to(engram_hashes.device, non_blocking=True)
    )


@engram_cpu_lookup_all.register_fake
def _engram_cpu_lookup_all_fake(
    engram_hashes: torch.Tensor,
    table_ids: list[int],
    packed_dim: int,
) -> torch.Tensor:
    n_tok = engram_hashes.shape[0]
    return engram_hashes.new_empty(
        (n_tok * engram_hashes.shape[1] * engram_hashes.shape[2] * packed_dim,),
        dtype=torch.uint8,
    )


def _engram_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Attach the full Engram table in CPU Host RAM (zero-copy safetensors mmap)."""
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    param.data = loaded_weight.to("cpu")


class TpuParallelEngramEmbedding(nn.Module):
    """CPU-offloaded Engram hash embedding table with TPU TensorCore dequant."""

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        n_hash_cols: int,
        block_size: int = 32,
    ) -> None:
        super().__init__()
        global _NEXT_ENGRAM_TABLE_ID
        self.table_id = _NEXT_ENGRAM_TABLE_ID
        _NEXT_ENGRAM_TABLE_ID += 1

        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.scale_dim = dim // block_size
        self.packed_dim = dim + self.scale_dim
        self.n_hash_cols = n_hash_cols

        # 1-row CPU placeholders so 0 bytes of TPU HBM are used during create_model().
        self.weight = nn.Parameter(
            torch.zeros(1, self.dim, dtype=torch.float8_e4m3fn, device="cpu"),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.zeros(1, self.scale_dim, dtype=torch.uint8, device="cpu"),
            requires_grad=False,
        )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(param, {"weight_loader": _engram_weight_loader})

    def finalize_cpu_offload(self, allow_dummy: bool = False) -> None:
        """Detach CPU Engram tables from _parameters so Dynamo/XLA skips them."""
        w = self._parameters.get("weight")
        s = self._parameters.get("weight_scale_inv")
        if w is not None and s is not None and w.shape[0] > 1:
            del self.weight, self.weight_scale_inv
            w_u8, s_u8 = w.data.view(torch.uint8), s.data.view(torch.uint8)
            try:
                libc = ctypes.CDLL(None)
                for t in (w_u8, s_u8):
                    ptr, aligned = t.data_ptr(), t.data_ptr() & ~4095
                    libc.madvise(
                        ctypes.c_void_p(aligned),
                        ctypes.c_size_t(t.numel() + ptr - aligned),
                        ctypes.c_int(MADV_HUGEPAGE),
                    )
            except (OSError, AttributeError) as e:
                logger.debug("madvise(MADV_HUGEPAGE) unavailable: %s", e)
            _CPU_ENGRAM_WEIGHTS[self.table_id] = (w_u8, s_u8)
        elif allow_dummy and self.table_id not in _CPU_ENGRAM_WEIGHTS:
            for attr in ("weight", "weight_scale_inv"):
                if attr in self._parameters:
                    delattr(self, attr)
            w_u8 = torch.zeros((2, self.dim), dtype=torch.uint8, device="cpu")
            s_u8 = torch.full((2, self.scale_dim), 127, dtype=torch.uint8, device="cpu")
            _CPU_ENGRAM_WEIGHTS[self.table_id] = (w_u8, s_u8)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Dequantize packed [T, n_hash_cols, dim + scale_dim] FP8+E8M0 rows to BF16."""
        if packed.shape[1] != self.n_hash_cols:
            packed = packed[:, : self.n_hash_cols]
        values = packed[..., : self.dim].view(torch.float8_e4m3fn)
        scales_f32 = e8m0_to_fp32(packed[..., self.dim :])
        return (
            (
                values.to(torch.float32).unflatten(
                    -1, (self.scale_dim, self.block_size)
                )
                * scales_f32.unsqueeze(-1)
            )
            .flatten(-2)
            .to(torch.bfloat16)
        )


class TpuEngram(nn.Module):
    """TPU implementation of DeepSeek-V4.1 Engram conditional memory module."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.inv_sqrt_dim = self.dim**-0.5

        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.embed = TpuParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index], layout.head_dim, n_hash_cols
        )
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.register_buffer(
            "qk_weight",
            torch.empty(self.hc_mult, self.dim, dtype=torch.float32),
            persistent=False,
        )
        self._qk_weight_ready = False
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        self.register_buffer(
            "tp_rank_idx",
            torch.tensor([tp_rank], dtype=torch.int64),
            persistent=False,
        )

    def process_weights_after_loading(self, allow_dummy: bool = False) -> None:
        self.embed.finalize_cpu_offload(allow_dummy=allow_dummy)
        qk_cpu = (
            self.q_weight.detach().cpu().float() * self.k_weight.detach().cpu().float()
        ).contiguous()
        self.qk_weight = qk_cpu.to(device=self.q_weight.device, dtype=torch.float32)
        self._qk_weight_ready = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_sharded: bool = False,
    ) -> torch.Tensor:
        """hidden_states: [T or T//tp, hc_mult, dim]; hash_ids: [T or T//tp, C, P]."""
        num_tokens = hidden_states.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        if token_sharded:
            if hash_ids.shape[0] > num_tokens and tp_size > 1:
                tp_rank = get_tensor_model_parallel_rank()
                hash_ids = tp_shard_slice(hash_ids, tp_size, tp_rank, self.tp_rank_idx)
            h_in = hidden_states
            sharded = False
        else:
            sharded = hash_ids.shape[0] < num_tokens
            if (
                not sharded
                and tp_size > 1
                and num_tokens > 0
                and num_tokens % tp_size == 0
            ):
                tp_rank = get_tensor_model_parallel_rank()
                hash_ids = tp_shard_slice(hash_ids, tp_size, tp_rank, self.tp_rank_idx)
                h_in = tp_shard_slice(hidden_states, tp_size, tp_rank, self.tp_rank_idx)
                sharded = True
            elif sharded:
                tp_rank = get_tensor_model_parallel_rank()
                h_in = tp_shard_slice(hidden_states, tp_size, tp_rank, self.tp_rank_idx)
            else:
                h_in = hidden_states

        emb = self.embed(hash_ids).flatten(-2)
        key, value = self.wkv(emb).split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = (
            self.qk_weight
            if self._qk_weight_ready
            else self.q_weight.float() * self.k_weight.float()
        )
        h = h_in.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.inv_sqrt_dim
        abs_sqrt = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0.0, -abs_sqrt, abs_sqrt))
        out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
            hidden_states.dtype
        )
        if sharded:
            out = tp_all_gather(out, dim=0)
        return out
