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
"""Synthetic verify-window attention inputs in the served cache layouts."""

import dataclasses

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L

Q_LEN = 7
WINDOW = 128
SWA_PAGE = 128  # SWA tokens per page of the CSA NoPE host (256 rows of 512 B)
HCA_RATIO = 128
HCA_PAGE = 8  # compressed rows per HCA page (block size 1024)
CSA_RATIO = 4
CSA_PAGE = 256  # compressed rows per CSA / indexer page
IDX_GROUPS = 64  # seq-along-lane word groups per indexer page (33 used)
IDX_HEADS = 64
IDX_DIM = 128


def cdiv(a, b):
    return -(-a // b)


def _bf16(rng, shape, scale=1.0):
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * scale, jnp.bfloat16)


def _table(rng, num_req, pages_per_req, pool, used):
    """Page tables whose first `used[r]` entries are distinct pool pages."""
    perm = rng.permutation(pool)
    table = rng.integers(0, pool, size=(num_req, pages_per_req)).astype(np.int32)
    off = 0
    for r in range(num_req):
        table[r, : used[r]] = perm[off : off + used[r]]
        off += used[r]
    assert off <= pool
    return table


def _bf16_token_pages(rng, num_pages, tokens_per_page):
    toks = _bf16(rng, (num_pages, tokens_per_page, L.HEAD_DIM))
    return L.bf16_tokens_to_words(toks).reshape(num_pages, 2 * tokens_per_page, L.LANES)


@dataclasses.dataclass
class SwaHcaCase:
    q: jnp.ndarray
    new_kv: jnp.ndarray
    swa_words: jnp.ndarray
    swa_table: jnp.ndarray
    hca_words: jnp.ndarray
    hca_table: jnp.ndarray
    seq_lens: jnp.ndarray
    sinks: jnp.ndarray
    sm_scale: float = L.HEAD_DIM**-0.5


def swa_hca_case(rng, seq_lens, heads, *, max_len=None):
    seq_lens = np.asarray(seq_lens, np.int32)
    R = len(seq_lens)
    max_len = max_len or int(seq_lens.max())
    p0 = seq_lens - Q_LEN
    # SWA: the pages under each window and the new rows are live (3 at most).
    swa_pp = cdiv(max_len, SWA_PAGE)
    swa_pool = 3 * R + 2
    lp = np.maximum(p0 - (WINDOW - 1), 0) // SWA_PAGE
    swa_table = rng.integers(0, swa_pool, size=(R, swa_pp)).astype(np.int32)
    perm = rng.permutation(swa_pool)
    for r in range(R):
        for i in range(3):
            if lp[r] + i < swa_pp:
                swa_table[r, lp[r] + i] = perm[3 * r + i]
    hca_pp = max(1, cdiv(cdiv(max_len, HCA_RATIO), HCA_PAGE))
    used = [cdiv(int(s) // HCA_RATIO, HCA_PAGE) for s in seq_lens]
    hca_pool = sum(used) + 2
    return SwaHcaCase(
        q=_bf16(rng, (R, Q_LEN, heads, L.HEAD_DIM)),
        new_kv=_bf16(rng, (R, Q_LEN, L.HEAD_DIM)),
        swa_words=_bf16_token_pages(rng, swa_pool, SWA_PAGE),
        swa_table=jnp.asarray(swa_table),
        hca_words=_bf16_token_pages(rng, hca_pool, HCA_PAGE),
        hca_table=jnp.asarray(_table(rng, R, hca_pp, hca_pool, used)),
        seq_lens=jnp.asarray(seq_lens),
        sinks=jnp.asarray(rng.standard_normal(heads, dtype=np.float32)),
    )


def _fp8_bytes(rng, shape):
    """Random finite e4m3 codes (no 0x7f / 0xff NaN codes)."""
    b = rng.integers(0, 256, size=shape, dtype=np.int64)
    return np.where((b & 0x7F) == 0x7F, b ^ 0x10, b).astype(np.uint32)


def _pack_bytes(b):
    """uint32 bytes [..., 4, n] -> int32 words [..., n] (byte i = bits 8i..)."""
    w = b[..., 0, :] | (b[..., 1, :] << 8) | (b[..., 2, :] << 16) | (b[..., 3, :] << 24)
    return w.astype(np.uint32).view(np.int32)


def idx_pages(rng, num_pages):
    """Seq-along-lane indexer K pages: int32 [N, 64, 256]."""
    g = IDX_DIM // 4
    keys = _pack_bytes(_fp8_normal_bytes(rng, (num_pages, g, 4, CSA_PAGE)))
    scale = rng.integers(118, 126, size=(num_pages, 1, CSA_PAGE)).astype(np.int32)
    junk = rng.integers(-(2**31), 2**31 - 1, size=(num_pages, IDX_GROUPS - g - 1, CSA_PAGE))
    return np.concatenate([keys, scale, junk.astype(np.int32)], axis=1)


def _fp8_normal_bytes(rng, shape):
    """e4m3 codes of N(0, 1) samples."""
    x = jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(jnp.float8_e4m3fn)
    return np.asarray(x.view(jnp.uint8)).astype(np.uint32)


def nope_pages(rng, num_pages):
    """CSA NoPE rows: int32 [N, 256, 128], e4m3 dims 0..447 + e8m0 scales."""
    b = _fp8_normal_bytes(rng, (num_pages, CSA_PAGE, 4, L.LANES))
    b[:, :, 3, 64:] = rng.integers(125, 129, size=(num_pages, CSA_PAGE, 64))
    return _pack_bytes(b)


def rope_pages(rng, num_pages):
    """CSA RoPE rows: int32 [N, 64, 128] (4 tokens of 32 words a row)."""
    vals = _bf16(rng, (num_pages, CSA_PAGE, L.ROPE_DIM))
    u = np.asarray(jnp.asarray(vals).view(jnp.uint16)).astype(np.uint32)
    words = u[..., :32] | (u[..., 32:] << 16)
    return words.view(np.int32).reshape(num_pages, CSA_PAGE // 4, L.LANES)


@dataclasses.dataclass
class CsaCase:
    win: SwaHcaCase  # q, new_kv, SWA cache, seq_lens, sinks (HCA parts unused)
    q_idx: jnp.ndarray  # fp8 [R, 7, 64, 128]
    weights: jnp.ndarray  # f32 [R, 7, 64]
    idx_words: jnp.ndarray
    nope_words: jnp.ndarray
    rope_words: jnp.ndarray
    csa_table: jnp.ndarray  # shared by the indexer K and the NoPE/RoPE arrays here


def csa_case(rng, seq_lens, heads, *, max_len=None):
    seq_lens = np.asarray(seq_lens, np.int32)
    R = len(seq_lens)
    max_len = max_len or int(seq_lens.max())
    win = swa_hca_case(rng, seq_lens, heads, max_len=max_len)
    pp = max(1, cdiv(cdiv(max_len, CSA_RATIO), CSA_PAGE))
    used = [cdiv(int(s) // CSA_RATIO, CSA_PAGE) for s in seq_lens]
    pool = sum(used) + 2
    q = rng.standard_normal((R, Q_LEN, IDX_HEADS, IDX_DIM), dtype=np.float32)
    return CsaCase(
        win=win,
        q_idx=jnp.asarray(q).astype(jnp.float8_e4m3fn),
        weights=jnp.asarray(rng.standard_normal((R, Q_LEN, IDX_HEADS), dtype=np.float32) * 0.05),
        idx_words=jnp.asarray(idx_pages(rng, pool)),
        nope_words=jnp.asarray(nope_pages(rng, pool)),
        rope_words=jnp.asarray(rope_pages(rng, pool)),
        csa_table=jnp.asarray(_table(rng, R, pp, pool, used)),
    )
