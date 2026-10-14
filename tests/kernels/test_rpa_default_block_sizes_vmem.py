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
"""Default RPA v3 block sizes must fit the chip's VMEM.

Regression test for Gemma-4-31B on one v7x chip (TP=2): the sliding-window
layers (8 KV heads, head_dim 256 per shard) got bq=512 / bkv=2048 at
``max_num_batched_tokens=1024`` -> 68 MiB of scratch, which fails to compile on
the 64 MiB VMEM of v7x. Runs on CPU; the TPU generation is patched.
"""

from unittest import mock

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa

V7X_VMEM = 64 * 1024 * 1024
MIB = 1024 * 1024


def _estimate(bs, *, num_q_heads, num_kv_heads, head_dim):
    return rpa.get_vmem_estimate_bytes(
        num_kv_heads,
        num_q_heads // num_kv_heads,
        head_dim,
        bs["bq_sz"],
        bs["bkv_sz"],
        jnp.bfloat16,
        jnp.bfloat16,
    )


def _default(max_num_tokens, *, num_q_heads, num_kv_heads, head_dim, case):
    return rpa.get_default_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size=16,
        max_num_tokens=max_num_tokens,
        max_num_seqs=128,
        pages_per_seq=1024,  # max_model_len 16384 / page 16
        case=case,
        vmem_limit_bytes=V7X_VMEM,
    )


# Gemma-4-31B at TP=2: sliding layers (16 q / 8 kv heads, hd 256) and global
# layers (16 q / 2 kv heads, hd 512).
GEMMA4_31B_TP2_SLIDING = dict(num_q_heads=16, num_kv_heads=8, head_dim=256)
GEMMA4_31B_TP2_GLOBAL = dict(num_q_heads=16, num_kv_heads=2, head_dim=512)


@pytest.fixture(autouse=True)
def _v7x():
    with mock.patch.object(rpa, "get_tpu_version", return_value=7):
        yield


@pytest.mark.parametrize("max_num_tokens", [512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("heads", [GEMMA4_31B_TP2_SLIDING, GEMMA4_31B_TP2_GLOBAL])
@pytest.mark.parametrize("case", [rpa.RpaCase.MIXED, rpa.RpaCase.DECODE])
def test_default_block_sizes_fit_v7x_vmem(max_num_tokens, heads, case):
    bs = _default(max_num_tokens, case=case, **heads)
    assert _estimate(bs, **heads) <= V7X_VMEM - rpa._VMEM_RESERVE_BYTES
    assert bs["bkv_sz"] % 16 == 0 and bs["bkv_csz"] % 16 == 0
    assert 1 <= bs["bq_csz"] <= bs["bq_sz"]
    assert 16 <= bs["bkv_csz"] <= bs["bkv_sz"]


def test_gemma4_sliding_1024_tokens_shrinks_bkv_only():
    # The failing config from b/573218860: bq=512, bkv=2048 (68 MiB).
    # One halving of the KV block (-> 50 MiB) is enough; bq is untouched.
    bs = _default(1024, case=rpa.RpaCase.MIXED, **GEMMA4_31B_TP2_SLIDING)
    assert (bs["bq_sz"], bs["bkv_sz"]) == (512, 1024)
    assert bs["bkv_csz"] == 512


def test_block_sizes_that_fit_are_unchanged():
    # max_num_batched_tokens=512 compiled before the fix (bq=256, bkv=2048).
    bs = _default(512, case=rpa.RpaCase.MIXED, **GEMMA4_31B_TP2_SLIDING)
    assert (bs["bq_sz"], bs["bkv_sz"], bs["bq_csz"], bs["bkv_csz"]) == (
        256,
        2048,
        256,
        512,
    )


def test_fit_prefers_bkv_then_bq_then_bkv_to_page_size():
    kw = dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        page_size=16,
        actual_num_q_heads=16,
        actual_num_kv_heads=8,
        head_dim=256,
    )
    # Plenty of VMEM: untouched.
    assert rpa.fit_block_sizes_to_vmem(
        512, 2048, 512, 512, vmem_limit_bytes=1024 * MIB, **kw
    ) == (512, 2048, 512, 512)
    # Tight budget: bkv first (2048 -> 512), then bq, then bkv down to a page.
    bq, bkv, bq_c, bkv_c = rpa.fit_block_sizes_to_vmem(
        512, 2048, 512, 512, vmem_limit_bytes=8 * MIB, **kw
    )
    assert bkv < 512 and bq < 512 and bkv % 16 == 0
    assert bq_c <= bq and bkv_c <= bkv
