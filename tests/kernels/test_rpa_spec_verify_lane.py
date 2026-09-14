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
"""Routing speculative verify windows through the RPA PREFILL lane.

A verify window is `1 + num_speculative_tokens` query tokens attending over the
whole cached history. The runner used to leave those windows in the MIXED
segment, whose fully-ragged loop is what makes per-step spec cost scale ~4.5x
from concurrency 1 to 32. The PREFILL segment runs a *static* q_len while its
KV loop stays bounded by `kv_len`, which is exactly the verify-window shape.

These tests pin the property that makes that routing legal: for a batch of
verify windows over cached history, the PREFILL lane and the MIXED lane must
produce the same output. That is not obvious -- the sibling `batched_rpa`
kernel's PREFILL case zeroes `bkv_p_cache` and so reads no cache at all, which
would make the same reroute silently wrong there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K

HEAD_DIM = 128
NUM_KV_HEADS = 2
NUM_Q_HEADS = 8
PAGE_SIZE = 16
PAGES_PER_SEQ = 16


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}.")


def _build_case(q_lens, kv_lens, seed=0):
    """Lay out a ragged batch: seq i owns pages [i*PAGES_PER_SEQ, ...).

    Returns everything the kernel needs plus the K/V for the *new* tokens; the
    cache is pre-filled with history for positions [0, kv_len - q_len).
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    total_pages = num_seqs * PAGES_PER_SEQ
    dtype = jnp.bfloat16

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    cache_shape = K.get_kv_cache_shape(total_pages, PAGE_SIZE, NUM_KV_HEADS,
                                       HEAD_DIM, dtype)
    cache = np.zeros(cache_shape, np.float32)
    flat = cache.reshape(total_pages * PAGE_SIZE, *cache_shape[2:])

    total_q = int(sum(q_lens))
    query = r(total_q, NUM_Q_HEADS, HEAD_DIM)
    new_k = r(total_q, NUM_KV_HEADS, HEAD_DIM)
    new_v = r(total_q, NUM_KV_HEADS, HEAD_DIM)

    for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        assert kv_len <= PAGES_PER_SEQ * PAGE_SIZE
        # History only: the kernel itself writes the new tokens at
        # [kv_len - q_len, kv_len), and both runs write the same thing.
        hist = kv_len - q_len
        hk, hv = r(hist, NUM_KV_HEADS, HEAD_DIM), r(hist, NUM_KV_HEADS,
                                                    HEAD_DIM)
        merged = np.asarray(
            K.merge_kv(jnp.asarray(hk, dtype),
                       jnp.asarray(hv, dtype)).astype(jnp.float32))
        base = i * PAGES_PER_SEQ * PAGE_SIZE
        flat[base:base + hist] = merged

    page_indices = np.arange(num_seqs * PAGES_PER_SEQ, dtype=np.int32)
    cu_q_lens = np.zeros(num_seqs + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(q_lens)

    # Kept as numpy: the kernel donates the query, k, v and cache buffers, so
    # every run needs its own copies.
    return dict(
        query=query,
        new_k=new_k,
        new_v=new_v,
        cache=cache.reshape(cache_shape),
        kv_lens=jnp.asarray(np.asarray(kv_lens, np.int32)),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q_lens),
        dtype=dtype,
    )


def _run(case, distribution, chunk_prefill_size, fn=None):
    fn = fn or K.ragged_paged_attention
    kwargs = {}
    if chunk_prefill_size is not None:
        kwargs["chunk_prefill_size"] = chunk_prefill_size
    dtype = case["dtype"]
    # All four are donated by the kernel -- hand each run its own copies.
    out, _cache = fn(
        jnp.asarray(case["query"], dtype),
        jnp.asarray(case["new_k"], dtype),
        jnp.asarray(case["new_v"], dtype),
        jnp.asarray(case["cache"], dtype),
        case["kv_lens"],
        case["page_indices"],
        case["cu_q_lens"],
        jnp.asarray(np.asarray(distribution, np.int32)),
        sm_scale=HEAD_DIM**-0.5,
        **kwargs,
    )
    return np.asarray(out.astype(jnp.float32))


@pytest.mark.parametrize("num_spec", [1, 3, 7])
def test_verify_windows_match_between_prefill_and_mixed_lanes(num_spec):
    """The reroute itself: same batch, PREFILL lane vs MIXED lane."""
    _require_tpu()
    width = 1 + num_spec
    num_seqs = 6
    q_lens = [width] * num_seqs
    # Histories that are not multiples of the page or block size, so the
    # cache-tail masking has to be right.
    kv_lens = [200, 37, 91, 128, 255, 16 + width]

    case = _build_case(q_lens, kv_lens, seed=num_spec)

    mixed = _run(case, (0, 0, num_seqs), chunk_prefill_size=None)
    prefill = _run(case, (0, num_seqs, num_seqs), chunk_prefill_size=width)

    # bf16 accumulation order differs between the two lanes.
    np.testing.assert_allclose(prefill, mixed, atol=2e-2, rtol=0)


def test_verify_lane_matches_reference():
    """Absolute check against the reference, which ignores lane boundaries."""
    _require_tpu()
    width = 4
    num_seqs = 4
    q_lens = [width] * num_seqs
    kv_lens = [64, 130, 33, 200]
    case = _build_case(q_lens, kv_lens, seed=11)

    # The reference walks every sequence generically and ignores the lane
    # boundaries, so it takes no `chunk_prefill_size` and is a lane-independent
    # oracle.
    ref = _run(case, (0, num_seqs, num_seqs),
               chunk_prefill_size=None,
               fn=K.ref_ragged_paged_attention)
    prefill = _run(case, (0, num_seqs, num_seqs), chunk_prefill_size=width)

    np.testing.assert_allclose(prefill, ref, atol=2e-2, rtol=0)


def test_three_lanes_together():
    """[decode][verify][prefill] in one batch, the real runner layout."""
    _require_tpu()
    width = 4
    # 3 one-token decodes, 3 verify windows, 2 ragged prefill chunks.
    q_lens = [1, 1, 1, width, width, width, 48, 17]
    kv_lens = [120, 33, 201, 88, 64, 150, 48, 17]
    num_seqs = len(q_lens)
    case = _build_case(q_lens, kv_lens, seed=5)

    # What the runner produces today: verify windows fall into MIXED.
    today = _run(case, (3, 3, num_seqs), chunk_prefill_size=None)
    # What it produces with the verify lane open.
    rerouted = _run(case, (3, 6, num_seqs), chunk_prefill_size=width)

    np.testing.assert_allclose(rerouted, today, atol=2e-2, rtol=0)


def test_shorter_than_static_width_is_safe():
    """A window narrower than `chunk_prefill_size` must not corrupt neighbours.

    The q-block trip count comes from the static width, but the bq fetch and bo
    store both clamp to the real `q_len` (`v3/kernel.py:706,730`). A drafter
    that proposes fewer than K tokens lands here.
    """
    _require_tpu()
    width = 4
    q_lens = [width, 2, 3, width]  # middle two are short proposals
    kv_lens = [96, 51, 77, 140]
    num_seqs = len(q_lens)
    case = _build_case(q_lens, kv_lens, seed=7)

    mixed = _run(case, (0, 0, num_seqs), chunk_prefill_size=None)
    prefill = _run(case, (0, num_seqs, num_seqs), chunk_prefill_size=width)

    np.testing.assert_allclose(prefill, mixed, atol=2e-2, rtol=0)
