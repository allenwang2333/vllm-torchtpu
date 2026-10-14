# SPDX-License-Identifier: Apache-2.0
"""Poisoned-VMEM NaN tests for batched ragged paged attention.

The kernel relies on zero-initialized VMEM for everything it reads but never
fetches (idle batch slots, lanes past `kv_len`, columns past the stitched new
tokens): `p = 0` only neutralizes `v` in the PV matmul if `v` is finite. With the
kernel's test-only hook `kernel.POISON_UNUSED_VMEM_FOR_TESTS` on, `rpa_kernel`
first fills every Q/KV pipeline slot with a NaN bit pattern and only then runs its
gated zero-fill, so a slot a step reads without having been zeroed shows up as NaN
in the outputs or in the written-back KV pages, which are checked against a
reference.

Coverage: num_seqs 1/4/8 at batch_size 8 (idle lanes), decode query sizes 1..4
(stitch columns, page straddles), 1-, 2- and 3-step schedules (every input slot),
a >= 2-chunk schedule (slots re-entered across chunks), pure-cache (CACHE_ONLY +
skip_kv_update, no stitch), zero steps (early exit), and both KV layouts.

Harness check, to be repeated whenever the kernel's VMEM initialization changes:
with the Q/KV zero-fill in `kernel.py` (`_run` -> `_init_vmem` ->
`_zero_input_slot`) disabled, the one-step, pure-cache, multi-step and multi-chunk
cases must FAIL on the NaN assertions; with the fill in place every case passes.
"""

import functools

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    utils,
    wrapper,
)

pytestmark = pytest.mark.tpu_test

# The E2E decode geometry (Qwen3.5 at TP8): 4 q heads over 1 kv head, fp8 KV.
_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 1
_HEAD_DIM = 256
_PAGE_SIZE = 128
_BATCH_SIZE = 8
_Q_DTYPE = jnp.bfloat16
_KV_DTYPE = jnp.float8_e4m3fn
_SM_SCALE = _HEAD_DIM**-0.5
_PREFILL_BLOCK_SIZES = configs.BlockSizes(
    bq_sz=128, bq_c_sz=64, bkv_sz=256, batch_size=2
)
_LAYOUTS = (
    configs.KVLayout.SEQ_ALONG_LANE,
    configs.KVLayout.HEAD_ALONG_SUBLANE,
)


@pytest.fixture(autouse=True, scope="module")
def _poison_inside_kernel():
    # The hook is read at trace time and the wrapper's jit cache does not key on
    # it, so clear the caches on both sides to keep it out of other test modules.
    jax.clear_caches()
    kernel.POISON_UNUSED_VMEM_FOR_TESTS = True
    yield
    kernel.POISON_UNUSED_VMEM_FOR_TESTS = False
    jax.clear_caches()


def _cdiv(a, b):
    return (a + b - 1) // b


def _vmem_capacity_bytes() -> int:
    return pltpu.get_tpu_info().vmem_capacity_bytes


@functools.partial(
    jax.jit,
    static_argnames=(
        "kv_layout",
        "decode_query_size",
        "decode_block_sizes",
        "attention_scope",
        "skip_kv_update",
    ),
)
def _ragged_paged_attention(
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    kv_layout,
    decode_query_size,
    decode_block_sizes,
    attention_scope,
    skip_kv_update,
):
    """Runs the kernel; the outer jit keeps the caller's kv_cache alive."""
    return wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=_SM_SCALE,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=_PREFILL_BLOCK_SIZES,
        vmem_limit_bytes=_vmem_capacity_bytes(),
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        attention_scope=attention_scope,
        skip_kv_update=skip_kv_update,
    )


def _dense_to_native(dense: jax.Array, kv_layout: configs.KVLayout):
    """[pages, page_size, 2 * kv_heads, head_dim] -> the kernel's cache layout."""
    num_pages, page_size, kv_x2, head_dim = dense.shape
    packing = utils.get_dtype_packing(dense.dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        return dense.reshape(
            num_pages, page_size, kv_x2, head_dim // packing, packing
        ).transpose(0, 2, 3, 4, 1)
    kv_x2_aligned = utils.align_to(kv_x2, packing)
    padded = jnp.pad(dense, ((0, 0), (0, 0), (0, kv_x2_aligned - kv_x2), (0, 0)))
    return padded.reshape(
        num_pages, page_size, kv_x2_aligned // packing, packing, head_dim
    )


def _native_to_dense(
    native: jax.Array, kv_layout: configs.KVLayout, kv_x2: int
) -> jax.Array:
    """Inverse of `_dense_to_native` (drops the HEAD layout's padding heads)."""
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_pages, _, head_dim_packed, packing, page_size = native.shape
        return native.transpose(0, 4, 1, 2, 3).reshape(
            num_pages, page_size, kv_x2, head_dim_packed * packing
        )
    num_pages, page_size, heads_packed, packing, head_dim = native.shape
    dense = native.reshape(num_pages, page_size, heads_packed * packing, head_dim)
    return dense[:, :, :kv_x2]


def _bits(x: jax.Array) -> jax.Array:
    """Bit pattern of `x`, for exact comparisons that ignore float semantics."""
    return lax.bitcast_convert_type(x, jnp.dtype(f"uint{8 * x.dtype.itemsize}"))


def _reference(q, k, v, dense, page_table, kv_lens, *, q_len, cache_only):
    """Vectorized decode reference; every sequence has `q_len` new tokens.

    Returns the attention output `[num_seqs * q_len, num_q_heads, head_dim]` and
    the per-sequence gathered KV pages `[num_seqs, pages_per_seq * page_size,
    2 * num_kv_heads, head_dim]` after the new tokens were written into them (or
    untouched when `cache_only`).
    """
    num_seqs, pages_per_seq = page_table.shape
    page_size = dense.shape[1]
    num_q_heads, head_dim = q.shape[1:]
    num_kv_heads = k.shape[1]
    group = num_q_heads // num_kv_heads
    out_dtype = q.dtype
    mask_value = jnp.finfo(out_dtype).min
    total_len = pages_per_seq * page_size

    gathered = dense[page_table].reshape(
        num_seqs, total_len, 2 * num_kv_heads, head_dim
    )
    cache_lens = kv_lens - q_len
    pos = jnp.arange(total_len, dtype=jnp.int32)[None, :]  # [S, L]
    if not cache_only:
        # Interleave K and V per head, matching the cache layout: 2h = K, 2h+1 = V.
        new_kv = jnp.stack([k, v], axis=2).reshape(-1, 2 * num_kv_heads, head_dim)
        new_kv = new_kv[: num_seqs * q_len].reshape(
            num_seqs, q_len, 2 * num_kv_heads, head_dim
        )
        is_new = (pos >= cache_lens[:, None]) & (pos < kv_lens[:, None])
        new_idx = jnp.clip(pos - cache_lens[:, None], 0, q_len - 1)  # [S, L]
        new_at_pos = new_kv[jnp.arange(num_seqs)[:, None], new_idx]  # [S, L, .]
        gathered = jnp.where(is_new[:, :, None, None], new_at_pos, gathered)

    k_all = jnp.repeat(gathered[:, :, 0::2], group, axis=2)  # [S, L, H, D]
    v_all = jnp.repeat(gathered[:, :, 1::2], group, axis=2)
    queries = q[: num_seqs * q_len].reshape(num_seqs, q_len, num_q_heads, head_dim)
    logits = jnp.einsum(
        "sqhd,slhd->shql", queries, k_all, preferred_element_type=jnp.float32
    )
    logits = (logits * _SM_SCALE).astype(out_dtype)
    if cache_only:
        valid = pos[:, None, :] < cache_lens[:, None, None]  # [S, 1, L]
        valid = jnp.broadcast_to(valid, (num_seqs, q_len, total_len))
    else:
        q_pos = cache_lens[:, None] + jnp.arange(q_len, dtype=jnp.int32)[None, :]
        valid = pos[:, None, :] <= q_pos[:, :, None]  # [S, Q, L]
    logits = jnp.where(valid[:, None], logits, mask_value)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(out_dtype)
    out = jnp.einsum(
        "shql,slhd->sqhd", probs, v_all, preferred_element_type=jnp.float32
    ).astype(out_dtype)
    return out.reshape(num_seqs * q_len, num_q_heads, head_dim), gathered


def _max_steps_ub(*, kv_layout, num_seqs, q_len, pages_per_seq, bkv_sz) -> int:
    """`max_steps_ub` (steps per schedule chunk) of the decode kernel."""
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=q_len, bq_c_sz=q_len, bkv_sz=bkv_sz, batch_size=_BATCH_SIZE
        ),
        model=configs.ModelConfigs(
            num_q_heads=_NUM_Q_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            mask_value=float(jnp.finfo(_Q_DTYPE).min),
        ),
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=_PAGE_SIZE,
            total_q_tokens=num_seqs * q_len,
            num_page_indices=num_seqs * pages_per_seq,
            dtype_q=_Q_DTYPE,
            dtype_kv=_KV_DTYPE,
            dtype_out=_Q_DTYPE,
            kv_layout=kv_layout,
            decode_query_size=q_len,
        ),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=_vmem_capacity_bytes(),
    )
    return cfgs.max_steps_ub


def _run_case(
    *,
    kv_layout,
    histories,  # cached tokens per active sequence
    q_len,
    bkv_sz,
    pages_per_seq,
    max_num_seqs,
    expected_steps,
    cache_only=False,
    seed=0,
):
    """Runs one poisoned decode call and checks outputs and the KV cache."""
    num_seqs = len(histories)
    assert num_seqs <= max_num_seqs
    rng = np.random.default_rng(seed)
    max_num_tokens = max_num_seqs * q_len
    num_pages = max_num_seqs * pages_per_seq
    kv_x2 = 2 * _NUM_KV_HEADS

    def normal(shape, dtype):
        return jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * 0.5).astype(
            dtype
        )

    # Every input is finite, including padding tokens and page tails, so any NaN
    # in the results can only have come from VMEM.
    q = normal((max_num_tokens, _NUM_Q_HEADS, _HEAD_DIM), _Q_DTYPE)
    k = normal((max_num_tokens, _NUM_KV_HEADS, _HEAD_DIM), _KV_DTYPE)
    v = normal((max_num_tokens, _NUM_KV_HEADS, _HEAD_DIM), _KV_DTYPE)
    dense = normal((num_pages, _PAGE_SIZE, kv_x2, _HEAD_DIM), _KV_DTYPE)
    page_table = rng.permutation(num_pages).reshape(max_num_seqs, pages_per_seq)
    page_table = jnp.asarray(page_table, dtype=jnp.int32)

    kv_lens = np.zeros((max_num_seqs,), np.int32)
    kv_lens[:num_seqs] = np.asarray(histories, np.int32) + q_len
    assert kv_lens.max(initial=0) <= pages_per_seq * _PAGE_SIZE
    cu_q_lens = np.full((max_num_seqs + 1,), num_seqs * q_len, np.int32)
    cu_q_lens[: num_seqs + 1] = np.arange(num_seqs + 1) * q_len
    distribution = jnp.asarray(np.array([num_seqs] * 3, np.int32))
    kv_lens = jnp.asarray(kv_lens)
    cu_q_lens = jnp.asarray(cu_q_lens)

    # Documents what the schedule looks like for this case; it is a property of
    # the inputs, not an observation of the kernel.
    schedule_lens = [h if cache_only else h + q_len for h in histories]
    tasks = sum(_cdiv(n, bkv_sz) for n in schedule_lens)
    assert _cdiv(tasks, _BATCH_SIZE) == expected_steps

    kv_cache = _dense_to_native(dense, kv_layout)
    assert kv_cache.shape == wrapper.get_kv_cache_shape(
        num_pages, _PAGE_SIZE, _NUM_KV_HEADS, _HEAD_DIM, _KV_DTYPE, kv_layout=kv_layout
    )

    out, updated = _ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_table.reshape(-1),
        cu_q_lens,
        distribution,
        kv_layout=kv_layout,
        decode_query_size=q_len,
        decode_block_sizes=configs.BlockSizes(
            bq_sz=q_len, bq_c_sz=q_len, bkv_sz=bkv_sz, batch_size=_BATCH_SIZE
        ),
        attention_scope=(
            configs.AttentionScope.CACHE_ONLY
            if cache_only
            else configs.AttentionScope.FULL
        ),
        skip_kv_update=cache_only,
    )
    out, updated = jax.block_until_ready((out, updated))
    assert updated.shape == kv_cache.shape

    if num_seqs == 0:
        assert bool(jnp.array_equal(_bits(updated), _bits(kv_cache))), (
            "kv_cache changed although no sequence was scheduled"
        )
        return

    expected_out, expected_pages = _reference(
        q,
        k,
        v,
        dense,
        page_table[:num_seqs],
        kv_lens[:num_seqs],
        q_len=q_len,
        cache_only=cache_only,
    )

    out = out[: num_seqs * q_len].astype(jnp.float32)
    assert bool(jnp.isfinite(out).all()), "NaN/Inf in the attention output"
    np.testing.assert_allclose(
        np.asarray(out),
        np.asarray(expected_out.astype(jnp.float32)),
        atol=0.2,
        rtol=0.2,
    )

    if cache_only:
        # skip_kv_update: the kernel must not write anything into the cache.
        assert bool(jnp.array_equal(_bits(updated), _bits(kv_cache))), (
            "kv_cache changed with skip_kv_update"
        )
        return

    total_len = pages_per_seq * _PAGE_SIZE
    active_lens = kv_lens[:num_seqs]
    got_pages = _native_to_dense(updated, kv_layout, kv_x2)[
        page_table[:num_seqs]
    ].reshape(num_seqs, total_len, kv_x2, _HEAD_DIM)
    pos = jnp.arange(total_len, dtype=jnp.int32)[None, :]
    num_valid_pages = _cdiv(active_lens, _PAGE_SIZE)  # [S]
    # Every page the kernel may have written back must be finite, including the
    # lanes past `kv_len` that the stitch clears.
    in_valid_page = pos < (num_valid_pages * _PAGE_SIZE)[:, None]
    finite = jnp.isfinite(got_pages.astype(jnp.float32))
    assert bool(jnp.where(in_valid_page[:, :, None, None], finite, True).all()), (
        "NaN/Inf in a written-back KV page"
    )
    # Tokens below `kv_len` must match the reference exactly (new tokens are
    # copied, not computed).
    in_seq = pos < active_lens[:, None]
    same = got_pages.astype(jnp.float32) == expected_pages.astype(jnp.float32)
    assert bool(jnp.where(in_seq[:, :, None, None], same, True).all()), (
        "written-back KV pages do not match the reference"
    )
    # Pages no active sequence reads must be bit-identical to the input: idle
    # lanes and inactive sequences must not write anything.
    valid_slot = jnp.arange(pages_per_seq)[None, :] < num_valid_pages[:, None]
    touched = (
        jnp.zeros((num_pages,), jnp.int32)
        .at[page_table[:num_seqs].reshape(-1)]
        .max(valid_slot.reshape(-1).astype(jnp.int32))
    ) > 0
    unchanged = jnp.all(
        (_bits(updated) == _bits(kv_cache)).reshape(num_pages, -1), axis=1
    )
    assert bool(jnp.all(unchanged | touched)), (
        "a KV page outside every active sequence was modified"
    )


@pytest.mark.parametrize("q_len", [1, 2, 3, 4])
@pytest.mark.parametrize("num_seqs", [1, 4, 8])
@pytest.mark.parametrize("kv_layout", _LAYOUTS)
def test_one_step(kv_layout, num_seqs, q_len):
    # One KV block per sequence and <= 8 sequences: a single step, so only input
    # slot 0 is ever read. Histories end mid-page (726, 854, 982), on a page
    # boundary (128), just below one (127, 255, 1019: q >= 2 straddles), and at a
    # single cached token (1). num_seqs 1 and 4 leave 7 and 4 idle lanes.
    histories = [726, 854, 982, 127, 128, 255, 1019, 1]
    _run_case(
        kv_layout=kv_layout,
        histories=histories[:num_seqs],
        q_len=q_len,
        bkv_sz=1152,
        pages_per_seq=8,
        max_num_seqs=_BATCH_SIZE,
        expected_steps=1,
        seed=num_seqs,
    )


@pytest.mark.parametrize("num_steps", [2, 3])
@pytest.mark.parametrize("kv_layout", _LAYOUTS)
def test_multi_step(kv_layout, num_steps):
    # Sequences spanning 2 (bkv 512) or 3 (bkv 384) KV blocks, 8 sequences: 15 or
    # 23 tasks, so 2 or 3 steps and every input slot is read. The first sequence
    # is one block shorter so that a later sequence's blocks fall on the last lane
    # of one step and the first lane of the next, exercising the m/l/acc carry
    # through the scratches across steps.
    if num_steps == 2:
        q_len, bkv_sz = 2, 512
        histories = [300, 726, 854, 982, 639, 640, 767, 1019]
    else:
        q_len, bkv_sz = 4, 384
        histories = [600, 900, 1000, 1100, 769, 1019, 1147, 850]
    _run_case(
        kv_layout=kv_layout,
        histories=histories,
        q_len=q_len,
        bkv_sz=bkv_sz,
        pages_per_seq=9,
        max_num_seqs=_BATCH_SIZE,
        expected_steps=num_steps,
        seed=num_steps,
    )


@pytest.mark.parametrize("kv_layout", _LAYOUTS)
def test_multi_chunk(kv_layout):
    # More steps than fit in one SMEM schedule chunk (`max_steps_ub`), so the
    # pipeline restarts from slot 0 on slots that earlier steps already used.
    # bkv 2304 (SEQ) / 1792 (HEAD) at page 128 is the E2E decode geometry and
    # keeps `max_steps_ub` small.
    q_len = 1
    pages_per_seq = 1
    bkv_sz = 2304 if kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 1792
    steps_ub = _max_steps_ub(
        kv_layout=kv_layout,
        num_seqs=_BATCH_SIZE,
        q_len=q_len,
        pages_per_seq=pages_per_seq,
        bkv_sz=bkv_sz,
    )
    num_seqs = _BATCH_SIZE * (steps_ub + 1)
    # More sequences only shrink the per-chunk step budget, so this stays >= 2
    # chunks.
    steps_ub = _max_steps_ub(
        kv_layout=kv_layout,
        num_seqs=num_seqs,
        q_len=q_len,
        pages_per_seq=pages_per_seq,
        bkv_sz=bkv_sz,
    )
    num_steps = num_seqs // _BATCH_SIZE
    assert _cdiv(num_steps, steps_ub) >= 2

    rng = np.random.default_rng(42)
    histories = rng.integers(1, _PAGE_SIZE - q_len, size=num_seqs)
    histories[::97] = _PAGE_SIZE - q_len  # fills the page exactly
    histories[1::89] = 1
    _run_case(
        kv_layout=kv_layout,
        histories=histories.tolist(),
        q_len=q_len,
        bkv_sz=bkv_sz,
        pages_per_seq=pages_per_seq,
        max_num_seqs=num_seqs,
        expected_steps=num_steps,
        seed=3,
    )


@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("kv_layout", _LAYOUTS)
def test_pure_cache(kv_layout, q_len):
    # CACHE_ONLY with skip_kv_update: no new tokens are stitched, so nothing
    # overwrites the lanes between `kv_len` and the end of the KV block; they must
    # come out of the zero-fill. The cache must not be written at all.
    histories = [726, 854, 982, 127, 128, 255, 1019, 1]
    _run_case(
        kv_layout=kv_layout,
        histories=histories,
        q_len=q_len,
        bkv_sz=1152,
        pages_per_seq=8,
        max_num_seqs=_BATCH_SIZE,
        expected_steps=1,
        cache_only=True,
        seed=5,
    )


def test_zero_steps():
    # No sequence scheduled: both kernels take the `actual_steps == 0` exit and
    # the cache must be untouched.
    _run_case(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        histories=[],
        q_len=1,
        bkv_sz=1152,
        pages_per_seq=8,
        max_num_seqs=_BATCH_SIZE,
        expected_steps=0,
        seed=7,
    )
