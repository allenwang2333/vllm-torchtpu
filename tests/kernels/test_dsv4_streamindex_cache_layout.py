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
"""TOKEN_MAJOR_3D indexer K cache: same bytes, no relayout.

`KVLayout.TOKEN_MAJOR_3D` stores the DSA indexer K cache as
`[pages, page_size, width]` instead of
`[pages, page_size // packing, packing, width]`. The two hold the *same bytes in
the same order* -- only the rank differs -- but XLA tiles a rank-3 u8 parameter
`T(32,128)(4,1)` and a rank-4 one `T(4,128)(4,1)`, and the scatter canonicalises
to a flat `[pages * page_size, width]` that only the rank-3 tile survives. So the
rank-4 write pays `relayout -> scatter -> relayout` over the whole cache and the
rank-3 write pays nothing.

This file is the correctness gate for that change. It moves `uint8` bytes and
computes nothing, so every check here is **exact byte equality**, never a
tolerance and never a statistical eval.

Three things have to hold, and each has its own trap:

1. The write lands in the same place. Traps: index 0 is a fixed point of stride
   bugs, so nothing here touches page 0 alone; and a decode writes one token at
   an arbitrary slot, where `slot % packing != 0` -- the case a page-aligned
   prefill chunk never exercises.
2. The oracle can fail. `test_negative_control_*` runs the same comparison
   against a deliberately wrong page multiplier and requires it to go red.
3. The *kernel* reads the rank-3 form. If any consumer still reads the rank-4
   view the relayout reappears at the kernel boundary and the change is worth
   exactly zero, so `test_kernel_*` compares top-k through the real kernel and
   `test_hlo_*` counts tile-changing ops in the compiled program.
"""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout, streamindex_topk)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import \
    get_kv_cache_shape

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Production geometry, from the DSA indexer on GLM/DeepSeek-V3.2:
# u8[424, 1024 // 4, 4, 256] == u8[424, 1024, 256] == 111,149,056 bytes.
PAGES = 424
PAGE_SIZE = 1024
PACKING = 4
WIDTH = 256
HEAD_DIM = 128  # align_to(128 + 1, 128) == WIDTH

# Small geometry for the kernel comparison: same layout question, fewer pages.
SMALL_PAGES = 16


# --------------------------------------------------------------------------
# The two write paths, spelled exactly as `mla_attention_op._streamidx_topk_jax`
# spells them.
# --------------------------------------------------------------------------
def scatter_4d(cache, rows, page, slot):
    """HEAD_ALONG_SUBLANE: the slot splits across the packing axis."""
    return cache.at[page, slot // PACKING, slot % PACKING].set(rows)


def scatter_3d(cache, rows, page, slot):
    """TOKEN_MAJOR_3D: the slot addresses a row directly."""
    return cache.at[page, slot].set(rows)


def scatter_3d_bad_page_multiplier(cache, rows, page, slot):
    """NEGATIVE CONTROL -- the page stride is `page_size // packing`, not
    `page_size`.

    This is the exact bug a page-0-only test cannot see: at page 0 the wrong
    multiplier and the right one agree. It must make the byte comparison fail.
    """
    pages, page_size, width = cache.shape
    flat = cache.reshape(pages * page_size, width)
    flat = flat.at[page * (page_size // PACKING) + slot].set(rows)
    return flat.reshape(pages, page_size, width)


def as_3d(cache_4d: np.ndarray) -> np.ndarray:
    """The rank-3 reading of a rank-4 cache.

    Row-major, so `[p, s // 4, s % 4, :]` and `[p, s, :]` are the same bytes at
    the same offset `(p * page_size + s) * width`. This reshape therefore
    *defines* the byte equality the layouts must satisfy; it moves nothing.
    """
    pages, per_packing, packing, width = cache_4d.shape
    return cache_4d.reshape(pages, per_packing * packing, width)


# --------------------------------------------------------------------------
# Token patterns. Every one of these keeps page 0 out of the picture, or at
# least out of it alone.
# --------------------------------------------------------------------------
def rows_for(token_ids: np.ndarray, width: int = WIDTH) -> np.ndarray:
    """Row `t` is a function of both the token and the lane.

    A row that depended only on the token would survive a lane-stride bug, and
    one that depended only on the lane would survive a token-stride bug.
    """
    lanes = np.arange(width, dtype=np.int64)[None, :]
    tok = token_ids.astype(np.int64)[:, None]
    return ((tok * 31 + lanes * 7 + 1) % 251 + 1).astype(np.uint8)


def prefill_chunk(num_tokens: int = PAGE_SIZE,
                  first_position: int = 3 * 1024 + 256):
    """A prefill chunk: `num_tokens` *consecutive* positions.

    `first_position` is non-zero, not page-aligned and not `packing`-aligned in
    absolute terms, so the chunk straddles a page boundary. Pages touched are
    {3, 4} shifted by the page map below -- never page 0 on its own.
    """
    pos = first_position + np.arange(num_tokens, dtype=np.int64)
    page = (pos // PAGE_SIZE).astype(np.int32)
    slot = (pos % PAGE_SIZE).astype(np.int32)
    return page, slot


def decode_tokens():
    """Decode: one token per sequence, each at an arbitrary slot.

    This is the case a contiguous prefill chunk never reaches -- `slot %
    packing` takes every value, so the rank-4 write hits every lane of the
    packing axis and the rank-3 write has to agree at each. Pages are spread
    across the cache and include the last page; page 0 appears but never alone.
    """
    page = np.array([7, 41, 41, 200, 423, 0, 118, 3], dtype=np.int32)
    slot = np.array([1, 2, 3, 517, 1023, 5, 4, 6], dtype=np.int32)
    assert set((slot % PACKING).tolist()) == {
        0, 1, 2, 3
    }, ("the decode pattern must cover every packing lane")
    return page, slot


# --------------------------------------------------------------------------
# Byte-equality gate
# --------------------------------------------------------------------------
def run_both_layouts(page: np.ndarray, slot: np.ndarray, *, scatter_3d_fn):
    """Write the same tokens through both layouts; return the two caches."""
    rows = rows_for(np.arange(len(page)))
    rows_dev = jnp.asarray(rows)
    page_dev = jnp.asarray(page)
    slot_dev = jnp.asarray(slot)

    cache_4d = jnp.zeros((PAGES, PAGE_SIZE // PACKING, PACKING, WIDTH),
                         dtype=jnp.uint8)
    cache_3d = jnp.zeros((PAGES, PAGE_SIZE, WIDTH), dtype=jnp.uint8)

    out_4d = jax.jit(scatter_4d)(cache_4d, rows_dev, page_dev, slot_dev)
    out_3d = jax.jit(scatter_3d_fn)(cache_3d, rows_dev, page_dev, slot_dev)
    return np.asarray(out_4d), np.asarray(out_3d)


def assert_byte_identical(out_4d: np.ndarray, out_3d: np.ndarray):
    reference = as_3d(out_4d)
    diff = int(np.count_nonzero(reference != out_3d))
    assert diff == 0, f"{diff} differing bytes between the two layouts"
    # A pair of all-zero caches would also compare equal; prove the write
    # happened at all.
    assert int(np.count_nonzero(out_3d)) > 0, "nothing was written"


def test_prefill_chunk_is_byte_identical():
    """A 1024-token prefill chunk, crossing a page boundary, off page 0."""
    page, slot = prefill_chunk()
    assert len(set(page.tolist())) > 1, "the chunk must cross a page boundary"
    assert 0 not in set(page.tolist()), "prefill arm must avoid page 0"
    out_4d, out_3d = run_both_layouts(page, slot, scatter_3d_fn=scatter_3d)
    assert_byte_identical(out_4d, out_3d)


def test_decode_single_slots_are_byte_identical():
    """Decode: single tokens at arbitrary, non-`packing`-aligned slots."""
    page, slot = decode_tokens()
    out_4d, out_3d = run_both_layouts(page, slot, scatter_3d_fn=scatter_3d)
    assert_byte_identical(out_4d, out_3d)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_negative_control_wrong_page_multiplier_is_red(phase):
    """The oracle must be able to fail.

    Same comparison, same tokens, one wrong constant: the page stride is
    `page_size // packing` instead of `page_size`. If this passes, the green
    results above prove nothing.
    """
    page, slot = prefill_chunk() if phase == "prefill" else decode_tokens()
    out_4d, out_3d = run_both_layouts(
        page, slot, scatter_3d_fn=scatter_3d_bad_page_multiplier)
    differing = int(np.count_nonzero(as_3d(out_4d) != out_3d))
    print(f"NEGATIVE CONTROL ({phase}): {differing} differing bytes")
    assert differing > 0, (
        "a wrong page multiplier went undetected -- the byte oracle is inert")


def test_get_kv_cache_shape_is_byte_neutral():
    """The two layouts must be the same number of bytes, or the allocator's
    region budget silently changes with the flag."""
    four_d = get_kv_cache_shape(PAGES, PAGE_SIZE, HEAD_DIM,
                                KVLayout.HEAD_ALONG_SUBLANE)
    three_d = get_kv_cache_shape(PAGES, PAGE_SIZE, HEAD_DIM,
                                 KVLayout.TOKEN_MAJOR_3D)
    assert four_d == (PAGES, PAGE_SIZE // PACKING, PACKING, WIDTH)
    assert three_d == (PAGES, PAGE_SIZE, WIDTH)
    assert int(np.prod(four_d)) == int(np.prod(three_d)) == 111_149_056


# --------------------------------------------------------------------------
# The kernel has to read the rank-3 form
# --------------------------------------------------------------------------
def _kernel_inputs(num_pages=SMALL_PAGES,
                   num_tokens=8,
                   num_heads=4,
                   pages_per_seq=4,
                   seq_len=4096,
                   rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    q = jnp.asarray(rng.standard_normal((num_tokens, num_heads, HEAD_DIM)),
                    dtype=jnp.float32)
    weights = jnp.asarray(rng.standard_normal((num_tokens, num_heads)),
                          dtype=jnp.float32)
    # A fragmented page table that never starts at 0.
    table = np.array([5, 11, 2, 9], dtype=np.int32)[:pages_per_seq]
    cache_bytes = rng.integers(1,
                               255,
                               size=(num_pages, PAGE_SIZE // PACKING, PACKING,
                                     WIDTH),
                               dtype=np.uint8)
    return dict(
        q=q,
        weights=weights,
        cache_4d=jnp.asarray(cache_bytes),
        cache_3d=jnp.asarray(as_3d(cache_bytes)),
        seq_lens=jnp.asarray([seq_len], dtype=jnp.int32),
        page_indices=jnp.asarray(table),
        cu_q_lens=jnp.asarray([0, num_tokens], dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
    )


def _run_topk(cache, kv_layout, inp, k=64):
    return streamindex_topk(
        inp["q"],
        inp["weights"],
        cache,
        inp["seq_lens"],
        inp["page_indices"],
        inp["cu_q_lens"],
        inp["distribution"],
        k=k,
        compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        kv_layout=kv_layout,
    )


def test_kernel_reads_3d_cache_identically():
    """The consumer, not just the writer.

    Byte equality of the stored cache is necessary and not sufficient: the win
    only survives if the *kernel* reads the rank-3 form. Same bytes, two
    layouts, one kernel -- the selected indices must match exactly.
    """
    inp = _kernel_inputs()
    got_4d = np.asarray(
        _run_topk(inp["cache_4d"], KVLayout.HEAD_ALONG_SUBLANE, inp))
    got_3d = np.asarray(
        _run_topk(inp["cache_3d"], KVLayout.TOKEN_MAJOR_3D, inp))
    mismatch = int(np.count_nonzero(got_4d != got_3d))
    assert mismatch == 0, (
        f"{mismatch} of {got_4d.size} top-k indices differ between layouts")
    assert int(
        np.count_nonzero(got_4d >= 0)) > 0, "the kernel selected nothing"


# --------------------------------------------------------------------------
# Relayout count in the compiled program
# --------------------------------------------------------------------------
_DEF = re.compile(r"^\s*%(\S+)\s*=\s*(\S+)\s+(\S+?)\(")
_TILE = re.compile(r"T\((\d+)(?:,(\d+))?\)(?:\((\d+),(\d+)\))?")
_ARG = re.compile(r"%([\w.\-]+)")
_RELAYOUT_OPS = ("reshape", "copy", "transpose", "bitcast")


def _tiles(shape_text):
    return tuple(_TILE.findall(shape_text or ""))


def _elem_count(shape_text):
    m = re.search(r"\[([\d,]+)\]", shape_text or "")
    if not m:
        return 0
    return int(np.prod([int(d) for d in m.group(1).split(",")]))


def relayouts_on_cache(hlo: str, cache_elems: int, all_sizes: bool = False):
    """Tile-changing ops whose operand or result is the whole cache.

    Scheduled HLO prints `reshape(%c.1)` *without* the operand's shape, so a
    naive regex over `reshape` flags every one of them and is worthless. Build a
    name -> shape map over the `%name = SHAPE opcode(` definitions first, then
    resolve each operand through it and compare tiles.
    """
    shape_of = {}
    for line in hlo.splitlines():
        m = _DEF.match(line)
        if m:
            shape_of[m.group(1)] = m.group(2)
    found = []
    for line in hlo.splitlines():
        m = _DEF.match(line)
        if not m or m.group(3) not in _RELAYOUT_OPS:
            continue
        name, result, op = m.group(1), m.group(2), m.group(3)
        arg = _ARG.search(line[m.end() - 1:])
        operand = shape_of.get(arg.group(1), "?") if arg else "?"
        if operand == "?" or _tiles(operand) == _tiles(result):
            continue
        if not all_sizes and cache_elems not in (_elem_count(operand),
                                                 _elem_count(result)):
            continue  # a small operand, not the cache
        found.append(f"{op} {name}: {operand}  ->  {result}")
    return found


def _write_then_read(cache, rows, page, slot, kv_layout, inp, k=64):
    """The production composition: scatter this step's rows, then score.

    Compiling the two together is the point. A scatter compiled on its own can
    show zero relayouts and the kernel can put one straight back at its own
    boundary -- which is exactly what a rank-3 store with a rank-4 reader does.

    Returning the cache alongside the indices is not cosmetic: it is what
    `_streamidx_topk_jax` returns, and it is what makes the donation usable. A
    version that returns only the indices has no output the 111 MB input can
    alias, so XLA refuses the donation ("donated buffers were not usable"),
    writes into a fresh buffer, and is free to pick a layout that never needs
    the relayout -- which reports zero for *both* arms and proves nothing.
    """
    if kv_layout == KVLayout.TOKEN_MAJOR_3D:
        cache = scatter_3d(cache, rows, page, slot)
    else:
        cache = scatter_4d(cache, rows, page, slot)
    return cache, _run_topk(cache, kv_layout, inp, k=k)


def _compiled_hlo(kv_layout, phase="prefill", num_pages=PAGES):
    inp = _kernel_inputs(num_pages=num_pages)
    cache = (inp["cache_3d"]
             if kv_layout == KVLayout.TOKEN_MAJOR_3D else inp["cache_4d"])
    # The measured 13.388 ms relayout is the prefill write: 1024 rows in one
    # scatter. A handful of decode rows is not the same program -- XLA lowers a
    # few-row update without canonicalising the operand to the flat 2-D form,
    # so it shows zero relayouts on *both* layouts and discriminates nothing.
    page, slot = prefill_chunk() if phase == "prefill" else decode_tokens()
    keep = page < num_pages
    page, slot = page[keep], slot[keep]
    rows = jnp.asarray(rows_for(np.arange(len(page))))
    fn = jax.jit(
        lambda c, r, p, s: _write_then_read(c, r, p, s, kv_layout, inp),
        donate_argnums=(0, ))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lowered = fn.lower(cache, rows, jnp.asarray(page), jnp.asarray(slot))
        hlo = lowered.compile().as_text()
    refused = [w for w in caught if "donated buffers" in str(w.message)]
    entry = re.search(r"entry_computation_layout=\S+", hlo)
    print(f"{kv_layout.name}/{phase} entry layout: "
          f"{entry.group(0)[:200] if entry else 'not found'}")
    assert not refused, (
        "XLA refused the donation, so this is not the production write: "
        f"{[str(w.message) for w in refused]}")
    return hlo, int(np.prod(cache.shape))


def test_hlo_incumbent_still_relayouts_the_cache():
    """The detector has to be able to say yes.

    The rank-4 arm is the incumbent and is expected to carry exactly the
    relayout this change exists to delete: `T(4,128)(4,1) -> T(8,128)(4,1)` over
    the whole cache. A run where both arms report zero means the detector is
    broken, not that the incumbent is free.
    """
    hlo, elems = _compiled_hlo(KVLayout.HEAD_ALONG_SUBLANE, "prefill")
    found = relayouts_on_cache(hlo, elems)
    for line in found:
        print("4-D prefill relayout:", line)
    if not found:
        for line in relayouts_on_cache(hlo, elems, all_sizes=True)[:40]:
            print("4-D prefill tile change (any size):", line)
    assert found, (
        "the rank-4 arm emitted no cache relayout -- the detector is not "
        "finding what the profile measured, so the rank-3 zero below is "
        "meaningless")


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_hlo_token_major_3d_emits_no_cache_relayout(phase):
    """The rank-3 arm, with the real kernel attached, must emit none."""
    hlo, elems = _compiled_hlo(KVLayout.TOKEN_MAJOR_3D, phase)
    found = relayouts_on_cache(hlo, elems)
    for line in found:
        print(f"3-D {phase} relayout:", line)
    assert not found, (
        f"TOKEN_MAJOR_3D still relayouts the cache ({phase}):\n" +
        "\n".join(found))
