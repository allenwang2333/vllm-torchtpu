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
"""Indexer K cache layouts: same bytes, and which one the scatter can consume.

Three spellings of the *same 111,149,056 bytes in the same order*, differing
only in how the token axis is grouped and therefore in the tiling XLA derives:

    HEAD_ALONG_SUBLANE  u8[424, 256, 4, 256]   T(4,128)(4,1)    incumbent
    TOKEN_MAJOR_3D      u8[424, 1024,   256]   T(32,128)(4,1)   non-shipping
    PACKING8            u8[424, 128, 8, 256]   T(8,128)(4,1)    default

The SparseCore `scatter_fusion` that inserts each step's rows requires
`T(8,128)(4,1)`. Only PACKING8 already has it. TOKEN_MAJOR_3D does not, so XLA
wraps the scatter in `copy T(32,128)->T(8,128)` / `copy back` over the whole
buffer -- which is *cheaper* than the incumbent's own round trip at prefill and
much dearer at decode, hence a net regression and hence this file's second job:
keeping TOKEN_MAJOR_3D measurable as the arm that explains PACKING8.

Everything here moves `uint8` bytes and computes nothing, so every check is
**exact byte equality**, never a tolerance and never a statistical eval.

Four things have to hold, and each has its own trap:

1. The write lands in the same place. Traps: index 0 is a fixed point of stride
   bugs, so nothing here touches page 0 alone; and a decode writes one token at
   an arbitrary slot, where `slot % packing != 0` -- the case a page-aligned
   prefill chunk never exercises. The decode pattern covers every residue of
   `slot % 8` *and* of `slot % 4`.
2. The oracle can fail. `test_negative_control_*` runs the same comparison
   against a deliberately wrong page multiplier and requires it to go red.
3. The *kernel* reads the new form. Byte equality of the store is necessary and
   not sufficient -- if any consumer reads the incumbent grouping the retiling
   reappears at the kernel boundary. `test_kernel_*` compares top-k through the
   real kernel.
4. The retiling is gone, not moved. `test_hlo_*` counts tile-changing ops in
   the compiled program, and `test_bytes_accessed_*` cross-checks that count
   against `cost_analysis()['bytes accessed']`, which needs no regex at all.

**Census correctness.** An earlier copy of the regex here matched only
`  %name = ...` and so silently missed every retiling copy, because they are
spelled `ROOT %copy.N = ... copy(...)` inside their own `called_computation`.
`_DEF` therefore carries `(?:ROOT\\s+)?` and `_RELAYOUT_OPS` carries `copy`, and
`test_hlo_token_major_3d_still_relayouts_the_cache` exists to fail loudly if
either is ever dropped again: a census that reports zero everywhere is
indistinguishable from a census that is broken.
"""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout, streamindex_topk)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import \
    PACKING8_KV_PACKING
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import \
    get_kv_cache_shape

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Production geometry, from the DSA indexer on GLM/DeepSeek-V3.2.
PAGES = 424
PAGE_SIZE = 1024
PACKING = 4  # uint8 default packing -- the incumbent's second-minor extent
PACKING8 = PACKING8_KV_PACKING  # == 8
WIDTH = 256
HEAD_DIM = 128  # align_to(128 + 1, 128) == WIDTH
CACHE_BYTES = PAGES * PAGE_SIZE * WIDTH  # 111,149,056

# Small geometry for the kernel comparison: same layout question, fewer pages.
SMALL_PAGES = 16

# The arms. `ref4d` is the incumbent and the oracle for both others.
ARMS = ("ref4d", "tm3d", "p8")
LAYOUT_OF = {
    "ref4d": KVLayout.HEAD_ALONG_SUBLANE,
    "tm3d": KVLayout.TOKEN_MAJOR_3D,
    "p8": KVLayout.PACKING8,
}


def shape_of(arm: str, pages: int = PAGES) -> tuple[int, ...]:
    return get_kv_cache_shape(pages, PAGE_SIZE, HEAD_DIM, LAYOUT_OF[arm])


# --------------------------------------------------------------------------
# The write paths, spelled exactly as `mla_attention_op._indexer_cache_set`
# spells them: rank-3 addresses a row, rank-4 splits the slot.
# --------------------------------------------------------------------------
def scatter(cache, rows, page, slot):
    if cache.ndim == 3:
        return cache.at[page, slot].set(rows)
    packing = cache.shape[2]
    return cache.at[page, slot // packing, slot % packing].set(rows)


def scatter_bad_page_multiplier(cache, rows, page, slot):
    """NEGATIVE CONTROL -- the page stride is `page_size // packing`, not
    `page_size`.

    This is the exact bug a page-0-only test cannot see: at page 0 the wrong
    multiplier and the right one agree. It must make the byte comparison fail.
    """
    flat = cache.reshape(-1, cache.shape[-1])
    packing = 1 if cache.ndim == 3 else cache.shape[2]
    stride = PAGE_SIZE // max(packing, PACKING)
    return flat.at[page * stride + slot].set(rows).reshape(cache.shape)


def token_major(cache: np.ndarray) -> np.ndarray:
    """The `[pages, page_size, width]` reading of any arm's cache.

    Row-major, so `[p, s // k, s % k, :]` and `[p, s, :]` are the same bytes at
    the same offset `(p * page_size + s) * width` for every packing `k`. This
    reshape therefore *defines* the byte equality the layouts must satisfy; it
    moves nothing.
    """
    return cache.reshape(cache.shape[0], -1, cache.shape[-1])


def to_arm(cache_4d: np.ndarray, arm: str) -> np.ndarray:
    """Regroup the incumbent's bytes into `arm`'s shape. A pure reshape."""
    pages = cache_4d.shape[0]
    return cache_4d.reshape(shape_of(arm, pages))


# --------------------------------------------------------------------------
# Token patterns. Every one keeps page 0 out of the picture, or out of it
# alone.
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
    {3, 4} -- never page 0.
    """
    pos = first_position + np.arange(num_tokens, dtype=np.int64)
    page = (pos // PAGE_SIZE).astype(np.int32)
    slot = (pos % PAGE_SIZE).astype(np.int32)
    return page, slot


def decode_tokens():
    """Decode: one token per sequence, each at an arbitrary slot.

    This is the case a contiguous prefill chunk never reaches. The slots cover
    every residue of `slot % 8` -- the whole second-minor axis of the PACKING8
    cache -- and every residue of `slot % 4` for the incumbent, so a wrong
    sublane within the right row-block cannot hide. Pages are spread across the
    cache and include the last one; page 0 appears but never alone.
    """
    page = np.array([7, 41, 41, 200, 423, 0, 118, 3, 63, 289], dtype=np.int32)
    slot = np.array([1, 2, 3, 517, 1023, 5, 4, 6, 8, 15], dtype=np.int32)
    assert set((slot % PACKING8).tolist()) == set(range(PACKING8)), (
        "the decode pattern must cover every PACKING8 sublane")
    assert set((slot % PACKING).tolist()) == set(range(PACKING)), (
        "the decode pattern must cover every incumbent sublane")
    assert len(set(page.tolist())) > 1 and 0 in set(page.tolist())
    return page, slot


# --------------------------------------------------------------------------
# Byte-equality gate
# --------------------------------------------------------------------------
def run_arm(arm: str, page, slot, *, scatter_fn=scatter):
    rows = jnp.asarray(rows_for(np.arange(len(page))))
    cache = jnp.zeros(shape_of(arm), dtype=jnp.uint8)
    out = jax.jit(scatter_fn)(cache, rows, jnp.asarray(page),
                              jnp.asarray(slot))
    return token_major(np.asarray(out))


def assert_byte_identical(reference: np.ndarray, got: np.ndarray, arm: str):
    diff = int(np.count_nonzero(reference != got))
    assert diff == 0, f"{diff} differing bytes between ref4d and {arm}"
    # A pair of all-zero caches would also compare equal; prove the write
    # happened at all.
    assert int(np.count_nonzero(got)) > 0, "nothing was written"


@pytest.mark.parametrize("arm", ["tm3d", "p8"])
def test_prefill_chunk_is_byte_identical(arm):
    """A 1024-token prefill chunk, crossing a page boundary, off page 0."""
    page, slot = prefill_chunk()
    assert len(set(page.tolist())) > 1, "the chunk must cross a page boundary"
    assert 0 not in set(page.tolist()), "prefill arm must avoid page 0"
    assert_byte_identical(run_arm("ref4d", page, slot),
                          run_arm(arm, page, slot), arm)


@pytest.mark.parametrize("arm", ["tm3d", "p8"])
def test_decode_single_slots_are_byte_identical(arm):
    """Decode: single tokens at arbitrary, non-`packing`-aligned slots."""
    page, slot = decode_tokens()
    assert_byte_identical(run_arm("ref4d", page, slot),
                          run_arm(arm, page, slot), arm)


@pytest.mark.parametrize("arm", ["tm3d", "p8"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_negative_control_wrong_page_multiplier_is_red(phase, arm):
    """The oracle must be able to fail.

    Same comparison, same tokens, one wrong constant: the page stride is
    `page_size // packing` instead of `page_size`. If this passes, the green
    results above prove nothing.
    """
    page, slot = prefill_chunk() if phase == "prefill" else decode_tokens()
    reference = run_arm("ref4d", page, slot)
    got = run_arm(arm, page, slot, scatter_fn=scatter_bad_page_multiplier)
    differing = int(np.count_nonzero(reference != got))
    print(f"NEGATIVE CONTROL ({arm}/{phase}): {differing} differing bytes")
    assert differing > 0, (
        "a wrong page multiplier went undetected -- the byte oracle is inert")


def test_get_kv_cache_shape_is_byte_neutral():
    """Every layout must be the same number of bytes, or the allocator's
    region budget silently changes with the flag."""
    shapes = {arm: shape_of(arm) for arm in ARMS}
    assert shapes["ref4d"] == (PAGES, PAGE_SIZE // PACKING, PACKING, WIDTH)
    assert shapes["tm3d"] == (PAGES, PAGE_SIZE, WIDTH)
    assert shapes["p8"] == (PAGES, PAGE_SIZE // PACKING8, PACKING8, WIDTH)
    for arm, shape in shapes.items():
        assert int(np.prod(shape)) == CACHE_BYTES == 111_149_056, (
            f"{arm} {shape} is not byte-neutral")


def test_packing8_page_size_guard():
    """The `% packing` guard has to move with the packing, or a page size that
    is a multiple of 4 but not of 8 would produce a truncated cache."""
    with pytest.raises(ValueError, match="multiple of 8"):
        get_kv_cache_shape(PAGES, 12, HEAD_DIM, KVLayout.PACKING8)
    # The same page size is legal for the incumbent -- so the guard is really
    # per-layout, not a constant that happens to be stricter.
    assert get_kv_cache_shape(PAGES, 12, HEAD_DIM,
                              KVLayout.HEAD_ALONG_SUBLANE)[1:3] == (3, 4)


def test_env_default_is_packing8():
    """`TPU_DSA_INDEXER_KV_LAYOUT` must default to the shipping arm.

    The env var reaches no Ray worker on the 2-node recipe, so the A/B arm has
    to live in the image's default. The `if TYPE_CHECKING` stub has no runtime
    effect, but a stub that contradicts the live default is a trap.
    """
    import inspect

    from vllm_torchtpu import envs
    assert KVLayout.parse(envs.TPU_DSA_INDEXER_KV_LAYOUT) is KVLayout.PACKING8
    source = inspect.getsource(envs)
    assert 'TPU_DSA_INDEXER_KV_LAYOUT: str = "packing8"' in source, (
        "the TYPE_CHECKING stub disagrees with the live default")


# --------------------------------------------------------------------------
# The kernel has to read the new form
# --------------------------------------------------------------------------
def _kernel_inputs(num_pages=SMALL_PAGES,
                   num_tokens=8,
                   num_heads=4,
                   pages_per_seq=4,
                   seq_len=4096,
                   rng_seed=0,
                   arms=ARMS):
    """Kernel operands. `arms=()` skips the cache -- a 111 MB array per arm is
    not worth materializing when the caller only needs to lower a program."""
    rng = np.random.default_rng(rng_seed)
    q = jnp.asarray(rng.standard_normal((num_tokens, num_heads, HEAD_DIM)),
                    dtype=jnp.float32)
    weights = jnp.asarray(rng.standard_normal((num_tokens, num_heads)),
                          dtype=jnp.float32)
    # A fragmented page table that never starts at 0.
    table = np.array([5, 11, 2, 9], dtype=np.int32)[:pages_per_seq]
    cache_by_arm = {}
    if arms:
        cache_bytes = rng.integers(1,
                                   255,
                                   size=shape_of("ref4d", num_pages),
                                   dtype=np.uint8)
        cache_by_arm = {
            arm: jnp.asarray(to_arm(cache_bytes, arm))
            for arm in arms
        }
    return dict(
        cache_by_arm=cache_by_arm,
        q=q,
        weights=weights,
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


@pytest.mark.parametrize("arm", ["tm3d", "p8"])
def test_kernel_reads_cache_identically(arm):
    """The consumer, not just the writer.

    Byte equality of the stored cache is necessary and not sufficient: the win
    only survives if the *kernel* reads the new grouping. Same bytes, two
    layouts, one kernel -- the selected indices must match exactly.
    """
    inp = _kernel_inputs()
    got_ref = np.asarray(
        _run_topk(inp["cache_by_arm"]["ref4d"], KVLayout.HEAD_ALONG_SUBLANE,
                  inp))
    got_arm = np.asarray(
        _run_topk(inp["cache_by_arm"][arm], LAYOUT_OF[arm], inp))
    mismatch = int(np.count_nonzero(got_ref != got_arm))
    assert mismatch == 0, (
        f"{mismatch} of {got_ref.size} top-k indices differ, ref4d vs {arm}")
    assert int(
        np.count_nonzero(got_ref >= 0)) > 0, "the kernel selected nothing"


def test_kernel_rejects_a_mismatched_packing8_cache():
    """A PACKING8-configured kernel handed the incumbent's shape must refuse.

    Both shapes are rank-4 and both are the right number of bytes, so nothing
    downstream would crash -- it would score the right bytes at the wrong
    strides. This is the allocator/kernel disagreement the guard exists for.
    """
    inp = _kernel_inputs()
    with pytest.raises(ValueError, match="second-minor extent"):
        _run_topk(inp["cache_by_arm"]["ref4d"], KVLayout.PACKING8, inp)


# --------------------------------------------------------------------------
# Relayout census in the compiled program, and its byte-count second opinion
# --------------------------------------------------------------------------
# NOTE the `(?:ROOT\s+)?`. Every retiling copy XLA emits around the SparseCore
# scatter is the ROOT of its own `called_computation`, so a pattern anchored on
# a leading `%` reports a confident, self-consistent zero for every arm.
_DEF = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(\S+)\s+([\w\-]+)\(")
_TILE = re.compile(r"T\((\d+)(?:,(\d+))?\)(?:\((\d+),(\d+)\))?")
_ARG = re.compile(r"%([\w.\-]+)")
# `copy` is the one that matters: a relayout is emitted as a copy, not as a
# reshape. Dropping it is the other half of the false-zero bug.
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

    Scheduled HLO prints `copy(%c.1)` *without* the operand's shape, so a naive
    regex over `copy` flags every one of them and is worthless. Build a
    name -> shape map over the `%name = SHAPE opcode(` definitions first, then
    resolve each operand through it and compare tiles.
    """
    shape_of_name = {}
    for line in hlo.splitlines():
        m = _DEF.match(line)
        if m:
            shape_of_name[m.group(1)] = m.group(2)
    found = []
    for line in hlo.splitlines():
        m = _DEF.match(line)
        if not m or m.group(3) not in _RELAYOUT_OPS:
            continue
        name, result, op = m.group(1), m.group(2), m.group(3)
        arg = _ARG.search(line[m.end() - 1:])
        operand = shape_of_name.get(arg.group(1), "?") if arg else "?"
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
    boundary -- which is exactly what a store in one grouping with a reader in
    another does.

    Returning the cache alongside the indices is not cosmetic: it is what
    `_streamidx_topk_jax` returns, and it is what makes the donation usable. A
    version that returns only the indices has no output the 111 MB input can
    alias, so XLA refuses the donation ("donated buffers were not usable"),
    writes into a fresh buffer, and is free to pick a layout that never needs
    the relayout -- which reports zero for *every* arm and proves nothing.
    """
    cache = scatter(cache, rows, page, slot)
    return cache, _run_topk(cache, kv_layout, inp, k=k)


_COMPILED: dict = {}


def _compiled(arm, phase, full=True, num_pages=PAGES):
    """Compile one arm/phase once; cache it, these are 111 MB programs."""
    key = (arm, phase, full, num_pages)
    if key in _COMPILED:
        return _COMPILED[key]
    kv_layout = LAYOUT_OF[arm]
    inp = _kernel_inputs(num_pages=num_pages, arms=())
    # Lowering needs the shape, not the bytes.
    cache = jax.ShapeDtypeStruct(shape_of(arm, num_pages), jnp.uint8)
    page, slot = prefill_chunk() if phase == "prefill" else decode_tokens()
    keep = page < num_pages
    page, slot = page[keep], slot[keep]
    rows = jnp.asarray(rows_for(np.arange(len(page))))

    def body(c, r, p, s):
        if full:
            return _write_then_read(c, r, p, s, kv_layout, inp)
        return scatter(c, r, p, s), jnp.int32(0)

    fn = jax.jit(body, donate_argnums=(0, ))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = fn.lower(cache, rows, jnp.asarray(page),
                            jnp.asarray(slot)).compile()
    refused = [w for w in caught if "donated buffers" in str(w.message)]
    assert not refused, (
        "XLA refused the donation, so this is not the production write: "
        f"{[str(w.message) for w in refused]}")
    try:
        ca = compiled.cost_analysis()
        ca = ca[0] if isinstance(ca, list) else ca
        bytes_accessed = float(ca.get("bytes accessed", -1))
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"cost_analysis unavailable for {key}: {exc}")
        bytes_accessed = -1.0
    result = (compiled.as_text(), int(np.prod(shape_of(arm,
                                                       num_pages))), bytes_accessed)
    _COMPILED[key] = result
    return result


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_hlo_census_can_say_yes(phase):
    """The detector has to be able to say yes, at both phases.

    TOKEN_MAJOR_3D carries exactly the retiling PACKING8 exists to delete --
    `T(32,128)(4,1) -> T(8,128)(4,1)` over the whole cache, and back. A run
    where it reports zero means the census is broken (a dropped `ROOT`, a
    dropped `copy`), not that the arm is free, and the PACKING8 zeros below
    would then mean nothing.
    """
    hlo, elems, _ = _compiled("tm3d", phase)
    found = relayouts_on_cache(hlo, elems)
    for line in found:
        print(f"tm3d {phase} relayout:", line)
    assert found, (
        f"TOKEN_MAJOR_3D emitted no cache relayout at {phase} -- the census "
        "is not finding what the profile measured, so the PACKING8 zero is "
        "meaningless")


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_hlo_packing8_emits_no_cache_relayout(phase):
    """PACKING8, with the real kernel attached, must emit none."""
    hlo, elems, _ = _compiled("p8", phase)
    found = relayouts_on_cache(hlo, elems)
    for line in found:
        print(f"p8 {phase} relayout:", line)
    assert not found, (f"PACKING8 still relayouts the cache ({phase}):\n" +
                       "\n".join(found))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_bytes_accessed_agrees_with_the_census(phase):
    """The honest second opinion: a byte count, computed by XLA, no regex.

    A relayouting arm reads and rewrites the whole 111 MB buffer twice; an
    in-place one touches only the rows it writes. The write-only program is
    where this is starkest -- at decode the probe measured 0.04 MB for the
    incumbent against 889.26 MB for TOKEN_MAJOR_3D.
    """
    counts, byts = {}, {}
    for arm in ARMS:
        hlo, elems, ba_full = _compiled(arm, phase, full=True)
        _, _, ba_write = _compiled(arm, phase, full=False)
        counts[arm] = len(relayouts_on_cache(hlo, elems))
        byts[arm] = (ba_write, ba_full)
        print(f"{phase} {arm:<6} relayouts={counts[arm]}  "
              f"write_only={ba_write / 1e6:9.2f} MB  "
              f"whole_op={ba_full / 1e6:9.2f} MB")
    if any(b[0] < 0 for b in byts.values()):
        pytest.skip("cost_analysis() did not report 'bytes accessed'")
    # `ref4d` is deliberately not asserted on: it relayouts at prefill and not
    # at decode, which is the asymmetry the whole change is about. `p8` is the
    # relayout-free arm at *both* phases, so it is the right denominator.
    assert counts["tm3d"] > 0 and counts["p8"] == 0
    assert byts["tm3d"][0] > 4 * byts["p8"][0], (
        "the census says tm3d relayouts and p8 does not, but the byte count "
        f"disagrees; one of the two is lying: {byts}")
    assert byts["p8"][0] <= 2 * byts["ref4d"][0] + 1e6, (
        "the census says p8 is relayout-free but it reads far more bytes than "
        f"the incumbent: {byts}")
