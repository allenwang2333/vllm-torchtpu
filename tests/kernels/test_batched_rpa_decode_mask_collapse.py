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
"""Gate for the collapsed decode mask in the batched RPA kernel.

`flash_attention_qk_softmax` builds its mask from two full-shape index tensors,
`kv_idx_b = iota + processed_kv_len[b]` and `q_idx_b = iota // heads + bq_start
+ processed_q_len[b]`, once per batch lane. When `bq_c_sz == 1` -- which is
every decode dispatch, because the wrapper builds decode blocks with
`fixed_bq_sz=1` -- `q_idx_b` is a scalar, so the whole mask collapses to one
shared iota compared against a per-lane scalar bound.

The collapse is a pure rewrite of a boolean expression, so the requirement is
not "close enough" but **bit-identical output**. `_COLLAPSE_DECODE_MASK` exists
to make that testable: flipping it off forces the general path at an otherwise
identical shape and identical block sizes, so the two can be compared exactly.

Three things this file is built to catch, all of which a random-data allclose
would miss:

1. A bound one column too loose. Positions at and beyond `kv_len` are poisoned
   in the cache, so reading one extra column changes the output far outside any
   tolerance. `test_a_loose_bound_is_detectable` proves the poison has teeth
   rather than assuming it.
2. A padded batch lane going live. `mask_out_steps` writes `s_idx=-1` for
   unused lanes, which lands as `processed_q_len == processed_kv_len ==
   effective_kv_len == 0`. The causal term alone evaluates `0 >= 0` -> true, so
   only the `q < effective_kv_len` term keeps column 0 of a dead lane masked.
   Every case here runs a sequence count that is not a multiple of the lane
   count, so dead lanes always exist and always carry poison.
3. The adds being re-materialized rather than removed. `test_collapse_removes_
   vector_arithmetic` asserts on the jaxpr, so it fails if the rewrite is
   correct but does not actually save anything.

fp8 is covered because fp8 is what ships: this checkpoint serves with
`--kv-cache-dtype=fp8`, and nothing in the repo previously ran the batched
kernel against an fp8 cache at all.
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import flash_attention
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB

HEAD_DIM = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
PAGE_SIZE = 256
SM_SCALE = HEAD_DIM**-0.5

# Planted out-of-range signal. Any cache position at or beyond a sequence's
# `kv_len` holds this; a mask that reads one column too far pulls it into the
# softmax and moves the output by orders of magnitude. 64.0 is exactly
# representable in float8_e4m3fn, so the signal survives the fp8 arm unrounded.
POISON = 64.0

# Lane count of the decode dispatch. Held fixed across cases so that a sequence
# count coprime to it reliably leaves padded lanes in the final step.
BATCH_SIZE = 8


def _blocks(bkv_sz: int) -> KBC.BlockSizes:
    """Decode block sizes. `bq_sz == bq_c_sz == 1` is what the collapse keys on
    and is what `wrapper.calculate_block_sizes` produces for decode."""
    return KBC.BlockSizes(bq_sz=1,
                          bq_c_sz=1,
                          bkv_sz=bkv_sz,
                          batch_size=BATCH_SIZE,
                          n_buffer=3)


# `kv_lens` deliberately avoid being multiples of `bkv_sz` or `PAGE_SIZE` except
# where the case exists to test that alignment. Sequence counts are coprime to
# BATCH_SIZE so padded lanes are always exercised.
CASES = {
    # Single kv block, single page. The smallest thing that can fail.
    "decode_single_block": dict(kv_lens=[100, 255, 3], bkv_sz=256),
    # kv_len spans several kv blocks, so `processed_kv_len` is non-zero on later
    # steps -- the term the collapse folds into the bound.
    "decode_multi_block": dict(kv_lens=[600, 1025, 257, 13, 900], bkv_sz=256),
    # Ragged: lanes in one step carry very different bounds.
    "decode_ragged": dict(kv_lens=[1, 2, 511, 512, 513, 1000], bkv_sz=256),
    # Exactly on a kv-block boundary, where an off-by-one in either direction
    # lands on a block edge rather than mid-tile.
    "decode_block_aligned": dict(kv_lens=[512, 256, 768], bkv_sz=256),
    # Production geometry: bkv_sz 3840 and page 256 are what
    # `calculate_block_sizes` returns for this checkpoint at TP=8, and 8192 is
    # the served context. Kept to 9 sequences to bound test memory; the shape
    # that matters for the mask is (kv block, lane), not the sequence count.
    "decode_production_shape": dict(kv_lens=[8192, 8191, 8000, 4097, 4096,
                                             1, 33, 7000, 8192],
                                    bkv_sz=3840),
}

DTYPES = {"bf16": jnp.bfloat16, "fp8": jnp.float8_e4m3fn}


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}.")


def _build_inputs(kv_lens, bkv_sz, kv_dtype, seed=0):
    """Build a decode call: one query token per sequence, a poisoned cache.

    The cache is filled directly in its packed layout rather than by running a
    prefill. Both arms read the identical cache, so it does not need to
    correspond to any logical K/V -- it needs only to be non-degenerate and to
    carry POISON at every position a correct mask must not read. Position `p` of
    a sequence lives at page `page_indices[seq * pages_per_seq + p //
    PAGE_SIZE]`, offset `p % PAGE_SIZE`, which is true regardless of how heads
    are packed into the trailing axes -- so the poison can be planted without
    depending on the layout's head packing at all.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(kv_lens)
    max_kv = max(kv_lens)
    pages_per_seq = (max_kv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_seqs * pages_per_seq

    cache_shape = KB.get_kv_cache_shape(total_pages, PAGE_SIZE, NUM_KV_HEADS,
                                        HEAD_DIM, kv_dtype)
    cache = (rng.standard_normal(cache_shape) * 0.5).astype(np.float32)

    # Each sequence owns a contiguous, disjoint run of pages.
    page_indices = np.arange(total_pages, dtype=np.int32)
    for s, kv_len in enumerate(kv_lens):
        pos = np.arange(kv_len, pages_per_seq * PAGE_SIZE)
        if pos.size:
            cache[page_indices[s * pages_per_seq + pos // PAGE_SIZE],
                  pos % PAGE_SIZE] = POISON

    q = (rng.standard_normal(
        (num_seqs, NUM_Q_HEADS, HEAD_DIM)) * 0.5).astype(np.float32)
    # One new token per sequence. Its K/V is written into the cache by the
    # kernel at position kv_len-1, overwriting the poison there, so the cache
    # must be poisoned from kv_len onward only -- which is what the loop above
    # does.
    kv = (rng.standard_normal(
        (num_seqs, NUM_KV_HEADS, HEAD_DIM)) * 0.5).astype(np.float32)

    return dict(
        q=q,
        k=kv,
        v=kv,
        cache=cache,
        cache_dtype=kv_dtype,
        kv_lens=np.asarray(kv_lens, np.int32),
        page_indices=page_indices,
        cu_q_lens=np.arange(num_seqs + 1, dtype=np.int32),
        # Cumulative (decode, prefill, mixed): all sequences are decode.
        distribution=np.asarray([num_seqs, num_seqs, num_seqs], np.int32),
        blocks=_blocks(bkv_sz),
    )


def _run(inp, q_dtype=jnp.bfloat16, skip_kv_update=False):
    # K/V must be handed in at the *cache* dtype -- `RpaConfigs.validate_inputs`
    # rejects a mismatch outright. Queries are the exception: the kernel reads
    # `queries.dtype` as `dtype_q` and quantizes them itself, so they stay
    # bf16, which is what the served model passes.
    out, cache = KB.ragged_paged_attention(
        jnp.asarray(inp["q"], q_dtype),
        jnp.asarray(inp["k"], inp["cache_dtype"]),
        jnp.asarray(inp["v"], inp["cache_dtype"]),
        jnp.asarray(inp["cache"], inp["cache_dtype"]),
        jnp.asarray(inp["kv_lens"]),
        jnp.asarray(inp["page_indices"]),
        jnp.asarray(inp["cu_q_lens"]),
        jnp.asarray(inp["distribution"]),
        sm_scale=SM_SCALE,
        decode_block_sizes=inp["blocks"],
        skip_kv_update=skip_kv_update,
    )
    return (np.asarray(out.astype(jnp.float32)),
            np.asarray(cache.astype(jnp.float32)))


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("kv_dtype", list(DTYPES), ids=list(DTYPES))
def test_collapsed_matches_general_bitwise(case, kv_dtype, monkeypatch):
    """The collapsed mask must be a bit-identical rewrite of the general one.

    Exact equality is the right bar here and a tolerance would be the wrong one:
    the two paths differ only in how a boolean mask is computed, feed the same
    `jnp.where` against the same `mask_value`, and therefore drive an identical
    softmax over identical inputs. Any difference at all means the masks differ
    somewhere, and a tolerance would hide exactly the one-column errors this
    file exists to catch.
    """
    _require_tpu()
    spec = CASES[case]
    inp = _build_inputs(spec["kv_lens"], spec["bkv_sz"], DTYPES[kv_dtype])

    monkeypatch.setattr(flash_attention, "_COLLAPSE_DECODE_MASK", True)
    collapsed, collapsed_cache = _run(inp)

    monkeypatch.setattr(flash_attention, "_COLLAPSE_DECODE_MASK", False)
    general, general_cache = _run(inp)

    # Finiteness before comparison, and as its own assertion rather than a flag
    # on the comparison. `assert_allclose` defaults to `equal_nan=True`, so
    # NaNs at matching indices compare equal -- and `array_equal` likewise
    # reports mismatch without saying the values were garbage in both arms.
    # Poisoned reads in this kernel can produce NaN (large values through
    # exp/softmax), so "both arms agree" is only meaningful once both are known
    # finite.
    for name, arr in (("collapsed", collapsed), ("general", general)):
        bad = int((~np.isfinite(arr)).sum())
        assert bad == 0, (f"{name} output has {bad} non-finite element(s) for "
                          f"{case}/{kv_dtype}; an equality check would have "
                          f"called two garbage arrays identical")

    np.testing.assert_array_equal(
        collapsed, general,
        err_msg=f"collapsed != general output for {case}/{kv_dtype}")
    np.testing.assert_array_equal(
        collapsed_cache, general_cache,
        err_msg=f"collapsed != general cache for {case}/{kv_dtype}")


@pytest.mark.parametrize("kv_dtype", list(DTYPES), ids=list(DTYPES))
def test_a_loose_bound_is_detectable(kv_dtype):
    """Positive control: prove the planted poison actually has teeth.

    Without this, every assertion above could pass on a kernel whose mask reads
    past `kv_len`, provided both arms read past it identically -- and a mask
    bug is far more likely to be symmetric than not, since both paths are
    derived from the same expression.

    The control runs the same collapsed kernel against `kv_lens` and
    `kv_lens + 1`, which is exactly the read a bound one column too loose would
    perform, and requires the outputs to differ. Both arms use
    `skip_kv_update=True`: without it the kernel writes the new token's K/V at
    position `kv_len`, overwriting the very poison the extra column is supposed
    to pick up, and the control would be measuring the difference between two
    legitimate values instead.
    """
    _require_tpu()
    # kv_lens stay inside their allocated pages when incremented.
    kv_lens = [100, 255, 3, 600]
    inp = _build_inputs(kv_lens, 256, DTYPES[kv_dtype])
    tight, _ = _run(inp, skip_kv_update=True)

    loose = dict(inp)
    loose["kv_lens"] = np.asarray([n + 1 for n in kv_lens], np.int32)
    loose_out, _ = _run(loose, skip_kv_update=True)

    assert np.isfinite(tight).all(), "tight arm is not finite"
    # `loose_out` is allowed to be non-finite -- reading POISON through a
    # softmax may overflow, and that is itself the signal.
    if np.isfinite(loose_out).all():
        delta = np.abs(loose_out - tight).max()
    else:
        delta = np.inf
    assert delta > 1e-2, (
        f"reading one extra kv column changed the output by only {delta}; the "
        f"poison is not reaching the softmax, so the equality tests above "
        f"cannot detect a loose bound")


def _mock_tpu_info():
    mock = MagicMock()
    mock.generation = 7
    mock.chip_version = "v7x"
    mock.num_lanes = 128
    mock.mxu_column_size = 256
    mock.vmem_capacity_bytes = 64 * 1024 * 1024
    mock.fp8_ops_per_second = 2
    mock.bf16_ops_per_second = 1
    return mock


def _decode_cfgs(bkv_sz=256):
    model = KBC.ModelConfigs(num_q_heads=NUM_Q_HEADS,
                             num_kv_heads=NUM_KV_HEADS,
                             head_dim=HEAD_DIM,
                             mask_value=float(jnp.finfo(jnp.bfloat16).min),
                             sm_scale=SM_SCALE)
    serve = KBC.ServingConfigs(num_seqs=BATCH_SIZE,
                               page_size=PAGE_SIZE,
                               total_q_tokens=BATCH_SIZE,
                               num_page_indices=BATCH_SIZE,
                               dtype_q=jnp.bfloat16,
                               dtype_kv=jnp.float8_e4m3fn,
                               dtype_out=jnp.float32)
    return KBC.RpaConfigs(block=_blocks(bkv_sz),
                          model=model,
                          serve=serve,
                          mode=KBC.RpaCase.DECODE,
                          vmem_limit_bytes=64 * 1024 * 1024)


def _count_primitives(jaxpr, counts=None):
    """Count primitives, descending into any nested jaxpr in `eqn.params`.

    Walks `params` generically rather than naming the closed-jaxpr keys, and
    recognizes a nested jaxpr by duck-typing on `.eqns` rather than by
    `isinstance` -- `jax.core.Jaxpr` was removed in jax 0.10, and the public
    replacement has moved more than once.
    """
    counts = {} if counts is None else counts
    for eqn in jaxpr.eqns:
        counts[eqn.primitive.name] = counts.get(eqn.primitive.name, 0) + 1
        # Elements produced by results that are not scalars. This is the unit
        # the optimization is denominated in: work that occupies the vector
        # units rather than the scalar core.
        produced = 0
        for var in eqn.outvars:
            shape = getattr(getattr(var, "aval", None), "shape", ())
            if shape:
                produced += int(np.prod(shape))
        counts["vector_elements"] = counts.get("vector_elements",
                                               0) + produced
        if eqn.primitive.name == "iota":
            counts["iota_elements"] = counts.get("iota_elements",
                                                 0) + produced
        for value in eqn.params.values():
            for item in (value if isinstance(value, (list, tuple)) else
                         (value,)):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    _count_primitives(inner, counts)
    return counts


def _trace_mask(cfgs, q, k, m_prev, l_prev, scalars, collapse):
    """Trace `flash_attention_qk_softmax` with the collapse forced on or off."""
    with patch.object(flash_attention, "_COLLAPSE_DECODE_MASK", collapse):
        return jax.make_jaxpr(
            lambda q_, k_, m_, l_: flash_attention.flash_attention_qk_softmax(
                q_,
                k_,
                m_,
                l_,
                processed_q_len=scalars,
                processed_kv_len=scalars,
                effective_kv_len=scalars,
                cfgs=cfgs,
                bq_start=0))(q, k, m_prev, l_prev)


def _mask_trace_inputs(cfgs, s=256):
    batch = cfgs.block.batch_size
    k_heads = NUM_KV_HEADS
    tq = cfgs.bq_c_sz * cfgs.aligned_num_q_heads_per_kv_head
    return dict(
        q=jnp.zeros((batch, k_heads, tq, HEAD_DIM), jnp.bfloat16),
        k=jnp.zeros((batch, k_heads, s, HEAD_DIM), jnp.bfloat16),
        m_prev=jnp.zeros((batch, k_heads, tq, 128), jnp.float32),
        l_prev=jnp.zeros((batch, k_heads, tq, 128), jnp.float32),
        scalars=[jnp.asarray(0, cfgs.serve.int_ty)] * batch,
    )


def test_collapse_removes_vector_arithmetic():
    """The rewrite must actually remove work, not just be equivalent.

    Runs no kernel and needs no TPU: it traces the masking function both ways
    and compares primitive counts. This is the assertion that fails if a future
    edit makes the collapsed path correct but re-materializes the per-lane
    index tensors -- which the bit-exact tests above would happily pass.
    """
    with patch("jax.experimental.pallas.tpu.get_tpu_info",
               return_value=_mock_tpu_info()):
        cfgs = _decode_cfgs()
        t = _mask_trace_inputs(cfgs)
        collapsed = _count_primitives(
            _trace_mask(cfgs, t["q"], t["k"], t["m_prev"], t["l_prev"],
                        t["scalars"], True).jaxpr)
        general = _count_primitives(
            _trace_mask(cfgs, t["q"], t["k"], t["m_prev"], t["l_prev"],
                        t["scalars"], False).jaxpr)

    # Raw op *count* is the wrong metric and would fail here for the right
    # reason: the collapsed path emits more `lt` than the general one (two per
    # lane, `pq < eff` and `kv_iota < bound`, against the general path's one),
    # but one of its two is a comparison of two scalars. The claim is not
    # "fewer operations", it is "the same masking decision made on the scalar
    # core instead of over a tile" -- so the thing to count is elements
    # produced by non-scalar results.
    assert collapsed["vector_elements"] < general["vector_elements"], (
        f"collapsed path does not reduce vector work: "
        f"{collapsed['vector_elements']} vs {general['vector_elements']} "
        f"elements")

    # The iotas are the specific tensors the rewrite exists to stop
    # materializing per lane: one shared kv iota instead of a kv iota and a q
    # iota for every lane.
    assert collapsed["iota_elements"] < general["iota_elements"], (
        f"collapsed path still materializes per-lane index tensors: "
        f"{collapsed['iota_elements']} vs {general['iota_elements']} iota "
        f"elements")

    # Guard the size of the win, so a future edit that keeps the rewrite
    # correct but gives most of the saving back fails here rather than
    # silently. At `bq_c_sz == 1` the general path builds two full-shape index
    # tensors per lane and the collapsed path builds one in total.
    ratio = general["iota_elements"] / collapsed["iota_elements"]
    assert ratio >= 2 * cfgs.block.batch_size, (
        f"expected the collapsed path to cut iota material by at least "
        f"{2 * cfgs.block.batch_size}x (two per-lane iotas -> one shared), "
        f"got {ratio:.1f}x")


@pytest.mark.parametrize("why", ["prefill_bq_c_sz", "sliding_window"])
def test_collapse_is_declined_when_it_does_not_apply(why):
    """Negative control on the gate itself.

    `bq_c_sz > 1` (prefill) and a set `sliding_window` must both fall through
    to the general path: the collapse's premise -- that `q_idx_b` is a scalar
    over the tile -- fails for the first, and the second adds a lower bound the
    collapsed path does not emit. A gate that fired anyway would be wrong in a
    way no decode test could see, because no decode test reaches either config.

    Asserted by tracing rather than by re-reading the gate's own condition:
    with the collapse enabled and disabled the jaxprs must be *identical*,
    which is only true if the gate declined. Restating the boolean would pass
    against any gate, including a broken one.
    """
    with patch("jax.experimental.pallas.tpu.get_tpu_info",
               return_value=_mock_tpu_info()):
        base = _decode_cfgs()
        if why == "prefill_bq_c_sz":
            cfgs = KBC.RpaConfigs(block=KBC.BlockSizes(bq_sz=1280,
                                                       bq_c_sz=40,
                                                       bkv_sz=1280,
                                                       batch_size=2,
                                                       n_buffer=3),
                                  model=base.model,
                                  serve=base.serve,
                                  mode=KBC.RpaCase.PREFILL,
                                  vmem_limit_bytes=base.vmem_limit_bytes)
        else:
            cfgs = KBC.RpaConfigs(block=base.block,
                                  model=KBC.ModelConfigs(
                                      num_q_heads=NUM_Q_HEADS,
                                      num_kv_heads=NUM_KV_HEADS,
                                      head_dim=HEAD_DIM,
                                      mask_value=base.model.mask_value,
                                      sm_scale=SM_SCALE,
                                      sliding_window=512),
                                  serve=base.serve,
                                  mode=KBC.RpaCase.DECODE,
                                  vmem_limit_bytes=base.vmem_limit_bytes)

        t = _mask_trace_inputs(cfgs)
        args = (t["q"], t["k"], t["m_prev"], t["l_prev"], t["scalars"])
        on = _count_primitives(_trace_mask(cfgs, *args, True).jaxpr)
        off = _count_primitives(_trace_mask(cfgs, *args, False).jaxpr)

    assert on == off, (
        f"the collapse fired for {why}, where it does not apply: primitive "
        f"counts differ between enabled and disabled "
        f"({ {k: v for k, v in on.items() if off.get(k) != v} } vs "
        f"{ {k: v for k, v in off.items() if on.get(k) != v} })")
