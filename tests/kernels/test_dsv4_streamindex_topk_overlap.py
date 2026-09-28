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
"""`streamindex_topk(overlap_s2=...)` (TPU_STREAMIDX_OVERLAP_S2) is bit-exact.

The flag only moves work (bit 0: order each chunk's scoring after earlier
chunks' SparseCore stages; bit 1: gather the winners from stage 1's flat
buffer), so flag on must equal flag off element for element, indices and
scores, in the same slot order. Cases are at the call site at the production
shape (1024 tokens in 8 chunks of 128, n=264192, k=2048) plus the shapes that
change the pipeline: 2 and 3 chunks (fewer than the barrier's reach), one
unchunked b=1024 call, a 16-token bucket, decodes and padded rows, and P=1
(the staged path does not apply). Controls show the comparison fails on a
one-element fault injected on the flag-on path and on a one-token input change.
"""

import difflib
import importlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import _align_to
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

# The package re-exports the function under the module's name.
sim = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)

K = 2048
PAGE = 1024
IH, D = 32, 128
NUM_TOKENS = 1024


def _build_cache(pages_per_seq, recent_ctx, hot, seed):
    """fp8 + UE8M0 HEAD_ALONG_SUBLANE cache with shuffled pages (as in the L1 test)."""
    num_pages = _align_to(pages_per_seq, 2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_pages).astype(np.int32)
    x = jax.random.normal(jax.random.PRNGKey(seed), (num_pages * PAGE, D))
    if hot:
        pos = jnp.arange(num_pages * PAGE)
        near = (pos >= recent_ctx - 12000) & (pos < recent_ctx)
        x = jnp.where(near[:, None], x * 4.0, x)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.clip(jnp.max(jnp.abs(x), -1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    quant = (x / scale).astype(jnp.float8_e4m3fn)
    rec = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(quant, jnp.uint8),
            jax.lax.bitcast_convert_type(scale.astype(jnp.float8_e8m0fnu), jnp.uint8),
        ],
        -1,
    )
    width = _align_to(D + D // 128, 128)
    rec = jnp.pad(rec, ((0, 0), (0, width - rec.shape[-1])))
    logical = rec.reshape(num_pages, PAGE // 4, 4, width)
    cache = jnp.zeros_like(logical).at[jnp.asarray(perm)].set(logical)
    return cache, perm[:pages_per_seq]


def _call_kwargs(
    T_list, S_list, pages_per_seq=257, hot=False, seed=0, num_tokens=NUM_TOKENS,
    chunk_tokens=128,
):
    cache, table = _build_cache(pages_per_seq, max(S_list), hot, seed)
    rng = np.random.default_rng(seed + 1)
    num_decodes = 0
    while num_decodes < len(T_list) and T_list[num_decodes] == 1:
        num_decodes += 1
    q = rng.standard_normal((num_tokens, IH, D), dtype=np.float32)
    w = rng.uniform(-1.5, 1.5, size=(num_tokens, IH)).astype(np.float32)
    return dict(
        q=jnp.asarray(q).astype(jnp.float8_e4m3fn),
        indexer_weights=jnp.asarray(w),
        cache_kv=cache,
        seq_lens=jnp.asarray(np.array(S_list, np.int32)),
        page_indices=jnp.asarray(np.tile(table, len(S_list))),
        cu_q_lens=jnp.asarray(np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)),
        distribution=jnp.asarray(
            np.array([num_decodes, num_decodes, len(T_list)], np.int32)
        ),
        k=K,
        compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        chunk_tokens=chunk_tokens,
    )


def _normalized(hlo: str) -> str:
    # Each trace reserves fresh scheduling group ids; renumber in order.
    ids = {}
    return re.sub(
        r"_scheduling_group_id(\W+)(\d+)",
        lambda m: f"_scheduling_group_id{m.group(1)}"
        f"{ids.setdefault(m.group(2), len(ids))}",
        hlo,
    )


def _run(kw, overlap_s2):
    idx, sc = streamindex_topk(**kw, overlap_s2=overlap_s2, return_scores=True)
    return np.asarray(idx), np.asarray(sc)


def _assert_bitwise(on, off, what):
    for name, a, b in (("indices", on[0], off[0]), ("scores", on[1], off[1])):
        assert a.shape == b.shape and a.dtype == b.dtype, (what, name)
        bad = np.argwhere(a != b)
        assert bad.size == 0, (
            f"{what}: {len(bad)} {name} differ, first at {bad[:4].tolist()}"
        )


CALL_CASES = [
    # Production: one long prefill, 8 chunks of 128.
    pytest.param((1024,), (230000,), False, {}, id="prefill-230k"),
    pytest.param((1024,), (230000,), True, {}, id="prefill-230k-hot"),
    pytest.param((1024,), (262144,), False, {}, id="prefill-262k"),
    # Short context at the production n: SC-bound regime, rows mostly empty.
    pytest.param((1024,), (4096,), False, {}, id="prefill-4k"),
    pytest.param((1024,), (29000,), True, {}, id="prefill-29k-hot"),
    # Decodes, long and short prefills, 120 padded (invalid) rows.
    pytest.param(
        (1, 1, 1, 1, 600, 300),
        (250000, 1500, 64, 200000, 230000, 3000),
        True,
        {},
        id="mixed-padded",
    ),
    # 2 chunks: no chunk reaches the barrier (m >= 2).
    pytest.param((256,), (230000,), True, dict(num_tokens=256), id="2-chunks"),
    # 3 chunks: chunk 2 has a stage-1 edge and no stage-2 edge.
    pytest.param((384,), (100000,), True, dict(num_tokens=384), id="3-chunks"),
    # Unchunked b=1024 call: staged (bit 1) applies, bit 0 does not.
    pytest.param((1024,), (230000,), True, dict(chunk_tokens=None), id="unchunked"),
    # A 16-token bucket (unchunked b=16, P=16).
    pytest.param((16,), (230000,), True, dict(num_tokens=16), id="bucket16"),
]


@pytest.mark.parametrize("overlap_s2", [1, 2, 3])
@pytest.mark.parametrize("T_list,S_list,hot,extra", CALL_CASES)
def test_flag_on_is_bitwise_equal_to_off(T_list, S_list, hot, extra, overlap_s2):
    kw = _call_kwargs(T_list, S_list, hot=hot, **extra)
    off_txt = _normalized(streamindex_topk.lower(**kw, return_scores=True).as_text())
    on_txt = _normalized(
        streamindex_topk.lower(**kw, overlap_s2=overlap_s2, return_scores=True).as_text()
    )
    chunked = extra.get("chunk_tokens", 128) is not None and extra.get(
        "num_tokens", NUM_TOKENS
    ) % 128 == 0
    # Otherwise the comparison below is off against off.
    if overlap_s2 & 2 or chunked:
        assert on_txt != off_txt, "flag did not change the program"
    n_chunks = extra.get("num_tokens", NUM_TOKENS) // 128 if chunked else 1
    has_barrier = "optimization_barrier" in on_txt
    assert has_barrier == bool(overlap_s2 & 1 and n_chunks >= 3), (
        f"barrier present={has_barrier} for overlap_s2={overlap_s2}, {n_chunks} chunks"
    )
    off = _run(kw, 0)
    on = _run(kw, overlap_s2)
    _assert_bitwise(on, off, f"overlap_s2={overlap_s2}")
    # Not vacuous: rows actually select something.
    assert (off[0][: sum(T_list)] >= 0).any()


def test_flag_zero_lowers_to_the_default_program():
    kw = _call_kwargs((1024,), (230000,))
    default = streamindex_topk.lower(**kw).as_text()
    zero = streamindex_topk.lower(**kw, overlap_s2=0).as_text()
    a, b = _normalized(default), _normalized(zero)
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))
    assert not diff, "\n".join(diff[:60])


def test_p1_shape_is_untouched():
    """n=10240 (P=1): the staged split does not apply, flag on == default program."""
    kw = _call_kwargs((1024,), (9000,), pages_per_seq=9)
    a = _normalized(streamindex_topk.lower(**kw).as_text())
    b = _normalized(streamindex_topk.lower(**kw, overlap_s2=3).as_text())
    assert a == b
    np.testing.assert_array_equal(
        np.asarray(streamindex_topk(**kw, overlap_s2=3)),
        np.asarray(streamindex_topk(**kw)),
    )


def test_l1_takes_precedence():
    """With sctopk_l1 the flag is ignored, so the L1 program is unchanged."""
    kw = _call_kwargs((1024,), (230000,))
    a = _normalized(streamindex_topk.lower(**kw, sctopk_l1=True).as_text())
    b = _normalized(streamindex_topk.lower(**kw, sctopk_l1=True, overlap_s2=3).as_text())
    assert a == b


# ----------------------------- positive controls --------------------------- #


def test_control_comparison_catches_one_perturbed_index():
    """A single index +1 on the flag-on output makes the check fail."""
    kw = _call_kwargs((1024,), (230000,))
    off = _run(kw, 0)
    on = _run(kw, 3)
    _assert_bitwise(on, off, "control baseline")
    bad = on[0].copy()
    r, c = 517, int(np.argmax(on[0][517] >= 0))
    bad[r, c] += 1
    with pytest.raises(AssertionError, match="1 indices differ"):
        _assert_bitwise((bad, on[1]), off, "perturbed")


def test_control_fault_in_flag_on_path_goes_red(monkeypatch):
    """An off-by-one in the flag-on remap, on device, is caught."""
    kw = _call_kwargs((1024,), (230000,))
    off = _run(kw, 0)
    real = sim.sparsecore_topk_stage2

    def faulty(*a, **k):
        idx, sc, raw = real(*a, **k)
        return idx.at[3, 5].add(1), sc, raw

    monkeypatch.setattr(sim, "sparsecore_topk_stage2", faulty)
    streamindex_topk.clear_cache()
    try:
        on = _run(kw, 3)
    finally:
        monkeypatch.undo()
        streamindex_topk.clear_cache()
    # 8 chunks: the fault lands on row 3 of every chunk.
    with pytest.raises(AssertionError, match="8 indices differ"):
        _assert_bitwise(on, off, "faulty remap")


def test_control_one_token_context_change_goes_red():
    """Same program, context 230000 vs 230001: the outputs must differ."""
    a = _run(_call_kwargs((1024,), (230000,)), 3)
    b = _run(_call_kwargs((1024,), (230001,)), 3)
    with pytest.raises(AssertionError, match="differ"):
        _assert_bitwise(a, b, "one-token shift")
