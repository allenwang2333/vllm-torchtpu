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
"""The fused kpool op: ``write -> score -> top-k -> expand``, composed.

The pooling, packing, expansion and query stages
and 12 already cover every non-Pallas stage on its own, so what is left to
prove here is the **composition**, and that is what these tests do with
``streamindex_topk`` replaced by an exact numpy top-k over the *real* pooled
cache the write stage just produced. The Pallas kernel itself needs a device;
a device run carries it.

Two properties are tested by watching them fail rather than by reading them:

* **Write before score.** :func:`test_the_scorer_sees_this_steps_own_pools`
  asserts the pooled record for a pool completed by *this* step is already in
  the cache the top-k receives -- checked against the cache the op was called
  with, so a reordering that scored the stale cache would fail.
* **The identity regime.** Below ``index_topk + index_kpool - 2`` the emitted
  id set must be exactly ``range(p + 1)``. That is a theorem, not a tolerance,
  and it is the property the whole shipped 2048-token baseline rests on.

And one that is a warning about a test rather than about the code:
:func:`test_the_early_exit_fast_path_never_reads_the_cache` shows *why*
``enable_early_exit`` must be off -- the fast path produces the identity
selection from ``seq_lens`` alone, so with it on the entire identity regime
would be blind to a completely broken write path.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch_tpu",
                    reason="the fused op imports torch_tpu._internal.pallas; "
                    "run the CPU rung with VLLM_TORCHTPU_CPU_STUB=1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu.kernels.glm5_kpool import (  # noqa: E402
    KPOOL_WRITE_DIAG_SIZE, derive_out_width, derive_pending_state_geometry,
    empty_pending_state, read_pooled_records, unpack_fp8_ue8m0)
from vllm_torchtpu.layers.vllm.custom_ops.glm5_next.kpool_indexer import (  # noqa: E402
    KPOOL_ENABLE_EARLY_EXIT, KpoolIndexerOpConfig, assert_topk_buffer_width,
    build_kpool_indexer_op, kpool_indexer_stages, kpool_request_distribution)
from vllm_torchtpu.models.vllm.glm5_next.indexer import (  # noqa: E402
    POOLED_CACHE_KV_PACKING, derive_pooled_indexer_geometry,
    pooled_record_bytes)

from . import kpool_oracle as O  # noqa: E402

# The step-8 "can actually catch a kpool bug" fixture geometry: index_topk 64
# gives select_k = 16 pools against 48 candidate pools at seq_len 192, so two
# thirds of the candidates are genuinely dropped. At the real 2048 the top-k is
# the identity at every testable length and a kpool bug scores green.
INDEX_KPOOL = 4
INDEX_TOPK = 64
INDEX_HEAD_DIM = 128
N_HEAD = 4
BLOCK_SIZE = 16

PARAMS = O.KpoolParams(index_kpool=INDEX_KPOOL,
                       index_topk=INDEX_TOPK,
                       index_head_dim=INDEX_HEAD_DIM,
                       index_n_heads=N_HEAD)


def _geometry(max_model_len: int):
    return derive_pooled_indexer_geometry(block_size=BLOCK_SIZE,
                                          index_kpool=INDEX_KPOOL,
                                          index_head_dim=INDEX_HEAD_DIM,
                                          max_model_len=max_model_len)


def _config(max_model_len: int) -> KpoolIndexerOpConfig:
    geo = _geometry(max_model_len)
    return KpoolIndexerOpConfig(
        index_kpool=INDEX_KPOOL,
        index_topk=INDEX_TOPK,
        index_head_dim=INDEX_HEAD_DIM,
        num_kv_pages_per_block=geo.num_kv_pages_per_block,
        layer_prefix="model.layers.3.self_attn.indexer",
    )


# ---------------------------------------------------------------------------
# An exact top-k standing in for the Pallas kernel
# ---------------------------------------------------------------------------


class RecordingExactTopk:
    """``streamindex_topk``'s contract, in numpy, plus a record of its inputs.

    Reads the pooled keys **out of the packed cache it is handed**, through the
    same page arithmetic and the same fp8 + e8m0 bitcast the kernel performs
    (``streamindex_topk.py:262-271``). That is what makes this a test of the
    composition: if the write stage put a record on the wrong page, or scored
    before writing, the scorer here sees exactly what the kernel would.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, q, indexer_weights, cache_kv, seq_lens, page_indices,
                 cu_q_lens, distribution, *, k, compression_ratio,
                 causal_row_offset, num_kv_pages_per_block,
                 num_queries_per_block, enable_early_exit):
        self.calls.append(
            dict(
                k=k,
                compression_ratio=compression_ratio,
                causal_row_offset=causal_row_offset,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
                enable_early_exit=enable_early_exit,
                cache_kv=np.asarray(cache_kv).copy(),
            ))
        head_dim = q.shape[-1]
        num_tokens = q.shape[0]
        num_seqs = seq_lens.shape[0]
        pages_per_seq = page_indices.shape[0] // num_seqs
        block_tables = np.asarray(page_indices).reshape(
            num_seqs, pages_per_seq)
        seq_lens = np.asarray(seq_lens)
        cu = np.asarray(cu_q_lens)
        qf = np.asarray(q.astype(jnp.float32)).astype(np.float64)
        w = np.asarray(indexer_weights).astype(np.float64)

        out = np.full((num_tokens, k), -1, np.int32)
        for t in range(num_tokens):
            if t >= cu[-1]:
                continue
            seq = int(np.searchsorted(cu[1:], t, side="right"))
            q_len = int(cu[seq + 1] - cu[seq])
            pos = int(seq_lens[seq]) - q_len + (t - int(cu[seq]))
            n_cand = min(
                int(seq_lens[seq]) // compression_ratio,
                (pos + 1 - causal_row_offset + compression_ratio - 1) //
                compression_ratio)
            if n_cand <= 0:
                continue
            recs = np.asarray(
                read_pooled_records(cache_kv, jnp.arange(n_cand),
                                    jnp.asarray(block_tables[seq])))
            keys = np.asarray(unpack_fp8_ue8m0(jnp.asarray(recs),
                                               head_dim)).astype(np.float64)
            # relu per head, then the per-head weighted sum -- the kernel's
            # arrangement (`streamindex_topk.py:326-338`), which is exact
            # because the folded factors are strictly positive.
            per_head = np.maximum(qf[t] @ keys.T, 0.0)  # [H, n_cand]
            scores = w[t] @ per_head
            take = min(k, n_cand)
            chosen = np.argsort(-scores, kind="stable")[:take]
            out[t, :take] = np.sort(chosen)
        return jnp.asarray(out)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _inputs(seq_len: int, *, seed: int = 0, max_model_len: int | None = None):
    """One full-prefill sequence, one step, everything the op needs."""
    max_model_len = max_model_len or max(seq_len, 192)
    geo = _geometry(max_model_len)
    rng = np.random.default_rng(seed)
    d = INDEX_HEAD_DIM

    k = rng.standard_normal((seq_len, d)).astype(np.float32)
    gate = rng.standard_normal((seq_len, d)).astype(np.float32)
    ape = rng.standard_normal((INDEX_KPOOL, d)).astype(np.float32)
    q = rng.standard_normal((seq_len, N_HEAD, d)).astype(np.float32)
    weights = rng.standard_normal((seq_len, N_HEAD)).astype(np.float32)**2

    # fp8 q with the scale folded into the weights, exactly as the
    # `project_query` produces it.
    amax = np.abs(q).max(axis=-1, keepdims=True)
    q_scale = np.where(amax == 0, 1.0, amax / 448.0).astype(np.float32)
    q_fp8 = jnp.asarray(q / q_scale).astype(jnp.float8_e4m3fn)
    q_bytes = jax.lax.bitcast_convert_type(q_fp8, jnp.uint8).reshape(
        seq_len, N_HEAD, d)
    folded = (weights * q_scale[..., 0] * PARAMS.softmax_scale *
              N_HEAD**-0.5).astype(np.float32)
    # The q the pipeline actually scores with. The oracle must be driven from
    # *this*, not from the pre-quantization `q`: fp8 has 3 mantissa bits, and
    # reshuffling the k/k+1 near-ties is exactly what it does. Comparing
    # selections computed from different q values would measure fp8, not the
    # composition -- and it would do so intermittently, which is worse.
    q_dequant = np.asarray(q_fp8.astype(jnp.float32)).astype(
        np.float64) * q_scale.astype(np.float64)

    num_seqs = 1
    pages_per_seq = geo.pages_per_seq
    block_tables = np.arange(1, 1 + pages_per_seq, dtype=np.int32)[None, :]
    cache = jnp.zeros(
        (2 + pages_per_seq, geo.page_size // POOLED_CACHE_KV_PACKING,
         POOLED_CACHE_KV_PACKING, pooled_record_bytes(d)), jnp.uint8)
    pend = derive_pending_state_geometry(index_kpool=INDEX_KPOOL,
                                         index_head_dim=d,
                                         max_num_reqs=4)
    pending_kv, pending_tag = empty_pending_state(pend)

    return dict(
        q_bytes=q_bytes,
        weights=jnp.asarray(folded),
        pooled_cache=cache,
        pending_kv=pending_kv,
        pending_tag=pending_tag,
        k=jnp.asarray(k),
        gate=jnp.asarray(gate),
        ape=jnp.asarray(ape),
        positions=jnp.arange(seq_len, dtype=jnp.int32),
        seq_lens=jnp.asarray([seq_len], jnp.int32),
        page_indices=jnp.asarray(block_tables.reshape(-1)),
        cu_q_lens=jnp.asarray([0, seq_len], jnp.int32),
        kpool_state_indices=jnp.asarray([1], jnp.int32),
        distribution=jnp.asarray([0, 0, num_seqs], jnp.int32),
    ), dict(k=k,
            gate=gate,
            ape=ape,
            q=q,
            q_dequant=q_dequant,
            weights=weights,
            block_tables=block_tables,
            geo=geo,
            max_model_len=max_model_len)


def _run(seq_len: int, *, seed: int = 0, max_model_len: int | None = None):
    args, raw = _inputs(seq_len, seed=seed, max_model_len=max_model_len)
    cfg = _config(raw["max_model_len"])
    topk = RecordingExactTopk()
    out = kpool_indexer_stages(*args.values(), config=cfg, topk_fn=topk)
    return out, args, raw, cfg, topk


# ---------------------------------------------------------------------------
# 1. Static contract
# ---------------------------------------------------------------------------


def test_the_static_arguments_are_what_the_kernel_gets():
    """Not "we set them" -- what ``streamindex_topk`` is actually handed."""
    (_, _, _, _, _), _, _, cfg, topk = _run(48)
    assert len(topk.calls) == 1
    got = topk.calls[0]
    assert got["enable_early_exit"] is False
    assert got["causal_row_offset"] == INDEX_KPOOL - 1 == 3
    assert got["compression_ratio"] == INDEX_KPOOL
    assert got["k"] == INDEX_TOPK // INDEX_KPOOL == 16
    assert got["num_queries_per_block"] == (1, 128, 128)
    # Derived from block_size // index_kpool, never DSv4's (3, 2, 2).
    assert got["num_kv_pages_per_block"] == cfg.num_kv_pages_per_block
    assert got["num_kv_pages_per_block"] != (3, 2, 2)


def test_enable_early_exit_is_asserted_not_conventional():
    """And it raises ``ValueError``, so ``python -O`` cannot delete it.

    A bare ``assert`` would be stripped under ``-O``, which would silently
    restore ``streamindex_topk``'s fast path -- the one that emits the
    identity selection without reading a single pooled record, i.e. the one
    that makes every test at or below ``index_topk`` tokens pass with
    ``kpool_write`` producing garbage.
    """
    cfg = _config(192)
    assert KPOOL_ENABLE_EARLY_EXIT is False
    import vllm_torchtpu.layers.vllm.custom_ops.glm5_next.kpool_indexer as mod
    mod.KPOOL_ENABLE_EARLY_EXIT = True
    try:
        with pytest.raises(ValueError,
                           match="enable_early_exit must be False"):
            cfg.topk_kwargs()
    finally:
        mod.KPOOL_ENABLE_EARLY_EXIT = False


def test_the_op_never_reads_the_deepseek_early_exit_toggle():
    """``STREAMIDX_ENABLE_EARLY_EXIT`` is a module-level constant another model
    may flip. It must not be reachable from this path.

    Checked by walking the module's **AST**, not by grepping: the docstring
    names the constant on purpose, and a text search would either trip on that
    or be loosened until it stopped catching anything.
    """
    import ast
    import inspect

    import vllm_torchtpu.layers.vllm.custom_ops.glm5_next.kpool_indexer as mod
    tree = ast.parse(inspect.getsource(mod))
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            referenced.update(a.name for a in node.names)
    assert "STREAMIDX_ENABLE_EARLY_EXIT" not in referenced
    assert "KPOOL_ENABLE_EARLY_EXIT" in referenced


def test_out_width_comes_from_the_single_home():
    cfg = _config(192)
    assert cfg.out_width == derive_out_width(INDEX_TOPK, INDEX_KPOOL) == 128
    assert_topk_buffer_width(cfg.out_width, cfg)
    with pytest.raises(ValueError, match="always-selected tail"):
        assert_topk_buffer_width(INDEX_TOPK, cfg)


def test_causal_row_offset_cannot_be_overridden():
    cfg = _config(192)
    assert cfg.causal_row_offset == INDEX_KPOOL - 1
    with pytest.raises((AttributeError, TypeError)):
        cfg.causal_row_offset = 0  # frozen, and a property besides


def test_request_distribution_collapses_j_onto_i():
    rd = jnp.asarray([2, 5, 9], jnp.int32)
    got = np.asarray(kpool_request_distribution(rd))
    assert got.tolist() == [2, 2, 9]


# ---------------------------------------------------------------------------
# 2. The composition
# ---------------------------------------------------------------------------


def _emitted_sets(topk_indices):
    return [O.row_id_set(row) for row in np.asarray(topk_indices)]


@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_identity_regime_selects_exactly_every_causal_token(seq_len):
    """Below ``index_topk + index_kpool - 2`` the union of selected pools and
    the always-selected tail is ``{0, ..., p}`` -- every causal token, nothing
    more and nothing less. This is the theorem the shipped 2048-token baseline
    rests on; at this fixture the bound is 66."""
    assert seq_len - 1 <= PARAMS.identity_bound
    (_, _, _, topk_indices, diag), args, raw, cfg, _ = _run(seq_len)
    assert int(diag[0]) == 0 and int(diag[1]) == 0
    for p, ids in enumerate(_emitted_sets(topk_indices)):
        assert ids == set(range(p + 1)), (p, sorted(ids)[:8], sorted(ids)[-8:])


def test_sparse_regime_matches_the_oracle_and_actually_drops_pools():
    """192 tokens, 48 candidate pools, a 16-pool budget: two thirds dropped.

    The oracle is driven from the pooled keys **as the cache holds them**
    (dequantized fp8), not from float64 keys: this is a test of the pipeline's
    composition, and the fp8 accuracy gate is test_pool_math's.
    """
    seq_len = 192
    (cache, _, _, topk_indices, diag), args, raw, cfg, topk = _run(seq_len)
    assert int(diag[0]) == 0 and int(diag[1]) == 0

    n_pools = seq_len // INDEX_KPOOL
    recs = read_pooled_records(cache, jnp.arange(n_pools),
                               jnp.asarray(raw["block_tables"][0]))
    pool_keys = np.asarray(unpack_fp8_ue8m0(recs,
                                            INDEX_HEAD_DIM)).astype(np.float64)

    # Score with the *reference's* arrangement (relu after softmax_scale, then
    # the per-head weights) from the un-quantized q -- the device path folds
    # q_scale into the weights and applies them after its own relu, and the two
    # agree only because every folded factor is positive. test_kpool_query
    # proves that;
    # here it means the oracle and the pipeline must select the same set.
    scores = O.pool_scores(raw["q_dequant"],
                           pool_keys,
                           raw["weights"].astype(np.float64),
                           softmax_scale=PARAMS.softmax_scale,
                           n_heads=N_HEAD)
    rows = O.select_pool_ids(scores,
                             seq_len=seq_len,
                             index_kpool=INDEX_KPOOL,
                             select_k=PARAMS.select_k)
    expected = O.expand_with_tail(rows,
                                  index_kpool=INDEX_KPOOL,
                                  out_width=cfg.out_width)

    got = np.asarray(topk_indices)
    dropped = 0
    for p in range(seq_len):
        assert O.is_prefix_contiguous(got[p]), p
        want = O.row_id_set(expected[p])
        assert O.row_id_set(
            got[p]) == want, (p, sorted(O.row_id_set(got[p]) ^ want)[:8])
        if len(want) < p + 1:
            dropped += 1
    assert dropped > seq_len // 2, (
        f"only {dropped} of {seq_len} rows dropped a pool; this fixture is "
        "supposed to be genuinely sparse, else it proves nothing")


def test_the_scorer_sees_this_steps_own_pools():
    """Write before score, checked against the cache the scorer received.

    The pooled record for a pool completed by *this* step must already be in
    it. Scoring the pre-write cache would leave those pages byte-zero, which
    scores exactly 0 through the ReLU -- a legal candidate that always ranks
    last, i.e. silent.
    """
    seq_len = 32
    (_, _, _, _, _), args, raw, cfg, topk = _run(seq_len)
    seen = jnp.asarray(topk.calls[0]["cache_kv"])
    n_pools = seq_len // INDEX_KPOOL
    recs = np.asarray(
        read_pooled_records(seen, jnp.arange(n_pools),
                            jnp.asarray(raw["block_tables"][0])))
    assert (recs[:, :INDEX_HEAD_DIM + 1] != 0).any(axis=1).all(), (
        "the scorer was handed a cache without this step's pooled records")
    # And the input cache really was empty, so the assertion above has teeth.
    assert not np.asarray(args["pooled_cache"]).any()


def test_decode_step_after_a_prefill_carries_the_pool_across_the_boundary():
    """Two steps, with the second step's pool completed from ring rows.

    The whole reason the pending state exists: three decode steps in four
    complete a pool from tokens the current step does not contain.
    """
    seq_len = 30
    args, raw = _inputs(seq_len + 2)
    cfg = _config(raw["max_model_len"])
    topk = RecordingExactTopk()

    def step(start, n):
        a = dict(args)
        a["k"] = args["k"][start:start + n]
        a["gate"] = args["gate"][start:start + n]
        a["q_bytes"] = args["q_bytes"][start:start + n]
        a["weights"] = args["weights"][start:start + n]
        a["positions"] = jnp.arange(start, start + n, dtype=jnp.int32)
        a["seq_lens"] = jnp.asarray([start + n], jnp.int32)
        a["cu_q_lens"] = jnp.asarray([0, n], jnp.int32)
        out = kpool_indexer_stages(*a.values(), config=cfg, topk_fn=topk)
        args["pooled_cache"], args["pending_kv"], args["pending_tag"] = out[:3]
        return out

    step(0, 30)  # pools 0..6 complete in-step
    out = step(30, 2)  # pool 7 = {28, 29, 30, 31}: two carried
    diag = out[4]
    assert int(diag[0]) == 0, "a carried row failed its position tag"
    recs = np.asarray(
        read_pooled_records(out[0], jnp.asarray([7]),
                            jnp.asarray(raw["block_tables"][0])))
    got = np.asarray(unpack_fp8_ue8m0(jnp.asarray(recs), INDEX_HEAD_DIM))
    ref = O.pooled_keys(raw["k"][28:32].astype(np.float64),
                        raw["gate"][28:32].astype(np.float64),
                        raw["ape"].astype(np.float64),
                        index_kpool=INDEX_KPOOL)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.07, rel


def test_padded_token_rows_are_all_minus_one():
    """The token axis is padded to a fixed width; padded rows must emit no ids
    at all, or the sparse MLA consumer gathers decoy rows at full weight."""
    seq_len, padded = 16, 24
    args, raw = _inputs(padded)
    cfg = _config(raw["max_model_len"])
    args["seq_lens"] = jnp.asarray([seq_len], jnp.int32)
    args["cu_q_lens"] = jnp.asarray([0, seq_len], jnp.int32)
    args["positions"] = jnp.concatenate([
        jnp.arange(seq_len, dtype=jnp.int32),
        jnp.zeros(padded - seq_len, jnp.int32)
    ])
    out = kpool_indexer_stages(*args.values(),
                               config=cfg,
                               topk_fn=RecordingExactTopk())
    rows = np.asarray(out[3])
    assert (rows[seq_len:] == -1).all()
    for p in range(seq_len):
        assert O.row_id_set(rows[p]) == set(range(p + 1))


def test_profiling_shape_returns_minus_one_without_touching_anything():
    """vLLM traces the model once with no KV cache bound; the op must survive
    it and must not claim to have selected anything."""
    args, raw = _inputs(16)
    cfg = _config(raw["max_model_len"])
    args["pooled_cache"] = jnp.zeros((0, 1, POOLED_CACHE_KV_PACKING, 256),
                                     jnp.uint8)
    topk = RecordingExactTopk()
    out = kpool_indexer_stages(*args.values(), config=cfg, topk_fn=topk)
    assert topk.calls == []
    assert np.asarray(out[3]).shape == (16, cfg.out_width)
    assert (np.asarray(out[3]) == -1).all()


# ---------------------------------------------------------------------------
# 3. Why the fast path must stay off
# ---------------------------------------------------------------------------


def test_the_early_exit_fast_path_never_reads_the_cache():
    """The real reason ``enable_early_exit`` must be False -- and a correction
    correct.

    The fast path was thought to "select pools containing the query's own
    future" at ratio 4. It does not: ``causal_row_offset``
    reaches ``_fast_path_max_valid_idx``, and this test shows the emitted set
    is exactly the identity selection, i.e. **correct**. The hazard is
    different and worse: the function's inputs are ``seq_lens`` and
    ``cu_q_lens`` and *not the cache*, so in the whole
    ``seq_len <= index_topk``
    regime -- every test that exists while the 2048 cap is in place -- the fast
    path would return the right answer with ``kpool_write`` producing garbage.
    """
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        _fast_path_max_valid_idx
    from vllm_torchtpu.kernels.glm5_kpool import expand_pools_with_tail

    seq_len = 48  # <= identity bound 66
    cfg = _config(192)
    seq_lens = jnp.asarray([seq_len], jnp.int32)
    cu_q_lens = jnp.asarray([0, seq_len], jnp.int32)
    max_valid = _fast_path_max_valid_idx(seq_lens, cu_q_lens, seq_len,
                                         INDEX_KPOOL, cfg.causal_row_offset)
    s_idx = jnp.arange(cfg.select_k, dtype=jnp.int32)[None, :]
    pool_ids = jnp.where(s_idx <= max_valid[:, None], s_idx, jnp.int32(-1))
    rows = expand_pools_with_tail(pool_ids,
                                  jnp.arange(seq_len, dtype=jnp.int32),
                                  jnp.ones(seq_len, bool),
                                  index_kpool=INDEX_KPOOL,
                                  out_width=cfg.out_width)
    for p, ids in enumerate(_emitted_sets(rows)):
        assert ids == set(range(p + 1)), p


# ---------------------------------------------------------------------------
# 4. The torch-visible op object
# ---------------------------------------------------------------------------


def test_every_traced_parameter_carries_a_type_annotation():
    """``pallas.jax_op`` derives the torch schema from the annotations and
    refuses a signature with any unannotated parameter.

    This is a regression test for a real, expensive failure: the op used to be
    built from ``functools.partial(kpool_indexer_stages, config=config)``.
    Binding ``config`` by *keyword* leaves it -- and the unannotated
    ``topk_fn`` hook after it -- in the inspected signature, and torch_tpu
    raises ``Missing argument type annotation for JAX function`` at
    ``pallas.py:339``. Nothing on the CPU rung saw it (the tests call
    ``kpool_indexer_stages`` directly) and nothing in a kernel-level device
    probe saw it (same), so it surfaced only at EngineCore startup, on a
    device:

        ValueError: Missing argument type annotation for JAX function:
        (q_bytes: jax.Array, ..., distribution: jax.Array, *,
         config: KpoolIndexerOpConfig = ..., topk_fn=<PjitFunction ...>)

    Two things now stand between that and a repeat: the explicit annotated
    wrapper in ``build_kpool_indexer_op``, and
    ``torch_tpu_cpu_stub._assert_every_parameter_is_annotated``, which makes
    the stub as strict as the real thing. This test pins both -- the second
    directly, by re-doing the mistake and requiring it to be caught.
    """
    import functools
    import inspect

    from torch_tpu._internal import pallas

    from vllm_torchtpu.layers.vllm.custom_ops.glm5_next import kpool_indexer

    cfg = _config(192)

    # (a) The wrapper the op is actually built from: fourteen traced operands,
    #     every one annotated, and NO leftover keyword parameters.
    built = {}
    real = pallas.jax_op

    def _capture(name, fn=None, **kwargs):
        if fn is not None:
            built["fn"] = fn
        return real(name, fn, **kwargs) if fn is not None else real(
            name, **kwargs)

    kpool_indexer._op_cache.clear()
    original = pallas.jax_op
    try:
        pallas.jax_op = _capture
        kpool_indexer.build_kpool_indexer_op(cfg, cfg.out_width)
    finally:
        pallas.jax_op = original
        kpool_indexer._op_cache.clear()

    signature = inspect.signature(built["fn"])
    params = list(signature.parameters.values())
    assert len(params) == 14, [p.name for p in params]
    assert all(p.annotation is not inspect.Parameter.empty
               for p in params), (signature)
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in params), signature
    # And the return, which `_verify_signature` checks separately, with a
    # different message ("The return annotation is invalid. Only jax.Arrays
    # and POD types are supported.") and therefore its own failed startup.
    assert signature.return_annotation is not inspect.Signature.empty, (
        signature)

    # (b) The mistake itself must still be caught, so the guard cannot rot.
    stale = functools.partial(kpool_indexer.kpool_indexer_stages, config=cfg)
    with pytest.raises(ValueError, match="Missing argument type annotation"):
        pallas.jax_op("pallas::glm5_kpool_annotation_regression", stale)


def test_the_op_builds_and_its_fake_matches_the_real_signature():
    """``register_fake`` is what vLLM's tracing sees. A fake whose shapes drift
    from the real op is a compile-time error at best and a wrong buffer at
    worst -- and it is never exercised by an eager CPU run, so it needs its own
    check."""
    import torch

    cfg = _config(192)
    op = build_kpool_indexer_op(cfg, cfg.out_width)
    assert op is build_kpool_indexer_op(cfg, cfg.out_width), "op cache"

    with pytest.raises(ValueError, match="always-selected tail"):
        build_kpool_indexer_op(_config(256), INDEX_TOPK)

    # The stub's jax_op keeps the fake it was given; call it directly.
    geo = _geometry(192)
    fake = None
    for obj in (getattr(op, "__closure__", None) or ()):
        cell = obj.cell_contents
        if hasattr(cell, "_fake"):
            fake = cell._fake
    assert fake is not None, "could not reach the registered fake"
    outs = fake(
        torch.zeros(8, N_HEAD, INDEX_HEAD_DIM, dtype=torch.uint8),
        torch.zeros(8, N_HEAD),
        torch.zeros(geo.cache_shape(4), dtype=torch.uint8),
        torch.zeros(5, 4, 2, INDEX_HEAD_DIM),
        torch.zeros(5, 4, dtype=torch.int32),
    )
    assert outs[3].shape == (8, cfg.out_width)
    assert outs[3].dtype == torch.int32
    assert outs[4].shape == (KPOOL_WRITE_DIAG_SIZE, )


def test_the_op_name_carries_every_field_the_cache_is_keyed_on():
    """A cache key wider than the op name is a name collision waiting.

    ``_op_cache`` used to be keyed on a 6-tuple while ``op_name`` was built
    from three of those fields, so two configs differing only in
    ``num_kv_pages_per_block`` or ``num_queries_per_block`` missed the cache
    and then handed ``pallas.jax_op`` an already-registered name. Keying the
    cache **on the name** makes the two impossible to disagree: a collision is
    now a cache hit by construction. This pins the other half -- that the name
    really does separate configs that must not share an op.
    """
    import dataclasses

    from vllm_torchtpu.layers.vllm.custom_ops.glm5_next.kpool_indexer import (
        _op_cache, _op_name_for)

    base = _config(192)
    variants = {
        "num_kv_pages_per_block": (64, 64, 64),
        "num_queries_per_block": (1, 64, 64),
        "index_head_dim": INDEX_HEAD_DIM * 2,
        "layer_prefix": "model.layers.7.self_attn.indexer",
    }
    names = {_op_name_for(base)}
    for field, value in variants.items():
        other = dataclasses.replace(base, **{field: value})
        name = _op_name_for(other)
        assert name != _op_name_for(base), (
            f"{field} changes the compiled op but not its registered name; "
            "two engine constructions differing only in it would re-register "
            f"{name}")
        names.add(name)
    assert len(names) == len(variants) + 1

    # Equal configs -> one name -> one cache entry, and the second build is a
    # hit rather than a second `pallas.jax_op` registration.
    before = len(_op_cache)
    op = build_kpool_indexer_op(base, base.out_width)
    assert op is build_kpool_indexer_op(base, base.out_width)
    assert _op_name_for(base) in _op_cache
    # A config that differs only in a field the old name ignored must get its
    # own entry.
    other = dataclasses.replace(base, num_kv_pages_per_block=(64, 64, 64))
    assert build_kpool_indexer_op(other, other.out_width) is not op
    assert len(_op_cache) >= before + 1
