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
"""``kpool_write``: the pending ring, the pooling, and the pooled scatter.

Tested against an independently written CPU oracle, over enumerated
schedules of chunked prefill, decode, preemption and speculative rollback.

Three things here are negative controls rather than confirmations, and they are
the reason this file is long:

* **The depth table.** A suite that only shows the chosen ``pending_depth``
  passing cannot distinguish "the depth is right" from "the schedule never
  stressed it". :func:`test_the_depth_table` runs the same schedules at
  ``pending_depth = index_kpool`` -- the naive ring, which is correct at
  ``num_speculative_tokens == 1`` -- and asserts it **fails** at 2 and above,
  plus the silent-wrong column the tag exists to zero.
* **float32.** :func:`test_a_bf16_ring_breaks_the_chunk_invariance` rounds the
  ring to bf16 between steps and asserts the chunked-prefill equality then
  *breaks*, so "keep it f32" is a measured property and not a comment.
* **The ape slot index.** :func:`test_ape_is_indexed_by_slot_within_its_pool`
  permutes ``ape``'s rows and asserts the pooled key moves. Indexing ``ape``
  wrong (or not at all) yields a finite pooled key of the right shape.

A schedule model whose ``k_of(pos)`` is a pure function of the position makes
a rejected draft and its replacement byte-identical, so only the *addressing*
hazard is visible. Here a rejected draft writes junk instead, so a pool
completed from a draft is genuinely wrong until the next step rewrites it.
That rewrite-before-completion property is what makes rollback free, and it is
asserted rather than argued.
"""

from __future__ import annotations

import functools
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import glm5_kpool as kpmod
from vllm_torchtpu.kernels.glm5_kpool import (kpool_write, pool_window,
                                              read_pooled_records)

# Same idiom as ``test_expand.py``: the oracle lives under
# tests/models/vllm/glm5_next and is imported as a package member so that no
# directory named ``vllm`` lands on sys.path.
sys.path.insert(0,
                str(Path(__file__).resolve().parents[2] / "models" / "vllm"))
from glm5_next.kpool_oracle import pooled_keys  # noqa: E402

INDEX_KPOOL = 4
#: Small on purpose for the sweeps -- every hazard here is in the addressing,
#: not in the width. The oracle-parity tests run at the real 128.
HEAD_DIM = 8
BLOCK_SIZE = 16  # GLM's cache_config.block_size; page_size = 4 records/page

# ---------------------------------------------------------------------------
# Content model: one (k, gate) pair per (position, variant)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _row(pos: int, variant: int, head_dim: int, which: int) -> np.ndarray:
    rng = np.random.default_rng((which * 1_000_003 + variant * 7919 + pos)
                                & 0x7FFFFFFF)
    return rng.standard_normal(head_dim).astype(np.float32)


def key_of(pos: int, variant: int = 0, head_dim: int = HEAD_DIM):
    """``LayerNorm(wk(h[pos]))`` stand-in. ``variant != 0`` is a rejected
    draft: a *different* token occupying the same absolute position."""
    return _row(pos, variant, head_dim, 0)


def gate_of(pos: int, variant: int = 0, head_dim: int = HEAD_DIM):
    return _row(pos, variant, head_dim, 1)


@functools.lru_cache(maxsize=None)
def _ape(index_kpool: int, head_dim: int) -> np.ndarray:
    return np.random.default_rng(7).standard_normal(
        (index_kpool, head_dim)).astype(np.float32)


def expected_records(pool_ids, *, index_kpool=INDEX_KPOOL, head_dim=HEAD_DIM):
    """The record bytes a correct writer must produce for committed content.

    Computed through the *same* ``pool_window`` / ``kpmod.pack_fp8_ue8m0`` the writer
    uses, so "correct" is an exact byte equality rather than a tolerance; the
    fp32-tolerance comparison against the numpy oracle is a separate test.
    """
    pool_ids = list(pool_ids)
    if not pool_ids:
        return jnp.zeros((0, head_dim + 1), jnp.uint8)
    k = np.stack([[
        key_of(j * index_kpool + m, 0, head_dim) for m in range(index_kpool)
    ] for j in pool_ids])
    g = np.stack([[
        gate_of(j * index_kpool + m, 0, head_dim) for m in range(index_kpool)
    ] for j in pool_ids])
    pooled = pool_window(jnp.asarray(k), jnp.asarray(g),
                         jnp.asarray(_ape(index_kpool, head_dim)))
    return kpmod.pack_fp8_ue8m0(pooled)


# ---------------------------------------------------------------------------
# The harness: one persistent batch, stepping the real kpool_write
# ---------------------------------------------------------------------------

_JIT_CACHE: dict = {}


def _jitted(index_kpool: int):
    if index_kpool not in _JIT_CACHE:
        _JIT_CACHE[index_kpool] = jax.jit(
            functools.partial(kpool_write, index_kpool=index_kpool))
    return _JIT_CACHE[index_kpool]


@dataclass
class Sim:
    """A fixed-shape padded batch, exactly as the runner presents one.

    The token axis is padded to ``max_tokens`` on every call, which keeps one
    compiled shape for the whole sweep *and* exercises the padded-tail routing
    on every single step rather than in one dedicated test.
    """

    pending_depth: int
    index_kpool: int = INDEX_KPOOL
    head_dim: int = HEAD_DIM
    max_num_reqs: int = 2
    max_tokens: int = 32
    pages_per_seq: int = 32
    num_blocks: int = 96
    block_size: int = BLOCK_SIZE

    cache: jnp.ndarray = field(init=False)
    pending_kv: jnp.ndarray = field(init=False)
    pending_tag: jnp.ndarray = field(init=False)
    tag_violations: int = field(default=0, init=False)
    position_mismatches: int = field(default=0, init=False)
    #: per-step ``(pools completed, tag violations reported)``
    history: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        page_size = self.block_size // self.index_kpool
        kv_packing = kpmod.POOLED_CACHE_KV_PACKING
        assert page_size % kv_packing == 0
        self.cache = jnp.zeros(
            (self.num_blocks, page_size // kv_packing, kv_packing,
             kpmod.pooled_record_bytes(self.head_dim)), jnp.uint8)
        self.pending_kv = jnp.zeros(
            (self.max_num_reqs + 1, self.pending_depth, 2, self.head_dim),
            jnp.float32)
        self.pending_tag = jnp.full(
            (self.max_num_reqs + 1, self.pending_depth),
            kpmod.PENDING_TAG_EMPTY, jnp.int32)
        # One block table row per request; disjoint physical blocks so a
        # cross-request write shows up as a wrong record rather than as a
        # coincidence.
        self.block_tables = np.zeros((self.max_num_reqs, self.pages_per_seq),
                                     np.int32)
        for r in range(self.max_num_reqs):
            base = 1 + r * self.pages_per_seq
            self.block_tables[r] = np.arange(base, base + self.pages_per_seq)

    # -- one scheduler step ------------------------------------------------

    def step(self,
             start: int,
             n_tokens: int,
             *,
             req: int = 0,
             slot: int | None = None,
             accepted: int | None = None,
             draft_variant: int = 0,
             position_offset: int = 0,
             ring_dtype=None):
        """Process absolute positions ``[start, start + n_tokens)``.

        ``accepted`` (default: all) is how many of them carry committed
        content; the rest carry ``draft_variant`` junk, i.e. they are drafts
        this schedule will reject.
        """
        assert n_tokens <= self.max_tokens
        accepted = n_tokens if accepted is None else accepted
        slot = req + 1 if slot is None else slot
        kp, d = self.index_kpool, self.head_dim

        k = np.zeros((self.max_tokens, d), np.float32)
        g = np.zeros((self.max_tokens, d), np.float32)
        for i in range(n_tokens):
            p = start + i
            v = 0 if i < accepted else draft_variant
            k[i] = key_of(p, v, d)
            g[i] = gate_of(p, v, d)

        num_seqs = self.max_num_reqs
        seq_lens = np.zeros(num_seqs, np.int32)
        seq_lens[req] = start + n_tokens
        cu = np.zeros(num_seqs + 1, np.int32)
        cu[req + 1:] = n_tokens
        positions = np.zeros(self.max_tokens, np.int32)
        positions[:n_tokens] = np.arange(start,
                                         start + n_tokens) + position_offset
        slots = np.zeros(num_seqs, np.int32)
        slots[req] = slot

        pending_kv = self.pending_kv
        if ring_dtype is not None:
            # What a `ring_dtype` buffer would have stored between steps.
            pending_kv = pending_kv.astype(ring_dtype).astype(jnp.float32)

        cache, pkv, ptag, diag = _jitted(kp)(
            self.cache, pending_kv, self.pending_tag, jnp.asarray(k),
            jnp.asarray(g), jnp.asarray(_ape(kp, d)), jnp.asarray(positions),
            jnp.asarray(seq_lens), jnp.asarray(cu),
            jnp.asarray(self.block_tables.reshape(-1)), jnp.asarray(slots))
        self.cache, self.pending_kv, self.pending_tag = cache, pkv, ptag
        viol = int(diag[kpmod.DIAG_TAG_VIOLATIONS])
        self.tag_violations += viol
        self.position_mismatches += int(diag[kpmod.DIAG_POSITION_MISMATCHES])
        completed = [
            p // kp for p in range(start, start + n_tokens)
            if (p + 1) % kp == 0
        ]
        self.history.append((completed, viol))
        return diag

    # -- readback ----------------------------------------------------------

    def records(self, pool_ids, *, req: int = 0):
        return read_pooled_records(self.cache,
                                   jnp.asarray(list(pool_ids), jnp.int32),
                                   jnp.asarray(self.block_tables[req]))

    def wrong_pools(self, final_pos: int, *, req: int = 0) -> list[int]:
        """Fully-committed pools whose record is not the committed truth."""
        pools = list(range(final_pos // self.index_kpool))
        if not pools:
            return []
        got = np.asarray(self.records(pools, req=req))[:, :self.head_dim + 1]
        want = np.asarray(
            expected_records(pools,
                             index_kpool=self.index_kpool,
                             head_dim=self.head_dim))
        return [
            j for i, j in enumerate(pools)
            if not np.array_equal(got[i], want[i])
        ]

    def flagged_pools(self) -> set[int]:
        """Pools completed by a step that reported at least one violation."""
        out: set[int] = set()
        for completed, viol in self.history:
            if viol:
                out.update(completed)
        return out


# ---------------------------------------------------------------------------
# Schedule generators
# ---------------------------------------------------------------------------


def prefill_chunks(total: int, chunk: int) -> list[tuple[int, int]]:
    steps, pos = [], 0
    while pos < total:
        steps.append((pos, min(chunk, total - pos)))
        pos += min(chunk, total - pos)
    return steps


def spec_decode_steps(start: int, num_steps: int, k_spec: int,
                      accept_pattern: list[int]) -> list[tuple[int, int, int]]:
    """``(start, n_tokens, accepted)`` per verify step."""
    steps, pos = [], start
    for i in range(num_steps):
        a = accept_pattern[i % len(accept_pattern)]
        steps.append((pos, 1 + k_spec, a))
        pos += a
    return steps


def run_schedule(depth: int,
                 prompt: int,
                 chunk: int,
                 k_spec: int,
                 accept: list[int],
                 num_steps: int = 10) -> tuple[Sim, int]:
    sim = Sim(pending_depth=depth, max_tokens=max(chunk, 1 + k_spec))
    for start, n in prefill_chunks(prompt, chunk):
        sim.step(start, n)
    pos = prompt
    for i, (start, n, a) in enumerate(
            spec_decode_steps(prompt, num_steps, k_spec, accept)):
        sim.step(start, n, accepted=a, draft_variant=i + 1)
        pos = start + a
    return sim, pos


# ---------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------


def test_pending_depth_is_derived_not_chosen():
    for mcb, k_spec in itertools.product((1, 2), (0, 1, 2, 3, 4, 7)):
        geo = kpmod.derive_pending_state_geometry(
            index_kpool=INDEX_KPOOL,
            index_head_dim=128,
            max_num_reqs=64,
            max_concurrent_batches=mcb,
            num_speculative_tokens=k_spec)
        assert geo.max_rollback_tokens == mcb * k_spec
        assert geo.pending_depth == INDEX_KPOOL + mcb * k_spec
        # The measured floor is `k_spec + index_kpool - 1`; the chosen depth
        # clears it by exactly one slot at mcb == 1.
        assert geo.pending_depth >= mcb * k_spec + INDEX_KPOOL - 1
        assert geo.num_slots == 65  # max_num_reqs + 1, slot 0 = sentinel


def test_pending_state_bytes_match_the_decision_table():
    """The table in ``docs/glm5-3-flash-kpool-pending-state.md`` §shape.

    Pinned because the entire argument for staying outside vLLM's KV-cache
    allocator is that these numbers are small *and* flat in ``max_model_len``.
    If the shape drifts, the argument has to be re-made.
    """
    table = {  # (mcb, num_spec): (depth, bytes/req/layer, whole buffer MiB)
        (1, 0): (4, 4112, 3.06),
        (1, 1): (5, 5140, 3.82),
        (1, 2): (6, 6168, 4.59),
        (1, 3): (7, 7196, 5.35),
        (2, 3): (10, 10280, 7.65),
        (1, 7): (11, 11308, 8.41),
    }
    for (mcb, k_spec), (depth, per_req, mib) in table.items():
        geo = kpmod.derive_pending_state_geometry(
            index_kpool=4,
            index_head_dim=128,
            max_num_reqs=64,
            max_concurrent_batches=mcb,
            num_speculative_tokens=k_spec)
        assert geo.pending_depth == depth, geo.describe()
        assert geo.pending_depth * geo.bytes_per_slot_row == per_req
        whole = geo.total_bytes * 12 / 2**20  # 11 DSA layers + MTP layer 45
        assert abs(whole - mib) < 0.01, (geo.describe(), whole)


def test_the_depth_floor_assertion_fires_with_its_numbers():
    """The "just use ``index_kpool``" shortcut, checked against the rule.

    ``kpmod.derive_pending_state_geometry`` satisfies the floor by construction, so
    the assertion inside it can never be *shown* to fire through the public
    derivation. It lives in its own function precisely so this test can pass it
    the depth a shortcut would have chosen, and so a caller can validate an
    externally allocated buffer against the same rule.
    """
    for k_spec in (1, 2, 3, 4, 7):
        floor = INDEX_KPOOL + k_spec  # index_kpool - 1 + k_spec + 1
        kpmod.assert_pending_depth_sufficient(pending_depth=floor,
                                              index_kpool=INDEX_KPOOL,
                                              max_rollback_tokens=k_spec)
        with pytest.raises(AssertionError) as exc:
            kpmod.assert_pending_depth_sufficient(pending_depth=INDEX_KPOOL,
                                                  index_kpool=INDEX_KPOOL,
                                                  max_rollback_tokens=k_spec)
        message = str(exc.value)
        assert "rollback floor" in message
        for number in (f"pending_depth={INDEX_KPOOL}",
                       f"index_kpool={INDEX_KPOOL}",
                       f"max_rollback_tokens={k_spec}"):
            assert number in message, (number, message)


# ---------------------------------------------------------------------------
# 2. The pooling itself, against the step-2 oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [8, 128])
def test_pooled_rows_match_the_oracle_before_quantization(head_dim):
    """Pre-quantization pooled rows match ``get_pooled_states``.

    Checked *before* ``kpmod.pack_fp8_ue8m0`` on purpose: fp8 has ~6% relative
    precision, so a comparison after quantization cannot tell a pooling bug
    from the quantizer.
    """
    kp, L = INDEX_KPOOL, 32
    k = np.stack([key_of(p, 0, head_dim) for p in range(L)])
    g = np.stack([gate_of(p, 0, head_dim) for p in range(L)])
    ape = _ape(kp, head_dim)

    sim = Sim(pending_depth=kp, head_dim=head_dim, max_tokens=L)
    idx = kpmod.pool_write.index_tokens(jnp.asarray([L, 0], jnp.int32),
                                        jnp.asarray([0, L, L], jnp.int32), L)
    pooled, boundary, pool_id, viol = kpmod.kpool_pooled_rows(
        sim.pending_kv,
        sim.pending_tag,
        jnp.asarray(k),
        jnp.asarray(g),
        jnp.asarray(ape),
        idx,
        jnp.asarray([1, 0], jnp.int32),
        jnp.asarray([L, 0], jnp.int32),
        jnp.asarray([0, L, L], jnp.int32),
        index_kpool=kp)

    assert int(viol) == 0
    assert np.array_equal(np.asarray(boundary), (np.arange(L) + 1) % kp == 0)
    ref = pooled_keys(k.astype(np.float64),
                      g.astype(np.float64),
                      ape.astype(np.float64),
                      index_kpool=kp)
    got = np.asarray(pooled)[np.asarray(boundary)]
    assert np.asarray(pool_id)[np.asarray(boundary)].tolist() == list(
        range(L // kp))
    np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-6)


def test_ape_is_indexed_by_slot_within_its_pool():
    """Permuting ``ape``'s rows must move the pooled key.

    The ``ape`` index is ``pos % index_kpool``, which equals the reference's
    ``(idx - first_key) % index_kpool`` only because ``first_key == 0``. If it
    were dropped, broadcast, or indexed by a constant, the pooled key would
    still be finite and correctly shaped -- so the only way to know is to
    perturb the axis and watch.
    """
    kp, d, L = INDEX_KPOOL, HEAD_DIM, 16
    k = jnp.asarray(np.stack([key_of(p) for p in range(L)]))
    g = jnp.asarray(np.stack([gate_of(p) for p in range(L)]))
    ape = _ape(kp, d)
    base = pool_window(k.reshape(-1, kp, d), g.reshape(-1, kp, d),
                       jnp.asarray(ape))
    rolled = pool_window(k.reshape(-1, kp, d), g.reshape(-1, kp, d),
                         jnp.asarray(np.roll(ape, 1, axis=0)))
    assert not np.allclose(np.asarray(base), np.asarray(rolled))


def test_packed_record_round_trips_through_the_kernel_bitcast():
    """``read_pooled_records`` inverts the scatter's page
    arithmetic, and ``kpmod.unpack_fp8_ue8m0`` is the bitcast ``streamindex_topk``
    performs at ``:264-271``."""
    kp, d, L = INDEX_KPOOL, 128, 32
    sim = Sim(pending_depth=kp, head_dim=d, max_tokens=L)
    sim.step(0, L)
    pools = list(range(L // kp))
    rec = np.asarray(sim.records(pools))
    assert rec.shape == (len(pools), kpmod.pooled_record_bytes(d))
    # The scale byte lives at offset `index_head_dim`, and everything past
    # `index_head_dim + 1` is padding the cache page owns, not the record.
    assert np.all(rec[:, d + 1:] == 0)
    got = np.asarray(kpmod.unpack_fp8_ue8m0(jnp.asarray(rec), d))

    k = np.stack([key_of(p, 0, d) for p in range(L)])
    g = np.stack([gate_of(p, 0, d) for p in range(L)])
    ref = pooled_keys(k.astype(np.float64),
                      g.astype(np.float64),
                      _ape(kp, d).astype(np.float64),
                      index_kpool=kp)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    # fp8_e4m3 with a per-row ue8m0 scale: 3 mantissa bits => 2**-4 = 6.25%
    # worst case relative to the row max.
    assert rel < 0.07, rel


# ---------------------------------------------------------------------------
# 3. Chunked prefill: exact equality, and the bf16 negative control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [13, 14, 15, 16, 33])
@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 7, 16])
def test_chunked_prefill_at_every_residue_equals_single_shot(prompt, chunk):
    """Chunked prefill equals one-shot prefill, as an **exact array equality**.

    Both gather sources are f32 and hold the same bits for the same position,
    so where the scheduler cut the chunk cannot change a single byte of the
    pooled cache. A tolerance here would pass with a bf16 ring; see the next
    test.
    """
    single = Sim(pending_depth=INDEX_KPOOL, max_tokens=prompt)
    single.step(0, prompt)

    chunked = Sim(pending_depth=INDEX_KPOOL, max_tokens=max(chunk, prompt))
    for start, n in prefill_chunks(prompt, chunk):
        chunked.step(start, n)

    assert chunked.tag_violations == 0
    assert chunked.position_mismatches == 0
    assert np.array_equal(np.asarray(single.cache), np.asarray(chunked.cache))


def test_a_bf16_ring_breaks_the_chunk_invariance():
    """The f32 ring is load-bearing, measured rather than asserted in a
    comment: with the ring rounded to bf16 between steps, the same prompt
    produces a different pooled cache depending on where it was cut."""
    prompt, chunk = 33, 3
    single = Sim(pending_depth=INDEX_KPOOL, max_tokens=prompt)
    single.step(0, prompt)

    chunked = Sim(pending_depth=INDEX_KPOOL, max_tokens=chunk)
    for start, n in prefill_chunks(prompt, chunk):
        chunked.step(start, n, ring_dtype=jnp.bfloat16)

    assert not np.array_equal(np.asarray(single.cache),
                              np.asarray(chunked.cache))


def test_the_ring_dtype_is_asserted():
    sim = Sim(pending_depth=INDEX_KPOOL, max_tokens=8)
    with pytest.raises(AssertionError, match="pending_kv must be float32"):
        kpool_write(sim.cache,
                    sim.pending_kv.astype(jnp.bfloat16),
                    sim.pending_tag,
                    jnp.zeros((8, HEAD_DIM), jnp.float32),
                    jnp.zeros((8, HEAD_DIM), jnp.float32),
                    jnp.asarray(_ape(INDEX_KPOOL, HEAD_DIM)),
                    jnp.arange(8, dtype=jnp.int32),
                    jnp.asarray([8, 0], jnp.int32),
                    jnp.asarray([0, 8, 8], jnp.int32),
                    jnp.asarray(sim.block_tables.reshape(-1)),
                    jnp.asarray([1, 0], jnp.int32),
                    index_kpool=INDEX_KPOOL)


# ---------------------------------------------------------------------------
# 4. Rollback, and the depth table (the negative control)
# ---------------------------------------------------------------------------


def test_a_rejected_draft_is_healed_by_overwriting_its_position():
    """The idempotence claim, at its smallest.

    Complete a pool from a draft token, then re-process the rejected position
    with the real token, and assert the pooled record equals the record a
    single correct write would have produced.
    """
    sim = Sim(pending_depth=INDEX_KPOOL + 1, max_tokens=4)
    sim.step(0, 3)  # positions 0,1,2 committed
    sim.step(3, 1, accepted=0, draft_variant=9)  # pos 3 is a rejected draft
    wrong = np.asarray(sim.records([0]))[:, :HEAD_DIM + 1]
    want = np.asarray(expected_records([0]))
    assert not np.array_equal(wrong, want), (
        "the draft must actually change the record, else this proves nothing")
    sim.step(3, 1)  # the real token at position 3
    got = np.asarray(sim.records([0]))[:, :HEAD_DIM + 1]
    assert np.array_equal(got, want)
    assert sim.tag_violations == 0


def test_the_plan_2_2_counterexample():
    """§2.2 verbatim: kpool=4, num_spec=2, positions 10/11/12, both rejected.

    At depth ``index_kpool`` position 12 overwrites the ring row holding
    position 8, so pool 2 cannot be rebuilt. At the derived depth 6 it can.
    """
    for depth, expect_ok in ((INDEX_KPOOL, False), (INDEX_KPOOL + 2, True)):
        sim = Sim(pending_depth=depth, max_tokens=4)
        for start, n in prefill_chunks(10, 4):
            sim.step(start, n)
        sim.step(10, 3, accepted=1, draft_variant=1)  # 11, 12 are drafts
        sim.step(11, 3, accepted=1, draft_variant=2)
        sim.step(12, 3, accepted=1, draft_variant=3)
        wrong = sim.wrong_pools(final_pos=13)
        if expect_ok:
            assert wrong == [], (depth, wrong)
            assert sim.tag_violations == 0
        else:
            assert wrong, depth
            assert sim.tag_violations > 0, (
                "the depth-4 failure must be loud, not silent")


_ACCEPT_PATTERNS = {
    k_spec: [[1, 1], [k_spec + 1, k_spec + 1], [1, k_spec + 1],
             [k_spec + 1, 1], [max(1, k_spec), 2]]
    for k_spec in (1, 2, 3, 4, 7)
}


@pytest.mark.parametrize("k_spec", [1, 2, 3, 4, 7])
def test_the_depth_table(k_spec, capsys):
    """The depth table, measured against the real ``kpool_write``.

    Three depths per ``k_spec``: the rejected ring (``index_kpool``), one slot
    short of the derived depth, and the derived depth. The derived depth must
    be clean; ``index_kpool`` must FAIL at ``k_spec >= 2``; and the
    *silent* column -- a wrong pool that no step's tag check flagged -- must be
    zero everywhere, which is the entire justification for spending an int32
    per ring row.
    """
    chosen = INDEX_KPOOL + k_spec
    rows = []
    for depth in (INDEX_KPOOL, chosen - 1, chosen):
        wrong = viol = silent = n = 0
        for prompt in (14, 16):
            for chunk in (3, 4):
                for accept in _ACCEPT_PATTERNS[k_spec]:
                    sim, final = run_schedule(depth, prompt, chunk, k_spec,
                                              accept)
                    bad = set(sim.wrong_pools(final))
                    wrong += len(bad)
                    viol += sim.tag_violations
                    silent += len(bad - sim.flagged_pools())
                    n += 1
        rows.append((depth, n, wrong, viol, silent))
        print(f"  k_spec={k_spec} depth={depth:>2} schedules={n:>3} "
              f"wrong={wrong:>4} tagviol={viol:>5} silent={silent:>3}  "
              f"{'FAIL' if wrong else 'pass'}")

    by_depth = {depth: (n, w, v, s) for depth, n, w, v, s in rows}
    assert by_depth[chosen][1] == 0, (
        f"the derived depth {chosen} must be clean")
    assert by_depth[chosen][2] == 0, "no tag violation at the derived depth"
    if k_spec >= 2:
        assert by_depth[INDEX_KPOOL][1] > 0, (
            f"depth={INDEX_KPOOL} must FAIL at k_spec={k_spec}; a suite that "
            "cannot show the rejected ring failing has not stressed the ring")
    for depth, n, wrong, viol, silent in rows:
        assert silent == 0, (
            f"depth={depth} produced {silent} wrong pooled record(s) that no "
            "tag check flagged -- the tag is supposed to make this impossible")


def test_num_spec_1_is_the_blind_spot():
    """The usual configuration is ``num_speculative_tokens = 1``.

    That is the one value at which a depth-``index_kpool`` ring survives, so
    such a ring would have shipped green. This pins that as a blind spot
    rather than as evidence.
    """
    clean = []
    for k_spec in (1, 2):
        wrong = 0
        for accept in _ACCEPT_PATTERNS[k_spec]:
            sim, final = run_schedule(INDEX_KPOOL, 16, 4, k_spec, accept)
            wrong += len(sim.wrong_pools(final))
        clean.append(wrong)
    assert clean[0] == 0, "k_spec=1 at depth 4 is green -- that is the trap"
    assert clean[1] > 0, "k_spec=2 at depth 4 must be caught"


def test_preemption_re_prefills_from_zero_without_clearing_the_slot():
    """``_preempt_request`` zeroes
    ``num_computed_tokens``, so the first re-prefill step has ``s == 0`` and by
    invariant R reads nothing. Run it with the slot deliberately *not* cleared,
    which is also the slot-reuse-across-requests case."""
    k_spec = 3
    sim = Sim(pending_depth=INDEX_KPOOL + k_spec, max_tokens=8)
    for start, n in prefill_chunks(30, 7):
        sim.step(start, n)
    for i, (start, n, a) in enumerate(
            spec_decode_steps(30, 6, k_spec, [1, k_spec + 1, 2])):
        sim.step(start, n, accepted=a, draft_variant=i + 1)
    # Preempted: same slot, re-prefill from 0. Nothing is cleared.
    for start, n in prefill_chunks(34, 7):
        sim.step(start, n)
    assert sim.wrong_pools(34) == []
    assert sim.tag_violations == 0


def test_slot_and_page_recycling_leaves_the_other_request_alone():
    """Slot reuse: run A to position 100, then run B from 0 in a
    different slot and different blocks; A's records must be untouched."""
    sim = Sim(pending_depth=INDEX_KPOOL + 2, max_tokens=8, max_num_reqs=2)
    for start, n in prefill_chunks(100, 8):
        sim.step(start, n, req=0)
    a_records = np.asarray(sim.records(range(25), req=0)).copy()
    for start, n in prefill_chunks(24, 5):
        sim.step(start, n, req=1)
    assert np.array_equal(np.asarray(sim.records(range(25), req=0)), a_records)
    assert sim.wrong_pools(24, req=1) == []
    assert sim.tag_violations == 0


def test_clear_pending_slots_resets_only_the_named_slots():
    sim = Sim(pending_depth=INDEX_KPOOL + 1, max_tokens=8, max_num_reqs=2)
    sim.step(0, 8, req=0)
    sim.step(0, 8, req=1)
    before = np.asarray(sim.pending_tag).copy()
    assert (before[1] != kpmod.PENDING_TAG_EMPTY).any()
    after = np.asarray(
        kpmod.clear_pending_slots(sim.pending_tag, jnp.asarray([1],
                                                               jnp.int32)))
    assert (after[1] == kpmod.PENDING_TAG_EMPTY).all()
    assert np.array_equal(after[2], before[2])


# ---------------------------------------------------------------------------
# 5. The addressing contract
# ---------------------------------------------------------------------------


def test_positions_disagreeing_with_the_derived_position_is_counted():
    """The ``first_key == 0`` check. ``ape`` is indexed by ``pos %
    index_kpool`` with ``pos`` derived from ``seq_lens``/``cu_q_lens``; the
    reference indexes by ``(idx - first_key) % index_kpool``. The two agree
    only under vLLM's unpadded ragged layout, so the disagreement is counted
    rather than assumed away."""
    sim = Sim(pending_depth=INDEX_KPOOL, max_tokens=8)
    diag = sim.step(0, 8, position_offset=5)
    assert int(diag[kpmod.DIAG_POSITION_MISMATCHES]) == 8
    sim2 = Sim(pending_depth=INDEX_KPOOL, max_tokens=8)
    assert int(sim2.step(0, 8)[kpmod.DIAG_POSITION_MISMATCHES]) == 0


def test_positions_must_be_one_dimensional():
    """A 3-D mrope position tensor would address the ring and the pooled pages
    with a rope-section index. GLM-5.3 is NoPE today; this fires at trace time
    if the vision path ever changes that."""
    sim = Sim(pending_depth=INDEX_KPOOL, max_tokens=8)
    with pytest.raises(AssertionError, match="positions must be 1-D"):
        kpool_write(sim.cache,
                    sim.pending_kv,
                    sim.pending_tag,
                    jnp.zeros((8, HEAD_DIM), jnp.float32),
                    jnp.zeros((8, HEAD_DIM), jnp.float32),
                    jnp.asarray(_ape(INDEX_KPOOL, HEAD_DIM)),
                    jnp.zeros((3, 8), jnp.int32),
                    jnp.asarray([8, 0], jnp.int32),
                    jnp.asarray([0, 8, 8], jnp.int32),
                    jnp.asarray(sim.block_tables.reshape(-1)),
                    jnp.asarray([1, 0], jnp.int32),
                    index_kpool=INDEX_KPOOL)


def test_padded_tokens_write_nothing_and_land_in_the_null_slot():
    """The padded tail of the token axis must not reach any real page or any
    real ring slot. Every other test runs padded too, but this one looks."""
    sim = Sim(pending_depth=INDEX_KPOOL + 1, max_tokens=16)
    sim.step(0, 5, req=0)
    tags = np.asarray(sim.pending_tag)
    # Slot 0 is the sentinel: it absorbs the padded tail and the non-carried
    # in-step positions, and its tag must stay unreadable.
    assert (tags[0] == kpmod.PENDING_TAG_EMPTY).all(), tags[0]
    assert (tags[2] == kpmod.PENDING_TAG_EMPTY).all(), "request 1 never ran"
    # Positions 0..4, of which the trailing `depth` = 5 are carried.
    assert sorted(t for t in tags[1]
                  if t != kpmod.PENDING_TAG_EMPTY) == [0, 1, 2, 3, 4]
    # Exactly one pool (0) is complete; nothing else was written anywhere.
    written = np.asarray(sim.cache).any(axis=(1, 2, 3))
    assert written.sum() == 1
    assert written[sim.block_tables[0][0]]


# ---------------------------------------------------------------------------
# 6. The slot contract -- the half the tag structurally cannot see
# ---------------------------------------------------------------------------


def _two_seq_step(sim: Sim, start: int, n_tokens: int, slots, variants=(0, 1)):
    """One step with **two live sequences** over the same absolute range.

    ``Sim.step`` runs a single live request, which is exactly the shape in
    which a slot collision is invisible: a slot is only shared if two live
    sequences hold it at once.
    """
    d = sim.head_dim
    k = np.zeros((sim.max_tokens, d), np.float32)
    g = np.zeros((sim.max_tokens, d), np.float32)
    positions = np.zeros(sim.max_tokens, np.int32)
    for r in range(2):
        for i in range(n_tokens):
            row = r * n_tokens + i
            p = start + i
            k[row] = key_of(p, variants[r], d)
            g[row] = gate_of(p, variants[r], d)
            positions[row] = p
    seq_lens = np.asarray([start + n_tokens] * 2, np.int32)
    cu = np.asarray([0, n_tokens, 2 * n_tokens], np.int32)
    cache, pkv, ptag, diag = _jitted(sim.index_kpool)(
        sim.cache, sim.pending_kv, sim.pending_tag, jnp.asarray(k),
        jnp.asarray(g), jnp.asarray(_ape(sim.index_kpool, d)),
        jnp.asarray(positions), jnp.asarray(seq_lens), jnp.asarray(cu),
        jnp.asarray(sim.block_tables.reshape(-1)),
        jnp.asarray(np.asarray(slots, np.int32)))
    sim.cache, sim.pending_kv, sim.pending_tag = cache, pkv, ptag
    return diag


@pytest.mark.parametrize(
    "slots,out_of_range,duplicates",
    [
        ((1, 2), 0, 0),  # the legal assignment
        ((1, 1), 0, 1),  # duplicated: the clip is a no-op, the tag blind
        ((3, 1), 1, 0),  # >= num_slots: folds onto slot 2 under the clip
        ((0, 1), 1, 0),  # slot 0 is the null sentinel, never handed out
        ((-5, -5), 2, 0),  # negative folds onto slot 0
    ],
)
def test_bad_ring_slots_are_counted_before_the_clip(slots, out_of_range,
                                                    duplicates):
    """``jnp.clip`` prevents an OOB write; these counters stop it being silent.

    ``num_slots`` here is ``max_num_reqs + 1 == 3``, so a legal slot is 1 or 2.
    Every illegal value is folded by the clip into a slot that a live request
    may hold, and the per-row int32 tag cannot detect the fold because a tag is
    an **absolute position** and absolute positions collide across requests.
    """
    sim = Sim(pending_depth=INDEX_KPOOL + 1, max_tokens=16, max_num_reqs=2)
    diag = _two_seq_step(sim, 0, 4, slots)
    print(f"slots={slots}: {kpmod.pool_write.describe_diagnostics(diag)}")
    assert int(diag[kpmod.DIAG_SLOT_OUT_OF_RANGE]) == out_of_range
    assert int(diag[kpmod.DIAG_SLOT_DUPLICATES]) == duplicates


def test_a_duplicated_slot_corrupts_a_pool_with_a_perfectly_matching_tag():
    """The negative control for the counter: wrong records, zero tag hits.

    Two requests share ring slot 1 across a chunk boundary. In step A both
    write positions 0..2 into the *same* rows, so the later write wins; in step
    B each completes pool 0 from those shared rows. Every tag matches -- the
    rows really do hold positions 0, 1 and 2 -- so ``tag_violations`` is 0
    while the pooled record is built from the other request's tokens. That is
    the
    class of wrong pooled record ``test_the_depth_table``'s
    ``silent == 0`` structurally cannot see.
    """

    def run(slots):
        sim = Sim(pending_depth=INDEX_KPOOL + 1, max_tokens=16, max_num_reqs=2)
        diag_a = _two_seq_step(sim, 0, 3, slots)
        diag_b = _two_seq_step(sim, 3, 3, slots)
        # Request 0 carries the committed content (variant 0), so its pool 0
        # record is comparable against `expected_records`.
        got = np.asarray(sim.records([0], req=0))[:, :HEAD_DIM + 1]
        want = np.asarray(expected_records([0]))
        tag_hits = int(diag_a[kpmod.DIAG_TAG_VIOLATIONS]) + int(
            diag_b[kpmod.DIAG_TAG_VIOLATIONS])
        dups = int(diag_a[kpmod.DIAG_SLOT_DUPLICATES]) + int(
            diag_b[kpmod.DIAG_SLOT_DUPLICATES])
        return np.array_equal(got, want), tag_hits, dups

    correct, tag_hits, dups = run((1, 2))
    assert correct, "distinct slots must reproduce the committed record"
    assert (tag_hits, dups) == (0, 0)

    correct, tag_hits, dups = run((1, 1))
    assert not correct, (
        "sharing a ring slot must corrupt request 0's pooled record; if this "
        "stops holding the negative control below is vacuous")
    assert tag_hits == 0, (
        "the tag is an absolute position, so a shared slot matches it "
        "perfectly -- this is exactly why kpmod.DIAG_SLOT_DUPLICATES exists")
    assert dups == 2, "one duplicated slot per step, two steps"


def test_only_the_trailing_depth_positions_are_carried():
    """A chunk longer than the ring must not alias itself -- the reason the
    reads and writes are split by source (the obvious design writes
    every in-step token into the ring first, which is wrong at any chunk longer
    than ``pending_depth``)."""
    depth = INDEX_KPOOL + 1
    sim = Sim(pending_depth=depth, max_tokens=32)
    sim.step(0, 20)
    tags = sorted(int(t) for t in np.asarray(sim.pending_tag)[1])
    assert tags == list(range(20 - depth, 20)), tags
    assert sim.wrong_pools(20) == []
