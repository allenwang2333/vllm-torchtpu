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
"""The stride-1 kpool slot pool: req_id -> pending-ring slot.

``runner/tpu_runner.py`` cannot be imported on the CPU rung (a pre-existing
``ModelOptNvFp4LinearMethod`` rebase drift), which is exactly why every
decision the mapping makes lives in ``runner/kpool_slots.py`` and the runner
keeps only the call. These are the tests that would not exist otherwise.
"""

import dataclasses

import pytest
import torch

from vllm_torchtpu.runner.kpool_slots import (KPOOL_NULL_SLOT, KpoolSlotPool,
                                              assert_pool_matches_ring,
                                              assert_slot_indices_sane,
                                              kpool_num_slots)


def _staging(n: int = 64) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.int32)


def _build(pool,
           req_ids,
           *,
           start_index=0,
           num_reqs=None,
           target_num_reqs=None,
           out=None,
           live=None):
    num_reqs = len(req_ids) if num_reqs is None else num_reqs
    target_num_reqs = num_reqs if target_num_reqs is None else target_num_reqs
    out = _staging() if out is None else out
    return pool.build_state_indices(req_ids,
                                    start_index=start_index,
                                    num_reqs=num_reqs,
                                    target_num_reqs=target_num_reqs,
                                    out=out,
                                    live_req_ids=live).tolist()


# --- the pool's shape ------------------------------------------------------


def test_the_pool_is_max_num_reqs_plus_one_and_never_hands_out_slot_zero():
    """Slot 0 is the sentinel every padded position and every non-carried
    token writes; a live request on it would be clobbered every step."""
    pool = KpoolSlotPool(max_num_reqs=8)
    assert pool.num_slots == kpool_num_slots(8) == 9
    assert pool.num_free == 8
    slots = _build(pool, [f"r{i}" for i in range(8)])
    assert sorted(slots) == list(range(1, 9))
    assert KPOOL_NULL_SLOT not in slots


def test_the_first_request_gets_slot_one():
    """Purely so a debug print reads the same way as the mamba pool's."""
    pool = KpoolSlotPool(max_num_reqs=4)
    assert _build(pool, ["a"], target_num_reqs=4) == [1, 0, 0, 0]


# --- what makes it a *pool* ------------------------------------------------


def test_the_slot_follows_the_req_id_through_a_reorder():
    """The mapping is keyed on req_id, not persistent-batch position.

    Upstream vLLM's ``InputBatch.condense`` reorders the batch and this repo
    does not subclass it, so a position-keyed pool would hand a request a
    different ring every time a neighbour finished -- and the ring rows it
    then read would carry another request's positions, whose *tags match*,
    because absolute positions collide across requests.
    """
    pool = KpoolSlotPool(max_num_reqs=4)
    first = _build(pool, ["a", "b", "c"], target_num_reqs=4)
    reordered = _build(pool, ["c", "a", "b"], target_num_reqs=4)
    assert reordered[:3] == [first[2], first[0], first[1]]


def test_a_departed_request_returns_its_slot_in_the_same_step():
    """Otherwise the pool is exhausted after ``max_num_reqs`` *cumulative*
    requests rather than ``max_num_reqs`` concurrent ones -- a leak that
    presents as a mid-soak crash."""
    pool = KpoolSlotPool(max_num_reqs=2)
    _build(pool, ["a", "b"])
    assert pool.num_free == 0
    slots = _build(pool, ["c"], target_num_reqs=2)
    assert slots[0] in (1, 2) and slots[1] == KPOOL_NULL_SLOT
    assert pool.slot_for("a") is None and pool.slot_for("b") is None


def test_two_live_requests_never_share_a_slot_over_a_long_churn():
    """The property ``DIAG_SLOT_DUPLICATES`` exists to catch, checked over a
    schedule that admits and retires requests at every offset."""
    pool = KpoolSlotPool(max_num_reqs=6)
    out = _staging()
    live: list[str] = []
    for step in range(200):
        if step % 3 and len(live) < 6:
            live.append(f"r{step}")
        if step % 5 == 0 and live:
            live.pop(0)
        slots = _build(pool, list(live), target_num_reqs=6, out=out)
        assert_slot_indices_sane(slots,
                                 num_slots=pool.num_slots,
                                 num_live=len(live),
                                 where=f"step {step}")


def test_exhaustion_raises_rather_than_folding_onto_a_live_slot():
    pool = KpoolSlotPool(max_num_reqs=2)
    with pytest.raises(RuntimeError, match="slot pool exhausted"):
        # `live_req_ids` deliberately keeps a and b alive while c arrives.
        _build(pool, ["a", "b", "c"], live=["a", "b", "c"])


# --- the DP-dummy case -----------------------------------------------------


def test_num_reqs_larger_than_the_live_batch_pads_with_the_null_slot():
    """``run_dp_dummy_draft``: an idle DP rank replays a peer's collective
    trace at the coordinated shape while its own batch is shorter. Without the
    clamp the compact layout raises ``IndexError`` and kills the engine."""
    pool = KpoolSlotPool(max_num_reqs=8)
    slots = _build(pool, ["a"], num_reqs=6, target_num_reqs=6)
    assert slots == [1, 0, 0, 0, 0, 0]


def test_an_entirely_empty_batch_at_a_coordinated_shape_is_all_null():
    pool = KpoolSlotPool(max_num_reqs=8)
    assert _build(pool, [], num_reqs=8, target_num_reqs=8) == [0] * 8
    assert pool.num_free == 8


def test_a_chunk_window_past_the_end_of_req_ids_is_clamped():
    """`start_index` walks the persistent batch in chunks; the last chunk can
    be shorter than `num_reqs`."""
    pool = KpoolSlotPool(max_num_reqs=8)
    all_slots = _build(pool, ["a", "b", "c"], target_num_reqs=3)
    tail = _build(pool, ["a", "b", "c"],
                  start_index=2,
                  num_reqs=4,
                  target_num_reqs=4)
    assert tail == [all_slots[2], 0, 0, 0]


def test_a_hole_inside_the_live_prefix_is_an_assertion_not_the_null_slot():
    pool = KpoolSlotPool(max_num_reqs=4)
    with pytest.raises(AssertionError, match="is None inside"):
        _build(pool, ["a", None, "c"], live=["a", "c"])


# --- the staging tensor ----------------------------------------------------


def test_the_staging_tensor_must_be_int32():
    pool = KpoolSlotPool(max_num_reqs=4)
    with pytest.raises(ValueError, match="must be int32"):
        _build(pool, ["a"], out=torch.zeros(8, dtype=torch.int64))


def test_a_short_staging_tensor_raises_rather_than_truncating():
    pool = KpoolSlotPool(max_num_reqs=64)
    with pytest.raises(ValueError, match="holds 4 entries, need 8"):
        _build(pool, ["a"],
               num_reqs=8,
               target_num_reqs=8,
               out=torch.zeros(4, dtype=torch.int32))


def test_the_padded_tail_is_rewritten_every_build():
    """A stale slot left in the tail from a longer previous chunk would be a
    live request's slot, written by every padded token."""
    pool = KpoolSlotPool(max_num_reqs=8)
    out = _staging()
    _build(pool, ["a", "b", "c", "d"], target_num_reqs=8, out=out)
    _build(pool, ["a"], target_num_reqs=8, out=out, live=["a"])
    assert out[:8].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


# --- the two host-side checks ---------------------------------------------


def test_assert_slot_indices_sane_catches_a_mamba_style_block_id():
    """The whole point of a dedicated field: a `mamba_state_indices` entry
    under unified-align mode is an attention-pool block id, orders of
    magnitude past `max_num_reqs + 1`."""
    with pytest.raises(ValueError, match="DIAG_SLOT_OUT_OF_RANGE"):
        assert_slot_indices_sane([1, 4097, 0],
                                 num_slots=9,
                                 num_live=2,
                                 where="test")


def test_assert_slot_indices_sane_catches_a_live_request_on_the_null_slot():
    with pytest.raises(ValueError, match="carry the null slot 0"):
        assert_slot_indices_sane([1, 0, 0],
                                 num_slots=9,
                                 num_live=2,
                                 where="test")


def test_assert_slot_indices_sane_catches_a_shared_slot():
    with pytest.raises(ValueError, match="DIAG_SLOT_DUPLICATES"):
        assert_slot_indices_sane([3, 3], num_slots=9, num_live=2, where="test")


def test_assert_slot_indices_sane_allows_the_all_null_padded_tail():
    assert_slot_indices_sane([1, 2, 0, 0, 0],
                             num_slots=9,
                             num_live=2,
                             where="test")


def test_assert_pool_matches_ring_ties_the_two_derivations():
    from vllm_torchtpu.kernels.glm5_kpool import derive_pending_state_geometry

    pool = KpoolSlotPool(max_num_reqs=16)
    geometry = derive_pending_state_geometry(index_kpool=4,
                                             index_head_dim=128,
                                             max_num_reqs=16)
    assert_pool_matches_ring(pool,
                             ring_num_slots=geometry.num_slots,
                             where="test")
    with pytest.raises(ValueError, match="must be max_num_reqs \\+ 1"):
        assert_pool_matches_ring(pool, ring_num_slots=16, where="test")


# --- the handoff itself ----------------------------------------------------


def test_attention_metadata_carries_the_field_and_it_is_a_jax_data_field():
    """Both halves matter. A field the dataclass has but the pytree
    registration omits is dropped on the way into every jitted op -- it would
    arrive as ``None`` inside the kernel and the indexer's refusal would fire
    on device instead of here."""
    from vllm_torchtpu.layers.common.attention_metadata import \
        AttentionMetadata

    names = [f.name for f in dataclasses.fields(AttentionMetadata)]
    assert "kpool_state_indices" in names
    assert "mamba_state_indices" in names
    assert names.index("kpool_state_indices") != names.index(
        "mamba_state_indices")

    registered = AttentionMetadata.__dataclass_fields__
    assert registered["kpool_state_indices"].default is None

    import jax
    metadata = AttentionMetadata(input_positions=None)
    leaves, treedef = jax.tree_util.tree_flatten(metadata)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.kpool_state_indices is None


def test_the_indexer_still_refuses_a_metadata_without_the_field():
    """The refusal is now a regression guard: if someone removes the field or
    hands the indexer a bare object, it must still raise rather than reach for
    ``mamba_state_indices``."""
    import inspect
    import linecache

    from vllm_torchtpu.models.vllm.glm5_next import indexer as indexer_mod

    # `inspect.getsource` slices by `co_firstlineno` out of linecache, which
    # holds whatever the file said when it was first read. Editing the module
    # while a long test run is in flight therefore makes this return a
    # *different function's* body -- a false failure that reads exactly like
    # a real one. Drop the cached copy first.
    linecache.checkcache(inspect.getfile(indexer_mod))
    body = inspect.getsource(indexer_mod.Glm5NextKpoolIndexer.forward)
    # The refusal, still present and still naming the right field.
    assert 'getattr(metadata, "kpool_state_indices", None)' in body
    assert "AttentionMetadata carries no" in body
    # And the one substitution that must never become a fallback. Comments
    # mentioning the name are fine (they are the reason it must not be used);
    # an *access* is not.
    assert "metadata.mamba_state_indices" not in body
    assert 'getattr(metadata, "mamba_state_indices"' not in body


def test_the_dummy_slot_vector_is_live_valid_not_all_null():
    """The dummy run's sequences are LIVE, so slot 0 is wrong for them.

    A source check, because `tpu_runner.py` is not importable here. The mamba
    dummy is legitimately all-zeros -- slot 0 is a real null block it may
    write. The kpool dummy is not: `kpool_write` classifies a sequence as
    live from its query length, the dummy gives every sequence q_len 1, and
    `DIAG_SLOT_OUT_OF_RANGE` counts a live sequence on slot 0 because slot 0
    is the sentinel every padded token also writes.

    All-zeros here cost a full 17-minute job on the 328 GB checkpoint
    which failed with
    `slot_out_of_range=16` -- 16 being the dummy bucket's live sequence
    count. The counter was correct; the dummy was not.
    """
    import pathlib as _pathlib

    source = (_pathlib.Path(__file__).resolve().parents[2] / "src" /
              "vllm_torchtpu" / "runner" / "tpu_runner.py").read_text()
    assert "dummy_kpool_state_indices = torch.clamp(" in source, (
        "the kpool dummy slot vector is no longer built with clamp(arange)")
    assert "torch.arange(1, num_reqs + 1, dtype=torch.int32)" in source, (
        "the kpool dummy slot vector must start at 1, not 0")
    # And the exact regression: it must not be zeros.
    assert "dummy_kpool_state_indices = (torch.zeros(" not in source, (
        "the kpool dummy slot vector is all-zeros again; every live dummy "
        "sequence would sit on the null sentinel and DIAG_SLOT_OUT_OF_RANGE "
        "will refuse the step on device.")


def test_a_live_sequence_on_the_null_slot_is_refused_by_the_host_twin():
    """The same rule the device counter applies, checked here.

    `assert_slot_indices_sane` and `DIAG_SLOT_OUT_OF_RANGE` must agree about
    what is illegal, or one of them is decoration.
    """
    with pytest.raises(ValueError, match="carry the null slot 0"):
        assert_slot_indices_sane([0] * 16,
                                 num_slots=257,
                                 num_live=16,
                                 where="dummy run")
