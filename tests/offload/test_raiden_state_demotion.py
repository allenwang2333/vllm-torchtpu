# SPDX-License-Identifier: Apache-2.0
"""Superseded GDN state demotion across a multi-turn PD-prefill conversation.

Drives the real upstream OffloadingConnectorScheduler under
TPURaidenOffloadingScheduler with the shape of the PP=8 Qwen3.5 prefill:
three Mamba "align" (GDN) groups and one full-attention group, GDN states
offloaded only through the scheduler's boundary hand-offs, and partial-tail
offload off. The KV cache manager also hands off a producer partial tail at
the prompt's last hash boundary, which no store job takes in that mode.
"""

import hashlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from test_raiden_store_offload import FakeKVCacheStore, make_manager

TPB = 16  # tokens per block = tokens per chunk
TPH = 4  # tokens per hash
GDN = (0, 1, 2)
FA = 3


class _RecordingStore(FakeKVCacheStore):
    def __init__(self):
        super().__init__()
        self.demoted: list[bytes] = []

    def demote(self, block_hashes):
        self.demoted.extend(block_hashes)
        return len(block_hashes)


class _Req:
    def __init__(self, req_id, tokens):
        from vllm.v1.request import RequestStatus

        self.request_id = req_id
        self.tokens = list(tokens)
        self.num_prompt_tokens = len(tokens)
        self.num_computed_tokens = 0
        self.status = RequestStatus.RUNNING
        self.kv_transfer_params = None
        self.kv_hints = None
        self.skip_reading_prefix_cache = False
        self.shared_prefix_boundary = 0
        self.max_tokens = 1

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def block_hashes(self):
        return [
            hashlib.sha256(repr(self.tokens[: (i + 1) * TPH]).encode()).digest()[:8]
            for i in range(len(self.tokens) // TPH)
        ]

    def is_finished(self):
        from vllm.v1.request import RequestStatus

        return RequestStatus.is_finished(self.status)


def _state_key(req, group_idx, boundary):
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(req.block_hashes[boundary // TPH - 1], group_idx)


class TestMultiTurnStateDemotion(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("TPU_RAIDEN_TIER_DEMOTE_STATE", None)
        self._next_block = 100

    def _sched(self, store):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            scheduler as up,
        )

        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

        def group(idx, recurrent):
            return up.GroupOffloadConfig(
                group_idx=idx,
                tokens_per_block=TPB,
                tokens_per_chunk=TPB,
                hashes_per_chunk=TPB // TPH,
                kv_event_group_spec=MagicMock(),
                sliding_window_size_in_chunks=1 if recurrent else None,
                kv_cache_spec=MagicMock(),
                manager_cls=MagicMock(),
                requires_cow_source=recurrent,
            )

        config = up.SchedulerOffloadConfig(
            kv_group_configs=tuple(group(g, True) for g in GDN) + (group(FA, False),),
            blocks_per_chunk=1,
            tokens_per_hash=TPH,
            num_workers=1,
            offload_prompt_only=False,
            supports_partial_tail=False,
        )
        manager = make_manager(store, num_blocks=256, world_size=1)
        spec = SimpleNamespace(
            get_manager=lambda: manager,
            dedup_layout=None,
            kv_events_config=SimpleNamespace(
                enable_kv_cache_events=False, self_describing_kv_events=False
            ),
        )
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True)
        )
        with (
            patch.object(
                up.SchedulerOffloadConfig, "from_spec", staticmethod(lambda *a: config)
            ),
            patch.object(up, "resolve_mamba_align_size", lambda *a: TPB),
        ):
            sched = TPURaidenOffloadingScheduler(spec, vllm_config, MagicMock())
        manager.demote = MagicMock(wraps=manager.demote)
        return sched, manager

    def _blocks(self, n):
        out = list(range(self._next_block, self._next_block + n))
        self._next_block += n
        return out

    def _step(self, sched, new=(), num_scheduled=None, offers=None, finished=()):
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            KVConnectorBlockState,
            SchedulerOutput,
        )

        out = SchedulerOutput.make_empty()
        out.scheduled_new_reqs = [
            SimpleNamespace(req_id=rid, block_ids=bids) for rid, bids in new
        ]
        out.scheduled_cached_reqs = CachedRequestData.make_empty()
        out.num_scheduled_tokens = dict(num_scheduled or {})
        out.finished_req_ids = set(finished)
        out.kv_connector_block_state = KVConnectorBlockState(
            req_ids=set(out.num_scheduled_tokens) | set(offers or {}),
            resolve_block_ids=lambda rid: None,
            boundary_state_offloads=dict(offers or {}),
        )
        return sched.build_connector_meta(out)

    def _admit(self, sched, req, local_hit):
        sched.on_new_request(req)
        hit, _ = sched.get_num_new_matched_tokens(req, local_hit)
        self.assertEqual(hit, 0)
        sched.update_state_after_alloc(req, MagicMock(), 0)

    def _compute_and_finish(self, sched, req, local_hit):
        """One prefill step for the whole prompt, then the request finishes."""
        from vllm.v1.request import RequestStatus

        prompt = req.num_prompt_tokens
        block_ids = tuple(self._blocks(-(-prompt // TPB)) for _ in range(4))
        aligned = prompt // TPB * TPB
        partial = prompt // TPH * TPH
        offers = [(g, block_ids[g][aligned // TPB - 1], aligned) for g in GDN]
        # The producer partial tail the KV cache manager also hands off.
        offers += [(g, self._blocks(1)[0], partial) for g in GDN]
        req.num_computed_tokens = local_hit
        self._step(
            sched,
            new=[(req.request_id, block_ids)],
            num_scheduled={req.request_id: prompt - local_hit},
            offers={req.request_id: offers},
        )
        self._land(sched)
        req.num_computed_tokens = prompt
        req.tokens.append(-1)  # the single sampled token
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        sched.request_finished(req)
        self._step(sched, finished=[req.request_id])
        self._land(sched)

    def _prefill(self, sched, req, local_hit):
        """One PD-prefill request: admission, one prefill step, finish."""
        self._admit(sched, req, local_hit)
        self._compute_and_finish(sched, req, local_hit)

    def _land(self, sched):
        """Acknowledges every parked store's fence and lands its save."""
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            scheduler as up,
        )

        from vllm_torchtpu.offload.raiden_connector import (
            RaidenOffloadingWorkerMetadata,
        )

        output = SimpleNamespace(
            kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
                fenced_jobs={jid: 1 for jid in sched._fence_pending}
            )
        )
        with patch.object(up.OffloadingConnectorScheduler, "update_connector_output"):
            sched.update_connector_output(output)
        store = sched._raiden_manager._store
        if store._saving:
            store.complete_save(list(store._saving))
        sched._raiden_manager.poll_finished_jobs()

    def test_turn_two_demotes_the_state_turn_one_stored(self):
        store = _RecordingStore()
        sched, manager = self._sched(store)
        self.assertEqual(sched._state_group_ids, frozenset(GDN))
        self.assertTrue(sched._demote_stale_state)

        turn1 = _Req("turn1", range(40))
        self._prefill(sched, turn1, local_hit=0)
        # Turn 1 stored only its chunk-aligned state (32); the partial tail at
        # 40 was handed off but not stored, so 32 is its newest state.
        manager.demote.assert_not_called()

        # Turn 2 extends turn 1 and is admitted from the HBM partial tail at
        # 40, past turn 1's stored state at 32.
        turn2 = _Req("turn2", list(range(40)) + list(range(1000, 1030)))
        self._prefill(sched, turn2, local_hit=40)

        (call,) = manager.demote.call_args_list
        demoted = set(call.args[0])
        turn1_state = {_state_key(turn2, g, 32) for g in GDN}
        turn2_state = {_state_key(turn2, g, 64) for g in GDN}
        self.assertEqual(turn1_state, {_state_key(turn1, g, 32) for g in GDN})
        self.assertLessEqual(turn1_state, demoted)
        self.assertFalse(turn2_state & demoted)
        self.assertEqual(set(store.demoted), set(manager._expand_keys(sorted(demoted))))
        tel = manager.telemetry
        self.assertEqual(tel["state_finished_reqs"], 2)
        self.assertEqual(tel["state_tracked_reqs"], 2)
        self.assertEqual(tel["state_demoting_reqs"], 1)
        self.assertEqual(tel["demote_req_keys"], len(call.args[0]))
        self.assertEqual(tel["demote_keys"], len(call.args[0]))
        self.assertEqual(tel["demote_unsupported_keys"], 0)
        self.assertEqual(sched._state_boundaries, {})
        self.assertEqual(sched._state_hits, {})

    def test_live_fork_keeps_the_state_it_was_admitted_from(self):
        store = _RecordingStore()
        sched, manager = self._sched(store)
        turn1 = _Req("turn1", range(40))
        self._prefill(sched, turn1, local_hit=0)
        # A fork of turn 1 is admitted from the same state and still running
        # when turn 2 finishes.
        fork = _Req("fork", list(range(40)) + list(range(2000, 2012)))
        self._admit(sched, fork, local_hit=40)
        turn2 = _Req("turn2", list(range(40)) + list(range(1000, 1030)))
        self._prefill(sched, turn2, local_hit=40)
        manager.demote.assert_not_called()
        self.assertEqual(manager.telemetry["demote_guarded_keys"], 2 * len(GDN))

        self._compute_and_finish(sched, fork, local_hit=40)
        (call,) = manager.demote.call_args_list
        demoted = set(call.args[0])
        self.assertLessEqual({_state_key(turn1, g, 32) for g in GDN}, demoted)
        self.assertFalse({_state_key(turn2, g, 64) for g in GDN} & demoted)
        self.assertFalse({_state_key(fork, g, 48) for g in GDN} & demoted)

    def test_shim_takes_exactly_turn_one_state_out_of_the_sweep(self):
        os.environ["TPU_RAIDEN_TIER_DEMOTE_SHIM"] = "1"
        store = FakeKVCacheStore()  # no demote(), like the shipped wheel
        sched, manager = self._sched(store)
        self.assertTrue(manager._shim)
        turn1 = _Req("turn1", range(40))
        self._prefill(sched, turn1, local_hit=0)
        turn1_state = {_state_key(turn1, g, 32) for g in GDN}
        self.assertLessEqual(turn1_state, set(manager._shadow))
        turn2 = _Req("turn2", list(range(40)) + list(range(1000, 1030)))
        self._prefill(sched, turn2, local_hit=40)

        (call,) = manager.demote.call_args_list
        turn2_state = {_state_key(turn2, g, 64) for g in GDN}
        live = set(manager._shadow)
        self.assertFalse(turn1_state & live)
        self.assertLessEqual(turn2_state, live)
        # Every full-attention chunk of both turns stays in the sweep.
        fa = {_state_key(turn2, FA, b) for b in (16, 32, 48, 64)}
        self.assertLessEqual(fa, live)
        tel = manager.telemetry
        self.assertEqual(tel["demote_emulated_keys"], len(turn1_state))
        self.assertEqual(tel["demote_unsupported_keys"], 0)

        # A full sweep refreshes every live key and none of the demoted ones.
        order = list(manager._shadow)
        touched = []
        real = manager._touch_key
        manager._touch_key = lambda key: touched.append(key) or real(key)
        manager._shim_last -= 10 * manager._shim_period_s
        manager.poll_finished_jobs()
        self.assertEqual(touched, order)
        self.assertFalse(turn1_state & set(touched))


class _LRUStore(FakeKVCacheStore):
    """The shipped KVCacheStore's LRU, as measured on tpu-sync-torch
    0.0.1.dev20261012084241: a strict LRU over unpinned entries; insert pins
    and evicts least recently used unpinned entries, all or nothing; release
    of a pinned entry moves it to the MRU end (the batch's first hash ends
    most recent); release of an unpinned entry and lookup(pin_found=False)
    change nothing."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity_hashes = capacity
        self.lru: dict[bytes, int] = {}  # hash -> pins, LRU first
        self.evicted: list[bytes] = []

    def lookup(self, block_hashes, enable_global=False, pin_found=True):
        from test_raiden_store_offload import FakeBlockStatus

        out = []
        for h in block_hashes:
            if h not in self.lru:
                break
            if pin_found:
                self.lru[h] += 1
            out.append((h, SimpleNamespace(status=FakeBlockStatus.HOST)))
        return out

    def release(self, block_hashes):
        for h in reversed(list(block_hashes)):
            if self.lru.get(h, 0) > 0:
                self.lru[h] -= 1
                self.lru[h] = self.lru.pop(h)

    def insert(self, block_hashes, slices, on_host):
        new = [h for h in block_hashes if h not in self.lru]
        free = self.capacity_hashes - len(self.lru)
        victims = [h for h, pins in self.lru.items() if pins == 0][
            : max(0, len(new) - free)
        ]
        if len(new) > free + len(victims):
            return False
        for h in victims:
            del self.lru[h]
        self.evicted.extend(victims)
        for h in block_hashes:
            self.lru[h] = self.lru.pop(h, 0) + 1
        if victims and self.eviction_callback is not None:
            self.eviction_callback(victims)
        return True


class TestDemoteShimOnNativeLRU(unittest.TestCase):
    """The shim against a model of the shipped store's LRU."""

    def _setup(self, shim):
        from vllm.v1.kv_offload.base import make_offload_key

        env = {"TPU_RAIDEN_TIER_DEMOTE_SHIM": "1" if shim else "0"}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        store = _LRUStore(capacity=8 * 4)
        manager = make_manager(store, num_blocks=8, world_size=1)
        self.assertEqual(manager._shim, shim)
        keys = {
            name: make_offload_key(name.encode(), 3 if name[0] == "L" else 0)
            for name in ("L0", "L1", "L2", "L3", "D0", "D1", "L4", "L5")
        }
        for name, key in keys.items():  # lands in this order: L0 oldest
            subs = manager._sub_hashes(key)
            self.assertTrue(store.insert(subs, [None] * len(subs), on_host=True))
            store.release(subs)
            manager._shadow_touch([key])
        return store, manager, keys

    def _evict_with(self, store, manager, names):
        from vllm.v1.kv_offload.base import make_offload_key

        before = list(store.evicted)
        for name in names:
            subs = manager._sub_hashes(make_offload_key(name.encode(), 3))
            self.assertTrue(store.insert(subs, [None] * len(subs), on_host=True))
            store.release(subs)
        gone = store.evicted[len(before) :]
        return {manager._sub_hash_to_offload_key(h)[0] for h in gone}

    def _demote_and_sweep(self, manager, keys):
        manager.demote([keys["D0"], keys["D1"]])
        manager._shim_last -= 10 * manager._shim_period_s
        manager.poll_finished_jobs()

    def test_demoted_states_are_evicted_before_older_live_keys(self):
        store, manager, keys = self._setup(shim=True)
        self._demote_and_sweep(manager, keys)
        self.assertEqual(manager.telemetry["demote_emulated_keys"], 2)
        self.assertEqual(manager.telemetry["shim_sweep_keys"], 6)
        self.assertEqual(
            self._evict_with(store, manager, ["N0", "N1"]), {keys["D0"], keys["D1"]}
        )
        # Among live keys the LRU order is kept: L0 is still the oldest.
        self.assertEqual(self._evict_with(store, manager, ["N2"]), {keys["L0"]})

    def test_without_the_shim_older_live_keys_go_first(self):
        store, manager, keys = self._setup(shim=False)
        self._demote_and_sweep(manager, keys)
        self.assertEqual(
            self._evict_with(store, manager, ["N0", "N1"]), {keys["L0"], keys["L1"]}
        )

    def test_a_key_with_an_inflight_store_is_not_taken_out(self):
        store, manager, keys = self._setup(shim=True)
        manager._inflight_store_keys.add(keys["D0"])
        self.assertEqual(manager.demote([keys["D0"], keys["D1"]]), 1)
        self.assertIn(keys["D0"], manager._shadow)
        self.assertNotIn(keys["D1"], manager._shadow)
        self.assertEqual(manager.telemetry["demote_emulated_skipped_keys"], 1)

    def test_eviction_leaves_the_sweep(self):
        store, manager, keys = self._setup(shim=True)
        self._evict_with(store, manager, ["N0"])
        manager.poll_finished_jobs()
        self.assertNotIn(keys["L0"], manager._shadow)


class TestDemoteWithoutStoreSupport(unittest.TestCase):
    def test_store_without_demote_counts_the_dropped_keys(self):
        from vllm.v1.kv_offload.base import make_offload_key

        from vllm_torchtpu.offload import raiden_store

        store = FakeKVCacheStore()
        self.assertFalse(hasattr(store, "demote"))
        manager = make_manager(store)
        keys = [make_offload_key(b"st%03d" % i, 1) for i in range(3)]
        with patch.object(raiden_store, "logger") as log:
            self.assertEqual(manager.demote(keys), 0)
            self.assertEqual(manager.demote(keys[:1]), 0)
        log.warning.assert_called_once()
        tel = manager.telemetry
        self.assertEqual(tel["demote_req_keys"], 4)
        self.assertEqual(tel["demote_unsupported_keys"], 4)
        self.assertEqual(tel["demote_keys"], 0)
        self.assertEqual(tel["demoted_sub_hashes"], 0)


if __name__ == "__main__":
    unittest.main()
