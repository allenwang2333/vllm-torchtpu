# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deduplicated host-tier layout (offload/tier_dedup.py).

The two class stores are FakeKVCacheStores, so the routing, the manager's
per-class launches and the connector's state demotion run on the VM.
"""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from test_raiden_store_offload import (
    FAKE_STORE_TYPES,
    FakeKVCacheStore,
    _ctx,
    make_manager,
    sub_hashes,
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key

from vllm_torchtpu.offload.tier_dedup import (
    FA_CLASS,
    STATE_CLASS,
    ClassRoutedStore,
    TierDedupLayout,
    fa_segment_key,
    remove_stale_fa_segments,
    resolve_tier_dedup_layout,
)

FACTOR = 4  # sub-hashes per key in make_manager


def _fa(n: int) -> bytes:
    return make_offload_key(b"fa%03d" % n, 0)


def _st(n: int) -> bytes:
    return make_offload_key(b"st%03d" % n, 1)


class DemotingFakeStore(FakeKVCacheStore):
    def __init__(self, capacity=64):
        super().__init__()
        self._capacity = capacity
        self.demoted: list[bytes] = []
        self.raiden_controller_address = "127.0.0.1:0"
        self.store_server_address = ""

    def capacity(self):
        return self._capacity

    def demote(self, block_hashes):
        moved = [
            h
            for h in block_hashes
            if h in self.entries and self.entries[h].pin_count == 0
        ]
        self.demoted.extend(moved)
        return len(moved)


LAYOUT = TierDedupLayout(
    fa_group_ids=frozenset({0}),
    replicas=4,
    world_size=8,
    fa_capacity_keys=16,
    state_capacity_keys=8,
    state_port=27800,
    fa_port=27900,
)


def _facade():
    stores = {FA_CLASS: DemotingFakeStore(), STATE_CLASS: DemotingFakeStore()}
    return ClassRoutedStore(stores, LAYOUT, FAKE_STORE_TYPES.RaidenBlockId), stores


class TestResolveLayout(unittest.TestCase):
    KW = dict(
        fa_group_ids=[0],
        total_num_kv_heads=2,
        tp_size=8,
        world_size=8,
        pcp_size=1,
        dcp_size=1,
        cpu_bytes=560_000_000_000,
        worker_bytes_per_block=5_242_880,
        state_port=27800,
    )

    def test_off_unless_requested(self):
        with patch.dict(os.environ, {"TPU_RAIDEN_TIER_DEDUP": "0"}):
            self.assertIsNone(resolve_tier_dedup_layout(**self.KW))

    def test_replicated_heads_split_the_budget(self):
        env = {"TPU_RAIDEN_TIER_DEDUP": "1", "TPU_RAIDEN_TIER_FA_FRACTION": "0.5"}
        with patch.dict(os.environ, env):
            layout = resolve_tier_dedup_layout(**self.KW)
        self.assertEqual(layout.replicas, 4)
        self.assertEqual(layout.num_head_groups, 2)
        # 280 GB at 2 copies x 5.24 MB, and 280 GB at 8 copies x 5.24 MB.
        self.assertEqual(layout.fa_capacity_keys, 280_000_000_000 // (2 * 5_242_880))
        self.assertEqual(layout.state_capacity_keys, 280_000_000_000 // (8 * 5_242_880))
        self.assertEqual(layout.leader_ranks(), [0, 4])
        self.assertEqual([layout.head_group(r) for r in range(8)], [0] * 4 + [1] * 4)
        self.assertEqual(layout.fa_port, 27900)
        self.assertEqual(layout.class_of(sub_hashes(_fa(0))[0]), FA_CLASS)
        self.assertEqual(layout.class_of(sub_hashes(_st(0))[0]), STATE_CLASS)

    def test_not_applied_without_replication_or_with_cp(self):
        with patch.dict(os.environ, {"TPU_RAIDEN_TIER_DEDUP": "1"}):
            self.assertIsNone(
                resolve_tier_dedup_layout(**{**self.KW, "total_num_kv_heads": 8})
            )
            self.assertIsNone(resolve_tier_dedup_layout(**{**self.KW, "pcp_size": 2}))


class TestClassRoutedStore(unittest.TestCase):
    def _seed_host(self, store, key):
        from test_raiden_store_offload import seed_host_entries

        seed_host_entries(store, key)

    def test_lookup_prefix_ends_at_first_class_miss_and_returns_extra_pins(self):
        facade, stores = _facade()
        for key in (_fa(0), _fa(1)):
            self._seed_host(stores[FA_CLASS], key)
        hashes = sub_hashes(_fa(0)) + sub_hashes(_st(0)) + sub_hashes(_fa(1))
        found = facade.lookup(hashes)
        self.assertEqual([h for h, _ in found], sub_hashes(_fa(0)))
        # _fa(1) was found and pinned by its store, then handed back.
        self.assertEqual(
            stores[FA_CLASS].pinned_hashes(), {h: 1 for h in sub_hashes(_fa(0))}
        )

    def test_insert_rolls_back_when_one_class_refuses(self):
        facade, stores = _facade()
        stores[STATE_CLASS].fail_insert = True
        hashes = sub_hashes(_fa(0)) + sub_hashes(_st(0))
        slices = [
            FAKE_STORE_TYPES.RaidenBlockId(
                raiden_id=facade.raiden_id,
                device_block_id=i,
                status=FAKE_STORE_TYPES.BlockStatus.HBM,
            )
            for i in range(len(hashes))
        ]
        self.assertFalse(facade.insert(hashes, slices, on_host=False))
        self.assertEqual(stores[FA_CLASS].pinned_hashes(), {})

    def test_insert_rebinds_slices_to_each_class_store(self):
        facade, stores = _facade()
        hashes = sub_hashes(_fa(0)) + sub_hashes(_st(0))
        slices = [
            FAKE_STORE_TYPES.RaidenBlockId(
                raiden_id=None,
                device_block_id=i,
                status=FAKE_STORE_TYPES.BlockStatus.HBM,
            )
            for i in range(len(hashes))
        ]
        self.assertTrue(facade.insert(hashes, slices, on_host=False))
        (fa_insert,) = stores[FA_CLASS].calls_named("insert")
        (st_insert,) = stores[STATE_CLASS].calls_named("insert")
        self.assertEqual(fa_insert.args[0], sub_hashes(_fa(0)))
        self.assertTrue(
            all(sl.raiden_id is stores[FA_CLASS].raiden_id for sl in fa_insert.args[1])
        )
        self.assertEqual([sl.device_block_id for sl in st_insert.args[1]], [4, 5, 6, 7])

    def test_eviction_callback_counts_per_class(self):
        facade, stores = _facade()
        seen = []
        facade.set_eviction_callback(seen.extend)
        stores[FA_CLASS].simulate_evict([b"a", b"b"])
        stores[STATE_CLASS].simulate_evict([b"c"])
        self.assertEqual(seen, [b"a", b"b", b"c"])
        self.assertEqual(facade.evicted_sub_hashes, {FA_CLASS: 2, STATE_CLASS: 1})

    def test_demote_routes_by_class(self):
        facade, stores = _facade()
        self._seed_host(stores[STATE_CLASS], _st(0))
        self._seed_host(stores[FA_CLASS], _fa(0))
        self.assertEqual(facade.demote(sub_hashes(_st(0)) + sub_hashes(_fa(0))), 8)
        self.assertEqual(stores[STATE_CLASS].demoted, sub_hashes(_st(0)))
        self.assertEqual(stores[FA_CLASS].demoted, sub_hashes(_fa(0)))


class TestManagerOverClassRoutedStore(unittest.TestCase):
    def _manager(self):
        facade, stores = _facade()
        manager = make_manager(facade, num_blocks=8, device_block_size_factor=FACTOR)
        return manager, stores

    def _complete_saves(self, stores):
        for store in stores.values():
            pending = list(store._saving)
            if pending:
                store.complete_save(pending)

    def test_store_and_load_span_both_classes(self):
        manager, stores = self._manager()
        keys = [_fa(0), _st(0)]
        manager.prepare_store(keys, _ctx())
        manager.submit_store_job(1, keys, [10, 11])
        self.assertEqual(len(stores[FA_CLASS].calls_named("save")), 1)
        self.assertEqual(len(stores[STATE_CLASS].calls_named("save")), 1)
        self._complete_saves(stores)
        (done,) = manager.poll_finished_jobs()
        self.assertTrue(done.success)

        spec = manager.prepare_load(keys, _ctx("r2"))
        self.assertTrue(spec.pinned)
        manager.submit_load_job(2, keys, [20, 21], "r2")
        (fa_load,) = stores[FA_CLASS].calls_named("load")
        self.assertEqual(fa_load.args[1], [80, 81, 82, 83])
        for store in stores.values():
            store.complete_load(list(store._loading))
        (done,) = manager.poll_finished_jobs()
        self.assertTrue(done.success)
        for store in stores.values():
            self.assertEqual(store.pinned_hashes(), {})

    def test_class_telemetry_counts_stores_and_loads_per_class(self):
        from vllm_torchtpu.offload import raiden_store

        manager, stores = self._manager()
        keys = [_fa(0), _st(0)]
        manager.prepare_store(keys, _ctx())
        manager.submit_store_job(1, keys, [10, 11])
        self._complete_saves(stores)
        manager.poll_finished_jobs()
        manager.prepare_load([_fa(0)], _ctx("r2"))
        manager.submit_load_job(2, [_fa(0)], [20], "r2")
        manager._tel_last_log -= 61.0
        with patch.object(raiden_store.logger, "info") as info:
            manager.poll_finished_jobs()
        (line,) = [
            call.args[0] % call.args[1:]
            for call in info.call_args_list
            if "offload-tier classes" in call.args[0]
        ]
        for part in (
            f"fa_stored={FACTOR}",
            f"state_stored={FACTOR}",
            f"fa_loaded={FACTOR}",
            "state_loaded=0",
            "fa_evicted=0",
        ):
            self.assertIn(part, line)

    def test_save_refused_by_one_class_fails_after_the_other_lands(self):
        manager, stores = self._manager()
        stores[FA_CLASS].fail_save = True
        keys = [_fa(0), _st(0)]
        manager.prepare_store(keys, _ctx())
        manager.submit_store_job(1, keys, [10, 11])
        # The state part is in flight: the job is not terminal yet.
        self.assertTrue(manager.is_job_launched(1))
        self.assertEqual(manager.poll_finished_jobs(), [])
        self._complete_saves(stores)
        (done,) = manager.poll_finished_jobs()
        self.assertFalse(done.success)
        for store in stores.values():
            self.assertEqual(store.pinned_hashes(), {})

    def test_load_refused_by_one_class_fails_after_the_other_lands(self):
        manager, stores = self._manager()
        from test_raiden_store_offload import seed_host_entries

        seed_host_entries(stores[FA_CLASS], _fa(0))
        seed_host_entries(stores[STATE_CLASS], _st(0))
        keys = [_fa(0), _st(0)]
        manager.prepare_load(keys, _ctx("r1"))
        stores[STATE_CLASS].fail_load = True
        manager.submit_load_job(3, keys, [20, 21], "r1")
        self.assertTrue(manager.is_job_launched(3))
        stores[FA_CLASS].complete_load(list(stores[FA_CLASS]._loading))
        (done,) = manager.poll_finished_jobs()
        self.assertFalse(done.success)
        self.assertEqual(done.failed_device_block_ids, [20, 21])
        for store in stores.values():
            self.assertEqual(store.pinned_hashes(), {})

    def test_demote_expands_keys_and_counts(self):
        manager, stores = self._manager()
        from test_raiden_store_offload import seed_host_entries

        seed_host_entries(stores[STATE_CLASS], _st(0))
        self.assertEqual(manager.demote([_st(0), _st(9)]), FACTOR)
        self.assertEqual(manager.telemetry["demote_keys"], 2)
        self.assertEqual(manager.telemetry["demoted_sub_hashes"], FACTOR)


class _Req:
    def __init__(self, req_id, hashes):
        self.request_id = req_id
        self.block_hashes = hashes
        self.num_prompt_tokens = len(hashes) * 4
        self.num_tokens = self.num_prompt_tokens
        self.kv_transfer_params = None


def _tracked(req):
    """The scheduler's per-request state the demotion hooks read."""
    return SimpleNamespace(req=req, req_context=ReqContext(req_id=req.request_id))


class TestStateDemotion(unittest.TestCase):
    TPC = 4

    def _sched(self, manager, on=True):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (  # noqa: E501
            GroupOffloadConfig,
            OffloadingConnectorScheduler,
            SchedulerOffloadConfig,
        )

        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

        def group(idx, recurrent):
            return GroupOffloadConfig(
                group_idx=idx,
                tokens_per_block=self.TPC,
                tokens_per_chunk=self.TPC,
                hashes_per_chunk=1,
                kv_event_group_spec=MagicMock(),
                sliding_window_size_in_chunks=1 if recurrent else None,
                kv_cache_spec=MagicMock(),
                manager_cls=MagicMock(),
                requires_cow_source=recurrent,
            )

        sched = object.__new__(TPURaidenOffloadingScheduler)
        sched.config = SchedulerOffloadConfig(
            kv_group_configs=(group(0, False), group(1, True)),
            blocks_per_chunk=1,
            tokens_per_hash=self.TPC,
            num_workers=2,
            offload_prompt_only=False,
            supports_partial_tail=False,
        )
        sched._raiden_manager = manager
        sched._req_status = {}
        sched._init_state_demotion(on)
        self._patch = patch.object(
            OffloadingConnectorScheduler,
            "_build_aligned_boundary_store_jobs",
            return_value={},
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        return sched

    def test_finish_demotes_older_boundaries_and_the_admission_hit(self):
        manager = MagicMock()
        sched = self._sched(manager)
        req = _Req("r1", [b"h%02d" % i for i in range(8)])
        sched._req_status["r1"] = _tracked(req)
        sched._record_state_hit(req, 8)
        sched._build_aligned_boundary_store_jobs({"r1": [(1, 5, 16), (0, 6, 16)]})
        sched._build_aligned_boundary_store_jobs({"r1": [(1, 7, 24)]})

        sched._demote_superseded_states(["r1"])

        (call,) = manager.demote.call_args_list
        self.assertEqual(
            call.args[0],
            [
                make_offload_key(b"h03", 1),  # boundary 16 < newest 24
                make_offload_key(b"h01", 1),  # admitted from boundary 8
            ],
        )
        self.assertEqual(sched._state_boundaries, {})
        self.assertEqual(sched._state_hits, {})

    def test_request_without_new_boundary_demotes_nothing(self):
        manager = MagicMock()
        sched = self._sched(manager)
        req = _Req("r1", [b"h%02d" % i for i in range(8)])
        sched._req_status["r1"] = _tracked(req)
        sched._record_state_hit(req, 8)
        self.assertIn("r1", sched._state_hits)
        sched._demote_superseded_states(["r1"])
        manager.demote.assert_not_called()
        self.assertEqual(sched._state_hits, {})

    def test_off_records_nothing(self):
        manager = MagicMock()
        sched = self._sched(manager, on=False)
        req = _Req("r1", [b"h%02d" % i for i in range(8)])
        sched._req_status["r1"] = _tracked(req)
        sched._record_state_hit(req, 8)
        sched._build_aligned_boundary_store_jobs({"r1": [(1, 5, 16)]})
        self.assertEqual(sched._state_boundaries, {})
        self.assertEqual(sched._state_hits, {})


class TestFaRegistration(unittest.TestCase):
    def test_worker_joins_its_group_segment_and_clears_stale_ones(self):
        from vllm_torchtpu.offload import raiden_store

        port = 29977
        token = f"{os.getpid()}xabc"
        marker = raiden_store.controller_ready_marker(port)
        with open(marker, "w") as f:
            f.write(token + "\n")
        self.addCleanup(lambda: os.path.exists(marker) and os.unlink(marker))
        stale = f"/dev/shm/raidenfa-{port}-oldrun-g1"
        with open(stale, "w"):
            pass
        self.addCleanup(lambda: os.path.exists(stale) and os.unlink(stale))

        created = {}

        class FakeManager:
            def __init__(self, **kwargs):
                created.update(kwargs)

        fake_mod = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
        fake_mod.KVCacheManager = FakeManager
        worker = object.__new__(raiden_store.RaidenStoreOffloadingWorker)
        worker._rank = 5
        worker._device_tensors = [["t"]]
        worker._fa_target = raiden_store.FaRegistration(
            controller_address=f"127.0.0.1:{port}",
            host_blocks=40,
            fa_port=port,
            head_group=1,
        )
        with patch.dict(
            sys.modules,
            {
                "tpu_sync": types.ModuleType("tpu_sync"),
                "tpu_sync.api": types.ModuleType("tpu_sync.api"),
                "tpu_sync.api.torch": types.ModuleType("tpu_sync.api.torch"),
                "tpu_sync.api.torch.kv_cache_manager": fake_mod,
            },
        ):
            sys.modules["tpu_sync.api.torch"].kv_cache_manager = fake_mod
            worker._register_fa()

        self.assertEqual(created["shm_group_key"], fa_segment_key(port, token, 1))
        self.assertEqual(created["host_blocks_to_allocate"], 40)
        self.assertEqual(created["node_id"], 5)
        self.assertEqual(created["raiden_controller_address"], f"127.0.0.1:{port}")
        self.assertFalse(os.path.exists(stale))

    def test_stale_cleanup_keeps_this_runs_segments(self):
        port = 29978
        keep = f"/dev/shm/{fa_segment_key(port, 'now', 0)}"
        drop = f"/dev/shm/{fa_segment_key(port, 'before', 0)}.lock"
        other = f"/dev/shm/raidenfa-{port + 1}-before-g0"
        for path in (keep, drop, other):
            with open(path, "w"):
                pass
            self.addCleanup(lambda p=path: os.path.exists(p) and os.unlink(p))
        removed = remove_stale_fa_segments(port, "now")
        self.assertEqual(removed, [os.path.basename(drop)])
        self.assertTrue(os.path.exists(keep))
        self.assertTrue(os.path.exists(other))
