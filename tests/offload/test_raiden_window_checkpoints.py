# SPDX-License-Identifier: Apache-2.0
"""Window-checkpoint offload policy (TPU_OFFLOAD_WINDOW_CHECKPOINT_INTERVAL).

Drives vLLM's real offloading scheduler (store-job building and the hybrid
lookup) through `TPURaidenOffloadingScheduler` over a FakeKVCacheStore, with a
DeepSeek-V4-shaped group geometry: one 1024-token full-attention group, two
128-token sliding-window groups (one of them the EAGLE/MTP group), a 16-token
state group with an 8-token window and a 32-token state group with a
128-token window.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_raiden_store_offload import FakeKVCacheStore, make_manager, sub_hashes
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    SchedulerOffloadConfig,
)
from vllm.v1.kv_offload.base import get_offload_group_idx
from vllm.v1.request import RequestStatus

FULL = 1024
TOKENS_PER_HASH = 16
# (tokens_per_chunk, window_in_chunks or None, is_eagle)
DSV4_GROUPS = [
    (128, 1, False),
    (128, 1, True),
    (FULL, None, False),
    (16, 1, False),
    (32, 4, False),
]
FULL_GROUP = 2


def dsv4_config() -> SchedulerOffloadConfig:
    groups = []
    for idx, (chunk, window, eagle) in enumerate(DSV4_GROUPS):
        groups.append(
            GroupOffloadConfig(
                group_idx=idx,
                tokens_per_block=chunk,
                tokens_per_chunk=chunk,
                hashes_per_chunk=chunk // TOKENS_PER_HASH,
                kv_event_group_spec=None,
                sliding_window_size_in_chunks=window,
                alignment_chunk_count=None if window is None else FULL // chunk,
                is_eagle_group=eagle,
            )
        )
    return SchedulerOffloadConfig(
        kv_group_configs=tuple(groups),
        blocks_per_chunk=1,
        tokens_per_hash=TOKENS_PER_HASH,
        num_workers=1,
        offload_prompt_only=True,
        supports_partial_tail=False,
    )


def make_scheduler(interval):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    store = FakeKVCacheStore()
    manager = make_manager(store, device_block_size_factor=1, num_blocks=1 << 20)
    spec = SimpleNamespace(
        get_manager=lambda: manager,
        kv_events_config=SimpleNamespace(
            enable_kv_cache_events=False, self_describing_kv_events=False
        ),
        tokens_per_block=[g[0] for g in DSV4_GROUPS],
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True)
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=None) for _ in DSV4_GROUPS]
    )
    env = (
        {}
        if interval is None
        else {"TPU_OFFLOAD_WINDOW_CHECKPOINT_INTERVAL": str(interval)}
    )
    with (
        patch.dict("os.environ", env),
        patch.object(SchedulerOffloadConfig, "from_spec", return_value=dsv4_config()),
    ):
        if interval is None:
            import os

            os.environ.pop("TPU_OFFLOAD_WINDOW_CHECKPOINT_INTERVAL", None)
        sched = TPURaidenOffloadingScheduler(spec, vllm_config, kv_cache_config)
    return sched, store


def make_request(req_id, hashes, num_prompt_tokens):
    req = SimpleNamespace(
        request_id=req_id,
        block_hashes=hashes[: num_prompt_tokens // TOKENS_PER_HASH],
        num_tokens=num_prompt_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        status=RequestStatus.RUNNING,
        kv_transfer_params=None,
        skip_reading_prefix_cache=False,
    )
    req.is_finished = lambda: False
    return req


class _Prefill:
    """Chunked prefill of one request, storing whatever the scheduler offers."""

    def __init__(self, sched, store, req):
        self.sched, self.store, self.req = sched, store, req
        self.next_block = 1
        sched.on_new_request(req)
        self.state = sched._req_status[req.request_id]
        self.stored = []

    def _allocate(self, upto_tokens):
        for config, gs in zip(
            self.sched.config.kv_group_configs, self.state.group_states
        ):
            need = -(-upto_tokens // config.tokens_per_block)
            while len(gs.block_ids) < need:
                gs.block_ids.append(self.next_block)
                self.next_block += 1

    # vLLM offers an EAGLE group's peek chunk only when a prefill step ends
    # near it, which steps of at most 256 tokens (the served budget) ensure.
    def run(self, step_tokens=256):
        req, sched, mgr = self.req, self.sched, self.sched._raiden_manager
        while req.num_computed_tokens < req.num_prompt_tokens:
            n = min(step_tokens, req.num_prompt_tokens - req.num_computed_tokens)
            self._allocate(req.num_computed_tokens + n)
            self.state.update_offload_keys()
            sched._admitted_window_keys.clear()
            jobs = sched._build_store_jobs(
                SimpleNamespace(
                    num_scheduled_tokens={req.request_id: n}, finished_req_ids=set()
                )
            )
            for job_id, job in jobs.items():
                keys = job.dst_spec.keys
                mgr.submit_store_job(job_id, keys, job.src_spec.block_ids.tolist())
                self.store.complete_save([h for k in keys for h in sub_hashes(k, 1)])
                self.stored.extend(keys)
            mgr.poll_finished_jobs()
            req.num_computed_tokens += n
        return self.stored


def stored_per_group(keys):
    out = [0] * len(DSV4_GROUPS)
    for k in keys:
        out[get_offload_group_idx(k)] += 1
    return out


# vLLM's own choice for a 10k prompt prefilled 256 tokens a step.
DENSE_256 = [39, 78, 9, 40, 156]
HASHES = [i.to_bytes(8, "big") for i in range(1, 4096)]
OTHER = [(10**6 + i).to_bytes(8, "big") for i in range(4096)]


class TestWindowCheckpointPolicy(unittest.TestCase):
    PROMPT = 10_000  # replay boundary 9216, 9 full chunks

    def _prefill(self, interval):
        sched, store = make_scheduler(interval)
        stored = _Prefill(sched, store, make_request("a", HASHES, self.PROMPT)).run()
        return sched, store, stored

    def _hit(self, sched, hashes, prompt, local=0, req_id="b"):
        req = make_request(req_id, hashes, prompt)
        sched.on_new_request(req)
        hit, _ = sched.get_num_new_matched_tokens(req, local)
        return hit

    def test_dense_stores_window_keys_at_every_step(self):
        _, _, stored = self._prefill(None)
        # Each prefill step stores the trailing window of the segment it ends
        # in, so windowed keys outnumber full-attention keys many times over.
        self.assertEqual(stored_per_group(stored), DENSE_256)

    def test_replay_boundary_only(self):
        _, _, stored = self._prefill(0)
        # Window keys only for the segment ending at the replay boundary.
        self.assertEqual(stored_per_group(stored), [1, 2, 9, 1, 4])

    def test_interval_adds_periodic_segments(self):
        _, _, stored = self._prefill(4096)
        # Segments ending at 4096, 8192 and the replay boundary 9216.
        self.assertEqual(stored_per_group(stored), [3, 6, 9, 3, 12])

    def test_next_turn_hits_replay_boundary_like_dense(self):
        longer = self.PROMPT + 3000
        for interval in (None, 0, 32768):
            sched, _, _ = self._prefill(interval)
            with self.subTest(interval=interval):
                self.assertEqual(self._hit(sched, HASHES, longer), 9216)
                # Extends a local (HBM) hit that stopped at an earlier checkpoint.
                self.assertEqual(
                    self._hit(sched, HASHES, longer, local=4096, req_id="c"),
                    9216 - 4096,
                )

    def test_replay_boundary_backs_off_without_peek_chunk(self):
        # 50 tokens past 9216: the EAGLE peek chunk at 9216 never completes.
        # vLLM's dense choice stops one 256-token step short of 9216; the
        # checkpoint policy backs off to the boundary before.
        prompt = 9216 + 50
        for interval, hit in ((None, 8960), (0, 8192)):
            sched, store = make_scheduler(interval)
            _Prefill(sched, store, make_request("a", HASHES, prompt)).run()
            with self.subTest(interval=interval):
                self.assertEqual(self._hit(sched, HASHES, prompt + 3000), hit)

    def test_divergent_prompt_falls_back_to_last_checkpoint(self):
        # Shares the first 5000 tokens, then diverges.
        hashes = HASHES[: 5000 // TOKENS_PER_HASH] + OTHER
        expected = {None: 4096, 0: 0, 4096: 4096, 32768: 0}
        for interval, hit in expected.items():
            sched, _, _ = self._prefill(interval)
            with self.subTest(interval=interval):
                self.assertEqual(self._hit(sched, hashes, 8000), hit)

    def test_rejects_unaligned_interval(self):
        with self.assertRaisesRegex(ValueError, "multiple of the full-attention chunk"):
            make_scheduler(1000)


if __name__ == "__main__":
    unittest.main()
