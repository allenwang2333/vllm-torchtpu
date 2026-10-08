# SPDX-License-Identifier: Apache-2.0
"""Owner-sharded KV block allocation (TPU_DEEPSEEK_V41_KV_SHARD=1).

Every block of a request in an owner-sharded group must come from its owner's
region, and every block of a replicated group from the replicated region. See
vllm_torchtpu.core.owner_affine_block_pool.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core import kv_cache_coordinator
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

import vllm_torchtpu
from vllm_torchtpu import _patch_vllm_block_pool_lifo_free
from vllm_torchtpu.core import owner_affine_block_pool as oab
from vllm_torchtpu.core.owner_affine_block_pool import (
    OwnerAffineBlockPool,
    OwnerLayout,
    owner_scope,
    replicated_scope,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
OWNERS = 8
# Replicated ids [0, 8), block 0 being the null block, then 8 owners x 8 ids.
LAYOUT = OwnerLayout(num_replicated=8, num_owners=OWNERS, blocks_per_owner=8)
NUM_BLOCKS = LAYOUT.num_blocks
REPLICATED = range(1, LAYOUT.num_replicated)


def _owner_range(owner: int) -> range:
    start = LAYOUT.num_replicated + owner * LAYOUT.blocks_per_owner
    return range(start, start + LAYOUT.blocks_per_owner)


def _spec(cls, **kwargs):
    return cls(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        **kwargs,
    )


FULL = _spec(FullAttentionSpec)
SWA = _spec(SlidingWindowSpec, sliding_window=32)
RING = _spec(CircularBufferSpec)


def _kv_config(num_blocks: int, specs) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


def _vllm_config(
    *,
    tp=OWNERS,
    pp=1,
    dcp=1,
    pcp=1,
    prefix_caching=False,
    kv_transfer=None,
    max_num_seqs=4,
    num_spec_tokens=None,
):
    spec_config = None
    if num_spec_tokens is not None:
        spec_config = SimpleNamespace(num_speculative_tokens=num_spec_tokens)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        kv_transfer_config=kv_transfer,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec_config,
        max_in_flight_tokens=64,
    )


def _request(request_id: str, num_tokens: int) -> Request:
    return Request(
        request_id, list(range(num_tokens)), SamplingParams(max_tokens=1), None
    )


def _manager(specs=(FULL, SWA, RING), watermark=0.0) -> KVCacheManager:
    with oab.layout_scope(LAYOUT):
        return KVCacheManager(
            _kv_config(NUM_BLOCKS, specs),
            max_model_len=1024,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=False,
            watermark=watermark,
        )


@pytest.fixture
def pool():
    with oab.layout_scope(LAYOUT):
        return OwnerAffineBlockPool(NUM_BLOCKS, False, BLOCK_SIZE)


@pytest.fixture
def restore(monkeypatch):
    """Undo install() after the test."""
    for name in ("__init__", "allocate_slots", "free", "pop_blocks_for_free"):
        monkeypatch.setattr(KVCacheManager, name, KVCacheManager.__dict__[name])
    monkeypatch.setattr(KVCacheManager, "_tpu_owner_affine_kv", False, raising=False)
    monkeypatch.setattr(Scheduler, "__init__", Scheduler.__dict__["__init__"])
    monkeypatch.setattr(
        kv_cache_coordinator, "BlockPool", kv_cache_coordinator.BlockPool
    )


@pytest.fixture
def installed(restore):
    oab.install()


@pytest.fixture
def v2_env(monkeypatch):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_IMPL", "v2")
    monkeypatch.setenv("TPU_DEEPSEEK_V41_SP", "0")
    monkeypatch.setenv("TPU_DEEPSEEK_V41_LOCAL_Q", "0")


def test_layout_maps_ids_to_regions_and_rank_local_ids():
    assert (LAYOUT.num_blocks, LAYOUT.blocks_per_rank) == (72, 16)
    assert LAYOUT.region_of(0) == LAYOUT.region_of(7) == OWNERS
    assert (LAYOUT.region_of(8), LAYOUT.region_of(71)) == (0, 7)
    assert LAYOUT.local_block_id(5, rank=3) == 5
    assert LAYOUT.local_block_id(_owner_range(3)[2], rank=3) == 10
    assert LAYOUT.local_block_id(_owner_range(3)[2], rank=4) == -1
    for rank in range(OWNERS):
        local = [LAYOUT.local_block_id(b, rank) for b in range(NUM_BLOCKS)]
        assert sorted(i for i in local if i >= 0) == list(range(16))
    for block_id in (-1, NUM_BLOCKS):
        with pytest.raises(ValueError, match="outside"):
            LAYOUT.region_of(block_id)


def test_layout_split_puts_the_remainder_in_the_replicated_region():
    assert OwnerLayout.split(75, OWNERS, 8) == OwnerLayout(11, OWNERS, 8)
    with pytest.raises(ValueError, match="Invalid"):
        OwnerLayout.split(10, OWNERS, 8)
    with pytest.raises(ValueError, match="at least 2 owners"):
        OwnerLayout.split(72, 1, 8)
    with pytest.raises(ValueError, match="Invalid"):
        OwnerLayout(0, OWNERS, 8)


def test_replicated_blocks_needed_bounds_window_groups():
    kv_config = _kv_config(200, (FULL, SWA, RING))
    # Null block + one ring block per sequence + per sequence a 31-token
    # window in 2 blocks plus 2, + the 64 in-flight tokens in 4 blocks.
    assert oab.replicated_blocks_needed(_vllm_config(), kv_config) == 1 + 4 + 16 + 4
    # Two lookahead tokens per sequence add 8 in-flight tokens.
    with_spec = _vllm_config(num_spec_tokens=2)
    assert oab.replicated_blocks_needed(with_spec, kv_config) == 1 + 4 + 16 + 5
    retained = _kv_config(
        200, (_spec(SlidingWindowSpec, sliding_window=32, extra_retained_tokens=16),)
    )
    assert oab.replicated_blocks_needed(_vllm_config(), retained) == 1 + 20 + 4
    assert oab.owner_layout(_vllm_config(), kv_config) == OwnerLayout(32, OWNERS, 21)


def test_pool_splits_free_blocks_by_region(pool):
    assert pool.free_block_queue is pool.region_free_queue
    assert pool.owner_free_blocks() == [8] * OWNERS
    assert pool.num_free_replicated_blocks == 7
    assert pool.get_num_free_blocks() == 71
    with owner_scope(3):
        assert pool.get_num_free_blocks() == 8
        with replicated_scope():
            assert pool.get_num_free_blocks() == 7
    assert pool.get_usage() == 0.0
    queues = pool.region_free_queue.queues
    for owner in range(OWNERS):
        ids = [b.block_id for b in queues[owner].get_all_free_blocks()]
        assert ids == list(_owner_range(owner))
    assert [b.block_id for b in queues[OWNERS].get_all_free_blocks()] == list(
        REPLICATED
    )


def test_allocation_draws_from_the_region_in_scope(pool):
    with owner_scope(5):
        owned = pool.get_new_blocks(3)
        with replicated_scope():
            shared = pool.get_new_blocks(2)
    assert {b.block_id for b in owned} <= set(_owner_range(5))
    assert {b.block_id for b in shared} <= set(REPLICATED)
    assert pool.owner_free_blocks()[5] == 5
    assert pool.num_free_replicated_blocks == 5
    assert pool.get_usage() == pytest.approx(5 / 71)
    with owner_scope(5):
        pool.get_new_blocks(5)
        # Other owners still have free blocks, but owner 5 has none.
        with pytest.raises(ValueError, match="Cannot get"):
            pool.get_new_blocks(1)


def test_allocation_outside_a_scope_raises(pool):
    with pytest.raises(RuntimeError, match="outside an owner scope"):
        pool.get_new_blocks(1)
    with pytest.raises(RuntimeError, match="outside an owner scope"):
        pool.free_block_queue.popleft()


def test_queue_operations_route_by_block_id(pool):
    queue = pool.region_free_queue
    with owner_scope(2):
        block = queue.popleft()
    assert block.block_id == _owner_range(2)[0]
    assert queue.num_free_blocks == 70
    queue.append(block)
    assert queue.queues[2].get_all_free_blocks()[-1] is block
    queue.remove(pool.blocks[25])
    assert (pool.owner_free_blocks()[2], queue.num_free_blocks) == (7, 70)
    queue.append_n([pool.blocks[25]])
    assert (pool.owner_free_blocks()[2], queue.num_free_blocks) == (8, 71)
    assert len(queue.get_all_free_blocks()) == 71
    with pytest.raises(NotImplementedError):
        queue.iter_blocks_after(None)


def test_bad_block_ids_change_nothing(pool):
    queue = pool.region_free_queue
    stranger = KVCacheBlock(NUM_BLOCKS)
    with owner_scope(1):
        good = pool.get_new_blocks(2)
    before = (pool.owner_free_blocks(), queue.num_free_blocks)
    with pytest.raises(ValueError, match="outside"):
        queue.prepend_n([*good, stranger])
    with pytest.raises(ValueError, match="outside"):
        queue.append(stranger)
    assert (pool.owner_free_blocks(), queue.num_free_blocks) == before


@pytest.mark.parametrize("plugin_lifo", [False, True])
def test_freed_blocks_return_to_their_region_lifo(pool, monkeypatch, plugin_lifo):
    """Holds for the installed free_blocks and for the plugin's LIFO one."""
    if plugin_lifo:
        monkeypatch.setattr(BlockPool, "free_blocks", BlockPool.free_blocks)
        monkeypatch.setattr(BlockPool, "_tpu_lifo_free_patch", False, raising=False)
        _patch_vllm_block_pool_lifo_free()
    with owner_scope(4):
        owned = pool.get_new_blocks(3)
    with replicated_scope():
        shared = pool.get_new_blocks(2)
    pool.free_blocks(reversed(owned + shared))
    assert pool.owner_free_blocks() == [8] * OWNERS
    assert pool.num_free_replicated_blocks == 7
    with owner_scope(4):
        assert pool.get_new_blocks(1)[0] is owned[-1]
    with replicated_scope():
        assert pool.get_new_blocks(1)[0] is shared[-1]


def test_pool_rejects_unsupported_setups():
    with pytest.raises(RuntimeError, match="needs a layout"):
        OwnerAffineBlockPool(NUM_BLOCKS, False, BLOCK_SIZE)
    with pytest.raises(ValueError, match="prefix caching"):
        OwnerAffineBlockPool(NUM_BLOCKS, True, BLOCK_SIZE, layout=LAYOUT)
    with pytest.raises(ValueError, match="does not cover"):
        OwnerAffineBlockPool(64, False, BLOCK_SIZE, layout=LAYOUT)
    pool = OwnerAffineBlockPool(NUM_BLOCKS, False, BLOCK_SIZE, layout=LAYOUT)
    assert pool.layout is LAYOUT


def test_owner_bookkeeping(pool):
    assert pool.owner_candidates() == list(range(OWNERS))
    pool.bind("a", 0)
    assert pool.owner_of("a") == 0
    with owner_scope(1):
        pool.get_new_blocks(2)
    # Fewest live requests first, then most free blocks, then the index.
    assert pool.owner_candidates() == [2, 3, 4, 5, 6, 7, 1, 0]
    with pytest.raises(ValueError, match="already has a KV owner"):
        pool.bind("a", 3)
    pool.release("a")
    pool.release("never-bound")
    assert pool.owner_of("a") is None
    assert pool.owner_candidates() == [0, 2, 3, 4, 5, 6, 7, 1]


def test_validate_config(monkeypatch, v2_env):
    oab.validate_config(_vllm_config())

    monkeypatch.setenv("TPU_DEEPSEEK_V41_IMPL", "v1")
    monkeypatch.setenv("TPU_DEEPSEEK_V41_SP", "1")
    monkeypatch.setenv("TPU_DEEPSEEK_V41_LOCAL_Q", "1")
    bad = _vllm_config(
        tp=1, pp=2, dcp=2, pcp=2, prefix_caching=True, kv_transfer=object()
    )
    with pytest.raises(ValueError) as excinfo:
        oab.validate_config(bad)
    message = str(excinfo.value)
    for problem in (
        "tensor_parallel_size",
        "pipeline parallelism",
        "decode context parallelism",
        "prefill context parallelism",
        "prefix caching",
        "KV transfer",
        "TPU_DEEPSEEK_V41_IMPL",
        "TPU_DEEPSEEK_V41_SP",
        "TPU_DEEPSEEK_V41_LOCAL_Q",
    ):
        assert problem in message


def test_request_blocks_follow_owner_and_replicated_regions(installed):
    manager = _manager()
    pool = manager.block_pool
    assert isinstance(pool, OwnerAffineBlockPool)
    request = _request("a", 40)
    assert manager.allocate_slots(request, 40) is not None
    assert pool.owner_of("a") == 0
    full, swa, ring = manager.get_block_ids("a")
    assert len(full) == len(swa) == 3 and len(ring) == 1
    assert set(full) <= set(_owner_range(0))
    assert set(swa + ring) <= set(REPLICATED)

    # A later step grows the request on the same owner.
    request.num_computed_tokens = 40
    assert manager.allocate_slots(request, 16) is not None
    full, swa, ring = manager.get_block_ids("a")
    assert len(full) == 4 and set(full) <= set(_owner_range(0))
    assert set(swa + ring) - {0} <= set(REPLICATED)

    manager.free(request)
    assert pool.owner_of("a") is None
    assert pool.owner_free_blocks() == [8] * OWNERS
    assert pool.num_free_replicated_blocks == 7


def test_full_owner_falls_through_to_the_next_candidate(installed):
    manager = _manager(specs=(FULL,))
    pool = manager.block_pool
    assert manager.allocate_slots(_request("big", 128), 128) is not None
    assert pool.owner_free_blocks()[0] == 0
    for owner in range(1, OWNERS):
        pool.bind(f"x{owner}a", owner)
        pool.bind(f"x{owner}b", owner)
    # Owner 0 has the fewest requests but no free block.
    assert pool.owner_candidates()[0] == 0
    assert manager.allocate_slots(_request("small", 16), 16) is not None
    assert pool.owner_of("small") == 1
    assert manager.get_block_ids("small")[0][0] in _owner_range(1)


def test_no_owner_fits_returns_none_without_binding(installed):
    manager = _manager(specs=(FULL,))
    pool = manager.block_pool
    for i in range(OWNERS):
        assert manager.allocate_slots(_request(f"r{i}", 64), 64) is not None
    assert [pool.owner_of(f"r{i}") for i in range(OWNERS)] == list(range(OWNERS))
    candidates = pool.owner_candidates()
    # 32 blocks are free in total, but no owner has the 5 this one needs.
    assert manager.allocate_slots(_request("big", 80), 80) is None
    assert pool.owner_of("big") is None
    assert pool.owner_free_blocks() == [4] * OWNERS
    assert pool.owner_candidates() == candidates


def test_replicated_shortage_blocks_admission(installed):
    manager = _manager(specs=(FULL, SWA))
    pool = manager.block_pool
    assert manager.allocate_slots(_request("r1", 64), 64) is not None
    assert pool.num_free_replicated_blocks == 3
    # The owner-sharded group fits on every owner, the replicated one nowhere.
    assert manager.allocate_slots(_request("r2", 64), 64) is None
    assert pool.owner_of("r2") is None
    assert pool.num_free_replicated_blocks == 3
    assert manager.allocate_slots(_request("r3", 48), 48) is not None
    assert pool.num_free_replicated_blocks == 0


def test_external_computed_blocks_use_the_replicated_region(installed):
    manager = _manager()
    ring_manager = manager.coordinator.single_type_managers[2]
    ring_manager.allocate_external_computed_blocks("x", 0, BLOCK_SIZE)
    assert ring_manager.req_to_blocks["x"][0].block_id in REPLICATED


def test_pop_blocks_for_free_releases_the_owner(installed):
    manager = _manager(specs=(FULL,))
    pool = manager.block_pool
    request = _request("a", 40)
    manager.allocate_slots(request, 40)
    blocks = manager.pop_blocks_for_free(request)
    assert pool.owner_of("a") is None
    assert pool.owner_free_blocks()[0] == 5
    pool.free_blocks(reversed(blocks))
    assert pool.owner_free_blocks()[0] == 8


def test_watermark_is_one_owners_share(installed):
    # int(0.1 * 72) = 7 blocks overall, so 1 block per owner.
    assert _manager(specs=(FULL,), watermark=0.1).watermark_blocks == 1


def test_plain_pool_is_left_alone(installed, monkeypatch):
    monkeypatch.setattr(kv_cache_coordinator, "BlockPool", BlockPool)
    manager = _manager(specs=(FULL, SWA), watermark=0.1)
    assert type(manager.block_pool) is BlockPool
    assert manager.watermark_blocks == 7
    assert "allocate_new_blocks" not in vars(
        manager.coordinator.single_type_managers[1]
    )
    first, second = _request("a", 40), _request("b", 40)
    assert manager.allocate_slots(first, 40) is not None
    manager.free(first)
    assert manager.allocate_slots(second, 40) is not None
    assert manager.pop_blocks_for_free(second)


def test_scheduler_init_validates_and_scopes_the_layout(restore, monkeypatch, v2_env):
    seen = []

    def fake_init(self, vllm_config, kv_cache_config, *args, **kwargs):
        seen.append(oab._LAYOUT.get())

    monkeypatch.setattr(Scheduler, "__init__", fake_init)
    oab.install()
    config, kv_config = _vllm_config(), _kv_config(200, (FULL, SWA, RING))
    expected = oab.owner_layout(config, kv_config)
    Scheduler.__init__(SimpleNamespace(), config, kv_config)
    Scheduler.__init__(SimpleNamespace(), vllm_config=config, kv_cache_config=kv_config)
    assert seen == [expected, expected]
    assert oab._LAYOUT.get() is None
    with pytest.raises(ValueError, match="prefix caching"):
        Scheduler.__init__(
            SimpleNamespace(), _vllm_config(prefix_caching=True), kv_config
        )
    assert len(seen) == 2


def test_install_is_idempotent(installed):
    patched = (
        Scheduler.__init__,
        KVCacheManager.__init__,
        KVCacheManager.allocate_slots,
        KVCacheManager.free,
        KVCacheManager.pop_blocks_for_free,
    )
    oab.install()
    assert kv_cache_coordinator.BlockPool is OwnerAffineBlockPool
    assert (
        Scheduler.__init__,
        KVCacheManager.__init__,
        KVCacheManager.allocate_slots,
        KVCacheManager.free,
        KVCacheManager.pop_blocks_for_free,
    ) == patched


def test_patch_installs_only_with_kv_shard(monkeypatch):
    calls = []
    monkeypatch.setattr(oab, "install", lambda: calls.append(True))
    monkeypatch.setenv("TPU_DEEPSEEK_V41_KV_SHARD", "0")
    vllm_torchtpu._patch_vllm_owner_affine_block_pool()
    assert calls == []
    monkeypatch.setenv("TPU_DEEPSEEK_V41_KV_SHARD", "1")
    vllm_torchtpu._patch_vllm_owner_affine_block_pool()
    assert calls == [True]
