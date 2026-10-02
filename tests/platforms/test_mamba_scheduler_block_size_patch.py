# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import _patch_vllm_mamba_split_scheduler_block_size


@pytest.fixture(autouse=True)
def _restore_scheduler_split_method():
    original_split = Scheduler._mamba_block_aligned_split
    try:
        yield
    finally:
        Scheduler._mamba_block_aligned_split = original_split


def _request(prompt_len: int):
    return SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        shared_prefix_boundary=0,
    )


def _scheduler():
    physical_cache_config = SimpleNamespace(block_size=768)
    scheduler = SimpleNamespace(
        cache_config=physical_cache_config,
        block_size=3072,
        use_eagle=True,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        hash_block_size=3072,
    )
    return scheduler, physical_cache_config


def test_mamba_split_uses_scheduler_block_size_and_restores_cache_config():
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler, physical_cache_config = _scheduler()

    split = Scheduler._mamba_block_aligned_split(
        scheduler,
        _request(prompt_len=8755),
        num_new_tokens=8755,
    )

    assert split == 3072
    assert scheduler.cache_config is physical_cache_config
    assert scheduler.cache_config.block_size == 768


def test_mamba_split_restores_cache_config_when_upstream_raises(monkeypatch):
    def raising_split(self, *args, **kwargs):
        raise RuntimeError("upstream split failed")

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", raising_split)
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler, physical_cache_config = _scheduler()

    with pytest.raises(RuntimeError, match="upstream split failed"):
        Scheduler._mamba_block_aligned_split(
            scheduler,
            _request(prompt_len=8755),
            num_new_tokens=8755,
        )

    assert scheduler.cache_config is physical_cache_config
    assert scheduler.cache_config.block_size == 768


def test_mamba_split_patch_is_idempotent():
    _patch_vllm_mamba_split_scheduler_block_size()
    patched_split = Scheduler._mamba_block_aligned_split

    _patch_vllm_mamba_split_scheduler_block_size()

    assert Scheduler._mamba_block_aligned_split is patched_split


def _dcp_scheduler(use_eagle: bool, dcp_world_size: int, pcp_world_size: int = 1):
    # Mirrors Qwen3.5 DCP4 on TPU: 1152-token physical blocks, a 4608-token
    # DCP-scaled scheduler block, and fine-grained (1152) prefix hashes.
    physical_cache_config = SimpleNamespace(block_size=1152)
    return SimpleNamespace(
        cache_config=physical_cache_config,
        block_size=1152 * dcp_world_size * pcp_world_size,
        use_eagle=use_eagle,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
        hash_block_size=1152,
    )


def test_dcp_eagle_split_backs_off_one_physical_block():
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler = _dcp_scheduler(use_eagle=True, dcp_world_size=4)

    split = Scheduler._mamba_block_aligned_split(
        scheduler, _request(prompt_len=10000), num_new_tokens=10000
    )

    # Full attention's EAGLE drop lands one hash unit before the prompt's last
    # hash boundary (9216 - 1152), so the Mamba state must be checkpointed
    # there, not one 4608-token scheduler block back (4608).
    assert split == 8064
    assert scheduler.cache_config.block_size == 1152


def test_pcp_eagle_split_still_uses_scheduler_block_size():
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler = _dcp_scheduler(use_eagle=True, dcp_world_size=1, pcp_world_size=4)

    split = Scheduler._mamba_block_aligned_split(
        scheduler, _request(prompt_len=10000), num_new_tokens=10000
    )

    assert split == 4608
