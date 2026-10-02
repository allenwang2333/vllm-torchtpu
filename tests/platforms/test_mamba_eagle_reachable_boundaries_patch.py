# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu import _patch_vllm_mamba_eagle_reachable_boundaries

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _restore_reachable_block_mask():
    original = MambaManager.__dict__["reachable_block_mask"]
    try:
        yield
    finally:
        MambaManager.reachable_block_mask = original


def _spec() -> MambaSpec:
    return MambaSpec(
        block_size=1152,
        shapes=((4, 4),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )


def _mask(use_eagle: bool, retention_interval: int | None):
    return MambaManager.reachable_block_mask(
        start_block=0,
        end_block=9,
        alignment_tokens=1152,
        kv_cache_spec=_spec(),
        use_eagle=use_eagle,
        retention_interval=retention_interval,
        reachable_boundaries=[9999],
    )


def test_sparse_eagle_keeps_shifted_replay_boundary():
    _patch_vllm_mamba_eagle_reachable_boundaries()

    mask = _mask(use_eagle=True, retention_interval=0)

    # Replay boundary 9999 -> state at 9216 (block 7); EAGLE hits land one
    # alignment unit earlier at 8064 (block 6).
    assert [i for i, keep in enumerate(mask) if keep] == [6, 7]


def test_sparse_without_eagle_is_unchanged():
    _patch_vllm_mamba_eagle_reachable_boundaries()

    mask = _mask(use_eagle=False, retention_interval=0)

    assert [i for i, keep in enumerate(mask) if keep] == [7]


def test_dense_retention_is_unchanged():
    _patch_vllm_mamba_eagle_reachable_boundaries()

    assert _mask(use_eagle=True, retention_interval=None) is None


def test_patch_is_idempotent():
    _patch_vllm_mamba_eagle_reachable_boundaries()
    patched = MambaManager.__dict__["reachable_block_mask"]

    _patch_vllm_mamba_eagle_reachable_boundaries()

    assert MambaManager.__dict__["reachable_block_mask"] is patched
