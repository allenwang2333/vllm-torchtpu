# SPDX-License-Identifier: Apache-2.0
"""TD8.78: vLLM-version import fallbacks and CED owner-balanced padding.

The import fallbacks run at module import time, so they are exercised by
re-executing the module source with ``runpy.run_path`` into a throwaway
namespace against injected fake upstream modules. Unlike
``importlib.reload``, that leaves the imported module objects (and the v1/v2
subclasses bound to their classes) untouched.
"""

import runpy
import sys
import types
from types import SimpleNamespace

import torch
import vllm.models.deepseek_v41.common as upstream_common_pkg
import vllm.models.deepseek_v41.nvidia as upstream_nvidia_pkg

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced, host_engram

_COMMON = "vllm.models.deepseek_v41.common.engram"
_NVIDIA = "vllm.models.deepseek_v41.nvidia.engram"
_FALLBACK_NAMES = (
    "engram_gathered_num_tokens",
    "engram_head_shard_rank",
    "gather_engram_hashes",
)


def _copy_upstream(*, drop=()):
    real = sys.modules[_COMMON]
    fake = types.ModuleType(_COMMON)
    for key, value in vars(real).items():
        if not key.startswith("__") and key not in drop:
            setattr(fake, key, value)
    return fake


def _install(monkeypatch, *, common, nvidia):
    monkeypatch.setitem(sys.modules, _COMMON, common)
    monkeypatch.setattr(upstream_common_pkg, "engram", common)
    monkeypatch.setitem(sys.modules, _NVIDIA, nvidia)
    if nvidia is None:  # `from pkg import engram` must not find a cached attr
        monkeypatch.delattr(upstream_nvidia_pkg, "engram", raising=False)
    else:
        monkeypatch.setattr(upstream_nvidia_pkg, "engram", nvidia, raising=False)


def test_common_engram_v031_without_nvidia_gather(monkeypatch):
    """v0.31: symbols come from common.engram; with no nvidia module the
    missing ``gather_engram_hashes`` is left unset rather than failing."""
    fake = _copy_upstream(drop=("gather_engram_hashes",))
    fake.engram_gathered_num_tokens = "v031_tokens"
    fake.engram_head_shard_rank = "v031_rank"
    _install(monkeypatch, common=fake, nvidia=None)
    ns = runpy.run_path(common_engram.__file__, run_name="_td878_common_v031")
    assert ns["engram_gathered_num_tokens"] == "v031_tokens"
    assert ns["engram_head_shard_rank"] == "v031_rank"
    assert not hasattr(fake, "gather_engram_hashes")
    assert fake.engram_gathered_num_tokens == "v031_tokens"


def test_common_engram_v030_backfills_from_nvidia(monkeypatch):
    """v0.30: symbols live in nvidia.engram and are back-filled upstream."""
    fake = _copy_upstream(drop=_FALLBACK_NAMES)
    nvidia = types.ModuleType(_NVIDIA)
    nvidia.engram_gathered_num_tokens = "v030_tokens"
    nvidia.engram_head_shard_rank = "v030_rank"
    nvidia.gather_engram_hashes = "v030_gather"
    _install(monkeypatch, common=fake, nvidia=nvidia)
    ns = runpy.run_path(common_engram.__file__, run_name="_td878_common_v030")
    assert ns["engram_gathered_num_tokens"] == "v030_tokens"
    assert ns["engram_head_shard_rank"] == "v030_rank"
    assert fake.gather_engram_hashes == "v030_gather"
    assert fake.engram_gathered_num_tokens == "v030_tokens"
    # The imported module itself is untouched (no reload).
    assert (
        sys.modules["vllm_torchtpu.models.vllm.deepseek_v41.common.engram"]
        is common_engram
    )


def test_host_engram_v030_imports_from_nvidia(monkeypatch):
    fake = _copy_upstream(drop=_FALLBACK_NAMES)
    nvidia = types.ModuleType(_NVIDIA)
    nvidia.engram_gathered_num_tokens = "v030_tokens"
    _install(monkeypatch, common=fake, nvidia=nvidia)
    ns = runpy.run_path(host_engram.__file__, run_name="_td878_host_v030")
    assert ns["engram_gathered_num_tokens"] == "v030_tokens"


def test_host_engram_gather_falls_back_to_nvidia(monkeypatch):
    fake = _copy_upstream(drop=_FALLBACK_NAMES)
    seen = []

    def gather(hashes, *, dp_shared_memory):
        seen.append(dp_shared_memory)
        return hashes * 2

    nvidia = types.ModuleType(_NVIDIA)
    nvidia.gather_engram_hashes = gather
    _install(monkeypatch, common=fake, nvidia=nvidia)
    hashes = torch.arange(6).view(3, 2)
    out = host_engram.gather_host_engram_hashes(hashes, dp_shared_memory=True)
    torch.testing.assert_close(out, hashes * 2)
    assert seen == [True]


def _ced_metadata(owners, tp_size):
    return SimpleNamespace(
        query_start_loc_cpu=[0, 3, 4],
        seq_lens_cpu=[3, 1],
        prompt_lens_cpu=[3, 1],
        owner_token_layout=SimpleNamespace(
            owners=owners, query_start_loc=torch.zeros(tp_size)
        ),
    )


def test_ced_plan_pads_for_the_busiest_owner():
    # Both requests (3 + 1 kept tokens) land on owner 0 of 2, so every owner
    # slot must hold 4 rows: 2 * 4 = 8, the smallest compile size >= 8.
    plan = ced.plan_decoder_replay(_ced_metadata([0, 0], 2), 128, 16, (4, 8, 16))
    assert plan.indices == (0, 1, 2, 3)
    assert plan.query_start_loc == (0, 3, 4)
    assert plan.padded_num_tokens == 8
    assert plan.is_identity is False


def test_ced_plan_balanced_owners_keep_the_tight_bucket():
    # Owners 0 and 1 hold 3 and 1 rows: max(4, 2 * 3) = 6 -> bucket 8; with
    # no owners listed the requirement stays at kept=4 -> bucket 4.
    assert (
        ced.plan_decoder_replay(
            _ced_metadata([0, 1], 2), 128, 16, (4, 8, 16)
        ).padded_num_tokens
        == 8
    )
    assert (
        ced.plan_decoder_replay(
            _ced_metadata([], 2), 128, 16, (4, 8, 16)
        ).padded_num_tokens
        == 4
    )
