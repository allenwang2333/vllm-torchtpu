# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import MagicMock

import pytest

from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import (MLATuningKey,
                                                       get_mla_kernel_config)


def _k3_mixed_key() -> MLATuningKey:
    return MLATuningKey(
        device_name="TPU v7",
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=128,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size=256,
    )


def test_k3_mixed_tuned_params() -> None:
    tuned = tuned_params.mla_tuned_params_mapping[_k3_mixed_key()]

    assert tuned.num_kv_pages_per_block == 4
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 1


def test_k3_kernel_config_combines_decode_and_mixed_slots(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    config = get_mla_kernel_config(
        device_name="TPU v7",
        max_num_tokens=4096,
        local_num_q_heads=12,
        global_num_q_heads=96,
        actual_lkv_dim=128,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=128,
        kv_packing=2,
        max_num_seqs=4,
        pages_per_seq=128,
    )

    # Decode has no K3-specific entry yet and keeps the historical fallback.
    assert config.num_kv_pages_per_block == (3, 1, 4)
    assert config.num_queries_per_block == (1, 16, 512)
    assert config.decode_batch_size == 4
    assert config.mixed_q_split == 1
    assert config.vmem_limit_bytes == 62914560


def test_k3_mixed_config_requires_matching_page_size(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    config = get_mla_kernel_config(
        device_name="TPU v7",
        max_num_tokens=4096,
        local_num_q_heads=12,
        global_num_q_heads=96,
        actual_lkv_dim=128,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=64,
        kv_packing=2,
        max_num_seqs=4,
        pages_per_seq=256,
    )

    assert config.num_kv_pages_per_block == (3, 1, 1)
    assert config.num_queries_per_block == (1, 16, 16)


def test_global_head_count_entry_remains_compatible() -> None:
    config = get_mla_kernel_config(
        device_name="TPU v7",
        max_num_tokens=16,
        local_num_q_heads=8,
        global_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="float8_e4m3fn",
        kv_dtype="float8_e4m3fn",
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    )

    assert config.num_kv_pages_per_block == (2, 1, 1)
    assert config.num_queries_per_block == (1, 16, 64)
    assert config.decode_batch_size == 16
    assert config.mixed_q_split == 16
