# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import (TuningKey,
                                                       get_tuned_params,
                                                       tuned_params_mapping)
from vllm_torchtpu.layers.common import attention_interface
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata


def _k3_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=128,
        kv_packing=2,
        max_num_seqs=8,
        pages_per_seq=128,
    )


def _k3_tp32_fp8_8k_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=8192,
        actual_num_q_heads=3,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        page_size_per_kv_packing=4,
        kv_packing=4,
        max_num_seqs=8,
        pages_per_seq=520,
    )


def _k3_tp32_fp8_8k_page256_mixed_key() -> TuningKey:
    return replace(
        _k3_tp32_fp8_8k_mixed_key(),
        page_size_per_kv_packing=64,
        pages_per_seq=40,
    )


def test_k3_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_mixed_key())

    assert tuned.num_kv_pages_per_block == 4
    assert tuned.num_queries_per_block == 256
    assert tuned.q_split == 1


def test_k3_tp32_fp8_8k_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_mixed_key())

    assert tuned.num_kv_pages_per_block == 256
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 4
    assert tuned.vmem_limit_bytes == 56 * 1024 * 1024


def test_k3_tp32_fp8_8k_page256_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_page256_mixed_key())

    assert tuned.num_kv_pages_per_block == 16
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 8
    assert tuned.vmem_limit_bytes == 52 * 1024 * 1024


def test_k3_mixed_tuned_params_require_matching_geometry(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    mismatched_key = replace(_k3_mixed_key(), page_size_per_kv_packing=64)

    tuned = get_tuned_params(mismatched_key)

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 16
    assert tuned.q_split == 1


# Kimi-K3 MLA geometry, per shard: kv_lora_rank 512, qk_rope_head_dim 64.
# `qk_nope_head_dim` is 128 and is deliberately absent -- it is not a dimension
# the v2 kernel ever sees.
K3_LKV_DIM = 512
K3_ROPE_DIM = 64


def _call_site_keys(monkeypatch: pytest.MonkeyPatch, *, max_num_tokens: int,
                    num_q_heads: int, kv_dtype, page_size_per_kv_packing: int,
                    kv_packing: int, max_num_seqs: int,
                    pages_per_seq: int) -> dict[str, TuningKey]:
    """The keys `mla_attention` really builds, traced on CPU over a stub kernel.

    Reading them out of the call site rather than re-listing the fields is the
    point: a hand-written key agrees with the call site by construction, which
    is exactly how `actual_lkv_dim=qk_nope_head_dim` survived in the decode key.
    """
    seen: dict[str, TuningKey] = {}
    real_get = attention_interface.get_tuned_params

    def spy(key: TuningKey):
        seen[key.case] = key
        return real_get(key)

    def stub_kernel(q, q_rope, k, k_rope, cache, *args, **kwargs):
        del q_rope, k, k_rope, args, kwargs
        return jnp.zeros(q.shape, q.dtype), cache

    monkeypatch.setattr(attention_interface, "get_tuned_params", spy)
    monkeypatch.setattr(attention_interface, "mla_ragged_paged_attention",
                        stub_kernel)

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model", ))
    attention_interface.mla_attention(
        jnp.zeros((max_num_tokens, num_q_heads, K3_LKV_DIM), jnp.bfloat16),
        jnp.zeros((max_num_tokens, num_q_heads, K3_ROPE_DIM), jnp.bfloat16),
        jnp.zeros((max_num_tokens, K3_LKV_DIM), jnp.bfloat16),
        jnp.zeros((max_num_tokens, K3_ROPE_DIM), jnp.bfloat16),
        jnp.zeros((16, page_size_per_kv_packing, kv_packing,
                   K3_LKV_DIM + K3_ROPE_DIM), kv_dtype),
        AttentionMetadata(
            input_positions=None,
            block_tables=jnp.zeros((max_num_seqs * pages_per_seq, ),
                                   jnp.int32),
            seq_lens=jnp.zeros((max_num_seqs, ), jnp.int32),
            query_start_loc=jnp.zeros((max_num_seqs + 1, ), jnp.int32),
            request_distribution=jnp.zeros((3, ), jnp.int32),
        ),
        mesh,
        sm_scale=1.0)
    return seen


# Every K3 serving shape this branch ships a tuned row or a recipe for.
K3_GEOMETRIES = {
    # tuned_params.py:1110 -- TP8, bf16, 4K chunk.
    "tp8-bf16-4k":
    dict(max_num_tokens=4096,
         num_q_heads=12,
         kv_dtype=jnp.bfloat16,
         page_size_per_kv_packing=128,
         kv_packing=2,
         max_num_seqs=8,
         pages_per_seq=128),
    # tuned_params.py:1131 -- TP32, fp8 KV, 8K chunk, 16-token pages (PR #598).
    "tp32-fp8-8k-page16":
    dict(max_num_tokens=8192,
         num_q_heads=3,
         kv_dtype=jnp.float8_e4m3fn,
         page_size_per_kv_packing=4,
         kv_packing=4,
         max_num_seqs=8,
         pages_per_seq=520),
    # tuned_params.py:1157 -- same, 256-token pages.
    "tp32-fp8-8k-page256":
    dict(max_num_tokens=8192,
         num_q_heads=3,
         kv_dtype=jnp.float8_e4m3fn,
         page_size_per_kv_packing=64,
         kv_packing=4,
         max_num_seqs=8,
         pages_per_seq=40),
    # configs/kimi-k3-sliced8-tp8-ep.sh as measured on dev build #289: no row.
    "tp8-sliced8-bf16-512":
    dict(max_num_tokens=512,
         num_q_heads=12,
         kv_dtype=jnp.bfloat16,
         page_size_per_kv_packing=8,
         kv_packing=2,
         max_num_seqs=8,
         pages_per_seq=576),
    # configs/kimi-k3-tp32-ep.sh, the full 93-layer perf recipe, as measured on
    # dev build #289: bf16 KV, so PR #598's fp8 rows cannot apply. No row.
    "tp32-full-bf16-512":
    dict(max_num_tokens=512,
         num_q_heads=3,
         kv_dtype=jnp.bfloat16,
         page_size_per_kv_packing=8,
         kv_packing=2,
         max_num_seqs=8,
         pages_per_seq=576),
}

# The named gap. `batched_decode` is uncovered for *every* K3 shape, so K3
# decode -- the whole output phase -- runs on default block sizes today.
# Tuning a shape means adding its case here; losing one turns this red.
K3_TUNED_CASES = {
    "tp8-bf16-4k": {"mixed"},
    "tp32-fp8-8k-page16": {"mixed"},
    "tp32-fp8-8k-page256": {"mixed"},
    "tp8-sliced8-bf16-512": set(),
    "tp32-full-bf16-512": set(),
}


@pytest.mark.parametrize("name", sorted(K3_GEOMETRIES))
def test_k3_tuning_coverage_from_the_call_site(
        name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    keys = _call_site_keys(monkeypatch, **K3_GEOMETRIES[name])

    covered = {
        case
        for case, key in keys.items() if key in tuned_params_mapping
    }
    assert covered == K3_TUNED_CASES[name]


@pytest.mark.parametrize("name", sorted(K3_GEOMETRIES))
def test_both_cases_key_on_the_arrays_the_kernel_receives(
        name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """One kernel call, one set of arrays: `case` is the only field that differs.

    The decode key used to carry `actual_lkv_dim=qk_nope_head_dim` (128) and
    dataclass defaults for the five serving-configuration fields, so no decode
    row could ever match for any DeepSeek-style MLA model. The expected key is
    spelled out from the geometry we built the arrays from, not from the call
    site, so it cannot agree with the call site by construction.
    """
    geometry = K3_GEOMETRIES[name]
    keys = _call_site_keys(monkeypatch, **geometry)

    assert keys["batched_decode"] == replace(keys["mixed"],
                                             case="batched_decode")
    assert keys["mixed"] == TuningKey(
        case="mixed",
        max_num_tokens=geometry["max_num_tokens"],
        actual_num_q_heads=geometry["num_q_heads"],
        actual_lkv_dim=K3_LKV_DIM,
        actual_r_dim=K3_ROPE_DIM,
        q_dtype="bfloat16",
        kv_dtype=jnp.dtype(geometry["kv_dtype"]).name,
        page_size_per_kv_packing=geometry["page_size_per_kv_packing"],
        kv_packing=geometry["kv_packing"],
        max_num_seqs=geometry["max_num_seqs"],
        pages_per_seq=geometry["pages_per_seq"],
    )


def test_a_missed_key_is_recorded_for_the_end_of_warmup_summary(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    monkeypatch.setattr(tuned_params, "_missed_keys", {})
    monkeypatch.setattr(tuned_params, "_hit_keys", {})
    missing = replace(_k3_mixed_key(), page_size_per_kv_packing=64)

    get_tuned_params(missing)
    get_tuned_params(missing)
    get_tuned_params(_k3_mixed_key())

    assert tuned_params.missed_tuning_keys() == {missing: 2}
    assert tuned_params.report_missed_tuning_keys() == 1
    # One warning for the first miss, one for the summary -- not one per trace.
    assert tuned_params.logger.warning.call_count == 2


def test_strict_mode_refuses_the_untuned_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuned_params.envs, "TPU_MLA_TUNING_STRICT", True)
    monkeypatch.setattr(tuned_params, "_missed_keys", {})

    with pytest.raises(tuned_params.TuningKeyMissError):
        get_tuned_params(replace(_k3_mixed_key(), page_size_per_kv_packing=64))
