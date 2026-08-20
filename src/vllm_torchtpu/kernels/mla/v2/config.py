# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from vllm_torchtpu.kernels.mla.v2.tuned_params import (TunableParams,
                                                       TuningKey,
                                                       get_tuned_params,
                                                       tuned_params_mapping)


@dataclass(frozen=True)
class MLAKernelConfig:
    """Complete MLA launch configuration for decode and mixed programs."""

    num_kv_pages_per_block: tuple[int, int, int]
    num_queries_per_block: tuple[int, int, int]
    vmem_limit_bytes: int
    decode_batch_size: int
    mixed_q_split: int


# Kimi-K3 MLA mixed extend, TP=8 on TPU v7x-8. This is the post-shard-map
# shape (96 / 8 = 12 local Q heads) for cold 4K prefill with a BF16 cache.
# Falcon exp-tn7ngntmg4 selected page_size=256, bkv=4, bq=512. Cache-hit and
# long-KV serving remain to be validated.
MLA_TUNED_CONFIGS: dict[str, dict[TuningKey, TunableParams]] = {
    "TPU v7": {
        TuningKey(
            case="mixed",
            max_num_tokens=4096,
            actual_num_q_heads=12,
            actual_lkv_dim=128,
            actual_r_dim=64,
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            page_size_per_kv_packing=128,
            kv_packing=2,
            max_num_seqs=4,
            pages_per_seq=128,
        ):
        TunableParams(
            num_kv_pages_per_block=4,
            num_queries_per_block=512,
            vmem_limit_bytes=62914560,
            q_split=1,
        ),
    },
}


def _get_legacy_tuned_params(
        candidates: tuple[TuningKey, ...]) -> TunableParams:
    for candidate in candidates:
        if candidate in tuned_params_mapping:
            return tuned_params_mapping[candidate]
    return get_tuned_params(candidates[0])


def get_mla_kernel_config(
    *,
    device_name: str,
    max_num_tokens: int,
    local_num_q_heads: int,
    global_num_q_heads: int,
    actual_lkv_dim: int,
    actual_r_dim: int,
    q_dtype: str,
    kv_dtype: str,
    page_size_per_kv_packing: int,
    kv_packing: int,
    max_num_seqs: int,
    pages_per_seq: int,
) -> MLAKernelConfig:
    """Resolve decode and mixed launch parameters for one compiled shape."""
    common = dict(
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=local_num_q_heads,
        actual_lkv_dim=actual_lkv_dim,
        actual_r_dim=actual_r_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=kv_packing,
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
    )
    global_common = common | {"actual_num_q_heads": global_num_q_heads}
    legacy_common = dict(
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=global_num_q_heads,
        actual_lkv_dim=actual_lkv_dim,
        actual_r_dim=actual_r_dim,
        kv_dtype=kv_dtype,
    )

    def _lookup(case: str) -> TunableParams:
        exact_key = TuningKey(case=case, **common)
        if tuned := MLA_TUNED_CONFIGS.get(device_name, {}).get(exact_key):
            return tuned
        return _get_legacy_tuned_params((
            exact_key,
            TuningKey(case=case, **global_common),
            TuningKey(case=case, **legacy_common),
        ))

    decode = _lookup("batched_decode")
    mixed = _lookup("mixed")
    return MLAKernelConfig(
        num_kv_pages_per_block=(decode.num_kv_pages_per_block, 1,
                                mixed.num_kv_pages_per_block),
        num_queries_per_block=(decode.num_queries_per_block, 16,
                               mixed.num_queries_per_block),
        vmem_limit_bytes=min(decode.vmem_limit_bytes, mixed.vmem_limit_bytes),
        decode_batch_size=decode.decode_batch_size,
        mixed_q_split=mixed.q_split,
    )
