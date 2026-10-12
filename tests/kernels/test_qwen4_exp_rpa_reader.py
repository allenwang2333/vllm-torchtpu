# SPDX-License-Identifier: Apache-2.0
"""Explicit reader overrides must reject modes outside their ownership contract."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.kernels.qwen4_exp.rpa_reader import QwenDecodeKVRef


@pytest.mark.parametrize(
    "unsupported",
    [
        {"all_decode": False},
        {"skip_kv_update": False},
        {"kv_layout": configs.KVLayout.SEQ_ALONG_LANE},
        {"cp_group_size": 2},
        {"return_lse": True},
    ],
)
def test_reader_override_rejects_unsupported_modes(unsupported):
    args = (
        jax.ShapeDtypeStruct((8, 24, 256), jnp.float32),
        jax.ShapeDtypeStruct((8, 2, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((8, 2, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((4, 4, 1, 4, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((8,), jnp.int32),
        jax.ShapeDtypeStruct((8 * 513,), jnp.int32),
        jax.ShapeDtypeStruct((9,), jnp.int32),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )
    options = dict(
        all_decode=True,
        skip_kv_update=True,
        kv_cache_alloc_cls=QwenDecodeKVRef,
    )
    options.update(unsupported)
    with pytest.raises(ValueError, match="KV allocator override requires"):
        jax.eval_shape(wrapper.ragged_paged_attention, *args, **options)
