# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for GDN v3 VMEM estimation and dynamic tile selection."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.gdn.v3 import tiling

_VMEM_CAPACITY_64MIB = 64 * 1024 * 1024
_NUM_LANES = 128
_D_K = 128
_D_V = 128
_KERNEL_SIZE = 4


def _dim(n_kq: int, n_v: int) -> int:
    return 2 * n_kq * _D_K + n_v * _D_V


def _derived_expected_picks(
    batch_size: int,
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    rec_dtype: jnp.dtype,
    vmem_limit_bytes: int,
    window_size: int,
) -> tuple[int, int]:
    """Derives expected (T, C) from get_vmem_estimate_bytes and target."""
    dim = _dim(n_kq, n_v)
    rec_bytes = jnp.dtype(rec_dtype).itemsize

    max_decode_b = min(padded_batch_size, num_seqs)
    if max_decode_b <= 0:
        vmem_fit_t = 1
    else:
        cands_t = [c for c in tiling.DECODE_TILE_SIZES if c <= max_decode_b]
        vmem_fit_t = cands_t[-1]
        for cand in cands_t:
            est = tiling.get_vmem_estimate_bytes(
                tile_b=cand,
                chunk_sz=1,
                n_kq=n_kq,
                n_v=n_v,
                d_k=_D_K,
                d_v=_D_V,
                kernel_size=_KERNEL_SIZE,
                act_out_bytes=2,
                rec_state_bytes=rec_bytes,
                num_lanes=_NUM_LANES,
                conv_state_dim_size=dim,
                is_decode=True,
                window_size=window_size,
            )
            if est <= vmem_limit_bytes:
                vmem_fit_t = cand
                break

    knee = tiling.decode_tile_target(num_seqs, n_v)
    exp_t = max(1, min(min(knee, vmem_fit_t), padded_batch_size))

    if batch_size <= 0:
        vmem_fit_c = 1
    else:
        cands_c = [c for c in tiling.MIXED_TILE_SIZES if c <= batch_size]
        vmem_fit_c = cands_c[-1]
        for cand in cands_c:
            est = tiling.get_vmem_estimate_bytes(
                tile_b=1,
                chunk_sz=cand,
                n_kq=n_kq,
                n_v=n_v,
                d_k=_D_K,
                d_v=_D_V,
                kernel_size=_KERNEL_SIZE,
                act_out_bytes=2,
                rec_state_bytes=rec_bytes,
                num_lanes=_NUM_LANES,
                conv_state_dim_size=dim,
                is_decode=False,
            )
            if est <= vmem_limit_bytes:
                vmem_fit_c = cand
                break
    exp_c = max(1, min(vmem_fit_c, batch_size))
    return exp_t, exp_c


@pytest.mark.parametrize("n_kq,n_v", [(2, 8), (8, 32), (16, 64)])
@pytest.mark.parametrize("num_seqs", [8, 16, 32, 64])
@pytest.mark.parametrize("batch_size", [1, 8, 64, 128, 16384])
@pytest.mark.parametrize("window_size", [1, 4])
@pytest.mark.parametrize("rec_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("fraction", [0.7, 0.8])
def test_get_tile_sizes_invariants_and_monotonicity(
    n_kq: int,
    n_v: int,
    num_seqs: int,
    batch_size: int,
    window_size: int,
    rec_dtype: jnp.dtype,
    fraction: float,
) -> None:
    dim = _dim(n_kq, n_v)
    padded_batch_size = ((batch_size + 1) // 2) * 2
    vmem_limit = int(fraction * _VMEM_CAPACITY_64MIB)
    rec_bytes = jnp.dtype(rec_dtype).itemsize

    t, c = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=_KERNEL_SIZE,
        conv_state_dim_size=dim,
        act_out_dtype=jnp.bfloat16,
        recurrent_state_dtype=rec_dtype,
        num_lanes=_NUM_LANES,
        vmem_limit_bytes=vmem_limit,
        window_size=window_size,
    )

    exp_t, exp_c = _derived_expected_picks(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        rec_dtype=rec_dtype,
        vmem_limit_bytes=vmem_limit,
        window_size=window_size,
    )
    assert (t, c) == (exp_t, exp_c)
    assert t in tiling.DECODE_TILE_SIZES
    assert c in tiling.MIXED_TILE_SIZES
    assert t != 32
    assert t <= min(padded_batch_size, num_seqs)
    assert c <= batch_size

    if t > 1:
        assert (
            tiling.get_vmem_estimate_bytes(
                tile_b=t,
                chunk_sz=1,
                n_kq=n_kq,
                n_v=n_v,
                d_k=_D_K,
                d_v=_D_V,
                kernel_size=_KERNEL_SIZE,
                act_out_bytes=2,
                rec_state_bytes=rec_bytes,
                num_lanes=_NUM_LANES,
                conv_state_dim_size=dim,
                is_decode=True,
                window_size=window_size,
            )
            <= vmem_limit
        )
    if c > 1:
        assert (
            tiling.get_vmem_estimate_bytes(
                tile_b=1,
                chunk_sz=c,
                n_kq=n_kq,
                n_v=n_v,
                d_k=_D_K,
                d_v=_D_V,
                kernel_size=_KERNEL_SIZE,
                act_out_bytes=2,
                rec_state_bytes=rec_bytes,
                num_lanes=_NUM_LANES,
                conv_state_dim_size=dim,
                is_decode=False,
            )
            <= vmem_limit
        )

    # Monotonicity in vmem_limit_bytes: larger limit never shrinks (t, c).
    t_hi, c_hi = tiling.get_tile_sizes(
        batch_size=batch_size,
        num_seqs=num_seqs,
        padded_batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=_KERNEL_SIZE,
        conv_state_dim_size=dim,
        act_out_dtype=jnp.bfloat16,
        recurrent_state_dtype=rec_dtype,
        num_lanes=_NUM_LANES,
        vmem_limit_bytes=int(0.9 * _VMEM_CAPACITY_64MIB),
        window_size=window_size,
    )
    assert t_hi >= t
    assert c_hi >= c


def test_get_tile_sizes_boundary_clamps() -> None:
    dim = _dim(2, 8)
    vmem_limit = int(0.7 * _VMEM_CAPACITY_64MIB)
    t, c = tiling.get_tile_sizes(
        batch_size=0,
        num_seqs=0,
        padded_batch_size=0,
        n_kq=2,
        n_v=8,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=_KERNEL_SIZE,
        conv_state_dim_size=dim,
        act_out_dtype=jnp.bfloat16,
        recurrent_state_dtype=jnp.float32,
        num_lanes=_NUM_LANES,
        vmem_limit_bytes=vmem_limit,
    )
    assert (t, c) == (1, 1)
