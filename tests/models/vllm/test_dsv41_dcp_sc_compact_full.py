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

"""Full statement and branch coverage tests for dcp_sc_compact.py."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import dcp_sc_compact


class _SliceAtProxy:
    def __init__(self, ref: "_SliceArrayRef"):
        self.ref = ref

    def __getitem__(self, idx):
        return _SliceArrayRef(self.ref.arr, self.ref.base_offset + int(idx.start))


class _SliceArrayRef:
    """NumPy-backed slice reference for exercising SparseCore Pallas body."""

    def __init__(self, arr: np.ndarray, base_offset: int = 0):
        self.arr = arr
        self.base_offset = base_offset

    @property
    def at(self) -> _SliceAtProxy:
        return _SliceAtProxy(self)

    def _to_slice(self, idx):
        if hasattr(idx, "start") and hasattr(idx, "size"):
            start = int(idx.start) + self.base_offset
            return slice(start, start + int(idx.size))
        return idx

    def __getitem__(self, idx):
        s = self._to_slice(idx)
        return jnp.asarray(self.arr[s])

    def __setitem__(self, idx, val):
        s = self._to_slice(idx)
        self.arr[s] = np.asarray(val)


class TestDcpScCompactFull:
    """100% statement and branch coverage tests for dcp_sc_compact.py."""

    def test_pack_nonnegative_validation_errors(self):
        with pytest.raises(ValueError, match="vals must be int32"):
            dcp_sc_compact.pack_nonnegative(jnp.zeros((2, 8), dtype=jnp.float32))

        with (
            patch.object(
                dcp_sc_compact.pltpu,
                "get_tpu_info",
                return_value=SimpleNamespace(sparse_core=None),
            ),
            pytest.raises(NotImplementedError, match="SparseCore is not available"),
        ):
            dcp_sc_compact.pack_nonnegative(jnp.zeros((2, 8), dtype=jnp.int32))

        fake_sc = SimpleNamespace(num_lanes=8, num_cores=1, num_subcores=2)
        fake_tpu = SimpleNamespace(sparse_core=fake_sc)
        with (
            patch.object(dcp_sc_compact.pltpu, "get_tpu_info", return_value=fake_tpu),
            pytest.raises(ValueError, match="must be a multiple of"),
        ):
            dcp_sc_compact.pack_nonnegative(jnp.zeros((2, 10), dtype=jnp.int32))

    def test_pack_body_and_wrappers_end_to_end(self):
        fake_sc = SimpleNamespace(num_lanes=8, num_cores=1, num_subcores=2)
        fake_tpu = SimpleNamespace(sparse_core=fake_sc)
        dcp_sc_compact._get_sc_mesh.cache_clear()
        dcp_sc_compact._get_pack_kernel.cache_clear()

        def _fake_parallel_loop(start, stop, unroll=1, carry=None):
            del unroll
            start_i = int(start)
            stop_i = int(stop)

            def _runner(body_fn):
                if carry is None:
                    for i in range(start_i, stop_i):
                        body_fn(i)
                    return None
                c = carry
                for i in range(start_i, stop_i):
                    c = body_fn(i, c)
                return c

            return _runner

        def _fake_store_compressed(
            dst_ref: _SliceArrayRef, v: jax.Array, mask: jax.Array
        ):
            v_np = np.asarray(v)
            m_np = np.asarray(mask, dtype=bool)
            kept = v_np[m_np]
            start = dst_ref.base_offset
            dst_ref.arr[start : start + len(kept)] = kept

        def _fake_popcount(mask: jax.Array) -> jax.Array:
            cnt = int(np.sum(np.asarray(mask, dtype=bool)))
            return jnp.full((mask.shape[0],), cnt, dtype=jnp.int32)

        def _fake_emit_pipeline(step_fn, *, grid, in_specs, out_specs):
            num_waves = grid[0]
            width = in_specs[0].block_shape[0]

            def _run(vals_hbm: _SliceArrayRef, out_hbm: _SliceArrayRef):
                idx = in_specs[0].index_map(0)[0]
                _ = out_specs.index_map(0)[0]
                for w in range(num_waves):
                    row_idx = int(w * 2 + idx)
                    vals_row = _SliceArrayRef(
                        vals_hbm.arr[row_idx * width : (row_idx + 1) * width].copy()
                    )
                    out_row = _SliceArrayRef(np.zeros((width,), dtype=np.int32))
                    with patch.object(dcp_sc_compact.pl, "program_id", return_value=w):
                        step_fn(vals_row, out_row)
                    out_hbm.arr[row_idx * width : (row_idx + 1) * width] = out_row.arr

            return _run

        def _fake_pl_kernel(
            body_fn, *, out_type, compiler_params, scratch_types, mesh, name
        ):
            del compiler_params, mesh, name
            scratch_shape = scratch_types[0].shape

            def _host_sim(flat_vals_np):
                vals_np = np.asarray(flat_vals_np, dtype=np.int32)
                out_np = np.zeros(out_type.shape, dtype=np.int32)
                for subcore_id in range(fake_sc.num_subcores):
                    vals_ref = _SliceArrayRef(vals_np)
                    out_ref = _SliceArrayRef(out_np)
                    rowbuf_ref = _SliceArrayRef(np.zeros(scratch_shape, dtype=np.int32))
                    with patch.object(
                        dcp_sc_compact.lax,
                        "axis_index",
                        side_effect=lambda axis, s=subcore_id: (
                            0 if axis == "core" else s
                        ),
                    ):
                        body_fn(vals_ref, out_ref, rowbuf_ref)
                return out_np

            def _call(flat_vals: jax.Array) -> jax.Array:
                return jax.pure_callback(_host_sim, out_type, flat_vals)

            return _call

        with (
            patch.object(dcp_sc_compact.pltpu, "get_tpu_info", return_value=fake_tpu),
            patch.object(
                dcp_sc_compact.plsc, "VectorSubcoreMesh", return_value=SimpleNamespace()
            ),
            patch.object(
                dcp_sc_compact.plsc, "parallel_loop", side_effect=_fake_parallel_loop
            ),
            patch.object(
                dcp_sc_compact.plsc,
                "store_compressed",
                side_effect=_fake_store_compressed,
            ),
            patch.object(
                dcp_sc_compact.plsc,
                "all_reduce_population_count",
                side_effect=_fake_popcount,
            ),
            patch.object(
                dcp_sc_compact.pltpu, "emit_pipeline", side_effect=_fake_emit_pipeline
            ),
            patch.object(dcp_sc_compact.pl, "kernel", side_effect=_fake_pl_kernel),
        ):
            # Case 1: num_waves == 1 and padded_b == b (b = 2, rows_per_wave = 2)
            vals_exact = jnp.array(
                [
                    [-1, 5, -1, 2, 9, -1, 0, -1],
                    [3, -1, -1, -1, 7, 1, -1, -1],
                ],
                dtype=jnp.int32,
            )
            packed_exact = dcp_sc_compact.pack_nonnegative(vals_exact)
            np.testing.assert_array_equal(
                np.asarray(packed_exact),
                np.array(
                    [
                        [5, 2, 9, 0, -1, -1, -1, -1],
                        [3, 7, 1, -1, -1, -1, -1, -1],
                    ],
                    dtype=np.int32,
                ),
            )

            # Case 2: num_waves > 1 and padded_b > b (b = 3 -> padded_b = 4)
            vals_padded = jnp.array(
                [
                    [-1, 4, -1, 8, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 2, 3, 4, 5, 6, 7, 8],
                ],
                dtype=jnp.int32,
            )
            packed_padded = dcp_sc_compact.pack_nonnegative(vals_padded)
            assert packed_padded.shape == (3, 8)
            np.testing.assert_array_equal(
                np.asarray(packed_padded[0]),
                np.array([4, 8, -1, -1, -1, -1, -1, -1], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                np.asarray(packed_padded[1]),
                np.full((8,), -1, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                np.asarray(packed_padded[2]),
                np.arange(1, 9, dtype=np.int32),
            )

            # Case 3: resolve_owned_winners
            local_idxs = jnp.array(
                [
                    [10, 11, 12, 13, 14, 15, 16, 17],
                    [20, 21, 22, 23, 24, 25, 26, 27],
                ],
                dtype=jnp.int32,
            )
            slots = jnp.array(
                [
                    [0, 9, -1, 12, 3, 15, -1, 8],
                    [11, 2, 10, -1, -1, -1, -1, -1],
                ],
                dtype=jnp.int32,
            )
            winners = dcp_sc_compact.resolve_owned_winners(
                local_idxs, slots, k=8, dcp_rank=1
            )
            np.testing.assert_array_equal(
                np.asarray(winners),
                np.array(
                    [
                        [11, 14, 17, 10, -1, -1, -1, -1],
                        [23, 22, -1, -1, -1, -1, -1, -1],
                    ],
                    dtype=np.int32,
                ),
            )
