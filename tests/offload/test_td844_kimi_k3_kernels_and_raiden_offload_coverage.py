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
"""CPU unit tests for TD8.44 Kimi-K3 kernels and Raiden offload connector."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferStats,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadKey
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.kernels.kimi_k3 import attention_residual as attn_res_mod
from vllm_torchtpu.kernels.kimi_k3 import decode_kda as decode_kda_mod
from vllm_torchtpu.kernels.kimi_k3 import kda as kda_mod
from vllm_torchtpu.kernels.kimi_k3 import mega_kda as mega_kda_mod
from vllm_torchtpu.offload import raiden_connector as rc_mod
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    FinishedJob,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
)


@pytest.mark.cpu_test
class TestAttentionResidualCoverage:
    """100% statement and branch coverage for attention_residual.py."""

    def test_validation_and_early_returns(self):
        prefix = jnp.ones((16, 8), dtype=jnp.bfloat16)
        history = jnp.ones((16, 2, 8), dtype=jnp.bfloat16)
        weight = jnp.ones((8,), dtype=jnp.float32)

        with pytest.raises(ValueError, match="prefix/history shapes disagree"):
            attn_res_mod.attention_residual(
                prefix, jnp.ones((8, 2, 8), dtype=jnp.bfloat16), weight, interpret=True
            )

        with pytest.raises(ValueError, match="prefix/history shapes disagree"):
            attn_res_mod.attention_residual(
                prefix, jnp.ones((16, 2, 4), dtype=jnp.bfloat16), weight, interpret=True
            )

        with pytest.raises(ValueError, match="one folded weight per hidden feature"):
            attn_res_mod.attention_residual(
                prefix, history, jnp.ones((4,), dtype=jnp.float32), interpret=True
            )

        empty_blocks = jnp.ones((16, 0, 8), dtype=jnp.bfloat16)
        out_empty_blocks = attn_res_mod.attention_residual(
            prefix, empty_blocks, weight, interpret=True
        )
        assert out_empty_blocks is prefix

        empty_tokens_prefix = jnp.ones((0, 8), dtype=jnp.bfloat16)
        empty_tokens_history = jnp.ones((0, 2, 8), dtype=jnp.bfloat16)
        out_empty_tokens = attn_res_mod.attention_residual(
            empty_tokens_prefix, empty_tokens_history, weight, interpret=True
        )
        assert out_empty_tokens is empty_tokens_prefix

        with pytest.raises(ValueError, match="requires 16-token tiles"):
            attn_res_mod.attention_residual(
                jnp.ones((8, 8), dtype=jnp.bfloat16),
                jnp.ones((8, 2, 8), dtype=jnp.bfloat16),
                weight,
                interpret=True,
            )

    def test_interpret_execution_and_helpers(self):
        assert attn_res_mod._idx_map_2d(3) == (3, 0)
        assert attn_res_mod._idx_map_3d(3) == (3, 0, 0)
        assert attn_res_mod._idx_map_1d(3) == (0,)

        in_specs_1, out_spec_1 = attn_res_mod._get_block_specs(8, 2)
        in_specs_2, out_spec_2 = attn_res_mod._get_block_specs(8, 2)
        assert in_specs_1 is in_specs_2
        assert out_spec_1 is out_spec_2

        k1 = attn_res_mod._get_mix_kernel(1e-5)
        k2 = attn_res_mod._get_mix_kernel(1e-5)
        assert k1 is k2

        prefix = jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) * 0.01
        history = jnp.ones((16, 2, 8), dtype=jnp.float32) * 0.5
        weight = jnp.linspace(-0.2, 0.3, 8, dtype=jnp.float32)
        out = attn_res_mod.attention_residual(
            prefix, history, weight, eps=1e-5, interpret=True
        )
        assert out.shape == (16, 8)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.cpu_test
class TestDecodeKdaCoverage:
    """100% statement and branch coverage for decode_kda.py."""

    def test_helpers_and_step(self):
        assert decode_kda_mod._seq_map_3d(2, [1, 2, 3], [0, 1, 1]) == (2, 0, 0)
        assert decode_kda_mod._seq_map_4d(2, [1, 2, 3], [0, 1, 1]) == (2, 0, 0, 0)
        assert decode_kda_mod._slot_map_5d(2, [1, 2, 5], [0, 1, 1]) == (5, 0, 0, 0, 0)
        assert decode_kda_mod._slot_map_4d(2, [1, 2, 5], [0, 1, 1]) == (5, 0, 0, 0)
        assert decode_kda_mod._zero_map_4d(2, [1, 2, 5], [0, 1, 1]) == (0, 0, 0, 0)
        assert decode_kda_mod._zero_map_2d(2, [1, 2, 5], [0, 1, 1]) == (0, 0)

        gs1 = decode_kda_mod._get_decode_grid_spec(2, 4, 8, 8, 4)
        gs2 = decode_kda_mod._get_decode_grid_spec(2, 4, 8, 8, 4)
        assert gs1 is gs2

        cp1 = decode_kda_mod._get_compiler_params(16 * 1024 * 1024)
        cp2 = decode_kda_mod._get_compiler_params(16 * 1024 * 1024)
        assert cp1 is cp2

        x = jnp.ones((2, 4), dtype=jnp.bfloat16)
        normed = decode_kda_mod._l2_norm(x)
        assert normed.dtype == jnp.bfloat16
        assert normed.shape == (2, 4)

        g = jnp.ones((2, 4), dtype=jnp.float32) * 0.2
        a_log = jnp.zeros((2, 1), dtype=jnp.float32)
        dt_bias = jnp.ones((2, 4), dtype=jnp.float32) * 0.1

        gate_no_bias_softplus = decode_kda_mod._activate_gate(g, a_log, None, None)
        gate_bias_sigmoid = decode_kda_mod._activate_gate(g, a_log, dt_bias, -5.0)
        assert gate_no_bias_softplus.shape == (2, 4)
        assert gate_bias_sigmoid.shape == (2, 4)

        q = jnp.ones((2, 1, 4), dtype=jnp.float32)
        k = jnp.ones((2, 1, 4), dtype=jnp.float32)
        k_t = jnp.ones((2, 4, 1), dtype=jnp.float32)
        v = jnp.ones((2, 1, 4), dtype=jnp.float32)
        decay = jnp.ones((2, 4, 1), dtype=jnp.float32) * 0.9
        beta = jnp.ones((2, 1, 1), dtype=jnp.float32) * 0.5
        state = jnp.zeros((2, 4, 4), dtype=jnp.float32)
        out, next_state = decode_kda_mod._decode_step(
            q, k, k_t, v, decay, beta, state, jnp.float32
        )
        assert out.shape == (2, 4)
        assert next_state.shape == (2, 4, 4)

    def test_kernel_body_execution(self):
        n_v, d_k, kernel_size = 2, 4, 3
        state_indices_ref = jnp.array([1, 0], dtype=jnp.int32)
        has_initial_state_ref = jnp.array([1, 0], dtype=jnp.int32)
        mixed_qkv_ref = jnp.ones((1, 3, n_v, d_k), dtype=jnp.float32) * 0.2
        g_ref = jnp.ones((1, n_v, d_k), dtype=jnp.float32) * 0.1
        beta_ref = jnp.ones((1, n_v, 1), dtype=jnp.float32) * 0.5
        conv_state_ref = jnp.ones((1, kernel_size - 1, 3, n_v, d_k), dtype=jnp.float32)
        conv_weight_ref = jnp.ones((kernel_size, 3, n_v, d_k), dtype=jnp.float32) * 0.1
        a_log_ref = jnp.zeros((n_v, 1), dtype=jnp.float32)
        dt_bias_ref = jnp.zeros((n_v, d_k), dtype=jnp.float32)
        state_ref = jnp.ones((1, n_v, d_k, d_k), dtype=jnp.float32) * 0.05

        out_ref = np.zeros((1, n_v, d_k), dtype=np.float32)
        conv_state_out_ref = np.zeros(
            (1, kernel_size - 1, 3, n_v, d_k), dtype=np.float32
        )
        state_out_ref = np.zeros((1, n_v, d_k, d_k), dtype=np.float32)

        with (
            patch.object(decode_kda_mod.pl, "program_id", return_value=0),
            patch.object(
                decode_kda_mod,
                "fused_transpose_broadcast",
                side_effect=lambda x, src_dim, dst_dim: jnp.swapaxes(
                    x, src_dim, dst_dim
                ),
            ),
        ):
            decode_kda_mod._kernel(
                state_indices_ref,
                has_initial_state_ref,
                mixed_qkv_ref,
                g_ref,
                beta_ref,
                conv_state_ref,
                conv_weight_ref,
                a_log_ref,
                dt_bias_ref,
                state_ref,
                out_ref,
                conv_state_out_ref,
                state_out_ref,
                n_v=n_v,
                d_k=d_k,
                kernel_size=kernel_size,
                lower_bound=-5.0,
                compute_dtype=jnp.float32,
                scale=0.5,
                state_transposed=True,
            )
            assert np.any(out_ref != 0.0)

            # Also cover kernel_size == 1 (0 loop iterations) and state_transposed=False
            conv_state_ref_k1 = jnp.ones((1, 1, 3, n_v, d_k), dtype=jnp.float32)
            conv_weight_ref_k1 = jnp.ones((1, 3, n_v, d_k), dtype=jnp.float32)
            conv_state_out_k1 = np.zeros((1, 1, 3, n_v, d_k), dtype=np.float32)
            decode_kda_mod._kernel(
                state_indices_ref,
                has_initial_state_ref,
                mixed_qkv_ref,
                g_ref,
                beta_ref,
                conv_state_ref_k1,
                conv_weight_ref_k1,
                a_log_ref,
                dt_bias_ref,
                state_ref,
                out_ref,
                conv_state_out_k1,
                state_out_ref,
                n_v=n_v,
                d_k=d_k,
                kernel_size=1,
                lower_bound=None,
                compute_dtype=jnp.float32,
                scale=0.5,
                state_transposed=False,
            )

    def test_decode_kda_validation_and_dispatch(self):
        num_seqs, n_v, d_k, kernel_size = 2, 2, 4, 3
        mixed_dim = 3 * n_v * d_k
        mixed_qkv = jnp.ones((num_seqs, mixed_dim), dtype=jnp.float32)
        g = jnp.ones((num_seqs, n_v, d_k), dtype=jnp.float32)
        beta = jnp.ones((num_seqs, n_v), dtype=jnp.float32)
        conv_state_5d = jnp.ones((4, kernel_size - 1, 3, n_v, d_k), dtype=jnp.float32)
        conv_weight = jnp.ones((kernel_size, 3, n_v, d_k), dtype=jnp.float32)
        state = jnp.zeros((4, n_v, d_k, d_k), dtype=jnp.float32)
        a_log = jnp.zeros((n_v,), dtype=jnp.float32)
        dt_bias = jnp.zeros((n_v, d_k), dtype=jnp.float32)
        state_indices = jnp.array([1, 2], dtype=jnp.int32)
        has_initial_state = jnp.array([1, 0], dtype=jnp.int32)

        with pytest.raises(ValueError, match="mixed_qkv must have width"):
            decode_kda_mod.decode_kda(
                jnp.ones((num_seqs, mixed_dim - 1), dtype=jnp.float32),
                g,
                beta,
                conv_state_5d,
                conv_weight,
                state,
                a_log,
                dt_bias,
                state_indices,
                has_initial_state,
            )

        with pytest.raises(ValueError, match="conv_weight must have shape"):
            decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                conv_state_5d,
                jnp.ones((kernel_size, 2, n_v, d_k), dtype=jnp.float32),
                state,
                a_log,
                dt_bias,
                state_indices,
                has_initial_state,
            )

        with pytest.raises(ValueError, match="conv_state must end in"):
            decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                jnp.ones((4, kernel_size, 3, n_v, d_k), dtype=jnp.float32),
                conv_weight,
                state,
                a_log,
                dt_bias,
                state_indices,
                has_initial_state,
            )

        with pytest.raises(ValueError, match="currently requires d_v == d_k"):
            decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                conv_state_5d,
                conv_weight,
                jnp.zeros((4, n_v, d_k, d_k + 1), dtype=jnp.float32),
                a_log,
                dt_bias,
                state_indices,
                has_initial_state,
            )

        with pytest.raises(ValueError, match="dt_bias must be"):
            decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                conv_state_5d,
                conv_weight,
                state,
                a_log,
                jnp.zeros((n_v, d_k + 1), dtype=jnp.float32),
                state_indices,
                has_initial_state,
            )

        def fake_pallas_call(kernel, *, grid_spec, out_shape, **kwargs):
            for spec in grid_spec.in_specs:
                spec.index_map(0, [1, 2], [1, 0])
            for spec in grid_spec.out_specs:
                spec.index_map(0, [1, 2], [1, 0])

            def _runner(*args):
                return tuple(jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape)

            return _runner

        fake_tpu_info = SimpleNamespace(vmem_capacity_bytes=16 * 1024 * 1024)
        with (
            patch.object(
                decode_kda_mod.pltpu, "get_tpu_info", return_value=fake_tpu_info
            ),
            patch.object(
                decode_kda_mod.pl, "pallas_call", side_effect=fake_pallas_call
            ),
        ):
            # 1. 5D conv_state, scale=None, dt_bias=None
            out, cs_out, st_out = decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                conv_state_5d,
                conv_weight,
                state,
                a_log,
                None,
                state_indices,
                has_initial_state,
            )
            assert out.shape == (num_seqs, n_v, d_k)
            assert cs_out.shape == conv_state_5d.shape
            assert st_out.shape == state.shape

            # 2. 3D conv_state with conv_state_dim_first=False, explicit scale & dt_bias
            conv_state_3d = jnp.ones((4, kernel_size - 1, mixed_dim), dtype=jnp.float32)
            out2, cs_out2, _ = decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                conv_state_3d,
                conv_weight,
                state,
                a_log,
                dt_bias,
                state_indices,
                has_initial_state,
                scale=0.5,
                conv_state_dim_first=False,
            )
            assert out2.shape == (num_seqs, n_v, d_k)
            assert cs_out2.shape == conv_state_3d.shape

            # 3. 3D conv_state with conv_state_dim_first=True
            conv_state_3d_df = jnp.ones(
                (4, mixed_dim, kernel_size - 1), dtype=jnp.float32
            )
            out3, cs_out3, _ = decode_kda_mod.decode_kda(
                mixed_qkv,
                g,
                beta,
                conv_state_3d_df,
                conv_weight,
                state,
                a_log,
                dt_bias,
                state_indices,
                has_initial_state,
                conv_state_dim_first=True,
            )
            assert out3.shape == (num_seqs, n_v, d_k)
            assert cs_out3.shape == conv_state_3d_df.shape


@pytest.mark.cpu_test
class TestKdaCoverage:
    """100% statement and branch coverage for kda.py."""

    def _valid_inputs(self):
        num_tokens, num_heads, head_dim = 4, 2, 4
        proj_dim = num_heads * head_dim
        mixed_qkv = jnp.ones((num_tokens, 3 * proj_dim), dtype=jnp.bfloat16) * 0.1
        raw_gate = jnp.zeros((num_tokens, proj_dim), dtype=jnp.bfloat16)
        beta = jnp.zeros((num_tokens, num_heads), dtype=jnp.bfloat16)
        output_gate = jnp.zeros((num_tokens, proj_dim), dtype=jnp.bfloat16)
        recurrent_state = jnp.zeros(
            (3, num_heads, head_dim, head_dim), dtype=jnp.float32
        )
        a_log = jnp.zeros((num_heads,), dtype=jnp.float32)
        dt_bias = jnp.zeros((proj_dim,), dtype=jnp.float32)
        norm_weight = jnp.ones((head_dim,), dtype=jnp.float32)
        query_start_loc = jnp.array([0, 2, 3], dtype=jnp.int32)
        state_indices = jnp.array([1, -1], dtype=jnp.int32)
        seq_lens = jnp.array([4, 1], dtype=jnp.int32)
        return (
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            recurrent_state,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )

    def test_ragged_kda_execution_and_bounds(self):
        args = self._valid_inputs()
        out_none, state_none = kda_mod.ragged_kda(*args, lower_bound=None, eps=1e-5)
        assert out_none.shape == (4, 2, 4)
        assert state_none.shape == (3, 2, 4, 4)

        args2 = self._valid_inputs()
        out_lb, state_lb = kda_mod.ragged_kda(*args2, lower_bound=-5.0, eps=1e-5)
        assert out_lb.shape == (4, 2, 4)
        assert state_lb.shape == (3, 2, 4, 4)

    def test_ragged_kda_validation_errors(self):
        (
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            recurrent_state,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        ) = self._valid_inputs()

        with pytest.raises(ValueError, match="Incompatible KDA activation and state"):
            kda_mod.ragged_kda(
                jnp.ones((4, 10), dtype=jnp.bfloat16),
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="Incompatible KDA gate shape"):
            kda_mod.ragged_kda(
                mixed_qkv,
                jnp.zeros((4, 3), dtype=jnp.bfloat16),
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="Incompatible KDA beta shape"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                jnp.zeros((4, 1), dtype=jnp.bfloat16),
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="Incompatible KDA output-gate shape"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                jnp.zeros((4, 3), dtype=jnp.bfloat16),
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="A_log shape does not match"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                jnp.zeros((1,), dtype=jnp.float32),
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="dt_bias shape does not match"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log,
                jnp.zeros((3,), dtype=jnp.float32),
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="norm weight shape does not match"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                jnp.ones((2,), dtype=jnp.float32),
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(ValueError, match="query starts and state indices disagree"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                jnp.array([0, 2], dtype=jnp.int32),
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(
            ValueError, match="sequence lengths and state indices disagree"
        ):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                jnp.array([2], dtype=jnp.int32),
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(TypeError, match="activations must have one common dtype"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate.astype(jnp.float32),
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(TypeError, match="recurrent state must be float32"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state.astype(jnp.bfloat16),
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(TypeError, match="A_log and dt_bias must be float32"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log.astype(jnp.bfloat16),
                dt_bias,
                norm_weight,
                query_start_loc,
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )

        with pytest.raises(TypeError, match="metadata tensors must be int32"):
            kda_mod.ragged_kda(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                recurrent_state,
                a_log,
                dt_bias,
                norm_weight,
                query_start_loc.astype(jnp.int16),
                state_indices,
                seq_lens,
                lower_bound=None,
                eps=1e-5,
            )


@pytest.mark.cpu_test
class TestMegaKdaCoverage:
    """100% statement and branch coverage for mega_kda/__init__.py."""

    def test_layout_supported(self):
        with pytest.raises(ValueError, match="requires full token tiles"):
            mega_kda_mod._layout_supported(
                jnp.array([0, 32], dtype=jnp.int32), num_tokens=96, chunk_size=64
            )

        assert bool(
            mega_kda_mod._layout_supported(
                jnp.array([0, 32, 64], dtype=jnp.int32), num_tokens=64
            )
        )
        assert bool(
            mega_kda_mod._layout_supported(
                jnp.array([0, 32, 64, 128], dtype=jnp.int32), num_tokens=128
            )
        )
        assert not bool(
            mega_kda_mod._layout_supported(
                jnp.array([0, 16, 32, 64], dtype=jnp.int32), num_tokens=64
            )
        )

    def test_kda_forward_inference_validation_and_dispatch(self):
        q = jnp.ones((1, 64, 2, 8), dtype=jnp.bfloat16)
        k = jnp.ones((1, 64, 2, 8), dtype=jnp.bfloat16)
        v = jnp.ones((1, 64, 2, 8), dtype=jnp.bfloat16)
        g = jnp.ones((1, 64, 2, 8), dtype=jnp.bfloat16)
        beta = jnp.ones((1, 64, 2), dtype=jnp.bfloat16)
        seg_1d = jnp.ones((64,), dtype=jnp.int32)
        seg_2d = jnp.ones((1, 64), dtype=jnp.int32)
        a_log = jnp.zeros((2,), dtype=jnp.float32)
        dt_bias = jnp.zeros((16,), dtype=jnp.float32)
        init_state = jnp.zeros((1, 2, 2, 8, 8), dtype=jnp.float32)

        common_kw = dict(
            segment_ids=seg_2d,
            A_log=a_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            N_max=2,
        )

        with pytest.raises(ValueError, match="currently requires BF16"):
            mega_kda_mod.kda_forward_inference(
                q.astype(jnp.float32), k, v, g, beta, **common_kw
            )

        with pytest.raises(ValueError, match="q, k, v, and g must have the same dtype"):
            mega_kda_mod.kda_forward_inference(
                q, k.astype(jnp.float32), v, g, beta, **common_kw
            )

        with pytest.raises(ValueError, match="beta must have the same dtype as q"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, beta.astype(jnp.float32), **common_kw
            )

        with pytest.raises(ValueError, match="requires chunk_size=64"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, beta, chunk_size=32, **common_kw
            )

        with pytest.raises(ValueError, match="q, k, and g must agree"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, jnp.ones((1, 64, 2, 4), dtype=jnp.bfloat16), beta, **common_kw
            )

        with pytest.raises(ValueError, match="v and beta must agree"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, jnp.ones((1, 64, 1), dtype=jnp.bfloat16), **common_kw
            )

        with pytest.raises(ValueError, match="same key dimension"):
            mega_kda_mod.kda_forward_inference(
                q, jnp.ones((1, 64, 2, 4), dtype=jnp.bfloat16), v, g, beta, **common_kw
            )

        q_unaligned = jnp.ones((1, 32, 2, 8), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="padded to a multiple of chunk_size"):
            mega_kda_mod.kda_forward_inference(
                q_unaligned,
                q_unaligned,
                q_unaligned,
                q_unaligned,
                jnp.ones((1, 32, 2), dtype=jnp.bfloat16),
                **common_kw,
            )

        with pytest.raises(ValueError, match="segment_ids must have shape"):
            mega_kda_mod.kda_forward_inference(
                q,
                k,
                v,
                g,
                beta,
                **(common_kw | {"segment_ids": jnp.ones((1, 32), dtype=jnp.int32)}),
            )

        with pytest.raises(ValueError, match="segment_ids must use int32"):
            mega_kda_mod.kda_forward_inference(
                q,
                k,
                v,
                g,
                beta,
                **(common_kw | {"segment_ids": seg_2d.astype(jnp.int16)}),
            )

        with pytest.raises(ValueError, match="requires A_log and dt_bias"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, beta, **(common_kw | {"A_log": None})
            )

        with pytest.raises(ValueError, match="requires in-kernel Q/K L2 normalization"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, beta, **(common_kw | {"use_qk_l2norm_in_kernel": False})
            )

        with pytest.raises(ValueError, match="requires safe fused gate activation"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, beta, **(common_kw | {"use_gate_in_kernel": False})
            )

        with pytest.raises(ValueError, match="lower_bound in"):
            mega_kda_mod.kda_forward_inference(
                q, k, v, g, beta, **(common_kw | {"lower_bound": 0.0})
            )

        with pytest.raises(ValueError, match="N_max is required"):
            mega_kda_mod.kda_forward_inference(
                q,
                k,
                v,
                g,
                beta,
                **(common_kw | {"N_max": None, "initial_state": None}),
            )

        with pytest.raises(ValueError, match="initial_state must use float32"):
            mega_kda_mod.kda_forward_inference(
                q,
                k,
                v,
                g,
                beta,
                **(common_kw | {"initial_state": init_state.astype(jnp.bfloat16)}),
            )

        with patch.object(
            mega_kda_mod,
            "_chunk_kda_fwd_native_segids",
            return_value=(v, init_state, None),
        ) as mock_fwd:
            # 1D segment_ids, N_max=None derived from initial_state, scale=None
            out1, st1 = mega_kda_mod.kda_forward_inference(
                q,
                k,
                v,
                g,
                beta,
                segment_ids=seg_1d,
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=init_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
                N_max=None,
                scale=None,
            )
            assert out1 is v
            assert st1 is init_state
            assert mock_fwd.call_args.kwargs["N_max"] == 2

            # 2D segment_ids, explicit N_max and scale, initial_state=None
            out2, st2 = mega_kda_mod.kda_forward_inference(
                q,
                k,
                v,
                g,
                beta,
                segment_ids=seg_2d,
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
                N_max=4,
                scale=0.25,
            )
            assert out2 is v
            assert st2 is init_state
            assert mock_fwd.call_args.kwargs["scale"] == 0.25


@pytest.mark.cpu_test
class TestRaidenConnectorCoverage:
    """100% statement and branch coverage for offload/raiden_connector.py."""

    def test_fence_helpers_and_worker_metadata_aggregate(self):
        fake_sync = MagicMock()
        fake_internal = MagicMock()
        fake_internal.sync = fake_sync
        t = torch.zeros((2, 4), dtype=torch.int8)
        with patch.dict(
            sys.modules,
            {
                "torch_tpu": MagicMock(_internal=fake_internal),
                "torch_tpu._internal": fake_internal,
                "torch_tpu._internal.sync": fake_sync,
            },
        ):
            rc_mod._enqueue_pool_writes([t])
        fake_sync.synchronize.assert_called_once_with([t], wait=False)

        fake_event = MagicMock()
        with patch.object(rc_mod.torch, "tpu", create=True) as mock_tpu:
            mock_tpu.Event.return_value = fake_event
            ev = rc_mod._record_fence_event()
            assert ev is fake_event
            fake_event.record.assert_called_once()

        fake_event.query.return_value = False
        assert rc_mod._event_is_retired(fake_event) is False
        fake_event.synchronize.assert_not_called()

        fake_event.query.return_value = True
        assert rc_mod._event_is_retired(fake_event) is True
        fake_event.synchronize.assert_called_once()

        m1 = rc_mod.RaidenOffloadingWorkerMetadata(
            completed_jobs={1: 1},
            transfer_stats=TransferStats(),
            fenced_jobs={10: 1},
        )
        m_empty = rc_mod.RaidenOffloadingWorkerMetadata(
            completed_jobs={},
            transfer_stats=TransferStats(),
            fenced_jobs={},
        )
        agg_empty = m1.aggregate(m_empty)
        assert agg_empty.completed_jobs == {1: 1}
        assert agg_empty.fenced_jobs == {10: 1}

        m2 = rc_mod.RaidenOffloadingWorkerMetadata(
            completed_jobs={1: 1, 2: 1},
            transfer_stats=TransferStats(),
            fenced_jobs={10: 1, 20: 1},
        )
        agg = m1.aggregate(m2)
        assert agg.completed_jobs == {1: 2, 2: 1}
        assert agg.fenced_jobs == {10: 2, 20: 1}

    def test_scheduler_and_connector_branches(self, monkeypatch):
        assert (
            rc_mod.TPURaidenOffloadingConnector.get_required_kvcache_layout(MagicMock())
            is None
        )

        # Test RAIDEN_DISABLE_SINGLETON_WORKER=False error
        monkeypatch.setattr(rc_mod.tpu_envs, "RAIDEN_DISABLE_SINGLETON_WORKER", False)
        with pytest.raises(ValueError, match="RAIDEN_DISABLE_SINGLETON_WORKER"):
            rc_mod.TPURaidenOffloadingConnector(
                MagicMock(), KVConnectorRole.WORKER, MagicMock()
            )
        monkeypatch.setattr(rc_mod.tpu_envs, "RAIDEN_DISABLE_SINGLETON_WORKER", True)

        # Test role that is neither SCHEDULER nor WORKER
        fake_spec = MagicMock(spec=TPURaidenStoreOffloadingSpec)
        with (
            patch.object(
                rc_mod.offloading_config_module,
                "build_offloading_config",
                return_value=MagicMock(),
            ),
            patch.object(
                rc_mod, "TPURaidenStoreOffloadingSpec", return_value=fake_spec
            ),
        ):
            conn_other = rc_mod.TPURaidenOffloadingConnector(
                MagicMock(),
                "NONE_ROLE",  # type: ignore[arg-type]
                MagicMock(),
            )
            assert conn_other.connector_scheduler is None
            assert conn_other.connector_worker is None
            assert conn_other.reset_cache() is False
            assert list(conn_other.take_events()) == []

            monkeypatch.setattr(rc_mod.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", True)
            assert conn_other.prefer_cross_layer_blocks is True
            monkeypatch.setattr(rc_mod.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)
            assert conn_other.prefer_cross_layer_blocks is False

        # Test SCHEDULER role & TPURaidenOffloadingScheduler branches
        fake_manager = MagicMock(spec=RaidenOffloadingManager)

        def fake_sched_init(self, spec, vllm_config, kv_cache_config):
            self.manager = fake_manager
            self.config = SimpleNamespace(num_workers=2)

        with (
            patch.object(
                rc_mod.offloading_config_module,
                "build_offloading_config",
                return_value=MagicMock(num_workers=2),
            ),
            patch.object(
                rc_mod, "TPURaidenStoreOffloadingSpec", return_value=fake_spec
            ),
            patch.object(
                rc_mod.OffloadingConnectorScheduler,
                "__init__",
                fake_sched_init,
            ),
        ):
            conn_sched = rc_mod.TPURaidenOffloadingConnector(
                MagicMock(), KVConnectorRole.SCHEDULER, MagicMock()
            )
            assert conn_sched.connector_scheduler is not None
            sched = conn_sched.connector_scheduler

            with patch.object(
                rc_mod.OffloadingConnectorScheduler,
                "take_events",
                return_value=["ev1"],
            ):
                assert list(conn_sched.take_events()) == ["ev1"]

            # 1. build_connector_meta with store_jobs, load_jobs, jobs_to_flush=set()
            key_a = OffloadKey(b"a" * 8)
            key_b = OffloadKey(b"b" * 8)
            gpu_src = MagicMock(spec=GPULoadStoreSpec)
            gpu_src.block_ids = np.array([1], dtype=np.int64)
            gpu_dst = MagicMock(spec=GPULoadStoreSpec)
            gpu_dst.block_ids = np.array([2], dtype=np.int64)
            store_job = SimpleNamespace(
                src_spec=gpu_src,
                dst_spec=RaidenLoadStoreSpec(keys=[key_a]),
                req_id="req-s",
            )
            load_job = SimpleNamespace(
                src_spec=RaidenLoadStoreSpec(keys=[key_b], pinned=True),
                dst_spec=gpu_dst,
                req_id="req-l",
            )
            base_meta_1 = OffloadingConnectorMetadata(
                load_jobs={20: load_job},
                store_jobs={10: store_job},
                jobs_to_flush=set(),
            )
            fake_manager.poll_finished_jobs.return_value = [
                FinishedJob(job_id=99, op=AdmissionOp.STORE, success=True),
                FinishedJob(
                    job_id=98,
                    op=AdmissionOp.LOAD,
                    success=False,
                    req_id="req-98",
                    failed_device_block_ids=[7],
                ),
            ]
            with patch.object(
                rc_mod.OffloadingConnectorScheduler,
                "build_connector_meta",
                return_value=base_meta_1,
            ):
                out_meta_1 = sched.build_connector_meta(MagicMock())
            assert isinstance(out_meta_1, rc_mod.RaidenOffloadingConnectorMetadata)
            assert out_meta_1.fence_job_ids == {10}
            assert out_meta_1.finished_store_job_ids == [99]
            assert out_meta_1.finished_load_jobs == {98: "req-98"}
            assert out_meta_1.failed_load_device_block_ids == [7]

            # 2. build_connector_meta with jobs_to_flush containing:
            # - parked job 10 (cancelled before launch)
            # - launched job 11 (drained)
            # - unknown job 12 (neither parked nor launched)
            fake_manager.is_job_launched.side_effect = lambda jid: jid == 11
            fake_manager.drain_jobs.return_value = [
                FinishedJob(job_id=11, op=AdmissionOp.STORE, success=True)
            ]
            fake_manager.poll_finished_jobs.return_value = []
            base_meta_2 = OffloadingConnectorMetadata(
                load_jobs={},
                store_jobs={},
                jobs_to_flush={10, 11, 12},
            )
            with patch.object(
                rc_mod.OffloadingConnectorScheduler,
                "build_connector_meta",
                return_value=base_meta_2,
            ):
                out_meta_2 = sched.build_connector_meta(MagicMock())
            assert set(out_meta_2.finished_store_job_ids) == {10, 11}

            # 3. build_connector_meta with jobs_to_flush where no job is launched
            base_meta_3 = OffloadingConnectorMetadata(
                load_jobs={},
                store_jobs={15: store_job},
                jobs_to_flush={15},
            )
            with patch.object(
                rc_mod.OffloadingConnectorScheduler,
                "build_connector_meta",
                return_value=base_meta_3,
            ):
                out_meta_3 = sched.build_connector_meta(MagicMock())
            assert out_meta_3.finished_store_job_ids == [15]
            assert out_meta_3.fence_job_ids == set()

            # 4. update_connector_output branches
            sched._fence_pending[30] = rc_mod._ParkedStore(
                keys=[key_a], device_block_ids=[1], acks=0
            )
            with patch.object(
                rc_mod.OffloadingConnectorScheduler, "update_connector_output"
            ):
                # None worker meta
                sched.update_connector_output(
                    KVConnectorOutput(kv_connector_worker_meta=None)
                )
                # Unknown job 999 + partial ack on job 30 (1 of 2 workers)
                sched.update_connector_output(
                    KVConnectorOutput(
                        kv_connector_worker_meta=rc_mod.RaidenOffloadingWorkerMetadata(
                            completed_jobs={},
                            transfer_stats=TransferStats(),
                            fenced_jobs={999: 1, 30: 1},
                        )
                    )
                )
                assert 30 in sched._fence_pending
                # Second ack on job 30 -> submits store job
                sched.update_connector_output(
                    KVConnectorOutput(
                        kv_connector_worker_meta=rc_mod.RaidenOffloadingWorkerMetadata(
                            completed_jobs={},
                            transfer_stats=TransferStats(),
                            fenced_jobs={30: 1},
                        )
                    )
                )
                assert 30 not in sched._fence_pending
                fake_manager.submit_store_job.assert_called_once_with(30, [key_a], [1])

    def test_worker_connector_branches(self, monkeypatch):
        monkeypatch.setattr(rc_mod.tpu_envs, "RAIDEN_DISABLE_SINGLETON_WORKER", True)
        fake_spec = MagicMock(spec=TPURaidenStoreOffloadingSpec)
        vllm_cfg = MagicMock()
        kv_cfg = MagicMock()
        kv_cfg.num_blocks = 4
        kv_cfg.kv_cache_groups = [MagicMock()]

        def fake_worker_init(self, spec, vllm_config, kv_cache_config):
            self.kv_cache_config = kv_cache_config
            self._init_worker = MagicMock()

        with (
            patch.object(
                rc_mod.offloading_config_module,
                "build_offloading_config",
                return_value=MagicMock(),
            ),
            patch.object(
                rc_mod, "TPURaidenStoreOffloadingSpec", return_value=fake_spec
            ),
            patch.object(
                rc_mod.OffloadingConnectorWorker,
                "__init__",
                fake_worker_init,
            ),
        ):
            worker_conn = rc_mod.TPURaidenOffloadingConnector(
                vllm_cfg, KVConnectorRole.WORKER, kv_cfg
            )

        # Cover empty kv_caches loop branch
        worker_conn.register_kv_caches({})
        assert worker_conn._pool_sync_tensors == []

        t_single = torch.zeros((4, 16), dtype=torch.int8)
        t_list = [torch.zeros((4, 8), dtype=torch.int8)]
        worker_conn.register_kv_caches({"l0": t_single, "l1": t_list})
        assert len(worker_conn._pool_sync_tensors) == 2

        # 1. get_finished with empty fence_job_ids, pending_fences, finished/failed
        meta_empty = rc_mod.RaidenOffloadingConnectorMetadata(
            load_jobs={}, store_jobs={}
        )
        with patch.object(
            worker_conn, "_get_connector_metadata", return_value=meta_empty
        ):
            sending, recving = worker_conn.get_finished(set())
        assert sending == set()
        assert recving == set()
        assert worker_conn.build_connector_worker_meta() is None

        # 2. get_finished with fence_job_ids and unretired event, then retired event
        meta_fence = rc_mod.RaidenOffloadingConnectorMetadata(
            load_jobs={},
            store_jobs={},
            fence_job_ids={101},
            finished_store_job_ids=[201],
            finished_load_jobs={301: "req-301"},
            failed_load_device_block_ids=[5],
        )
        fake_ev = MagicMock()
        with (
            patch.object(
                worker_conn, "_get_connector_metadata", return_value=meta_fence
            ),
            patch.object(rc_mod, "_enqueue_pool_writes"),
            patch.object(rc_mod, "_record_fence_event", return_value=fake_ev),
            patch.object(rc_mod, "_event_is_retired", return_value=False),
        ):
            _, recving = worker_conn.get_finished(set())
        assert recving == {"req-301"}
        assert worker_conn.get_block_ids_with_load_errors() == {5}
        assert worker_conn.get_block_ids_with_load_errors() == set()
        wmeta_1 = worker_conn.build_connector_worker_meta()
        assert isinstance(wmeta_1, rc_mod.RaidenOffloadingWorkerMetadata)
        assert wmeta_1.completed_jobs == {201: 1, 301: 1}
        assert wmeta_1.fenced_jobs == {}

        # Next step: event retires (include empty job_ids set to cover 0-iter loop)
        worker_conn._pending_fences.append((set(), fake_ev))
        with (
            patch.object(
                worker_conn, "_get_connector_metadata", return_value=meta_empty
            ),
            patch.object(rc_mod, "_event_is_retired", return_value=True),
        ):
            worker_conn.get_finished(set())
        wmeta_2 = worker_conn.build_connector_worker_meta()
        assert isinstance(wmeta_2, rc_mod.RaidenOffloadingWorkerMetadata)
        assert wmeta_2.fenced_jobs == {101: 1}
