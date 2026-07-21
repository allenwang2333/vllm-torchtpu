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
"""Correctness test for DeepSeek-V4 MoE routing and EP index mapping."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import shard_map

# Import the core MoE JAX functions
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func


def gen_random(rng, shape, dtype=jnp.bfloat16):
    return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)


def ref_moe_implementation(
    hidden_states: np.ndarray,
    w1: np.ndarray,  # [global_num_experts, hidden_size, intermediate_size * 2]
    w2: np.ndarray,  # [global_num_experts, hidden_size, intermediate_size]
    topk_weights: np.ndarray,  # [num_tokens, topk]
    topk_ids: np.ndarray,  # [num_tokens, topk]
    topk: int,
    activation: str = "silu",
) -> np.ndarray:
    num_tokens, hidden_size = hidden_states.shape
    output = np.zeros((num_tokens, hidden_size), dtype=np.float32)
    
    # Iterate over all global experts sequentially
    for e in range(w1.shape[0]):
        # mask for tokens that route to expert e
        mask = (topk_ids == e)
        expert_weights = np.where(mask, topk_weights, 0.0).sum(axis=1)
        
        # Expert gate & up projections (silu activation)
        gate_up = hidden_states @ w1[e]
        
        # Split gate_up into gate and up projections
        mid = gate_up.shape[-1] // 2
        gate = gate_up[..., :mid]
        up = gate_up[..., mid:]
        
        # SwiGLU / Gated SiLU activation
        act = (gate / (1.0 + np.exp(-gate))) * up
        
        # Out projection (w2 is transposed to match [intermediate_size, hidden_size] multiplication)
        out = act @ w2[e].T
        output += out * expert_weights[:, None]
        
    return output


class MoERoutingCorrectnessTest(parameterized.TestCase):

    def test_moe_routing_correctness_ep8(self):
        rng = np.random.default_rng(1234)

        devices = jax.devices()
        assert len(devices) >= 8, f"Need at least 8 devices to run EP=8 sharding test, found {len(devices)}"
        devices = devices[:8]
        print(f"Running on JAX backend: {jax.default_backend()} using devices: {devices}")

        # Configuration
        num_tokens = 16
        hidden_size = 128
        intermediate_size = 256
        global_num_experts = 256
        ep_size = 8
        local_num_experts = global_num_experts // ep_size  # 32 experts per rank
        topk = 6
        activation = "silu"

        # Generate inputs
        hidden_states = gen_random(rng, (num_tokens, hidden_size))
        w1_global = gen_random(rng, (global_num_experts, hidden_size, intermediate_size * 2))
        w2_global = gen_random(rng, (global_num_experts, hidden_size, intermediate_size))

        # Generate top-K routing assignments
        topk_ids_list = []
        for _ in range(num_tokens):
            topk_ids_list.append(rng.choice(global_num_experts, size=topk, replace=False))
        topk_ids = jnp.array(topk_ids_list, dtype=jnp.int32)
        topk_weights = jnp.array(rng.random(size=(num_tokens, topk), dtype=np.float32))
        # Normalize weights
        topk_weights /= topk_weights.sum(axis=1, keepdims=True)

        # Run CPU Reference (float32)
        ref_out = ref_moe_implementation(
            np.array(hidden_states, dtype=np.float32),
            np.array(w1_global, dtype=np.float32),
            np.array(w2_global, dtype=np.float32),
            np.array(topk_weights, dtype=np.float32),
            np.array(topk_ids, dtype=np.int32),
            topk=topk,
            activation=activation,
        )

        # Run non-sharded JAX operator check directly
        non_sharded_out = fused_moe_func(
            hidden_states,
            w1_global,
            w2_global,
            None,
            None,
            None,
            None,
            topk_weights.astype(hidden_states.dtype),
            topk_ids,
            experts_start=None,
            topk=topk,
            activation=activation,
            use_ep=False,
            use_sparse_core=False,
        )
        non_sharded_np = np.array(non_sharded_out, dtype=np.float32)
        non_sharded_diff = np.abs(ref_out - non_sharded_np)
        print(f"[TEST_MOE] Non-sharded Max Diff: {np.max(non_sharded_diff)}")
        np.testing.assert_allclose(ref_out, non_sharded_np, rtol=1e-2, atol=1e-2)

        # Build JAX Mesh
        mesh = Mesh(np.array(devices).reshape(ep_size), ('expert',))

        # Create NamedSharding for weights along the expert dimension
        sharding_w = NamedSharding(mesh, P('expert', None, None))
        
        # Distribute / slice global weights to the 8 devices
        w1_sharded = jax.device_put(w1_global, sharding_w)
        w2_sharded = jax.device_put(w2_global, sharding_w)

        # Sharding specifications for the shard_map inputs
        w1_spec = P("expert", None, None)
        w2_spec = P("expert", None, None)
        input_partition_specs = (
            P(None, None),      # hidden_states (replicated)
            w1_spec,            # w1 (expert-sharded)
            w2_spec,            # w2 (expert-sharded)
            None,               # w1_scale (None)
            None,               # w2_scale (None)
            None,               # w1_bias (None)
            None,               # w2_bias (None)
            P(None, None),      # topk_weights (replicated)
            P(None, None),      # topk_ids (replicated)
        )

        # Define local JAX execution block with remapping
        @jax.jit
        def run_sharded_moe(hs, w1_s, w2_s, tw, ti):
            
            def _local_step(hs_l, w1_l, w2_l, w1_sc_l, w2_sc_l, w1_b_l, w2_b_l, tw_l, ti_l):
                # Remap global expert indices to local rank's expert indices inside shard_map
                # Using dynamic rank tracking:
                axis_env = jax.lax.axis_index(axis_name="expert")
                experts_start = axis_env * local_num_experts
                
                # remap global indices to local rank
                local_ids = ti_l - experts_start
                valid = (local_ids >= 0) & (local_ids < local_num_experts)
                
                # Mask out out-of-rank expert assignments
                tw_masked = jnp.where(valid, tw_l, 0.0)
                ti_masked = jnp.where(valid, local_ids, -1)
                
                local_out = fused_moe_func(
                    hs_l,
                    w1_l,
                    w2_l,
                    None,
                    None,
                    None,
                    None,
                    tw_masked,
                    ti_masked,
                    experts_start=None,  # Handled manual remap above to avoid tracer conflict
                    topk=topk,
                    activation=activation,
                    use_ep=True,
                    use_sparse_core=False,  # Replicated inputs path (GMM reference matmul)
                )
                # psum aggregate outputs across devices (re-syncs tokens)
                return jax.lax.psum(local_out, axis_name="expert")

            return shard_map(
                _local_step,
                mesh=mesh,
                in_specs=input_partition_specs,
                out_specs=P(None, None),
                check_vma=False,
            )(hs, w1_s, w2_s, None, None, None, None, tw, ti)

        # Execute
        sharded_out = run_sharded_moe(
            hidden_states,
            w1_sharded,
            w2_sharded,
            topk_weights.astype(hidden_states.dtype),
            topk_ids,
        )
        
        # Block until ready to fetch output
        sharded_out.block_until_ready()

        # Compare results
        out_np = np.array(sharded_out, dtype=np.float32)
        diff = np.abs(ref_out - out_np)
        print(f"[TEST_MOE] Max Diff: {np.max(diff)}")
        np.testing.assert_allclose(ref_out, out_np, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    absltest.main()
