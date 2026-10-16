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

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4.o_projection import (
    LANE,
    fused_reverse_rope_wo_a_projection,
    fused_reverse_rope_wo_a_projection_pair,
    gather_cos_sin,
    wo_a_projection,
)

from .rope_test import make_cos_sin_cache, rope_ref_impl

pytestmark = pytest.mark.tpu_test

# DeepSeek-V4: head_dim 512, qk_rope_head_dim 64, 8 heads per group.
HEAD_DIM = 512
ROTARY_DIM = 64
MAX_POSITION = 4096
HEADS_PER_GROUP = 8


def o_projection_ref_impl(activations, wo_a, wo_a_scale, *, num_groups):
    """``DeepseekV4Attention._o_proj``'s wo_a einsum, up to (not incl.) ``wo_b``."""
    num_tokens = activations.shape[0]
    o_f = activations.reshape(num_tokens, num_groups, -1)  # [t, g, d]
    reduction = o_f.shape[-1]
    w = wo_a.reshape(reduction, num_groups, -1)
    s = wo_a_scale.reshape(num_groups, -1)
    z = (
        jnp.einsum(
            "tgd,dgr->tgr",
            o_f,
            w.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        * s.astype(jnp.bfloat16)[None, ...]
    )
    return z.astype(jnp.bfloat16).reshape(num_tokens, -1)


def make_inputs(
    rng, *, num_tokens, num_groups, lora_rank, heads_per_group=HEADS_PER_GROUP
):
    """Activations in the ``[T, G * H, head_dim]`` view the kernel takes."""
    reduction = heads_per_group * HEAD_DIM
    num_heads = num_groups * heads_per_group
    activations = jnp.asarray(
        rng.standard_normal((num_tokens, num_heads, HEAD_DIM)),
        dtype=jnp.bfloat16,
    )
    wo_a = jnp.asarray(
        rng.standard_normal((reduction, num_groups * lora_rank)),
        dtype=jnp.float8_e4m3fn,
    )
    wo_a_scale = jnp.asarray(
        rng.uniform(0.5, 1.5, size=num_groups * lora_rank), dtype=jnp.float32
    )
    return activations, wo_a, wo_a_scale


# TODO: improve test coverage.
class OProjectionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(  # DeepSeek-V4-Flash, unsharded.
            testcase_name="dsv4_flash",
            num_tokens=256,
            num_groups=8,
            lora_rank=1024,
        ),
        dict(  # The benchmarked shape: activations [1024, 128, 512].
            testcase_name="dsv4_pro",
            num_tokens=1024,
            num_groups=16,
            lora_rank=1024,
        ),
    )
    def test_matches_reference(
        self,
        *,
        num_tokens,
        num_groups,
        lora_rank,
        tile_t=None,
        tile_r=None,
        sub_t=None,
    ):
        """The fused inverse-RoPE path against ``rope`` + the wo_a einsum."""
        rng = np.random.default_rng(0)
        activations, wo_a, wo_a_scale = make_inputs(
            rng,
            num_tokens=num_tokens,
            num_groups=num_groups,
            lora_rank=lora_rank,
        )
        positions = jnp.asarray(
            rng.integers(0, MAX_POSITION, size=(num_tokens,)), dtype=jnp.int32
        )
        cos_sin_cache = jnp.asarray(
            make_cos_sin_cache(MAX_POSITION, ROTARY_DIM), dtype=jnp.float32
        )
        cos_sin = gather_cos_sin(positions, cos_sin_cache, inverse=True)
        self.assertEqual(cos_sin.shape, (num_tokens, 2 * LANE))
        out = wo_a_projection(
            activations,
            wo_a,
            wo_a_scale,
            cos_sin,
            tile_t=tile_t,
            tile_r=tile_r,
            sub_t=sub_t,
            quantize_activations=False,
        )

        roped = rope_ref_impl(activations, positions, cos_sin_cache, inverse=True)
        expected = o_projection_ref_impl(roped, wo_a, wo_a_scale, num_groups=num_groups)

        np.testing.assert_allclose(
            np.asarray(out, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )

    @parameterized.product(num_tokens=(8, 32, 64, 128, 256), rank=(0, 1))
    def test_pair_matches_fused_bitwise(self, *, num_tokens, rank):
        """The pair op == the fused op on the assembled [2T, G*H, D] input.

        Rank 0's first half is its own all-head `o` (heads [0, G*H)) and its
        second half is rows [T, 2T) of a gathered buffer; rank 1's first half
        is gathered rows [0, T) and its second half its `o` from head G*H.
        """
        num_groups, lora_rank = 8, 1024
        heads = num_groups * HEADS_PER_GROUP
        rng = np.random.default_rng(2)
        own = jnp.asarray(
            rng.standard_normal((num_tokens, 2 * heads, HEAD_DIM)), jnp.bfloat16
        )
        gathered = jnp.asarray(
            rng.standard_normal((2 * num_tokens, heads, HEAD_DIM)), jnp.bfloat16
        )
        _, wo_a, wo_a_scale = make_inputs(
            rng, num_tokens=1, num_groups=num_groups, lora_rank=lora_rank
        )
        positions = jnp.asarray(
            rng.integers(0, MAX_POSITION, size=(2 * num_tokens,)), dtype=jnp.int32
        )
        cos_sin_cache = jnp.asarray(
            make_cos_sin_cache(MAX_POSITION, ROTARY_DIM), dtype=jnp.float32
        )
        if rank == 0:
            assembled = jnp.concatenate([own[:, :heads], gathered[num_tokens:]])
            first, second, second_row0, second_head0 = own, gathered, num_tokens, 0
        else:
            assembled = jnp.concatenate([gathered[:num_tokens], own[:, heads:]])
            first, second, second_row0, second_head0 = gathered, own, 0, heads
        expected = fused_reverse_rope_wo_a_projection(
            assembled, positions, cos_sin_cache, wo_a, wo_a_scale, head_dim=HEAD_DIM
        )
        actual = fused_reverse_rope_wo_a_projection_pair(
            first,
            second,
            positions,
            cos_sin_cache,
            wo_a,
            wo_a_scale,
            first_row0=0,
            first_head0=0,
            second_row0=second_row0,
            second_head0=second_head0,
            num_heads=heads,
            head_dim=HEAD_DIM,
        )
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16)
        )


if __name__ == "__main__":
    absltest.main()
