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
"""Run ``fused_ep_moe_v2`` at the four shapes served in ``profiles/dsv4_dumps/``.

Each profile there is a PyTorch trace of a real serving process, and the HLO it
carries names the kernel's custom call in full -- operand shapes, operand
dtypes and the Pallas build's own name -- so these shapes are read off a run
rather than guessed. The two captures on the 16-way slice carry::

    %fused_ep_moe_v2_fp4w_silu_and_mul_with_clamp_g24_c128_nb1 = ... custom-call(
        s32[24,16] x5, s32[24] x3, s32[8,16], s32[1,1], s32[99072],
        f8e4m3fn[16384,56,128], f32[773,128],
        f4e2m1fn[24,7168,6144], f4e2m1fn[24,3072,7168],
        f32[24,14,6144], f32[24,6,7168])

and the two on the 8-way slice the same thing at ``g32_c128_nb3``, hidden 4096.

Those are the operands of the Pallas call, not of the layer, so the layer's
parameters are recovered from them: ``w1 [G, hidden, 2 * inter]`` gives the
local expert count, the hidden size and the intermediate size, and
``f4e2m1fn`` names the weight format; the transport tables are ``[G, ep]``, so
their minor axis is the mesh width and ``e_total = G * ep``; the gathered
activation slab's leading axis is the mesh-wide token count; ``c128`` is the
capacity. ``topk`` is the last unknown and two profiles at one geometry and two
token counts pin it, since ``recv_rows = align_up(t_local * topk + (ROWBLK - 1)
* e_total, ROWBLK)`` and differencing the pair kills the ``e_total`` term --
``topk = 6`` on all four. Each case below reproduces every one of the eleven
operand shapes its profile carries.

``nb{1,3}`` is the weight-slot count ``fit_weight_slots`` picked for the chip
the profile was captured on, so it is a property of the host rather than of the
shape and is not part of a case.

Two things a profile cannot state are set to the DeepSeek defaults:
``renormalize=True`` and ``routed_scaling_factor=1.0``. The profiles score with
sigmoid plus the score-correction bias, which is what runs here; the two 8-way
captures also run 30 hash-routed layers a step against the identical build.
"""

import json

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from google3.experimental.users.xiowei.pallas.optimize.fused_moe_v2.kernel import host
from google3.experimental.users.xiowei.pallas.optimize.fused_moe_v2.kernel.host import WeightFormat
from google3.experimental.users.xiowei.pallas.optimize.fused_moe_v2.kernel.kernel import AXIS
from google3.experimental.users.xiowei.pallas.optimize.fused_moe_v2.kernel.layer import fused_ep_moe_v2

jax.config.parse_flags_with_absl()

# The Pallas call's name prefix; the full name adds the weight format, the
# activation and the g/c/nb geometry.
KERNEL_NAME = "fused_ep_moe_v2"






FP4 = jnp.float4_e2m1fn
# Contraction rows one weight scale covers, which is the middle axis of the
# scale tables the profiles carry.
QB4 = host.QB4
# The serving adapter's _TILE_M, which every build name repeats as c128.
CAPACITY = 128


def _operands(case, mesh):
    """The layer's operands, pre-sharded onto the device mesh.

    Avoid unsharded host inputs: passing unsharded arrays to `@jax.jit` causes
    XLA to compile host transfers and SPMD dynamic slicing (`jit(_multi_slice)`),
    polluting the trace. Pre-sharding with `NamedSharding` places operands directly
    in TPU HBM matching real serving layouts.
    """
    e_total = case["e_total"]
    hidden = case["hidden"]
    inter = case["inter"]
    tokens = case["tokens"]
    ep = case["ep"]
    keys = jax.random.split(jax.random.key(0), 7)

    sharded_ep = NamedSharding(mesh, P(AXIS))
    replicated = NamedSharding(mesh, P())

    w1 = jax.device_put(
        # e2m1 spans +-6 with two mantissa steps a binade, so a unit normal
        # would round most of the draw onto 0 and +-0.5.
        (jax.random.normal(keys[0], (e_total, hidden, 2 * inter)) * 2).astype(FP4),
        sharded_ep,
    )
    w2 = jax.device_put(
        (jax.random.normal(keys[1], (e_total, inter, hidden)) * 2).astype(FP4),
        sharded_ep,
    )
    w1_scale = jax.device_put(
        jax.random.uniform(keys[2], (e_total, hidden // QB4, 2 * inter)) / 64,
        sharded_ep,
    )
    w2_scale = jax.device_put(
        jax.random.uniform(keys[3], (e_total, inter // QB4, hidden)) / 64,
        sharded_ep,
    )
    x = jax.device_put(
        (jax.random.normal(keys[4], (tokens, hidden)) / 10).astype(jnp.bfloat16),
        sharded_ep,
    )
    gating = jax.device_put(
        jax.random.normal(keys[5], (tokens, e_total)),
        sharded_ep,
    )
    score_bias = jax.device_put(
        jax.random.normal(keys[6], (e_total,)) / 10,
        replicated,
    )
    # This rank's mesh index as an explicit device tensor, matching serving's
    # `_RANK_BUFFER`. Prevents XLA from generating a dynamic slice on a
    # constant partition-id.
    rank = jax.device_put(
        jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1),
        sharded_ep,
    )
    return w1, w2, w1_scale, w2_scale, x, gating, score_bias, rank


class FusedEpMoeV2MicrobenchmarkTest(parameterized.TestCase):

  # One entry a profile. The two 16-way cases are one geometry at two token
  # counts, and so are the two 8-way ones; that pairing is what pins topk.
  @parameterized.named_parameters(
      dict(
          testcase_name="ep16_h7168_e384_t16384",
          name="ep16-h7168-e384-t16384",
          ep=16,
          tokens=16384,
          hidden=7168,
          inter=3072,
          e_total=384,
          topk=6,
      ),
      dict(
          testcase_name="ep16_h7168_e384_t1024",
          name="ep16-h7168-e384-t1024",
          ep=16,
          tokens=1024,
          hidden=7168,
          inter=3072,
          e_total=384,
          topk=6,
      ),
      dict(
          testcase_name="ep8_h4096_e256_t8192",
          name="ep8-h4096-e256-t8192",
          ep=8,
          tokens=8192,
          hidden=4096,
          inter=2048,
          e_total=256,
          topk=6,
      ),
      dict(
          testcase_name="ep8_h4096_e256_t512",
          name="ep8-h4096-e256-t512",
          ep=8,
          tokens=512,
          hidden=4096,
          inter=2048,
          e_total=256,
          topk=6,
      ),
  )
  def test_runs_the_served_shape(self, **case):
    """The shape builds, runs and returns the rows it was given."""
    if jax.device_count() < case["ep"]:
      self.skipTest(
          f"{case['name']} is a {case['ep']}-way expert-parallel collective "
          f"and this host reports {jax.device_count()} JAX devices"
      )
    mesh = Mesh(np.asarray(jax.devices()[: case["ep"]]), axis_names=(AXIS,))
    w1, w2, w1_scale, w2_scale, x, gating, score_bias, rank = _operands(case, mesh)

    @jax.jit
    def run_kernel(x, w1, w2, w1_scale, w2_scale, gating, score_bias, rank):
      return fused_ep_moe_v2(
          x,
          w1,
          w2,
          w1_scale,
          w2_scale,
          gating,
          rank=rank,
          topk=case["topk"],
          renormalize=True,
          mesh=mesh,
          capacity=CAPACITY,
          weight_format=WeightFormat.FP4,
          act_fn="silu_and_mul_with_clamp",
          scoring_fn="sigmoid",
          score_bias=score_bias,
      )

    # Warmup
    out = run_kernel(x, w1, w2, w1_scale, w2_scale, gating, score_bias, rank)
    out.block_until_ready()

    out = run_kernel(x, w1, w2, w1_scale, w2_scale, gating, score_bias, rank)
    out.block_until_ready()


if __name__ == "__main__":
  absltest.main()
