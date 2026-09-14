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
"""What the fused EP MoE kernel's VMEM accounting promises about shard size.

The kernel refuses to build when its declared buffers do not fit, and that
refusal is a hard raise: an over-budget shard takes the server down at load
rather than falling back. So which real models fit is a property worth
holding still, and it is a property of pure arithmetic -- shapes, dtypes and
a device record -- which is why these run on any host, with or without a
chip. The kernel tests beside them need four.

The device record is a stand-in for TPU v7x rather than a queried one, for
the same reason: `pltpu.get_tpu_info()` needs a chip, and a test that
silently skipped off-TPU would leave exactly the arithmetic that decides
whether DeepSeek-V4-Pro can serve at all with no coverage anywhere.
"""

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402

MIB = 1024 * 1024


class _V7x:
    """The v7x fields `array_vmem_bytes` and `vmem_limit` read."""

    generation = 7
    num_lanes = 128
    vmem_capacity_bytes = 64 * MIB

    def get_sublane_tiling(self, dtype):
        return {32: 8, 16: 16, 8: 32}[jnp.dtype(dtype).itemsize * 8]


INFO = _V7x()
LIMIT = int(INFO.vmem_capacity_bytes * host.VMEM_FRACTION)

# The two DeepSeek-V4 shards this kernel is being brought up on: (local
# experts, hidden, intermediate). Pro is the target and Flash is the
# four-chip vehicle it is developed against.
PRO_EP16 = (24, 7168, 3072)
FLASH_EP4 = (64, 4096, 2048)


def _estimate(shard, nbuf, capacity=128):
    g_local, hidden, inter = shard
    return host.vmem_estimate_bytes(g_local,
                                    capacity,
                                    hidden,
                                    inter,
                                    nbuf=nbuf,
                                    weight_format=host.WeightFormat.FP4,
                                    rhs_qb=512,
                                    info=INFO)


def _scratch(shard, nbuf, capacity=128):
    g_local, hidden, inter = shard
    return {
        name: (shape, dtype)
        for name, shape, dtype in host.vmem_scratch_arrays(
            g_local,
            capacity,
            hidden,
            inter,
            nbuf=nbuf,
            weight_format=host.WeightFormat.FP4,
            rhs_qb=512)
    }


@pytest.mark.parametrize("nbuf", [1, 2, 3])
def test_scale_tables_are_slot_deep_not_expert_deep(nbuf):
    """The four-bit scale tables ride a weight slot, not a resident table.

    They used to be `g_local` deep, which on Pro's EP16 shard cost 14.2MiB
    of a 62.7MiB budget to hold tables the kernel reads one expert of at a
    time. If they ever go back to being expert-deep the shard stops fitting
    and the failure is a build-time raise on a sixteen-host job.
    """
    for shard in (PRO_EP16, FLASH_EP4):
        arrays = _scratch(shard, nbuf)
        for name in ("w1s_vm", "w2s_vm"):
            shape, _ = arrays[name]
            assert shape[0] == nbuf, (
                f"{name} on a {shard[0]}-expert shard is {shape[0]} deep, "
                f"expected the {nbuf} weight slots")


def test_estimate_is_flat_in_local_expert_count():
    """Nothing left in the four-bit estimate scales with `g_local`.

    The corollary of the test above, stated as the thing that actually
    matters: doubling a shard's experts must not cost VMEM, because that is
    what lets a model be served at a smaller expert-parallel width.
    """
    hidden, inter = 7168, 3072
    sizes = {
        host.vmem_estimate_bytes(g,
                                 capacity=128,
                                 hidden=hidden,
                                 inter=inter,
                                 nbuf=2,
                                 weight_format=host.WeightFormat.FP4,
                                 rhs_qb=512,
                                 info=INFO)
        for g in (12, 24, 48, 96)
    }
    assert len(sizes) == 1, (
        f"the four-bit estimate still varies with the local expert count: "
        f"{sorted(s / MIB for s in sizes)} MiB")


def test_deepseek_v4_pro_fits_single_buffered():
    """Pro at EP16 fits, and only without a prefetch runway.

    Two of Pro's expert slabs are 63MiB against a 62.7MiB budget, so no
    tuning knob reaches a double-buffered build -- the single-buffered one
    is the whole of what is available, and this is the assertion that says
    the margin is real rather than incidental.
    """
    single = _estimate(PRO_EP16, nbuf=1)
    assert single <= LIMIT, (
        f"DeepSeek-V4-Pro EP16 needs {single/MIB:.1f}MiB single-buffered, "
        f"over the {LIMIT/MIB:.1f}MiB budget")
    assert _estimate(PRO_EP16, nbuf=2) > LIMIT, (
        "Pro now fits double-buffered; the comment above and the "
        "recipe's TPU_MOE_FUSED_EP_PREFETCH_DISTANCE=0 are both stale")


def test_deepseek_v4_flash_fits_at_the_default_distance():
    """Flash at EP4 fits with the module's default prefetch distance.

    It did not before the scale tables were streamed -- 64.4MiB against
    62.7 -- and the recipe carried a distance override to get under. This
    is what says the override is no longer load-bearing.
    """
    est = _estimate(FLASH_EP4, nbuf=host.WEIGHT_PREFETCH_DISTANCE + 1)
    assert est <= LIMIT, (
        f"DeepSeek-V4-Flash EP4 needs {est/MIB:.1f}MiB at the default "
        f"prefetch distance {host.WEIGHT_PREFETCH_DISTANCE}, over the "
        f"{LIMIT/MIB:.1f}MiB budget")


def test_prefetch_distance_zero_is_one_slot():
    """Distance and slot count stay in the relation the kernel asserts.

    `_build_fused_ep_moe_kernel` asserts `0 <= distance < NBUF`; the module
    derives one from the other, so the two can only disagree by an edit to
    this line, and distance zero is the edge the Pro build sits on.
    """
    assert host.NBUF == host.WEIGHT_PREFETCH_DISTANCE + 1
    assert host.WEIGHT_PREFETCH_DISTANCE >= 0


def test_estimate_sums_the_declared_arrays():
    """The estimate is the scratch list plus the tile body, and nothing else.

    `vmem_scratch_arrays` is also what the kernel declares its scratch from,
    so this is the host-side half of "a buffer cannot be resized on one side
    only": if the estimate ever stops summing that exact list, the number
    the build checks against stops describing the buffers it builds.
    """
    g_local, hidden, inter = PRO_EP16
    arrays = list(
        host.vmem_scratch_arrays(g_local,
                                 128,
                                 hidden,
                                 inter,
                                 nbuf=1,
                                 weight_format=host.WeightFormat.FP4,
                                 rhs_qb=512))
    arrays += list(
        host.vmem_tile_body_arrays(128,
                                   hidden,
                                   inter,
                                   weight_format=host.WeightFormat.FP4,
                                   rhs_qb=512))
    summed = sum(
        host.array_vmem_bytes(shape, dtype, INFO)
        for _, shape, dtype in arrays)
    assert summed == _estimate(PRO_EP16, nbuf=1)
