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
"""GLM5_MK_TOPK_ROLL: vtopk.threshold_roll + extract_roll (rolled over 128-row key blocks) against
the megakernel's unrolled slow path (key mask, threshold passes, extract), bitwise, in Pallas
interpret mode on CPU. Random keys, heavy ties at the threshold (the position search), all-ties,
fewer valid keys than K, and several seq_len / ROWS combinations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.glm5_megakernel import vtopk

INT_MIN = vtopk.INT_MIN


def _run(keys, seq_len, K, roll, kc):
    ROWS = keys.shape[0]

    def kern(sl_ref, k_in, idx_out, keys_s, acc, cc):
        keys_s[...] = k_in[...]
        seq = sl_ref[0]
        if roll:
            T = vtopk.threshold_roll(keys_s, K, seq)
            vtopk.extract_roll(keys_s, T, K, idx_out, acc, cc, kc=kc)
        else:      # kernel.py topk() slow path, unrolled
            p_ = (jax.lax.broadcasted_iota(jnp.int32, (ROWS, 128), 0) * 128
                  + jax.lax.broadcasted_iota(jnp.int32, (ROWS, 128), 1))
            keys_s[...] = jnp.where(p_ >= seq, INT_MIN, keys_s[...])
            T = vtopk.threshold(keys_s, K)
            vtopk.extract(keys_s, T, K, idx_out, kc=kc)

    return pl.pallas_call(
        kern, out_shape=jax.ShapeDtypeStruct((K // 128, 128), jnp.int32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((ROWS, 128), jnp.int32), pltpu.VMEM((kc, 256), jnp.float32),
                        pltpu.VMEM((2, 128, 128), jnp.float32)],
        interpret=True)(jnp.asarray([seq_len], jnp.int32), jnp.asarray(keys))


CASES = [  # (ROWS, K, seq_len, mode)
    (256, 256, 30000, "rand"), (256, 256, 20000, "ties"), (384, 512, 49152, "rand"),
    (384, 512, 40000, "ties"), (256, 256, 200, "rand"), (256, 256, 30000, "allties"),
    (512, 256, 65536, "ties"),
]


@pytest.mark.parametrize("rows,K,seq_len,mode", CASES)
def test_topk_roll_bitwise(rows, K, seq_len, mode):
    rng = np.random.default_rng(rows * 7 + K + seq_len)
    if mode == "rand":
        keys = rng.integers(-2 ** 31, 2 ** 31 - 1, (rows, 128), dtype=np.int64).astype(np.int32)
    elif mode == "ties":        # few distinct values: many ties at the threshold
        keys = rng.integers(0, 6, (rows, 128)).astype(np.int32) * 1000
    else:
        keys = np.full((rows, 128), 7, np.int32)
    keys[rng.random((rows, 128)) < 0.05] = INT_MIN      # some invalid scores
    kc = min(256, K)
    ref = np.asarray(_run(keys, seq_len, K, False, kc))
    got = np.asarray(_run(keys, seq_len, K, True, kc))
    assert np.array_equal(got, ref), (np.argwhere(got != ref)[:5], got.ravel()[:8], ref.ravel()[:8])
    assert (ref >= 0).sum() == min(K, int((keys.ravel()[:seq_len] != INT_MIN).sum())) or mode != "rand"
