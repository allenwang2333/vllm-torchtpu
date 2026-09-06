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
"""Option C, TensorCore half only: the scores buffer is written as bf16.

This is NOT option C. It is C's TensorCore half with C's SparseCore half left
undone, and the difference is deliberate and is the point.

  incumbent   TC writes 285 MB f32          SC reads 285 MB
  this        TC writes 142 MB bf16         XLA widens (reads 142, writes 285)
                                            SC reads 285 MB
  option C    TC writes 142 MB packed i32   SC reads 142 MB

In bytes moved per `mixed_64k` call, relative to the incumbent, option C is
-285 MB and this arm is +285 MB. They are mirror images. So the measured
`this - incumbent` gap prices the marginal cost of a megabyte of scores
traffic at this shape *from the wrong side*, and C's prize can be read off it
without building the packing, the SparseCore unpack or the index remap.

That inference assumes time is locally linear in scores bytes, which is
exactly what measurement_and_design.md 7's VPU-saturation hypothesis
disputes. So the arm is worth running for either answer: if the gap is large,
C's prize is real and worth the build; if the gap is ~0, halving the buffer
buys nothing and C is dead for the cost of a one-line patch.

Correctness: this arm rounds the scores to bf16 on the way to HBM, so it is a
real half-width key and the accuracy gate in `keyfmt_probe.py --mode margins`
covers it directly (bf16 = 7 explicit mantissa bits). Unlike the probe's
`.astype` round trip, the rounding here CANNOT be elided by
`--xla_allow_excess_precision`: the value is written to HBM as bf16 by a DMA,
so the narrow bits are the only bits that exist.
"""

import jax.numpy as jnp  # noqa: F401  (referenced by the patched source)

from _load import load_patched

BASE = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"

SUBS = [
    (
        "    def _common_path(_):\n"
        "        out_dtype = jnp.float32\n",
        "    def _common_path(_):\n"
        "        out_dtype = jnp.bfloat16  # HALF-WIDTH KEY (option C, TC "
        "half)\n",
    ),
]

_mod = load_patched(BASE, "streamindex_topk_bf16out", SUBS)

streamindex_topk = _mod.streamindex_topk
kernel = _mod.kernel
