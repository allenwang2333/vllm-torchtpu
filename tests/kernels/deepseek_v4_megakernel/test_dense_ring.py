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
"""Dense weight ring (DSv4 megakernel block B2): CPU-interpret numerics.

Reference: the same formulas in plain JAX with the kernel's f32 summation
order. Up to the first Sinkhorn output the kernel is bit-exact (layer 0
qkv / xproj probes); after it, XLA:CPU codegen of the same ops differs in
rare last ulps, which the next layers' fp8 activation quantization
amplifies. The bound there is the reassociation band: the distance of the
stock-order reference from the kernel-order one. The positive controls (a
wrong tile order, a refill before read) land far outside it.
"""

import json
import os
import subprocess
import sys

import pytest

_CHECK = "tests.kernels.deepseek_v4_megakernel.dense_ring_interpret_check"


def _run(*args: str) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-m", _CHECK, *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
    ).stdout  # fmt: skip
    line = next(ln for ln in out.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("tp", [1, 4])
def test_matches_kernel_order_reference(tp):
    r = _run("--tp", str(tp))
    assert not r["races"], r
    assert r["layer0_probes"] == 0.0, r
    assert r["x_rel"] <= r["band_x_rel"] and r["res_rel"] <= r["band_res_rel"], r


def test_positive_control_wrong_tile_order():
    r = _run("--fault", "swap")
    assert r["x_rel"] > 3 * r["band_x_rel"], r


def test_positive_control_refill_before_read():
    # 64 heads give wq_b 8 tiles for 4 banks: the early refill of a bank
    # with the stream's own next tile clobbers the tile about to be read.
    r = _run("--fault", "early", "--heads", "64")
    assert r["races"], r
    assert r["x_rel"] > 3 * r["band_x_rel"], r
