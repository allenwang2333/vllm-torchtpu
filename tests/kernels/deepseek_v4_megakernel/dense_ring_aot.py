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
"""Deviceless tpu7x compile of the dense ring kernel at DSv4-Pro size.

    python -m tests.kernels.deepseek_v4_megakernel.dense_ring_aot --tp 16

Reports Mosaic lowering success and the generated code size (needs libtpu; no
TPU device).
"""

import argparse
import os

os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import topologies  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_ring  # noqa: E402


def specs(cfg, sharding=None):
    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)

    tiles = {}
    for n, s in cfg.streams.items():
        nl = max(1, len(cfg.stream_layers(n)))
        tiles[n] = sds((nl, s.n_tiles, s.tile_rows, s.tile_cols), jnp.dtype(s.dtype))
    m = cfg.rows
    inputs = {
        "x": sds((m, cfg.hidden), jnp.bfloat16),
        "res": sds((m, cfg.hc_hidden), jnp.bfloat16),
        "post": sds((m, cfg.hc_mult), jnp.float32),
        "comb": sds((m, cfg.hc_mult**2), jnp.float32),
        "cos_sin": sds((m, 256), jnp.float32),
    }
    smalls = sds((cfg.n_layers, cfg.smalls_rows, 128), jnp.float32)
    return tiles, smalls, inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=16)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--banks", type=int, default=10)
    p.add_argument("--layers", type=int, default=0, help="0 = all 61")
    p.add_argument("--no-probes", action="store_true")
    a = p.parse_args()
    kw = dict(tp=a.tp, rows=a.rows, n_banks=a.banks)
    if a.layers:
        kw["ratios"] = dc.DSV4_PRO_RATIOS[: a.layers]
    cfg = dc.DenseChainConfig(**kw)
    print(dc.describe(cfg))
    topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1")
    sh = jax.sharding.SingleDeviceSharding(topo.devices[0])
    run = dense_ring.build(cfg, probes=not a.no_probes)
    c = jax.jit(run).lower(*specs(cfg, sh)).compile()
    ma = c.memory_analysis()
    print(
        f"RESULT compiled tp={a.tp} rows={a.rows} code_bytes="
        f"{ma.generated_code_size_in_bytes} temp={ma.temp_size_in_bytes}"
    )


if __name__ == "__main__":
    main()
