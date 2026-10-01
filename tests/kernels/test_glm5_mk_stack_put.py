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
"""GLM5_MK_STACK_W stack put (Glm5Megakernel._put_op) on an emulated 16-device EP mesh (CPU).

j-427279cb: with the stack sharded [{}, {"d"}, {}] over EP16, the put op's module was specialised
on the global stack ([78, 16 * k, ...]) and torch_tpu, which calls it with this rank's shard,
refused it ("invalid refinement ... 78x32768x1024 -> 78x2048x1024"). For every owned kind, sharded
and replicated, this builds the op exactly as the adapter does (sharded_jax_callable over the
adapter's input specs, placeholders from per-rank torch tensors) and checks:
  - the update in the exported module is on this rank's shard shape, never the global one;
  - writing every row reproduces the unstacked per-layer operands bitwise, both through the jitted
    callable and through its jax.export (as torch_tpu registers it), globally and per device.
The body runs in a subprocess so the 16 host devices exist whatever jax state pytest already has.
"""

import os
import re
import subprocess
import sys

N_DEV = 16
N_LAYERS = 5
K_LOCAL = 4      # rows of one rank's shard of a sharded kind (the 2048 of q_b at EP16)
COLS = 256


def _main() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax.sharding import Mesh, NamedSharding
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from vllm_torchtpu.kernels.glm5_megakernel import vllm_adapter as A

    assert jax.device_count() >= N_DEV, jax.devices()
    mesh = Mesh(np.array(jax.devices()[:N_DEV]), (A.EP_AXIS_NAME,))
    dtypes = {"q_b": torch.float8_e4m3fn, "w_uk": torch.bfloat16, "o_proj": torch.float8_e4m3fn,
              "rw13": torch.float8_e4m3fn, "rs2": torch.float32, "qkv_a": torch.float8_e4m3fn,
              "gate": torch.float32, "idx_wk": torch.bfloat16}
    kinds = sorted(set(A.OWNED_KINDS) & set(dtypes))
    assert any(k in A.SHARDED_KINDS for k in kinds) and any(k not in A.SHARDED_KINDS for k in kinds)
    ad = A.Glm5Megakernel.__new__(A.Glm5Megakernel)     # only the put-op state
    ad.mesh, ad._put_ops = mesh, {}
    bad = []
    rng = np.random.default_rng(0)
    for kind in kinds:
        sh = kind in A.SHARDED_KINDS
        tdt = dtypes[kind]
        jdt = pallas_impl.TORCH_TO_JAX_DTYPE_MAP[tdt]
        k_glob = K_LOCAL * N_DEV if sh else K_LOCAL
        op = ad._put_op(kind)
        specs = op.input_partition_specs
        # this rank's torch tensors, as the adapter passes them (stack, x, row)
        local = [torch.zeros((N_LAYERS, K_LOCAL, COLS), dtype=tdt),
                 torch.zeros((K_LOCAL, COLS), dtype=tdt), torch.zeros((1,), dtype=torch.int32)]
        ph = pallas_impl.jax_placeholders(local, mesh=mesh, partition_specs=specs)
        assert ph[0].shape == (N_LAYERS, k_glob, COLS), (kind, ph[0].shape)
        exp = op.exported(*ph)
        txt = exp.mlir_module()
        ty = {"float8_e4m3fn": "f8E4M3FN", "bfloat16": "bf16", "float32": "f32"}[jnp.dtype(jdt).name]
        dus = re.findall(r"stablehlo\.dynamic_update_slice[^\n]*->\s*tensor<([0-9x]+)x" + ty, txt)
        want = f"{N_LAYERS}x{K_LOCAL}x{COLS}"
        if not dus or any(d != want for d in dus):
            bad.append(f"{kind}: update on tensor<{dus}> in the exported module, want <{want}> "
                       "(the rank's shard)")
        # bitwise: every row written == the unstacked per-layer operand
        # random bit patterns, finite only (CPU XLA canonicalizes NaN payloads in bf16 copies)
        rows = []
        for _ in range(N_LAYERS):
            x = (rng.integers(0, 256, (k_glob, COLS * jnp.dtype(jdt).itemsize), np.uint8)
                 .view(jnp.dtype(jdt)).reshape(k_glob, COLS))
            rows.append(np.where(np.isfinite(x.astype(np.float32)), x, np.zeros((), x.dtype)))
        shs = [NamedSharding(mesh, s) for s in specs]
        # the same export (op.exported is TPU-only), lowered for the CPU devices to run it here
        call_exp = jax.export.export(op.jit_fn, platforms=["cpu"])(*ph).call
        for path, f in (("jit", op.jit_fn), ("export", call_exp)):
            stk = jax.device_put(jnp.zeros((N_LAYERS, k_glob, COLS), jdt), shs[0])
            for r, x in enumerate(rows):
                stk = f(stk, jax.device_put(jnp.asarray(x), shs[1]),
                        jax.device_put(jnp.asarray([r], jnp.int32), shs[2]))
            got = np.asarray(stk).view(np.uint8)
            ref = np.stack(rows).view(np.uint8)
            if not np.array_equal(got, ref):
                bad.append(f"{kind} [{path}]: stack != stacked per-layer operands")
            for s in stk.addressable_shards:
                d = s.device.id
                sl = slice(d * K_LOCAL, (d + 1) * K_LOCAL) if sh else slice(None)
                if (np.asarray(s.data).shape != (N_LAYERS, K_LOCAL, COLS) or not np.array_equal(
                        np.asarray(s.data).view(np.uint8), ref[:, sl])):
                    bad.append(f"{kind} [{path}]: device {d} shard != its per-layer rows")
                    break
        print(f"KIND {kind} sharded={sh} dus={dus}", flush=True)
    for b in bad:
        print("BAD", b, flush=True)
    print(f"STACK_PUT_BAD={len(bad)}", flush=True)
    sys.exit(1 if bad else 0)


def test_stack_put_local_shard_bitwise():
    import pytest

    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "")
                          + f" --xla_force_host_platform_device_count={N_DEV}").strip())
    p = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, capture_output=True,
                       text=True, timeout=900)
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-40:])
    if "No module named 'torch_tpu'" in tail:
        pytest.skip("torch_tpu not installed")
    assert p.returncode == 0 and "STACK_PUT_BAD=0" in p.stdout, tail


if __name__ == "__main__":
    _main()
