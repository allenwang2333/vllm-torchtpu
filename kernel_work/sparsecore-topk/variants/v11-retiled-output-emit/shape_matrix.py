"""AOT-compile every frozen-test shape id on both arms.

`probes/17` closed feasibility at ONE shape (b=1024, n=10240, k=2048, P=1).
The public entry also reaches `_sc_topk_direct` twice on the P>1 path -- stage 1
at (b*p, n_p) with `emit_keys=True`, stage 2 at (b, p*k_p) -- and neither had
ever been AOT-compiled with a 2-D `out_type`. This compiles all seven ids plus
the emit_keys arm, and counts the retile ops XLA is left holding.

STATIC COMPILE-TIME ANALYSIS ONLY. No TPU, no timing.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

from jax._src.pallas.mosaic import tpu_info as _t

_TPU7X = _t.get_tpu_info_for_chip(_t.ChipVersion.TPU_7X, 1)
_t.get_tpu_info = lambda: _TPU7X          # type: ignore[assignment]
_t.is_tpu_device = lambda: True           # type: ignore[assignment]
_t.get_device_kind = lambda: "TPU7x"      # type: ignore[assignment]
_t.get_num_device_cores = lambda: 1       # type: ignore[assignment]

import jax                                # noqa: E402
import jax.numpy as jnp                   # noqa: E402
from jax.experimental import topologies   # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding    # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X        # type: ignore[assignment]

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[3]
SRC = {"base": REPO / "src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py",
       "v11": HERE / "sparsecore_topk.py"}
N_SHORT, N_128K = 10240, 133120
SHAPES = [("p1-b64-n10240", 64, N_SHORT, 2048),
          ("p2-b16-n10240", 16, N_SHORT, 2048),
          ("p4-b16-n133120", 16, N_128K, 2048),
          ("p4-b64-n133120", 64, N_128K, 2048),
          ("p4-b16-n133120-k512", 16, N_128K, 512),
          ("p4-b1024-n133120", 1024, N_128K, 2048),
          ("p1-b1024-n10240", 1024, N_SHORT, 2048)]


def load(name):
    spec = importlib.util.spec_from_file_location(f"sm_{name}", SRC[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def counts(txt):
    """Retile work XLA is left holding, and the kernel's own result shapes."""
    ent = re.search(r"^ENTRY .*?^}", txt, re.S | re.M)
    ent = ent.group(0) if ent else txt
    body = [l for l in ent.splitlines() if " = " in l]
    return dict(
        entry_instrs=len(body),
        entry_reshape=sum(" reshape(" in l for l in body),
        entry_copy=sum(" copy(" in l or " copy-start(" in l for l in body),
        kernel_out_shapes=sorted({m.group(1) for m in re.finditer(
            r"call-done = (\S+) call-done\(%sc_topk_direct", txt)}),
    )


def main():
    sh = SingleDeviceSharding(topologies.get_topology_desc(
        topology_name="tpu7x:2x2x1", platform="tpu").devices[0])
    rows = {}
    for name in ("v11", "base"):
        mod = load(name)
        for sid, b, n, k in SHAPES:
            # `k` is the SECOND positional parameter and is a static int;
            # `row_lengths` is the third. Passing them the other way round
            # makes `k` a tracer and jit rejects it as a static argument.
            f = jax.jit(lambda s, rl, m=mod, k=k: m.sparsecore_topk(s, k, rl))
            try:
                txt = f.lower(
                    jax.ShapeDtypeStruct((b, n), jnp.float32, sharding=sh),
                    jax.ShapeDtypeStruct((b, ), jnp.int32, sharding=sh),
                ).compile().as_text()
                rows[(sid, name)] = counts(txt)
            except Exception as e:  # noqa: BLE001
                rows[(sid, name)] = {"FAILED": f"{type(e).__name__}: {e}"[:300]}
            print(f"{sid:24s} {name:5s} {json.dumps(rows[(sid, name)])}",
                  flush=True)
    print("\n=== base -> v11, per shape id ===")
    for sid, *_ in SHAPES:
        a, c = rows[(sid, "base")], rows[(sid, "v11")]
        if "FAILED" in a or "FAILED" in c:
            print(f"{sid:24s} FAILED somewhere"); continue
        print(f"{sid:24s} ENTRY {a['entry_instrs']:3d} -> {c['entry_instrs']:3d}"
              f"   reshape {a['entry_reshape']} -> {c['entry_reshape']}"
              f"   copy {a['entry_copy']} -> {c['entry_copy']}"
              f"   kernel out {a['kernel_out_shapes']} -> {c['kernel_out_shapes']}")


if __name__ == "__main__":
    main()
