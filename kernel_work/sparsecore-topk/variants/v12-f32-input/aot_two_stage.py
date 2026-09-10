"""Does the two-stage (P>1) path still lower after the f32-input edit?

STATIC COMPILE-TIME ANALYSIS ONLY.

`aot_check.py` compiles `_sc_topk_direct` at the P=1 shipping shape. But 54 of
the frozen test's 68 ids are P=2 or P=4, and those reach `_sc_topk_direct`
through three further call sites (`:607`, `:612`, `:637`) -- one of which feeds
it `cand_scores`, which is itself produced by a `bitcast_convert_type` at
`:516`. A lowering failure there is a hard block on the correctness job, and it
costs seconds to rule out here instead of a pod.

Every shape in `SHAPES` and `TWO_STAGE_SHAPES` from the frozen test is compiled
for both arms, and the count of `bitcast-convert` fusions left in each ENTRY is
reported -- that is the op v12 exists to delete.

Run: uv_venv/bin/python3 aot_two_stage.py
"""

import importlib.util
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

HERE = pathlib.Path(__file__).parent
INCUMBENT = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" /
             "deepseek_v4" / "sparsecore_topk.py")
VARIANT = HERE / "sparsecore_topk.py"

DEV = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                   platform="tpu").devices[0]
SH = SingleDeviceSharding(DEV)

# The frozen test's SHAPES and TWO_STAGE_SHAPES, verbatim by (b, n, k).
N_SHORT, N_128K = 10240, 133120
CASES = [
    (64, N_SHORT, 2048, "p1-b64-n10240"),
    (1024, N_SHORT, 2048, "p1-b1024-n10240"),
    (16, N_SHORT, 2048, "p2-b16-n10240"),
    (16, N_128K, 2048, "p4-b16-n133120"),
    (64, N_128K, 2048, "p4-b64-n133120"),
    (16, N_128K, 512, "p4-b16-n133120-k512"),
    (1024, N_128K, 2048, "p4-b1024-n133120"),
]


def load(tag, path):
    spec = importlib.util.spec_from_file_location(f"ts_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    mods = {t: load(t, p) for t, p in (("incumbent", INCUMBENT),
                                       ("variant", VARIANT))}
    print(f"{'case':24s} {'P':>2s} | "
          f"{'incumbent':>26s} | {'variant':>26s}")
    print(f"{'':24s} {'':>2s} | {'lowers  bcvt  entry':>26s} | "
          f"{'lowers  bcvt  entry':>26s}")
    for b, n, k, cid in CASES:
        row, p = f"{cid:24s}", None
        for tag in ("incumbent", "variant"):
            mod = mods[tag]
            p = mod._pick_partition(b, n, k)
            scores = jax.ShapeDtypeStruct((b, n), jnp.float32, sharding=SH)
            lens = jax.ShapeDtypeStruct((b, ), jnp.int32, sharding=SH)
            try:
                c = jax.jit(lambda s, l: mod.sparsecore_topk(s, k, l),
                            out_shardings=SH).lower(scores, lens).compile()
                e = re.search(r"^ENTRY .*?^}", c.as_text(), re.S | re.M).group(0)
                bcvt = len(re.findall(r"bitcast-convert", e))
                res = f"{'ok':>6s} {bcvt:5d} {len(e.splitlines()) - 2:6d}"
            except Exception as exc:  # noqa: BLE001
                res = f"{'FAIL':>6s}  {type(exc).__name__[:16]}"
            if tag == "incumbent":
                row = f"{row} {p:2d} | {res:>26s}"
            else:
                row = f"{row} | {res:>26s}"
        print(row)
