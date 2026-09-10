"""AOT-compile the REAL sparsecore_topk at production dims, three ways, and
diff the post-optimization ENTRY.

No TPU is attached. `jax.experimental.topologies.get_topology_desc` builds a
tpu7x:2x2x1 topology from libtpu on the host, so `.compile()` runs the full XLA
TPU pipeline and emits real optimized HLO with `used_scoped_memory_configs`.
STATIC COMPILE-TIME ANALYSIS ONLY -- no timing is produced.

  base : src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py, unmodified
  u2   : f32 kernel input + f32 resident scratch + in-register bitcast
  v11  : out_type = (b, k) i32, out DMA writes out_hbm.at[row]

The variants are string patches applied to a COPY of the source; the shipped
file is never touched.
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
SRC = (HERE.parents[3] / "src" / "vllm_torchtpu" / "kernels" / "deepseek_v4" /
       "sparsecore_topk.py")
B, N, K = 1024, 10240, 2048  # production: f32[1024,10240] -> i32[1024,2048]

# --- U2: f32 in, f32 resident scratch, bitcast in-register -------------------
U2_PATCHES = [
    # sparsecore_topk.py:438 -- the op we are trying to delete
    ("    words = jax.lax.bitcast_convert_type(scores, jnp.int32)\n",
     "    words = scores  # U2: no bitcast_convert_type\n"),
    # :455 -- the resident slice scratch must follow the input dtype
    ("        pltpu.VMEM((slice_len, ), jnp.int32),\n",
     "        pltpu.VMEM((slice_len, ), jnp.float32),  # U2\n"),
    # :171-173 -- cvt reads f32, keys in-register, writes f32 back
    ("""            def cvt(i):
                bits = resident_vmem[pl.ds(i * LANES, LANES)]
                resident_vmem[pl.ds(i * LANES, LANES)] = monotone_key(bits)
""", """            def cvt(i):
                bits = jax.lax.bitcast_convert_type(
                    resident_vmem[pl.ds(i * LANES, LANES)], jnp.int32)
                resident_vmem[pl.ds(i * LANES, LANES)] = (
                    jax.lax.bitcast_convert_type(monotone_key(bits),
                                                 jnp.float32))
"""),
    # :178 -- every later read of the slice retypes in-register
    ("            key = resident_vmem[pl.ds(i * LANES, LANES)]\n",
     "            key = jax.lax.bitcast_convert_type(\n"
     "                resident_vmem[pl.ds(i * LANES, LANES)], jnp.int32)\n"),
]

# --- v11: 2-D tiled output emitted by the kernel ------------------------------
V11_PATCHES = [
    # :451
    ("    slot_type = jax.ShapeDtypeStruct((b * k, ), jnp.int32)\n",
     "    slot_type = jax.ShapeDtypeStruct((b, k), jnp.int32)  # v11\n"),
    # :315-319 -- out DMA writes a whole tiled row
    ("""                rowbuf_vmem.at[pl.ds(0, k)],
                out_hbm.at[pl.ds(row_c * k, k)],
""", """                rowbuf_vmem.at[pl.ds(0, k)],
                out_hbm.at[row_c],
"""),
    ("""                    keybuf_vmem.at[pl.ds(0, k)],
                    keys_hbm.at[pl.ds(row_c * k, k)],
""", """                    keybuf_vmem.at[pl.ds(0, k)],
                    keys_hbm.at[row_c],
"""),
    # :491 -- the reshape that XLA turns into ROOT %reshape.1
    ("    return out.reshape(b, k)\n", "    return out  # v11\n"),
    ("        return slots.reshape(b, k), jax.lax.bitcast_convert_type(\n"
     "            keys.reshape(b, k), jnp.float32)\n",
     "        return slots, jax.lax.bitcast_convert_type(keys, jnp.float32)\n"),
]


def load(name, patches):
    text = SRC.read_text()
    for old, new in patches:
        if old not in text:
            raise SystemExit(f"[{name}] patch anchor not found:\n{old!r}")
        text = text.replace(old, new)
    path = HERE / f"_gen_{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"sctopk_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def entry(hlo):
    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    return m.group(0) if m else hlo


def compile_variant(name, mod):
    dev = topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                       platform="tpu").devices[0]
    sh = SingleDeviceSharding(dev)
    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=sh)
    fn = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l), out_shardings=sh)
    hlo = fn.lower(scores, lens).compile().as_text()
    e = entry(hlo)
    (HERE / "hlo" / f"entry-{name}.txt").write_text(e)
    (HERE / "hlo" / f"full-{name}.txt").write_text(hlo)
    n_bc = len(re.findall(r"= \S+ bitcast-convert\(", e))
    roots = re.findall(r"^\s*ROOT %\S+ = .*$", e, re.M)
    print(f"--- {name}")
    print(f"  ENTRY instructions      : {len(e.splitlines()) - 2}")
    print(f"  bitcast-convert in ENTRY: {n_bc}")
    for r in roots:
        print(f"  ROOT: {r.strip()[:170]}")
    for pat in ("bitcast-convert", "reshape("):
        for ln in e.splitlines():
            if pat in ln and "ROOT" not in ln:
                print(f"  [{pat}] {ln.strip()[:200]}")
    return e


if __name__ == "__main__":
    print("jax", jax.__version__, "| src", SRC)
    ents = {}
    for name, patches in [("base", []), ("u2", U2_PATCHES),
                          ("v11", V11_PATCHES)]:
        try:
            ents[name] = compile_variant(name, load(name, patches))
        except Exception as exc:
            print(f"--- {name}: FAILED {type(exc).__name__}: {exc}"[:3000])
            import traceback
            traceback.print_exc()

    import difflib
    for a, b in (("base", "u2"), ("base", "v11")):
        if a in ents and b in ents:
            d = "\n".join(
                difflib.unified_diff(ents[a].splitlines(),
                                     ents[b].splitlines(),
                                     fromfile=f"ENTRY-{a}",
                                     tofile=f"ENTRY-{b}",
                                     lineterm="",
                                     n=1))
            (HERE / "hlo" / f"diff-{a}-vs-{b}.txt").write_text(d)
            print(f"\n===== ENTRY diff {a} -> {b} "
                  f"({len(d.splitlines())} lines) =====")
