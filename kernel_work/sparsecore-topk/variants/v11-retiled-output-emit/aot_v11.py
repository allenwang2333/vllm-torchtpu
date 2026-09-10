"""v11 static-compile evidence: standalone ENTRY, the IN-MODEL staging copy, controls.

STATIC COMPILE-TIME ANALYSIS ONLY. No TPU is attached and no timing is produced.
`jax.experimental.topologies.get_topology_desc("tpu7x:2x2x1")` builds the topology from
libtpu on the host, so `.compile()` runs the real XLA-TPU + Mosaic SparseCore backend.

Arms:
  base : the incumbent   src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py
  v11  : THIS variant    variants/v11-retiled-output-emit/sparsecore_topk.py
Both are loaded as modules; neither file is written to.

Sections:
  1. standalone `_sc_topk_direct` at production dims  -> ENTRY count, ROOT, temp bytes
  2. IN-MODEL: the real enclosing entry `_streamidx_topk_jax` (probe 14's rebuild)
     -> does the `S(1)` staging copy + `copy-start/copy-done` survive?
  3. negative control: an 8 MiB VMEM scratch must fail in the spmem allocator
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info as _tpu_info

# probe 12/17's device-pinning prologue, verbatim.
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
REPO = HERE.parents[3]
BASE_SRC = REPO / "src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py"
V11_SRC = HERE / "sparsecore_topk.py"
OUT = HERE / "hlo"
OUT.mkdir(exist_ok=True)

B, N, K = 1024, 10240, 2048  # production prefill: f32[1024,10240] -> i32[1024,2048]
MIB = 1024 * 1024

# probe 14's PROD block, for the enclosing entry's avals.
PROD = dict(num_tokens=1024, num_index_heads=80, head_dim=128, max_num_seqs=64,
            lkv_dim=256, page_size=1024, kv_packing=4, max_model_len=9216,
            num_kv_pages_per_block=(2, 2, 2), num_queries_per_block=(1, 128, 128),
            k=2048, compression_ratio=1)


def dev():
    return topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                        platform="tpu").devices[0]


def load(name, path):
    spec = importlib.util.spec_from_file_location(f"sctopk_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def entry(hlo):
    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    return m.group(0) if m else hlo


def mem(compiled):
    a = compiled.memory_analysis()
    return {k: getattr(a, k) for k in
            ("argument_size_in_bytes", "output_size_in_bytes",
             "temp_size_in_bytes", "alias_size_in_bytes",
             "generated_code_size_in_bytes") if hasattr(a, k)}


_BC = re.compile(r"backend_config=(\{.*\})\s*$")


def used_scoped(line):
    m = _BC.search(line.rstrip().rstrip(","))
    if not m:
        return None
    try:
        cfg = json.loads(m.group(1))
    except Exception:
        return None
    return [u.get("size") for u in (cfg.get("used_scoped_memory_configs") or [])]


# ---------------------------------------------------------------- 1. standalone
def standalone(name, mod):
    sh = SingleDeviceSharding(dev())
    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=sh)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=sh)
    fn = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l), out_shardings=sh)
    t0 = time.perf_counter()
    c = fn.lower(scores, lens).compile()
    txt = c.as_text()
    (OUT / f"standalone-{name}.txt").write_text(txt)
    e = entry(txt)
    (OUT / f"standalone-entry-{name}.txt").write_text(e)
    print(f"--- standalone {name}   ({time.perf_counter() - t0:.1f}s)")
    print(f"  ENTRY instructions : {len(e.splitlines()) - 2}")
    for ln in e.splitlines():
        if ln.strip().startswith("ROOT"):
            print(f"  ROOT               : {ln.strip()[:150]}")
    for ln in e.splitlines():
        if " reshape(" in ln or " copy-start(" in ln or " copy(" in ln:
            print(f"  [move] {ln.strip()[:130]}   used_scoped={used_scoped(ln)}")
    print(f"  memory_analysis    : {json.dumps(mem(c))}")
    return e


# ------------------------------------------------------------------ 2. in-model
def _derived():
    pps = -(-PROD["max_model_len"] // PROD["page_size"])
    return dict(pages_per_seq=pps, num_pages=PROD["max_num_seqs"] * pps)


def _outer_avals(sh):
    p, d = PROD, _derived()
    nt, hi, hd, ms, npg = (p["num_tokens"], p["num_index_heads"], p["head_dim"],
                           p["max_num_seqs"], d["num_pages"])
    s = lambda shape, dt: jax.ShapeDtypeStruct(shape, dt, sharding=sh)
    return (
        s((nt, hi, hd), jnp.uint8),                                   # q_bytes
        s((nt, hi), jnp.bfloat16),                                    # weights
        s((npg, p["page_size"] // p["kv_packing"], p["kv_packing"],
           p["lkv_dim"]), jnp.uint8),                                 # cache_kv
        s((nt, hd + 1), jnp.uint8),                                   # k_packed
        s((ms, ), jnp.int32),                                         # seq_lens
        s((npg, ), jnp.int32),                                        # page_indices
        s((ms + 1, ), jnp.int32),                                     # cu_q_lens
        s((3, ), jnp.int32),                                          # distribution
    )


SCT_MODNAME = "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk"
SIT_MODNAME = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"


def fresh_sit(mod):
    """Reload `streamindex_topk` against `mod` as its `sparsecore_topk`.

    `streamindex_topk.py:25` does `from ...sparsecore_topk import
    sparsecore_topk`, so rebinding the attribute is not enough: `jax.jit`
    caches the inner pjit *jaxpr* on (function object, avals), and the second
    arm would otherwise silently reuse the first arm's trace. Seeding
    `sys.modules` and reloading gives each arm its own jitted callables, and
    `jax.clear_caches()` drops anything already keyed on the old ones.
    """
    import importlib
    sys.modules[SCT_MODNAME] = mod
    sit = importlib.import_module(SIT_MODNAME)
    sit = importlib.reload(sit)
    jax.clear_caches()
    assert sit.sparsecore_topk is mod.sparsecore_topk, "module swap did not take"
    return sit


def in_model(name, mod, early_exit):
    """probe 14's `_streamidx_topk_jax` rebuild, with `mod` supplying the top-k."""
    sit = fresh_sit(mod)
    topk = PROD["k"]

    def _streamidx_topk_jax(q_bytes, weights, cache_kv, k_packed, seq_lens,
                            page_indices, cu_q_lens, distribution):
        _, bs_per_pk, kv_packing, lkv_dim = cache_kv.shape
        block_size = bs_per_pk * kv_packing
        k_packed = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
        tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
        page = jnp.where(
            tok < cu_q_lens[-1],
            page_indices.reshape(seq_lens.shape[0], -1)[seq_id, pos // block_size],
            jnp.int32(2**30))
        slot = pos % block_size
        cache_kv = cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)
        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        topk_indices = sit.streamindex_topk(
            q, weights, cache_kv, seq_lens, page_indices, cu_q_lens, distribution,
            k=topk, compression_ratio=PROD["compression_ratio"],
            num_kv_pages_per_block=PROD["num_kv_pages_per_block"],
            num_queries_per_block=PROD["num_queries_per_block"],
            enable_early_exit=early_exit)
        return cache_kv, topk_indices

    sh = SingleDeviceSharding(dev())
    fn = jax.jit(_streamidx_topk_jax, static_argnums=(), donate_argnums=(2, ),
                 keep_unused=True)
    t0 = time.perf_counter()
    c = fn.lower(*_outer_avals(sh)).compile()
    txt = c.as_text()
    (OUT / f"inmodel-{name}.txt").write_text(txt)
    import hashlib
    print(f"--- in-model {name}   ({time.perf_counter() - t0:.1f}s, "
          f"{len(txt.splitlines())} lines, "
          f"hlo md5 {hashlib.md5(txt.encode()).hexdigest()})")
    print(f"  sparsecore_topk module file: {mod.__file__}")
    kname = f"sc_topk_direct_b{B}_n{N}_k{K}"
    print(f"  shape gate `{kname}` present: {kname in txt}")
    print(f"  memory_analysis: {json.dumps(mem(c))}")

    # Walk the chain by name from the kernel's call-done.
    done = [m.group(1) for m in
            re.finditer(r"%(\S*" + re.escape(kname) + r"\S*call-done)\s*=", txt)]
    print(f"  call-done producers: {done}")
    lines = txt.splitlines()

    def users(nm):
        return [l for l in lines if nm in l and f"%{nm} =" not in l
                and not l.strip().startswith("//")]

    chain, seen, frontier = [], set(), list(done)
    for _ in range(6):
        nxt = []
        for nm in frontier:
            for l in users(nm):
                d = re.match(r"\s*(?:ROOT )?%(\S+) =", l)
                if not d or d.group(1) in seen:
                    continue
                seen.add(d.group(1))
                chain.append(l.strip())
                nxt.append(d.group(1))
        frontier = nxt
    print("  --- chain from call-done ---")
    for l in chain[:24]:
        print(f"    {l[:190]}   used_scoped={used_scoped(l)}")

    # The three tells probe 14 named, counted over the whole module.
    for pat, label in ((r"S\(1\)\}", "S(1)-annotated shapes"),
                       (r"copy-start", "copy-start"),
                       (r"= s32\[1024,2048\]", "s32[1024,2048] results"),
                       (r"= s32\[2097152\]", "s32[2097152] results")):
        hits = [l.strip() for l in lines if re.search(pat, l)]
        print(f"  {label:28s}: {len(hits)}")
        for h in hits[:8]:
            print(f"      {h[:170]}")
    return txt


# ------------------------------------------------------- 3. negative control
def negative_control():
    """probe 17 `aot_controls.py:37` verbatim in shape: 8 MiB of VMEM scratch on
    a 512 KiB SC tile must be refused by the SparseCore *allocator*, not the
    MLIR verifier. If this compiles, every green above is worthless."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu_sc as plsc
    sc = _TPU7X.sparse_core

    def kern(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, K)], o_hbm.at[0], sems.at[0])
        cp.start()
        cp.wait()

    def build():
        return pl.kernel(
            kern,
            out_type=jax.ShapeDtypeStruct((B, K), jnp.int32),
            mesh=plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                        num_subcores=sc.num_subcores,
                                        core_axis_name="core",
                                        subcore_axis_name="subcore"),
            scratch_types=(pltpu.VMEM((2 * 1024 * 1024, ), jnp.int32),
                           pltpu.SemaphoreType.DMA((1, ))),
            name="aot_negctl_oversized_vmem")()

    try:
        jax.jit(build, out_shardings=SingleDeviceSharding(dev())).lower().compile()
        print("AOT-NEGCTL-8MiB-vmem: *** COMPILED -- CONTROL DID NOT FIRE ***")
    except Exception as exc:
        s = f"{type(exc).__name__}: {exc}"
        ok = "E3000" in s and "spmem" in s
        print(f"AOT-NEGCTL-8MiB-vmem: FAILED as required={ok}")
        print("   " + s.replace("\n", " ")[:400])


if __name__ == "__main__":
    print(f"jax {jax.__version__}")
    import hashlib
    for nm, p in (("base", BASE_SRC), ("v11", V11_SRC)):
        print(f"  subject {nm:5s} {p}  md5 "
              f"{hashlib.md5(p.read_bytes()).hexdigest()}")
    negative_control()

    ents = {}
    mods = {}
    for nm, p in (("base", BASE_SRC), ("v11", V11_SRC)):
        mods[nm] = load(nm, p)
        try:
            ents[nm] = standalone(nm, mods[nm])
        except Exception:
            traceback.print_exc()

    import difflib
    if len(ents) == 2:
        d = "\n".join(difflib.unified_diff(ents["base"].splitlines(),
                                           ents["v11"].splitlines(),
                                           fromfile="ENTRY-base",
                                           tofile="ENTRY-v11", lineterm="", n=1))
        (OUT / "diff-standalone-base-vs-v11.txt").write_text(d)
        print(f"\n===== standalone ENTRY diff base -> v11 "
              f"({len(d.splitlines())} lines) =====\n{d}")

    if os.environ.get("SKIP_INMODEL"):
        sys.exit(0)
    sys.path.insert(0, str(REPO / "src"))
    try:
        import vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op as mao
        early = mao.STREAMIDX_ENABLE_EARLY_EXIT
        print(f"STREAMIDX_ENABLE_EARLY_EXIT = {early} (from mla_attention_op)")
    except Exception as exc:
        src = (REPO / "src/vllm_torchtpu/layers/vllm/custom_ops/"
               "mla_attention_op.py").read_text()
        m = re.search(r"^STREAMIDX_ENABLE_EARLY_EXIT\s*=\s*(\w+)", src, re.M)
        early = m.group(1) == "True"
        print(f"STREAMIDX_ENABLE_EARLY_EXIT = {early} (read from source; "
              f"import failed: {type(exc).__name__})")
    # Order matters only as a check: run v11 FIRST so a stale-cache bug would
    # show as base looking like v11, not the other way round.
    for nm in ("v11", "base"):
        try:
            in_model(nm, mods[nm], early)
        except Exception:
            traceback.print_exc()
