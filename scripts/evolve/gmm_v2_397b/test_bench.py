"""CPU checks of bench.py against golden: run with a python holding jax 0.10.2 + libtpu 0.0.47.

    python scripts/evolve/gmm_v2_397b/test_bench.py

1. The Pallas patch bench.py applies is golden env_override's, verbatim.
2. SERVED_LIBTPU_INIT_ARGS is what golden env_override assembles from the served
   recipe's LIBTPU_INIT_ARGS on tpu7x.
3. Compiled ahead of time for one tpu7x core, each config with a
   `trace_signature` launches exactly that gmm_v2 op name.
4. The fp32 reference matches a direct grouped matmul on a small case.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECIPE_LIBTPU_INIT_ARGS = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
    " --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false"
    " --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
    " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=8388608"
    " --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144")


def function_source(path: Path, name: str) -> str:
    src = path.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{path} has no {name}")


def check_patch_is_golden() -> None:
    name = "_patch_jax_pallas_fori_lowering"
    golden = function_source(REPO / "src/vllm_torchtpu/env_override.py", name)
    assert function_source(HERE / "bench.py", name) == golden, "the patch drifted from golden"


def check_libtpu_args(bench) -> None:
    code = ("import os, sys; sys.path.insert(0, 'src'); "
            "import importlib.util as u, types; "
            "p = types.ModuleType('vllm_torchtpu'); p.__path__ = ['src/vllm_torchtpu']; "
            "sys.modules['vllm_torchtpu'] = p; "
            "s = u.spec_from_file_location('vllm_torchtpu.env_override', 'src/vllm_torchtpu/env_override.py'); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); print(os.environ['LIBTPU_INIT_ARGS'])")
    env = dict(os.environ, LIBTPU_INIT_ARGS=RECIPE_LIBTPU_INIT_ARGS, TPU_ACCELERATOR_TYPE="tpu7x")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True,
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    assert out == bench.SERVED_LIBTPU_INIT_ARGS, f"served flags differ:\n{out!r}\n{bench.SERVED_LIBTPU_INIT_ARGS!r}"


def check_signatures(bench) -> dict:
    handle = bench.setup(accept_device_kind="cpu", root=REPO / "src/vllm_torchtpu/kernels/megablox")
    import jax
    from jax._src import mesh as mesh_lib
    from jax.experimental import topologies

    dev = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices[0]
    sharding = jax.sharding.SingleDeviceSharding(dev)
    jnp = handle.jnp
    seen = {}
    for name, cfg in bench.CONFIGS.items():
        if not cfg.get("trace_signature") and not name.startswith("prefill"):
            continue
        m, k, n = cfg["size_m"], cfg["size_k"], cfg["size_n"]
        specs = [((m, k), jnp.bfloat16), ((bench.NUM_GROUPS, k, n), jnp.float8_e4m3fn),
                 ((bench.NUM_GROUPS,), jnp.int32), ((bench.NUM_GROUPS, 1, 1, n), jnp.float32),
                 ((1,), jnp.int32)]
        args = [jax.ShapeDtypeStruct(s, d, sharding=sharding) for s, d in specs]
        with mesh_lib.use_abstract_mesh(mesh_lib.AbstractMesh((), (), abstract_device=mesh_lib.AbstractDevice(
                device_kind="TPU7x", num_cores=1, platform="tpu"))):
            text = jax.jit(bench.kernel_call(handle, cfg)).trace(*args).lower(
                lowering_platforms=("tpu",)).compile().as_text()
        seen[name] = bench.signature(text)
        if cfg.get("trace_signature"):
            assert seen[name] == cfg["trace_signature"], f"{name}: {seen[name]} != trace"
    return seen


def check_reference(bench) -> None:
    import numpy as np

    cfg = {"size_m": 8, "size_k": 4, "size_n": 6, "fuse_act": None, "active": 2, "rows": 5, "seed": 3}
    gs = bench.group_sizes_for(cfg)
    act = np.flatnonzero(gs)
    rng = np.random.default_rng(0)
    host = {"lhs": rng.standard_normal((8, 4)).astype(np.float32), "gs": gs, "active": act,
            "w": rng.standard_normal((2, 4, 6)).astype(np.float32),
            "scale": rng.uniform(0.5, 1.5, (2, 6)).astype(np.float32)}
    ref = bench.fp32_reference(cfg, host)
    a = int(gs[act[0]])
    want = np.concatenate([host["lhs"][:a] @ (host["w"][0] * host["scale"][0]),
                           host["lhs"][a:5] @ (host["w"][1] * host["scale"][1])])
    assert np.allclose(ref, want), "reference rows are not the grouped matmul"


def main() -> int:
    spec = importlib.util.spec_from_file_location("gmm_v2_397b_bench", HERE / "bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    check_patch_is_golden()
    check_libtpu_args(bench)
    seen = check_signatures(bench)
    check_reference(bench)
    print(json.dumps({"ok": True, "signatures": seen}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
