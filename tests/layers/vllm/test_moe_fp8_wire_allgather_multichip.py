# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_MOE_FP8_WIRE_ALLGATHER on real devices.

`test_moe_fp8_wire_allgather.py` pins the arithmetic on CPU. Four things it
cannot answer, each of which is a hard stop for the design:

  1. ``.to(torch.float8_e4m3fn)`` lowers at all -- eagerly and inside a
     ``torch.compile``d graph. No fp8 *activation* tensor exists anywhere in
     this program today; the weights are fp8, the activations are not.
  2. A vLLM group ``all_gather`` accepts an ``f8e4m3fn`` tensor, and the
     emitted HLO really carries ``f8e4m3fn`` rather than silently upcasting to
     bf16 -- an upcast compiles, runs, passes every correctness check and
     delivers none of the win. If fp8 is refused, the ``view(torch.int8)``
     bitcast around the collective is the fallback, and this file tests both.
  3. The gathered collective stays on the SparseCore. The offload threshold is
     the per-tensor-core VMEM capacity (64 MiB on tpu7x); the gathered tensor
     is 96 MiB in bf16 and 48 MiB in fp8, so it crosses the line, which is why
     the knob halves the threshold to 32 MiB.
  4. The round trip stays bit-exact under the TPU's own bf16 arithmetic, not
     just under CPU's.

Shapes are the production ones. The *gathered* tensor is what the SparseCore
threshold is compared against, so it is held at bf16[8192, 6144] = 96 MiB --
the shape the DP=16 prefill step gathers -- and the per-rank tile is sized to
reach it at this slice's world size (8192 / 8 = 1024 rows) rather than copied
from DP=16's 512.

Run on one host with 4 chips x 2 cores.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
# The production MoE dispatch shapes. HIDDEN is the GMM's contracting
# dimension (the emitted kernel is `...-k_6144-...-tk_6144-...`); GATHERED_ROWS
# is the token count the DP group gathers to.
HIDDEN = 6144
GATHERED_ROWS = 8192
ROWS = GATHERED_ROWS // WORLD_SIZE
LAUNCH_TIMEOUT_SECONDS = 1500

# Mirrors of the constants under test; deliberately restated rather than
# imported, so a change to either side shows up as a failure here.
BLOCK = 512
E4M3_MAX = 448.0
VMEM_BYTES_PER_TENSOR_CORE = 67108864
SC_AG_FLAG = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------
def _kernel_quantize(x):
    """What `kernels/megablox/gmm_v2.py` does to its lhs, in `x`'s own dtype.

    `block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1)`;
    `block_scale = block_abs_max / dtype_max`;
    `block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)`.
    """
    import torch
    rows, hidden = x.shape
    blocks = hidden // BLOCK
    blocked = x.reshape(rows, blocks, BLOCK)
    scale = blocked.abs().amax(dim=-1, keepdim=True) / E4M3_MAX
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale),
                            scale.reciprocal())
    codes = (blocked * scale_inv).to(torch.float8_e4m3fn)
    return codes, scale


def _run_compiled(fn, arg, results, name):
    """Run `fn` under the TPU backend, recording whether fullgraph held.

    Dynamo traces at call time, not at `torch.compile` time, so the fullgraph
    refusal has to be caught around the call. `_all_gather_activations` reads
    `envs.TPU_MOE_FP8_WIRE_ALLGATHER`, which is a module `__getattr__` over
    `os.getenv`; whether Dynamo folds that or breaks on it is something only a
    run can say, and a graph break inside the compiled decode graph is a
    finding, so record it rather than hide it.
    """
    import torch
    from torch_tpu._internal import compile as tpu_compile
    try:
        out = torch.compile(fn,
                            fullgraph=True,
                            dynamic=False,
                            backend=tpu_compile.TpuBackend())(arg)
        results[f"{name}_fullgraph"] = True
        return out
    except Exception as exc:  # noqa: BLE001
        results[f"{name}_fullgraph"] = False
        results[f"{name}_fullgraph_error"] = repr(exc)[:900]
        out = torch.compile(fn,
                            fullgraph=False,
                            dynamic=False,
                            backend=tpu_compile.TpuBackend())(arg)
        return out


def _measure(rank, arm, dump_dir):
    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (get_world_group,
                                                 init_model_parallel_group)

    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.layers.vllm import pipelined_fused_moe as mod

    # The world group is built with `use_device_communicator=False`, so its
    # `all_gather` raises "No device communicator found". The DP group the MoE
    # path actually calls is an `init_model_parallel_group` coordinator with
    # the TPU device communicator attached; build one spanning all ranks, which
    # is what DP=world does in production.
    world = get_world_group()
    group = init_model_parallel_group(
        [list(range(WORLD_SIZE))],
        world.local_rank,
        dist.get_backend(world.device_group),
        group_name="fp8_wire_dp",
    )
    r = {
        "rank": rank,
        "arm": arm,
        "world_size": group.world_size,
        "group_name": group.unique_name,
        "device_communicator": type(group.device_communicator).__name__,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "knob": bool(envs.TPU_MOE_FP8_WIRE_ALLGATHER),
        "quant_block": mod._FP8_WIRE_QUANT_BLOCK,
        "fp8_default_constant":
        __import__("vllm_torchtpu.env_override", fromlist=["x"]
                   ).FP8_WIRE_ALLGATHER_DEFAULT,
    }
    assert mod._FP8_WIRE_QUANT_BLOCK == BLOCK

    # (P3a) the flag the knob is supposed to move, read back from the env the
    # worker will hand libtpu. Derived from the *effective* knob, so the
    # `shipped` arm -- which sets no env var and lets the image's own module
    # constant decide -- checks the artifact the e2e A/B will build.
    expected = (VMEM_BYTES_PER_TENSOR_CORE //
                2 if r["knob"] else VMEM_BYTES_PER_TENSOR_CORE)
    tokens = r["libtpu_init_args"].split()
    got = [t for t in tokens if t.startswith(SC_AG_FLAG + "=")]
    r["sc_ag_flag"] = got[0] if got else None
    r["sc_ag_flag_expected"] = f"{SC_AG_FLAG}={expected}"
    r["sc_ag_flag_ok"] = bool(got) and got[0] == r["sc_ag_flag_expected"]

    torch.manual_seed(20260921 + rank)
    x = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16)
    x_tpu = x.to("tpu")

    # ------------------------------------------------------------------
    # P1: does .to(torch.float8_e4m3fn) lower, eager and compiled?
    # fp8 e4m3 -> bf16 is exact (3 mantissa bits into 7), so equality of the
    # dequantized tensors is equality of the codes.
    # ------------------------------------------------------------------
    def _cast_roundtrip(t):
        return t.to(torch.float8_e4m3fn).to(torch.bfloat16)

    cpu_ref = _cast_roundtrip(x)
    try:
        eager = _cast_roundtrip(x_tpu).cpu()
        r["p1_eager_ok"] = True
        r["p1_eager_mismatches"] = int((eager != cpu_ref).sum().item())
    except Exception as exc:  # noqa: BLE001
        r["p1_eager_ok"] = False
        r["p1_eager_error"] = repr(exc)[:600]

    try:
        comp = _run_compiled(_cast_roundtrip, x_tpu, r, "p1_compiled").cpu()
        r["p1_compiled_ok"] = True
        r["p1_compiled_mismatches"] = int((comp != cpu_ref).sum().item())
    except Exception as exc:  # noqa: BLE001
        r["p1_compiled_ok"] = False
        r["p1_compiled_error"] = repr(exc)[:600]

    # Does an fp8 tensor survive a device->host copy and a bitcast at all?
    try:
        q = x_tpu.to(torch.float8_e4m3fn)
        r["p1_fp8_cpu_view_int8_ok"] = bool(
            q.cpu().view(torch.int8).dtype == torch.int8)
    except Exception as exc:  # noqa: BLE001
        r["p1_fp8_cpu_view_int8_ok"] = False
        r["p1_fp8_cpu_view_int8_error"] = repr(exc)[:300]
    try:
        r["p1_fp8_device_view_int8_ok"] = bool(
            x_tpu.to(torch.float8_e4m3fn).view(
                torch.int8).cpu().dtype == torch.int8)
    except Exception as exc:  # noqa: BLE001
        r["p1_fp8_device_view_int8_ok"] = False
        r["p1_fp8_device_view_int8_error"] = repr(exc)[:300]

    # ------------------------------------------------------------------
    # P2: the collective. Four arms, each its own compiled program so the
    # dumped HLO modules are separable by their shapes.
    # ------------------------------------------------------------------
    def arm_bf16(t):
        return group.all_gather(t, dim=0)

    def arm_wire(t):
        # The code under test, exactly as the MoE path calls it.
        return mod._all_gather_activations(group, t)

    def arm_int8_bitcast(t):
        """The fallback, if f8e4m3fn is refused by the collective."""
        rows, hidden = t.shape
        blocks = hidden // BLOCK
        blocked = t.reshape(rows, blocks, BLOCK)
        scale = blocked.abs().amax(dim=-1, keepdim=True) / E4M3_MAX
        scale_inv = torch.where(scale == 0, torch.zeros_like(scale),
                                scale.reciprocal())
        q = (blocked * scale_inv).to(torch.float8_e4m3fn).reshape(rows, hidden)
        ag = group.all_gather(q.view(torch.int8), dim=0).view(
            torch.float8_e4m3fn)
        ag_scale = group.all_gather(scale.reshape(rows, blocks), dim=0)
        ag_rows = ag.shape[0]
        return (ag.reshape(ag_rows, blocks, BLOCK).to(t.dtype) *
                ag_scale.reshape(ag_rows, blocks, 1)).reshape(ag_rows, hidden)

    def arm_promoted(t):
        """Negative control: the same move, but the scale promoted to f32.

        The commit claims the round trip is bit-exact *because* the scale stays
        in the input dtype; promoted, ~3.42% of codes move by one step. If this
        arm also comes back with 0 differing codes, the P4 comparison cannot
        fail and proves nothing.
        """
        rows, hidden = t.shape
        blocks = hidden // BLOCK
        blocked = t.reshape(rows, blocks, BLOCK).to(torch.float32)
        scale = blocked.abs().amax(dim=-1, keepdim=True) / E4M3_MAX
        scale_inv = torch.where(scale == 0, torch.zeros_like(scale),
                                scale.reciprocal())
        q = (blocked * scale_inv).to(torch.float8_e4m3fn).reshape(rows, hidden)
        ag = group.all_gather(q, dim=0)
        ag_scale = group.all_gather(scale.reshape(rows, blocks), dim=0)
        ag_rows = ag.shape[0]
        return (ag.reshape(ag_rows, blocks, BLOCK).to(torch.float32) *
                ag_scale.reshape(ag_rows, blocks, 1)).to(t.dtype).reshape(
                    ag_rows, hidden)

    def arm_f32_dequant(t):
        """Candidate fix: quantize in `t`'s dtype, dequantize in float32.

        The commit's f32 hazard is on the *quantize* side -- an f32 scale moves
        codes. This arm leaves the scale exactly as the change computes it and
        promotes only the reconstruction, which is what CPU torch already does
        internally and TPU apparently does not.
        """
        rows, hidden = t.shape
        blocks = hidden // BLOCK
        blocked = t.reshape(rows, blocks, BLOCK)
        scale = blocked.abs().amax(dim=-1, keepdim=True) / E4M3_MAX
        scale_inv = torch.where(scale == 0, torch.zeros_like(scale),
                                scale.reciprocal())
        q = (blocked * scale_inv).to(torch.float8_e4m3fn).reshape(rows, hidden)
        ag = group.all_gather(q, dim=0)
        ag_scale = group.all_gather(scale.reshape(rows, blocks), dim=0)
        ag_rows = ag.shape[0]
        return (ag.reshape(ag_rows, blocks, BLOCK).to(torch.float32) *
                ag_scale.reshape(ag_rows, blocks, 1).to(torch.float32)).to(
                    t.dtype).reshape(ag_rows, hidden)

    outputs = {}
    for name, fn in (("bf16", arm_bf16), ("wire", arm_wire),
                     ("int8_bitcast", arm_int8_bitcast),
                     ("f32_dequant", arm_f32_dequant), ("promoted",
                                                        arm_promoted)):
        try:
            out = _run_compiled(fn, x_tpu, r, f"p2_{name}")
            outputs[name] = out
            r[f"p2_{name}_ok"] = True
            r[f"p2_{name}_shape"] = list(out.shape)
            r[f"p2_{name}_dtype"] = str(out.dtype)
        except Exception as exc:  # noqa: BLE001
            r[f"p2_{name}_ok"] = False
            r[f"p2_{name}_error"] = repr(exc)[:900]

    # Eager too: a compiled-only result cannot separate "the collective
    # refuses fp8" from "XLA sank the dequantize back across the collective".
    try:
        eager_wire = arm_wire(x_tpu)
        r["p2_wire_eager_ok"] = True
        r["p2_wire_eager_shape"] = list(eager_wire.shape)
        r["p2_wire_eager_dtype"] = str(eager_wire.dtype)
        if "bf16" in outputs:
            r["p2_wire_eager_vs_bf16_differing"] = int(
                (eager_wire != outputs["bf16"]).sum().item())
        del eager_wire
    except Exception as exc:  # noqa: BLE001
        r["p2_wire_eager_ok"] = False
        r["p2_wire_eager_error"] = repr(exc)[:900]

    # `_all_gather_activations` must be inert with the knob off: the trap is an
    # "inert" path nobody runs. Run it, and require it to equal the plain
    # gather element for element.
    r["p2_wire_applies"] = bool(mod.fp8_wire_allgather_applies(x_tpu))
    if "bf16" in outputs and "wire" in outputs:
        diff = (outputs["wire"] != outputs["bf16"]).sum().item()
        r["p2_wire_vs_bf16_differing_elements"] = int(diff)
        r["p2_wire_vs_bf16_total"] = int(outputs["bf16"].numel())

    # ------------------------------------------------------------------
    # P4: the thing downstream actually sees. The GMM kernel re-quantizes
    # whatever it is handed; compare the codes and the scales it would derive.
    # ------------------------------------------------------------------
    if "bf16" in outputs:
        ref_codes, ref_scale = _kernel_quantize(outputs["bf16"])
        ref_codes_f = ref_codes.to(torch.float32).cpu()
        ref_scale_c = ref_scale.cpu()
        r["p4_total_codes"] = int(ref_codes_f.numel())
        for name in ("wire", "int8_bitcast", "f32_dequant", "promoted"):
            if name not in outputs:
                continue
            codes, scale = _kernel_quantize(outputs[name])
            codes_f = codes.to(torch.float32).cpu()
            neq = (codes_f != ref_codes_f)
            n = int(neq.sum().item())
            r[f"p4_{name}_differing_codes"] = n
            r[f"p4_{name}_differing_fraction"] = n / ref_codes_f.numel()
            if n:
                # fp8 e4m3 codes are ordered by value within an exponent, so a
                # "one step" move shows up as the smallest nonzero ratio.
                d = (codes_f - ref_codes_f).abs()
                r[f"p4_{name}_max_abs_code_delta"] = float(d.max().item())
            sc = scale.cpu()
            neqs = (sc != ref_scale_c)
            ns = int(neqs.sum().item())
            r[f"p4_{name}_scale_differing"] = ns
            r[f"p4_{name}_scale_total"] = int(ref_scale_c.numel())
            if ns:
                # bf16 ULP distance: for same-sign finite values the bit
                # pattern read as an integer is monotone in the value.
                bits_a = sc.view(torch.int16).to(torch.int64)
                bits_b = ref_scale_c.view(torch.int16).to(torch.int64)
                ulp = (bits_a - bits_b).abs()
                r[f"p4_{name}_scale_max_ulp"] = int(ulp.max().item())
                r[f"p4_{name}_scale_ulp_histogram"] = {
                    str(int(v)): int((ulp == int(v)).sum().item())
                    for v in torch.unique(ulp)[:8]
                }
                rel = ((sc.float() - ref_scale_c.float()).abs() /
                       ref_scale_c.float().clamp_min(1e-30))
                r[f"p4_{name}_scale_max_rel_error"] = float(rel.max().item())

    # ------------------------------------------------------------------
    # P2/P3: read the HLO back. Never infer a wire format from the fact that
    # it ran.
    # ------------------------------------------------------------------
    r["hlo"] = _scan_hlo(dump_dir)
    return r


def _scan_hlo(dump_dir):
    """Every all-gather in the dumped modules, plus any sparsecore mention."""
    out = {"dump_dir": str(dump_dir), "files": 0, "all_gathers": [],
           "sparsecore_lines": [], "async_threads": []}
    if not dump_dir.exists():
        return out
    files = sorted(dump_dir.glob("*.txt"))
    out["files"] = len(files)
    out["file_names"] = [f.name for f in files][:80]
    for f in files:
        if not f.name.endswith(".after_optimizations.txt"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            s = line.strip()
            if "all-gather" in s:
                out["all_gathers"].append({"file": f.name, "hlo": s[:600]})
            if "sparsecore" in s.lower():
                out["sparsecore_lines"].append({
                    "file": f.name,
                    "hlo": s[:400]
                })
            if "execution_thread=" in s:
                out["async_threads"].append({"file": f.name, "hlo": s[:400]})
    for key in ("all_gathers", "sparsecore_lines", "async_threads"):
        out[key] = out[key][:200]
    # The one number that settles P2: which element types actually appear on
    # the wire. A dequantize sunk back across the collective, or a silent
    # upcast, both show up here as bf16 where f8e4m3fn was asked for.
    import re
    types: dict[str, int] = {}
    for hit in out["all_gathers"]:
        for m in re.finditer(r"\b([a-z0-9]+)\[([0-9,]*)\]", hit["hlo"]):
            types[f"{m.group(1)}[{m.group(2)}]"] = types.get(
                f"{m.group(1)}[{m.group(2)}]", 0) + 1
    out["all_gather_shape_histogram"] = dict(
        sorted(types.items(), key=lambda kv: -kv[1])[:40])
    return out


def _run_worker(result_dir: Path, arm: str) -> None:
    rank = int(os.environ["RANK"])
    # XLA_FLAGS is read when the PJRT client starts, so it has to be in place
    # before torch_tpu loads libtpu. Per-rank directories keep the ranks
    # symmetric -- an XLA flag that differs across ranks is its own bug.
    dump_dir = result_dir / "hlo" / f"rank{rank}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (
        f"{existing} --xla_dump_to={dump_dir}").strip()

    result_path = result_dir / f"rank_{rank}.json"
    result: dict
    initialized = False
    try:
        # Import order matters: env_override assembles LIBTPU_INIT_ARGS (and
        # the halved SparseCore threshold) and must run before libtpu loads.
        # `vllm_torchtpu/__init__.py` imports it on its first line.
        import vllm_torchtpu  # noqa: F401

        import torch
        import torch_tpu  # noqa: F401
        from torch import distributed as dist

        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1, ), device="tpu").cpu()

        from vllm.distributed.parallel_state import \
            init_distributed_environment
        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result = _measure(rank, arm, dump_dir)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result = {"rank": rank, "arm": arm, "error": repr(exc)[:2000]}
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                from torch import distributed as dist
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


# --------------------------------------------------------------------------
# parent side
# --------------------------------------------------------------------------
def _worker_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
        "--arm",
        "ARM",
    ]


def _prepare_worker_env(arm: str) -> dict[str, str]:
    try:
        from torch_tpu._internal.distributed.launchers.singlehost_wrapper import \
            prepare_tpu_environment
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")

    prepared_keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    original_values = {key: os.environ.get(key) for key in prepared_keys}
    try:
        for key in prepared_keys:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                   "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    # Each worker must assemble LIBTPU_INIT_ARGS itself: env_override only adds
    # the SparseCore flag when the caller has not already supplied it, so an
    # inherited value from this pytest process would pin the 64 MiB threshold
    # and the ON arm would silently test the OFF flag.
    env.pop("LIBTPU_INIT_ARGS", None)
    if arm == "shipped":
        # Let the image's own FP8_WIRE_ALLGATHER_DEFAULT decide. This is the
        # arm that tests the artifact the e2e A/B builds, rather than an env
        # var the e2e A/B does not use.
        env.pop("TPU_MOE_FP8_WIRE_ALLGATHER", None)
    else:
        env["TPU_MOE_FP8_WIRE_ALLGATHER"] = "1" if arm == "knobon" else "0"
    # env_override resolves the threshold from the accelerator type; the pod is
    # tpu7x by construction (the recipe hardcodes topology 2x2x1 on tpu7x), so
    # pin it rather than depend on a metadata round trip inside the worker.
    env.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x")
    return env


def _run_worker_group(result_dir: Path, arm: str):
    cmd = [a if a != "ARM" else arm for a in _worker_command(result_dir)]
    process = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=_prepare_worker_env(arm),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        process.wait(timeout=60)
    return subprocess.CompletedProcess(cmd, process.returncode, output, "")


def _collect(result_dir: Path):
    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else
            {
                "rank": rank,
                "error": "missing worker result"
            })
    return results


def _publish(result_dir: Path, arm: str, results, completed) -> None:
    """Copy everything a reader needs off the pod."""
    out = os.environ.get("CDK_OUTPUT_DIR")
    if not out:
        return
    dest = Path(out) / "fp8_wire" / arm
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(results, indent=2),
                                       encoding="utf-8")
    (dest / "worker_output.log").write_text(completed.stdout or "",
                                            encoding="utf-8")
    src = result_dir / "hlo" / "rank0"
    if src.exists():
        for f in src.glob("*after_optimizations.txt"):
            try:
                shutil.copy2(f, dest / f.name)
            except OSError:
                pass


@pytest.mark.parametrize("arm", ["knoboff", "knobon", "shipped"])
def test_fp8_wire_allgather_on_device(tmp_path, arm):
    result_dir = tmp_path / f"fp8_wire_{arm}"
    result_dir.mkdir()
    completed = _run_worker_group(result_dir, arm)
    results = _collect(result_dir)
    _publish(result_dir, arm, results, completed)

    errors = [r for r in results if "error" in r]
    print(f"\n===== FP8 WIRE ALLGATHER arm={arm} =====")
    print(json.dumps(results[0] if results else {}, indent=2)[:60000])
    print(f"===== end arm={arm} (returncode={completed.returncode}) =====")
    assert completed.returncode == 0 and not errors, (
        f"worker failure: returncode={completed.returncode}, "
        f"errors={errors}\noutput tail:\n{completed.stdout[-16000:]}")

    r = results[0]
    # (P3a) the flag the knob moves.
    assert r["sc_ag_flag_ok"], (r["sc_ag_flag"], r["sc_ag_flag_expected"])
    # The knob's own state. `shipped` sets no env var, so this reads back the
    # module constant the image ships.
    if arm == "knoboff":
        assert r["knob"] is False, r["knob"]
    elif arm == "knobon":
        assert r["knob"] is True, r["knob"]
    else:
        assert r["knob"] is bool(r["fp8_default_constant"]), r
    # (P1) the cast lowers and matches CPU.
    assert r["p1_eager_ok"] and r["p1_eager_mismatches"] == 0, r
    assert r["p1_compiled_ok"] and r["p1_compiled_mismatches"] == 0, r
    # (P2) the reference gather ran and produced the production shape.
    assert r.get("p2_bf16_ok"), r.get("p2_bf16_error")
    assert r["p2_bf16_shape"] == [GATHERED_ROWS, HIDDEN], r["p2_bf16_shape"]
    assert r.get("p2_wire_ok"), r.get("p2_wire_error")
    # The knobs-off path must be inert -- executed, not argued.
    if not r["knob"]:
        assert r["p2_wire_applies"] is False
        assert r["p2_wire_vs_bf16_differing_elements"] == 0, r
    else:
        assert r["p2_wire_applies"] is True
        # (P4) bit equality is the bar.
        assert r["p4_wire_differing_codes"] == 0, r
        assert r["p4_wire_scale_differing"] == 0, (
            "the GMM would re-derive a different per-block scale from the "
            "round-tripped tensor; candidate fix (dequantize in f32) gives "
            f"{r.get('p4_f32_dequant_scale_differing')} differing scales and "
            f"{r.get('p4_f32_dequant_differing_codes')} differing codes")
        # The negative control: if the promoted arm also matches, the
        # comparison above could not have failed and says nothing.
        assert r.get("p2_promoted_ok"), r.get("p2_promoted_error")
        assert r["p4_promoted_differing_codes"] > 0, (
            "the fp32-promoted control did not differ, so the bit-equality "
            "check above does not discriminate")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--arm", default="off")
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        raise ValueError("expected --worker")
    _run_worker(args.result_dir, args.arm)


if __name__ == "__main__":
    _main()
