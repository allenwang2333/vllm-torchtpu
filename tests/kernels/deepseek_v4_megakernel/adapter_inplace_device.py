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
"""The adapter's cache paths on 8 TPU ranks: in place vs donated + aliased.

    python -m tests.kernels.deepseek_v4_megakernel.adapter_inplace_device

R = 8 DP ranks (torch.distributed.run, tpu_dist). Each rank binds the fake
four-layer cache tree (test_vllm_adapter.make_fake_layers, seeded random
contents) twice, once per path, with `page_copy_kernel.PageCopyKernel` as the
kernel: a communicating pallas_call that DMA-writes a page of every cache
array. Each adapter runs an idle warmup and two DP steps (ranks 0 and 7
busy). Gates, per rank: both paths' caches equal a numpy replay of the page
copies (idle ranks: unchanged), hidden / aux bitwise equal across the paths.
Rank 0 dumps its optimized HLO; the parent counts copies of cache-shaped
arrays in each path's step program.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

WORLD = 8
Q, B, T = 7, 2, 16
BUSY = (0, WORLD - 1)
HIDDEN = 128
# MK_HOSTBENCH_WEIGHTS=<n>: n weight operands (the served step has ~440 plus
# the caches) and a host-time comparison of the in-place step's dispatch
N_WEIGHTS = int(os.environ.get("MK_HOSTBENCH_WEIGHTS", "3"))
STEP_IDS = (1001, 2006)  # ids[0] per step: page ids[0] % n <- (ids[0] + 1) % n
MODES = ("aliased", "inplace")


def _seeded(t, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    hi = 256 if t.dtype == torch.uint8 else 2**30
    return torch.randint(0, hi, tuple(t.shape), generator=g, dtype=t.dtype)


def _worker(result_dir: Path) -> None:
    rank = int(os.environ["RANK"])
    if rank == 0:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_dump_to={result_dir / 'xla_r0'} --xla_dump_hlo_as_text"
        )
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    out: dict = {"rank": rank}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from torch_tpu._internal.pallas import get_pallas_mesh

        from tests.kernels.deepseek_v4_megakernel.page_copy_kernel import (
            PageCopyKernel,
        )
        from tests.kernels.deepseek_v4_megakernel.test_vllm_adapter import (
            fake_metadata,
            make_fake_layers,
        )
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA

        spec = KI.ModelSpec(
            hidden_size=HIDDEN, num_layers=4, compress_ratios=(128, 4, 128, 4),
            dtype="float32", q=Q, mode="dp", axis_name="ep", num_ranks=WORLD,
            aux_layers=(2, 3),
        )  # fmt: skip
        mesh = get_pallas_mesh(axis_names=("ep",))
        busy = rank in BUSY
        rows = KI.request_block_rows(Q)
        entries = [0, 0] + [rows if r in BUSY else 0 for r in range(WORLD)]
        saved = {}
        for mode in MODES:
            layers, groups, distinct = make_fake_layers("tpu")
            for i, t in enumerate(distinct):
                t.copy_(_seeded(t, 1000 * rank + i).to("tpu"))
            bound = VA.bind_layer_caches(layers, groups)
            widths = [3 if g % 2 else 5 for g in bound.group_ids]
            mk = VA.DSv4Megakernel(
                spec=spec,
                buckets=VA.parse_row_buckets(str(T), Q, B),
                kernel=PageCopyKernel(mode == "inplace", N_WEIGHTS, HIDDEN),
                mesh=mesh,
                rank=rank,
                expert_block=rank,
                weights=[
                    torch.full((8,), float(i)).to("tpu") for i in range(N_WEIGHTS)
                ],
                decoder_layers=layers,
                device="tpu",
                op_namespace=f"dsv4_mk_inplace_dev_{mode}",
            )
            mk.bind(bound, widths, swa_page=128)
            saved[f"{mode}_before"] = [t.cpu().numpy() for t in mk.caches.arrays]
            mk.warmup(NS(num_tokens_paddings=[T], num_reqs_max_model_len=4))
            saved[f"{mode}_warm"] = [t.cpu().numpy() for t in mk.caches.arrays]
            if busy:
                md = fake_metadata(mk, num_reqs=1, padded_reqs=4, bt_cols=widths,
                                   device="tpu")  # fmt: skip
                VA._current_attn_metadata = lambda md=md: md
            for s, id0 in enumerate(STEP_IDS):
                local = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, rows) if busy else None
                d = mk.decide_dp(local, max_tokens=Q, max_chunks=1, any_prefill=False,
                                 entries=entries)  # fmt: skip
                assert d.run, d
                ids = torch.arange(T, dtype=torch.int32) + id0
                if not busy:
                    ids = torch.zeros(T, dtype=torch.int32)
                pos = torch.arange(T, dtype=torch.int32) + 900
                hidden, aux = mk.forward(None, ids.to("tpu"), pos.to("tpu"), None, None)
                saved[f"{mode}_hidden{s}"] = hidden.cpu().numpy()
                saved[f"{mode}_aux{s}"] = np.stack([a.cpu().numpy() for a in aux])
            saved[f"{mode}_after"] = [t.cpu().numpy() for t in mk.caches.arrays]
            if mode == "inplace" and "MK_HOSTBENCH_WEIGHTS" in os.environ:
                out["host_us"] = _host_bench(mk, torch)
        flat = {}
        for k, v in saved.items():
            if isinstance(v, list):
                flat.update({f"{k}.{i}": a for i, a in enumerate(v)})
            else:
                flat[k] = v
        np.savez(result_dir / f"rank_{rank}.npz", **flat)
        out["n_arrays"] = len(saved["inplace_after"])
        out["shapes"] = [[list(a.shape), str(a.dtype)] for a in saved["inplace_after"]]
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        out["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(out))
    if "error" in out:
        raise SystemExit(1)


def _host_bench(mk, torch, iters=30) -> dict:
    """Host time to enqueue one in-place step: through DirectStepCall, and
    through the JaxCallable front end it bypasses (idle operands, no sync)."""
    import time

    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA

    (bucket,) = mk.buckets
    step = mk._steps[bucket]
    direct = next(
        c.cell_contents
        for c in step.__closure__
        if isinstance(c.cell_contents, VA.DirectStepCall)
    )
    z = mk._idle_operands[bucket]
    head = (mk.meta, z[0], z[1], z[2], *z[3:])
    w, c = mk.weights, mk.caches.arrays

    def timed(fn):
        fn()
        torch.tpu.synchronize() if hasattr(torch, "tpu") else None
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters * 1e6
        mk.caches.arrays[0].cpu()
        return dt

    return {
        "operands": len(head) + len(w) + len(c),
        "direct": timed(lambda: direct(head, w, c)),
        "jaxcallable": timed(lambda: direct.flat(*head, *w, *c)),
    }


def _replay(before: list[np.ndarray], busy: bool) -> list[np.ndarray]:
    out = [a.copy() for a in before]
    if busy:
        for id0 in STEP_IDS:
            for a in out:
                n = a.shape[0]
                a[id0 % n] = a[(id0 + 1) % n]
    return out


_HLO_DT = {"int32": "s32", "uint8": "u8"}


def _hlo_copies(dump: Path, shapes) -> dict:
    """Per path: optimized step modules found, and their copy instructions
    whose result is a cache array's per-rank or global shape."""
    pats = set()
    for shape, dt in shapes:
        for lead in (shape[0], shape[0] * WORLD):
            dims = ",".join(map(str, [lead, *shape[1:]]))
            pats.add(f"{_HLO_DT[dt]}[{dims}]")
    res = {m: {"modules": 0, "cache_copies": 0, "sample": []} for m in MODES}
    files = sorted(dump.glob("*after_optimizations.txt")) if dump.exists() else []
    for f in files:
        txt = f.read_text(errors="replace")
        for m in MODES:
            if f"page_copy_{m}" not in txt:
                continue
            res[m]["modules"] += 1
            for ln in txt.splitlines():
                hit = re.search(r"= (\S+) copy(-start)?\(", ln)
                if hit and any(hit.group(1).startswith(p) for p in pats):
                    res[m]["cache_copies"] += 1
                    if len(res[m]["sample"]) < 3:
                        res[m]["sample"].append(ln.strip()[:140])
    res["dump_files"] = len(files)
    return res


def _launch(result_dir: Path) -> int:
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )

    saved = dict(os.environ)
    prepare_tpu_environment(world_size=WORLD)
    env = os.environ.copy()
    os.environ.clear()
    os.environ.update(saved)
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    me = "tests.kernels.deepseek_v4_megakernel.adapter_inplace_device"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
           f"--nproc-per-node={WORLD}", "-m", me, "--worker",
           "--result-dir", str(result_dir)]  # fmt: skip
    return subprocess.run(cmd, env=env, check=False).returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--result-dir", type=Path, default=Path("/tmp/dsv4_adapter_inplace"))
    args = p.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        _worker(args.result_dir)
        return 0
    bad = _launch(args.result_dir) != 0
    shapes = None
    for r in range(WORLD):
        f = args.result_dir / f"rank_{r}.json"
        res = json.loads(f.read_text()) if f.exists() else {"error": "missing"}
        if "error" in res:
            bad = True
            print("RESULT", json.dumps({"rank": r, **res}), flush=True)
            continue
        shapes = res["shapes"]
        z = np.load(args.result_dir / f"rank_{r}.npz")
        n = res["n_arrays"]
        busy = r in BUSY
        row = {"rank": r, "busy": busy}
        if "host_us" in res:
            row["host_us"] = res["host_us"]
        for m in MODES:
            before = [z[f"{m}_before.{i}"] for i in range(n)]
            warm = [z[f"{m}_warm.{i}"] for i in range(n)]
            after = [z[f"{m}_after.{i}"] for i in range(n)]
            want = _replay(before, busy)
            row[f"{m}_warmup_wrote_nothing"] = all(
                np.array_equal(a, b) for a, b in zip(warm, before)
            )
            row[f"{m}_caches_match_replay"] = all(
                np.array_equal(a, b) for a, b in zip(after, want)
            )
            row[f"{m}_bytes_changed"] = int(
                sum(np.count_nonzero(a != b) for a, b in zip(after, before))
            )
        row["caches_inplace_eq_aliased"] = all(
            np.array_equal(z[f"inplace_after.{i}"], z[f"aliased_after.{i}"])
            for i in range(n)
        )
        row["head_inplace_eq_aliased"] = all(
            np.array_equal(z[f"inplace_{k}{s}"], z[f"aliased_{k}{s}"])
            for k in ("hidden", "aux")
            for s in range(len(STEP_IDS))
        )
        row["busy_flag"] = float(z["inplace_hidden0"][0, 1])
        ok = all(
            v
            for k, v in row.items()
            if k.endswith(("replay", "nothing", "_eq_aliased"))
        )
        ok &= (row["inplace_bytes_changed"] > 0) == busy
        bad |= not ok
        print("RESULT", json.dumps(row), flush=True)
    if shapes is not None:
        hlo = _hlo_copies(args.result_dir / "xla_r0", shapes)
        print("RESULT hlo", json.dumps(hlo), flush=True)
        bad |= hlo["inplace"]["cache_copies"] != 0
    print(f"RESULT ok={not bad}", flush=True)
    return int(bad)


if __name__ == "__main__":
    sys.exit(main())
