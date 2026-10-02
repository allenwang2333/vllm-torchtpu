# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_DRAFT_HOIST_ATTN_META through the real torch adapter, on 8 ranks.

`tests/layers/core/test_sparse_mla_dcp_draft_prep.py` pins the JAX functions;
it cannot see the torch registration, which is where the first served boot
died (`pallas._verify_signature` rejected a `tuple[jax.Array, ...]` return).
This builds both adapter ops the drafter builds --
`_build_sparse_mla_dcp_draft_prep_op` (behind eagle3's `_compiled_draft_prep`,
as `_build_draft_prep` calls it) and `_build_sparse_mla_dcp_op` with the flag
on -- over real per-rank torch tensors, and checks three loop steps of the
hoisted attention bit-identical to the flag-0 per-call path (nope, rope, out,
lse), plus a positive control: a stale write plan must change the nope cache.

Run on one host (4 chips x 2 cores = 8 ranks).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
NUM_HEADS = 16
TOPK = 2048
NUM_TOKENS = 16
BUF_ROWS = 64
NUM_STEPS = 3
NUM_REQS = 4
KERNEL_NUM_REQS = 8
PAGES_PER_SEQ = 32
NUM_PAGES = 64
INTERLEAVE = 16
LAUNCH_TIMEOUT_SECONDS = 1200

_SERVED_ENV = {
    "TPU_DRAFT_HOIST_ATTN_META": "1",
    "TPU_DCP_MLA_TOPK_CAP": "512",
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": "256",
    "TPU_DCP_MLA_CAPPED_PREP": "1",
    "TPU_DCP_MLA_SPLIT_Q_LSE": "1",
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": "4",
    "TPU_DCP_MLA_PACKED_GATHER": "0",
}


def _prepare_worker_env() -> dict[str, str]:
    try:
        from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
            prepare_tpu_environment,
        )
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    env.update(_SERVED_ENV)
    return env


# --------------------------------------------------------------------- worker
def _problem(rank, world):
    """This rank's torch inputs; the global problem is identical on all ranks."""
    import numpy as np
    import torch

    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )

    rng = np.random.default_rng(20261002)
    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, NUM_PAGES, PAGE_SIZE, LKV_DIM
    )
    rope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, NUM_PAGES, PAGE_SIZE, ROPE_DIM
    )

    def cache(spec):
        # Every byte a finite fp8 e4m3fn of magnitude < 2.
        shape = (world * spec.shape[0],) + tuple(spec.shape[1:])
        raw = rng.integers(0, 256, size=shape + (4,), dtype=np.uint8) & 0xBF
        local = raw.view(np.int32).reshape(shape)[
            rank * spec.shape[0] : (rank + 1) * spec.shape[0]
        ]
        return torch.from_numpy(np.ascontiguousarray(local)).view(spec.torch_dtype)

    nope, rope = cache(nope_spec), cache(rope_spec)
    qsl = np.minimum(np.arange(KERNEL_NUM_REQS + 1), NUM_REQS).astype(np.int32)
    rd = np.array([NUM_REQS] * 3, np.int32)
    max_len = (PAGES_PER_SEQ * world - 2) * PAGE_SIZE
    seq_lens = rng.integers(PAGE_SIZE, max_len, size=KERNEL_NUM_REQS).astype(np.int32)
    seq_lens[0] = PAGE_SIZE * world  # crosses a virtual page between steps
    seq_lens[1] = 63
    block_tables = rng.integers(
        0, 4 * NUM_PAGES, size=KERNEL_NUM_REQS * PAGES_PER_SEQ
    ).astype(np.int32)
    pool = np.array([0, 1, 130, 511, 512, 513, 700, 1025, 2048])
    counts = rng.choice(pool, size=(world, NUM_TOKENS))
    idx = rng.integers(0, PAGES_PER_SEQ * PAGE_SIZE, size=(world, NUM_TOKENS, TOPK))
    live = np.arange(TOPK)[None, None, :] < counts[..., None]
    topk = np.where(live, idx, -1).astype(np.int32)
    # The layer's shared top-k buffer: this rank's rows first, stale rows after.
    buf = np.full((BUF_ROWS, TOPK), 7, np.int32)
    buf[:NUM_TOKENS] = topk[rank]

    def q(dim):
        return torch.from_numpy(
            rng.normal(size=(NUM_TOKENS, NUM_HEADS * world, dim)).astype(np.float32)
        ).to(torch.bfloat16)

    def latent(dim):
        v = rng.normal(size=(NUM_TOKENS, dim)).astype(np.float32)
        return torch.from_numpy(v).to(torch.float8_e4m3fn)

    steps = [
        (q(LKV_DIM), q(ROPE_DIM), latent(LKV_DIM), latent(ROPE_DIM))
        for _ in range(NUM_STEPS)
    ]
    t = torch.from_numpy
    return dict(
        specs=(nope_spec, rope_spec),
        nope=nope,
        rope=rope,
        buf=t(buf),
        seq_lens=t(seq_lens),
        block_tables=t(block_tables),
        qsl=t(qsl),
        rd=t(rd),
        steps=steps,
    )


_STAGE = ["init"]


def _stage(name):
    """Record the step about to run, so a worker failure names it."""
    _STAGE[0] = name
    print(f"STAGE rank{os.environ.get('RANK')} {name}", flush=True)


def _bits(x):
    import torch

    x = x.cpu()
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.view(torch.int16)
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    return x


def _check(rank, world, group):
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.distributed.mesh_utils import get_or_create_cp_mesh
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
    from vllm_torchtpu.layers.core.attention_interface import (
        DCP_DRAFT_PREP_SHARED,
        DcpDraftPrep,
    )
    from vllm_torchtpu.spec_decode.eagle3 import _compiled_draft_prep

    mesh = get_or_create_cp_mesh(axis_name=DCP_AXIS_NAME, group=group)
    pr = _problem(rank, world)
    impl = SimpleNamespace(
        _dcp_mesh=mesh, _get_kv_scales=lambda layer: (None, 0.75, None)
    )
    layer = SimpleNamespace(
        mla_kv_spec=pr["specs"],
        layer_name="model.layers.78.mtp_block.self_attn.mla_attn",
        scale=0.1,
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
    )
    cls = PallasMLAttentionBackendImpl
    _stage("build_ops")
    prep_op = cls._build_sparse_mla_dcp_draft_prep_op(
        impl, layer, world, INTERLEAVE, NUM_STEPS
    )
    attn = cls._build_sparse_mla_dcp_op(impl, layer, world, INTERLEAVE)

    dev = {k: v.to("tpu") for k, v in pr.items() if isinstance(v, torch.Tensor)}
    steps = [tuple(x.to("tpu") for x in s) for s in pr["steps"]]
    topk = dev["buf"][:NUM_TOKENS]
    common = (dev["block_tables"], dev["qsl"], dev["rd"])
    report = {}

    _stage("prep_compiled")
    # The prep, as eagle3's `_build_draft_prep` calls it (compiled region,
    # `buf[:num_tokens]` inside), and eagerly through the op.
    prep = tuple(
        _compiled_draft_prep(prep_op)(
            dev["buf"], dev["seq_lens"], *common, NUM_TOKENS
        )
    )
    torch.equal(_bits(prep[0]), _bits(prep[0]))
    _stage("prep_eager")
    eager_prep = tuple(prep_op(topk, dev["seq_lens"], *common))
    want = DCP_DRAFT_PREP_SHARED + 3 * NUM_STEPS
    _stage("prep_compare")
    report["prep_len"] = len(prep)
    report["prep_shapes"] = [list(p.shape) for p in prep]
    report["prep_compiled_eq_eager"] = len(prep) == want and all(
        torch.equal(_bits(a), _bits(b)) for a, b in zip(prep, eager_prep)
    )

    def fresh():
        return (dev["nope"].clone(), dev["rope"].clone())

    def run(caches, step, sl, draft=None):
        out, lse = attn(caches, *steps[step], topk, sl, *common, draft_index_prep=draft)
        return out, lse

    ref_c, hoist_c = fresh(), fresh()
    sl = dev["seq_lens"]
    mismatches = []
    for step in range(NUM_STEPS):
        _stage(f"attn_flag0_step{step}")
        r = run(ref_c, step, sl)
        _bits(r[0])
        _stage(f"attn_hoisted_step{step}")
        g = run(hoist_c, step, sl, tuple(DcpDraftPrep.for_step(prep, step)))
        for name, a, b in (
            ("nope", ref_c[0], hoist_c[0]),
            ("rope", ref_c[1], hoist_c[1]),
            ("out", r[0], g[0]),
            ("lse", r[1], g[1]),
        ):
            if not torch.equal(_bits(a), _bits(b)):
                mismatches.append(f"step{step}:{name}")
        sl = sl + 1
    report["mismatches"] = mismatches
    report["out_finite"] = bool(torch.isfinite(r[0].cpu().float()).all())

    _stage("control")
    # Positive control: step 0's write plan at step 1 moves the write.
    right_c, wrong_c = fresh(), fresh()
    run(right_c, 0, dev["seq_lens"])
    run(wrong_c, 0, dev["seq_lens"])
    sl1 = dev["seq_lens"] + 1
    run(right_c, 1, sl1, tuple(DcpDraftPrep.for_step(prep, 1)))
    run(wrong_c, 1, sl1, tuple(DcpDraftPrep.for_step(prep, 0)))
    report["control_stale_plan_differs_local"] = not torch.equal(
        _bits(right_c[0]), _bits(wrong_c[0])
    )
    return report


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        world_group = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="dcp_draft_prep_test",
        )
        result["check"] = _check(rank, world, group)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)[:2000]
        result["stage"] = _STAGE[0]
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        (result_dir / f"rank_{rank}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


# --------------------------------------------------------------------- pytest
def _results(result_dir: Path) -> list[dict]:
    out = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


def assert_results(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    want = 4 + 3 * NUM_STEPS
    for r in results:
        c = r["check"]
        assert c["prep_len"] == want, (r["rank"], c)
        assert c["prep_compiled_eq_eager"], (r["rank"], c)
        assert not c["mismatches"], (r["rank"], c["mismatches"])
        assert c["out_finite"], (r["rank"], c)
    # A stale plan writes to another position; some rank's shard must move.
    assert any(r["check"]["control_stale_plan_differs_local"] for r in results), (
        "positive control: a stale write plan left every nope shard unchanged"
    )


def test_hoisted_draft_prep_through_torch_adapter(tmp_path):
    result_dir = tmp_path / "dcp_draft_prep"
    result_dir.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={WORLD_SIZE}",
            str(Path(__file__).resolve()),
            "--result-dir",
            str(result_dir),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        process.wait(timeout=60)
    results = _results(result_dir)
    print(json.dumps(results[0], indent=1))
    if any("error" in r for r in results):
        print((output or "")[-12000:])
    assert_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    _run_worker(args.result_dir)
