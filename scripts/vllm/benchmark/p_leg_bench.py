#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measures the Qwen3.5-397B AgentX prefill leg's step time on one 2x2x1 host.

The engine is the offline vLLM engine with the P leg's serving configuration:
TP8 + SP (fp8 all-gather) + EP8 with fused EP v3, fp8 KV cache, block size
640 with Mamba "align" prefix caching, MTP with one draft token, and the
offload-v7 PREFILL_* knobs (``P_LEG_ENV`` below is the prefill container env of
job j-03cc9fd7, the D1 trace, without the KV transfer, offload tier and
frontend knobs). Weights are synthetic: the unittest pod cannot see the
checkpoint PVC, and step time does not depend on weight values as long as the
activations stay finite and the router spreads tokens over the experts.

Each case builds REQS prefixes of CTX random tokens, then runs ROUNDS rounds in
which every request is its prefix plus NEW fresh tokens, so a measured step
reads CTX tokens of cached K/V and starts from a restored GDN state. Reported
per round: wall time, computed rows, rows/s for the host, and the step interval
from a scheduler hook (one record per engine step). With --profile the last
round of each case runs under the xprof profiler; its trace lands in
OUT/xprof/case<i>/ for p_leg_xprof.py.

Run in a vllm-torchtpu-unittest pod through
tests/benchmarks/p_leg/test_p_leg_bench.py, or directly:

    python scripts/vllm/benchmark/p_leg_bench.py --case 92160:10240:16 \
        --rounds 3 --profile --env TPU_SOME_KNOB=1

Results: OUT/result.json, plus a ``P_LEG_RESULT {...}`` line on stdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Prefill container env of j-03cc9fd7 (agentx-397b-pd-baseline, scripts/agentx-cell.sh
# 1 1 16), minus the KV transfer, offload tier, compile-cache path and frontend
# knobs, which do not reach the forward step.
P_LEG_ENV = {
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_OFFLINE": "1",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PLATFORMS": "tpu,cpu",
    "MODEL_IMPL_TYPE": "vllm",
    "MOE_FUSED_EP_KERNEL_IMPL": "v3",
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS": "1024",
    "MOE_LOCAL_EXPERT_AFFINITY_EPSILON": "0.01",
    "NEW_MODEL_DESIGN": "0",
    "OMP_NUM_THREADS": "8",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "2048",
    "PJRT_DEVICE": "TPU",
    "RAGGED_GATED_DELTA_RULE_IMPL": "chunked_kernel_v3_pd",
    "RAGGED_GATHER_REDUCE_VERSION": "v3",
    "SKIP_JAX_PRECOMPILE": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
    "TORCHINDUCTOR_MAX_WORKERS": "4",
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
    "TPUMISC_HMA_CHECKSUM_TRACE": "0",
    "TPU_BACKEND_TYPE": "jax",
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
    "TPU_ENABLE_GDN_DYNAMIC_TILING": "0",
    "TPU_EXPERT_PLACEMENT": str(
        REPO_ROOT / "expert_placement/expert_placement_tp8_ep8_agentx_corr_c32.json"
    ),
    "TPU_FUSE_POST_ATTN_NORM": "1",
    "TPU_FUSE_PRE_ATTN_NORM": "1",
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": "1",
    "TPU_GDN_CONV_STATE_TILE_PAD": "1",
    "TPU_GDN_FUSED_NORM_QUANT": "1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_HYBRID_RETENTION_MAX_CONVS": "256",
    "TPU_HYBRID_RETENTION_MAX_PINNED_BLOCKS": "1800",
    "TPU_HYBRID_STATE_RETENTION": "1",
    "TPU_KERNEL_PAGE_TOKENS": "128",
    "TPU_KV_CACHE_HEADROOM_MIB": "6144",
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER": "1",
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": "33",
    "TPU_MOE_LOWTOK": "0",
    "TPU_MOE_LOWTOK_MAX_TOKENS": "256",
    "TPU_MOE_SEQUENCE_PARALLEL": "1",
    "TPU_MOE_SKIP_PADDED_TOKENS": "1",
    "TPU_PJRT_EVENT_TRACE_ENABLE": "1",
    "TPU_PREFILL_SKIP_VERIFY_WARMUP": "1",
    "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    "TPU_QWEN35_AUTO_DEFAULTS": "1",
    "TPU_ROPE_CACHE_TRUNCATE": "1",
    "TPU_RPA_FOLD_KV_HEAD_DIM": "0",
    "TPU_SMALL_ALL_REDUCE": "0",
    "TPU_SP_FP8_GATHER": "1",
    "TPU_SP_MIN_ROWS": "512",
    "TPU_TOPOLOGY": "2x2x1",
    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
    "TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG": "1",
    "USE_BATCHED_RPA_KERNEL": "1",
    "USE_BATCHED_RPA_LONGCTX": "1",
    "USE_FUSED_MOE_GMM": "1",
    "USE_MOE_COUNTING_SORT": "1",
    "USE_MOE_FUSED_EP_KERNEL": "1",
    "USE_MOE_FUSED_TP_AR": "0",
    "USE_MOE_SPARSE_CORE": "1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": "2700",
    "VLLM_KV_CACHE_LAYOUT": "HND",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_MOE_CHUNK_SIZE": "256",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": "1",
    "VLLM_TPU_QWEN35_MEGAKERNEL": "0",
    "VLLM_TPU_QWEN35_MTP_MK": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_XLA_CHECK_RECOMPILATION": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # The recipe's string; vllm_torchtpu.env_override rewrites it the same way
    # it does in the server.
    "LIBTPU_INIT_ARGS": (
        " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
        " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
        " --xla_tpu_ars_combiner_threshold_in_bytes=0"
        " --xla_tpu_enable_async_collective_merger=false"
        " --xla_enable_async_all_gather=true"
        " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
        " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
    ),
}

# Loaded at interpreter start in the engine-core and worker processes (they
# are spawned, and inherit PYTHONPATH). Patches two modules as they load:
# DummyModelLoader, so each tensor is filled once with values that survive the
# fp8 cast, and Scheduler.update_from_output, which logs one record per step.
SITECUSTOMIZE = r"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

_POOL_N = 16_777_259  # prime: tiled rows never repeat at power-of-two widths
_pools = {}


def _pool(torch, dtype, lo, hi):
    key = (dtype, lo, hi)
    if key not in _pools:
        gen = torch.Generator().manual_seed(1234)
        vals = torch.rand(_POOL_N, generator=gen, dtype=torch.float32)
        _pools[key] = (vals * (hi - lo) + lo).to(dtype)
    return _pools[key]


def _range(name, t):
    # fp8 weights pair with per-block float scales; [-1, 1] keeps most values
    # above the e4m3 subnormal floor, so the dequantized weight is nonzero.
    if t.element_size() == 1:
        return -1.0, 1.0
    if name.endswith(("scale", "scale_inv")):
        return 1e-3, 2e-3
    return -1e-3, 1e-3


def _fill_model(model):
    import torch

    seen = set()
    start = time.time()
    nbytes = 0
    # The state dict holds what a checkpoint load writes: parameters and
    # persistent buffers. Non-persistent buffers (the RoPE cos/sin cache) are
    # computed on device at init in the layout the compiled programs expect;
    # a host copy into them changes that layout.
    named = list(model.state_dict(keep_vars=True).items())
    for i, (name, t) in enumerate(named):
        if id(t) in seen or t.device.type == "meta":
            continue
        seen.add(id(t))
        # Scalars are KV/activation scales; their defaults keep fp8 KV finite.
        if not torch.is_floating_point(t) or t.numel() <= 1:
            continue
        lo, hi = _range(name, t)
        pool = _pool(torch, t.dtype, lo, hi)
        n = t.numel()
        off = (i * 7919) % _POOL_N
        reps = -(-(off + n) // _POOL_N)
        src = pool[off:off + n] if reps == 1 else pool.repeat(reps)[off:off + n]
        with torch.no_grad():
            t.data.copy_(src.view(t.shape))
        nbytes += n * t.element_size()
    print(f"[p_leg_bench] synthetic weights: {nbytes / 2**30:.1f} GiB in "
          f"{time.time() - start:.0f} s", flush=True)


def _patch_dummy_loader(module):
    cls = module.DummyModelLoader
    original = cls.load_weights

    def load_weights(self, model, model_config):
        if any(module.get_layerwise_info(m).can_load() for m in model.modules()):
            return original(self, model, model_config)
        _fill_model(model)

    cls.load_weights = load_weights


def _patch_scheduler(module):
    path = os.environ.get("P_LEG_STEP_LOG")
    if not path:
        return
    cls = module.Scheduler
    original = cls.update_from_output

    def update_from_output(self, scheduler_output, model_runner_output):
        rec = {
            "t": time.time(),
            "tok": scheduler_output.total_num_scheduled_tokens,
            "reqs": len(scheduler_output.num_scheduled_tokens),
        }
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return original(self, scheduler_output, model_runner_output)

    cls.update_from_output = update_from_output


_PATCHES = {
    "vllm.model_executor.model_loader.dummy_loader": _patch_dummy_loader,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
}


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.pop(name, None)
        if patch is None:
            return None
        if not _PATCHES:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched_exec_module(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = patched_exec_module
        return spec


sys.meta_path.insert(0, _PatchFinder())

# This file shadows any sitecustomize further down sys.path; run that one too.
_self = sys.modules.get(__name__)
sys.modules.pop(__name__, None)
_here = __file__
for _entry in sys.path:
    _spec = (
        importlib.machinery.PathFinder.find_spec(__name__, [_entry]) if _entry else None
    )
    if _spec and _spec.origin and _spec.origin != _here:
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        break
if _self is not None:
    sys.modules[__name__] = _self
"""


@dataclasses.dataclass
class Case:
    ctx: int
    new: int
    reqs: int

    @classmethod
    def parse(cls, text: str) -> Case:
        ctx, new, reqs = (int(x) for x in text.split(":"))
        return cls(ctx, new, reqs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--case",
        action="append",
        type=Case.parse,
        help="CTX:NEW:REQS, repeatable (default 92160:10240:16). CTX should be "
        "a multiple of the block size so the GDN state at the boundary is cached.",
    )
    p.add_argument("--rounds", type=int, default=3, help="unprofiled rounds per case")
    p.add_argument(
        "--profile", action="store_true", help="add one xprof round per case"
    )
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--compile-sizes", default="2048,8192")
    p.add_argument("--block-size", type=int, default=640)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument(
        "--mtp", type=int, default=1, help="MTP draft tokens on the P leg (0: off)"
    )
    p.add_argument(
        "--scheduler-cls",
        default="",
        help="empty: vLLM's scheduler (no cold-chunk cap or step cap)",
    )
    p.add_argument("--no-async-scheduling", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="stages; above 1 the engine runs TP1 per stage, no MTP, no async "
        "scheduling, no SP and no fused EP kernel (pp_validation.py)",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        help="KEY=VALUE set after P_LEG_ENV; KEY= with no value unsets KEY",
    )
    p.add_argument("--model-dir", default="/workspace/tmp/agentx-p2/qwen35-397b-config")
    p.add_argument("--out-dir", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--cache-root", default="/tmp/p-leg-cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dry-run", action="store_true", help="print the env and engine args only"
    )
    a = p.parse_args(argv)
    if not a.case:
        a.case = [Case(92160, 10240, 16)]
    return a


def default_out_dir(a: argparse.Namespace) -> str:
    name = os.environ.get("CDK_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    if a.tag:
        name += "-" + a.tag
    return f"/tmp/p-leg-bench/{name}"


def build_env(a: argparse.Namespace, out_dir: str) -> tuple[dict[str, str], list[str]]:
    """Returns the variables to set and the ones to remove."""
    env = dict(P_LEG_ENV)
    unset = []
    if a.pipeline_parallel_size > 1:
        # One rank per stage: no TP collectives, so SP and the fused EP kernel
        # have nothing to shard, and the tp8_ep8 placement table does not
        # apply. Hybrid-state retention is untested under PP.
        env.update(
            TPU_MOE_SEQUENCE_PARALLEL="0",
            TPU_SP_FP8_GATHER="0",
            USE_MOE_FUSED_EP_KERNEL="0",
            TPU_HYBRID_STATE_RETENTION="0",
        )
        env.pop("TPU_EXPERT_PLACEMENT", None)
        unset.append("TPU_EXPERT_PLACEMENT")
    for kv in a.env:
        key, _, val = kv.partition("=")
        if val:
            env[key] = val
        else:
            env.pop(key, None)
            unset.append(key)
    # Not every kernel knob is in vllm_torchtpu's compile-cache key, so each
    # set of overrides compiles into its own directory.
    overrides = hashlib.sha256("\n".join(sorted(a.env)).encode()).hexdigest()[:8]
    cache = os.path.join(a.cache_root, _stack_tag(), "env-" + overrides)
    for key, sub in (
        ("VLLM_CACHE_ROOT", "vllm"),
        ("JAX_COMPILATION_CACHE_DIR", "jax_cache"),
        ("JAX_COMPILER_CACHE_DIR", "jax_cache"),
        ("TORCH_TPU_CACHE_DIR", "torch_tpu_cache"),
        ("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "torch_tpu_cache"),
        ("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT", "torch_tpu_tier3"),
        ("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", "torch_tpu_tier3"),
        ("VLLM_XLA_CACHE_PATH", "xla_cache"),
    ):
        env.setdefault(key, os.path.join(cache, sub))
    env["P_LEG_STEP_LOG"] = os.path.join(out_dir, "steps.jsonl")
    site_dir = os.path.join(out_dir, "sitecustomize")
    env["PYTHONPATH"] = os.pathsep.join(
        x for x in (site_dir, os.environ.get("PYTHONPATH", "")) if x
    )
    return env, unset


def _stack_tag() -> str:
    """Names the installed runtime, so compile caches never mix stacks."""
    from importlib.metadata import PackageNotFoundError, version

    parts = []
    for dist in ("torch-tpu", "libtpu", "jax", "vllm"):
        try:
            parts.append(f"{dist}-{version(dist)}")
        except PackageNotFoundError:
            parts.append(f"{dist}-none")
    return "_".join(parts).replace("+", ".")


def engine_argv(a: argparse.Namespace, profile_dir: str) -> list[str]:
    compile_sizes = [int(x) for x in a.compile_sizes.split(",") if x]
    argv = [
        "--model", a.model_dir,
        "--load-format", "dummy",
        "--trust-remote-code",
        "--seed", "42",
        "--max-model-len", str(a.max_model_len),
        "--disable-custom-all-reduce",
        "--gpu-memory-utilization", str(a.gpu_memory_utilization),
        "--kv-cache-dtype", "fp8",
        "--quantization", "fp8",
        "--language-model-only",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--max-num-batched-tokens", str(a.max_num_batched_tokens),
        "--long-prefill-token-threshold", "262144",
        "--max-num-seqs", str(a.max_num_seqs),
        "--no-disable-hybrid-kv-cache-manager",
        "--attention-backend", "CUSTOM",
        "--mamba-cache-mode", "align",
        "--compilation-config", json.dumps({
            "backend": "vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor",
            "compile_sizes": compile_sizes,
            "inductor_compile_config": {
                "enable_auto_functionalized_v2": False,
                "size_asserts": False,
                "alignment_asserts": False,
                "scalar_asserts": False,
            },
        }),
        "--enable-prefix-caching",
        "--block-size", str(a.block_size),
        "--disable-sliding-window",
        "--tensor-parallel-size", str(a.tensor_parallel_size),
        "--pipeline-parallel-size", str(a.pipeline_parallel_size),
        "--prefill-context-parallel-size", "1",
        "--cp-kv-cache-interleave-size", "1",
        "--data-parallel-size", "1",
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", profile_dir,
        "--profiler-config.ignore_frontend", "true",
    ]  # fmt: skip
    pp = a.pipeline_parallel_size > 1
    if not pp:
        argv.append("--enable-expert-parallel")
    if pp:
        argv.append("--no-async-scheduling")
    elif not a.no_async_scheduling:
        argv.append("--async-scheduling")
    if a.mtp > 0 and not pp:
        argv += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": a.mtp}),
        ]
    if a.scheduler_cls:
        argv += ["--scheduler-cls", a.scheduler_cls]
    return argv


def _read_steps(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            # Appends from another pod over NFS can leave NUL-filled gaps.
            line = line.strip("\x00 \n")
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _step_stats(steps: list[dict], t0: float, t1: float, full: int) -> dict:
    """Summarizes the step records that landed inside one round."""
    win = [s for s in steps if t0 <= s["t"] <= t1 + 0.5]
    out = {
        "steps": len(win),
        "step_tokens": [s["tok"] for s in win],
        "step_reqs": [s["reqs"] for s in win],
    }
    # An interval between two consecutive full steps is one step of device
    # work plus whatever host time the pipeline does not hide.
    iv = [
        (b["t"] - a["t"]) * 1e3
        for a, b in zip(win, win[1:])
        if a["tok"] == full and b["tok"] == full
    ]
    if not iv and win:
        top_tok = max(s["tok"] for s in win)
        iv = [
            (b["t"] - a["t"]) * 1e3
            for a, b in zip(win, win[1:])
            if a["tok"] == top_tok and b["tok"] == top_tok
        ]
        full = top_tok
    if iv:
        q = statistics.quantiles(iv, n=10) if len(iv) >= 2 else [iv[0]] * 9
        out.update(
            full_step_tok=full,
            full_step_ms_median=round(statistics.median(iv), 2),
            full_step_ms_p10=round(q[0], 2),
            full_step_ms_p90=round(q[-1], 2),
            full_step_intervals=len(iv),
            rows_per_s_steady=round(full / (statistics.median(iv) / 1e3)),
        )
    return out


def run(a: argparse.Namespace, out_dir: str) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    profile_dir = os.path.join(out_dir, "xprof")
    from vllm.config import FaultToleranceConfig

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(engine_argv(a, profile_dir))
    # The CLI default for the config is a dict, and EngineArgs turns a dict
    # into enable_fault_tolerance=True, which only external-LB DP accepts.
    ns.fault_tolerance_config = FaultToleranceConfig()
    engine_args = EngineArgs.from_cli_args(ns)
    t_start = time.time()
    llm = LLM.from_engine_args(engine_args)
    startup_s = time.time() - t_start
    step_log = os.environ["P_LEG_STEP_LOG"]
    rng = np.random.default_rng(a.seed)
    full = a.max_num_batched_tokens
    sp_probe = SamplingParams(
        max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=1, detokenize=False
    )
    sp = SamplingParams(
        max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False
    )

    def tokens(n: int) -> list[int]:
        return rng.integers(1000, 200000, n).tolist()

    cases = []
    for ci, case in enumerate(a.case):
        prefixes = [tokens(case.ctx) for _ in range(case.reqs)]
        t0 = time.time()
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in prefixes], sp_probe, use_tqdm=False
        )
        build_s = time.time() - t0
        lps = [
            next(iter(o.outputs[0].logprobs[0].values())).logprob
            for o in outs
            if o.outputs[0].logprobs
        ]
        case_out = {
            "ctx": case.ctx,
            "new": case.new,
            "reqs": case.reqs,
            "prefix_build_s": round(build_s, 2),
            "prefix_build_cached": [o.num_cached_tokens or 0 for o in outs],
            "first_token_logprob_finite": bool(lps) and all(np.isfinite(lps)),
            "rounds": [],
        }
        n_rounds = a.rounds + (1 if a.profile else 0)
        for r in range(n_rounds):
            profiled = a.profile and r == n_rounds - 1
            prompts = [{"prompt_token_ids": p + tokens(case.new)} for p in prefixes]
            if profiled:
                llm.start_profile(profile_prefix=f"case{ci}")
            t0 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            t1 = time.time()
            if profiled:
                llm.stop_profile()
            cached = [o.num_cached_tokens or 0 for o in outs]
            rows = sum(len(p["prompt_token_ids"]) - c for p, c in zip(prompts, cached))
            rd = {
                "round": r,
                "profiled": profiled,
                "wall_s": round(t1 - t0, 3),
                "rows": rows,
                "rows_per_s_wall": round(rows / (t1 - t0)),
                "cached_min": min(cached),
                "cached_max": max(cached),
            }
            rd.update(_step_stats(_read_steps(step_log), t0, t1, full))
            case_out["rounds"].append(rd)
            print(f"[p_leg_bench] case {ci} round {r}: {json.dumps(rd)}", flush=True)
        plain = [x for x in case_out["rounds"] if not x["profiled"]]
        if plain:
            case_out["rows_per_s_wall_median"] = round(
                statistics.median(x["rows_per_s_wall"] for x in plain)
            )
            med = [
                x["full_step_ms_median"] for x in plain if "full_step_ms_median" in x
            ]
            if med:
                case_out["full_step_ms_median"] = round(statistics.median(med), 2)
                case_out["rows_per_s_steady"] = round(
                    full / (case_out["full_step_ms_median"] / 1e3)
                )
        # A measured request must start from the cached prefix (attention and
        # GDN state); one block of slack covers the align-mode boundary.
        case_out["cached_ok"] = all(
            x["cached_min"] >= case.ctx - a.block_size for x in case_out["rounds"]
        )
        cases.append(case_out)
    return {"startup_s": round(startup_s, 1), "cases": cases}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    out_dir = a.out_dir or default_out_dir(a)
    env, unset = build_env(a, out_dir)
    argv_engine = engine_argv(a, os.path.join(out_dir, "xprof"))
    if a.dry_run:
        print(
            json.dumps(
                {"env": env, "unset": unset, "engine_argv": argv_engine}, indent=1
            )
        )
        return 0
    os.makedirs(os.path.join(out_dir, "sitecustomize"), exist_ok=True)
    with open(os.path.join(out_dir, "sitecustomize", "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE)
    os.environ.update(env)
    for key in unset:
        os.environ.pop(key, None)
    config = {
        "argv": sys.argv[1:] if argv is None else argv,
        "cases": [dataclasses.asdict(c) for c in a.case],
        "env_overrides": a.env,
        "engine_argv": argv_engine,
        "stack": _stack_tag(),
        "out_dir": out_dir,
    }
    result = {"config": config}
    try:
        result.update(run(a, out_dir))
        result["ok"] = all(
            c["cached_ok"] and c["first_token_logprob_finite"] for c in result["cases"]
        )
    finally:
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=1)
    print("P_LEG_RESULT " + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
