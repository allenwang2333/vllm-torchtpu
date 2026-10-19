#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measures the Qwen3.5-397B AgentX decode leg's step time on one 2x2x1 host.

The engine is the offline vLLM engine with the D leg's serving configuration
(``D_LEG_ENV`` is the decode container env of j-8602a6cb, the w13 1P1D DCP4
canonical run, minus the KV transfer / Raiden / offload knobs): TP8 + EP8 with
DCP=4 (interleave 128), fp8 KV, block 640 with Mamba "align" prefix caching,
async scheduling, MTP K=3 with the synthetic rejection sampler at acceptance
length 3.39. Weights are synthetic (see p_leg_bench.py).

One shared prefix of CTX tokens is prefilled once (CTX is a multiple of the
block size so the GDN state at the boundary is cached). Each case then runs N
requests of prefix + SUFFIX fresh tokens, generating enough tokens for about
STEPS pure decode steps with all N requests running, so every decode step
reads CTX tokens of K/V per sequence and the GDN state of N sequences.

Per case and round: the median interval between consecutive pure decode steps
(N running, 4N scheduled tokens) from a scheduler hook. With --profile-n the
case gets one more round in which --profile-steps pure decode steps run under
the profiler; traces land in OUT/xprof/n<N>/.

Results: OUT/result.json plus a ``D_LEG_RESULT {...}`` line on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p_leg_bench as PL  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

D_LEG_ENV = {
    "ATTN_BUCKETIZED_NUM_REQS": "true",
    "ATTN_CUSTOM_NUM_REQS_BUCKETS": "8,16,32,64",
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
    "TPU_ENABLE_GDN_DYNAMIC_TILING": "1",
    "TPU_EXPERT_PLACEMENT": str(
        REPO_ROOT / "expert_placement/expert_placement_tp8_ep8_agentx_corr_c32.json"
    ),
    "TPU_FUSE_POST_ATTN_NORM": "1",
    "TPU_FUSE_PRE_ATTN_NORM": "1",
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": "1",
    "TPU_GDN_CONV_STATE_TILE_PAD": "1",
    "TPU_GDN_FUSED_NORM_QUANT": "1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_KERNEL_PAGE_TOKENS": "128",
    "TPU_KV_CACHE_HEADROOM_MIB": "6144",
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER": "1",
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": "33",
    "TPU_MOE_LOWTOK": "1",
    "TPU_MOE_LOWTOK_MAX_TOKENS": "256",
    "TPU_MOE_SEQUENCE_PARALLEL": "1",
    "TPU_MOE_SKIP_PADDED_TOKENS": "1",
    "TPU_PREFILL_SKIP_VERIFY_WARMUP": "1",
    "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    "TPU_QWEN35_AUTO_DEFAULTS": "1",
    "TPU_ROPE_CACHE_TRUNCATE": "1",
    # Production decode pods set it; env_override keys decode defaults on it
    # (it keeps TPU_MOE_FUSED_EP_STEP_MIN_TOKENS at 33 instead of 64).
    "TPU_RAIDEN_JOB_NAME": "decode",
    "TPU_RPA_FOLD_KV_HEAD_DIM": "0",
    "TPU_SCHED_FINISH_DECODE_STEPS": "0",
    # Production caps running at --max-num-seqs; the MK-derived cap is off.
    "TPU_SCHED_MK_MAX_RUNNING": "0",
    "TPU_SCHED_WARM_STREAK": "0",
    "TPU_SMALL_ALL_REDUCE": "1",
    "TPU_SP_FP8_GATHER": "1",
    "TPU_SP_MIN_ROWS": "512",
    "TPU_TOKEN_BUCKET_EXTRA": "4,8,48,2048",
    "TPU_TOKEN_BUCKET_LINEAR_INTERVAL": "16",
    "TPU_TOKEN_BUCKET_LINEAR_UNTIL": "64",
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
    "VLLM_ENGINE_READY_TIMEOUT_S": "5400",
    "VLLM_KV_CACHE_LAYOUT": "HND",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_MOE_CHUNK_SIZE": "256",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": "1",
    # Megakernel off by default: it only serves <= 104 verify rows (26 seqs),
    # below every case measured here, and its warmup costs about an hour.
    "VLLM_TPU_QWEN35_MEGAKERNEL": "0",
    "VLLM_TPU_QWEN35_MTP_MK": "0",
    "VLLM_TPU_QWEN35_MK_BUCKETS": "0",
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B": "160",
    "VLLM_TPU_QWEN35_MK_MODE": "ep",
    "VLLM_TPU_QWEN35_MK_SPLIT_WIDE": "1",
    "VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS": "4,8,12,16,20,24,32,40,48,56,64,80,84,88,96,104",
    "VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS": "4,8,12,16,20,24,32,40,48,56,64,72,80",
    # The spec-decode warmup sweeps every request count up to max_num_seqs
    # (~20 min) but without it the async glue recompiles at run time on
    # live-count/acceptance shapes and steps stall (j-ed8e470b: 135 ms median,
    # 700 ms mean at 40 seqs). Production runs it.
    "SPEC_WARMUP": "1",
    # Production value. =1 logs every graph compiled after warmup, but walks
    # torch.tpu's ~3000-entry cache several times per step: ~150 ms per step
    # of host time at any batch (j-7a352ac8), so timing runs keep it off.
    "VLLM_XLA_CHECK_RECOMPILATION": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
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

# compile_sizes of the large-batch decode runs (j-849474b7, hyb96, max-num-seqs 96).

SAMPLER = r"""
import collections as _col
import json as _json
import os as _os
import sys as _sys
import threading as _thr
import time as _time


def _start_sampler(tag, target_ident=None, period=0.004):
    flag = _os.environ.get("D_LEG_SAMPLE_FLAG")
    if not flag:
        return
    out = _os.path.join(_os.path.dirname(flag), "stacks_%s_%d.json" % (tag, _os.getpid()))
    ident = target_ident or _thr.main_thread().ident
    data = {}

    def key(f):
        c = f.f_code
        return "%s:%d:%s" % (c.co_filename.split("site-packages/")[-1][-60:], f.f_lineno, c.co_name)

    def run():
        last_dump = _time.time()
        while True:
            _time.sleep(period)
            try:
                with open(flag) as fh:
                    win = fh.read().strip()
            except OSError:
                win = ""
            if win:
                fr = _sys._current_frames().get(ident)
                if fr is not None:
                    d = data.setdefault(win, {"n": 0, "leaf": _col.Counter(), "cum": _col.Counter(),
                                              "fn": _col.Counter()})
                    d["n"] += 1
                    d["leaf"][key(fr)] += 1
                    seen = set()
                    seenf = set()
                    depth = 0
                    while fr is not None and depth < 80:
                        k = key(fr)
                        if k not in seen:
                            d["cum"][k] += 1
                            seen.add(k)
                        c = fr.f_code
                        kf = "%s:%s" % (c.co_filename.split("site-packages/")[-1][-60:], c.co_name)
                        if kf not in seenf:
                            d["fn"][kf] += 1
                            seenf.add(kf)
                        fr = fr.f_back
                        depth += 1
            if _time.time() - last_dump > 10:
                last_dump = _time.time()
                try:
                    with open(out + ".tmp", "w") as fh:
                        _json.dump({w: {"n": d["n"], "leaf": d["leaf"].most_common(60),
                                        "cum": d["cum"].most_common(150),
                                        "fn": d["fn"].most_common(150)}
                                    for w, d in data.items()}, fh)
                    _os.replace(out + ".tmp", out)
                except Exception:
                    pass

    _thr.Thread(target=run, daemon=True, name="d-leg-sampler").start()


def _patch_moe_router_noise(module):
    # Synthetic weights + one shared prefix make every sequence's hidden state
    # nearly the same, so all rows pick the same few experts (j-ed8e470b: the
    # few-token MoE kernel read ~5-15 experts per rank per layer instead of
    # ~all 64 at 256 rows). Adds a deterministic pseudo-random term to the
    # router logits that dominates them: per sequence (4 rows) with "corr",
    # per row with "indep", different per layer, global row index under SP.
    mode = _os.environ.get("D_LEG_ROUTER_NOISE", "")
    if not mode:
        return
    cls = module.VllmFp8MoEMethodTPU
    original = cls.apply_monolithic
    group = 4 if mode == "corr" else 1
    scale = float(_os.environ.get("D_LEG_ROUTER_NOISE_SCALE", "20"))

    def apply_monolithic(self, layer, x, router_logits, *a, **k):
        import re as _re
        import torch
        from vllm_torchtpu.layers.adapter import sequence_parallel as _sp
        T, E = router_logits.shape[0], router_logits.shape[-1]
        m = _re.search(r"layers\.(\d+)", str(getattr(layer, "layer_name", "")))
        lid = (int(m.group(1)) if m else 0) + (100 if "mtp" in str(getattr(layer, "layer_name", "")) else 0)
        off = 0
        if _sp.is_active() and bool(getattr(getattr(layer, "moe_config", None), "is_sequence_parallel", False)):
            from vllm.distributed import get_tensor_model_parallel_rank
            off = get_tensor_model_parallel_rank() * T
        dev = router_logits.device
        r = (torch.arange(T, device=dev, dtype=torch.int32) + off) // group
        e = torch.arange(E, device=dev, dtype=torch.int32)
        # integer hash (int32 wrap-around), uniform in [0, scale)
        h = (r[:, None] * 73856093) ^ (e[None, :] * 19349663) ^ (lid * 83492791 + 1)
        h = (h ^ (h >> 13)) * 1274126177
        h = h ^ (h >> 16)
        v = (h & 0xFFFF).to(torch.float32) * (scale / 65536.0)
        return original(self, layer, x, router_logits + v.to(router_logits.dtype), *a, **k)

    cls.apply_monolithic = apply_monolithic


def _patch_engine_core(module):
    cls = module.EngineCoreProc
    original = cls.run_busy_loop

    def run_busy_loop(self, *a, **k):
        _start_sampler("enginecore", _thr.get_ident())
        return original(self, *a, **k)

    cls.run_busy_loop = run_busy_loop
"""

PROD_COMPILE_SIZES = "4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512,2048"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--n", default="40,48,64,80,96,112,128", help="running seqs per case"
    )
    p.add_argument("--ctx", type=int, default=80000)
    p.add_argument("--suffix", type=int, default=64)
    p.add_argument(
        "--steps", type=int, default=80, help="target full decode steps/round"
    )
    p.add_argument(
        "--rounds", type=int, default=3, help="rounds per case (first warms)"
    )
    p.add_argument("--profile-n", default="", help="cases that get an xprof round")
    p.add_argument("--profile-steps", type=int, default=6)
    p.add_argument("--acceptance", type=float, default=3.39)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--compile-sizes", default=PROD_COMPILE_SIZES)
    p.add_argument("--block-size", type=int, default=640)
    p.add_argument("--dcp", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE (KEY= unsets)")
    p.add_argument("--model-dir", default=str(REPO_ROOT / "qwen35-397b-config"))
    p.add_argument("--out-dir", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--cache-root", default="/tmp/d-leg-cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checksum",
        type=int,
        default=1,
        help="greedy output-token checksum per case (bit-exactness A/B)",
    )
    p.add_argument("--router-noise", default="corr", help="corr | indep | none")
    p.add_argument(
        "--sample",
        type=int,
        default=1,
        help="stack-sample the engine core and driver during round 1 of each case",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_env(a, out_dir):
    env = dict(D_LEG_ENV)
    unset = []
    for kv in a.env:
        key, _, val = kv.partition("=")
        if val:
            env[key] = val
        else:
            env.pop(key, None)
            unset.append(key)
    overrides = hashlib.sha256("\n".join(sorted(a.env)).encode()).hexdigest()[:8]
    cache = os.path.join(a.cache_root, PL._stack_tag(), "env-" + overrides)
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
    # The step log, the sampler flag (read every 4 ms) and the stack dumps
    # live on local tmpfs: the out dir is a gcsfuse mount under CDK, where one
    # append per engine step cost ~150 ms and made every step host-bound
    # (j-ed8e470b, j-dd39b313). They are copied to the out dir at the end.
    scratch = os.path.join(
        "/dev/shm", "d-leg-" + hashlib.sha256(out_dir.encode()).hexdigest()[:8]
    )
    os.makedirs(scratch, exist_ok=True)
    env["D_LEG_SCRATCH"] = scratch
    env["P_LEG_STEP_LOG"] = os.path.join(scratch, "steps.jsonl")
    if a.router_noise != "none":
        env["D_LEG_ROUTER_NOISE"] = a.router_noise
    if a.sample:
        env["D_LEG_SAMPLE_FLAG"] = os.path.join(scratch, "sample_flag")
    site_dir = os.path.join(out_dir, "sitecustomize")
    env["PYTHONPATH"] = os.pathsep.join(
        x for x in (site_dir, os.environ.get("PYTHONPATH", "")) if x
    )
    return env, unset


def engine_argv(a, profile_dir):
    compile_sizes = [int(x) for x in a.compile_sizes.split(",") if x]
    return [
        "--model", a.model_dir,
        "--load-format", "dummy",
        "--trust-remote-code",
        "--seed", "42",
        "--max-model-len", str(a.max_model_len),
        "--enable-expert-parallel",
        "--disable-custom-all-reduce",
        "--gpu-memory-utilization", str(a.gpu_memory_utilization),
        "--kv-cache-dtype", "fp8",
        "--quantization", "fp8",
        "--language-model-only",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--max-num-batched-tokens", str(a.max_num_batched_tokens),
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
        "--async-scheduling",
        "--block-size", str(a.block_size),
        "--disable-sliding-window",
        "--tensor-parallel-size", "8",
        "--prefill-context-parallel-size", "1",
        "--decode-context-parallel-size", str(a.dcp),
        "--cp-kv-cache-interleave-size", "128",
        "--data-parallel-size", "1",
        "--speculative-config", json.dumps({
            "method": "mtp",
            "num_speculative_tokens": a.k,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": a.acceptance,
        }),
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", profile_dir,
        "--profiler-config.ignore_frontend", "true",
    ]  # fmt: skip


def _full_steps(steps, n, k):
    return [s for s in steps if s["reqs"] == n and s["tok"] == n * (1 + k)]


def _intervals(win, n, k):
    full = n * (1 + k)
    return [
        (b["t"] - a["t"]) * 1e3
        for a, b in zip(win, win[1:])
        if a["reqs"] == n and b["reqs"] == n and a["tok"] == full and b["tok"] == full
    ]


def _summ(iv):
    if not iv:
        return {}
    q = statistics.quantiles(iv, n=10) if len(iv) >= 2 else [iv[0]] * 9
    return {
        "step_ms_median": round(statistics.median(iv), 2),
        "step_ms_mean": round(statistics.fmean(iv), 2),
        "step_ms_p10": round(q[0], 2),
        "step_ms_p90": round(q[-1], 2),
        "intervals": len(iv),
    }


def run(a, out_dir):
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.config import FaultToleranceConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    profile_dir = os.path.join(out_dir, "xprof")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(engine_argv(a, profile_dir))
    ns.fault_tolerance_config = FaultToleranceConfig()
    engine_args = EngineArgs.from_cli_args(ns)
    t_start = time.time()
    llm = LLM.from_engine_args(engine_args)
    startup_s = time.time() - t_start
    print(f"[d_leg_bench] engine up in {startup_s:.0f} s", flush=True)
    if os.environ.get("D_LEG_SAMPLE_FLAG"):
        ns_ = {}
        exec(SAMPLER, ns_)
        ns_["_start_sampler"]("driver")
    step_log = os.environ["P_LEG_STEP_LOG"]
    rng = np.random.default_rng(a.seed)

    def tokens(n):
        return rng.integers(1000, 200000, n).tolist()

    prefix = tokens(a.ctx)
    t0 = time.time()
    outs = llm.generate(
        [{"prompt_token_ids": prefix}],
        SamplingParams(
            max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False
        ),
        use_tqdm=False,
    )
    prefix_s = time.time() - t0
    print(f"[d_leg_bench] prefix {a.ctx} tokens built in {prefix_s:.0f} s", flush=True)

    k = a.k
    profile_ns = {int(x) for x in a.profile_n.split(",") if x}
    cases = []
    for n in [int(x) for x in a.n.split(",") if x]:
        # Enough tokens for `steps` full steps after the suffix prefill ramp.
        max_tokens = int(math.ceil(a.acceptance * (a.steps + 6))) + 4
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False
        )
        case = {"n": n, "rows": n * (1 + k), "max_tokens": max_tokens, "rounds": []}
        n_rounds = a.rounds + (1 if n in profile_ns else 0)
        for r in range(n_rounds):
            profiled = n in profile_ns and r == n_rounds - 1
            # Same suffixes every round (same seed per case): the measured
            # greedy outputs are comparable across runs for bit-exactness.
            srng = np.random.default_rng(1000 + n)
            prompts = [
                {
                    "prompt_token_ids": prefix
                    + srng.integers(1000, 200000, a.suffix).tolist()
                }
                for _ in range(n)
            ]
            flag = os.environ.get("D_LEG_SAMPLE_FLAG")
            if flag:
                with open(flag, "w") as fh:
                    fh.write(f"n{n}" if r == 1 and not profiled else "")
            t0 = time.time()
            if not profiled:
                outs = llm.generate(prompts, sp, use_tqdm=False)
            else:
                outs = _profiled_generate(
                    llm, prompts, sp, n, k, a.profile_steps, step_log, f"n{n}"
                )
            t1 = time.time()
            if flag:
                with open(flag, "w") as fh:
                    fh.write("")
            steps = [s for s in PL._read_steps(step_log) if t0 <= s["t"] <= t1 + 0.5]
            iv = _intervals(steps, n, k)
            rd = {
                "round": r,
                "profiled": profiled,
                "sampled": bool(flag) and r == 1 and not profiled,
                "wall_s": round(t1 - t0, 2),
                "steps": len(steps),
                "full_steps": len(_full_steps(steps, n, k)),
            }
            rd.update(_summ(iv))
            toks = [list(o.outputs[0].token_ids) for o in outs]
            rd["gen_tokens"] = sum(len(t) for t in toks)
            rd["cached_min"] = min((o.num_cached_tokens or 0) for o in outs)
            if a.checksum:
                h = hashlib.md5(json.dumps(toks).encode()).hexdigest()[:12]
                rd["out_md5"] = h
            case["rounds"].append(rd)
            print(f"[d_leg_bench] n={n} round {r}: {json.dumps(rd)}", flush=True)
        plain = [
            x for x in case["rounds"][1:] if not x["profiled"] and "step_ms_median" in x
        ]
        if any(not x["sampled"] for x in plain):
            plain = [x for x in plain if not x["sampled"]]
        if plain:
            case["step_ms_median"] = round(
                statistics.median(x["step_ms_median"] for x in plain), 2
            )
            case["tok_per_s"] = round(n * a.acceptance / (case["step_ms_median"] / 1e3))
        case["cached_ok"] = all(
            x["cached_min"] >= a.ctx - a.block_size for x in case["rounds"]
        )
        cases.append(case)
        print(
            f"[d_leg_bench] CASE {json.dumps({k2: v for k2, v in case.items() if k2 != 'rounds'})}",
            flush=True,
        )
    time.sleep(12)
    scratch = os.environ.get("D_LEG_SCRATCH")
    if scratch:
        import shutil

        for f in os.listdir(scratch):
            try:
                shutil.copy(os.path.join(scratch, f), out_dir)
            except OSError:
                pass
    _print_stacks(out_dir)
    return {
        "startup_s": round(startup_s, 1),
        "prefix_s": round(prefix_s, 1),
        "cases": cases,
    }


def _print_stacks(out_dir, top=30):
    import glob

    for f in sorted(glob.glob(os.path.join(out_dir, "stacks_*.json"))):
        try:
            data = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[d_leg_bench] stacks {f}: unreadable ({e})")
            continue
        for win, d in sorted(data.items()):
            n = max(d["n"], 1)
            print(f"[d_leg_bench] STACKS {os.path.basename(f)} {win}: {d['n']} samples")
            for k, c in d["fn"][:top]:
                print(f"    fn  {100 * c / n:5.1f}%  {k}")
            for k, c in d["leaf"][:15]:
                print(f"    leaf {100 * c / n:5.1f}%  {k}")


def _profiled_generate(llm, prompts, sp, n, k, profile_steps, step_log, prefix):
    """Runs one round by hand-stepping the engine; profiles `profile_steps`
    pure decode steps once all n requests are decoding."""
    from vllm.sampling_params import RequestOutputKind

    sp = sp.clone()
    sp.output_kind = RequestOutputKind.FINAL_ONLY
    eng = llm.llm_engine
    base = len(PL._read_steps(step_log))
    for i, p in enumerate(prompts):
        eng.add_request(f"prof-{prefix}-{i}", p, sp)
    finals = {}
    state = "wait"
    n_full_at_start = 0
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                finals[o.request_id] = o
        steps = PL._read_steps(step_log)[base:]
        nfull = len(_full_steps(steps, n, k))
        if state == "wait" and nfull >= 4:
            llm.start_profile(profile_prefix=prefix)
            state, n_full_at_start = "on", nfull
            print(f"[d_leg_bench] profile start at full step {nfull}", flush=True)
        elif state == "on" and nfull >= n_full_at_start + profile_steps:
            llm.stop_profile()
            state = "done"
            print(f"[d_leg_bench] profile stop at full step {nfull}", flush=True)
    if state == "on":
        llm.stop_profile()
    return [finals[f"prof-{prefix}-{i}"] for i in range(len(prompts))]


def _sitecustomize():
    src = PL.SITECUSTOMIZE
    anchor = "_PATCHES = {"
    assert anchor in src
    src = src.replace(
        anchor,
        SAMPLER + "\n" + anchor + '\n    "vllm.v1.engine.core": _patch_engine_core,'
        '\n    "vllm_torchtpu.layers.adapter.quantization.fp8": _patch_moe_router_noise,',
        1,
    )
    return src


def main(argv=None):
    a = parse_args(argv)
    name = os.environ.get("CDK_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_dir or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp/d-leg-bench"),
        name + ("-" + a.tag if a.tag else ""),
    )
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
        f.write(_sitecustomize())
    os.environ.update(env)
    for key in unset:
        os.environ.pop(key, None)
    result = {
        "config": {
            "argv": sys.argv[1:] if argv is None else argv,
            "env_overrides": a.env,
            "engine_argv": argv_engine,
            "stack": PL._stack_tag(),
            "out_dir": out_dir,
        }
    }
    try:
        result.update(run(a, out_dir))
        result["ok"] = all(
            c["cached_ok"] and "step_ms_median" in c for c in result["cases"]
        )
    finally:
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=1)
    print("D_LEG_RESULT " + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
