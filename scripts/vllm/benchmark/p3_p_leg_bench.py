#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prefill leg step rate of Qwen3.5-397B under a chosen layout (phase 3 harness).

Port of agentx-p2-d2 / agentx-pp-p scripts/vllm/benchmark/p_leg_bench.py onto
the vLLM 0.31 / torch_tpu 0.2.0rc1 tree, extended for the phase 4 matrix:

  * layout: --tensor-parallel-size, --pipeline-parallel-size,
    --prefill-context-parallel-size, --data-parallel-size (attention DP),
    --expert-parallel, --num-hosts (TPU_* bounds for 2x2xN slices);
  * routing: --routing hash (near-uniform per token), model (the router on
    synthetic weights, skewed), trace (replay of a recorded table given with
    --routing-file); routing, source identity, layout and quantization are in
    the compile-cache key;
  * requests: --mode warm (cached CTX prefix plus NEW fresh tokens, the AgentX
    turn shape) and --mode cold (CTX+NEW uncached tokens), from --case
    CTX:NEW:REQS or from --prompt-file (frozen input, one JSON per line with
    "prefix" and "new" token id lists);
  * chunk grid: --chunk-grid 4096,8192 reruns the bench in a fresh process per
    max-num-batched-tokens value, each with its own run directory;
  * accounting: computed rows (prompt minus cached tokens, and the scheduler's
    scheduled tokens) never padded rows; padded rows are reported next to them;
  * timing: synchronized wall time per round, per-step intervals from the
    scheduler hook (steps.jsonl), and with --profile one xprof round per case
    for device time (parse with p3_xplane_device_time.py); queue, init, load,
    compile, warmup and timed windows are reported separately.

Weights are synthetic unless --weights real (then --model-dir must hold the
checkpoint and --load-format the loader). Run in a pod through the
p3-bench-pod recipe (BENCH=p_leg) or directly:

    python scripts/vllm/benchmark/p3_p_leg_bench.py --case 87040:10240:32 \
        --tensor-parallel-size 8 --routing hash --rounds 3

Results: <out_dir>/result.json (summary), steps.jsonl (per engine step),
env.json (environment id), plus a ``P3_P_LEG_RESULT {...}`` line on stdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_bench_common as common  # noqa: E402

REPO_ROOT = common.REPO_ROOT

# Prefill container env of the agentx-397b-pd-baseline cell on this tree
# (scripts/agentx-cell.sh 1 1 16, PREFILL_* prefix stripped) layered over the
# D1 trace container env captured by agentx-p2-d2, minus the KV transfer,
# offload tier, compile-cache path and frontend knobs, which do not reach the
# forward step.
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
    "TPU_DRAFT_LOAD_BARRIER": "1",
    # The values PrefixAwareScheduler resolves to as a kv_producer; the
    # offline engine is not one, and its defaults (64, 4096) run the
    # hybrid-retention tail pieces as steps of their own.
    "TPU_SCHED_FINISH_DECODE_STEPS": "0",
    "TPU_SCHED_STEP_TOTAL_CAP": "0",
    "TPU_ENABLE_GDN_DYNAMIC_TILING": "0",
    "TPU_EXPERT_PLACEMENT": str(
        common.PLACEMENT_DIR / "expert_placement_tp8_ep8_agentx_corr_c32.json"
    ),
    "TPU_FUSE_POST_ATTN_NORM": "1",
    "TPU_FUSE_PRE_ATTN_NORM": "1",
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": "1",
    "TPU_GDN_CONV_STATE_TILE_PAD": "1",
    "TPU_GDN_FUSED_NORM_QUANT": "1",
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
    # D2a: fp8 PV with a static scale (phase 2 record 40.3 to 41.3k rows/s).
    "TPU_RPA_NATIVE_FP8_PREFILL": "pvs",
    "TPU_SCHED_COLD_CHUNK_CAP": "2560",
    "TPU_SCHED_COLD_THRESHOLD": "4096",
    "TPU_SCHED_PAW_AGING_TOK_PER_S": "0",
    "TPU_SCHED_PAW_MAX_WAIT_S": "0",
    "TPU_SMALL_ALL_REDUCE": "0",
    "TPU_SP_FP8_GATHER": "1",
    "TPU_SP_MIN_ROWS": "512",
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
        "--case", action="append", type=Case.parse,
        help="CTX:NEW:REQS, repeatable (default 87040:10240:32). CTX should be a "
        "multiple of the block size (640, and 2176 under PP) so the GDN state at "
        "the boundary is cached.",
    )
    p.add_argument(
        "--prompt-file", default="",
        help="JSONL of {\"prefix\": [ids], \"new\": [ids]} from the frozen input; "
        "replaces --case (REQS from --reqs, records cycled)",
    )
    p.add_argument("--reqs", type=int, default=32, help="requests per round with --prompt-file or --shapes-from")
    p.add_argument(
        "--shapes-from", default="",
        help="per_request.csv(.gz) of the frozen input: each request of a round takes a real "
        "(context, uncached suffix) pair (random token content), see p3_bench_common.load_shapes",
    )
    p.add_argument("--shapes-select", choices=("sample", "quantiles"), default="sample")
    p.add_argument("--shapes-subset", choices=("all", "first_turn", "later_turn", "cold"), default="all")
    p.add_argument("--shapes-seed", type=int, default=0)
    p.add_argument(
        "--mode", action="append", choices=("warm", "cold"),
        help="warm: cached CTX prefix plus NEW fresh tokens (default); cold: CTX+NEW "
        "uncached tokens; repeatable",
    )
    p.add_argument("--rounds", type=int, default=3, help="unprofiled rounds per case")
    p.add_argument("--profile", action="store_true", help="add one xprof round per case")
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--chunk-grid", default="",
        help="comma list of max-num-batched-tokens; each runs in a fresh process with "
        "its own run directory",
    )
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument(
        "--compile-sizes", default="2048,8192",
        help="per-rank token sizes (under PCP the scheduler chunk is split over the "
        "PCP ranks, so give the per-rank ladder)",
    )
    p.add_argument("--block-size", type=int, default=640)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--mtp", type=int, default=1, help="MTP draft tokens on the P leg (0: off)")
    p.add_argument(
        "--scheduler-cls", default="",
        help="empty: vLLM's scheduler (no cold-chunk cap or step cap)",
    )
    p.add_argument("--no-async-scheduling", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument(
        "--pipeline-parallel-size", type=int, default=1,
        help="stages; above 1 this tree's pp_validation rejects stage TP>1, async "
        "scheduling, speculation other than producer MTP K=1 and SP, so those are "
        "turned off here",
    )
    p.add_argument("--prefill-context-parallel-size", type=int, default=1)
    p.add_argument("--cp-kv-cache-interleave-size", type=int, default=1)
    p.add_argument(
        "--data-parallel-size", type=int, default=1,
        help="attention DP replicas inside one engine (DP ranks x TP x PCP x PP ranks "
        "in total)",
    )
    p.add_argument("--expert-parallel", type=int, choices=(0, 1), default=1)
    p.add_argument("--num-hosts", type=int, default=1, help="2x2xN slice; sets TPU_* bounds")
    p.add_argument(
        "--distributed-executor-backend", default="",
        help="empty: vLLM default (mp); ray for multi-host engines",
    )
    p.add_argument(
        "--routing", choices=("hash", "model", "trace"), default="hash",
        help="hash: near-uniform expert choice per token (default); model: the router's "
        "own choice on synthetic weights, which is skewed; trace: replay --routing-file",
    )
    p.add_argument("--routing-file", default="", help="npz with picks [L, N, K] for --routing trace")
    p.add_argument("--weights", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--load-format", default="", help="empty: dummy for synthetic, auto for real")
    p.add_argument("--kv-cache-dtype", default="fp8")
    p.add_argument("--quantization", default="fp8")
    p.add_argument(
        "--rpa-native-fp8-prefill", default="pvs",
        help="TPU_RPA_NATIVE_FP8_PREFILL operand set for the prefill attention kernel (pvs: the "
        "phase 2 fp8 PV path with a static scale, the 40.3 to 41.3k rows/s record; empty: bf16 "
        "matmuls). The shared serving jsonnet has no passthrough for it yet (REPORT.md).",
    )
    p.add_argument(
        "--env", action="append", default=[],
        help="KEY=VALUE set after P_LEG_ENV; KEY= with no value unsets KEY",
    )
    p.add_argument("--model-dir", default=common.MODEL_CONFIG_DIR)
    p.add_argument("--out-root", default="", help="run directory root (default P3_OUT_ROOT)")
    p.add_argument("--out-dir", default="", help="exact run directory (overrides --out-root)")
    p.add_argument("--tag", default="")
    p.add_argument("--cache-root", default="/workspace/tmp/p3-harness/compile-cache/p-leg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print the env and engine args only")
    a = p.parse_args(argv)
    if not a.case and not a.prompt_file and not a.shapes_from:
        a.case = [Case(87040, 10240, 32)]
    if not a.mode:
        a.mode = ["warm"]
    if a.routing == "trace" and not a.routing_file:
        p.error("--routing trace needs --routing-file")
    return a


def layout_tag(a: argparse.Namespace) -> str:
    parts = [f"tp{a.tensor_parallel_size}"]
    if a.pipeline_parallel_size > 1:
        parts.append(f"pp{a.pipeline_parallel_size}")
    if a.prefill_context_parallel_size > 1:
        parts.append(f"pcp{a.prefill_context_parallel_size}")
    if a.data_parallel_size > 1:
        parts.append(f"dp{a.data_parallel_size}")
    parts.append("ep" if a.expert_parallel else "noep")
    parts.append(f"c{a.max_num_batched_tokens}")
    parts.append(a.routing)
    return "-".join(parts)


def build_env(a: argparse.Namespace, out_dir: str) -> tuple[dict[str, str], list[str]]:
    """Returns the variables to set and the ones to remove."""
    env = dict(P_LEG_ENV)
    env.update(common.topology_env(a.num_hosts))
    unset: list[str] = []
    pp = a.pipeline_parallel_size > 1
    if pp:
        # TP1 per stage: no TP collectives, so SP and the fused EP kernel have
        # nothing to shard, and the tp8_ep8 placement table does not apply.
        # Hybrid-state retention is untested under PP; dynamic GDN tiling is
        # what the PP8 leg needed (agentx-pp-p).
        env.update(
            TPU_MOE_SEQUENCE_PARALLEL="0",
            TPU_SP_FP8_GATHER="0",
            USE_MOE_FUSED_EP_KERNEL="0",
            TPU_HYBRID_STATE_RETENTION="0",
            TPU_ENABLE_GDN_DYNAMIC_TILING="1",
        )
        env.pop("TPU_EXPERT_PLACEMENT", None)
        unset.append("TPU_EXPERT_PLACEMENT")
    if a.tensor_parallel_size != 8 or not a.expert_parallel:
        # The placement table is tp8_ep8 specific.
        env.pop("TPU_EXPERT_PLACEMENT", None)
        unset.append("TPU_EXPERT_PLACEMENT")
    if a.distributed_executor_backend == "ray":
        env["TPU_MULTIHOST_BACKEND"] = "ray"
    if a.rpa_native_fp8_prefill:
        env["TPU_RPA_NATIVE_FP8_PREFILL"] = a.rpa_native_fp8_prefill
    else:
        env.pop("TPU_RPA_NATIVE_FP8_PREFILL", None)
        unset.append("TPU_RPA_NATIVE_FP8_PREFILL")
    env, more = common.parse_env_overrides(a.env, env)
    unset += more
    src = common.source_identity()
    key = common.cache_key([
        *sorted(a.env),
        "routing=" + a.routing,
        "routing_file=" + (os.path.basename(a.routing_file) if a.routing_file else ""),
        "source=" + src["source_sha"] + "+" + src["source_dirty_sha"],
        "layout=" + layout_tag(a),
        f"quant={a.quantization}/{a.kv_cache_dtype}/rpa={a.rpa_native_fp8_prefill}",
        f"weights={a.weights}",
        f"mtp={a.mtp}",
        f"block={a.block_size}",
        f"compile={a.compile_sizes}",
        common.SITECUSTOMIZE,
    ])
    for k, v in common.cache_dirs(a.cache_root, key).items():
        env.setdefault(k, v)
    env["P3_STEP_LOG"] = os.path.join(out_dir, "steps.jsonl")
    env["P3_LOG_PREFIX"] = "p3_p_leg"
    env["P3_ROUTING"] = a.routing
    if a.routing_file:
        env["P3_ROUTING_FILE"] = os.path.abspath(a.routing_file)
    env["P3_WEIGHTS"] = a.weights
    env["P3_COMPILE_SIZES"] = a.compile_sizes
    site_dir = common.install_sitecustomize(out_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        x for x in (site_dir, os.environ.get("PYTHONPATH", "")) if x
    )
    env["P3_CACHE_KEY"] = key
    return env, unset


def engine_argv(a: argparse.Namespace, profile_dir: str) -> list[str]:
    compile_sizes = [int(x) for x in a.compile_sizes.split(",") if x]
    load_format = a.load_format or ("dummy" if a.weights == "synthetic" else "auto")
    argv = [
        "--model", a.model_dir,
        "--load-format", load_format,
        "--trust-remote-code",
        "--seed", "42",
        "--max-model-len", str(a.max_model_len),
        "--disable-custom-all-reduce",
        "--gpu-memory-utilization", str(a.gpu_memory_utilization),
        "--kv-cache-dtype", a.kv_cache_dtype,
        "--quantization", a.quantization,
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
        "--prefill-context-parallel-size", str(a.prefill_context_parallel_size),
        "--cp-kv-cache-interleave-size", str(a.cp_kv_cache_interleave_size),
        "--data-parallel-size", str(a.data_parallel_size),
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", profile_dir,
        "--profiler-config.ignore_frontend", "true",
    ]  # fmt: skip
    pp = a.pipeline_parallel_size > 1
    if a.expert_parallel:
        argv.append("--enable-expert-parallel")
    if pp or a.no_async_scheduling:
        argv.append("--no-async-scheduling")
    else:
        argv.append("--async-scheduling")
    if a.mtp > 0 and not pp:
        # pp_validation admits only producer MTP K=1 with a kv_transfer_config,
        # which this offline engine has not, so PP runs without a draft.
        argv += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": a.mtp})]
    if a.scheduler_cls:
        argv += ["--scheduler-cls", a.scheduler_cls]
    if a.distributed_executor_backend:
        argv += ["--distributed-executor-backend", a.distributed_executor_backend]
    return argv


def _step_stats(steps: list[dict], t0: float, t1: float) -> dict:
    """Summarizes the step records that landed inside one round.

    With Mamba align prefix caching a chunk ends on a block boundary, so a
    full step is the largest block multiple under the token budget (7680 rows
    for 8192 and block 640), not the budget itself. Async scheduling also
    logs empty steps; they are skipped, so an interval spans one step that
    computed rows plus whatever host time the pipeline did not hide.
    """
    win = [s for s in steps if t0 <= s["t"] <= t1 + 0.5]
    work = [s for s in win if s["tok"] > 0]
    out = {
        "steps": len(work),
        "empty_steps": len(win) - len(work),
        "computed_rows_steps": sum(s["tok"] for s in work),
        "padded_rows_steps": sum(s.get("padded", s["tok"]) for s in work),
        "step_tokens": [s["tok"] for s in work],
        "step_reqs": [s["reqs"] for s in work],
    }
    if len(work) < 3:
        return out
    full = max(s["tok"] for s in work)
    iv = [(b["t"] - a["t"]) * 1e3 for a, b in zip(work, work[1:])]
    iv_full = [
        (b["t"] - a["t"]) * 1e3
        for a, b in zip(work, work[1:])
        if a["tok"] == full and b["tok"] == full
    ]
    span = work[-1]["t"] - work[0]["t"]
    out.update(
        full_step_rows=full,
        step_ms_median=round(statistics.median(iv), 2),
        # Rows after the first step over the time from its end to the last
        # step's end: the rate with the round's ramp-up left out.
        rows_per_s_steady=round(sum(s["tok"] for s in work[1:]) / span) if span > 0 else None,
    )
    if len(iv_full) >= 2:
        q = statistics.quantiles(iv_full, n=10)
        out.update(
            full_step_ms_median=round(statistics.median(iv_full), 2),
            full_step_ms_p10=round(q[0], 2),
            full_step_ms_p90=round(q[-1], 2),
            full_step_intervals=len(iv_full),
            rows_per_s_full_steps=round(full / (statistics.median(iv_full) / 1e3)),
        )
    return out


def _load_prompt_file(path: str) -> list[tuple[list[int], list[int]]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append((list(rec["prefix"]), list(rec["new"])))
    if not out:
        raise SystemExit(f"{path}: no records")
    return out


def run(a: argparse.Namespace, out_dir: str, timeline: common.Timeline) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.config import FaultToleranceConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    profile_dir = os.path.join(out_dir, "xprof")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(engine_argv(a, profile_dir))
    # The parser's default for --fault-tolerance-config is a dict, and
    # EngineArgs.__post_init__ turns any dict there into
    # enable_fault_tolerance=True, which a single engine rejects.
    if isinstance(ns.fault_tolerance_config, dict):
        ns.fault_tolerance_config = FaultToleranceConfig()
    engine_args = EngineArgs.from_cli_args(ns)
    step_log = os.environ["P3_STEP_LOG"]
    # A retried pod reuses the run directory; start the log afresh.
    open(step_log, "w").close()
    timeline.mark("engine_init_start")
    llm = LLM.from_engine_args(engine_args)
    timeline.mark("engine_ready")
    startup_s = timeline.marks["engine_ready"] - timeline.marks["engine_init_start"]
    print(f"[p3_p_leg] engine ready in {startup_s:.0f} s", flush=True)
    rng = np.random.default_rng(a.seed)
    sp_probe = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=1, detokenize=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False)

    def tokens(n: int) -> list[int]:
        return rng.integers(1000, 200000, n).tolist()

    if a.prompt_file:
        records = _load_prompt_file(a.prompt_file)
        file_cases = [("file", records)]
    else:
        file_cases = []
    shape_cases = []
    if a.shapes_from:
        shapes = common.load_shapes(a.shapes_from, a.reqs, a.shapes_seed, a.shapes_select, a.shapes_subset)
        # Random token content at the real lengths: prefix = context minus the
        # uncached suffix (what the cache would hold), new = the uncached suffix.
        shape_cases.append(("shapes", [(tokens(s["ctx"] - s["new"]), tokens(s["new"])) for s in shapes], shapes))

    cases_out = []
    timeline.mark("timed_start")
    jobs: list[tuple[str, Case | None, list, list]] = []
    for mode in a.mode:
        for case in a.case or []:
            jobs.append((mode, case, [], []))
        for _, records in file_cases:
            jobs.append((mode, None, records, []))
        for _, records, shapes in shape_cases:
            jobs.append((mode, None, records, shapes))
    for ci, (mode, case, records, shapes) in enumerate(jobs):
        if case is not None:
            prefixes = [tokens(case.ctx) for _ in range(case.reqs)]
            news = [tokens(case.new) for _ in range(case.reqs)]
            label = {"ctx": case.ctx, "new": case.new, "reqs": case.reqs, "source": "random"}
        else:
            n = a.reqs if not shapes else len(records)
            prefixes = [records[i % len(records)][0] for i in range(n)]
            news = [records[i % len(records)][1] for i in range(n)]
            label = {
                "ctx": round(statistics.mean(len(p) + len(x) for p, x in zip(prefixes, news))),
                "new": round(statistics.mean(len(x) for x in news)),
                "reqs": n,
                "source": os.path.basename(a.shapes_from if shapes else a.prompt_file),
                "ctx_min": min(len(p) + len(x) for p, x in zip(prefixes, news)),
                "ctx_max": max(len(p) + len(x) for p, x in zip(prefixes, news)),
                "new_min": min(len(x) for x in news), "new_max": max(len(x) for x in news),
                "cold_requests": sum(1 for p in prefixes if len(p) == 0),
            }
            if shapes:
                label["shapes"] = shapes
        case_out = {**label, "mode": mode, "rounds": []}
        if mode == "warm":
            t0 = time.time()
            # Fully cold requests (empty prefix) have nothing to warm.
            warm_prefixes = [p for p in prefixes if len(p) > 0]
            outs = llm.generate([{"prompt_token_ids": p} for p in warm_prefixes], sp_probe,
                                use_tqdm=False) if warm_prefixes else []
            case_out["prefix_build_s"] = round(time.time() - t0, 2)
            case_out["prefix_build_cached"] = [o.num_cached_tokens or 0 for o in outs]
            lps = [next(iter(o.outputs[0].logprobs[0].values())).logprob
                   for o in outs if o.outputs[0].logprobs]
            case_out["first_token_logprob_finite"] = bool(lps) and all(np.isfinite(lps))
        else:
            case_out["first_token_logprob_finite"] = None
        n_rounds = a.rounds + (1 if a.profile else 0)
        for r in range(n_rounds):
            profiled = a.profile and r == n_rounds - 1
            if mode == "warm" and case is not None:
                prompts = [{"prompt_token_ids": p + tokens(len(nw))} for p, nw in zip(prefixes, news)]
            elif mode == "warm":
                # Frozen-input suffixes: round 0 replays them as recorded; later
                # rounds change the suffix's first token, which under align
                # prefix caching invalidates every suffix block while the
                # prefix stays cached, so the suffix is computed again.
                prompts = [{"prompt_token_ids": p + (nw if r == 0 else tokens(1) + nw[1:])}
                           for p, nw in zip(prefixes, news)]
            else:
                # Cold: fresh tokens every round so no prefix is in the cache.
                prompts = [{"prompt_token_ids": tokens(len(p) + len(nw))}
                           for p, nw in zip(prefixes, news)]
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
                "t0": t0, "t1": t1,
                "wall_s": round(t1 - t0, 3),
                "computed_rows": rows,
                "prompt_rows": sum(len(p["prompt_token_ids"]) for p in prompts),
                "rows_per_s_wall": round(rows / (t1 - t0)),
                "cached_min": min(cached),
                "cached_max": max(cached),
            }
            rd.update(_step_stats(common.step_records(common.read_jsonl(step_log)), t0, t1))
            case_out["rounds"].append(rd)
            print(f"[p3_p_leg] case {ci} {mode} round {r}: {json.dumps(rd)}", flush=True)
        plain = [x for x in case_out["rounds"] if not x["profiled"]]
        if plain:
            for key in ("rows_per_s_wall", "rows_per_s_steady", "rows_per_s_full_steps",
                        "full_step_ms_median", "step_ms_median"):
                vals = [x[key] for x in plain if x.get(key) is not None]
                if vals:
                    case_out[key + "_median"] = round(statistics.median(vals), 2)
        if mode == "warm":
            # A measured request must start from the cached prefix (attention
            # and GDN state); one block of slack covers the align-mode boundary.
            lo = min(len(p) for p in prefixes)
            case_out["cached_ok"] = all(x["cached_min"] >= lo - a.block_size for x in case_out["rounds"])
        else:
            case_out["cached_ok"] = all(x["cached_max"] <= a.block_size for x in case_out["rounds"])
        cases_out.append(case_out)
    timeline.mark("timed_end")
    return {"startup_s": round(startup_s, 1), "cases": cases_out}


def _chunk_grid(a: argparse.Namespace, argv: list[str]) -> int:
    """Runs one fresh process per chunk size; the TPU is released in between."""
    sizes = [int(x) for x in a.chunk_grid.split(",") if x]
    base = [x for x in argv]
    # Drop the grid flag and any explicit chunk size from the child argv.
    out_argv = []
    skip = False
    for tok in base:
        if skip:
            skip = False
            continue
        if tok in ("--chunk-grid", "--max-num-batched-tokens", "--tag"):
            skip = True
            continue
        if tok.startswith(("--chunk-grid=", "--max-num-batched-tokens=", "--tag=")):
            continue
        out_argv.append(tok)
    results = []
    rc_all = 0
    for size in sizes:
        tag = f"{a.tag + '-' if a.tag else ''}chunk{size}"
        cmd = [sys.executable, os.path.abspath(__file__), *out_argv,
               "--max-num-batched-tokens", str(size), "--tag", tag]
        print("[p3_p_leg] grid:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, check=False, env={**os.environ, "P3_CHILD": "1"})
        rc_all |= proc.returncode
        results.append({"chunk": size, "rc": proc.returncode, "tag": tag})
    grid_dir = common.run_dir("p_leg_grid", a.tag, a.out_root or None)
    common.write_json(os.path.join(grid_dir, "grid.json"), {"sizes": sizes, "runs": results})
    print("P3_P_LEG_GRID " + json.dumps({"sizes": sizes, "runs": results, "dir": grid_dir}), flush=True)
    return rc_all


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    if a.chunk_grid and not os.environ.get("P3_CHILD"):
        return _chunk_grid(a, argv)
    timeline = common.Timeline()
    out_dir = a.out_dir or common.run_dir("p_leg", "-".join(x for x in (a.tag, layout_tag(a)) if x),
                                          a.out_root or None)
    os.makedirs(out_dir, exist_ok=True)
    env, unset = build_env(a, out_dir)
    argv_engine = engine_argv(a, os.path.join(out_dir, "xprof"))
    if a.dry_run:
        print(json.dumps({"env": env, "unset": unset, "engine_argv": argv_engine,
                          "out_dir": out_dir}, indent=1))
        return 0
    os.environ.update(env)
    for key in unset:
        os.environ.pop(key, None)
    env_id = common.write_env(out_dir, {"bench": "p_leg", "cache_key": env["P3_CACHE_KEY"]})
    config = {
        "argv": argv,
        "cases": [dataclasses.asdict(c) for c in (a.case or [])],
        "prompt_file": a.prompt_file,
        "modes": a.mode,
        "layout": {
            "tp": a.tensor_parallel_size, "pp": a.pipeline_parallel_size,
            "pcp": a.prefill_context_parallel_size, "dp": a.data_parallel_size,
            "ep": bool(a.expert_parallel), "num_hosts": a.num_hosts,
            "ranks": a.tensor_parallel_size * a.pipeline_parallel_size
            * a.prefill_context_parallel_size * a.data_parallel_size,
        },
        "routing": a.routing,
        "routing_file": a.routing_file,
        "weights": a.weights,
        "max_num_batched_tokens": a.max_num_batched_tokens,
        "compile_sizes": a.compile_sizes,
        "block_size": a.block_size,
        "mtp": a.mtp,
        "env_overrides": a.env,
        "engine_argv": argv_engine,
        "stack": common.stack_tag(),
        "cache_key": env["P3_CACHE_KEY"],
        "out_dir": out_dir,
    }
    result = {"bench": "p_leg", "environment_id": env_id["id"], "config": config}
    try:
        result.update(run(a, out_dir, timeline))
        result["ok"] = all(
            c["cached_ok"] and c["first_token_logprob_finite"] is not False
            for c in result["cases"]
        )
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["ok"] = False
        raise
    finally:
        records = common.read_jsonl(os.environ["P3_STEP_LOG"])
        timeline.absorb_phases(records)
        result["windows"] = timeline.windows()
        result["phases"] = common.summarize_phases(records)
        common.write_json(os.path.join(out_dir, "result.json"), result)
        print("P3_P_LEG_RESULT " + json.dumps(result, default=str), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
