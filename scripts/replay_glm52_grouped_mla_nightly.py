# SPDX-License-Identifier: Apache-2.0
"""Run the GLM-5.2 8K/1K scheduler workload through one MLA layer on TPU.

This is an attention-only serving replay, not a full-model benchmark. It uses
vLLM's real ``AsyncScheduler`` to produce one representative data-parallel
rank's steps for the exact GLM-5.2 nightly workload, then executes every step
through the production-equivalent cache-update plus MLA wrapper from
``benchmark_masked_dense_vs_sparse_mla.py``.

The feature arm applies the runner's policy-free
``[decode][long][medium][short]`` ordering. The MLA kernel then derives and
validates its own split and cost tier from existing device tensors. The
control arm runs the identical scheduler trace and tensors through an
independently compiled sparse-only wrapper.
Compilation, input construction, and scheduler time are reported separately
and excluded from the paired TPU timings.

Run the scheduler-only check on CPU:

    JAX_PLATFORMS=cpu .venv/bin/python \
      scripts/replay_glm52_grouped_mla_nightly.py --schedule-only

After acquiring one chip according to the local TPU allocation policy, run:

    TPU_MLA_MASKED_DENSE_ENABLED=1 \
      python scripts/replay_glm52_grouped_mla_nightly.py \
      --warmup 5 --iters 20 --out /tmp/glm52_grouped_scheduler_replay.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from vllm import SamplingParams
from vllm.config import SchedulerConfig
from vllm.v1 import kv_cache_interface
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 1024
MAX_MODEL_LEN = PROMPT_TOKENS + OUTPUT_TOKENS
MAX_NUM_BATCHED_TOKENS = 1024
MAX_NUM_SEQS = 64
DATA_PARALLEL_SIZE = 16
FLEET_CONCURRENCY = 1024
FLEET_REQUESTS = 2048
PAGE_SIZE = 1024
PAGES_PER_SEQ = MAX_MODEL_LEN // PAGE_SIZE
TOKEN_BUCKETS = (16, 32, 64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class Step:
    index: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return sum(self.q_lens)

    @property
    def num_decodes(self) -> int:
        return sum(q_len == 1 for q_len in self.q_lens)

    @property
    def num_prefills(self) -> int:
        return len(self.q_lens) - self.num_decodes


@dataclass(frozen=True)
class ReplayStep:
    index: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    token_bucket: int
    grouped: bool
    tier: str

    @property
    def num_tokens(self) -> int:
        return sum(self.q_lens)

    @property
    def num_decodes(self) -> int:
        return sum(q_len == 1 for q_len in self.q_lens)

    @property
    def num_prefills(self) -> int:
        return len(self.q_lens) - self.num_decodes


class _NoMultimodalRegistry:

    @staticmethod
    def supports_multimodal_inputs(_model_config) -> bool:
        return False


class _NoStructuredOutput:

    @staticmethod
    def should_advance(*_args, **_kwargs) -> bool:
        return False


def _make_scheduler() -> AsyncScheduler:
    """Construct the installed vLLM scheduler without loading model weights."""
    scheduler_config = SchedulerConfig(
        max_model_len=MAX_MODEL_LEN,
        is_encoder_decoder=False,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
        async_scheduling=True,
    )
    cache_config = SimpleNamespace(
        num_gpu_blocks=1024,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    parallel_config = SimpleNamespace(
        data_parallel_index=0,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        is_diffusion=False,
        max_model_len=MAX_MODEL_LEN,
        enable_return_routed_experts=False,
    )
    observability_config = SimpleNamespace(
        kv_cache_metrics=False,
        kv_cache_metrics_sample=0,
        enable_mfu_metrics=False,
        enable_logging_iteration_details=False,
    )
    ec_manager_config = SimpleNamespace(
        get_encoder_cache_manager_obj=lambda: None, )
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=None,
        kv_events_config=None,
        parallel_config=parallel_config,
        observability_config=observability_config,
        model_config=model_config,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        max_concurrent_batches=2,
        max_in_flight_tokens=2 * MAX_NUM_BATCHED_TOKENS,
        use_v2_model_runner=False,
        ec_manager_config=ec_manager_config,
    )
    cache_spec = kv_cache_interface.FullAttentionSpec(
        block_size=PAGE_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    kv_cache_config = kv_cache_interface.KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[
            kv_cache_interface.KVCacheGroupSpec(["attention"], cache_spec)
        ],
    )
    return AsyncScheduler(
        vllm_config,
        kv_cache_config,
        _NoStructuredOutput(),
        block_size=PAGE_SIZE,
        mm_registry=_NoMultimodalRegistry(),
    )


def _request(index: int) -> Request:
    return Request(
        request_id=f"request-{index}",
        prompt_token_ids=[0] * PROMPT_TOKENS,
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
        ),
        pooling_params=None,
        arrival_time=float(index),
    )


def simulate() -> list[Step]:
    """Replay one rank, including the engine's two-entry asynchronous queue."""
    scheduler = _make_scheduler()
    requests_per_rank = FLEET_REQUESTS // DATA_PARALLEL_SIZE
    concurrency_per_rank = FLEET_CONCURRENCY // DATA_PARALLEL_SIZE
    next_request = 0
    for _ in range(concurrency_per_rank):
        scheduler.add_request(_request(next_request))
        next_request += 1

    steps: list[Step] = []
    pending_outputs = deque()
    completed = 0
    while completed < requests_per_rank or pending_outputs:
        if len(steps) > 10_000:
            raise RuntimeError(
                "simulation exceeded 10,000 steps: "
                f"{completed=}, {next_request=}, pending={len(pending_outputs)}, "
                f"running={len(scheduler.running)}, waiting={len(scheduler.waiting)}"
            )
        scheduled_step = False
        if completed < requests_per_rank and len(pending_outputs) < 2:
            output = scheduler.schedule()
            if output.num_scheduled_tokens:
                scheduled_step = True
                pairs = [(
                    q_len,
                    scheduler.requests[req_id].num_computed_tokens,
                ) for req_id, q_len in output.num_scheduled_tokens.items()]
                # This is the runner's first partition. The grouped feature's
                # long/short partition is applied later only when its host gate
                # accepts the complete step.
                pairs.sort(key=lambda pair: pair[0] != 1)
                steps.append(
                    Step(
                        index=len(steps),
                        q_lens=tuple(pair[0] for pair in pairs),
                        kv_lens=tuple(pair[1] for pair in pairs),
                    ))
                req_ids = list(output.num_scheduled_tokens)
                sampled = tuple(not scheduler.requests[req_id].is_prefill_chunk
                                for req_id in req_ids)
                pending_outputs.append((output, req_ids, sampled))
            elif not pending_outputs:
                raise RuntimeError(
                    "scheduler produced an empty step with no result to drain")

        if (scheduled_step and len(pending_outputs) < 2
                and completed < requests_per_rank):
            continue

        output, req_ids, will_sample = pending_outputs.popleft()
        sampled = [[1] if sample else [] for sample in will_sample]
        engine_outputs = scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={
                    req_id: i
                    for i, req_id in enumerate(req_ids)
                },
                sampled_token_ids=sampled,
            ))
        newly_completed = sum(item.finish_reason is not None
                              for group in engine_outputs.values()
                              for item in group.outputs)
        completed += newly_completed
        for _ in range(newly_completed):
            if next_request < requests_per_rank:
                scheduler.add_request(_request(next_request))
                next_request += 1
    return steps


def _token_bucket(num_tokens: int) -> int:
    for bucket in TOKEN_BUCKETS:
        if num_tokens <= bucket:
            return bucket
    raise ValueError(
        f"step has {num_tokens} tokens, above the 1,024-token cap")


def _selected_tier(seq_lens, q_lens, mla_dispatch) -> str:
    """Run the production device selector for result accounting."""
    import jax.numpy as jnp

    starts = np.zeros(len(seq_lens) + 1, np.int32)
    starts[1:] = np.cumsum(q_lens, dtype=np.int32)
    num_decode = sum(q_len == 1 for q_len in q_lens)
    tier = int(
        mla_dispatch.masked_dense_prefill_mla_tier(
            jnp.asarray(seq_lens, jnp.int32),
            jnp.asarray(
                [num_decode, num_decode, len(seq_lens)], jnp.int32),
            jnp.asarray(starts), (mla_dispatch.MASKED_DENSE_ANALYTIC_LIMIT,
                                  mla_dispatch.MASKED_DENSE_LIMIT)))
    return ("analytic", "bitmap", "sparse")[tier]


def prepare_replay(steps: list[Step]):
    """Apply PR #641's policy-free request ordering."""
    from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch

    bounds = mla_dispatch.grouped_partition_limits()
    replay = []
    for step in steps:
        pairs = list(zip(step.q_lens, step.kv_lens))

        def category(pair):
            q_len, kv_len = pair
            if q_len == 1:
                return 0
            return 2 + sum(kv_len <= bound for bound in bounds)

        # Stable sorting is attention-equivalent to the runner's sequence of
        # threshold partitions.
        pairs.sort(key=category)
        q_lens = tuple(pair[0] for pair in pairs)
        kv_lens = tuple(pair[1] for pair in pairs)
        tier = _selected_tier(kv_lens, q_lens, mla_dispatch)
        replay.append(
            ReplayStep(
                index=step.index,
                q_lens=q_lens,
                kv_lens=kv_lens,
                token_bucket=_token_bucket(sum(q_lens)),
                grouped=bool(bounds),
                tier=tier,
            ))
    return replay


def summarize(replay: list[ReplayStep]) -> dict:
    kinds = Counter("decode-only" if step.num_prefills ==
                    0 else "prefill-only" if step.num_decodes == 0 else "mixed"
                    for step in replay)
    return {
        "config": {
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "fleet_requests": FLEET_REQUESTS,
            "fleet_concurrency": FLEET_CONCURRENCY,
            "data_parallel_size": DATA_PARALLEL_SIZE,
            "requests_per_rank": FLEET_REQUESTS // DATA_PARALLEL_SIZE,
            "concurrency_per_rank": FLEET_CONCURRENCY // DATA_PARALLEL_SIZE,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
        },
        "steps":
        len(replay),
        "step_kinds":
        dict(kinds),
        "token_buckets":
        dict(sorted(Counter(step.token_bucket for step in replay).items())),
        "feature_routes":
        dict(Counter(step.tier for step in replay)),
        "grouped_steps":
        sum(step.grouped for step in replay),
    }


def _causal_topk(positions: np.ndarray, token_bucket: int, topk: int):
    """Deterministic causal selection with uniform long-context dispersion."""
    padded = np.full(token_bucket, -1, np.int64)
    padded[:len(positions)] = positions
    span = padded[:, None] + 1
    columns = np.arange(topk, dtype=np.int64)[None, :]
    short = np.where(columns < span, columns, -1)
    spread = columns * span // topk
    return np.where(span >= topk, spread, short).astype(np.int32)


def _positions(step: ReplayStep) -> np.ndarray:
    positions = []
    for q_len, kv_len in zip(step.q_lens, step.kv_lens):
        positions.extend(range(kv_len - q_len, kv_len))
    return np.asarray(positions, np.int64)


def _stage_tpu_args(replay, bench):
    """Stage scheduler-derived metadata and causal top-k tensors on one chip."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(20260827)
    total_pages = MAX_NUM_SEQS * PAGES_PER_SEQ
    nope, rope = bench._build_caches(rng, total_pages)
    page_indices = jnp.asarray(rng.permutation(total_pages).astype(np.int32))
    q_by_bucket = {}
    for bucket in sorted({step.token_bucket for step in replay}):
        q = jnp.asarray(
            rng.standard_normal(
                (bucket, bench.NUM_HEADS, bench.LKV_DIM + bench.ROPE_DIM),
                dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        ql_nope = q[..., :bench.LKV_DIM]
        q_pe = q[..., bench.LKV_DIM:]
        q_by_bucket[bucket] = (
            ql_nope,
            q_pe,
            ql_nope[:, 0, :].astype(jnp.float8_e4m3fn),
            q_pe[:, 0, :].astype(jnp.float8_e4m3fn),
        )
    jax.block_until_ready((nope, rope, page_indices, q_by_bucket))

    staged = []
    for offset, step in enumerate(replay, 1):
        num_seqs = len(step.q_lens)
        starts = np.full(MAX_NUM_SEQS + 1, step.num_tokens, np.int32)
        starts[0] = 0
        starts[1:num_seqs + 1] = np.cumsum(step.q_lens, dtype=np.int32)
        seq_lens = np.zeros(MAX_NUM_SEQS, np.int32)
        seq_lens[:num_seqs] = step.kv_lens
        topk = jnp.asarray(
            _causal_topk(_positions(step), step.token_bucket, bench.TOPK))
        seq_lens_device = jnp.asarray(seq_lens)
        starts_device = jnp.asarray(starts)
        distribution = jnp.asarray(
            [step.num_decodes, step.num_decodes, num_seqs], jnp.int32)
        # Bound host memory and make input transfer an explicit untimed phase.
        jax.block_until_ready(
            (topk, seq_lens_device, starts_device, distribution))
        q_args = q_by_bucket[step.token_bucket]
        staged.append(q_args + (
            nope,
            rope,
            topk,
            seq_lens_device,
            page_indices,
            starts_device,
            distribution,
        ))
        if offset % 250 == 0 or offset == len(replay):
            print(f"staged {offset}/{len(replay)} scheduler steps", flush=True)
    return staged


def _time_tpu_replay(replay, staged, bench, warmup: int, iters: int):
    import jax

    grouped = bench._build_wrapper(True)
    sparse_only = bench._build_wrapper(False)

    # Compile every production token bucket once for each arm.
    sparse_examples = {}
    grouped_examples = {}
    for step, args in zip(replay, staged):
        sparse_examples.setdefault(step.token_bucket, args)
        if step.grouped:
            grouped_examples.setdefault(step.token_bucket, args)
    for bucket, args in sorted(sparse_examples.items()):
        jax.block_until_ready(sparse_only(*args))
        print(f"compiled sparse-only bucket {bucket}", flush=True)
    if not grouped_examples:
        raise RuntimeError("scheduler workload contains no grouped MLA step")
    for bucket, args in sorted(grouped_examples.items()):
        jax.block_until_ready(grouped(*args))
        print(f"compiled grouped bucket {bucket}", flush=True)

    sparse_fns = [sparse_only] * len(replay)
    feature_fns = [grouped if step.grouped else sparse_only for step in replay]

    def run_trace(fns):
        start = time.perf_counter()
        for fn, args in zip(fns, staged):
            # A serving model step cannot advance the scheduler until device
            # execution completes. Synchronizing every attention call preserves
            # that dependency and includes the same host launch boundary in
            # both arms.
            jax.block_until_ready(fn(*args))
        return (time.perf_counter() - start) * 1e3

    for iteration in range(warmup):
        run_trace(feature_fns)
        run_trace(sparse_fns)
        print(f"warmup {iteration + 1}/{warmup}", flush=True)

    samples = {"grouped_ms": [], "sparse_only_ms": []}
    for iteration in range(iters):
        arms = ((feature_fns, samples["grouped_ms"]),
                (sparse_fns, samples["sparse_only_ms"]))
        if iteration % 2:
            arms = arms[::-1]
        for fns, values in arms:
            values.append(run_trace(fns))
        print(
            f"sample {iteration + 1}/{iters}: "
            f"grouped={samples['grouped_ms'][-1]:.3f} ms "
            f"sparse={samples['sparse_only_ms'][-1]:.3f} ms",
            flush=True)

    grouped_median = statistics.median(samples["grouped_ms"])
    sparse_median = statistics.median(samples["sparse_only_ms"])
    return {
        "warmup_traces": warmup,
        "measured_traces": iters,
        **samples,
        "grouped_median_ms": grouped_median,
        "sparse_only_median_ms": sparse_median,
        "ratio": grouped_median / sparse_median,
        "reduction_fraction": 1.0 - grouped_median / sparse_median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps-out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error(
            "--warmup must be nonnegative and --iters must be positive")

    schedule_start = time.perf_counter()
    scheduled = simulate()
    replay = prepare_replay(scheduled)
    schedule_ms = (time.perf_counter() - schedule_start) * 1e3
    result = {
        "summary": summarize(replay),
        "scheduler_cpu_ms": schedule_ms,
        "tpu_replay": None,
    }
    print(json.dumps(result, indent=2), flush=True)

    if not args.schedule_only:
        import benchmark_masked_dense_vs_sparse_mla as bench
        import jax

        device = str(jax.devices()[0])
        result["measurement"] = {
            "device":
            device,
            "scope":
            "cache update plus one MLA layer per scheduled step",
            "synchronization":
            "after every scheduled attention call",
            "topk_inputs":
            "deterministic causal, uniformly dispersed",
            "excluded": [
                "compilation", "input staging", "scheduler CPU time",
                "DSA indexer", "remaining model"
            ],
        }
        print(f"device: {device}", flush=True)
        staged = _stage_tpu_args(replay, bench)
        result["tpu_replay"] = _time_tpu_replay(replay, staged, bench,
                                                args.warmup, args.iters)
        print(json.dumps(result["tpu_replay"], indent=2), flush=True)

    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
    if args.steps_out:
        args.steps_out.write_text(
            json.dumps([asdict(step) for step in replay], indent=2) + "\n")
        print(f"wrote {args.steps_out}", flush=True)


if __name__ == "__main__":
    main()
