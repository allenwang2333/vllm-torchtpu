# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""The owner-sharded KV DSpark 100k decode config fits KV_SHARD's buckets."""

import json
import pathlib
import shlex
import subprocess

CONFIG = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "configs"
    / "deepseek-v41-flash-tp8-ep-100k-decode-dspark-kvshard.sh"
)
KNOBS = (
    "TPU_DEEPSEEK_V41_KV_SHARD",
    "TPU_DEEPSEEK_V41_IMPL",
    "TPU_DEEPSEEK_V41_FUSED_SWA",
    "ENABLE_PREFIX_CACHING",
    "TENSOR_PARALLELISM",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "CONCURRENCY_OPTIONS",
    "COMPILE_SIZES",
    "KV_CACHE_MEMORY_BYTES",
    "DSPARK_K",
    "EXTRA_SERVE_ARGS",
)
# Everything the config chain reads, cleared so the calling shell cannot leak in.
INPUTS = (
    *KNOBS,
    "OFFLOAD",
    "HOST_DMA",
    "MODEL_URI",
    "DEEPSEEK_V41_MODEL",
    "KV_CACHE_DTYPE",
    "TPU_DEEPSEEK_V41_CED",
    "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD",
    "TPU_DEEPSEEK_V41_ENGRAM_HOST_DMA",
)


def _source(**overrides: str) -> dict[str, str]:
    lines = [f"unset {' '.join(INPUTS)}"]
    lines += [
        f"export {name}={shlex.quote(value)}" for name, value in overrides.items()
    ]
    lines.append(f"source {shlex.quote(str(CONFIG))}")
    lines += [f'printf "%s=%s\\n" {name} "${{{name}}}"' for name in KNOBS]
    result = subprocess.run(
        ["bash", "-c", "\n".join(lines)], capture_output=True, text=True, check=True
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_defaults_shard_kv_for_c224_on_the_dspark_workload():
    cfg = _source()

    assert cfg["TPU_DEEPSEEK_V41_KV_SHARD"] == "1"
    assert cfg["TPU_DEEPSEEK_V41_IMPL"] == "v2"
    assert cfg["TPU_DEEPSEEK_V41_FUSED_SWA"] == "1"
    assert cfg["MAX_NUM_SEQS"] == "224"
    assert cfg["CONCURRENCY_OPTIONS"] == "96 128 192 224"
    assert cfg["KV_CACHE_MEMORY_BYTES"] == str(20 * 2**30)
    # validate_config rejects prefix caching under KV_SHARD.
    assert cfg["ENABLE_PREFIX_CACHING"] == "false"
    args = cfg["EXTRA_SERVE_ARGS"]
    assert f'--compilation-config={{"compile_sizes":{cfg["COMPILE_SIZES"]}}}' in args
    assert '--speculative-config={"method":"dspark","num_speculative_tokens":5}' in args


def test_every_bucket_divides_by_tp_and_each_sweep_point_has_its_own():
    cfg = _source()
    tp = int(cfg["TENSOR_PARALLELISM"])
    sizes = json.loads(cfg["COMPILE_SIZES"])

    assert all(size % tp == 0 for size in sizes)
    # The scheduler holds each owner to max_num_scheduled_tokens // tp tokens a
    # step (owner_affine_block_pool._steer_preemption). The largest bucket is
    # tp x that budget, so a step at the budget always has a bucket, also under
    # a min(..., max(compile_sizes) // tp) guard.
    budget = int(cfg["MAX_NUM_BATCHED_TOKENS"]) // tp
    assert max(sizes) == tp * budget
    tokens_per_request = 1 + int(cfg["DSPARK_K"])
    for concurrency in map(int, cfg["CONCURRENCY_OPTIONS"].split()):
        busiest_owner = -(-concurrency // tp) * tokens_per_request
        assert busiest_owner <= budget
        assert tp * busiest_owner in sizes


def test_caller_overrides_keep_precedence():
    cfg = _source(
        TPU_DEEPSEEK_V41_KV_SHARD="0",
        OFFLOAD="1",
        MAX_NUM_SEQS="128",
        CONCURRENCY_OPTIONS="96 128",
    )

    assert cfg["TPU_DEEPSEEK_V41_KV_SHARD"] == "0"
    assert cfg["MAX_NUM_SEQS"] == "128"
    assert cfg["CONCURRENCY_OPTIONS"] == "96 128"
    assert cfg["KV_CACHE_MEMORY_BYTES"] == str(40 * 2**30)
