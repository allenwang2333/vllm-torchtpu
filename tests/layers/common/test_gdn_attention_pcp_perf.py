# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PCP=8 GDN wrapper performance test for Qwen3.5-35B-A3B-FP8.

This test deliberately bypasses the vLLM engine and OpenAI server.  It invokes
the production ``gdn_attention_core_tpu_pcp_prefill`` Pallas bridge with the
same physical shapes and dtypes observed in the 4096-token service baseline:

* 4096 physical token rows per native PCP rank (512 valid + 3584 padding);
* 32768 physical rows after the token-to-head all-to-all;
* all 16 K/Q and 32 V heads at the custom-op entrance;
* the 35-block BF16/FP32 state layout used by the unified KV pool;
* physical PCP-local state, Torch-side weight slicing, and whole-state refill.

The service-side 7.246 ms number is an XPlane anchored-region proxy from a
single monolithic backbone containing 30 GDN layers.  It is recorded here for
comparison, but it is not used as an equality assertion: this test measures an
isolated wrapper's synchronized host wall time.  The isolated result is the
fast regression metric to compare before and after a GDN implementation change.

The pre-optimization replicated-state implementation measured 10.344 ms host
wall p50 (and 9.753 ms isolated XPlane anchored p50) on this machine.  Those
numbers are the A/B baseline for this test, not a service-latency assertion.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

WORLD_SIZE = 8
MODEL_PATH = Path(
    os.getenv(
        "QWEN35_35B_MODEL",
        "/home/xiaohao_yxh/workspace/models/Qwen3.5-35B-A3B-FP8",
    ))

LOCAL_PADDED_TOKENS = 4096
GLOBAL_VALID_TOKENS = 4096
GLOBAL_PADDED_TOKENS = LOCAL_PADDED_TOKENS * WORLD_SIZE
INTERLEAVE_SIZE = 256
NUM_BLOCKS = 35
MAX_REQUESTS = 8

NUM_KQ_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL_SIZE = 4
QKV_WIDTH = 2 * NUM_KQ_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM
OUTPUT_WIDTH = NUM_V_HEADS * HEAD_V_DIM
LOCAL_QKV_WIDTH = QKV_WIDTH // WORLD_SIZE
LOCAL_NUM_V_HEADS = NUM_V_HEADS // WORLD_SIZE

WARMUP_ITERS = 2
MEASURE_ITERS = 9
LAUNCH_TIMEOUT_SECONDS = 1200

PRE_OPT_ISOLATED_HOST_P50_MS = 10.344
PRE_OPT_ISOLATED_DEVICE_P50_MS = 9.753
PRE_OPT_ISOLATED_FUSED_PER_SEQ_MS = 1.079

# Cross-reference only.  It is not an exact isolated-op wall-time target; see
# the module docstring and PROFILE_SUMMARY.md in the associated baseline bundle.
SERVICE_ANCHORED_PROXY_P50_MS = 7.246
SERVICE_WRAPPER_AMORTIZED_MS = 7.940
SERVICE_BASELINE_GCS = (
    "gs://xiaohao-tran/xiaohao_ignore/"
    "qwen35_35b_gdn_prefill_pcp8_xprof_20260713T073305Z.tar.zst")
SERVICE_BASELINE_SHA256 = (
    "611a9bd2e81f24021f91cdd559d2984d3d0261d8ea76a96173bace9c5146043c")


def _load_and_validate_model_config() -> dict[str, int]:
    config_path = MODEL_PATH / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config["text_config"]
    actual = {
        "num_kq_heads": int(text["linear_num_key_heads"]),
        "num_v_heads": int(text["linear_num_value_heads"]),
        "head_k_dim": int(text["linear_key_head_dim"]),
        "head_v_dim": int(text["linear_value_head_dim"]),
        "conv_kernel_size": int(text["linear_conv_kernel_dim"]),
        "hidden_size": int(text["hidden_size"]),
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "full_attention_interval": int(text["full_attention_interval"]),
    }
    expected = {
        "num_kq_heads": NUM_KQ_HEADS,
        "num_v_heads": NUM_V_HEADS,
        "head_k_dim": HEAD_K_DIM,
        "head_v_dim": HEAD_V_DIM,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "full_attention_interval": 4,
    }
    if actual != expected:
        raise AssertionError(
            f"Unexpected Qwen3.5-35B text config: actual={actual}, "
            f"expected={expected}, path={config_path}")
    return actual


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
    ]


def _append_libtpu_flag(current: str, flag: str) -> str:
    parts = current.split()
    flag_name = flag.split("=", 1)[0]
    parts = [part for part in parts if part.split("=", 1)[0] != flag_name]
    parts.append(flag)
    return " ".join(parts)


def _prepare_worker_env() -> dict[str, str]:
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
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["RAGGED_GATED_DELTA_RULE_IMPL"] = "chunked_kernel_v3_pd"
    env["LIBTPU_INIT_ARGS"] = _append_libtpu_flag(
        env.get("LIBTPU_INIT_ARGS", ""),
        "--xla_tpu_scoped_vmem_limit_kib=65536",
    )
    env["QWEN35_35B_MODEL"] = str(MODEL_PATH)
    return env


def _run_worker_group(result_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[3],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        raise AssertionError(
            "Qwen3.5-35B PCP GDN perf workers timed out. Output tail:\n"
            f"{output[-16000:]}") from exc
    return subprocess.CompletedProcess(process.args, process.returncode,
                                       output, "")


def _gather_global_device_ids(torch, dist, tpu_distributed) -> tuple[int, ...]:
    local_id = torch.tensor(
        [int(tpu_distributed.global_device_id())],
        dtype=torch.int32,
        device="tpu",
    )
    gathered = [torch.empty_like(local_id) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local_id)
    return tuple(int(tensor.cpu().item()) for tensor in gathered)


def _build_mesh(jax, np, global_device_ids: tuple[int, ...]):
    """Use stable torchrun rank order retained by the isolated A/B baseline."""
    devices_by_id = {int(device.id): device for device in jax.devices()}
    missing = [
        device_id for device_id in global_device_ids
        if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(f"JAX does not expose TPU ids {missing}; "
                           f"available={sorted(devices_by_id)}")
    devices = np.asarray(
        [devices_by_id[device_id] for device_id in global_device_ids])
    return jax.sharding.Mesh(devices, axis_names=("pcp", ))


def _make_inputs(torch, sync):
    mixed_qkv = torch.zeros(
        (LOCAL_PADDED_TOKENS, QKV_WIDTH),
        dtype=torch.bfloat16,
        device="tpu",
    )
    b = torch.zeros(
        (LOCAL_PADDED_TOKENS, NUM_V_HEADS),
        dtype=torch.bfloat16,
        device="tpu",
    )
    a = torch.zeros_like(b)
    conv_state = torch.zeros(
        (NUM_BLOCKS, CONV_KERNEL_SIZE - 1, LOCAL_QKV_WIDTH),
        dtype=torch.bfloat16,
        device="tpu",
    )
    recurrent_state = torch.zeros(
        (NUM_BLOCKS, LOCAL_NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        dtype=torch.float32,
        device="tpu",
    )
    conv_weight = torch.zeros(
        (QKV_WIDTH, 1, CONV_KERNEL_SIZE),
        dtype=torch.bfloat16,
        device="tpu",
    )
    a_log = torch.zeros((NUM_V_HEADS, ), dtype=torch.float32, device="tpu")
    dt_bias = torch.zeros((NUM_V_HEADS, ), dtype=torch.bfloat16, device="tpu")

    # vLLM's PCP runner pads request metadata to PCP size.  Only request 0 is
    # live; non-decode requests use the mixed bucket [0, 0, mixed_end].
    state_indices = torch.tensor([1] + [0] * (MAX_REQUESTS - 1),
                                 dtype=torch.int32,
                                 device="tpu")
    query_start_loc = torch.tensor(
        [0] + [GLOBAL_VALID_TOKENS] * MAX_REQUESTS,
        dtype=torch.int32,
        device="tpu",
    )
    distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device="tpu")
    seq_lens = torch.tensor(
        [GLOBAL_VALID_TOKENS] + [0] * (MAX_REQUESTS - 1),
        dtype=torch.int32,
        device="tpu",
    )

    immutable = (
        mixed_qkv,
        b,
        a,
        conv_weight,
        a_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
    )
    sync.synchronize([conv_state, recurrent_state, *immutable], wait=True)
    return conv_state, recurrent_state, immutable


def _build_compiled_gdn(torch, mesh, pcp_rank: int):
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import compile as tpu_compile

    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import \
        pcp_streaming_jax_op
    from vllm_torchtpu.layers.common.gdn_attention import GdnAttentionConfig
    from vllm_torchtpu.layers.common.ragged_gated_delta_rule_wrapper import \
        RaggedGatedDeltaRuleImpl
    from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import (
        _pcp_slice_dim_torch, _pcp_slice_reordered_concatenated_torch,
        gdn_attention_core_tpu_pcp_prefill)

    config = GdnAttentionConfig(ragged_gated_delta_rule_impl=(
        RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_V3_PD))
    wrapped_fn = functools.partial(
        gdn_attention_core_tpu_pcp_prefill,
        mesh=mesh,
        n_kq=NUM_KQ_HEADS,
        n_v=NUM_V_HEADS,
        d_k=HEAD_K_DIM,
        d_v=HEAD_V_DIM,
        kernel_size=CONV_KERNEL_SIZE,
        pcp_size=WORLD_SIZE,
        interleave_size=INTERLEAVE_SIZE,
        config=config,
    )
    input_partition_specs = (
        P("pcp"),  # mixed_qkv
        P("pcp"),  # b
        P("pcp"),  # a
        P(None, None, "pcp"),  # conv_state
        P(None, "pcp", None, None),  # recurrent_state
        P("pcp", None, None),  # conv_weight
        None,  # conv_bias
        P("pcp"),  # A_log
        P("pcp"),  # dt_bias
        P(),  # state_indices
        P(),  # query_start_loc
        P(),  # request_distribution
        P(),  # seq_lens
    )
    gdn = pcp_streaming_jax_op(
        "pallas::qwen35_35b_pcp8_gdn_perf",
        wrapped_fn,
        donate_argnums=(3, 4),
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=(
            P(None, None, "pcp"),
            P(None, "pcp", None, None),
            P("pcp"),
        ),
    )

    def _fake(mixed_qkv, _b, _a, conv_state, recurrent_state, *_args):
        output = torch.empty(
            (mixed_qkv.size(0), OUTPUT_WIDTH),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        return (torch.empty_like(conv_state),
                torch.empty_like(recurrent_state), output)

    gdn.register_fake(_fake)

    def run_one(conv_state, recurrent_state, mixed_qkv, b, a, conv_weight,
                a_log, dt_bias, state_indices, query_start_loc, distribution,
                seq_lens):
        local_key_dim = NUM_KQ_HEADS * HEAD_K_DIM
        local_value_dim = NUM_V_HEADS * HEAD_V_DIM
        split_sizes = [local_key_dim, local_key_dim, local_value_dim]
        conv_weight_arg = _pcp_slice_reordered_concatenated_torch(conv_weight,
                                                                  split_sizes,
                                                                  pcp_rank,
                                                                  WORLD_SIZE,
                                                                  axis=0)
        a_log_arg = _pcp_slice_dim_torch(a_log, pcp_rank, WORLD_SIZE,
                                         axis=0).contiguous()
        dt_bias_arg = _pcp_slice_dim_torch(dt_bias,
                                           pcp_rank,
                                           WORLD_SIZE,
                                           axis=0).contiguous()
        new_conv, new_recurrent, output = gdn(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight_arg,
            None,
            a_log_arg,
            dt_bias_arg,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
        )
        conv_state.copy_(new_conv)
        recurrent_state.copy_(new_recurrent)
        return conv_state, recurrent_state, output

    backend = tpu_compile.TpuBackend()
    return torch.compile(run_one,
                         fullgraph=True,
                         dynamic=False,
                         backend=backend)


def _run_once(torch, dist, sync, cpu_group, compiled_gdn, conv_state,
              recurrent_state, immutable):
    dist.barrier(group=cpu_group)
    sync.synchronize([conv_state, recurrent_state, *immutable], wait=True)
    start = time.perf_counter()
    new_conv, new_recurrent, output = compiled_gdn(conv_state, recurrent_state,
                                                   *immutable)
    sync.synchronize([new_conv, new_recurrent, output], wait=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    dist.barrier(group=cpu_group)
    return new_conv, new_recurrent, output, elapsed_ms


def _run_worker(result_dir: Path) -> None:
    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist
    from torch_tpu._internal import sync
    from torch_tpu._internal.distributed import tpu_distributed
    from torch_tpu._internal.profiler import profiler_api

    rank = int(os.environ["RANK"])
    result_path = result_dir / f"rank_{rank}.json"
    initialized = False
    result: dict[str, object]
    try:
        model_config = _load_and_validate_model_config()
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        cpu_group = dist.new_group(ranks=list(range(WORLD_SIZE)),
                                   backend="gloo")

        torch.empty((1, ), device="tpu").cpu()
        import jax

        global_device_ids = _gather_global_device_ids(torch, dist,
                                                      tpu_distributed)
        mesh = _build_mesh(jax, np, global_device_ids)
        conv_state, recurrent_state, immutable = _make_inputs(torch, sync)
        compiled_gdn = _build_compiled_gdn(torch, mesh, rank)

        warmup_ms = []
        output = None
        for _ in range(WARMUP_ITERS):
            conv_state, recurrent_state, output, elapsed_ms = _run_once(
                torch,
                dist,
                sync,
                cpu_group,
                compiled_gdn,
                conv_state,
                recurrent_state,
                immutable,
            )
            warmup_ms.append(elapsed_ms)

        # Optional device trace. Rank 0 captures the two TensorCore planes of
        # its bound chip, matching the service-side XPlane device scope.
        profile_context = None
        capture_xprof = os.getenv("QWEN35_GDN_PCP_PERF_CAPTURE_XPROF") == "1"
        if rank == 0 and capture_xprof:
            profile_dir = result_dir / "profile"
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_context = profiler_api.profile(
                activities=[profiler_api.ProfilerActivity.TPU],
                on_trace_ready=profiler_api.xprof_trace_handler(profile_dir),
            )
            profile_context.__enter__()
        dist.barrier(group=cpu_group)
        measured_ms = []
        try:
            for _ in range(MEASURE_ITERS):
                conv_state, recurrent_state, output, elapsed_ms = _run_once(
                    torch,
                    dist,
                    sync,
                    cpu_group,
                    compiled_gdn,
                    conv_state,
                    recurrent_state,
                    immutable,
                )
                measured_ms.append(elapsed_ms)
        finally:
            dist.barrier(group=cpu_group)
            if profile_context is not None:
                profile_context.__exit__(None, None, None)
            dist.barrier(group=cpu_group)

        assert output is not None
        assert tuple(output.shape) == (LOCAL_PADDED_TOKENS, OUTPUT_WIDTH)
        assert output.dtype == torch.bfloat16
        assert tuple(conv_state.shape) == (NUM_BLOCKS, CONV_KERNEL_SIZE - 1,
                                           LOCAL_QKV_WIDTH)
        assert conv_state.dtype == torch.bfloat16
        assert tuple(recurrent_state.shape) == (NUM_BLOCKS, LOCAL_NUM_V_HEADS,
                                                HEAD_K_DIM, HEAD_V_DIM)
        assert recurrent_state.dtype == torch.float32
        result = {
            "rank": rank,
            "global_device_id": int(tpu_distributed.global_device_id()),
            "rank_order_device_ids": global_device_ids,
            "model_path": str(MODEL_PATH),
            "model_config": model_config,
            "pcp_size": WORLD_SIZE,
            "interleave_size": INTERLEAVE_SIZE,
            "local_valid_tokens": GLOBAL_VALID_TOKENS // WORLD_SIZE,
            "local_padded_tokens": LOCAL_PADDED_TOKENS,
            "global_valid_tokens": GLOBAL_VALID_TOKENS,
            "global_padded_tokens": GLOBAL_PADDED_TOKENS,
            "qkv_width": QKV_WIDTH,
            "physical_pcp_local_state": True,
            "capture_xprof": capture_xprof,
            "output_shape": list(output.shape),
            "conv_state_shape": list(conv_state.shape),
            "recurrent_state_shape": list(recurrent_state.shape),
            "warmup_ms": warmup_ms,
            "measured_ms": measured_ms,
        }
    except BaseException as exc:
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _assert_and_print_perf_summary(tmp_path: Path):
    if not (MODEL_PATH / "config.json").is_file():
        pytest.skip(f"Qwen3.5-35B model config is unavailable: {MODEL_PATH}")
    model_config = _load_and_validate_model_config()

    result_dir_env = os.getenv("QWEN35_GDN_PCP_PERF_RESULT_DIR")
    result_dir = (Path(result_dir_env) if result_dir_env else tmp_path /
                  "qwen35_35b_pcp8_gdn_perf")
    result_dir.mkdir(parents=True, exist_ok=True)
    completed = _run_worker_group(result_dir)

    results = []
    for rank in range(WORLD_SIZE):
        result_path = result_dir / f"rank_{rank}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            results.append({"rank": rank, "error": "missing worker result"})
    errors = [result for result in results if "error" in result]
    assert completed.returncode == 0 and not errors, (
        "Qwen3.5-35B PCP8 GDN perf failure: "
        f"returncode={completed.returncode}, errors={errors}\n"
        f"worker output tail:\n{completed.stdout[-16000:]}")

    iteration_max_ms = [
        max(float(result["measured_ms"][iteration]) for result in results)
        for iteration in range(MEASURE_ITERS)
    ]
    warmup_iteration_max_ms = [
        max(float(result["warmup_ms"][iteration]) for result in results)
        for iteration in range(WARMUP_ITERS)
    ]
    p50_ms = statistics.median(iteration_max_ms)
    summary = {
        "metric": ("isolated_production_head_sharded_pcp_gdn_wrapper_"
                   "synchronized_host_wall"),
        "min_ms":
        min(iteration_max_ms),
        "p50_ms":
        p50_ms,
        "mean_ms":
        statistics.fmean(iteration_max_ms),
        "p90_ms":
        _percentile(iteration_max_ms, 0.90),
        "max_ms":
        max(iteration_max_ms),
        "iteration_max_ms":
        iteration_max_ms,
        "warmup_iteration_max_ms":
        warmup_iteration_max_ms,
        "measure_iters":
        MEASURE_ITERS,
        "service_anchored_proxy_p50_ms":
        SERVICE_ANCHORED_PROXY_P50_MS,
        "service_wrapper_amortized_ms":
        SERVICE_WRAPPER_AMORTIZED_MS,
        "delta_vs_service_proxy_percent":
        ((p50_ms / SERVICE_ANCHORED_PROXY_P50_MS - 1.0) * 100.0),
        "pre_optimization_isolated_host_p50_ms":
        (PRE_OPT_ISOLATED_HOST_P50_MS),
        "pre_optimization_isolated_device_p50_ms":
        (PRE_OPT_ISOLATED_DEVICE_P50_MS),
        "pre_optimization_isolated_fused_per_seq_ms":
        (PRE_OPT_ISOLATED_FUSED_PER_SEQ_MS),
        "speedup_vs_pre_optimization_host_p50":
        (PRE_OPT_ISOLATED_HOST_P50_MS / p50_ms),
        "model_path":
        str(MODEL_PATH),
        "model_config":
        model_config,
        "pcp_size":
        WORLD_SIZE,
        "interleave_size":
        INTERLEAVE_SIZE,
        "local_valid_tokens":
        GLOBAL_VALID_TOKENS // WORLD_SIZE,
        "local_padded_tokens":
        LOCAL_PADDED_TOKENS,
        "global_valid_tokens":
        GLOBAL_VALID_TOKENS,
        "global_padded_tokens":
        GLOBAL_PADDED_TOKENS,
        "qkv_width":
        QKV_WIDTH,
        "num_blocks":
        NUM_BLOCKS,
        "physical_pcp_local_state":
        True,
        "service_baseline_gcs":
        SERVICE_BASELINE_GCS,
        "service_baseline_sha256":
        SERVICE_BASELINE_SHA256,
        "rank_order_device_ids":
        results[0]["rank_order_device_ids"],
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    (result_dir / "workers.log").write_text(completed.stdout, encoding="utf-8")
    print("Qwen3.5-35B PCP8 complete GDN wrapper perf summary:",
          json.dumps(summary, sort_keys=True))

    # Allow machine noise while catching a regression beyond the accepted
    # pre-optimization baseline. Override for a different TPU topology.
    max_p50_ms = float(os.getenv("QWEN35_GDN_PCP_PERF_MAX_P50_MS", "13.0"))
    assert p50_ms <= max_p50_ms, (
        f"isolated PCP GDN p50 {p50_ms:.3f} ms exceeds "
        f"QWEN35_GDN_PCP_PERF_MAX_P50_MS={max_p50_ms:.3f} ms")


@pytest.mark.nightly
@pytest.mark.timeout(LAUNCH_TIMEOUT_SECONDS + 60)
def test_qwen35_35b_pcp8_4k_complete_gdn_wrapper_perf(tmp_path):
    _assert_and_print_perf_summary(tmp_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        _run_worker(args.result_dir)
        return
    raise ValueError("expected --worker")


if __name__ == "__main__":
    _main()
