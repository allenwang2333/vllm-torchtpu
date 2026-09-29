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
"""Exercise actual vLLM bucket specialization and AOT reload on 32 TPU ranks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from qualification import audit_hlo, require_graph_agreement
from two_shot_benchmark import TorchRunner, check_outputs, reference, write_json
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import DynamicShapesType
from vllm.forward_context import set_forward_context

import vllm_torchtpu
from vllm_torchtpu.compilation import shape_variants, tpu_compiler
from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import KimiTwoShotAllReduce


@support_torch_compile(dynamic_arg_dims={"x": 0})
class ReductionModel(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, operation, prefix=""):
        super().__init__()
        # Synthetic admission exercises both branches even before promotion.
        self.reduction = KimiTwoShotAllReduce(operation, (1,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(x, TorchRunner.native)


def make_config(model, rank):
    from vllm_torchtpu.models.vllm import register_models

    register_models()
    config = VllmConfig(
        model_config=ModelConfig(
            model=model, trust_remote_code=True, dtype="bfloat16", max_model_len=32
        ),
        device_config=DeviceConfig(device="tpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=32, distributed_executor_backend="external_launcher"
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=32,
            max_num_seqs=4,
            max_model_len=32,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="",
            compile_sizes=[1, 3],
            splitting_ops=[],
            cudagraph_capture_sizes=[],
            pass_config={"fuse_rope_kvcache_cat_mla": False},
        ),
    )
    config.parallel_config.rank = rank
    config.compilation_config.dynamic_shapes_config.type = (
        DynamicShapesType.BACKED_SIZE_OBLIVIOUS
    )
    return config


def count_starts():
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    counts = {"traces": 0, "aot_loads": 0}
    load = decorators._try_load_aot_compiled_fn
    compile_ = TorchCompileWithNoGuardsWrapper.aot_compile

    def counted_load(model, path):
        found = load(model, path)
        counts["aot_loads"] += found is not None
        return found

    def counted_compile(self, *args, **kwargs):
        counts["traces"] += 1
        return compile_(self, *args, **kwargs)

    decorators._try_load_aot_compiled_fn = counted_load
    TorchCompileWithNoGuardsWrapper.aot_compile = counted_compile
    return counts


def recording_backend(directory):
    from torch_tpu._internal.compile import TpuBackend
    from torch_tpu._internal.compiler_options import custom_compiler_options

    class RecordingBackend(TpuBackend):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __call__(self, graph, inputs, **kwargs):
            rows = {int(x.shape[0]) for x in inputs if isinstance(x, torch.Tensor)}
            if len(rows) != 1 or next(iter(rows)) not in (1, 3):
                raise ValueError(f"expected one specialized token bucket, got {rows}")
            batch = rows.pop()
            dump = directory / f"batch-{batch}"
            dump.mkdir(parents=True, exist_ok=True)
            with custom_compiler_options({"xla_dump_to": str(dump.resolve())}):
                result = super().__call__(graph, inputs, **kwargs)
            self.calls.append(batch)
            return result

    return RecordingBackend()


def run(args):
    os.environ.update(
        VLLM_CACHE_ROOT=str(Path(args.cache_root).resolve()),
        VLLM_DISABLE_COMPILE_CACHE="0",
        VLLM_USE_AOT_COMPILE="1",
        VLLM_USE_MEGA_AOT_ARTIFACT="0",
        TPU_K3_TWO_SHOT_ALL_REDUCE="1",
    )
    runner = TorchRunner(SimpleNamespace(manifest=None))
    config = make_config(args.model, runner.rank)
    vllm_torchtpu._patch_vllm_aot_compile_cache_key()
    output = Path(args.output) / f"rank-{runner.rank}"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "collective": runner.identity,
        "tpu_compilation_hash": tpu_compiler.compute_tpu_compilation_hash(config),
        "vllm_config_hash": config.compute_hash(),
        "order": args.order,
        "admitted_batches": [1],
    }
    record = {"identity": identity, "phase": args.phase, "ok": False}
    cache_metadata = Path(args.cache_root) / f"bucket-proof-rank-{runner.rank}.json"
    if args.phase == "reload":
        previous = json.loads(cache_metadata.read_text())
        if previous["identity"] != identity:
            raise ValueError(
                "cold and reload source/runtime/placement identities differ"
            )
    elif cache_metadata.exists():
        raise ValueError("cold bucket verification requires a fresh cache root")
    counts = count_starts()
    backend = recording_backend(output / "hlo")
    tpu_compiler._tpu_backend = backend
    with set_current_vllm_config(config):
        model = ReductionModel(vllm_config=config, operation=runner.operation)
    order = (1, 3) if args.order == "eligible-first" else (3, 1)
    checks = []

    def invoke(batch, pattern, iteration):
        value = runner.input(batch, 7168, "bfloat16", pattern, iteration)
        with set_forward_context({}, config):
            result = model(value)
        runner.sync(result)
        if result.shape != value.shape or result.dtype != value.dtype:
            raise RuntimeError("compiled bucket changed the shape or dtype")
        expected, tolerance = reference(
            batch,
            7168,
            "bfloat16",
            pattern,
            iteration,
            "two_shot" if batch == 1 else "native",
        )
        checks.extend(
            check_outputs(
                runner.outputs(result),
                expected,
                tolerance,
                batch=batch,
                pattern=pattern,
                iteration=iteration,
            )
        )

    retraces = 0
    with set_current_vllm_config(config), shape_variants.warmup() as tracked:
        for batch in order:
            if tracked.needs_retrace(batch):
                shape_variants.retrace(model, config)
                retraces += 1
            invoke(batch, "exact", 0)
    calls_after_warmup = list(backend.calls)
    for pattern in ("exact", "cancellation"):
        for iteration in range(2):
            for batch in reversed(order):
                invoke(batch, pattern, iteration)
    if backend.calls != calls_after_warmup:
        raise RuntimeError("bucket execution compiled after warmup")
    if args.phase == "cold":
        if sorted(backend.calls) != [1, 3] or counts["traces"] != 2:
            raise RuntimeError(
                f"expected two specialized compilations, got {backend.calls}, {counts}"
            )
        proofs = []
        for batch in (1, 3):
            implementation = "two_shot" if batch == 1 else "native"
            candidates = list(
                (output / "hlo" / f"batch-{batch}").rglob("*after_optimizations.txt")
            )
            matches = [
                (path, audit_hlo(path.read_text(), implementation, 1))
                for path in candidates
            ]
            valid = [(path, audit) for path, audit in matches if audit["ok"]]
            if len(valid) != 1:
                raise RuntimeError(
                    f"bucket {batch} did not emit its expected collective"
                )
            path, audit = valid[0]
            proofs.append(
                {
                    "batch": batch,
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "audit": audit,
                }
            )
        write_json(cache_metadata, {"identity": identity, "hlo": proofs})
    else:
        if backend.calls or counts != {"traces": 0, "aot_loads": 2}:
            raise RuntimeError(
                f"AOT reload retraced or recompiled: {backend.calls}, {counts}"
            )
        proofs = previous["hlo"]
        for item in proofs:
            path = Path(item["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("cold executable HLO evidence changed")
            implementation = "two_shot" if item["batch"] == 1 else "native"
            if audit_hlo(path.read_text(), implementation, 1) != item["audit"]:
                raise ValueError("cold executable HLO audit changed")
    record.update(
        correctness=checks,
        hlo=proofs,
        starts=counts,
        backend_calls=backend.calls,
        retraces=retraces,
        ok=all(check["ok"] for check in checks),
    )
    write_json(output / "bucket-result.json", record)
    records = runner.gather(record)
    for batch in (1, 3):
        require_graph_agreement(
            [
                {**proof["audit"], "process": process}
                for process, item in enumerate(records)
                for proof in item["hlo"]
                if proof["batch"] == batch
            ],
            runner.process_count,
            (1,),
        )
    if runner.rank == 0:
        write_json(
            Path(args.output) / "results.json",
            {"ok": all(item["ok"] for item in records), "per_rank": records},
        )
    if any(not item["ok"] for item in records):
        raise RuntimeError("bucket validation failed independent numerical reference")
    runner.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="canonical K3 model/config path; weights are not loaded",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--order", choices=("eligible-first", "native-first"), required=True
    )
    parser.add_argument("--phase", choices=("cold", "reload"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
