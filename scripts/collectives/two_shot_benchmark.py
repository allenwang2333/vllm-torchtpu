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
"""Measure Two Shot through standalone JAX and the production Torch bridge."""

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from qualification import (
    BATCHES,
    PROTOCOL,
    audit_hlo,
    qualify,
    read_run,
    require_graph_agreement,
    summarize,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_array(rank, batch, width, dtype, pattern, iteration):
    import ml_dtypes
    import numpy as np

    if pattern == "exact":
        values = (
            (np.arange(batch * width).reshape(batch, width) % 5 - 2) + rank % 7 - 3
        ) / 8 + iteration / 16
    else:
        rng = np.random.default_rng(2027 + rank // 2 + iteration * 101)
        values = rng.normal(size=(batch, width)) * (-1 if rank % 2 else 1)
        values += (rank % 3 - 1) / 128
    return values.astype(ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float32)


def reference(batch, width, dtype, pattern, iteration, implementation="native"):
    import numpy as np

    values = [
        input_array(r, batch, width, dtype, pattern, iteration).astype(np.float64)
        for r in range(32)
    ]
    expected = sum(values)
    absolute_sum = sum(np.abs(x) for x in values)
    epsilon = 2**-7 if dtype == "bfloat16" else 2**-23
    if implementation == "two_shot":
        # FP32 accumulation followed by one output rounding, even under cancellation.
        unit = 2**-24
        gamma = 31 * unit / (1 - 31 * unit)
        tolerance = (
            absolute_sum * gamma * (1 + epsilon / 2)
            + np.abs(expected) * epsilon / 2
            + 1e-7
        )
    else:
        tolerance = absolute_sum * epsilon * 2 + 1e-6
    return expected, tolerance


def check_outputs(outputs, expected, tolerance, **metadata):
    import numpy as np

    checks = []
    for rank, actual in outputs:
        error = np.abs(actual.astype(np.float64) - expected)
        scaled = error / tolerance
        checks.append(
            {
                **metadata,
                "rank": rank,
                "ok": bool(np.isfinite(scaled).all() and np.max(scaled) <= 1),
                "max_abs_error": float(np.max(error)),
                "max_scaled_error": float(np.max(scaled)),
                "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            }
        )
    return checks


class TorchRunner:
    def __init__(self, args):
        import torch
        import torch.distributed as dist

        import vllm_torchtpu  # noqa: F401
        from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import (
            build_kimi_two_shot_all_reduce,
            prepare_two_shot_context,
        )

        dist.init_process_group(backend="tpu_dist")
        self.rank = dist.get_rank()
        self.process_count = self.world = dist.get_world_size()
        self.cpu_group = dist.new_group(backend="gloo")
        torch.empty(1, device="tpu").cpu()
        group = SimpleNamespace(
            ranks=list(range(self.world)),
            world_size=self.world,
            cpu_group=self.cpu_group,
            unique_name="two_shot_benchmark",
        )
        _, self.identity = prepare_two_shot_context(group)
        self.operation = build_kimi_two_shot_all_reduce("two_shot_benchmark", group)
        self.batches = ()
        if args.manifest:
            from vllm_torchtpu.distributed.two_shot_qualification import (
                qualified_batches,
            )
            from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import (
                KimiTwoShotAllReduce,
            )

            manifest = json.loads(Path(args.manifest).read_text())
            self.batches = qualified_batches(self.identity, manifest)
            self.automatic = KimiTwoShotAllReduce(self.operation, self.batches)

    def gather(self, value):
        import torch.distributed as dist

        result = [None] * self.process_count
        dist.all_gather_object(result, value, group=self.cpu_group)
        return result

    def barrier(self):
        import torch.distributed as dist

        dist.barrier(group=self.cpu_group)

    def input(self, batch, width, dtype, pattern="exact", iteration=0):
        import numpy as np
        import torch

        values = input_array(self.rank, batch, width, dtype, pattern, iteration)
        return torch.from_numpy(values.astype(np.float32)).to(
            dtype=getattr(torch, dtype), device="tpu"
        )

    def outputs(self, result):
        return [(self.rank, result.float().cpu().numpy())]

    def sync(self, result):
        from torch_tpu._internal import sync

        sync.synchronize([result], wait=True)

    @staticmethod
    def native(value):
        import torch.distributed as dist

        result = value.clone()
        dist.all_reduce(result)
        return result

    def chain(self, implementation, length):
        import torch

        operation = self.native if implementation == "native" else self.operation
        if implementation == "auto":

            def operation(x):
                return self.automatic(x, self.native)

        def function(initial):
            current = initial
            for _ in range(length):
                result = operation(current)
                result = torch.ops.tpu.optimization_barrier([result])[0]
                factor = (1.0 + result.float().sum() * 1e-30).to(initial.dtype)
                current = initial * factor
            return result

        return function

    def compile(self, implementation, length, value, directory):
        import torch
        from torch_tpu._internal.compile import TpuBackend
        from torch_tpu._internal.compiler_options import custom_compiler_options

        directory.mkdir(parents=True, exist_ok=True)
        backend = TpuBackend()
        function = torch.compile(
            self.chain(implementation, length),
            backend=backend,
            dynamic=False,
            fullgraph=True,
            isolate_recompiles=True,
        )
        start = time.perf_counter()
        with custom_compiler_options({"xla_dump_to": str(directory.resolve())}):
            result = function(value)
            self.sync(result)
        paths = list(directory.rglob("*after_optimizations.txt"))
        expected = (
            (
                "two_shot"
                if value.shape[0] in self.batches
                and value.shape[1] == 7168
                and value.dtype == torch.bfloat16
                else "native"
            )
            if implementation == "auto"
            else implementation
        )
        matches = [
            (path, audit_hlo(path.read_text(), expected, length)) for path in paths
        ]
        valid = [(path, audit) for path, audit in matches if audit["ok"]]
        if len(valid) != 1:
            raise RuntimeError(
                f"expected one optimized chain HLO in {directory}; found {matches}"
            )
        path, audit = valid[0]
        audit.update(
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            process=self.rank,
            compile_seconds=time.perf_counter() - start,
        )
        count = len(backend._compiled_executables)

        def run(x):
            result = function(x)
            if len(backend._compiled_executables) != count:
                raise RuntimeError("benchmark recompiled after initialization")
            return result

        run.executables = backend._compiled_executables
        return run, audit


class JaxRunner:
    def __init__(self, args):
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        if args.distributed:
            kwargs = {
                key: getattr(args, key)
                for key in ("coordinator_address", "num_processes", "process_id")
                if getattr(args, key) is not None
            }
            jax.distributed.initialize(**kwargs)
        from vllm_torchtpu.distributed.two_shot_qualification import (
            qualification_identity,
        )

        devices = jax.devices()
        self.rank, self.process_count, self.world = (
            jax.process_index(),
            jax.process_count(),
            len(devices),
        )
        self.mesh = Mesh(np.asarray(devices), ("k3_ar",))
        self.sharding = NamedSharding(self.mesh, P("k3_ar", None, None))
        inventory = [
            {
                "id": d.id,
                "device_kind": d.device_kind,
                "coords": list(d.coords),
                "core_on_chip": d.core_on_chip,
            }
            for d in devices
        ]
        self.identity = qualification_identity(inventory, [d.id for d in devices])
        if args.manifest:
            raise ValueError("automatic serving verification uses --framework torch")

    def gather(self, value):
        import numpy as np
        from jax.experimental import multihost_utils

        raw = json.dumps(value).encode()
        lengths = multihost_utils.process_allgather(
            np.asarray(len(raw), np.int32)
        ).reshape(-1)
        buffer = np.zeros(int(max(lengths)), np.uint8)
        buffer[: len(raw)] = np.frombuffer(raw, np.uint8)
        gathered = multihost_utils.process_allgather(buffer, tiled=False).reshape(
            self.process_count, -1
        )
        return [
            json.loads(row[: int(length)].tobytes())
            for row, length in zip(gathered, lengths, strict=True)
        ]

    def barrier(self):
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("two-shot-benchmark")

    def input(self, batch, width, dtype, pattern="exact", iteration=0):
        import jax
        import numpy as np

        host = np.stack(
            [
                input_array(r, batch, width, dtype, pattern, iteration)
                for r in range(self.world)
            ]
        )
        return jax.make_array_from_callback(
            host.shape, self.sharding, lambda index: host[index]
        )

    def outputs(self, result):
        import numpy as np

        return [
            (shard.index[0].start, np.asarray(shard.data[0]).astype(np.float32))
            for shard in result.addressable_shards
        ]

    def sync(self, result):
        result.block_until_ready()

    def chain(self, implementation, length):
        import jax
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec as P

        from vllm_torchtpu.kernels.collectives.two_shot import two_shot_all_reduce

        def function(x):
            initial = x[0]
            current = initial
            for _ in range(length):
                result = (
                    jax.lax.psum(current, "k3_ar")
                    if implementation == "native"
                    else two_shot_all_reduce(
                        current,
                        axis_name="k3_ar",
                        axis_size=self.world,
                        collective_id=1,
                    )
                )
                result = jax.lax.optimization_barrier(result)
                factor = (1.0 + jnp.sum(result.astype(jnp.float32)) * 1e-30).astype(
                    initial.dtype
                )
                current = initial * factor
            return result[None]

        return jax.jit(
            jax.shard_map(
                function,
                mesh=self.mesh,
                in_specs=P("k3_ar", None, None),
                out_specs=P("k3_ar", None, None),
                check_vma=False,
            )
        )

    def compile(self, implementation, length, value, directory):
        directory.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        executable = self.chain(implementation, length).lower(value).compile()
        text = executable.as_text()
        path = directory / "optimized.hlo.txt"
        path.write_text(text)
        audit = audit_hlo(text, implementation, length)
        audit.update(
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            process=self.rank,
            compile_seconds=time.perf_counter() - start,
        )
        if not audit["ok"]:
            raise RuntimeError(f"optimized JAX chain failed HLO audit: {audit}")
        return executable, audit


def save_executable(function, path, identity, case, implementation):
    import cloudpickle

    if len(function.executables) != 1:
        raise ValueError("cache validation requires exactly one executable")
    data = cloudpickle.dumps(function.executables[0])
    path.write_bytes(data)
    write_json(
        path.with_suffix(".json"),
        {
            "identity": identity,
            "case": case,
            "implementation": implementation,
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )


def reload_executables(runner, source, output):
    import cloudpickle

    checks = []
    paths = sorted((Path(source) / f"process-{runner.rank}").rglob("executable.pkl"))
    if not paths:
        raise ValueError("no saved executables for this rank")
    for path in paths:
        metadata = json.loads(path.with_suffix(".json").read_text())
        data = path.read_bytes()
        if (
            metadata["identity"] != runner.identity
            or metadata["sha256"] != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(
                "saved executable source/runtime/placement or bytes changed"
            )
        executable = cloudpickle.loads(data)
        batch, width, dtype = metadata["case"]
        for pattern in ("exact", "cancellation"):
            for iteration in range(2):
                value = runner.input(batch, width, dtype, pattern, iteration)
                result = executable([value])[0]
                runner.sync(result)
                expected, tolerance = reference(
                    batch, width, dtype, pattern, iteration, metadata["implementation"]
                )
                checks.extend(
                    check_outputs(
                        runner.outputs(result),
                        expected,
                        tolerance,
                        pattern=pattern,
                        iteration=iteration,
                        executable=str(path),
                        length=1,
                    )
                )
    gathered = [item for group in runner.gather(checks) for item in group]
    if runner.rank == 0:
        write_json(
            Path(output) / "reload.json",
            {
                "identity": runner.identity,
                "checks": gathered,
                "ok": all(c["ok"] for c in gathered),
            },
        )
    if any(not check["ok"] for check in gathered):
        raise RuntimeError(
            "serialized executable failed independent numerical reference"
        )
    runner.barrier()


def run(args):
    runner = TorchRunner(args) if args.framework == "torch" else JaxRunner(args)
    if runner.world != 32:
        raise ValueError("Two Shot qualification requires 32 devices")
    identities = runner.gather(runner.identity)
    if any(item != runner.identity for item in identities):
        raise ValueError("workers disagree on qualification identity")
    output = Path(args.output)
    occupied = output.exists() and any(output.iterdir()) if runner.rank == 0 else False
    if any(runner.gather(occupied)):
        raise ValueError("use an empty output directory for each fresh launch")
    if args.reload_executables:
        if args.framework != "torch":
            raise ValueError("cache reload validates the Torch bridge")
        reload_executables(runner, args.reload_executables, output)
        return
    run_id = runner.gather(str(uuid.uuid4()))[0]
    protocol = (
        {"low": 2, "high": 3, "trials": 3, "warmup": 1} if args.smoke else PROTOCOL
    )
    mode = "auto" if args.manifest else "forced"
    document = {
        "schema_version": 1,
        "run_id": run_id,
        "started_unix": time.time(),
        "framework": args.framework,
        "world_size": runner.world,
        "process_count": runner.process_count,
        "identity": runner.identity,
        "harness_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("qualification.py").read_bytes()
        ).hexdigest(),
        "protocol": protocol,
        "mode": mode,
        "smoke": args.smoke,
        "complete": False,
        "cases": [],
    }
    if args.manifest:
        document["manifest_sha256"] = hashlib.sha256(
            Path(args.manifest).read_bytes()
        ).hexdigest()
        document["admitted_batches"] = list(runner.batches)
    batches = (
        [int(x) for x in args.batches.split(",")]
        if args.batches
        else ([4] if args.smoke else list(BATCHES))
    )
    if args.manifest and 3 not in batches:
        batches.append(3)
    cases = [(batch, 7168, "bfloat16") for batch in batches]
    if args.correctness_extras:
        cases.extend([(3, 129, "float32"), (2, 7169, "bfloat16")])
    for batch, width, dtype in cases:
        if runner.rank == 0:
            print(f"Validating {args.framework} {dtype} [{batch}, {width}]", flush=True)
        case = {"batch": batch, "width": width, "dtype": dtype, "arms": {}}
        value = runner.input(batch, width, dtype)
        implementations = ("native", "auto" if args.manifest else "two_shot")
        functions = {}
        for implementation in implementations:
            actual = (
                (
                    "two_shot"
                    if batch in runner.batches and width == 7168 and dtype == "bfloat16"
                    else "native"
                )
                if implementation == "auto"
                else implementation
            )
            checks, eager_checks, audits = [], [], []
            directory = (
                output
                / f"process-{runner.rank}"
                / f"{batch}-{width}-{dtype}"
                / implementation
            )
            chain_functions = {}
            lengths = sorted({1, 2, 3, protocol["low"], protocol["high"]})
            for length in lengths:
                function, audit = runner.compile(
                    implementation, length, value, directory / str(length)
                )
                chain_functions[length] = function
                if args.save_executables and args.framework == "torch" and length == 1:
                    save_executable(
                        function,
                        directory / "executable.pkl",
                        runner.identity,
                        (batch, width, dtype),
                        actual,
                    )
                if length in (protocol["low"], protocol["high"]):
                    audits.append(audit)
            for pattern in ("exact", "cancellation"):
                for iteration in range(2):
                    changed = runner.input(batch, width, dtype, pattern, iteration)
                    expected, tolerance = reference(
                        batch, width, dtype, pattern, iteration, actual
                    )
                    if args.framework == "torch":
                        eager = runner.chain(implementation, 1)(changed)
                        runner.sync(eager)
                        eager_checks.extend(
                            check_outputs(
                                runner.outputs(eager),
                                expected,
                                tolerance,
                                pattern=pattern,
                                iteration=iteration,
                                length=1,
                            )
                        )
                    for length in lengths:
                        result = chain_functions[length](changed)
                        runner.sync(result)
                        checks.extend(
                            check_outputs(
                                runner.outputs(result),
                                expected,
                                tolerance,
                                pattern=pattern,
                                iteration=iteration,
                                length=length,
                            )
                        )
            all_eager = [
                item for group in runner.gather(eager_checks) for item in group
            ]
            all_checks = [item for group in runner.gather(checks) for item in group]
            if any(not item["ok"] for item in [*all_checks, *all_eager]):
                write_json(
                    directory / "failed-correctness.json",
                    {"compiled": all_checks, "eager": all_eager},
                )
                raise RuntimeError("independent all-rank numerical reference failed")
            all_audits = [item for group in runner.gather(audits) for item in group]
            try:
                require_graph_agreement(
                    all_audits,
                    runner.process_count,
                    (protocol["low"], protocol["high"]),
                )
            except ValueError:
                write_json(directory / "failed-graph-agreement.json", all_audits)
                raise
            case["arms"][implementation] = {
                "implementation": actual,
                "correctness": all_checks,
                "eager_correctness": all_eager,
                "hlo": all_audits,
            }
            functions[implementation] = chain_functions
        lengths = (protocol["low"], protocol["high"])
        for warmup in range(protocol["warmup"]):
            for implementation in implementations[:: 1 if warmup % 2 == 0 else -1]:
                for length in lengths:
                    runner.sync(functions[implementation][length](value))
        samples = {key: [] for key in implementations}
        per_process = {key: [] for key in implementations}
        for trial in range(protocol["trials"]):
            times = {key: {} for key in implementations}
            for implementation in implementations[:: 1 if trial % 2 == 0 else -1]:
                for length in lengths[:: 1 if trial % 2 == 0 else -1]:
                    runner.barrier()
                    start = time.perf_counter()
                    result = functions[implementation][length](value)
                    runner.sync(result)
                    times[implementation][length] = runner.gather(
                        time.perf_counter() - start
                    )
            for implementation in implementations:
                all_times = [times[implementation][length] for length in lengths]
                per_process[implementation].append(all_times)
                samples[implementation].append([max(item) for item in all_times])
        for implementation in implementations:
            case["arms"][implementation].update(
                summarize(samples[implementation], *lengths)
            )
            case["arms"][implementation]["per_process_seconds"] = per_process[
                implementation
            ]
        document["cases"].append(case)
        if runner.rank == 0:
            write_json(output / "results.json", document)
            print(
                json.dumps(
                    {
                        "shape": [batch, width],
                        "dtype": dtype,
                        "median_us": {
                            arm: row["median_us"] for arm, row in case["arms"].items()
                        },
                    }
                ),
                flush=True,
            )
        del value, functions
    document["complete"] = True
    if runner.rank == 0:
        write_json(output / "results.json", document)
    runner.barrier()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    measurement = sub.add_parser("run")
    measurement.add_argument("--framework", choices=("jax", "torch"), required=True)
    measurement.add_argument("--output", required=True)
    measurement.add_argument("--smoke", action="store_true")
    measurement.add_argument(
        "--batches", help="diagnostic subset; full admission requires all seven batches"
    )
    measurement.add_argument("--correctness-extras", action="store_true")
    measurement.add_argument(
        "--manifest", help="independent automatic dispatch validation (Torch only)"
    )
    measurement.add_argument("--save-executables", action="store_true")
    measurement.add_argument(
        "--reload-executables",
        help="validate saved Torch executables in a fresh launch",
    )
    measurement.add_argument("--distributed", action="store_true")
    measurement.add_argument("--coordinator-address")
    measurement.add_argument("--num-processes", type=int)
    measurement.add_argument("--process-id", type=int)
    admission = sub.add_parser("qualify")
    admission.add_argument("--primary", required=True)
    admission.add_argument("--confirmation", required=True)
    admission.add_argument("--output", required=True)
    admission.add_argument("--report", required=True)
    return result


def main():
    args = parser().parse_args()
    if args.command == "qualify":
        manifest, report = qualify(read_run(args.primary), read_run(args.confirmation))
        write_json(Path(args.output), manifest)
        write_json(Path(args.report), report)
    else:
        run(args)


if __name__ == "__main__":
    main()
