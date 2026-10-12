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

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.env_override import _TPU_VMEM_BYTES_PER_TENSOR_CORE


class TestTpuCompilerCache:
    @staticmethod
    def _config(*, max_num_seqs=16, num_gpu_blocks=1024):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=num_gpu_blocks,
                num_gpu_blocks_override=None,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
                max_num_seqs=max_num_seqs,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            speculative_config=None,
        )

    def test_tpu_hash_covers_derived_shapes(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            base = compute_tpu_compilation_hash(self._config())
            different_batch = compute_tpu_compilation_hash(
                self._config(max_num_seqs=32)
            )
            different_kv = compute_tpu_compilation_hash(
                self._config(num_gpu_blocks=2048)
            )

        assert base != different_batch
        assert base != different_kv

    def test_tpu_hash_covers_plugin_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "0"}):
                disabled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "4096"}):
                enabled = compute_tpu_compilation_hash(self._config())

        assert disabled != enabled

    def test_tpu_hash_ignores_the_phased_profiler_switch(self):
        """Selecting a profiler must not bust the compile cache: otherwise a
        profiled run recompiles everything, and so does the next plain one."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "false"}):
                unprofiled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "true"}):
                phased = compute_tpu_compilation_hash(self._config())

        assert unprofiled == phased

    def test_tpu_hash_covers_native_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/a"}):
                first = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/b"}):
                second = compute_tpu_compilation_hash(self._config())

        assert first != second

    def test_tpu_hash_covers_compiler_versions(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        versions = {
            "torch-tpu": "1",
            "jax": "1",
            "jaxlib": "1",
            "libtpu": "1",
        }
        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch(
                "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                side_effect=versions.get,
            ):
                base = compute_tpu_compilation_hash(self._config())

            for package in versions:
                changed_versions = {**versions, package: "2"}
                with patch(
                    "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                    side_effect=changed_versions.get,
                ):
                    changed = compute_tpu_compilation_hash(self._config())
                assert base != changed

    def test_tpu_hash_covers_effective_data_parallel_size(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"TORCH_TPU_DP_SIZE": "1"}):
                dp1 = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"TORCH_TPU_DP_SIZE": "4"}):
                dp4 = compute_tpu_compilation_hash(self._config())

        assert dp1 != dp4

    def test_runtime_hash_covers_custom_op_boundaries(self):
        from vllm_torchtpu.compilation import tpu_compiler

        source_root = Path(tpu_compiler.__file__).resolve().parents[2]
        hashed = {
            path.relative_to(source_root).as_posix()
            for path in tpu_compiler._iter_runtime_cache_key_files(source_root)
        }
        custom_op_boundaries = {
            path.relative_to(source_root).as_posix()
            for path in (source_root / "vllm_torchtpu").rglob("*.py")
            if "pallas.jax_op" in path.read_text()
        }
        assert custom_op_boundaries <= hashed
        assert {
            "vllm_torchtpu/kernels/quantized_matmul/blockwise_kernel.py",
            "vllm_torchtpu/kernels/pool_adapters.py",
            "vllm_torchtpu/models/vllm/kimi_k3/collective_ops.py",
            "vllm_torchtpu/models/vllm/kimi_k3/layers.py",
        } <= hashed

    def test_serving_only_kv_knobs_stay_out_of_the_compile_cache_key(self):
        """Serving-time knobs must stay out of the compile-cache key.

        Every `envs.py` entry not in `_TPU_COMPILE_ENV_IGNORED` is hashed, so
        a knob added there recompiles the world once on upgrade and again
        every time an operator retunes it.
        """
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED,
            _tpu_compile_env_factors,
        )

        serving_only = {
            "TPU_IPC_SOCKET_DIR",
            "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS",
            "TPU_KV_COORD_EXECUTOR_MAX_WORKERS",
            "TPU_KV_LATENCY_LOG_INTERVAL",
            "TPU_KV_PIN_SHM",
            "TPU_KV_SHM_POOL_GB",
            "TPU_KV_STAGE_WAITER_POOL_SIZE",
            "TPU_KV_STAGE_WAIT_TIMEOUT_SECS",
            "TPU_KV_TRANSFER_CHANNEL_NUMBER",
            "TPU_KV_TRANSFER_NAMESPACE",
            "TPU_KV_TRANSFER_PORT",
            "TPU_KV_WARMUP_ENABLED",
            "TPU_NODE_ID",
            "TPU_P2P_WAIT_PULL_TIMEOUT",
            "TPU_RAIDEN_INLINE_LOAD",
            "TPU_RAIDEN_POOL_STAGING_LEASES",
            "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT",
            "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S",
            "TPU_RAIDEN_STAGE3_STATUS_PROBE_S",
            "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S",
            "TPU_RAIDEN_TRANSFER_NUM_SLOTS",
            "TPU_SIDE_CHANNEL_PORT",
            "TPU_USE_RAIDEN_CONNECTOR",
            "DEBUG_TPU_LOCAL_RANK_OFFSET",
            "RAIDEN_DISABLE_SINGLETON_WORKER",
            "RAIDEN_SHM_KEY",
            "SPEC_WARMUP",
            "TORCH_TPU_BASE_PORT",
            "TORCH_TPU_MP_RENDEZVOUS_PORT",
            "TPU_LOCAL_RANK_OFFSET",
            "TPU_SHARDED_LOAD_SYNC_EVERY",
            "VLLM_TORCHTPU_IPC_KEY",
            "VLLM_TPU_DEBUG_PCP_LAYOUT",
            "VLLM_TPU_OFFLOAD_SAVE_RETRIES",
            "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S",
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        }

        assert serving_only <= _TPU_COMPILE_ENV_IGNORED
        assert not serving_only & set(_tpu_compile_env_factors())

    def test_every_registry_entry_is_hashed_or_deliberately_ignored(self):
        """The two sets must partition the registry, so a new entry forces a
        deliberate hash-or-ignore choice, and a stale exemption cannot sit
        waiting for a later knob to reuse its name."""
        from vllm_torchtpu import envs
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED,
            _tpu_compile_env_factors,
        )

        registry = set(envs.environment_variables)
        stale = _TPU_COMPILE_ENV_IGNORED - registry
        assert not stale, f"ignored but no longer in envs.py: {sorted(stale)}"
        assert set(_tpu_compile_env_factors()) | _TPU_COMPILE_ENV_IGNORED == registry

    def test_aot_hash_includes_tpu_compiler_hash(self):
        from vllm.compilation import caching

        from vllm_torchtpu import _patch_vllm_aot_compile_cache_key

        def upstream_factors(_):
            return ["upstream"]

        with patch.object(caching, "aot_compile_hash_factors", upstream_factors):
            _patch_vllm_aot_compile_cache_key()
            with patch(
                "vllm_torchtpu.compilation.tpu_compiler.compute_tpu_compilation_hash",
                return_value="tpu",
            ):
                factors = caching.aot_compile_hash_factors(self._config())

        assert factors == ["upstream", "tpu"]

    def test_env_override_mapping(self):
        """Verify that setting TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT sets
        TORCH_TPU_TIER2_COMPILATION_CACHE default.
        """
        env_mock = {
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/test_tier3_cache_env",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Reload env_override to re-execute its top-level statements
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert (
                os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") == "tpu_tier2_cache"
            )

    @staticmethod
    def _sc_offload_flags() -> list[str]:
        return [
            token
            for token in os.environ.get("LIBTPU_INIT_ARGS", "").split()
            if "offload_min_size_in_bytes" in token
        ]

    @pytest.mark.parametrize(
        "value,expected_threshold",
        [
            ("12345", 12345),
            ("auto", _TPU_VMEM_BYTES_PER_TENSOR_CORE["v6e"]),
            ("", _TPU_VMEM_BYTES_PER_TENSOR_CORE["v6e"]),
            ("0", None),
        ],
    )
    def test_env_override_sc_offload_threshold(self, value, expected_threshold):
        """An explicit value is used as given, unset/empty/auto resolves
        from the chip family, and 0 sets no flag at all."""
        env_mock = {
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            expected = (
                []
                if expected_threshold is None
                else [
                    f"--xla_tpu_sparse_core_{op}_offload_min_size_in_bytes="
                    f"{expected_threshold}"
                    for op in ("all_reduce", "all_gather")
                ]
            )
            assert self._sc_offload_flags() == expected

    @pytest.mark.parametrize("value", ["banana", "-1"])
    def test_env_override_sc_offload_bad_value_raises(self, value):
        """A value that is not "auto" or a non-negative integer stops the
        package import with an error that names the variable, rather than
        silently running without the SC-offload flag."""
        env_mock = {
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            with pytest.raises(
                ValueError, match="SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES"
            ):
                importlib.reload(vllm_torchtpu.env_override)

    def test_envs_module_imports_only_the_standard_library(self):
        """env_override.py imports envs while the package's own __init__ is
        still running, which is only safe while envs.py depends on nothing
        in this package."""
        import ast
        import sys

        import vllm_torchtpu.envs

        tree = ast.parse(Path(vllm_torchtpu.envs.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= sys.stdlib_module_names, (
            f"envs.py imports non-stdlib modules: "
            f"{sorted(imported - sys.stdlib_module_names)}"
        )

    def test_env_override_disables_breakable_cudagraph(self):
        """Verify that the TPU plugin always disables breakable cudagraph."""
        with patch.dict(os.environ, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"

    def test_env_override_no_clobber(self):
        """Verify that existing native variables are not overwritten by env_override."""
        env_mock = {
            "VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_env",
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "custom_tier2",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "custom_tier2",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert (
                os.environ.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
                == "/custom/tier3/path"
            )
            assert (
                os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
                == "/custom/tier3/path"
            )

            assert (
                os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE")
                == "custom_tier2"
            )
            assert os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") == "custom_tier2"

    def test_tpu_compilation_handle_serialization(self):
        """Round-trip the artifact payload TpuCompilerAdaptor.compile writes."""
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilationHandle

        handle = TpuCompilationHandle(
            key="test_key_123",
            entry="mock_entry",
            signature=(("torch.int32", (16,)),),
            was_wrapped=True,
        )
        loaded = pickle.loads(pickle.dumps(handle))
        assert isinstance(loaded, TpuCompilationHandle)
        assert loaded.key == "test_key_123"
        assert loaded.entry == "mock_entry"
        assert loaded.signature == (("torch.int32", (16,)),)
        assert loaded.was_wrapped is True

    def test_replay_refuses_artifacts_it_cannot_vouch_for(self, tmp_path):
        """Stale or damaged artifacts recompile instead of being replayed.

        The older format stored a bare executable picked off a shared list by
        index, with nothing tying it to the graph it was saved under, so
        replaying one can run the wrong program. Restoring such a pickle leaves
        `entry` at its class default of None, and it is refused on that basis.
        """
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilationHandle,
            TpuCompilerAdaptor,
        )

        compiler = TpuCompilerAdaptor()
        path = tmp_path / "artifact"

        legacy = object.__new__(TpuCompilationHandle)
        legacy.__dict__.update(key="k", executable="mock_exe")
        assert legacy.entry is None
        path.write_bytes(pickle.dumps(legacy))
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        path.write_bytes(pickle.dumps({"key": "k", "executable": "mock_exe"}))
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        path.write_bytes(b"not a pickle")
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        assert (
            compiler._replay(str(tmp_path / "missing"), graph=None, was_wrapped=False)
            is None
        )

    def test_compiler_initialize_cache(self):
        """Verify that TpuCompilerAdaptor sets cache_dir accurately."""
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir="/tmp/test_cache_dir")
        assert compiler.cache_dir == "/tmp/test_cache_dir"

    def test_run_backend_restarts_scheduling_group_ids(self, monkeypatch):
        """A bucket's HLO, and so its cache key, must not depend on what the
        rank compiled before it."""
        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        firsts = []

        def fake_backend(graph, example_inputs):
            firsts.append(config.reserve_scheduling_group_ids(3))
            return lambda *args: args

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        compiler = TpuCompilerAdaptor()
        compiler._run_backend(None, [])
        compiler._run_backend(None, [])
        assert firsts == [1, 1]

    @staticmethod
    def _peer_compilers(tmp_path, monkeypatch, n=2):
        """Compilers for local ranks 0..n-1 of one host, with a fake backend
        that records which ranks really compiled."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiled, replayed = [], []

        class Compiled:
            def __init__(self, rank):
                self.rank = rank

            def __call__(self, *args):
                return (self.rank,)

            def serialize(self):
                return f"entry-from-rank-{self.rank}"

        def compilers():
            for rank in range(n):
                compiler = TpuCompilerAdaptor()
                compiler.initialize_cache(
                    cache_dir=str(tmp_path / "hash" / f"rank_{rank}_0" / "backbone")
                )
                compiler.rank = rank
                yield compiler

        current = {}

        def fake_backend(graph, example_inputs):
            compiled.append(current["rank"])
            return Compiled(current["rank"])

        def fake_deserialize(entry):
            replayed.append(entry)
            return lambda *args: (entry,)

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        monkeypatch.setattr(tpu_compiler, "_deserialize_entry", fake_deserialize)
        monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 0.01)

        def compile(compiler, graph=None):
            current["rank"] = compiler.rank
            if graph is None:
                graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
            return compiler.compile(
                graph, [], {}, Range(start=1, end=4096), key="artifact_bucket"
            )

        return list(compilers()), compile, compiled, replayed

    def test_a_peer_artifact_is_replayed_instead_of_compiled(
        self, tmp_path, monkeypatch
    ):
        """The second rank of a host replays what the first one saved, and its
        own cache entry is the same file, not a copy."""
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a)
        fn, handle = compile(b)

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert path == os.path.join(b.cache_dir, key)
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))
        assert sorted(os.listdir(tmp_path / "hash")) == ["rank_0_0", "rank_1_0"]
        assert os.listdir(b.cache_dir) == [key]

    def test_a_peer_artifact_of_another_graph_is_not_replayed(
        self, tmp_path, monkeypatch
    ):
        """vLLM's key names a bucket, not a graph: a rank that traced
        something else, e.g. with its rank baked in, compiles its own."""
        import torch

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, torch.fx.symbolic_trace(lambda x: x * 2 + 1))
        compile(b, torch.fx.symbolic_trace(lambda x: x * 2 + 2))

        assert compiled == [0, 1]
        assert replayed == []
        assert not os.path.samefile(
            os.path.join(a.cache_dir, "artifact_bucket"),
            os.path.join(b.cache_dir, "artifact_bucket"),
        )

    def test_a_rank_waits_for_the_peer_that_compiles_a_bucket(
        self, tmp_path, monkeypatch
    ):
        import threading

        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        result = {}

        def waiter():
            with wait_for_peer_artifact(30.0):
                result["fn"], _ = compile(b)

        thread = threading.Thread(target=waiter)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        compile(a)
        thread.join(30.0)

        assert not thread.is_alive()
        assert compiled == [0]
        assert result["fn"]() == "entry-from-rank-0"

    def test_a_rank_compiles_itself_when_no_peer_saves_in_time(
        self, tmp_path, monkeypatch
    ):
        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (_, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        with wait_for_peer_artifact(0.05):
            compile(b)
        assert compiled == [1]
        assert replayed == []

    def test_no_peer_replay_outside_a_rank_cache_dir(self, tmp_path, monkeypatch):
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        a.initialize_cache(cache_dir=str(tmp_path / "x" / "backbone"))
        b.initialize_cache(cache_dir=str(tmp_path / "y" / "backbone"))
        compile(a)
        compile(b)
        assert compiled == [0, 1]

    def test_peer_replay_off_compiles_every_rank_itself(self, tmp_path, monkeypatch):
        """TPU_PRECOMPILE_PEER_REPLAY=0: no peer lookup, no wait, no link,
        and no fingerprint recorded; each rank compiles and saves its own."""
        import pickle

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilerAdaptor,
            wait_for_peer_artifact,
        )

        monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "0")
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )

        def no_peer_lookup(*args, **kwargs):
            raise AssertionError("looked for a peer artifact")

        monkeypatch.setattr(TpuCompilerAdaptor, "_peer_paths", no_peer_lookup)
        monkeypatch.setattr(TpuCompilerAdaptor, "_link_artifact", no_peer_lookup)
        monkeypatch.setattr(tpu_compiler, "_graph_fingerprint", no_peer_lookup)

        compile(a)
        with wait_for_peer_artifact(30.0):
            _, handle = compile(b)

        assert compiled == [0, 1]
        assert replayed == []
        key, path, _ = handle
        assert not os.path.samefile(path, os.path.join(a.cache_dir, key))
        with open(path, "rb") as f:
            assert pickle.load(f).fingerprint is None


def _graph_with_layers(layers, nodes_per_layer=2, path="L['self'].layers.{}"):
    """An FX graph whose nodes carry the nn_module_stack Dynamo records, with
    `nodes_per_layer[i]` (or the same count for every) nodes in layer i."""
    import torch

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    for layer in range(layers):
        count = (
            nodes_per_layer[layer]
            if isinstance(nodes_per_layer, list)
            else nodes_per_layer
        )
        for _ in range(count):
            x = graph.call_function(torch.relu, (x,))
            x.meta["nn_module_stack"] = {
                f"L__self___layers_{layer}": (path.format(layer), torch.nn.Module)
            }
    graph.output(x)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _eager_artifact(gm):
    """A torch_tpu CompiledArtifact that runs `gm` on the CPU, picklable like
    a real executable."""
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        CompiledArtifact,
    )

    global _EagerArtifact
    if "_EagerArtifact" not in globals():

        class _EagerArtifact(CompiledArtifact):
            def __init__(self, gm):
                self.gm = gm

            def __call__(self, *args, **kwargs):
                if len(args) == 1 and isinstance(args[0], list):
                    args = args[0]
                return self.gm(*args)

            def __reduce__(self):
                return (_eager_artifact, (self.gm,))

            @property
            def updates_default_generator_state(self):
                return False

            def fingerprint(self):
                return ""

    return _EagerArtifact(gm)


class TestHostSplit:
    """TPU_PRECOMPILE_SPLIT_MIN_TOKENS: one bucket compiled by every rank of
    a host, one piece each, cut between decoder layers."""

    @pytest.mark.parametrize(
        "path, layer",
        [
            ("L['self'].layers.12", 12),
            ("L['self'].model.layers[12].mlp", 12),
            ("getattr(L['self'].layers, '12')", 12),
            ("L['self']._modules['layers']._modules['12']", 12),
            ("L['self'].players.12", None),
            ("L['self'].norm", None),
        ],
    )
    def test_node_layer_reads_the_decoder_layer(self, path, layer):
        import torch

        from vllm_torchtpu.compilation import host_split

        node = SimpleNamespace(meta={"nn_module_stack": {"k": (path, torch.nn.Module)}})
        assert host_split.node_layer(node) == layer

    def test_layers_are_dealt_in_contiguous_runs_of_about_equal_size(self):
        from vllm_torchtpu.compilation import host_split

        gm = _graph_with_layers(6)
        assert host_split.layer_groups(gm.graph, 2) == {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
        }
        # Balanced by node count, not by layer count.
        gm = _graph_with_layers(4, nodes_per_layer=[6, 1, 1, 4])
        assert host_split.layer_groups(gm.graph, 2) == {0: 0, 1: 1, 2: 1, 3: 1}
        # Never more pieces than layers.
        gm = _graph_with_layers(3)
        assert sorted(set(host_split.layer_groups(gm.graph, 8).values())) == [0, 1, 2]

    def test_a_graph_without_layers_is_not_split(self):
        from vllm_torchtpu.compilation import host_split

        assert host_split.layer_groups(_graph_with_layers(1).graph, 8) == {}
        assert not host_split.can_split(_graph_with_layers(1), 8)
        assert not host_split.can_split(object(), 8)
        assert host_split.can_split(_graph_with_layers(2), 8)

    def test_a_wait_stays_with_its_collective(self):
        """A cut never separates a collective's wait_tensor from it, and the
        piece never moves inside an SPMD safe region."""
        import torch

        from vllm_torchtpu.compilation import host_split

        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        def at(node, layer, safe=False):
            node.meta["nn_module_stack"] = {"k": (f"layers.{layer}", None)}
            if safe:
                node.meta["custom"] = {"spmd_safe": True}
            return node

        a = at(graph.call_function(torch.relu, (x,)), 0)
        coll = at(
            graph.call_function(
                torch.ops._c10d_functional.all_reduce.default, (a, "sum", "0")
            ),
            0,
        )
        wait = at(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, (coll,)
            ),
            1,
        )
        safe = at(graph.call_function(torch.relu, (wait,)), 1, safe=True)
        b = at(graph.call_function(torch.relu, (safe,)), 1)
        graph.output(b)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        ids = host_split.piece_partitions(gm, lambda node: 0, {0: 0, 1: 1})
        assert ids[a] == ids[coll] == ids[wait] == ids[safe] == 0
        assert ids[b] == 1

    def test_a_piece_fingerprint_reads_no_tensor_values(self, monkeypatch):
        """Pieces are fingerprinted inside torch_tpu's FX trace, where reading
        a device tensor fails; constants count by dtype and shape, and the
        whole graph's fingerprint pins their values."""
        import torch

        from vllm_torchtpu.compilation import host_split

        root = torch.nn.Module()
        root.register_buffer("const", torch.ones(4))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        c = graph.get_attr("const")
        graph.output(graph.call_function(torch.add, (x, c)))
        gm = torch.fx.GraphModule(root, graph)
        inputs = [torch.zeros(4)]

        def no_read(*args, **kwargs):
            raise RuntimeError("cannot record EventSnapshot during an FX trace")

        monkeypatch.setattr(torch.Tensor, "to", no_read)
        monkeypatch.setattr(torch.Tensor, "numpy", no_read)
        a = host_split.piece_fingerprint(gm, inputs, "graph-a")
        assert a == host_split.piece_fingerprint(gm, inputs, "graph-a")
        assert a != host_split.piece_fingerprint(gm, inputs, "graph-b")
        assert a != host_split.piece_fingerprint(gm, [torch.zeros(8)], "graph-a")

    @staticmethod
    def _model_graph(layers=4):
        """A Dynamo graph of a small decoder stack, as vLLM hands it over."""
        import torch

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.proj(x)) + x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(layers)])
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.norm(x)

        torch.manual_seed(0)
        model = Model()
        captured = []

        def capture(gm, example_inputs):
            captured.append((gm, example_inputs))
            return gm.forward

        x = torch.randn(4, 8)
        with torch.no_grad():
            torch._dynamo.reset()
            torch.compile(model, backend=capture, dynamic=False)(x)
            expected = model(x)
        gm, example_inputs = captured[0]
        return gm, list(example_inputs), expected

    @staticmethod
    def _fake_lowering(monkeypatch):
        """torch_tpu's whole pipeline -- aot_autograd, SplitCompiler,
        serialization -- with only FX->MLIR->PjRt replaced by running the
        partition on the CPU. Returns the node counts of what was lowered."""
        from torch_tpu._internal.compile import compiler

        lowered = []

        def lower(self, gm, example_inputs, *args, **kwargs):
            lowered.append(len(gm.graph.nodes))
            return _eager_artifact(gm)

        monkeypatch.setattr(compiler.StaticCompiler, "_compile", lower)
        return lowered

    @staticmethod
    def _compile(tmp_path, rank, gm, example_inputs, wait_s, pieces=2):
        """Rank `rank` of a host of `pieces` compiles the bucket in a split.

        Under no_grad, as vLLM serves: aot_autograd then traces an inference
        graph, whose artifact can be saved.
        """
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import host_split
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(
            cache_dir=str(tmp_path / "hash" / f"rank_{rank}_0" / "backbone")
        )
        with torch.no_grad(), host_split.split_across_host(pieces, rank, wait_s):
            fn, handle = compiler.compile(
                gm, example_inputs, {}, Range(4, 4), key="artifact_bucket"
            )
        return compiler, fn, handle

    @staticmethod
    def _out(result):
        return result if not isinstance(result, (tuple, list)) else result[0]

    def test_each_rank_compiles_its_piece_and_replays_the_others(
        self, tmp_path, monkeypatch
    ):
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import host_split

        monkeypatch.setattr(host_split, "_POLL_S", 0.01)
        lowered = self._fake_lowering(monkeypatch)
        gm, example_inputs, expected = self._model_graph()

        # Rank 1 first, with no wait: it compiles its own piece, then piece 0
        # too, since rank 0 has not saved it.
        _, fn1, handle1 = self._compile(tmp_path, 1, gm, example_inputs, 0.0)
        assert len(lowered) == 2
        assert torch.allclose(self._out(fn1(*example_inputs)), expected)
        # Its whole bucket is in its cache dir as always; hide it so rank 0
        # cannot replay the bucket whole and has to assemble it from pieces.
        os.remove(handle1[1])

        lowered.clear()
        compiler0, fn0, handle0 = self._compile(tmp_path, 0, gm, example_inputs, 30.0)
        assert len(lowered) == 1, "rank 0 lowered more than its own piece"
        assert torch.allclose(self._out(fn0(*example_inputs)), expected)
        assert sorted(os.listdir(compiler0.cache_dir)) == [
            "artifact_bucket",
            "artifact_bucket.part0of2",
        ]

        # A warm start loads the assembled bucket unchanged.
        lowered.clear()
        loaded = compiler0.load(handle0, gm, example_inputs, 0, Range(4, 4))
        assert lowered == []
        assert torch.allclose(self._out(loaded(*example_inputs)), expected)

    def test_a_rank_waits_for_the_piece_a_peer_compiles(self, tmp_path, monkeypatch):
        import threading

        import torch

        from vllm_torchtpu.compilation import host_split

        monkeypatch.setattr(host_split, "_POLL_S", 0.01)
        self._fake_lowering(monkeypatch)
        gm, example_inputs, expected = self._model_graph()

        # Rank 1 publishes its piece only once rank 0 is already waiting.
        started = threading.Event()
        original_resolve = host_split._Split.resolve

        def resolve(self, original):
            started.set()
            return original_resolve(self, original)

        monkeypatch.setattr(host_split._Split, "resolve", resolve)
        result = {}

        def rank0():
            result["fn"] = self._compile(tmp_path, 0, gm, example_inputs, 30.0)[1]

        thread = threading.Thread(target=rank0)
        thread.start()
        assert started.wait(30.0)
        assert thread.is_alive()
        self._compile(tmp_path, 1, gm, example_inputs, 0.0)
        thread.join(30.0)

        assert not thread.is_alive()
        assert torch.allclose(self._out(result["fn"](*example_inputs)), expected)

    def test_a_piece_of_another_graph_is_compiled_here(self, tmp_path, monkeypatch):
        """A peer's piece built from a different graph is not replayed, and
        is not waited on either."""
        import time

        import torch

        from vllm_torchtpu.compilation import host_split

        monkeypatch.setattr(host_split, "_POLL_S", 0.01)
        lowered = self._fake_lowering(monkeypatch)
        gm, example_inputs, expected = self._model_graph()
        _, _, handle1 = self._compile(tmp_path, 1, gm, example_inputs, 0.0)
        os.remove(handle1[1])
        # Rank 0 sees its pieces as built from another graph than rank 1's.
        real = host_split.piece_fingerprint
        monkeypatch.setattr(
            host_split,
            "piece_fingerprint",
            lambda gm, inputs, graph_fp: real(gm, inputs, "rank-0:" + graph_fp),
        )

        lowered.clear()
        start = time.monotonic()
        _, fn0, _ = self._compile(tmp_path, 0, gm, example_inputs, 30.0)
        assert time.monotonic() - start < 20.0
        assert len(lowered) == 2
        assert torch.allclose(self._out(fn0(*example_inputs)), expected)

    def test_a_piece_its_owner_could_not_save_is_compiled_here(
        self, tmp_path, monkeypatch
    ):
        import pickle
        import time

        from vllm_torchtpu.compilation import host_split

        monkeypatch.setattr(host_split, "_POLL_S", 0.01)
        lowered = self._fake_lowering(monkeypatch)
        gm, example_inputs, _ = self._model_graph()
        real_dumps = pickle.dumps

        def unpicklable(obj, *args, **kwargs):
            raise pickle.PicklingError("no")

        monkeypatch.setattr(host_split.pickle, "dumps", unpicklable)
        _, _, handle1 = self._compile(tmp_path, 1, gm, example_inputs, 0.0)
        monkeypatch.setattr(host_split.pickle, "dumps", real_dumps)
        os.remove(handle1[1])

        lowered.clear()
        start = time.monotonic()
        self._compile(tmp_path, 0, gm, example_inputs, 30.0)
        assert time.monotonic() - start < 20.0
        assert len(lowered) == 2

    def test_the_cache_entry_is_pickled_outside_the_fx_trace(
        self, tmp_path, monkeypatch
    ):
        """aot_autograd pickles the entry of a split executable, device
        constants included, in torch_tpu's eager mode from outside the
        backend: inside the FX trace a device-to-host copy fails."""
        import contextlib

        from torch._functorch._aot_autograd import autograd_cache

        from vllm_torchtpu.compilation import host_split

        host_split._install()
        pickled_outside = []
        state = {"outside": False}

        @contextlib.contextmanager
        def outside_trace(self):
            state["outside"] = True
            try:
                yield
            finally:
                state["outside"] = False

        monkeypatch.setattr(host_split._Split, "outside_trace", outside_trace)

        def original(*args, **kwargs):
            pickled_outside.append(state["outside"])
            return b""

        monkeypatch.setitem(host_split._originals, "pickle_entry", original)
        installed = autograd_cache.AOTAutogradCache._pickle_entry

        installed(None, False)
        split = host_split._Split(host_split.Plan(2, 0, 0.0), "k", None, "r")
        monkeypatch.setattr(host_split._state, "split", split, raising=False)
        installed(None, False)
        assert pickled_outside == [False, True]

    def test_outside_a_split_the_backend_is_untouched(self, tmp_path, monkeypatch):
        """Once installed, the hooks do nothing for a bucket built outside
        split_across_host: one partition, one executable, no pieces."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import host_split
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        lowered = self._fake_lowering(monkeypatch)
        gm, example_inputs, _ = self._model_graph()
        host_split._install()
        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(
            cache_dir=str(tmp_path / "hash" / "rank_0_0" / "backbone")
        )
        with torch.no_grad():
            compiler.compile(gm, example_inputs, {}, Range(4, 4), key="artifact_bucket")
        assert len(lowered) == 1
        assert os.listdir(compiler.cache_dir) == ["artifact_bucket"]
