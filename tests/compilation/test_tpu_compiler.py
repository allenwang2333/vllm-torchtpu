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
import sys
from unittest.mock import MagicMock, patch

import torch
import torch.fx as fx


class TestTpuCompilerCache:

    def test_env_override_mapping(self):
        """Verify that VLLM_XLA_CACHE_PATH sets the corresponding native variables."""
        env_mock = {
            "VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_env",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Reload env_override to re-execute its top-level statements
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get(
                "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
            ) == "/tmp/test_xla_cache_env/torch_tpu_tier3"
            assert os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"
                                  ) == "tpu_tier2_cache"

    def test_env_override_no_clobber(self):
        """Verify that existing native variables are not overwritten by env_override."""
        env_mock = {
            "VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_env",
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
            "/custom/tier3/path",
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "custom_tier2",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get(
                "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
            ) == "/custom/tier3/path"
            assert os.environ.get(
                "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE") == "custom_tier2"

    def test_compiler_cache_dir_relocation(self):
        """Verify that TpuCompilerAdaptor relocates cache_dir under VLLM_XLA_CACHE_PATH."""
        with patch.dict(os.environ,
                        {"VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_dir"}):
            # Import adaptor
            from vllm_torchtpu.compilation.tpu_compiler import \
                TpuCompilerAdaptor

            compiler = TpuCompilerAdaptor()

            # Simulate initialize_cache call with default path
            from vllm.envs import VLLM_CACHE_ROOT
            default_cache_dir = os.path.join(VLLM_CACHE_ROOT,
                                             "torch_compile_cache")

            compiler.initialize_cache(cache_dir=default_cache_dir)

            expected_cache_dir = os.path.join("/tmp/test_xla_cache_dir",
                                              "torch_compile_cache")
            assert compiler.cache_dir == expected_cache_dir

    def test_compiler_cache_dir_fallback_on_error(self):
        """Verify that initialize_cache falls back to default cache_dir on exception."""
        with patch.dict(os.environ,
                        {"VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_dir"}):
            from vllm_torchtpu.compilation.tpu_compiler import \
                TpuCompilerAdaptor
            compiler = TpuCompilerAdaptor()

            # Force relpath exception by passing incompatible path types or mocking relpath
            with patch("os.path.relpath",
                       side_effect=ValueError("mock error")):
                compiler.initialize_cache(cache_dir="/original/cache/dir")
                # Should fallback to the original directory
                assert compiler.cache_dir == "/original/cache/dir"

    def test_ensure_tuple_output(self):
        """Verify that _ensure_tuple_output correctly wraps single-tensor FX graph outputs into tuples."""
        from vllm_torchtpu.compilation.tpu_compiler import _ensure_tuple_output

        def single_output_fn(x):
            return x * 2

        def multi_output_fn(x):
            return x * 2, x * 3

        # Single output graph
        g1 = fx.symbolic_trace(single_output_fn)
        g1_wrapped, was_wrapped1 = _ensure_tuple_output(g1)
        assert was_wrapped1
        # Verify it now returns a tuple
        output_node = next(n for n in g1_wrapped.graph.nodes
                           if n.op == "output")
        assert isinstance(output_node.args[0], (tuple, list))
        assert len(output_node.args[0]) == 1

        # Multi output graph
        g2 = fx.symbolic_trace(multi_output_fn)
        g2_wrapped, was_wrapped2 = _ensure_tuple_output(g2)
        assert not was_wrapped2

    @patch("vllm_torchtpu.compilation.tpu_compiler._tpu_backend")
    @patch("torch._dynamo.backends.common.aot_autograd")
    def test_compile_and_load_serialization(self, mock_aot_autograd,
                                            mock_backend, tmp_path):
        """Verify compiled executables are serialized to disk on cache miss and deserialized on hit."""
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        def sample_fn(x):
            return x * 2

        graph = fx.symbolic_trace(sample_fn)
        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(tmp_path))

        mock_exe = "mock_compiled_executable_content"
        mock_backend._compiled_executables = [mock_exe]
        mock_backend.return_value = lambda x: x * 2

        from torch._subclasses.fake_tensor import FakeTensorMode
        fake_mode = FakeTensorMode(shape_env=None)
        fake_tensor = fake_mode.from_tensor(torch.tensor([1.0]))
        example_inputs = [fake_tensor]
        cache_key = "test_hash_key"

        with patch("torch._guards.tracing"):
            compiled_fn, handle = compiler.compile(
                graph=graph,
                example_inputs=example_inputs,
                compiler_config={},
                compile_range=(0, 1),
                key=cache_key)

        assert handle is not None
        saved_key, save_path, was_wrapped = handle
        assert saved_key == cache_key
        assert os.path.exists(save_path)
        assert was_wrapped is True

        import pickle
        with open(save_path, "rb") as f:
            loaded_exe = pickle.load(f)
        assert loaded_exe == mock_exe

        # Configure mock aot_autograd
        mock_aot_autograd.return_value = lambda g, inputs: lambda x: "loaded_value"

        with patch("torch._guards.tracing"):
            loaded_fn = compiler.load(handle=handle,
                                      graph=graph,
                                      example_inputs=example_inputs,
                                      graph_index=0,
                                      compile_range=(0, 1))

        assert loaded_fn is not None
        # test unwrapping (wrapper turns single-element tuple return into value)
        # However, our mock_aot_autograd just returns "loaded_value" directly, so unwrapping sees "loaded_value".
        # Because "loaded_value" is a string, not a tuple/list, the wrapper just returns it.
        assert loaded_fn(1) == "loaded_value"

    @patch("vllm_torchtpu.compilation.tpu_compiler.envs")
    def test_compiler_cache_key_invalidation(self, mock_envs, tmp_path):
        """Verify that modifying runtime cache key files triggers recompilation by changing the hash."""
        import vllm_torchtpu.compilation.tpu_compiler as tpu_compiler_mod
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        mock_file = MagicMock()
        mock_file.relative_to.return_value.as_posix.return_value = "mock_kernel.py"
        initial_content = b"initial content"
        mock_file.read_bytes.return_value = initial_content

        with patch.object(tpu_compiler_mod,
                          "_iter_runtime_cache_key_files",
                          return_value=[mock_file]):
            compiler = TpuCompilerAdaptor()

            mock_config = MagicMock()
            mock_config.cache_config.gpu_memory_utilization = 0.9
            mock_config.scheduler_config.max_num_batched_tokens = 2048
            mock_config.cache_config.num_gpu_blocks_override = None
            mock_config.speculative_config = None
            mock_envs.TPU_KV_CACHE_HEADROOM_MIB = 0

            # Since torch_tpu might not be present in the mock, patch sys.modules if needed
            with patch.dict(sys.modules,
                            {'torch_tpu': MagicMock(__version__="1.0.0")}):
                hash1 = compiler.compute_hash(mock_config)

                # Modify file content
                mock_file.read_bytes.return_value = b"modified content"

                hash2 = compiler.compute_hash(mock_config)

                # Restore original file content to test config change
                mock_file.read_bytes.return_value = initial_content

                # Modify config
                mock_config.cache_config.gpu_memory_utilization = 0.8
                hash3 = compiler.compute_hash(mock_config)

            # Hash should change because the file content changed
            assert hash1 != hash2
            # Hash should change because the config changed
            assert hash1 != hash3
