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
from unittest.mock import patch

import torch
from torch.utils import _pytree as pytree
from vllm.config.utils import Range


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


def test_exact_bucket_static_scalar_output_round_trip():
    """Static scalar piece outputs are reconstructed on the host."""

    class StaticScalarOutput(torch.nn.Module):

        def forward(self, value):
            return value + 1, 4096

    from vllm_torchtpu.compilation.tpu_compiler import (
        _restore_static_scalar_outputs, _strip_exact_static_scalar_outputs)

    graph = torch.fx.symbolic_trace(StaticScalarOutput())
    plan = _strip_exact_static_scalar_outputs(
        graph,
        Range(start=512, end=512),
    )
    assert plan is not None
    assert plan[1] == ((1, 4096), )

    output = graph.graph.find_nodes(op="output")[0]
    leaves, _ = pytree.tree_flatten(output.args[0])
    assert leaves[1] is None

    # AOTAutograd returns a list even when the Dynamo FX graph returns a tuple.
    restore = _restore_static_scalar_outputs(lambda value: [value, None], plan)
    value = torch.tensor([1.0])
    restored_value, restored_scalar = restore(value)
    assert restored_value is value
    assert restored_scalar == 4096
