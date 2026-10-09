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

"""TD8.46 / EXT-8AT: 100% statement and branch coverage tests for tpu_compiler,
tpu_rank_binding, runai_repull, flash_attention/tuned_params,
ragged_kv_cache_update, and ragged_paged_attention/v3/util.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import torch
import torch.fx as fx
from jax.sharding import PartitionSpec as P
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config.utils import Range

from tests.compilation.test_tpu_compiler import *  # noqa: F401, F403
from tests.worker.test_runai_repull import *  # noqa: F401, F403
from tests.worker.test_tpu_worker_rank_binding import *  # noqa: F401, F403
from vllm_torchtpu.compilation import tpu_compiler as tc
from vllm_torchtpu.kernels.flash_attention import tuned_params
from vllm_torchtpu.kernels.ragged_paged_attention.v2 import (
    ragged_kv_cache_update as rkv,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import util as rpa_util
from vllm_torchtpu.worker import tpu_rank_binding as binding


class _SingleOutMod(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _TupleOutMod(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + 1,)


class TestTpuCompilerFullCoverage:
    """100% statement and branch coverage for compilation/tpu_compiler.py."""

    def test_deserialize_entry_delegates_to_torch(self) -> None:
        with patch(
            "torch._functorch._aot_autograd.aot_autograd_result.deserialize_bundled_cache_entry",
            return_value="rebuilt_fn",
        ) as mock_deser:
            assert tc._deserialize_entry("raw_entry") == "rebuilt_fn"
            mock_deser.assert_called_once_with("raw_entry")

    def test_iter_runtime_cache_key_files_missing_and_iter_mode(
        self, tmp_path: Path
    ) -> None:
        # Missing paths trigger warning; create one file and one dir
        file_rel = "vllm_torchtpu/layers/adapter/attention.py"
        dir_rel = "vllm_torchtpu/layers/core"
        (tmp_path / file_rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file_rel).write_text("# file\n")
        (tmp_path / dir_rel).mkdir(parents=True, exist_ok=True)
        (tmp_path / dir_rel / "sub.py").write_text("# sub\n")

        with (
            patch.object(tc.envs, "TPU_KERNEL_ITER_MODE", True),
            patch.object(
                tc,
                "_reloadable_kernel_relpaths",
                return_value=[file_rel],
            ),
        ):
            files = tc._iter_runtime_cache_key_files(tmp_path)
        assert files == [tmp_path / dir_rel / "sub.py"]

    def test_reloadable_kernel_relpaths_all_branches(self, tmp_path: Path) -> None:
        inside_file = tmp_path / "pkg" / "mod_in.py"
        inside_file.parent.mkdir(parents=True, exist_ok=True)
        inside_file.write_text("# inside\n")
        outside_file = tmp_path.parent / "outside_mod_td846.py"

        specs = {
            "mod_no_spec": None,
            "mod_no_origin": SimpleNamespace(origin=None),
            "mod_outside": SimpleNamespace(origin=str(outside_file)),
            "mod_inside": SimpleNamespace(origin=str(inside_file)),
        }
        with (
            patch(
                "vllm_torchtpu.compilation.kernel_reload.default_reload_modules",
                return_value=list(specs),
            ),
            patch(
                "vllm_torchtpu.compilation.tpu_compiler.importlib.util.find_spec",
                side_effect=specs.get,
            ),
        ):
            relpaths = tc._reloadable_kernel_relpaths(tmp_path)
        assert relpaths == ["pkg/mod_in.py"]

    def test_env_factors_exception_and_hash_file_read_branches(
        self, tmp_path: Path
    ) -> None:
        bad_getters = {
            "TEST_BAD_ENV_VAR": MagicMock(side_effect=RuntimeError("boom")),
        }
        with patch.dict(tc.envs.environment_variables, bad_getters):
            factors = tc._tpu_compile_env_factors()
            assert "TEST_BAD_ENV_VAR" not in factors

        good_file = tmp_path / "good.py"
        good_file.write_text("x = 1\n")
        missing_file = tmp_path / "missing.py"

        cfg = SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=128,
            ),
            model_config=SimpleNamespace(max_model_len=1024),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=512,
                max_num_seqs=8,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            speculative_config=SimpleNamespace(
                model="draft",
                draft_tensor_parallel_size=1,
                num_speculative_tokens=2,
            ),
        )
        with (
            patch.object(tc, "_REPO_ROOT", tmp_path),
            patch.object(
                tc,
                "_iter_runtime_cache_key_files",
                return_value=[good_file, missing_file],
            ),
        ):
            adaptor = tc.TpuCompilerAdaptor()
            digest = adaptor.compute_hash(cfg)
            assert len(digest) == 10

    def test_ensure_tuple_output_and_unwrap_and_single_size(self) -> None:
        gm_single = fx.symbolic_trace(_SingleOutMod())
        gm_wrapped, was_wrapped = tc._ensure_tuple_output(gm_single)
        assert was_wrapped is True
        x = torch.tensor([1, 2])
        assert isinstance(gm_wrapped(x), tuple)

        # Already tuple output
        gm_tuple = fx.symbolic_trace(_TupleOutMod())
        _, was_wrapped_tuple = tc._ensure_tuple_output(gm_tuple)
        assert was_wrapped_tuple is False

        # Graph with no output node
        empty_graph = fx.Graph()
        empty_gm = fx.GraphModule(torch.nn.Module(), empty_graph)
        _, was_wrapped_empty = tc._ensure_tuple_output(empty_gm)
        assert was_wrapped_empty is False

        unwrap_single = tc.TpuCompilerAdaptor._unwrap_compiled_fn(lambda a: (a + 1,))
        assert unwrap_single(3) == 4
        unwrap_multi = tc.TpuCompilerAdaptor._unwrap_compiled_fn(lambda a: (a, a + 1))
        assert unwrap_multi(3) == (3, 4)

        assert tc._single_size(Range(start=8, end=8)) == 8
        assert tc._single_size(Range(start=1, end=8)) is None

    def test_run_backend_fake_tensor_and_serialize_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adaptor = tc.TpuCompilerAdaptor()
        gm = fx.symbolic_trace(_TupleOutMod())

        # FakeTensor with shape_env = None and with shape_env already set
        fake_mode_no_env = FakeTensorMode(shape_env=None)
        fake_mode_no_env.shape_env = None
        with fake_mode_no_env:
            fake_t1 = torch.empty((2, 2))

        class _SerializableFn:
            def __call__(self, *args):
                return args

            def serialize(self):
                return "serialized_bundle"

        class _BrokenSerializableFn:
            def __call__(self, *args):
                return args

            def serialize(self):
                raise RuntimeError("cannot serialize")

        monkeypatch.setattr(tc, "_tpu_backend", lambda g, ins: _SerializableFn())
        fn1, entry1 = adaptor._run_backend(gm, [fake_t1])
        assert entry1 == "serialized_bundle"
        assert fake_t1.fake_mode.shape_env is not None

        # Second call: fake_mode.shape_env is already non-None, serialize raises
        monkeypatch.setattr(tc, "_tpu_backend", lambda g, ins: _BrokenSerializableFn())
        _, entry2 = adaptor._run_backend(gm, [fake_t1])
        assert entry2 is None

    def test_save_handle_and_replay_all_branches(self, tmp_path: Path) -> None:
        adaptor = tc.TpuCompilerAdaptor()
        gm = fx.symbolic_trace(_TupleOutMod())
        artifact_path = str(tmp_path / "sub" / "artifact.pkl")

        assert (
            adaptor._save_handle("k1", artifact_path, "entry_data", gm, False) is True
        )

        # Save failure branch
        with patch("pickle.dump", side_effect=OSError("disk full")):
            assert (
                adaptor._save_handle("k1", artifact_path, "entry_data", gm, False)
                is False
            )

        # Replay success
        with patch.object(tc, "_deserialize_entry", return_value="replayed_fn"):
            assert adaptor._replay(artifact_path, gm, False) == "replayed_fn"

        # Replay deserialize failure
        with patch.object(
            tc, "_deserialize_entry", side_effect=RuntimeError("corrupt")
        ):
            assert adaptor._replay(artifact_path, gm, False) is None

        # Replay was_wrapped mismatch
        assert adaptor._replay(artifact_path, gm, True) is None

        # Replay signature mismatch
        with patch.object(
            tc.shape_variants, "graph_signature", return_value=(("other",),)
        ):
            assert adaptor._replay(artifact_path, gm, False) is None

    def test_compile_and_load_all_branches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adaptor = tc.TpuCompilerAdaptor()
        adaptor.initialize_cache(str(tmp_path), disable_cache=False)
        gm_single = fx.symbolic_trace(_SingleOutMod())
        gm_tuple = fx.symbolic_trace(_TupleOutMod())
        single_range = Range(start=4, end=4)
        multi_range = Range(start=1, end=8)

        # 1. compile hits runnable_for shortcut
        with patch.object(tc.shape_variants, "runnable_for", return_value="reused_fn"):
            fn, handle = adaptor.compile(gm_single, [], {}, single_range, key="k_short")
            assert fn == "reused_fn"
            assert handle is None

        # 2. compile with wrapped single output, serializable entry, save succeeds
        class _FakeCompiled:
            def __call__(self, x):
                return (x + 10,)

            def serialize(self):
                return "bundle_1"

        monkeypatch.setattr(tc, "_tpu_backend", lambda g, ins: _FakeCompiled())
        with (
            patch.object(tc.shape_variants, "runnable_for", return_value=None),
            patch.object(tc.shape_variants, "remember") as mock_remember,
        ):
            fn, handle = adaptor.compile(
                gm_single,
                [torch.tensor([1])],
                {},
                single_range,
                key="k_save_ok",
            )
            assert fn(torch.tensor([2])).item() == 12
            assert handle == (
                "k_save_ok",
                str(tmp_path / "k_save_ok"),
                True,
            )
            mock_remember.assert_called_once()

        # 3. compile where _save_handle returns False and range is multi-size
        with (
            patch.object(adaptor, "_save_handle", return_value=False),
            patch.object(tc.shape_variants, "remember") as mock_remember,
        ):
            fn, handle = adaptor.compile(
                gm_tuple,
                [torch.tensor([1])],
                {},
                multi_range,
                key="k_save_fail",
            )
            assert handle is None
            mock_remember.assert_not_called()

        # 4. compile where entry is None (logs no serializable artifact) and key=None
        monkeypatch.setattr(tc, "_tpu_backend", lambda g, ins: lambda x: (x,))
        fn_no_entry, handle_no_entry = adaptor.compile(
            gm_tuple,
            [torch.tensor([1])],
            {},
            single_range,
            key="k_no_entry",
        )
        assert handle_no_entry is None

        fn_no_key, handle_no_key = adaptor.compile(
            gm_tuple,
            [torch.tensor([1])],
            {},
            single_range,
            key=None,
        )
        assert handle_no_key is None

        # 5. load hits runnable_for shortcut
        saved_handle = ("k_save_ok", str(tmp_path / "k_save_ok"), True)
        with patch.object(
            tc.shape_variants, "runnable_for", return_value="loaded_elsewhere"
        ):
            assert (
                adaptor.load(saved_handle, gm_single, [], 0, single_range)
                == "loaded_elsewhere"
            )

        # 6. load replays successfully (was_wrapped=False, multi_range)
        tuple_handle_path = str(tmp_path / "k_tuple")
        adaptor._save_handle("k_tuple", tuple_handle_path, "bundle_t", gm_tuple, False)
        with (
            patch.object(tc.shape_variants, "runnable_for", return_value=None),
            patch.object(tc, "_deserialize_entry", return_value=lambda x: (x + 5,)),
        ):
            loaded_tuple_fn = adaptor.load(
                ("k_tuple", tuple_handle_path, False),
                gm_tuple,
                [],
                0,
                multi_range,
            )
            assert loaded_tuple_fn(torch.tensor([1]))[0].item() == 6

        # 7. load fallback when _replay returns None: self-heals when entry is not None,
        #    and skips _save_handle when entry is None
        monkeypatch.setattr(tc, "_tpu_backend", lambda g, ins: _FakeCompiled())
        with (
            patch.object(tc.shape_variants, "runnable_for", return_value=None),
            patch.object(adaptor, "_replay", return_value=None),
            patch.object(tc.shape_variants, "remember") as mock_remember,
        ):
            healed_fn = adaptor.load(
                saved_handle, gm_single, [torch.tensor([1])], 0, single_range
            )
            assert healed_fn(torch.tensor([3])).item() == 13
            mock_remember.assert_called_once()

        monkeypatch.setattr(tc, "_tpu_backend", lambda g, ins: lambda x: (x,))
        with (
            patch.object(tc.shape_variants, "runnable_for", return_value=None),
            patch.object(adaptor, "_replay", return_value=None),
            patch.object(adaptor, "_save_handle") as mock_save,
        ):
            adaptor.load(saved_handle, gm_single, [torch.tensor([1])], 0, multi_range)
            mock_save.assert_not_called()


class TestRankBindingRemainingBranches:
    """Cover remaining branches in worker/tpu_rank_binding.py."""

    def test_invalid_int_env_and_slice_local_world_and_dp_fallback(
        self,
    ) -> None:
        assert binding._get_int_env({"BAD": "not_an_int"}, "BAD", 7) == 7

        # slice_local_world out of range (line 182)
        env = {
            "TORCH_TPU_DP_SIZE": "2",
            **binding.slice_binding_env(
                rank=1, local_rank=0, world_size=4, local_world_size=0
            ),
        }
        pc = SimpleNamespace(
            world_size=2,
            prefill_context_parallel_size=1,
            data_parallel_size=2,
            data_parallel_index=1,
            data_parallel_rank_local=None,
            data_parallel_size_local=2,
            enable_expert_parallel=True,
            local_world_size=2,
        )
        with pytest.raises(ValueError, match="SLICE_LOCAL_WORLD_SIZE"):
            binding.get_tpu_worker_binding(pc, rank=0, local_rank=0, env=env)

        # dp_rank_local is None fallback + spawned PCP local rank
        pc_pcp_dp = SimpleNamespace(
            world_size=2,
            prefill_context_parallel_size=2,
            data_parallel_size=2,
            data_parallel_index=1,
            data_parallel_rank_local=None,
            data_parallel_size_local=2,
            enable_expert_parallel=True,
            local_world_size=2,
        )
        b = binding.get_tpu_worker_binding(
            pc_pcp_dp,
            rank=1,
            local_rank=1,
            env={"LOCAL_RANK": "3"},
            use_spawned_pcp_local_rank=True,
        )
        assert b.local_rank == 3
        assert b.native_local_rank == 3


class TestFlashAttentionTunedParamsCoverage:
    """100% statement and branch coverage for tuned_params.py."""

    def test_current_device_kind_and_make_tuning_key_and_lookup(self) -> None:
        with patch.object(tuned_params.jax, "local_devices", return_value=[]):
            assert tuned_params.current_device_kind() == "unknown"

        fake_dev = SimpleNamespace(device_kind="TPU7x")
        with patch.object(tuned_params.jax, "local_devices", return_value=[fake_dev]):
            assert tuned_params.current_device_kind() == "TPU7x"
            q = jnp.zeros((1, 12, 256, 128), dtype=jnp.bfloat16)
            k = jnp.zeros((1, 12, 256, 128), dtype=jnp.bfloat16)
            v = jnp.zeros((1, 12, 256, 128), dtype=jnp.bfloat16)
            key = tuned_params.make_tuning_key(
                q,
                k,
                v,
                causal=False,
                has_segment_ids=True,
                has_attention_bias=False,
                vmem_limit_bytes=64 * 1024 * 1024,
            )
            params = tuned_params.get_tuned_params(key)
            assert params == tuned_params.TunableParams(256, 256, 256, 1)

        miss_key = tuned_params.make_tuning_key(
            q,
            k,
            v,
            causal=True,
            has_segment_ids=False,
            has_attention_bias=False,
            vmem_limit_bytes=None,
            device_kind="CPU",
        )
        assert tuned_params.get_tuned_params(miss_key) is None


class TestRpaV3UtilCoverage:
    """100% statement and branch coverage for v3/util.py."""

    def test_util_functions_all_branches(self) -> None:
        assert rpa_util.cdiv(7, 4) == 2
        assert rpa_util.align_to(7, 4) == 8
        assert rpa_util.get_dtype_bitwidth(jnp.bfloat16) == 16
        assert rpa_util.get_dtype_packing(jnp.bfloat16) == 2
        assert rpa_util.next_power_of_2(1) == 1
        assert rpa_util.next_power_of_2(5) == 8

        for kind, expected in [
            ("cpu", -1),
            ("TPU v5 lite", 5),
            ("TPU v5p", 5),
            ("TPU v6e", 6),
            ("TPU7x", 7),
            ("TPU v4", 4),
        ]:
            with patch.object(
                rpa_util.jax,
                "devices",
                return_value=[SimpleNamespace(device_kind=kind)],
            ):
                assert rpa_util.get_tpu_version() == expected


class _FakeRef:
    def __init__(self, shape: tuple[int, ...], data=None) -> None:
        self.shape = shape
        self._data = data
        self.at = self

    def __getitem__(self, idx):
        if self._data is not None:
            return self._data[idx]
        return self


class TestRaggedKvCacheUpdateFullCoverage:
    """100% statement and branch coverage for ragged_kv_cache_update.py."""

    def test_kernel_body_non_pipelined_and_pipelined(self) -> None:
        slices_data = jnp.array([[0, 4], [0, 2], [2, 2]], dtype=jnp.int32)
        num_slices_ref = jnp.array([2], dtype=jnp.int32)
        new_kv_ref = _FakeRef((4, 2, 128))
        kv_cache_ref = _FakeRef((8, 2, 128))
        scratch_ref = _FakeRef((2, 4, 2, 128))
        empty_scratch_ref = _FakeRef((0, 4, 2, 128))

        copies = []

        class _FakeCopy:
            def start(self) -> None:
                copies.append("start")

            def wait(self) -> None:
                copies.append("wait")

        with (
            patch.object(rkv.pl, "program_id", return_value=0),
            patch.object(rkv.pltpu, "make_async_copy", return_value=_FakeCopy()),
        ):
            rkv._kv_cache_update_kernel(
                slices_data,
                num_slices_ref,
                new_kv_ref,
                kv_cache_ref,
                None,
                scratch_ref,
                "sem",
                pipelined_dma=False,
            )
            assert copies.count("start") == 4
            assert copies.count("wait") == 4

            copies.clear()
            rkv._kv_cache_update_kernel(
                slices_data,
                num_slices_ref,
                new_kv_ref,
                kv_cache_ref,
                None,
                scratch_ref,
                "sem",
                pipelined_dma=True,
            )
            assert copies.count("start") == 4
            assert copies.count("wait") == 4

            # Cover num_slices_per_block == 0 so write_copy is None
            copies.clear()
            rkv._kv_cache_update_kernel(
                slices_data,
                num_slices_ref,
                new_kv_ref,
                kv_cache_ref,
                None,
                empty_scratch_ref,
                "sem",
                pipelined_dma=True,
            )
            assert copies == ["start"]

    def test_dynamic_validate_inputs_all_branches(self) -> None:
        # Valid case with 2 slices
        valid_slices = jnp.array([[0, 4], [0, 2], [2, 2]], dtype=jnp.int32)
        rkv._dynamic_validate_inputs(
            valid_slices,
            new_token_num=4,
            kv_cache_token_num=8,
            page_size=4,
            num_slices=jnp.array([2], dtype=jnp.int32),
        )

        n1 = jnp.array([1], dtype=jnp.int32)
        n2 = jnp.array([2], dtype=jnp.int32)

        # 1. new_kv_start < 0
        with pytest.raises(ValueError, match="new_kv_start"):
            rkv._dynamic_validate_inputs(jnp.array([[0], [-1], [2]]), 4, 8, 4, n1)
        # 2. kv_cache_start < 0
        with pytest.raises(ValueError, match="kv_cache_start"):
            rkv._dynamic_validate_inputs(jnp.array([[-1], [0], [2]]), 4, 8, 4, n1)
        # 3. slice_len <= 0 or > page_size
        with pytest.raises(ValueError, match="slice_len"):
            rkv._dynamic_validate_inputs(jnp.array([[0], [0], [0]]), 4, 8, 4, n1)
        # 4. new_kv_start + slice_len > new_token_num
        with pytest.raises(ValueError, match="new_token_num"):
            rkv._dynamic_validate_inputs(jnp.array([[0], [3], [2]]), 4, 8, 4, n1)
        # 5. kv_cache_start + slice_len > kv_cache_token_num
        with pytest.raises(ValueError, match="kv_cache_token_num"):
            rkv._dynamic_validate_inputs(jnp.array([[7], [0], [2]]), 4, 8, 4, n1)
        # 6. Cross-page slice
        with pytest.raises(ValueError, match="same page"):
            rkv._dynamic_validate_inputs(jnp.array([[3], [0], [2]]), 4, 8, 4, n1)
        # 7. Discontinuous new_kv_intervals
        with pytest.raises(ValueError, match="expeced to equal"):
            rkv._dynamic_validate_inputs(
                jnp.array([[0, 4], [0, 3], [2, 1]]), 4, 8, 4, n2
            )
        # 8. Overlapping kv_cache_intervals
        with pytest.raises(ValueError, match="Overlap detected"):
            rkv._dynamic_validate_inputs(
                jnp.array([[0, 1], [0, 2], [2, 2]]), 4, 8, 4, n2
            )

    def test_kv_cache_update_and_sizing_helpers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert rkv._prev_power_of_2(0) == 0
        assert rkv._prev_power_of_2(-4) == 0
        assert rkv._prev_power_of_2(6) == 4

        page_bytes = rkv._get_page_size_bytes(32, 2, 128, jnp.bfloat16)
        assert page_bytes > 0
        slices_per_block = rkv._get_num_slices_per_kv_cache_update_block(
            page_bytes, 40 * 1024 * 1024
        )
        assert slices_per_block == 64

        new_kv = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((8, 2, 128), dtype=jnp.bfloat16)
        slices = jnp.array([[0, 4], [0, 2], [2, 2]], dtype=jnp.int32)
        num_slices = jnp.array([2], dtype=jnp.int32)

        rkv._get_kv_update_specs.cache_clear()
        monkeypatch.setattr(rkv.pltpu, "MemorySpace", SimpleNamespace(ANY="any"))
        with patch.object(
            rkv.pl,
            "pallas_call",
            side_effect=lambda *args, **kwargs: (lambda *call_args: [call_args[-1]]),
        ):
            out1 = rkv._kv_cache_update(
                new_kv,
                slices,
                kv_cache,
                num_slices,
                page_size=4,
                num_slices_per_block=2,
                dynamic_validate_inputs=True,
                pipelined_dma=False,
            )
            assert out1.shape == kv_cache.shape

            # Second call with num_slices_per_block=None via kv_cache_update.__wrapped__
            out2 = rkv.kv_cache_update.__wrapped__(
                new_kv,
                slices,
                kv_cache,
                num_slices,
                page_size=4,
                num_slices_per_block=None,
                dynamic_validate_inputs=False,
                pipelined_dma=True,
            )
            assert out2.shape == kv_cache.shape

        # mesh provided without kv_cache_pspec raises ValueError
        fake_mesh = MagicMock()
        with pytest.raises(ValueError, match="kv_cache_pspec must be provided"):
            rkv.kv_cache_update.__wrapped__(
                new_kv,
                slices,
                kv_cache,
                num_slices,
                page_size=4,
                num_slices_per_block=2,
                mesh=fake_mesh,
                kv_cache_pspec=None,
            )

        # mesh provided with kv_cache_pspec delegates through jax.shard_map
        with (
            patch.object(
                rkv.jax,
                "shard_map",
                side_effect=lambda fn, **kw: fn,
            ),
            patch.object(
                rkv,
                "_kv_cache_update",
                return_value=kv_cache,
            ) as mock_inner,
        ):
            out3 = rkv.kv_cache_update.__wrapped__(
                new_kv,
                slices,
                kv_cache,
                num_slices,
                page_size=4,
                num_slices_per_block=2,
                mesh=fake_mesh,
                kv_cache_pspec=P(),
            )
            assert out3 is kv_cache
            mock_inner.assert_called_once()
