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

import dataclasses
import importlib
import pickle
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu import env_override, envs, tokenizers
from vllm_torchtpu.layers.adapter import moe_hierarchical
from vllm_torchtpu.layers.core import quant_methods
from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as wrapper_ctx
from vllm_torchtpu.runner import stage_timing


def _set_env(monkeypatch, name: str, value: str | None) -> None:
    envs.__dict__.pop(name, None)
    if value is None:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def test_env_override_all_branches(monkeypatch):
    for var in (
        "TORCH_TPU_TIER2_COMPILATION_CACHE",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
        "TPU_PREMAPPED_BUFFER_SIZE",
        "TPU_GDN_CONV_QK_PAIR_LAYOUT",
        "LIBTPU_INIT_ARGS",
    ):
        _set_env(monkeypatch, var, None)
    with patch("vllm_torchtpu.patch_registry.apply"):
        # 1. Reload with Tier-3 cache set, Raiden enabled with degree >= 8,
        #    and empty LIBTPU_INIT_ARGS + TPU_ACCELERATOR_TYPE="tpu7x-8"
        _set_env(
            monkeypatch, "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", "/tmp/tpu_tier3"
        )
        _set_env(monkeypatch, "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", None)
        monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
        monkeypatch.setenv("TPU_RAIDEN_TRANSFER_PARALLELISM", "8")
        monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
        monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
        importlib.reload(env_override)
        assert (
            "--xla_tpu_use_dynamic_smem_negotiation=true"
            in (env_override.os.environ["LIBTPU_INIT_ARGS"])
        )

        # 2. Reload with Tier-3 cache unset, Raiden enabled via manager with
        #    degree < 8, existing LIBTPU_INIT_ARGS containing one SC flag,
        #    and SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES > 0
        _set_env(monkeypatch, "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", None)
        _set_env(monkeypatch, "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "1048576")
        monkeypatch.delenv("TPU_KV_RESHARD_TRANSPORT", raising=False)
        monkeypatch.setenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", "1")
        monkeypatch.setenv("TPU_RAIDEN_TRANSFER_PARALLELISM", "4")
        monkeypatch.setenv(
            "LIBTPU_INIT_ARGS",
            "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=99",
        )
        importlib.reload(env_override)
        assert (
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1048576"
            in (env_override.os.environ["LIBTPU_INIT_ARGS"])
        )

        # 3. Reload with Raiden disabled, SC override == 0 (None threshold),
        #    and LIBTPU_INIT_ARGS already containing
        #    --xla_tpu_use_dynamic_smem_negotiation so _libtpu_extra_args is empty.
        monkeypatch.delenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER", raising=False)
        _set_env(monkeypatch, "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "0")
        monkeypatch.setenv(
            "LIBTPU_INIT_ARGS",
            "--xla_tpu_use_dynamic_smem_negotiation=false",
        )
        importlib.reload(env_override)

    # 4. Direct tests of _fetch_tpu_type_from_metadata and
    #    _resolve_sc_offload_threshold_bytes
    fake_requests = SimpleNamespace()
    with patch.dict(sys.modules, {"requests": fake_requests}):
        fake_requests.get = lambda *a, **k: SimpleNamespace(
            status_code=200, text="v6e-8"
        )
        assert env_override._fetch_tpu_type_from_metadata() == "v6e-8"

        fake_requests.get = lambda *a, **k: SimpleNamespace(status_code=404, text="")
        assert env_override._fetch_tpu_type_from_metadata() is None

        def _raise_err(*a, **k):
            raise RuntimeError("network error")

        fake_requests.get = _raise_err
        assert env_override._fetch_tpu_type_from_metadata() is None

    _set_env(monkeypatch, "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", None)
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    with patch.object(env_override, "_fetch_tpu_type_from_metadata", return_value=None):
        assert env_override._resolve_sc_offload_threshold_bytes() is None

    with patch.object(
        env_override,
        "_fetch_tpu_type_from_metadata",
        return_value="unknown-tpu",
    ):
        assert env_override._resolve_sc_offload_threshold_bytes() is None

    # 5. Direct tests of _patch_jax_pallas_fori_lowering() across all guard branches
    # 5a. Import failure
    with patch.dict(sys.modules, {"jax._src.pallas.mosaic": None}):
        env_override._patch_jax_pallas_fori_lowering()

    fake_mosaic = types.ModuleType("jax._src.pallas.mosaic")
    fake_lowering = types.ModuleType("jax._src.pallas.mosaic.lowering")
    fake_lowering.__file__ = "<fake_lowering>"
    fake_mosaic.lowering = fake_lowering  # type: ignore[attr-defined]

    with patch.dict(
        sys.modules,
        {
            "jax._src.pallas.mosaic": fake_mosaic,
            "jax._src.pallas.mosaic.lowering": fake_lowering,
        },
    ):
        # 5b. Already patched
        fake_lowering._torchtpu_fori_patch_applied = True  # type: ignore[attr-defined]
        env_override._patch_jax_pallas_fori_lowering()

        # 5c. orig is None
        fake_lowering._torchtpu_fori_patch_applied = False  # type: ignore[attr-defined]
        env_override._patch_jax_pallas_fori_lowering()

        # 5d. inspect.getsource raises
        fake_lowering._lower_jaxpr_to_for_loop = "not_a_func"  # type: ignore[attr-defined]
        env_override._patch_jax_pallas_fori_lowering()

        # 5e. needle not in src
        fake_lowering._lower_jaxpr_to_for_loop = lambda: None  # type: ignore[attr-defined]
        with patch("inspect.getsource", return_value="def foo():\n    pass\n"):
            env_override._patch_jax_pallas_fori_lowering()

        # 5f. exec raises
        with (
            patch(
                "inspect.getsource",
                return_value=(
                    "def _lower_jaxpr_to_for_loop():\n"
                    "    if False:\n        pass\n"
                    "    elif has_dynamic_remainder:\n"
                    "        raise RuntimeError()\n"
                ),
            ),
            patch("builtins.compile", side_effect=SyntaxError("bad")),
        ):
            env_override._patch_jax_pallas_fori_lowering()

        # 5g. new_fn is None (function name wasn't _lower_jaxpr_to_for_loop)
        with patch(
            "inspect.getsource",
            return_value=(
                "def _other_name(has_dynamic_remainder=False, unroll=1):\n"
                "    if False:\n        pass\n"
                "    elif has_dynamic_remainder:\n"
                "        return 1\n"
            ),
        ):
            env_override._patch_jax_pallas_fori_lowering()

        # 5h. Successful patch
        with patch(
            "inspect.getsource",
            return_value=(
                "def _lower_jaxpr_to_for_loop("
                "has_dynamic_remainder=False, unroll=1):\n"
                "    if False:\n        pass\n"
                "    elif has_dynamic_remainder:\n"
                "        return 1\n"
                "    return 0\n"
            ),
        ):
            env_override._patch_jax_pallas_fori_lowering()
            assert fake_lowering._torchtpu_fori_patch_applied is True  # type: ignore[attr-defined]
            assert fake_lowering._lower_jaxpr_to_for_loop(True, 1) == 0  # type: ignore[attr-defined]


def test_stage_timing_all_branches():
    class _FakeTpuEvent:
        def record(self):
            return None

        def synchronize(self):
            return None

    with patch.object(stage_timing, "TpuEvent", _FakeTpuEvent):
        timer = stage_timing.RunnerStageTiming(rank=0)
        with timer.host_stage("host_op"):
            pass

        with timer.stage("device_op"):
            pass

        # Exception inside stage still records end and CPU sample
        with pytest.raises(ValueError, match="stage error"), timer.stage("failing_op"):
            raise ValueError("stage error")

        # Wait briefly for daemon collector thread to drain the 2 queued items
        deadline = stage_timing.time.perf_counter() + 2.0
        while (
            len(timer._device_samples) < 2
            and stage_timing.time.perf_counter() < deadline
        ):
            stage_timing.time.sleep(0.005)

        # When semaphore is exhausted, stage falls back to host_stage
        with (
            patch.object(timer._slots, "acquire", return_value=False),
            timer.stage("fallback_host_op"),
        ):
            pass

        cpu_samples, dev_samples = timer.take_samples()
        assert len(cpu_samples) == 4
        assert len(dev_samples) == 2

        # Also exercise _collect_device_time directly on the main thread so
        # single-thread coverage records lines 51-65 deterministically.
        timer._slots.acquire(blocking=False)
        one_item = stage_timing._PendingTiming(
            "direct_op", _FakeTpuEvent(), _FakeTpuEvent()
        )
        one_item.end_recorded.set()
        timer._pending = SimpleNamespace(  # type: ignore[assignment]
            get=MagicMock(side_effect=[one_item, StopIteration("done")])
        )
        with pytest.raises(StopIteration, match="done"):
            timer._collect_device_time()
        _, direct_dev = timer.take_samples()
        assert len(direct_dev) == 1


class _BaseTok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools=None, **kwargs):
        self.calls.append((messages, tools, kwargs))
        return "ok"


def test_tokenizers_all_branches(monkeypatch):
    base = _BaseTok()
    wrapped = tokenizers._tpu_deepseek_v4_tokenizer(base)

    # 1. continue_final_message=False / omitted
    assert wrapped.apply_chat_template([{"role": "assistant", "content": "a"}]) == "ok"
    assert "conversation" not in base.calls[-1][2]

    # 2. continue_final_message=True with trailing assistant
    assert (
        wrapped.apply_chat_template(
            [{"role": "assistant", "content": "a"}],
            continue_final_message=True,
        )
        == "ok"
    )
    assert base.calls[-1][2]["conversation"][-1]["wo_eos"] is True

    # 3. continue_final_message=True with trailing user
    wrapped.apply_chat_template(
        [{"role": "user", "content": "u"}], continue_final_message=True
    )
    assert "wo_eos" not in base.calls[-1][2]["conversation"][-1]

    # 4. continue_final_message=True with empty conversation
    wrapped.apply_chat_template([], continue_final_message=True)
    assert base.calls[-1][2]["conversation"] == []

    # 5. Pickle roundtrip via __reduce__
    unpickled = pickle.loads(pickle.dumps(wrapped))
    assert type(unpickled).__name__ == "_TpuDeepseekV4Tokenizer"

    # 6. TpuDeepseekV4Tokenizer.from_pretrained
    fake_dsv4_mod = SimpleNamespace(
        DeepseekV4Tokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: _BaseTok())
    )
    with patch.dict(sys.modules, {"vllm.tokenizers.deepseek_v4": fake_dsv4_mod}):
        inst = tokenizers.TpuDeepseekV4Tokenizer.from_pretrained("model-id")
        assert type(inst).__name__ == "_TpuDeepseekV4Tokenizer"

    # 7. register_tokenizers when flag is False vs True
    fake_reg = MagicMock()
    with patch.dict(
        sys.modules,
        {"vllm.tokenizers.registry": SimpleNamespace(TokenizerRegistry=fake_reg)},
    ):
        _set_env(monkeypatch, "TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", "0")
        tokenizers.register_tokenizers()
        fake_reg.register.assert_not_called()

        _set_env(monkeypatch, "TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", "1")
        tokenizers.register_tokenizers()
        fake_reg.register.assert_called_once()


def test_vllm_model_wrapper_context_all_branches():
    with pytest.raises(AssertionError, match="VllmModelWrapperContext is not set"):
        wrapper_ctx.get_vllm_model_wrapper_context()

    fake_mesh = MagicMock()
    with wrapper_ctx.set_vllm_model_wrapper_context(
        mesh=fake_mesh,
        kv_caches=[],
        layer_name_to_kvcache_index={"l0": 0},
        vllm_config="cfg",
        kv_cache_bundle=None,
    ):
        ctx = wrapper_ctx.get_vllm_model_wrapper_context()
        assert ctx.mesh is fake_mesh
        assert ctx.vllm_config == "cfg"

    with pytest.raises(AssertionError, match="VllmModelWrapperContext is not set"):
        wrapper_ctx.get_vllm_model_wrapper_context()


def test_moe_hierarchical_and_quant_methods_all_branches(monkeypatch):
    # quant_methods
    quant_methods.get_tpu_quant_method.cache_clear()
    assert quant_methods.get_tpu_quant_method(quant_methods.FP8) == "tpu-fp8"
    assert quant_methods.get_tpu_quant_method(quant_methods.MXFP4) == "tpu-mxfp4"

    # hierarchical_split_or_none:
    # 1. env flag False
    _set_env(monkeypatch, "TPU_MOE_HIERARCHICAL_EP", "0")
    assert moe_hierarchical.hierarchical_split_or_none() is None

    # 2. env flag True, world_size <= 1
    _set_env(monkeypatch, "TPU_MOE_HIERARCHICAL_EP", "1")
    with patch.object(
        moe_hierarchical, "get_tensor_model_parallel_world_size", return_value=1
    ):
        assert moe_hierarchical.hierarchical_split_or_none() is None

    # 3. env flag True, world_size > 1, hierarchical_moe_split returns None vs tuple
    with (
        patch.object(
            moe_hierarchical,
            "get_tensor_model_parallel_world_size",
            return_value=8,
        ),
        patch.object(
            moe_hierarchical, "get_tensor_model_parallel_rank", return_value=0
        ),
    ):
        with patch.object(
            moe_hierarchical, "hierarchical_moe_split", return_value=None
        ):
            assert moe_hierarchical.hierarchical_split_or_none() is None

        with patch.object(
            moe_hierarchical,
            "hierarchical_moe_split",
            return_value=(4, 0, 2, 0),
        ):
            assert moe_hierarchical.hierarchical_split_or_none() == (4, 0, 2, 0)

    # align_ep_weight_filter:
    # 1. already patched
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", True)
    moe_hierarchical.align_ep_weight_filter()

    # 2. not patched, split=None, hierarchical_split_or_none() returns None
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    with patch.object(
        moe_hierarchical, "hierarchical_split_or_none", return_value=None
    ):
        moe_hierarchical.align_ep_weight_filter()
        assert moe_hierarchical._filter_patched is False

    # 3. not patched, split=None, hierarchical_split_or_none() returns tuple
    from vllm.model_executor.model_loader import default_loader

    orig_compute = default_loader.compute_local_expert_ids
    monkeypatch.setattr(
        default_loader,
        "compute_local_expert_ids",
        lambda n, es, er, placement="linear": {n, es, er},
    )
    with patch.object(
        moe_hierarchical,
        "hierarchical_split_or_none",
        return_value=(4, 1, 2, 0),
    ):
        moe_hierarchical.align_ep_weight_filter()
        assert moe_hierarchical._filter_patched is True
        assert default_loader.compute_local_expert_ids(224, 8, 3) == {224, 4, 1}
    monkeypatch.setattr(default_loader, "compute_local_expert_ids", orig_compute)

    # hierarchical_moe_parallel_config:
    # 1. split is None
    with (
        patch.object(moe_hierarchical, "hierarchical_split_or_none", return_value=None),
        moe_hierarchical.hierarchical_moe_parallel_config() as s,
    ):
        assert s is None

    # 2. split is not None (with _filter_patched=False so align_ep_weight_filter(split)
    #    exercises the split is not None branch!), both config.use_ep=False and True
    monkeypatch.setattr(moe_hierarchical, "_filter_patched", False)
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
    )

    cfg_no_ep = FusedMoEParallelConfig(
        tp_size=8,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )
    cfg_ep = dataclasses.replace(cfg_no_ep, ep_size=8, tp_size=1, use_ep=True)
    with (
        patch.object(
            moe_hierarchical,
            "hierarchical_split_or_none",
            return_value=(4, 2, 2, 1),
        ),
        patch.object(
            FusedMoEParallelConfig,
            "make",
            staticmethod(lambda use_ep_flag: cfg_ep if use_ep_flag else cfg_no_ep),
        ),
        moe_hierarchical.hierarchical_moe_parallel_config() as s,
    ):
        assert s == (4, 2, 2, 1)
        res_no_ep = FusedMoEParallelConfig.make(False)
        assert res_no_ep is cfg_no_ep
        res_ep = FusedMoEParallelConfig.make(True)
        assert (res_ep.ep_size, res_ep.ep_rank, res_ep.tp_size, res_ep.tp_rank) == (
            4,
            2,
            2,
            1,
        )
