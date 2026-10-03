# SPDX-License-Identifier: Apache-2.0
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from .mock_omni import vllm_omni_mock  # isort: split

import vllm_omni
from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

from vllm_torchtpu.omni import register_omni_tpu_platform
from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl
from vllm_torchtpu.omni.patches import (
    _unwrap_tpu_patch,
    apply_omni_model_specific_patches,
    apply_omni_tpu_patches,
    apply_omni_worker_patches,
)
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [pytest.mark.cpu_test, pytest.mark.omni]


@pytest.fixture(scope="module", autouse=True)
def cleanup_mock_omni():
    yield
    for mod_name in list(sys.modules.keys()):
        if mod_name == "vllm_omni" or mod_name.startswith("vllm_omni."):
            del sys.modules[mod_name]


def test_register_omni_tpu_platform(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 0)
    assert register_omni_tpu_platform() is None

    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 8)
    assert register_omni_tpu_platform() == "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_platform_attributes():
    platform = OmniTpuPlatform()
    assert platform.device_name == "tpu"
    assert platform.device_type == "tpu"
    assert platform.dist_backend == "gloo"
    assert platform.supports_torch_inductor() is False
    assert platform.supports_diffusion_dense_flash_attention() is False
    assert (
        platform.get_diffusion_attn_backend_cls(None, 64)
        == "vllm_torchtpu.omni.attention.TpuSDPABackend"
    )
    with patch("torch.device") as mock_dev:
        platform.get_torch_device()
        mock_dev.assert_called_once_with("tpu")


def test_device_control_env_var():
    assert OmniTpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


def test_shared_slicebuilder_addresses(monkeypatch):
    # Every worker derives the same list from the shared MASTER_PORT.
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "localhost:1")
    OmniTpuPlatform._set_shared_slicebuilder_addresses(2)
    assert (
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"]
        == "localhost:29501,localhost:29502"
    )
    # A correctly sized inherited list is kept.
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "a:1,b:2")
    OmniTpuPlatform._set_shared_slicebuilder_addresses(2)
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == "a:1,b:2"


def test_dist_backend_after_world_size_pop(monkeypatch):
    # Building the worker's VllmConfig pops WORLD_SIZE after set_device; the
    # size set_device recorded must keep multi-rank groups on tpu_dist.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "a:1,b:2")
    monkeypatch.setattr(OmniTpuPlatform, "_worker_world_size", 1)
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        staticmethod(lambda world_size: None),
    )
    monkeypatch.setattr(torch, "accelerator", MagicMock(), raising=False)
    with patch("torch.empty"), patch("torch.device"):
        OmniTpuPlatform.set_device(None)
    monkeypatch.delenv("WORLD_SIZE")
    assert OmniTpuPlatform().dist_backend == "tpu_dist"


def test_memory_methods(monkeypatch):
    class MockAccelerator:
        @staticmethod
        def get_memory_info(device=None):
            return (16 * 1024**3, 32 * 1024**3)

        @staticmethod
        def max_memory_reserved(device=None):
            return 16 * 1024**3

        @staticmethod
        def max_memory_allocated(device=None):
            return 16 * 1024**3

    monkeypatch.setattr(torch, "accelerator", MockAccelerator(), raising=False)
    with patch(
        "torch_tpu._internal.utils.hardware.get_hbm_bytes_per_device",
        return_value=32 * 1024**3,
    ):
        assert OmniTpuPlatform.get_device_total_memory() == 32 * 1024**3
        assert OmniTpuPlatform.get_free_memory() == 16 * 1024**3
        assert OmniTpuPlatform.get_device_memory() == (16 * 1024**3, 32 * 1024**3)
        assert OmniTpuPlatform.max_memory_reserved() == 16 * 1024**3
        assert OmniTpuPlatform.max_memory_allocated() == 16 * 1024**3


def test_tpu_sdpa_attention_backend():
    assert TpuSDPABackend.get_name() == "TPU_SDPA"
    assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
    assert TpuSDPABackend.supports_attention_mask() is True

    impl = TpuSDPAImpl(num_heads=4, head_size=16, softmax_scale=0.25)
    query = torch.randn(2, 8, 4, 16)
    key = torch.randn(2, 8, 4, 16)
    value = torch.randn(2, 8, 4, 16)

    res = impl.forward(query, key, value)
    assert res.shape == (2, 8, 4, 16)

    # GQA
    impl_gqa = TpuSDPAImpl(
        num_heads=8, head_size=16, softmax_scale=0.25, num_kv_heads=2
    )
    query_gqa = torch.randn(1, 16, 8, 16)
    key_gqa = torch.randn(1, 16, 2, 16)
    value_gqa = torch.randn(1, 16, 2, 16)
    res_gqa = impl_gqa.forward(query_gqa, key_gqa, value_gqa)
    assert res_gqa.shape == (1, 16, 8, 16)


@pytest.mark.parametrize(
    "seq_len, expected",
    [
        (4224, (1408, 4224, 1408)),  # Z-Image 1024^2: 4096 image + 128 caption
        (4096, (2048, 4096, 1024)),  # Z-Image noise refiner
        (1024, (1024, 1024, 1024)),
        (16384, (2048, 4096, 1024)),  # longer than the resident-KV cap: chunked
        (4160, None),  # 65 x 64: no lane-aligned q block divides it
        (100, None),
    ],
)
def test_splash_block_sizes(seq_len, expected):
    from vllm_torchtpu.omni.attention import splash_block_sizes

    assert splash_block_sizes(seq_len) == expected


@pytest.mark.parametrize(
    "seq_len, expected",
    [
        (4128, (4224, (1408, 4224, 384))),  # Z-Image 1024^2 + 32-token caption
        (4224, (4224, (1408, 4224, 1408))),  # already aligned: no padding, no mask
        (4096, (4096, (2048, 4096, 1024))),
        (4160, (4224, (1408, 4224, 384))),  # 65 x 64 -> next bucket
        (8320, (8448, (1408, 1408, 1408))),  # 65 x 128 only admits q=128: use 66 x 128
        (1024, (1024, (1024, 1024, 1024))),
        (1056, (1408, (1408, 1408, 1408))),  # 512^2 warm-up shape
    ],
)
def test_splash_plan(seq_len, expected):
    from vllm_torchtpu.omni.attention import _SPLASH_MIN_BLOCK_Q, splash_plan

    plan = splash_plan(seq_len)
    assert plan == expected
    padded, (block_q, block_kv, block_kv_compute) = plan
    assert padded >= seq_len and padded % 128 == 0
    assert padded % block_q == 0 and block_kv % block_kv_compute == 0
    assert block_q >= _SPLASH_MIN_BLOCK_Q


def test_padding_segment_ids():
    pytest.importorskip("jax.experimental.pallas.ops.tpu.splash_attention")
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.omni.attention import _padding_segment_ids

    heads, padded, valid, dim = 2, 256, 200, 128
    q = jnp.full((heads, padded, dim), 3.5, dtype=jnp.bfloat16)
    k = jnp.full((heads, padded, dim), -7.25, dtype=jnp.bfloat16)
    ids = _padding_segment_ids(q, k, valid)
    assert ids.q.shape == (padded,) and ids.kv.shape == (padded,)
    assert ids.q.dtype == jnp.int32 and ids.kv.dtype == jnp.int32
    # Queries are one segment; keys are segment 0 if valid, 1 if padding.
    np.testing.assert_array_equal(np.asarray(ids.q), np.zeros(padded, np.int32))
    np.testing.assert_array_equal(
        np.asarray(ids.kv),
        np.r_[np.zeros(valid, np.int32), np.ones(padded - valid, np.int32)],
    )
    # Resulting mask: every query sees exactly the valid keys (no empty rows).
    allowed = np.asarray(ids.q)[:, None] == np.asarray(ids.kv)[None, :]
    assert allowed[:, :valid].all() and not allowed[:, valid:].any()
    assert allowed.sum(axis=1).tolist() == [valid] * padded


def _fake_tpu_tensor(shape, dtype=torch.bfloat16):
    t = MagicMock(spec=torch.Tensor)
    t.device = MagicMock()
    t.device.type = "tpu"
    t.dtype = dtype
    t.shape = torch.Size(shape)
    t.dim.return_value = len(shape)
    return t


def test_can_use_splash_attention():
    from vllm_torchtpu.omni.attention import can_use_splash_attention

    q = _fake_tpu_tensor((1, 4224, 30, 128))
    assert can_use_splash_attention(q, q, q, None, causal=False) is True
    # Unaligned lengths are padded inside the kernel path.
    q_unaligned = _fake_tpu_tensor((1, 4128, 30, 128))
    assert can_use_splash_attention(q_unaligned, q_unaligned, q_unaligned, None, False)
    # Metadata without a mask is fine; a mask is not.
    assert can_use_splash_attention(q, q, q, MagicMock(attn_mask=None), False)
    assert not can_use_splash_attention(
        q, q, q, MagicMock(attn_mask=torch.ones(1)), False
    )
    assert not can_use_splash_attention(q, q, q, None, causal=True)
    # GQA / mismatched shapes, small sequences, odd head dims, other dtypes.
    assert not can_use_splash_attention(
        q, _fake_tpu_tensor((1, 4224, 6, 128)), q, None, False
    )
    assert not can_use_splash_attention(
        *(_fake_tpu_tensor((1, 128, 30, 128)),) * 3, None, False
    )
    assert not can_use_splash_attention(
        *(_fake_tpu_tensor((1, 4224, 30, 64)),) * 3, None, False
    )
    qf = _fake_tpu_tensor((1, 4224, 30, 128), dtype=torch.float32)
    assert not can_use_splash_attention(qf, qf, qf, None, False)
    # CPU tensors never take the kernel path.
    cpu = torch.zeros(1, 4224, 30, 128, dtype=torch.bfloat16)
    assert not can_use_splash_attention(cpu, cpu, cpu, None, False)


def test_tpu_sdpa_impl_routes_to_splash(monkeypatch):
    from vllm_torchtpu.omni import attention as omni_attention

    impl = TpuSDPAImpl(num_heads=30, head_size=128, softmax_scale=0.125)
    calls = []
    monkeypatch.setattr(
        impl, "_forward_splash", lambda q, k, v: calls.append("splash") or q
    )
    monkeypatch.setattr(
        omni_attention,
        "can_use_splash_attention",
        lambda *a, **k: a[0].device.type == "tpu",
    )
    tpu_q = _fake_tpu_tensor((1, 4224, 30, 128))
    assert impl.forward(tpu_q, tpu_q, tpu_q) is tpu_q
    assert calls == ["splash"]

    cpu_q = torch.randn(1, 64, 2, 16)
    assert impl.forward(cpu_q, cpu_q, cpu_q).shape == cpu_q.shape
    assert calls == ["splash"]


def test_vllm_omni_is_mocked():
    assert sys.modules["vllm_omni"] is vllm_omni_mock
    assert not hasattr(vllm_omni, "__file__") or vllm_omni.__file__ is None


def test_ltx2_safe_decode_output():
    apply_omni_model_specific_patches()

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    runtime = ltx2_runtime.LTXRuntime()

    decode_kwargs = dict(
        latents=torch.randn(1, 128, 4, 4),
        audio_latents=torch.randn(1, 8, 16),
        output_type="pt",
        connector_prompt_embeds=torch.randn(1, 10, 64),
        generator=None,
        device=torch.device("cpu"),
        decode_timestep=0.0,
        decode_noise_scale=None,
        prompt_batch_size=1,
    )

    # Case 1: Video and audio are both Tensors (verify .cpu() is invoked)
    v = torch.randn(1, 3, 16, 16).as_subclass(TrackedTensor)
    a = torch.randn(1, 100).as_subclass(TrackedTensor)
    runtime.output = (v, a)
    res = runtime._decode_output(**decode_kwargs)
    assert v.cpu_called is True
    assert a.cpu_called is True
    assert res.output[0].device.type == "cpu"
    assert res.output[1].device.type == "cpu"

    # Case 2: Video is Tensor, audio is None
    v = torch.randn(1, 3, 16, 16).as_subclass(TrackedTensor)
    runtime.output = (v, None)
    res = runtime._decode_output(**decode_kwargs)
    assert v.cpu_called is True
    assert res.output[0].device.type == "cpu"
    assert res.output[1] is None

    # Case 3: Both are None (verify non-tensor outputs pass through safely)
    runtime.output = (None, None)
    res = runtime._decode_output(**decode_kwargs)
    assert res.output == (None, None)


def test_safe_pack_tensor_if_large(monkeypatch):
    from unittest.mock import MagicMock

    from vllm_omni.diffusion import ipc

    from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

    # Provide an original pack function to mock_omni's ipc
    orig_mock = MagicMock(
        side_effect=lambda tensor, d2h_stream=None: {
            "packed": True,
            "device": getattr(tensor, "device", None).type
            if hasattr(tensor, "device")
            else "unknown",
        }
    )
    monkeypatch.setattr(ipc, "_pack_tensor_if_large", orig_mock)

    apply_omni_model_specific_patches.cache_clear()
    apply_omni_model_specific_patches()

    class MockDeviceTensor:
        def __init__(self, device_type="tpu"):
            self.device = MagicMock()
            self.device.type = device_type
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return torch.empty(1, device="cpu")

    t = MockDeviceTensor("tpu")
    res = ipc._pack_tensor_if_large(t)
    assert t.cpu_called is True
    assert res["device"] == "cpu"


def test_omni_tpu_patches_configuration(monkeypatch):
    # Call the undecorated body so the run-once cache is left alone, and
    # restore the process-global state it edits (env var, F.interpolate).
    orig_interpolate = torch.nn.functional.interpolate
    monkeypatch.setattr(torch.nn.functional, "interpolate", orig_interpolate)
    monkeypatch.setattr(
        torch.nn.functional, "group_norm", torch.nn.functional.group_norm
    )
    with patch.dict(os.environ):
        os.environ.pop("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", None)
        apply_omni_tpu_patches.__wrapped__()
        assert os.environ["VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT"] == "1800.0"


def test_flow_match_scheduler_step_flush(monkeypatch):
    """Scheduler.step launches the pending TPU graph asynchronously per step."""
    from vllm_torchtpu.omni import patches as omni_patches

    calls = []

    class _Scheduler:
        def step(self, model_output, timestep, sample, return_dict=True):
            return (sample,)

    fake_schedulers = MagicMock()
    fake_schedulers.FlowMatchEulerDiscreteScheduler = _Scheduler
    fake_diffusers = MagicMock()
    fake_diffusers.schedulers = fake_schedulers
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.schedulers", fake_schedulers)
    monkeypatch.setattr(
        omni_patches, "_tpu_materialize_async", lambda t: calls.append(t) or True
    )

    orig_step = _Scheduler.step
    omni_patches._patch_flow_match_scheduler_step_flush()
    omni_patches._patch_flow_match_scheduler_step_flush()  # idempotent
    assert _unwrap_tpu_patch(_Scheduler.step) is orig_step

    tpu_sample = MagicMock(spec=torch.Tensor)
    tpu_sample.device = MagicMock()
    tpu_sample.device.type = "tpu"
    cpu_sample = torch.zeros(1)

    assert _Scheduler().step(None, None, tpu_sample, return_dict=False) == (tpu_sample,)
    assert _Scheduler().step(None, None, cpu_sample, return_dict=False) == (cpu_sample,)
    assert calls == [tpu_sample]


@pytest.mark.parametrize(
    "shape, affine",
    [
        ((2, 64, 8, 8), True),
        ((2, 64, 8, 8), False),
        ((1, 32, 17), True),
        ((4, 128), True),
    ],
)
def test_tpu_group_norm_matches_reference(shape, affine):
    """The fused-reduce GroupNorm matches F.group_norm (fp32 and bf16)."""
    from vllm_torchtpu.omni.patches import _tpu_group_norm

    g = torch.Generator().manual_seed(0)
    x = torch.randn(*shape, generator=g) * 2 + 0.5
    c = shape[1]
    weight = torch.rand(c, generator=g) + 0.5 if affine else None
    bias = torch.rand(c, generator=g) - 0.5 if affine else None

    ref = torch.nn.functional.group_norm(x, 32, weight, bias, 1e-6)
    out = _tpu_group_norm(x, 32, weight, bias, 1e-6)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    xb = x.to(torch.bfloat16)
    wb = weight.to(torch.bfloat16) if affine else None
    bb = bias.to(torch.bfloat16) if affine else None
    out_b = _tpu_group_norm(xb, 32, wb, bb, 1e-6)
    assert out_b.dtype == torch.bfloat16
    torch.testing.assert_close(out_b.float(), ref, rtol=2e-2, atol=2e-2)


def test_group_norm_patch_dispatches_on_tpu_only(monkeypatch):
    from vllm_torchtpu.omni import patches as omni_patches

    orig_group_norm = _unwrap_tpu_patch(torch.nn.functional.group_norm)
    monkeypatch.setattr(torch.nn.functional, "group_norm", orig_group_norm)
    calls = []
    monkeypatch.setattr(
        omni_patches,
        "_tpu_group_norm",
        lambda *a: calls.append(a) or torch.zeros(1),
    )

    omni_patches._patch_group_norm_tpu()
    omni_patches._patch_group_norm_tpu()  # idempotent
    assert _unwrap_tpu_patch(torch.nn.functional.group_norm) is orig_group_norm

    x = torch.randn(2, 64, 4, 4)
    torch.testing.assert_close(
        torch.nn.functional.group_norm(x, 32), orig_group_norm(x, 32)
    )
    assert calls == []

    tpu_x = MagicMock(spec=torch.Tensor)
    tpu_x.device = MagicMock()
    tpu_x.device.type = "tpu"
    tpu_x.dim.return_value = 4
    torch.nn.functional.group_norm(tpu_x, 32, eps=1e-6)
    assert calls == [(tpu_x, 32, None, None, 1e-6)]


def test_worker_proc_async_output_loop_zero_gpu_id():
    from vllm_omni.diffusion.worker import diffusion_worker

    apply_omni_worker_patches()
    proc = diffusion_worker.WorkerProc(gpu_id=5)
    seen_gpu_id = proc._async_output_loop()
    assert seen_gpu_id == 0
    assert proc.gpu_id == 5
    assert proc._async_output_pending == 0


def test_set_device_installs_worker_patches(monkeypatch):
    # DiffusionWorker.__init__ calls init_device() -> set_device before
    # WorkerProc starts the DiffusionAsyncOutput thread, on both the default
    # and the custom_pipeline_args (skip_load_model) paths. set_device must
    # therefore be the place the _async_output_loop patch is installed.
    from vllm_omni.diffusion.worker import diffusion_worker

    orig_loop = _unwrap_tpu_patch(diffusion_worker.WorkerProc._async_output_loop)
    monkeypatch.setattr(diffusion_worker.WorkerProc, "_async_output_loop", orig_loop)
    monkeypatch.setattr(
        "vllm_torchtpu.omni.platform.apply_omni_worker_patches",
        apply_omni_worker_patches.__wrapped__,
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        staticmethod(lambda world_size: None),
    )
    monkeypatch.setattr(torch, "accelerator", MagicMock(), raising=False)
    assert not getattr(
        diffusion_worker.WorkerProc._async_output_loop, "_is_tpu_patch_wrapper", False
    )
    with patch("torch.empty"), patch("torch.device"):
        OmniTpuPlatform.set_device(None)
    assert diffusion_worker.WorkerProc._async_output_loop._is_tpu_patch_wrapper
    assert diffusion_worker.WorkerProc(gpu_id=3)._async_output_loop() == 0


@pytest.mark.parametrize(
    ("model_class_name", "ulysses_degree", "ulysses_mode", "expected"),
    [
        # 30 heads % 8 != 0 and user left the default -> promoted.
        ("ZImagePipeline", 8, "strict", "advanced_uaa"),
        ("ZImagePipeline", 4, "strict", "advanced_uaa"),
        # Divisors of 30 stay in strict mode.
        ("ZImagePipeline", 2, "strict", "strict"),
        ("ZImagePipeline", 6, "strict", "strict"),
        # No sequence parallelism -> untouched.
        ("ZImagePipeline", 1, "strict", "strict"),
        # An explicit user choice is never overridden.
        ("ZImagePipeline", 8, "advanced_uaa", "advanced_uaa"),
        # Models outside the table keep vLLM-Omni's default behaviour.
        ("FluxPipeline", 8, "strict", "strict"),
        (None, 8, "strict", "strict"),
    ],
)
def test_init_diffusion_model_runner_runtime_ulysses_mode(
    model_class_name, ulysses_degree, ulysses_mode, expected
):
    from types import SimpleNamespace

    od_config = SimpleNamespace(
        model_class_name=model_class_name,
        parallel_config=SimpleNamespace(
            ulysses_degree=ulysses_degree, ulysses_mode=ulysses_mode
        ),
    )
    OmniTpuPlatform.init_diffusion_model_runner_runtime(
        vllm_config=None, od_config=od_config, device=torch.device("cpu")
    )
    assert od_config.parallel_config.ulysses_mode == expected


def test_init_diffusion_model_runner_runtime_without_parallel_config():
    from types import SimpleNamespace

    # Configs without a parallel_config (or with None) are left alone.
    OmniTpuPlatform.init_diffusion_model_runner_runtime(
        vllm_config=None,
        od_config=SimpleNamespace(model_class_name="ZImagePipeline"),
        device=torch.device("cpu"),
    )
