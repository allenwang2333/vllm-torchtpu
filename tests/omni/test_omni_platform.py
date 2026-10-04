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
        (
            8192,
            (2048, 4096, 1024),
        ),  # sp=2 shape: resident (2048, 8192) failed to compile
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

    # First-step index resolution avoids the blocking (timesteps == timestep).nonzero()
    # D2H readback when _step_index and _begin_index are unset on TPU.
    sched = _Scheduler()
    sched._step_index = None
    sched._begin_index = None
    sched.timesteps = torch.tensor([1000.0, 500.0])
    tpu_ts = MagicMock(spec=torch.Tensor)
    tpu_ts.device = MagicMock()
    tpu_ts.device.type = "tpu"
    sched.step(None, tpu_ts, tpu_sample, return_dict=False)
    assert sched._step_index == 0

    # Explicit _begin_index (e.g. img2img) is left untouched.
    sched2 = _Scheduler()
    sched2._step_index = None
    sched2._begin_index = 3
    sched2.timesteps = torch.tensor([1000.0, 500.0])
    sched2.step(None, tpu_ts, tpu_sample, return_dict=False)
    assert sched2._step_index is None


def _install_fake_z_image(monkeypatch, events):
    """Fake ``z_image_transformer`` + ``forward_context`` modules for patch tests."""
    import dataclasses
    from types import SimpleNamespace

    from vllm_torchtpu.omni import patches as omni_patches

    @dataclasses.dataclass(frozen=True)
    class _SPInput:
        split_dim: int = 1
        auto_pad: bool = False

    class _Model:
        _sp_plan = {"unified_prepare": {0: _SPInput(1, False), 1: _SPInput(1, False)}}

        def __init__(self, hooks=None):
            self.unified_prepare = SimpleNamespace(
                _hook_registry=SimpleNamespace(_hooks=hooks)
                if hooks is not None
                else None
            )

        def forward(self, x):
            return x

    class _Block:
        def __init__(self, layer_id: int):
            self.layer_id = layer_id

        def forward(self, x, *args, **kwargs):
            events.append(("forward", self.layer_id))
            return x

    fake_ctx = SimpleNamespace(sp_plan_hooks_applied=False)
    fake_fc = SimpleNamespace(
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: fake_ctx,
    )
    fake_z_img = MagicMock()
    fake_z_img.ZImageTransformerBlock = _Block
    fake_z_img.ZImageTransformer2DModel = _Model
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.models.z_image.z_image_transformer",
        fake_z_img,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.forward_context",
        fake_fc,
    )
    monkeypatch.setattr(
        omni_patches,
        "_tpu_materialize_async",
        lambda t=None: events.append(("flush", t)) or True,
    )
    return _Block, _Model, fake_ctx


def _fake_tpu_block_input():
    tpu_x = MagicMock(spec=torch.Tensor)
    tpu_x.device = MagicMock()
    tpu_x.device.type = "tpu"
    return tpu_x


def test_z_image_transformer_block_flush(monkeypatch):
    """ZImageTransformerBlock.forward flushes setup on 0/1000 and output per block."""
    from vllm_torchtpu.omni import patches as omni_patches

    events = []
    _Block, _Model, fake_ctx = _install_fake_z_image(monkeypatch, events)
    monkeypatch.setattr(omni_patches, "_ZIMAGE_BLOCK_COMPILE", False)

    orig_forward = _Block.forward
    orig_model_forward = _Model.forward
    omni_patches._patch_z_image_transformer_block_flush()
    omni_patches._patch_z_image_transformer_block_flush()  # idempotent
    assert _unwrap_tpu_patch(_Block.forward) is orig_forward
    assert _unwrap_tpu_patch(_Model.forward) is orig_model_forward
    assert all(v.auto_pad for v in _Model._sp_plan["unified_prepare"].values())

    # Without SP hooks on unified_prepare (or after remove_sequence_parallel),
    # sp_plan_hooks_applied is left untouched.
    _Model(hooks=None).forward("x")
    assert fake_ctx.sp_plan_hooks_applied is False
    _Model(hooks={}).forward("x")
    assert fake_ctx.sp_plan_hooks_applied is False

    # With SP hooks registered on unified_prepare, forward restores
    # sp_plan_hooks_applied = True on the step's ForwardContext.
    _Model(hooks={"_sp_input_hook_unified_prepare": object()}).forward("x")
    assert fake_ctx.sp_plan_hooks_applied is True

    tpu_x = _fake_tpu_block_input()
    cpu_x = torch.zeros(1)

    assert _Block(1000).forward(tpu_x) is tpu_x
    assert _Block(0).forward(tpu_x) is tpu_x
    assert _Block(1).forward(tpu_x) is tpu_x
    assert _Block(0).forward(cpu_x) is cpu_x
    assert events == [
        ("flush", None),
        ("forward", 1000),
        ("flush", tpu_x),
        ("flush", None),
        ("forward", 0),
        ("flush", tpu_x),
        ("forward", 1),
        ("flush", tpu_x),
        ("forward", 0),
    ]


def test_z_image_transformer_block_compile(monkeypatch):
    """TPU blocks run through one shared torch.compile(backend="tpu") function.

    CPU inputs bypass it, and a failing compiled call falls back to eager for
    the rest of the process (the block is never compiled again).
    """
    from vllm_torchtpu.omni import patches as omni_patches

    events = []
    _Block, _, _ = _install_fake_z_image(monkeypatch, events)
    monkeypatch.setattr(omni_patches, "_ZIMAGE_BLOCK_COMPILE", True)

    compile_calls = []

    def fake_compile(fn, *, backend, dynamic):
        compile_calls.append((fn, backend, dynamic))

        def compiled(*args, **kwargs):
            events.append(("compiled",))
            return fn(*args, **kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    dynamo_cfg = torch._dynamo.config
    limit_attr = next(
        (a for a in ("recompile_limit", "cache_size_limit") if hasattr(dynamo_cfg, a)),
        None,
    )
    if limit_attr is not None:
        monkeypatch.setattr(dynamo_cfg, limit_attr, 8)

    orig_forward = _Block.forward
    omni_patches._patch_z_image_transformer_block_flush()
    tpu_x = _fake_tpu_block_input()
    cpu_x = torch.zeros(1)

    assert _Block(1000).forward(tpu_x) is tpu_x
    assert _Block(1).forward(tpu_x) is tpu_x
    assert _Block(0).forward(cpu_x) is cpu_x
    # One compiled function (dynamic=False, tpu backend) shared by all blocks.
    assert [(f is orig_forward, b, d) for f, b, d in compile_calls] == [
        (True, "tpu", False)
    ]
    if limit_attr is not None:
        assert getattr(dynamo_cfg, limit_attr) >= 64
    assert events == [
        ("flush", None),
        ("compiled",),
        ("forward", 1000),
        ("flush", tpu_x),
        ("compiled",),
        ("forward", 1),
        ("flush", tpu_x),
        ("forward", 0),
    ]

    # A failing compiled call falls back to eager and disables compilation.
    events.clear()
    compile_calls.clear()

    def failing_compile(fn, *, backend, dynamic):
        compile_calls.append((fn, backend, dynamic))

        def compiled(*args, **kwargs):
            raise RuntimeError("backend exploded")

        return compiled

    monkeypatch.setattr(torch, "compile", failing_compile)
    omni_patches._patch_z_image_transformer_block_flush()  # fresh compile_state
    assert _Block(1).forward(tpu_x) is tpu_x
    assert _Block(2).forward(tpu_x) is tpu_x
    assert len(compile_calls) == 1
    assert events == [
        ("forward", 1),
        ("flush", tpu_x),
        ("forward", 2),
        ("flush", tpu_x),
    ]


def test_z_image_noise_refiner_sp_sharding(monkeypatch):
    """noise_refiner runs sequence-sharded under SP with sp_active raised.

    The refiner inputs (x, attn_mask, cos, sin) are sharded on dim 1 before
    noise_refiner[0], ``_sp_shard_depth`` is 1 only inside the refiner, the
    output of noise_refiner[-1] is gathered back, hooks are installed once per
    model, and sequences that do not divide by the SP size stay unsharded.
    """
    from types import SimpleNamespace

    import torch.nn as nn

    from vllm_torchtpu.omni import patches as omni_patches

    events = []
    _, _, fake_ctx = _install_fake_z_image(monkeypatch, events)
    monkeypatch.setattr(omni_patches, "_ZIMAGE_BLOCK_COMPILE", False)
    fake_ctx._sp_shard_depth = 0
    fake_ctx._sp_equal_pad_stack = []
    world = {"size": 2}
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.distributed.sp_sharding",
        SimpleNamespace(
            get_sequence_parallel_world_size=lambda: world["size"],
            sp_shard=lambda t, dim, validate=True: t.chunk(world["size"], dim)[-1],
            sp_gather=lambda t, dim, validate=True: torch.cat([t] * world["size"], dim),
        ),
    )

    seen = []
    x_views = []

    class _Refiner(nn.Module):
        def forward(self, x, attn_mask, cos, sin, adaln_input=None):
            x_views.append(x._base is not None or x.storage_offset() != 0)
            seen.append(
                (
                    tuple(x.shape),
                    tuple(attn_mask.shape),
                    tuple(cos.shape),
                    tuple(sin.shape),
                    tuple(adaln_input.shape),
                    fake_ctx._sp_shard_depth,
                    list(fake_ctx._sp_equal_pad_stack),
                )
            )
            return x + 1

    class _Model:
        _sp_plan = {}

        def __init__(self):
            self.unified_prepare = SimpleNamespace(
                _hook_registry=SimpleNamespace(_hooks={"sp": object()})
            )
            self.noise_refiner = nn.ModuleList([_Refiner(), _Refiner()])

        def forward(self, x, attn_mask, cos, sin, adaln):
            for layer in self.noise_refiner:
                x = layer(x, attn_mask, cos, sin, adaln)
            return x

    z_img_mod = sys.modules["vllm_omni.diffusion.models.z_image.z_image_transformer"]
    z_img_mod.ZImageTransformer2DModel = _Model
    omni_patches._patch_z_image_transformer_block_flush()

    model = _Model()
    x = torch.zeros(1, 16, 4)
    mask = torch.ones(1, 16, dtype=torch.bool)
    cos, sin = torch.ones(1, 16, 2), torch.ones(1, 16, 2)
    adaln = torch.ones(1, 4)
    out = model.forward(x, mask, cos, sin, adaln)
    assert fake_ctx.sp_plan_hooks_applied is True
    assert tuple(out.shape) == (1, 16, 4) and torch.all(out == 2)
    assert seen == [
        ((1, 8, 4), (1, 8), (1, 8, 2), (1, 8, 2), (1, 4), 1, [True]),
        ((1, 8, 4), (1, 8), (1, 8, 2), (1, 8, 2), (1, 4), 1, [True]),
    ]
    assert fake_ctx._sp_shard_depth == 0 and fake_ctx._sp_equal_pad_stack == []
    # noise_refiner[0] gets a fresh copy of its x shard, never the
    # ``x[:, lo:hi]`` view of the full sequence (the sharded attn_mask /
    # cos / sin stay views).
    assert x_views == [False, False]
    assert [s[1] for s in seen] == [(1, 8), (1, 8)]

    # Hooks are installed once per model instance.
    model.forward(x, mask, cos, sin, adaln)
    assert len(model.noise_refiner[0]._forward_pre_hooks) == 1
    assert len(model.noise_refiner[1]._forward_hooks) == 1

    # A sequence that does not divide by the SP size stays unsharded.
    seen.clear()
    out = model.forward(
        torch.zeros(1, 15, 4), mask[:, :15], cos[:, :15], sin[:, :15], adaln
    )
    assert tuple(out.shape) == (1, 15, 4)
    assert [s[0] for s in seen] == [(1, 15, 4), (1, 15, 4)]
    assert [s[5] for s in seen] == [0, 0]

    # SP size 1: no sharding, no depth change.
    seen.clear()
    world["size"] = 1
    model = _Model()
    out = model.forward(x, mask, cos, sin, adaln)
    assert tuple(out.shape) == (1, 16, 4)
    assert [s[5] for s in seen] == [0, 0]


def test_tpu_prompt_encoder_build_mask():
    from vllm_torchtpu.omni.patches import _TpuCompiledPromptEncoder

    mask = _TpuCompiledPromptEncoder.build_mask(
        torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    )
    assert mask.shape == (2, 1, 4, 4) and mask.dtype == torch.bool
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    # Right padding: causal & key padding; padded query rows keep the diagonal.
    assert torch.equal(
        mask[0, 0],
        causal & torch.tensor([True, True, False, False])
        | torch.eye(4, dtype=torch.bool),
    )
    # Left padding: the first (padded) row is not fully masked.
    assert mask[1, 0, 0].tolist() == [True, False, False, False]
    assert torch.equal(
        mask[1, 0, 1:], causal[1:] & torch.tensor([False, True, True, True])
    )


def test_z_image_prompt_encoder_patch(monkeypatch):
    """ZImagePipeline._encode_prompt uses the compiled encoder on TPU only."""
    from types import SimpleNamespace

    from vllm_torchtpu.omni import patches as omni_patches

    calls = []
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)

    class _Tokenizer:
        def __init__(self, masks):
            self._masks = masks

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages[0]["content"], kwargs))
            return "<chat>" + messages[0]["content"]

        def __call__(self, prompts, **kwargs):
            calls.append(("tokenize", list(prompts), kwargs))
            ids = MagicMock()
            ids.to.return_value = "ids_on_tpu"
            return SimpleNamespace(input_ids=ids, attention_mask=self._masks)

    class _Pipeline:
        def __init__(self, masks):
            self.tokenizer = _Tokenizer(masks)
            self.text_encoder = object()
            self._execution_device = SimpleNamespace(type="tpu")

        def _encode_prompt(
            self, prompt, device=None, prompt_embeds=None, max_sequence_length=512
        ):
            calls.append(("orig", prompt, device, prompt_embeds, max_sequence_length))
            return "orig_result"

    class _Encoder:
        fail = False

        def __init__(self, text_encoder):
            calls.append(("build", text_encoder))

        def __call__(self, input_ids, attention_mask):
            calls.append(("encode", input_ids, attention_mask))
            if _Encoder.fail:
                raise RuntimeError("boom")
            return hidden

    fake_mod = MagicMock()
    fake_mod.ZImagePipeline = _Pipeline
    monkeypatch.setitem(
        sys.modules, "vllm_omni.diffusion.models.z_image.pipeline_z_image", fake_mod
    )
    monkeypatch.setattr(omni_patches, "_TpuCompiledPromptEncoder", _Encoder)

    orig = _Pipeline._encode_prompt
    omni_patches._patch_z_image_prompt_encoder_tpu()
    omni_patches._patch_z_image_prompt_encoder_tpu()  # idempotent
    assert _unwrap_tpu_patch(_Pipeline._encode_prompt) is orig

    masks = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0]])
    pipe = _Pipeline(masks)
    out = pipe._encode_prompt(["a dog", "a cat"], max_sequence_length=5)
    assert len(out) == 2
    assert torch.equal(out[0], hidden[0, :3])  # contiguous mask -> static slice
    assert torch.equal(out[1], hidden[1][[0, 2]])  # gaps -> boolean index
    assert calls[0] == ("build", pipe.text_encoder)
    assert calls[1][:2] == ("template", "a dog") and calls[1][2] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert calls[3] == (
        "tokenize",
        ["<chat>a dog", "<chat>a cat"],
        {
            "padding": "max_length",
            "max_length": 5,
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    assert calls[4] == ("encode", "ids_on_tpu", masks)
    assert not any(c[0] == "orig" for c in calls)
    # The encoder is built once per pipeline.
    pipe._encode_prompt("again", max_sequence_length=5)
    assert sum(c[0] == "build" for c in calls) == 1

    # Non-TPU devices and precomputed embeddings go straight to upstream.
    calls.clear()
    assert pipe._encode_prompt("x", device=torch.device("cpu")) == "orig_result"
    assert pipe._encode_prompt("x", prompt_embeds=["e"]) == "orig_result"
    assert [c[0] for c in calls] == ["orig", "orig"]

    # A failing compiled encoder falls back to upstream and stays disabled.
    calls.clear()
    _Encoder.fail = True
    try:
        assert pipe._encode_prompt("y", max_sequence_length=5) == "orig_result"
        assert pipe._tpu_prompt_encoder is False
        assert calls[-1][0] == "orig"
        assert pipe._encode_prompt("z", max_sequence_length=5) == "orig_result"
        assert sum(c[0] == "encode" for c in calls) == 1
    finally:
        _Encoder.fail = False


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


def _reference_rotary(x, cos, sin, interleaved):
    # Mirrors vllm_omni.diffusion.layers.rope.apply_rotary_emb_torch for
    # rotary_dim == head_dim with half-dim cos/sin of shape [seq, head_dim/2].
    if interleaved:
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        cos_full = cos.repeat_interleave(2, dim=-1)
        sin_full = sin.repeat_interleave(2, dim=-1)
    else:
        x1, x2 = x.chunk(2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        cos_full = torch.cat((cos, cos), dim=-1)
        sin_full = torch.cat((sin, sin), dim=-1)
    return x * cos_full.unsqueeze(-2) + rot * sin_full.unsqueeze(-2)


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tpu_apply_rotary_matches_reference(interleaved, dtype):
    """The matmul-based rotate_half is exact (0/+-1 matrix entries)."""
    from vllm_torchtpu.omni.patches import _rope_matrices, _tpu_apply_rotary

    g = torch.Generator().manual_seed(0)
    seq, heads, dim = 16, 4, 32
    x = torch.randn(1, seq, heads, dim, generator=g).to(dtype)
    ang = torch.rand(seq, dim // 2, generator=g) * 6.28
    cos, sin = ang.cos().to(dtype), ang.sin().to(dtype)

    r, e = _rope_matrices(dim, interleaved, dtype, x.device)
    assert r.shape == (dim, dim) and e.shape == (dim // 2, dim)
    assert (r.abs().sum(0) == 1).all() and (r.abs().sum(1) == 1).all()  # permutation
    assert (e.sum(0) == 1).all() and (e.sum(1) == 2).all()

    out = _tpu_apply_rotary(x, cos, sin, interleaved)
    ref = _reference_rotary(x, cos, sin, interleaved)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)


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


def test_vae_image_processor_pil_output_patch(monkeypatch):
    """The torch-native PIL tail is byte-identical to diffusers' numpy/PIL path."""
    import numpy as np
    from PIL import Image

    from vllm_torchtpu.omni import patches as omni_patches

    class _Config:
        do_normalize = True

    class _VaeImageProcessor:
        # diffusers 0.40 ``VaeImageProcessor.postprocess`` minus validation.
        config = _Config()

        @staticmethod
        def denormalize(images):
            return (images * 0.5 + 0.5).clamp(0, 1)

        def postprocess(self, image, output_type="pil", do_denormalize=None):
            if do_denormalize is None:
                do_denormalize = [self.config.do_normalize] * image.shape[0]
            image = torch.stack(
                [
                    self.denormalize(image[i]) if do_denormalize[i] else image[i]
                    for i in range(image.shape[0])
                ]
            )
            if output_type == "pt":
                return image
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            if output_type == "np":
                return image
            image = (image * 255).round().astype("uint8")
            if image.shape[-1] == 1:
                return [Image.fromarray(im.squeeze(), mode="L") for im in image]
            return [Image.fromarray(im) for im in image]

    fake_mod = MagicMock()
    fake_mod.VaeImageProcessor = _VaeImageProcessor
    monkeypatch.setitem(sys.modules, "diffusers.image_processor", fake_mod)

    orig_postprocess = _VaeImageProcessor.postprocess
    torch.manual_seed(0)
    inputs = [
        (torch.rand(2, 3, 16, 24) * 2.4 - 1.2).to(torch.bfloat16),
        torch.rand(1, 3, 8, 8) * 2.4 - 1.2,  # float32
        torch.rand(1, 4, 8, 8) * 2 - 1,  # RGBA
        torch.rand(2, 1, 8, 8) * 2 - 1,  # grayscale
    ]
    refs = [orig_postprocess(_VaeImageProcessor(), x) for x in inputs]
    ref_np = orig_postprocess(_VaeImageProcessor(), inputs[0], output_type="np")

    omni_patches._patch_vae_image_processor_pil_output()
    omni_patches._patch_vae_image_processor_pil_output()  # idempotent
    assert _unwrap_tpu_patch(_VaeImageProcessor.postprocess) is orig_postprocess

    proc = _VaeImageProcessor()
    for x, ref in zip(inputs, refs):
        x_before = x.clone()
        out = proc.postprocess(x)
        assert torch.equal(x, x_before)  # input is not modified in place
        assert len(out) == len(ref)
        for pil_out, pil_ref in zip(out, ref):
            assert isinstance(pil_out, Image.Image)
            assert pil_out.mode == pil_ref.mode
            assert pil_out.size == pil_ref.size
            assert np.array_equal(np.asarray(pil_out), np.asarray(pil_ref))

    # Partial denormalization is forwarded to the original.
    out = proc.postprocess(inputs[0], do_denormalize=[True, False])
    ref = orig_postprocess(proc, inputs[0], do_denormalize=[True, False])
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out, ref))

    # Non-PIL outputs and non-4D inputs go through the original unchanged.
    assert np.array_equal(proc.postprocess(inputs[0], output_type="np"), ref_np)
    assert torch.equal(
        proc.postprocess(inputs[1], output_type="pt"),
        orig_postprocess(proc, inputs[1], output_type="pt"),
    )
    sentinel = object()
    monkeypatch.setattr(
        _VaeImageProcessor, "postprocess", lambda self, *a, **k: sentinel
    )
    omni_patches._patch_vae_image_processor_pil_output()
    assert proc.postprocess(torch.zeros(3, 8, 8)) is sentinel
    assert proc.postprocess("not a tensor") is sentinel


def test_worker_proc_async_output_loop_zero_gpu_id():
    from vllm_omni.diffusion.worker import diffusion_worker

    apply_omni_worker_patches()
    proc = diffusion_worker.WorkerProc(gpu_id=5)
    seen_gpu_id = proc._async_output_loop()
    assert seen_gpu_id == 0
    assert proc.gpu_id == 5
    assert proc._async_output_pending == 0


def test_diffusion_worker_load_model_trims_torch_threads(monkeypatch):
    # vllm_omni spawns diffusion workers with OMP_NUM_THREADS=<all cores>
    # (set_multiprocessing_worker_envs() default local_world_size=1). After
    # load_model the worker must apply vLLM's steady-state thread policy,
    # exactly like TPUWorker.compile_or_warm_up_model does for LLM workers.
    from vllm_omni.diffusion.worker import diffusion_worker

    from vllm_torchtpu.omni import patches as omni_patches

    events: list[str] = []
    monkeypatch.setattr(
        "vllm.utils.torch_utils.set_torch_threads_for_runtime",
        lambda: events.append("trim"),
        raising=False,  # local vLLM may predate the 0.29 helper
    )
    worker_cls = diffusion_worker.DiffusionWorker
    orig_load = _unwrap_tpu_patch(worker_cls.load_model)
    monkeypatch.setattr(worker_cls, "load_model", orig_load)

    omni_patches._patch_diffusion_worker_runtime_threads(diffusion_worker)
    omni_patches._patch_diffusion_worker_runtime_threads(diffusion_worker)
    assert _unwrap_tpu_patch(worker_cls.load_model) is orig_load

    worker = worker_cls()
    assert worker.load_model(load_format="safetensors") == "loaded"
    assert worker.load_calls == [("safetensors", None)]
    assert events == ["trim"]
    worker.load_model(custom_pipeline_name="custom")
    assert worker.load_calls[-1] == ("default", "custom")
    assert events == ["trim", "trim"]


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
