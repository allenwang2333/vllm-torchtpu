"""The few-token kernel's scores entry takes the served layer: the FP8 checkpoint keeps its MoE scales under
``w13_weight_scale`` / ``w2_weight_scale`` (only block-quantised ones use ``_inv``).
Both scale names must engage the kernel. Runs on the CPU under fake tensors."""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def _layer(scale_name):
    tables = {
        k: torch.zeros(512, dtype=torch.int32)
        for k in ("_lowtok_gid", "_lowtok_prim", "_lowtok_copy")
    }
    tables.update({
        k: torch.full((128, ), -1, dtype=torch.int32)
        for k in ("_lowtok_hot", "_lowtok_hprim", "_lowtok_hcopy")
    })
    tables["_lowtok_hot1h"] = torch.zeros(512, 128, dtype=torch.bfloat16)
    tables["_lowtok_slot"] = torch.full((512, ), -1, dtype=torch.int32)
    layer = SimpleNamespace(renormalize=True,
                            scoring_func="softmax",
                            custom_routing_function=None,
                            w13_bias=None,
                            w2_bias=None,
                            _experts_start=torch.tensor(192,
                                                        dtype=torch.int32),
                            _lowtok_owned=64,
                            moe_config=SimpleNamespace(experts_per_token=10,
                                                       ep_size=8),
                            **tables)
    layer.w13_weight = torch.empty(64, 4096, 2048, dtype=torch.float8_e4m3fn)
    layer.w2_weight = torch.empty(64, 1024, 4096, dtype=torch.float8_e4m3fn)
    setattr(layer, f"w13_{scale_name}", torch.empty(64, 1, 1, 2048))
    setattr(layer, f"w2_{scale_name}", torch.empty(64, 1, 1, 4096))
    return layer


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_scores_entry_takes_the_served_layer(monkeypatch, scale_name):
    import vllm_torchtpu.layers.vllm.moe_lowtok_scores_op as scores_op
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm.quantization import fp8
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_FROM_SCORES", 1, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 32, raising=False)
    monkeypatch.setattr(
        envs, "TPU_MOE_SKIP_PADDED_TOKENS", 1,
        raising=False)  # the served setting: no longer a refusal
    scores_op.prebuild(10, 8, 64)
    with FakeTensorMode():
        layer = _layer(scale_name)
        x = torch.empty(4, 4096, dtype=torch.bfloat16)
        logits = torch.empty(4, 512, dtype=torch.bfloat16)
        out = fp8._lowtok_scores_output(layer, x, logits, "silu")
        assert out is not None and tuple(out.shape) == (4, 4096)
        ids = torch.zeros(4, 10, dtype=torch.int32)
        assert tuple(fp8._lowtok_lift_tables(layer, ids).shape) == (
            4, 10)  # the grouped side lifts the tables too
        big = torch.empty(64, 4096, dtype=torch.bfloat16)
        assert fp8._lowtok_scores_output(
            layer, big, torch.empty(64, 512, dtype=torch.bfloat16),
            "silu") is None
    assert fp8._lowtok_scores_static_refusal(layer, "silu") is None
    assert fp8._lowtok_scores_static_refusal(layer,
                                             "gelu") == "activation gelu"


def test_the_traced_guard_has_no_logging():
    """Dynamo refuses logging calls inside the compiled forward."""
    import inspect

    from vllm_torchtpu.layers.vllm.quantization import fp8
    for fn in (fp8._lowtok_scores_output, fp8._lowtok_scores_static_refusal):
        assert "logger." not in inspect.getsource(
            fn) and "print(" not in inspect.getsource(fn)
