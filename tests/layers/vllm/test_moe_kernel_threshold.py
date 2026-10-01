"""The two MoE kernels meet at one row count: settings that leave steps
between them are refused at startup."""
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm import fused_moe_ep_v3


def _set(monkeypatch, cap, gate, sp_start=0):
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", cap, raising=False)
    monkeypatch.setattr(envs,
                        "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS",
                        gate,
                        raising=False)
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", sp_start, raising=False)


@pytest.mark.parametrize("threshold", [128, 256, 512])
def test_recipe_settings_meet(monkeypatch, threshold):
    # data parallel: the gate is the first row count per engine above it
    _set(monkeypatch, threshold, threshold // 8 + 1)
    fused_moe_ep_v3.check_kernel_threshold(ep=8, tp=1)
    # tensor parallel with sequence parallelism from the next row
    _set(monkeypatch, threshold, (threshold + 8) // 8, threshold + 1)
    fused_moe_ep_v3.check_kernel_threshold(ep=8, tp=8)


def test_data_parallel_gap_refused(monkeypatch):
    # A 128-row cap and 48-row gate leave 17-32 rows per engine uncovered.
    _set(monkeypatch, 128, 48)
    with pytest.raises(ValueError, match="leave steps between"):
        fused_moe_ep_v3.check_kernel_threshold(ep=8, tp=1)


def test_tensor_parallel_gap_refused(monkeypatch):
    # A 32-row cap and 1,024-row SP boundary leave 33-1,023 rows uncovered.
    _set(monkeypatch, 32, 128, 1024)
    with pytest.raises(ValueError, match="leave steps between"):
        fused_moe_ep_v3.check_kernel_threshold(ep=8, tp=8)


def test_few_token_kernel_off_is_not_a_gap(monkeypatch):
    _set(monkeypatch, 0, 48)
    fused_moe_ep_v3.check_kernel_threshold(ep=8, tp=1)
