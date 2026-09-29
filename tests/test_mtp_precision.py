# SPDX-License-Identifier: Apache-2.0
"""TL;DR：保护 oracle 自身的拒绝回滚、causal mask 和逐步覆盖契约。

CPU 执行：python -m pytest tests/test_mtp_precision.py -q
这些长期 case 保护参考实现，无法替代真实 TPU 编译/模型端到端验证。
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "mtp_precision",
    Path(__file__).parent / "entrypoints/llm/mtp_precision.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DenseHistory = _MODULE.DenseHistory
StepCoverage = _MODULE.StepCoverage
check_logits = _MODULE.check_logits

pytestmark = pytest.mark.cpu_test


def test_rejection_overwrites_and_truncates_independent_history():
    history = DenseHistory()
    key = torch.arange(5.0).reshape(5, 1, 1)
    history.update(torch.arange(5), key, key + 10)
    replacement = torch.tensor([[[42.0]]])
    keys, values = history.update(torch.tensor([2]), replacement, replacement + 10)
    assert keys.flatten().tolist() == [0, 1, 42]
    assert values.flatten().tolist() == [10, 11, 52]
    replacement.fill_(-1)
    assert keys[-1].item() == 42
    with pytest.raises(AssertionError, match="gap"):
        history.update(torch.tensor([4]), replacement, replacement)


def test_chunked_queries_cannot_attend_future_tokens():
    history = DenseHistory()
    query = torch.ones(2, 2, 1)
    keys = torch.ones(2, 1, 1)
    values = torch.tensor([[[2.0]], [[6.0]]])
    first = history.attend(query, keys, values, torch.arange(2), scale=1.0)
    torch.testing.assert_close(first[:, 0, 0], torch.tensor([2.0, 4.0]))
    later = history.attend(query, keys, values + 8, torch.tensor([2, 3]), scale=1.0)
    torch.testing.assert_close(later[:, 0, 0], torch.tensor([6.0, 8.0]))


def test_later_call_cache_corruption_changes_reference_logits():
    reference, corrupted = DenseHistory(), DenseHistory()
    qkv = torch.ones(1, 1, 1)
    for history in (reference, corrupted):
        history.attend(qkv, qkv, qkv, torch.tensor([0]), scale=1.0)
    corrupted.values[0] = 9
    actual = corrupted.attend(qkv, qkv, qkv, torch.tensor([1]), scale=1.0)
    expected = reference.attend(qkv, qkv, qkv, torch.tensor([1]), scale=1.0)
    assert check_logits(actual, expected, atol=0.01, rtol=0.01, context="step=1")


def test_every_request_round_requires_all_k3_steps():
    coverage = StepCoverage(3)
    for step in range(3):
        for req in ("first", "second"):
            coverage.add(0, req, step)
    assert coverage.finish() == 2
    coverage.add(1, "first", 0)
    coverage.add(1, "first", 1)
    with pytest.raises(AssertionError, match="missing draft steps"):
        coverage.finish()


def test_logits_gate_checks_nonwinning_vocab_entries_and_nonfinite():
    reference = torch.tensor([10.0, 1.0, 0.0])
    actual = torch.tensor([10.0, 1.0, 2.0])
    assert actual.argmax() == reference.argmax()
    assert check_logits(actual, reference, atol=0.1, rtol=0.01, context="step=2")
    actual[-1] = float("nan")
    assert "non-finite" in check_logits(
        actual, reference, atol=0.1, rtol=0.01, context="step=2"
    )


def test_empty_or_out_of_order_capture_never_passes():
    coverage = StepCoverage(3)
    with pytest.raises(AssertionError, match="no real MTP"):
        coverage.finish()
    with pytest.raises(AssertionError, match="expected draft step 0"):
        coverage.add(0, "request", 1)


def test_fp8_kv_reference_preserves_static_scale_and_saturation():
    values = torch.tensor([0.125, 1.125, 10000.0, -10000.0])
    result = _MODULE.kv_roundtrip(values, torch.float8_e4m3fn, 0.5)
    torch.testing.assert_close(result, torch.tensor([0.125, 1.125, 224.0, -224.0]))


def test_replay_uses_reference_carry_and_checks_later_steps(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace

    def context(*args, **kwargs):
        return contextlib.nullcontext()

    for module, exports in (
        ("vllm.compilation.backends", {"set_model_tag": context}),
        ("vllm.forward_context", {"set_forward_context": context}),
        (
            "vllm_torchtpu.models.vllm.vllm_model_wrapper_context",
            {"set_vllm_model_wrapper_context": context},
        ),
    ):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**exports))

    def unused(*args):
        raise AssertionError("production attention must not be used by the oracle")

    attention = SimpleNamespace(
        forward=unused,
        num_heads=1,
        num_kv_heads=1,
        head_size=1,
        impl=SimpleNamespace(scale=1.0, kv_cache_quantized_dtype=None),
    )
    seen_hidden = []

    class Model:
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))]
        )

        def __call__(self, input_ids, positions, hidden_states, spec_step_idx):
            seen_hidden.append(hidden_states.clone())
            query = torch.ones_like(hidden_states)
            return hidden_states + attention.forward(query, query, hidden_states)

        def compute_logits(self, hidden):
            return torch.cat((hidden, -hidden), dim=-1)

    proposer = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        draft_model=Model(),
        draft_vllm_config=None,
        runner=SimpleNamespace(device="cpu", mesh=None),
    )
    recorder = _MODULE._Recorder(proposer, atol=1e-5, rtol=1e-5)
    # 正确 hidden 递推为 1 -> 2 -> 3.5 -> 17/3；只损坏第二次actual logits。
    for step, actual in enumerate((2.0, 99.0, 17 / 3)):
        recorder.coverage.add(0, "req", step)
        path = Path(recorder.directory.name) / f"{step}.pt"
        torch.save(
            {
                "round": 0,
                "step": step,
                "req_ids": ["req"],
                "qsl": [0, 1],
                "positions": torch.tensor([step]),
                "input_ids": torch.tensor([0]),
                "hidden": torch.tensor([[1.0 if step == 0 else 999.0]]),
                "gathers": torch.tensor([0]),
                "tokens": torch.tensor([0]),
                "logits": torch.tensor([[actual, -actual]]),
            },
            path,
        )
        recorder.paths.append(path)
    try:
        summary = recorder.finish()
        assert summary["checked_request_steps"] == 3
        assert summary["max_abs_error"] == pytest.approx(95.5)
        assert summary["max_tolerance_ratio"] > 1
        assert len(summary["failures"]) == 1
        assert "step=1" in summary["failures"][0]
        assert [x.item() for x in seen_hidden] == [1.0, 2.0, 3.5]
        assert attention.forward is unused
        assert not any(Path(recorder.directory.name).iterdir())
    finally:
        recorder.directory.cleanup()
