# SPDX-License-Identifier: Apache-2.0
"""100% whole-module coverage tests for tokenizers.py."""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

if "vllm_torchtpu" not in sys.modules:
    _pkg = types.ModuleType("vllm_torchtpu")
    _pkg.__path__ = [str(Path(_SRC_DIR, "vllm_torchtpu"))]
    sys.modules["vllm_torchtpu"] = _pkg

if "vllm.tokenizers.protocol" not in sys.modules:
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules.setdefault("vllm.tokenizers", types.ModuleType("vllm.tokenizers"))
    _proto_mod = types.ModuleType("vllm.tokenizers.protocol")
    _proto_mod.TokenizerLike = object  # type: ignore[attr-defined]
    sys.modules["vllm.tokenizers.protocol"] = _proto_mod

from vllm_torchtpu import envs, tokenizers  # noqa: E402


class _BaseTok:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, dict[str, object]]] = []

    def apply_chat_template(
        self, messages: object, tools: object = None, **kwargs: object
    ) -> str:
        self.calls.append((messages, tools, kwargs))
        return "ok"


def test_tokenizers_full_coverage(monkeypatch) -> None:
    tokenizers._SUBCLASS_CACHE.clear()
    base1 = _BaseTok()
    wrapped1 = tokenizers._tpu_deepseek_v4_tokenizer(base1)
    # Second call hits _SUBCLASS_CACHE[base_cls]
    base2 = _BaseTok()
    wrapped2 = tokenizers._tpu_deepseek_v4_tokenizer(base2)
    assert type(wrapped1) is type(wrapped2)

    # 1. No kwargs at all -> fast positional return
    assert wrapped1.apply_chat_template([{"role": "assistant", "content": "a"}]) == "ok"
    assert base1.calls[-1][2] == {}

    # 2. continue_final_message=False only -> popped, kwargs becomes empty
    assert (
        wrapped1.apply_chat_template(
            [{"role": "assistant", "content": "a"}],
            continue_final_message=False,
        )
        == "ok"
    )
    assert base1.calls[-1][2] == {}

    # 3. continue_final_message=False with extra kwargs -> forwards remaining kwargs
    assert (
        wrapped1.apply_chat_template(
            [{"role": "assistant", "content": "a"}],
            continue_final_message=False,
            add_generation_prompt=True,
        )
        == "ok"
    )
    assert base1.calls[-1][2] == {"add_generation_prompt": True}

    # 4. continue_final_message=True with trailing assistant message
    assert (
        wrapped1.apply_chat_template(
            [{"role": "assistant", "content": "a"}],
            continue_final_message=True,
        )
        == "ok"
    )
    conv_last = base1.calls[-1][2]["conversation"]
    assert isinstance(conv_last, list) and conv_last[-1]["wo_eos"] is True

    # 5. continue_final_message=True with trailing user message
    assert (
        wrapped1.apply_chat_template(
            [{"role": "user", "content": "u"}],
            continue_final_message=True,
        )
        == "ok"
    )
    conv_user = base1.calls[-1][2]["conversation"]
    assert isinstance(conv_user, list) and "wo_eos" not in conv_user[-1]

    # 6. continue_final_message=True with empty conversation
    assert wrapped1.apply_chat_template([], continue_final_message=True) == "ok"
    assert base1.calls[-1][2]["conversation"] == []

    # 7. Pickle round-trip via __reduce__
    unpickled = pickle.loads(pickle.dumps(wrapped1))
    assert type(unpickled) is type(wrapped1)

    # 8. TpuDeepseekV4Tokenizer.from_pretrained
    fake_dsv4_mod = types.ModuleType("vllm.tokenizers.deepseek_v4")
    fake_dsv4_mod.DeepseekV4Tokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **k: _BaseTok()
    )
    with patch.dict(sys.modules, {"vllm.tokenizers.deepseek_v4": fake_dsv4_mod}):
        inst = tokenizers.TpuDeepseekV4Tokenizer.from_pretrained("model-id")
        assert type(inst) is type(wrapped1)

    # 9. register_tokenizers when TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE is False vs True
    fake_reg = MagicMock()
    fake_reg_mod = types.ModuleType("vllm.tokenizers.registry")
    fake_reg_mod.TokenizerRegistry = fake_reg  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"vllm.tokenizers.registry": fake_reg_mod}):
        envs.__dict__.pop("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", None)
        monkeypatch.setenv("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", "0")
        tokenizers.register_tokenizers()
        fake_reg.register.assert_not_called()

        envs.__dict__.pop("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", None)
        monkeypatch.setenv("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", "1")
        tokenizers.register_tokenizers()
        fake_reg.register.assert_called_once_with(
            "deepseek_v4", tokenizers.__name__, "TpuDeepseekV4Tokenizer"
        )
