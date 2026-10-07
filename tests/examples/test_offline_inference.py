# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the chat-template default in examples/offline_inference.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _load_offline_inference():
    path = Path(__file__).resolve().parents[2] / "examples" / "offline_inference.py"
    spec = importlib.util.spec_from_file_location("offline_inference", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


offline_inference = _load_offline_inference()


class _FakeTokenizer:
    """The tokenizer attributes that the BOS check reads."""

    def __init__(
        self, *, bos_token, raw_token_ids, rendered_chat_prompt, bos_token_id=2
    ):
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id
        self.chat_template = None if rendered_chat_prompt is None else "<template>"
        self._raw_token_ids = raw_token_ids
        self._rendered_chat_prompt = rendered_chat_prompt
        self.chat_template_calls = []

    def encode(self, text):
        return self._raw_token_ids

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        self.chat_template_calls.append(kwargs)
        if isinstance(self._rendered_chat_prompt, Exception):
            raise self._rendered_chat_prompt
        return self._rendered_chat_prompt


# Gemma-4 instruction-tuned: raw text gets no BOS, the chat template adds it.
GEMMA4_IT_TOKENIZER = _FakeTokenizer(
    bos_token="<bos>",
    raw_token_ids=[1234],
    rendered_chat_prompt="<bos><|turn>user\nx<turn|>\n",
)
# Llama-3: the tokenizer adds BOS to raw text too.
LLAMA3_TOKENIZER = _FakeTokenizer(
    bos_token="<|begin_of_text|>",
    raw_token_ids=[2, 1234],
    rendered_chat_prompt="<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
)
# Qwen3: no BOS token at all.
QWEN3_TOKENIZER = _FakeTokenizer(
    bos_token=None, raw_token_ids=[1234], rendered_chat_prompt="<|im_start|>user"
)
# A base model: BOS token never added to raw text, and no chat template.
NO_CHAT_TEMPLATE_TOKENIZER = _FakeTokenizer(
    bos_token="<bos>", raw_token_ids=[1234], rendered_chat_prompt=None
)
# A chat template that does not start with BOS.
TEMPLATE_WITHOUT_BOS_TOKENIZER = _FakeTokenizer(
    bos_token="<s>", raw_token_ids=[1234], rendered_chat_prompt="<|im_start|>user\nx"
)
# A BOS string without a BOS id (not a real special token).
NO_BOS_ID_TOKENIZER = _FakeTokenizer(
    bos_token="<bos>",
    raw_token_ids=[1234],
    rendered_chat_prompt="<bos>x",
    bos_token_id=None,
)
# A chat template that fails to render, e.g. because it needs extra kwargs.
RAISING_TEMPLATE_TOKENIZER = _FakeTokenizer(
    bos_token="<bos>",
    raw_token_ids=[1234],
    rendered_chat_prompt=ValueError("template needs tools"),
)


def _fake_llm(tokenizer):
    return SimpleNamespace(get_tokenizer=lambda: tokenizer)


@pytest.mark.parametrize(
    ("tokenizer", "expected"),
    [
        (GEMMA4_IT_TOKENIZER, True),
        (LLAMA3_TOKENIZER, False),
        (QWEN3_TOKENIZER, False),
        (NO_CHAT_TEMPLATE_TOKENIZER, False),
        (TEMPLATE_WITHOUT_BOS_TOKENIZER, False),
        (NO_BOS_ID_TOKENIZER, False),
        (RAISING_TEMPLATE_TOKENIZER, False),
    ],
    ids=[
        "gemma4-it",
        "llama3",
        "qwen3",
        "no-chat-template",
        "template-without-bos",
        "no-bos-id",
        "template-raises",
    ],
)
def test_chat_template_adds_missing_bos(tokenizer, expected):
    assert offline_inference._chat_template_adds_missing_bos(tokenizer, {}) is expected


def test_probe_renders_with_the_chat_template_kwargs():
    tokenizer = _FakeTokenizer(
        bos_token="<bos>", raw_token_ids=[1234], rendered_chat_prompt="<bos>x"
    )
    kwargs = {"enable_thinking": False}

    assert offline_inference._chat_template_adds_missing_bos(tokenizer, kwargs)

    assert tokenizer.chat_template_calls == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    ]


def test_probe_failure_falls_back_to_raw_prompts(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(offline_inference, "logger", logger)

    resolved = offline_inference._resolve_use_chat_template(
        None, _fake_llm(RAISING_TEMPLATE_TOKENIZER), {}
    )

    assert resolved is False
    logger.warning.assert_called_once()
    logger.info.assert_not_called()


@pytest.mark.parametrize(
    ("use_chat_template", "tokenizer", "expected"),
    [
        (None, GEMMA4_IT_TOKENIZER, True),
        (None, LLAMA3_TOKENIZER, False),
        (None, QWEN3_TOKENIZER, False),
        (False, GEMMA4_IT_TOKENIZER, False),
        (True, QWEN3_TOKENIZER, True),
    ],
    ids=[
        "default-gemma4-it",
        "default-llama3",
        "default-qwen3",
        "no-use-chat-template-gemma4-it",
        "use-chat-template-qwen3",
    ],
)
def test_resolve_use_chat_template(use_chat_template, tokenizer, expected):
    resolved = offline_inference._resolve_use_chat_template(
        use_chat_template, _fake_llm(tokenizer), {}
    )
    assert resolved is expected


def test_resolve_warns_when_raw_prompts_would_miss_bos(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(offline_inference, "logger", logger)

    offline_inference._resolve_use_chat_template(
        False, _fake_llm(GEMMA4_IT_TOKENIZER), {}
    )

    logger.warning.assert_called_once()


def test_resolve_does_not_load_tokenizer_when_chat_template_requested():
    def get_tokenizer():
        raise AssertionError("the tokenizer should not be needed")

    llm = SimpleNamespace(get_tokenizer=get_tokenizer)
    assert offline_inference._resolve_use_chat_template(True, llm, {}) is True


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--use-chat-template"], True),
        (["--no-use-chat-template"], False),
    ],
    ids=["default", "use-chat-template", "no-use-chat-template"],
)
def test_parser_use_chat_template_flag(monkeypatch, argv, expected):
    # EngineArgs.add_cli_args builds a DeviceConfig, which needs the TPU torch
    # device; only the example's own flags matter here.
    monkeypatch.setattr(offline_inference.EngineArgs, "add_cli_args", lambda _: None)
    args = offline_inference.create_parser().parse_args(argv)
    assert args.use_chat_template is expected
