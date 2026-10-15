# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_toy_proxy_server = _load_toy_proxy_server()
_AsciiSafeStreamEncoder = _toy_proxy_server._AsciiSafeStreamEncoder
_replace_prompt_with_rendered_token_ids = (
    _toy_proxy_server._replace_prompt_with_rendered_token_ids
)


def test_ascii_safe_stream_encoder_handles_split_utf8() -> None:
    encoder = _AsciiSafeStreamEncoder()
    text = "\u044f\U0001f600"
    message = f'data: {{"choices": [{{"text": "{text}"}}]}}\n\n'
    encoded = message.encode("utf-8")
    split_at = encoded.index(text.encode("utf-8")) + 1

    output = encoder.encode(encoded[:split_at])
    output += encoder.encode(encoded[split_at:])
    output += encoder.encode(final=True)

    assert output.isascii()
    assert b"\\u044f" in output
    assert b"\\ud83d\\ude00" in output

    payload = output.decode("ascii").removeprefix("data: ").strip()
    assert json.loads(payload)["choices"][0]["text"] == text


def test_ascii_safe_stream_encoder_leaves_ascii_unchanged() -> None:
    encoder = _AsciiSafeStreamEncoder()
    message = b'data: {"choices": [{"text": "ok"}]}\n\n'

    assert encoder.encode(message) == message
    assert encoder.encode(final=True) == b""


def test_replace_prompt_with_rendered_token_ids_single_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
        "prompt_embeds": b"unused-after-render",
    }
    rendered = [{"token_ids": [1, 2, 3]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [1, 2, 3]
    assert "prompt_embeds" not in converted
    assert request["prompt"] == "hello"


def test_replace_prompt_with_rendered_token_ids_multi_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": ["hello", "world"],
    }
    rendered = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 5]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [[1, 2, 3], [4, 5]]


def test_replace_prompt_with_rendered_token_ids_rejects_bad_response() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
    }

    try:
        _replace_prompt_with_rendered_token_ids(request, [{"token_ids": ["1"]}])
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("malformed render response should fail")


def test_attach_reused_chat_prompt_token_ids(monkeypatch) -> None:
    attach = _toy_proxy_server._attach_reused_chat_prompt_token_ids
    kv_orig = {"uuid": "u-123", "num_tokens": 5}
    resp_json = {
        "prompt_token_ids": [10, 20, 30, 40, 50],
        "kv_transfer_params": kv_orig,
    }
    monkeypatch.setenv("PROXY_REUSE_CHAT_TOKEN_IDS", "1")
    kv_out, n_prompt = attach("/v1/chat/completions", resp_json, kv_orig)
    assert n_prompt == 5
    assert kv_out is not None and kv_out is not kv_orig
    assert kv_out["prompt_token_ids"] == [10, 20, 30, 40, 50]
    assert "prompt_token_ids" not in kv_orig

    # Disabled when PROXY_REUSE_CHAT_TOKEN_IDS=0
    monkeypatch.setenv("PROXY_REUSE_CHAT_TOKEN_IDS", "0")
    kv_out_off, n_prompt_off = attach("/v1/chat/completions", resp_json, kv_orig)
    assert n_prompt_off == 5
    assert kv_out_off == kv_orig
    assert "prompt_token_ids" not in (kv_out_off or {})

