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
"""Kimi-K3 chat prefill: the final assistant turn must be left open.

The stand-in below reimplements the checkpoint's XTML grammar rather than
importing it -- the real ``encoding_k3.py`` ships with the weights and is not
available to CI. It is faithful in the only respect these tests depend on:
``build_chat_segments`` takes ``**kwargs`` and acts on four of them, an
assistant message is emitted closed, and ``add_generation_prompt`` emits the
same opening and stops at the channel tag.

``build_chat_segments`` is module-level on purpose: ``tokenizers._k3_encoder``
identifies a K3 tokenizer by finding that name in the globals of whichever
``apply_chat_template`` implementation the MRO resolves to.
"""
import re
import types

import pytest
from vllm.exceptions import VLLMValidationError

OPEN, CLOSE, SEP, EOM = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"
SPECIALS = (OPEN, CLOSE, SEP, EOM)
_SPLIT_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def build_chat_segments(messages,
                        tools=None,
                        *,
                        add_generation_prompt=True,
                        thinking=True,
                        image_prompts=None,
                        **kwargs):
    """The subset of the checkpoint's encoder these tests exercise.

    Like the real one, it reads only ``tool_choice`` out of ``**kwargs`` (plus
    three others it has no use for here) and drops everything else in silence.
    """
    out = []
    if kwargs.get("tool_choice") == "required":
        out.append(
            f"{OPEN}message role=\"system\"{SEP}tools{CLOSE}message{SEP}{EOM}")
    for message in messages:
        role = message["role"]
        name = f" name=\"{message['name']}\"" if message.get("name") else ""
        out.append(f"{OPEN}message role=\"{role}\"{name}{SEP}")
        if role == "assistant":
            if thinking:
                out.append(f"{OPEN}think{SEP}")
                out.append(message.get("reasoning_content") or "")
                out.append(f"{CLOSE}think{SEP}")
            out.append(f"{OPEN}response{SEP}{message.get('content') or ''}"
                       f"{CLOSE}response{SEP}")
            if message.get("tool_calls"):
                out.append(f"{OPEN}tools{SEP}{CLOSE}tools{SEP}")
        else:
            out.append(message.get("content") or "")
        out.append(f"{CLOSE}message{SEP}{EOM}")
    if add_generation_prompt:
        out.append(f"{OPEN}message role=\"assistant\"{SEP}")
        out.append(f"{OPEN}{'think' if thinking else 'response'}{SEP}")
    return "".join(out)


class FakeK3Tokenizer:
    """The public surface `tokenizers.py` and the vLLM K3 renderer use."""

    def encode(self, text, allow_special_tokens=True, **kwargs):
        assert not kwargs
        pieces = _SPLIT_RE.split(text) if allow_special_tokens else [text]
        return [
            hash(p) % 90_000 if p not in SPECIALS else 100_000 +
            SPECIALS.index(p) for p in pieces if p
        ]

    def apply_chat_template(self,
                            conversation,
                            tools=None,
                            tokenize=False,
                            add_generation_prompt=True,
                            thinking=True,
                            return_dict=False,
                            **kwargs):
        kwargs.setdefault("thinking_effort", "max")
        rendered = build_chat_segments(
            conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            thinking=thinking,
            **kwargs)
        return self.encode(rendered) if tokenize else rendered


QUESTION = {"role": "user", "content": "Q\nA. a\nB. b\nC. c\nD. d"}
PREFILL = {"role": "assistant", "content": "The best answer is"}
MMLU_LLAMA = [QUESTION, PREFILL]
CLOSE_TAIL = f"{CLOSE}response{SEP}{CLOSE}message{SEP}{EOM}"


@pytest.fixture
def wrap():
    """Imported here, not at module scope, so the test below that documents the
    unpatched defect still runs on a tree without the fix."""
    from vllm_torchtpu.tokenizers import _tpu_kimi_k3_tokenizer

    return _tpu_kimi_k3_tokenizer


@pytest.fixture
def stock():
    return FakeK3Tokenizer()


@pytest.fixture
def patched(wrap, stock):
    return wrap(stock)


def test_stock_tokenizer_drops_the_kwarg(stock):
    """The defect, pinned: asking to continue changes nothing at all."""
    asked = stock.apply_chat_template(MMLU_LLAMA,
                                      thinking=False,
                                      add_generation_prompt=False,
                                      continue_final_message=True)
    assert asked == stock.apply_chat_template(MMLU_LLAMA,
                                              thinking=False,
                                              add_generation_prompt=False)
    assert asked.endswith(CLOSE_TAIL)


def test_prefill_turn_is_left_open(patched):
    rendered = patched.apply_chat_template(MMLU_LLAMA,
                                           thinking=False,
                                           add_generation_prompt=False,
                                           continue_final_message=True)
    assert rendered.endswith(f"{OPEN}response{SEP}The best answer is")
    assert CLOSE_TAIL not in rendered


def test_open_form_is_the_closed_form_minus_its_tail(patched, stock):
    """Exactness: continuing is truncation, not a re-render that looks similar."""
    closed = stock.apply_chat_template(MMLU_LLAMA,
                                       tokenize=True,
                                       thinking=False,
                                       add_generation_prompt=False)
    opened = patched.apply_chat_template(MMLU_LLAMA,
                                         tokenize=True,
                                         thinking=False,
                                         add_generation_prompt=False,
                                         continue_final_message=True)
    assert opened == closed[:len(opened)]
    assert closed[len(opened):] == stock.encode(CLOSE_TAIL)


def test_history_is_untouched(patched, stock):
    """Only the final turn changes; earlier turns keep their closed form."""
    history = [
        QUESTION, {
            "role": "assistant",
            "content": "The best answer is A."
        }
    ]
    rendered = patched.apply_chat_template(history + [QUESTION, PREFILL],
                                           thinking=False,
                                           continue_final_message=True)
    prefix = stock.apply_chat_template(history + [QUESTION],
                                       thinking=False,
                                       add_generation_prompt=True)
    assert rendered == prefix + "The best answer is"


def test_reasoning_prefill_uses_the_think_channel(patched):
    conversation = [
        QUESTION, {
            "role": "assistant",
            "reasoning_content": "Let me"
        }
    ]
    rendered = patched.apply_chat_template(conversation,
                                           thinking=True,
                                           continue_final_message=True)
    assert rendered.endswith(f"{OPEN}think{SEP}Let me")


def test_requests_without_prefill_are_unchanged(patched, stock):
    """The wrapper must not disturb the ordinary path."""
    for kwargs in ({}, {
            "thinking": False
    }, {
            "tokenize": True
    }, {
            "add_generation_prompt": False
    }, {
            "tool_choice": "required"
    }, {
            "return_dict": False,
            "thinking_effort": "low"
    }):
        assert (patched.apply_chat_template(
            MMLU_LLAMA,
            **kwargs) == stock.apply_chat_template(MMLU_LLAMA, **kwargs))


def test_wrapper_is_inert_without_the_k3_encoder(wrap):
    """Anything that is not the K3 checkpoint keeps stock behaviour.

    Built in a foreign module namespace because detection is by module scope:
    a class defined *here* would see this module's ``build_chat_segments``.
    """
    foreign = types.ModuleType("not_kimi_k3")
    exec(  # noqa: S102 - the point is a namespace without the K3 encoder
        "class Other:\n"
        "    def apply_chat_template(self, conversation, tools=None, **kw):\n"
        "        return f'other:{sorted(kw)}'\n", foreign.__dict__)

    other = foreign.Other()
    assert (wrap(other).apply_chat_template(MMLU_LLAMA,
                                            continue_final_message=True,
                                            documents=[{
                                                "a": "b"
                                            }]) == other.apply_chat_template(
                                                MMLU_LLAMA,
                                                continue_final_message=True,
                                                documents=[{
                                                    "a": "b"
                                                }]))


@pytest.mark.parametrize("kwargs,expected", [
    ({
        "documents": [{
            "title": "t"
        }]
    }, "documents"),
    ({
        "chat_template": "{{ messages }}"
    }, "chat_template"),
    ({
        "return_assistant_tokens_mask": True
    }, "return_assistant_tokens_mask"),
])
def test_unhonored_kwarg_raises(patched, kwargs, expected):
    with pytest.raises(VLLMValidationError, match=expected):
        patched.apply_chat_template(MMLU_LLAMA, **kwargs)


def test_none_valued_kwarg_is_not_an_error(patched, stock):
    """``None`` is vLLM's "unset"; rejecting it would break every request."""
    assert (patched.apply_chat_template(
        MMLU_LLAMA, documents=None) == stock.apply_chat_template(MMLU_LLAMA))


@pytest.mark.parametrize("conversation,expected", [
    ([QUESTION], "assistant"),
    ([], "conversation"),
    ([[QUESTION, PREFILL]], "conversation"),
    ([QUESTION, {
        "role": "assistant",
        "content": ""
    }], "content"),
    ([QUESTION, {
        **PREFILL, "name": "bot"
    }], "name"),
    ([QUESTION, {
        **PREFILL, "tool_calls": [{
            "id": "1"
        }]
    }], "tool_calls"),
])
def test_unsupported_prefill_raises(patched, conversation, expected):
    with pytest.raises(VLLMValidationError, match=expected):
        patched.apply_chat_template(conversation,
                                    thinking=False,
                                    continue_final_message=True)


def test_content_prefill_in_thinking_mode_raises(patched):
    """``add_generation_prompt`` opens ``<think|``; a content prefill would need
    ``</think|<response|`` first, which no public encoder call emits."""
    with pytest.raises(VLLMValidationError, match="reasoning_content"):
        patched.apply_chat_template(MMLU_LLAMA,
                                    thinking=True,
                                    continue_final_message=True)


def test_both_flags_raise(patched):
    with pytest.raises(VLLMValidationError, match="add_generation_prompt"):
        patched.apply_chat_template(MMLU_LLAMA,
                                    thinking=False,
                                    add_generation_prompt=True,
                                    continue_final_message=True)


def test_registered_for_the_k3_tokenizer_mode(monkeypatch):
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_torchtpu import tokenizers

    seen: dict[str, tuple[str, str]] = {}
    monkeypatch.setattr(
        TokenizerRegistry, "register",
        lambda mode, mod, cls: seen.__setitem__(mode, (mod, cls)))
    tokenizers.register_tokenizers()
    assert seen["kimi_k3"] == ("vllm_torchtpu.tokenizers",
                               "TpuKimiK3Tokenizer")
