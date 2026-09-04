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
"""Tokenizers for checkpoints that render chat in Python instead of Jinja.

Both models here ship no Jinja chat template; vLLM renders their chat by
calling code that comes with the checkpoint. That code takes ``**kwargs`` and
acts on the handful of keys it recognises, so a request parameter it has never
heard of is **dropped without a word** -- and
``continue_final_message`` is one of them for both.

That parameter is not cosmetic. Tasks in the ``mmlu_llama`` family end their
prompt with a partially written assistant turn (``"The best answer is"``) and
exact-match only the continuation (``"B."``). Dropped, the turn is rendered
*closed*, the model starts a fresh one and restates the stem, and the score
reads ~0 for a model that is answering correctly.

DeepSeek-V4 hit this first and is fixed below. Kimi-K3 has the same defect and
is fixed the same way -- one model at a time is how it got missed.
"""
import copy
import inspect
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.tokenizers.protocol import TokenizerLike


def _tpu_deepseek_v4_tokenizer(tokenizer):
    """Return a copy of ``tokenizer`` that honors ``continue_final_message``.

    The concrete class is built per instance because vLLM constructs its own
    DeepSeek-V4 tokenizer class dynamically from the checkpoint's HF backend.
    """
    base_cls = tokenizer.__class__

    class _TpuDeepseekV4Tokenizer(base_cls):

        def apply_chat_template(self, messages, tools=None, **kwargs):
            """Set ``wo_eos`` on a trailing assistant turn when
            ``continue_final_message`` is set, so the encoder omits
            ``eos_token`` and the model continues that turn."""
            if kwargs.pop("continue_final_message", False):
                # The parent renders the conversation from kwargs when it is
                # there, so the marked copy goes back the same way.
                conversation = list(kwargs.get("conversation", messages))
                if conversation and conversation[-1].get(
                        "role") == "assistant":
                    conversation[-1] = {**conversation[-1], "wo_eos": True}
                kwargs["conversation"] = conversation
            return super().apply_chat_template(messages, tools, **kwargs)

        def __reduce__(self):
            return _tpu_deepseek_v4_tokenizer, (tokenizer, )

    tpu_tokenizer = copy.copy(tokenizer)
    tpu_tokenizer.__class__ = _TpuDeepseekV4Tokenizer
    return tpu_tokenizer


class TpuDeepseekV4Tokenizer(TokenizerLike):
    """Registry entry point for ``tokenizer_mode=deepseek_v4``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

        return _tpu_deepseek_v4_tokenizer(
            DeepseekV4Tokenizer.from_pretrained(*args, **kwargs))


# Keys ``encoding_k3.build_chat_segments`` digs out of its ``**kwargs``. Its
# named parameters are discovered by introspection (they cannot go stale); these
# four are not parameters, so this list is the only thing separating them from a
# kwarg that is silently dropped.
_K3_ENCODER_KWARG_READS = frozenset({
    "response_format",
    "response_schema",
    "thinking_effort",
    "tool_choice",
})


def _k3_chat_impl(tokenizer):
    """The checkpoint's own ``apply_chat_template``, or ``None``.

    Found by MRO scan, not ``type(tokenizer).apply_chat_template``, so it still
    resolves once this module's wrapper is sitting on top of it.
    """
    for cls in type(tokenizer).__mro__:
        fn = cls.__dict__.get("apply_chat_template")
        if "build_chat_segments" in getattr(fn, "__globals__", {}):
            return fn
    return None


def _k3_encoder(tokenizer):
    """K3's ``build_chat_segments``, or ``None`` if this is not that tokenizer.

    The XTML rendering below is only valid for the checkpoint's own encoder, so
    everything in this wrapper is a no-op when it is absent.
    """
    fn = _k3_chat_impl(tokenizer)
    return None if fn is None else fn.__globals__["build_chat_segments"]


def _k3_honored_kwargs(tokenizer) -> frozenset[str]:
    """Every kwarg name the K3 chat path can actually act on."""
    names = set(_K3_ENCODER_KWARG_READS)
    for fn in (_k3_chat_impl(tokenizer), _k3_encoder(tokenizer)):
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            continue
        names.update(p.name for p in params if p.kind is not p.VAR_KEYWORD)
    names.discard("self")
    return frozenset(names)


def _reject(message: str, parameter: str, value: Any = None):
    return VLLMValidationError(f"Kimi K3: {message}",
                               parameter=parameter,
                               value=value)


def _split_k3_prefill(conversation, *, thinking: bool):
    """Split a conversation into (history, text) for a turn to be continued.

    K3 renders a closed assistant message as ``<message role="assistant"|``
    [``<think|`` .. ``</think|``] ``<response|`` CONTENT ``</response|``
    ``</message|`` ``<end_of_msg|>``, and ``add_generation_prompt`` emits
    exactly that same opening, stopping at the channel tag. So continuing the
    final turn *is* "render the history with a generation prompt, then append
    the prefill as ordinary text" -- token-for-token identical to truncating
    the closed form, not an approximation of it.

    That identity only holds for the channel ``add_generation_prompt`` opens:
    ``<response|`` when thinking is off, ``<think|`` when it is on. Everything
    it does not cover raises instead of rendering something almost right.

    One case is deliberately *not* a truncation: ``tool_choice`` and
    ``response_format`` make the encoder append an internal system message
    after every message, so the closed form puts that instruction *after* the
    turn it is meant to steer. Rendering the history with a generation prompt
    puts it before -- which is the ordering the encoder intends.
    """
    if not conversation or isinstance(conversation[0], (list, tuple)):
        raise _reject(
            "continue_final_message needs a single non-empty conversation",
            "continue_final_message", True)

    last = conversation[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        raise _reject(
            "continue_final_message needs the last message to be an assistant "
            "turn; there is nothing else this encoder can leave open",
            "messages",
            last.get("role") if isinstance(last, dict) else None)
    for unsupported in ("tool_calls", "name"):
        if last.get(unsupported):
            raise _reject(
                f"cannot continue an assistant turn carrying {unsupported!r}: "
                "the encoder emits it after the content it would continue",
                unsupported, last[unsupported])

    # In thinking mode the open channel is <think|, so the only prefill that
    # lands in the right place is reasoning; a content prefill would need
    # </think|<response| in between, which no public encoder call emits.
    field = "reasoning_content" if thinking else "content"
    other = "content" if thinking else "reasoning_content"
    text = last.get(field)
    if thinking and text is None:
        text = last.get("reasoning")
    if not isinstance(text, str) or not text:
        raise _reject(
            f"continue_final_message needs a non-empty string {field!r} on the "
            f"final turn when thinking={thinking}", field, text)
    if last.get(other):
        raise _reject(
            f"cannot continue a turn with both {field!r} and {other!r} when "
            f"thinking={thinking}; pass enable_thinking="
            f"{str(not thinking).lower()} to prefill the other channel", other,
            last[other])

    return list(conversation[:-1]), text


def _tpu_kimi_k3_tokenizer(tokenizer):
    """Return a copy of ``tokenizer`` that honors ``continue_final_message``
    and refuses, loudly, any other kwarg its encoder would drop."""
    base_cls = tokenizer.__class__

    class _TpuKimiK3Tokenizer(base_cls):

        def apply_chat_template(self, conversation, tools=None, **kwargs):
            if _k3_encoder(self) is None:
                return super().apply_chat_template(conversation, tools,
                                                   **kwargs)

            continue_final = kwargs.pop("continue_final_message", False)
            honored = _k3_honored_kwargs(self)
            # A ``None`` means "not set": vLLM's own kwarg merge strips those
            # from request overrides, so anything left is a caller asking for
            # something this encoder will not do.
            dropped = sorted(k for k, v in kwargs.items()
                             if k not in honored and v is not None)
            if dropped:
                raise _reject(
                    f"chat kwargs {dropped} are not honored by this "
                    "checkpoint's encoder and would be silently dropped",
                    dropped[0], kwargs[dropped[0]])

            if not continue_final:
                return super().apply_chat_template(conversation, tools,
                                                   **kwargs)

            if kwargs.pop("add_generation_prompt", False):
                raise _reject(
                    "continue_final_message and add_generation_prompt are "
                    "mutually exclusive", "add_generation_prompt", True)
            for name in ("padding", "truncation", "return_tensors",
                         "return_dict"):
                if kwargs.get(name):
                    raise _reject(
                        f"{name} is not supported with "
                        "continue_final_message", name, kwargs[name])
                kwargs.pop(name, None)

            thinking = kwargs.get("thinking", True)
            history, prefill = _split_k3_prefill(conversation,
                                                 thinking=thinking)
            tokenize = kwargs.pop("tokenize", False)
            prefix = super().apply_chat_template(history,
                                                 tools,
                                                 tokenize=tokenize,
                                                 add_generation_prompt=True,
                                                 **kwargs)
            if not tokenize:
                return prefix + prefill
            # allow_special_tokens=False: a prefill is caller text, and the
            # encoder gives caller text ordinary token ids.
            return list(prefix) + self.encode(prefill,
                                              allow_special_tokens=False)

        def __reduce__(self):
            return _tpu_kimi_k3_tokenizer, (tokenizer, )

    tpu_tokenizer = copy.copy(tokenizer)
    tpu_tokenizer.__class__ = _TpuKimiK3Tokenizer
    return tpu_tokenizer


class TpuKimiK3Tokenizer(TokenizerLike):
    """Registry entry point for ``tokenizer_mode=kimi_k3``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from vllm.tokenizers.hf import CachedHfTokenizer

        return _tpu_kimi_k3_tokenizer(
            CachedHfTokenizer.from_pretrained(*args, **kwargs))


def register_tokenizers() -> None:
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_torchtpu import envs

    if envs.TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE:
        TokenizerRegistry.register("deepseek_v4", __name__,
                                   "TpuDeepseekV4Tokenizer")
    if envs.TPU_K3_HONOR_CONTINUE_FINAL_MESSAGE:
        TokenizerRegistry.register("kimi_k3", __name__, "TpuKimiK3Tokenizer")
