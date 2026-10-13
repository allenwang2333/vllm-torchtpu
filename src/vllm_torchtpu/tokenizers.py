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
"""DeepSeek-V4 tokenizer that honors ``continue_final_message``.

vLLM renders DeepSeek-V4 chat by calling the checkpoint's ``encode_messages``
instead of a Jinja template, and builds its encode config from ``thinking_mode``,
``drop_thinking`` and ``reasoning_effort`` alone. ``continue_final_message`` is
dropped, so a trailing assistant turn is always rendered with ``eos_token`` and
a request that prefills one asks the model to generate past end-of-sentence.

The encoder reads a per-message ``wo_eos`` flag and has a template for that
case. This module registers a tokenizer that sets the flag.
"""

import copy

from vllm.tokenizers.protocol import TokenizerLike

_SUBCLASS_CACHE: dict[type, type] = {}


def _tpu_deepseek_v4_tokenizer(tokenizer):
    """Return a copy of ``tokenizer`` that honors ``continue_final_message``.

    The concrete class is built per instance because vLLM constructs its own
    DeepSeek-V4 tokenizer class dynamically from the checkpoint's HF backend.
    """
    base_cls = tokenizer.__class__
    tpu_cls = _SUBCLASS_CACHE.get(base_cls)
    if tpu_cls is not None:
        tpu_tokenizer = copy.copy(tokenizer)
        tpu_tokenizer._tpu_orig_tokenizer = tokenizer
        tpu_tokenizer.__class__ = tpu_cls
        return tpu_tokenizer
    base_apply = base_cls.apply_chat_template

    class _TpuDeepseekV4Tokenizer(base_cls):
        def apply_chat_template(
            self, messages, tools=None, _apply=base_apply, **kwargs
        ):
            """Set ``wo_eos`` on a trailing assistant turn when
            ``continue_final_message`` is set, so the encoder omits
            ``eos_token`` and the model continues that turn."""
            if not kwargs:
                return _apply(self, messages, tools)
            if kwargs.pop("continue_final_message", False):
                # The parent renders the conversation from kwargs when it is
                # there, so the marked copy goes back the same way.
                conversation = list(kwargs.get("conversation", messages))
                if (
                    conversation
                    and (last_msg := conversation[-1]).get("role") == "assistant"
                ):
                    conversation[-1] = {**last_msg, "wo_eos": True}
                kwargs["conversation"] = conversation
            if not kwargs:
                return _apply(self, messages, tools)
            return _apply(self, messages, tools, **kwargs)

        def __reduce__(self):
            return _tpu_deepseek_v4_tokenizer, (self._tpu_orig_tokenizer,)

    _SUBCLASS_CACHE[base_cls] = _TpuDeepseekV4Tokenizer
    tpu_tokenizer = copy.copy(tokenizer)
    tpu_tokenizer._tpu_orig_tokenizer = tokenizer
    tpu_tokenizer.__class__ = _TpuDeepseekV4Tokenizer
    return tpu_tokenizer


class TpuDeepseekV4Tokenizer(TokenizerLike):
    """Registry entry point for ``tokenizer_mode=deepseek_v4``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

        return _tpu_deepseek_v4_tokenizer(
            DeepseekV4Tokenizer.from_pretrained(*args, **kwargs)
        )


def register_tokenizers() -> None:
    from vllm_torchtpu import envs

    if not envs.TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE:
        return

    from vllm.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.register("deepseek_v4", __name__, "TpuDeepseekV4Tokenizer")
