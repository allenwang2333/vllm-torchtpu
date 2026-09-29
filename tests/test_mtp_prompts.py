# SPDX-License-Identifier: Apache-2.0
"""保护自然语言 MTP 输入的长度、确定性与末尾任务指令。"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mtp_prompts",
    Path(__file__).parent / "entrypoints" / "llm" / "mtp_prompts.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

pytestmark = pytest.mark.cpu_test


class CharacterTokenizer:
    """可逆字符 tokenizer，让边界断言不依赖模型下载。"""

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


@pytest.mark.parametrize("prompt_tokens", [512, 65536])
def test_exact_budget_preserves_the_writing_task(prompt_tokens):
    tokenizer = CharacterTokenizer()
    prompts = _MODULE.build_mtp_prompts(tokenizer, prompt_tokens)
    assert len(prompts) == 2
    assert prompts == _MODULE.build_mtp_prompts(tokenizer, prompt_tokens)
    texts = []
    for prompt in prompts:
        token_ids = prompt["prompt_token_ids"]
        assert len(token_ids) == prompt_tokens
        text = "".join(map(chr, token_ids))
        assert "Write the next chapter" in text
        assert text.endswith("Begin the chapter now.\n")
        texts.append(text)
    assert texts[0] != texts[1]


def test_long_document_has_distinct_natural_language_paragraphs():
    prompts = _MODULE.build_mtp_prompts(CharacterTokenizer(), 65536)
    for prompt in prompts:
        text = "".join(map(chr, prompt["prompt_token_ids"]))
        # 最后一个正文段落允许按 token 截断，只检查其前面的完整段落。
        paragraphs = text.split("\n\n")[1:-2]
        assert len(paragraphs) > 50
        assert len(set(paragraphs)) == len(paragraphs)
        assert all(len(paragraph.split()) > 30 for paragraph in paragraphs)


def test_budget_must_leave_room_for_document_and_instruction():
    with pytest.raises(ValueError, match="prompt_tokens"):
        _MODULE.build_mtp_prompts(CharacterTokenizer(), 1)
