# SPDX-License-Identifier: Apache-2.0
"""Exactness of VLLM_TPU_SEGMENT_TOKENIZE (vllm_torchtpu/segment_tokenize.py).

The piecewise path must return the same ids as the backend's own
``encode(text, add_special_tokens=False)`` for every input, cold or cached.
"""

import json
import os
import random

import pytest
import tokenizers
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

from vllm_torchtpu.segment_tokenize import SegmentTokenizer

SPECIALS = [
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "[gMASK]",
    "<sop>",
    "<think>",
    "</think>",
    "/nothink",
]
# Overlapping added tokens exercise leftmost-longest matching.
OVERLAP = ["<a>", "<a>b", "b<a"]

CORPUS = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "  def f(x):\n\treturn x  # 123 4567 89\r\n"
    "naïve café — 東京 😀  \n\n\n   tabs\t\tand  spaces   \n"
)


def _toy_tokenizer(extra_added=()):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                tokenizers.Regex(
                    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
                    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator([CORPUS] * 50, trainer=trainer)
    tok.add_special_tokens([AddedToken(s, normalized=False) for s in SPECIALS])
    tok.add_tokens([AddedToken(s, normalized=False) for s in OVERLAP])
    for t in extra_added:
        tok.add_tokens([t])
    return tok


def _random_text(rng, n_parts):
    atoms = (
        SPECIALS
        + OVERLAP
        + [
            "<",
            "|",
            "<|user",
            "user|>",
            "<a",
            "a>",
            "b",
            "\n",
            "\r\n",
            " ",
            "  ",
            "\t",
            "'s",
            "123456",
            "😀",
            "東京",
        ]
    )
    out = []
    for _ in range(n_parts):
        if rng.random() < 0.5:
            out.append(rng.choice(atoms))
        else:
            s = rng.randrange(len(CORPUS))
            out.append(CORPUS[s : s + rng.randrange(1, 120)])
    return "".join(out)


def test_toy_tokenizer_matches_full_encode():
    tok = _toy_tokenizer()
    assert SegmentTokenizer.eligible(tok) is None
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    rng = random.Random(0)
    for i in range(3000):
        text = _random_text(rng, rng.randrange(1, 40))
        want = tok.encode(text, add_special_tokens=False).ids
        assert seg.encode(text) == want, repr(text)
        # Second pass is served (partly) from the cache and must not change.
        assert seg.encode(text) == want, repr(text)
    assert seg.hits > 0 and seg.misses > 0


def test_edge_inputs():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    for text in [
        "",
        "<|user|>",
        "<|user|><|user|>",
        "<a>b<a>",
        "b<a>b",
        "<a>bb<a",
        " <|user|> ",
        "\n<|assistant|>\n",
        "x" * 5000 + "<sop>" + "y" * 5000,
    ]:
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids, repr(
            text
        )


def test_cache_is_bounded():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=500)
    rng = random.Random(1)
    for _ in range(200):
        text = "<|user|>" + _random_text(rng, 20)
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids
        assert seg._cached_tokens <= 500


def test_shared_backend_truncation_does_not_leak():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    tok.enable_truncation(max_length=3)  # what transformers does per call
    text = CORPUS * 3
    tok.no_truncation()
    want = tok.encode(text, add_special_tokens=False).ids
    tok.enable_truncation(max_length=3)
    assert seg.encode(text) == want


@pytest.mark.parametrize("flag", ["lstrip", "rstrip", "single_word"])
def test_ineligible_added_tokens(flag):
    tok = _toy_tokenizer(extra_added=[AddedToken("<x>", **{flag: True})])
    assert SegmentTokenizer.eligible(tok) is not None


def test_normalized_added_tokens_need_identity_normalizer():
    tok = _toy_tokenizer(extra_added=[AddedToken("<x>", normalized=True)])
    assert SegmentTokenizer.eligible(tok) is None
    tok.normalizer = normalizers.Sequence([])
    assert SegmentTokenizer.eligible(tok) is None
    tok.normalizer = normalizers.Lowercase()
    assert SegmentTokenizer.eligible(tok) is not None


# normalized=True tokens that overlap the normalized=False ones: tokenizers
# matches normalized=False first over the whole text, so "q<a>bq" is
# "q" + "<a>b" + "q", while one leftmost-longest pass would take "q<a" first.
NORMALIZED = ["q<a", "<a>bq", "zz", "<|user|>x"]


def test_normalized_and_raw_added_tokens_match_full_encode():
    tok = _toy_tokenizer(
        extra_added=[AddedToken(s, normalized=True) for s in NORMALIZED]
    )
    tok.normalizer = normalizers.Sequence([])  # as in DeepSeek tokenizer.json
    assert SegmentTokenizer.eligible(tok) is None
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    for text in ["q<a>bq", "q<a", "zzz<|user|>x", "<a>bq<a>bqzz", "q<a>b<a"]:
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids
    rng = random.Random(4)
    for _ in range(2000):
        text = "".join(
            rng.choice(NORMALIZED + ["q", "<a", ">", "z"]) + _random_text(rng, 2)
            for _ in range(rng.randrange(1, 12))
        )
        want = tok.encode(text, add_special_tokens=False).ids
        assert seg.encode(text) == want, repr(text)
        assert seg.encode(text) == want, repr(text)


def test_encode_special_tokens_ineligible():
    tok = _toy_tokenizer()
    tok.encode_special_tokens = True
    assert SegmentTokenizer.eligible(tok) is not None


GLM = os.environ.get("SEGMENT_TOKENIZE_TEST_MODEL", "zai-org/GLM-5.2-FP8")


def _glm_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(GLM)
    except Exception as e:  # no network / no cache
        pytest.skip(f"{GLM} tokenizer unavailable: {e}")


def _conversation(rng, n_msgs, words):
    vocab = CORPUS.split() + ["<|user|>", "</tool_call>", "{", "}", "\n\n", "```", "😀"]
    msgs = [
        {"role": "system", "content": " ".join(rng.choice(vocab) for _ in range(words))}
    ]
    for i in range(n_msgs):
        body = " ".join(rng.choice(vocab) for _ in range(rng.randrange(1, words)))
        role = ["user", "assistant", "tool"][i % 3]
        m = {"role": role, "content": body}
        if role == "assistant":
            m["tool_calls"] = [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": {"command": body[:40]}},
                }
            ]
        if role == "tool":
            m["tool_call_id"] = f"c{i - 1}"
        msgs.append(m)
    msgs.append({"role": "user", "content": "go on"})
    return msgs


def test_glm_chat_prompts_match_full_encode():
    hf = _glm_tokenizer()
    backend = hf.backend_tokenizer
    assert SegmentTokenizer.eligible(backend) is None
    seg = SegmentTokenizer(backend, max_cached_tokens=1 << 22)
    rng = random.Random(2)
    for _ in range(40):
        msgs = _conversation(rng, rng.randrange(1, 30), 300)
        # Continue the conversation turn by turn, as an agent does.
        for turn in range(3):
            prompt = hf.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            want = hf(prompt, add_special_tokens=False)["input_ids"]
            assert seg.encode(prompt) == want
            msgs = msgs[:-1] + _conversation(rng, 3, 200)[1:]
    assert seg.hits > 0


def test_glm_same_block_hashes():
    """Same ids give the same vLLM prefix-cache block hashes (and so hits)."""
    pytest.importorskip("vllm")
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

    hf = _glm_tokenizer()
    seg = SegmentTokenizer(hf.backend_tokenizer, max_cached_tokens=1 << 22)
    rng = random.Random(3)
    msgs = _conversation(rng, 60, 400)
    prompt = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    a = hf(prompt, add_special_tokens=False)["input_ids"]
    b = seg.encode(prompt)
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)

    def hashes(ids, bs=256):
        out, prev = [], None
        for s in range(0, len(ids) - bs + 1, bs):
            prev = hash_block_tokens(fn, prev, ids[s : s + bs])
            out.append(prev)
        return out

    assert len(a) >= 256 and hashes(a) == hashes(b)


@pytest.fixture
def restore_renderers(monkeypatch):
    """Undo patch_renderer_segment_tokenize on both patched renderer classes."""
    pytest.importorskip("vllm")
    from vllm.renderers.base import BaseRenderer
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

    for cls, name in (
        (BaseRenderer, "_tokenize_prompt"),
        (DeepseekV4Renderer, "_apply_chat_template"),
    ):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
        monkeypatch.setattr(cls, "_tpu_segment_tokenize", False, raising=False)


def test_renderer_patch_routes_and_falls_back(monkeypatch, restore_renderers):
    from vllm.renderers import base as renderer_base

    from vllm_torchtpu import envs
    from vllm_torchtpu import segment_tokenize as st

    hf = _glm_tokenizer()
    cls = renderer_base.BaseRenderer
    saved = cls.__dict__["_tokenize_prompt"]
    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE", "1")
    try:
        st.patch_renderer_segment_tokenize()
        assert cls._tpu_segment_tokenize

        class P:  # minimal TokenizeParams stand-in
            def __init__(self, add_special_tokens=False, max_length=None):
                self.a, self.m = add_special_tokens, max_length

            def get_encode_kwargs(self):
                return dict(
                    truncation=self.m is not None,
                    max_length=self.m,
                    add_special_tokens=self.a,
                )

        class R:
            _tokenize_prompt = cls._tokenize_prompt

            def get_tokenizer(self):
                return hf

            def _wants_offsets(self, prompt, params):
                return False

            def _build_tokens_prompt(self, ids, prompt, offset_mapping=None):
                return {"prompt_token_ids": ids, "fast": True}

        calls = []
        monkeypatch.setattr(st, "_INSTANCES", {})
        r = R()
        text = "[gMASK]<sop><|user|>hello world<|assistant|>"
        out = r._tokenize_prompt({"prompt": text}, P())
        assert (
            out["fast"]
            and out["prompt_token_ids"]
            == hf(text, add_special_tokens=False)["input_ids"]
        )

        # Over-long prompts and add_special_tokens go to the original path.
        def orig(self, prompt, params):
            return calls.append(params) or {"orig": True}

        monkeypatch.setattr(st, "_INSTANCES", {})
        cls._tpu_segment_tokenize = False
        cls._tokenize_prompt = orig
        st.patch_renderer_segment_tokenize()
        R._tokenize_prompt = cls._tokenize_prompt
        assert r._tokenize_prompt({"prompt": text}, P(add_special_tokens=True)) == {
            "orig": True
        }
        assert r._tokenize_prompt({"prompt": text}, P(max_length=2)) == {"orig": True}
        assert len(calls) == 2
    finally:
        cls._tokenize_prompt = saved
        cls._tpu_segment_tokenize = False
        assert envs.VLLM_TPU_SEGMENT_TOKENIZE is True


# A local DeepSeek-V4 tokenizer directory (tokenizer.json + tokenizer_config.json).
DSV4 = os.environ.get("SEGMENT_TOKENIZE_TEST_DSV4_TOKENIZER")


def _dsv4_tokenizer():
    if not DSV4:
        pytest.skip("SEGMENT_TOKENIZE_TEST_DSV4_TOKENIZER is not set")
    pytest.importorskip("vllm")
    from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

    return DeepseekV4Tokenizer.from_pretrained(DSV4)


DSV4_ATOMS = [
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜end▁of▁sentence｜>",
    "<｜begin▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
    "<｜DSML｜",
    "<｜User",
    "｜>",
    "<tool_result>",
    "\n\n",
    "😀",
    "东京",
]


def _dsv4_conversation(rng, n_msgs, words):
    vocab = CORPUS.split() + DSV4_ATOMS
    body = lambda n: " ".join(rng.choice(vocab) for _ in range(n))  # noqa: E731
    msgs = [{"role": "system", "content": body(words)}]
    for i in range(n_msgs):
        if i % 3 == 1:
            call = {"name": "Bash", "arguments": json.dumps({"command": body(8)})}
            msgs.append(
                {
                    "role": "assistant",
                    "content": body(rng.randrange(1, words)),
                    "tool_calls": [
                        {"id": f"c{i}", "type": "function", "function": call}
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": body(40)})
        else:
            role = "assistant" if msgs[-1]["role"] in ("user", "tool") else "user"
            msgs.append({"role": role, "content": body(rng.randrange(1, words))})
    if msgs[-1]["role"] == "assistant":
        msgs.append({"role": "user", "content": "go on"})
    return msgs


def test_dsv4_chat_prompts_match_full_encode():
    """DeepSeek-V4 marks its chat tokens normalized=True over an empty normalizer."""
    tok = _dsv4_tokenizer()
    assert SegmentTokenizer.eligible(tok.backend_tokenizer) is None
    seg = SegmentTokenizer(tok.backend_tokenizer, max_cached_tokens=1 << 22)
    rng = random.Random(5)
    for _ in range(30):
        msgs = _dsv4_conversation(rng, rng.randrange(1, 20), 300)
        for _turn in range(3):
            want = tok.apply_chat_template(msgs, conversation=msgs)
            text = tok.apply_chat_template(msgs, conversation=msgs, tokenize=False)
            assert seg.encode(text) == want
            msgs = msgs + _dsv4_conversation(rng, 2, 200)[1:]
    assert seg.hits > 0


def test_dsv4_renderer_patch_routes_and_falls_back(monkeypatch, restore_renderers):
    tok = _dsv4_tokenizer()
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

    from vllm_torchtpu import segment_tokenize as st

    cls = DeepseekV4Renderer
    saved = cls.__dict__["_apply_chat_template"]
    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE", "1")
    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE_VERIFY_EVERY", "1")
    monkeypatch.setattr(st, "_INSTANCES", {})
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("tokenize", True))
        return saved(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_apply_chat_template", spy)
    st.patch_renderer_segment_tokenize()
    assert cls.__dict__["_tpu_segment_tokenize"]
    r = object.__new__(cls)
    r.get_tokenizer = lambda: tok
    msgs = _dsv4_conversation(random.Random(6), 8, 100)
    want = tok.apply_chat_template(msgs, conversation=msgs)
    out = r._apply_chat_template(conversation=msgs, messages=msgs)
    # Rendered as text by the tokenizer, encoded piecewise, checked.
    assert out == want and calls == [False]
    assert st.get_segment_tokenizer(tok, 0).misses > 0
    assert r._apply_chat_template(
        conversation=msgs, messages=msgs, tokenize=False
    ) == tok.apply_chat_template(msgs, conversation=msgs, tokenize=False)
    r._apply_chat_template(conversation=msgs, messages=msgs, max_length=8)
    assert calls == [False, False, True]


def test_verify_raises_on_difference():
    from vllm_torchtpu.segment_tokenize import _Engaged

    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    engaged = _Engaged(verify_every=2)
    text = "<|user|>" + CORPUS
    want = tok.encode(text, add_special_tokens=False).ids
    assert engaged.encode(seg, text, lambda: 1 / 0) == want  # 1st: not checked
    assert engaged.encode(seg, text, lambda: want) == want  # 2nd: checked
    engaged.encode(seg, text, lambda: 1 / 0)
    with pytest.raises(RuntimeError, match="first difference at 1"):
        engaged.encode(seg, text, lambda: want[:1] + [-1] + want[2:])
