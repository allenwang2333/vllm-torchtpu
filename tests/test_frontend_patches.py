# SPDX-License-Identifier: Apache-2.0
"""TPU_DETOK_PRIME_TAIL_TOKENS streams the same text as full-prompt priming."""

import random
from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from vllm_torchtpu import frontend_patches  # noqa: E402

_TEXT = (
    "def step(self, ids):\n    return [i + 1 for i in ids]  # comment\n"
    "中文测试：深度求索模型的推理速度。😀🎉 emoji and ünïcödé text. "
) * 50


@pytest.fixture(scope="module")
def backend():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [_TEXT],
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<think>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return SimpleNamespace(_tokenizer=tok)


def _request(prompt_ids, skip_special_tokens):
    params = SimpleNamespace(
        stop=None,
        min_tokens=0,
        include_stop_str_in_output=False,
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=True,
    )
    return SimpleNamespace(
        request_id="r", prompt_token_ids=prompt_ids, sampling_params=params
    )


def _stream(cls, backend, prompt_ids, gen_ids, skip):
    detok = cls(backend, _request(prompt_ids, skip))
    return [detok.decode_next(t) for t in gen_ids]


def test_tail_priming_streams_identical_text(backend, monkeypatch):
    from vllm.v1.engine import detokenizer

    ids = backend._tokenizer.encode(_TEXT, add_special_tokens=False).ids
    think = backend._tokenizer.token_to_id("<think>")
    rng = random.Random(0)
    cases = []
    for i in range(60):
        cut = rng.randrange(40, len(ids) - 60)
        prompt = ids[:cut] + ([think] if i % 3 == 0 else [])
        cases.append((prompt, ids[cut : cut + rng.randrange(1, 50)], i % 2 == 0))

    cls = detokenizer.FastIncrementalDetokenizer
    expected = [_stream(cls, backend, *c) for c in cases]

    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.delattr(cls, "_tpu_prime_tail_tokens", raising=False)
    frontend_patches.patch_detok_prime_tail(8)
    assert [_stream(cls, backend, *c) for c in cases] == expected


def test_tail_request_view_keeps_other_fields():
    req = _request(list(range(100)), True)
    view = frontend_patches._TailPromptRequest(req, 8)
    assert view.prompt_token_ids == list(range(92, 100))
    assert view.sampling_params is req.sampling_params
    assert view.request_id == "r"
    assert (
        frontend_patches._TailPromptRequest(_request(None, True), 8).prompt_token_ids
        is None
    )
