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


def _dsv4_renderer(backend, workers):
    from transformers import TokenizersBackend
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
    from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
    from vllm.tokenizers.hf import get_cached_tokenizer

    hf = TokenizersBackend(tokenizer_object=backend._tokenizer)
    tok = get_cached_tokenizer(get_deepseek_v4_tokenizer(hf))
    model_config = SimpleNamespace(
        renderer_num_workers=workers,
        multimodal_config=None,
        is_multimodal_model=False,
        hf_config=SimpleNamespace(),
        skip_tokenizer_init=False,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(_api_process_rank=0),
    )
    return DeepseekV4Renderer(config, tok), tok


def _concurrent_tokenize(renderer, n=48):
    from concurrent.futures import ThreadPoolExecutor

    tokenizer = renderer.get_tokenizer()

    def one(i):
        # vLLM's _tokenize_prompt passes truncation for some requests, which
        # reconfigures the shared HF tokenizer mid-encode.
        kw = {"truncation": True, "max_length": 10**7 - i} if i % 2 else {}
        return tokenizer(_TEXT * 40, add_special_tokens=False, **kw)["input_ids"]

    with ThreadPoolExecutor(8) as ex:
        futures = [ex.submit(one, i) for i in range(n)]
    errors = [f.exception() for f in futures if f.exception() is not None]
    return [f.result() for f in futures if f.exception() is None], errors


def test_dsv4_renderer_thread_pool(backend, monkeypatch):
    pytest.importorskip("transformers")
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

    reference = backend._tokenizer.encode(_TEXT * 40, add_special_tokens=False).ids
    monkeypatch.setattr(DeepseekV4Renderer, "__init__", DeepseekV4Renderer.__init__)
    monkeypatch.delattr(DeepseekV4Renderer, "_tpu_thread_pool_patched", raising=False)
    frontend_patches.patch_dsv4_renderer_thread_pool()

    renderer, shared = _dsv4_renderer(backend, workers=16)
    ids, errors = _concurrent_tokenize(renderer)
    assert not errors
    assert all(x == reference for x in ids)
    # The caller's tokenizer (shared with the detokenizer) keeps its class.
    assert type(shared) is not type(renderer.get_tokenizer())
