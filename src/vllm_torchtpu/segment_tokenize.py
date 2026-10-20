# SPDX-License-Identifier: Apache-2.0
"""Exact segment-cached prompt tokenization for long multi-turn chat prompts.

Opt-in with ``VLLM_TPU_SEGMENT_TOKENIZE=1``.

Why: the API server tokenizes the whole chat-templated prompt on every turn.
HF fast tokenizers run ~1.5-2.5 us/token on one core, so a 500k-token agent
turn spends ~0.75-1.25 s in ``tokenizer(prompt)`` before the request reaches
the engine -- directly on TTFT, with the device idle. Consecutive turns of a
conversation share almost all of that text.

How (and why it is exact): an HF ``tokenizers`` pipeline first splits the
input on added-token occurrences -- leftmost-longest over the
``normalized=False`` tokens, then, inside the remaining pieces, over the
``normalized=True`` ones -- then normalizes, pre-tokenizes and BPE-encodes
each remaining piece independently. When every added token is
``lstrip=False``, ``rstrip=False`` and ``single_word=False``, no
``normalized=True`` token exists unless the normalizer is the identity, and
the tokenizer does not encode special tokens as text, that split is a pure
function of the raw string, so

    encode(text) == concat(encode(piece) for piece in split(text))

holds piece by piece. Chat templates put added tokens (``<|user|>``,
``<|assistant|>``, ``<tool_response>``, ...) around every message, so each
message body is one piece. This module performs the same split, serves
pieces already seen from an LRU cache keyed by the piece text, and encodes
only new pieces -- in one ``encode_batch`` call, so a cold long prompt is
also encoded across cores. Tokenizers that do not satisfy the conditions
above, or requests that need offsets, special-token insertion or tokenizer
truncation, take vLLM's original path unchanged.

Two entry points use it: ``BaseRenderer._tokenize_prompt`` (renderers that
return the chat prompt as text, e.g. HF Jinja templates) and
``DeepseekV4Renderer._apply_chat_template`` (DeepSeek-V4 renders and encodes
the chat inside its tokenizer's ``apply_chat_template``).
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Pieces shorter than this are re-encoded every time (cheap) rather than
# cached, so the cache holds message bodies, not template glue.
_MIN_CACHED_CHARS = 64


class SegmentTokenizer:
    """Encodes text exactly as ``backend.encode(text, add_special_tokens=False)``."""

    def __init__(self, backend: Any, max_cached_tokens: int):
        # Private copy: transformers mutates truncation/padding on the shared
        # backend per call, which must not leak into these encodes.
        self._backend = copy.deepcopy(backend)
        self._backend.no_truncation()
        self._backend.no_padding()
        added = self._backend.get_added_tokens_decoder()
        self._added_id = {tok.content: tid for tid, tok in added.items()}
        # tokenizers matches the normalized=False tokens on the whole input
        # first, then the normalized=True ones inside the pieces left over.
        self._split_raw = _alternation(
            t.content for t in added.values() if not t.normalized
        )
        self._split_norm = _alternation(
            t.content for t in added.values() if t.normalized
        )
        # Cached pieces are lists over one shared table of int objects: 8 B
        # per cached token, and ``list.extend`` copies pointers instead of
        # boxing a fresh int per token.
        vocab = max(
            self._backend.get_vocab_size(with_added_tokens=True),
            max(added, default=0) + 1,
        )
        self._ints = list(range(vocab))
        self._cache: OrderedDict[str, list[int]] = OrderedDict()
        self._cached_tokens = 0
        self._max_cached_tokens = max_cached_tokens
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def eligible(backend: Any) -> str | None:
        """Return why ``backend`` cannot use the piecewise path, or None."""
        if getattr(backend, "encode_special_tokens", False):
            return "encode_special_tokens is set"
        identity_normalizer = None
        for tok in backend.get_added_tokens_decoder().values():
            if tok.lstrip or tok.rstrip or tok.single_word:
                return f"added token {tok.content!r} is not a hard split point"
            if tok.normalized:
                if identity_normalizer is None:
                    spec = json.loads(backend.to_str())["normalizer"]
                    identity_normalizer = _is_identity_normalizer(spec)
                if not identity_normalizer:
                    return (
                        f"added token {tok.content!r} is matched after a "
                        "non-identity normalizer"
                    )
        return None

    def encode(self, text: str) -> list[int]:
        pieces: list[tuple[bool, str]] = []  # (is_added_token, text)
        for is_added, piece in _split_on(self._split_raw, text):
            if is_added:
                pieces.append((True, piece))
            else:
                pieces.extend(_split_on(self._split_norm, piece))

        with self._lock:
            found: dict[str, list[int]] = {}
            todo: list[str] = []
            for is_added, piece in pieces:
                if is_added or piece in found:
                    continue
                ids = self._cache.get(piece)
                if ids is not None:
                    self._cache.move_to_end(piece)
                    found[piece] = ids
                    self.hits += 1
                else:
                    found[piece] = None  # type: ignore[assignment]
                    todo.append(piece)
            self.misses += len(todo)
            if todo:
                encs = self._backend.encode_batch(todo, add_special_tokens=False)
                for piece, enc in zip(todo, encs):
                    ids = list(map(self._ints.__getitem__, enc.ids))
                    found[piece] = ids
                    if len(piece) >= _MIN_CACHED_CHARS:
                        self._insert(piece, ids)

        out: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
            else:
                out.extend(found[piece])
        return out

    def _insert(self, piece: str, ids: list[int]) -> None:
        if len(ids) > self._max_cached_tokens:
            return
        if piece in self._cache:
            return
        self._cache[piece] = ids
        self._cached_tokens += len(ids)
        while self._cached_tokens > self._max_cached_tokens:
            _, old = self._cache.popitem(last=False)
            self._cached_tokens -= len(old)


def _alternation(contents) -> re.Pattern[str] | None:
    alts = sorted(contents, key=len, reverse=True)
    if not alts:
        return None
    # Python's alternation takes the first alternative matching at the
    # leftmost position; longest-first order makes that leftmost-longest,
    # the same rule as tokenizers' Aho-Corasick added-token matcher.
    return re.compile("|".join(re.escape(a) for a in alts))


def _split_on(pattern: re.Pattern[str] | None, text: str):
    """Yield ``(is_match, piece)`` covering ``text``, without empty pieces."""
    pos = 0
    if pattern is not None:
        for m in pattern.finditer(text):
            if m.start() > pos:
                yield False, text[pos : m.start()]
            yield True, m.group()
            pos = m.end()
    if pos < len(text):
        yield False, text[pos:]


def _is_identity_normalizer(spec: dict | None) -> bool:
    if spec is None:
        return True
    return spec.get("type") == "Sequence" and all(
        _is_identity_normalizer(n) for n in spec.get("normalizers", ())
    )


def _backend_of(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or not hasattr(backend, "encode_batch"):
        return None
    return backend


_INSTANCES: dict[int, SegmentTokenizer | None] = {}
_INSTANCES_LOCK = threading.Lock()


def get_segment_tokenizer(tokenizer: Any, max_cached_tokens: int):
    """One ``SegmentTokenizer`` per vLLM tokenizer object, or None if ineligible."""
    key = id(tokenizer)
    with _INSTANCES_LOCK:
        if key in _INSTANCES:
            return _INSTANCES[key]
        seg = None
        backend = _backend_of(tokenizer)
        if backend is None:
            reason = "no HF tokenizers backend"
        elif getattr(tokenizer, "split_special_tokens", False):
            reason = "split_special_tokens is set"
        else:
            reason = SegmentTokenizer.eligible(backend)
        if reason is None:
            seg = SegmentTokenizer(backend, max_cached_tokens)
            logger.info(
                "VLLM_TPU_SEGMENT_TOKENIZE: piecewise prompt tokenization on "
                "(%d added tokens, cache %d tokens, ~12 B/token).",
                len(seg._added_id),
                max_cached_tokens,
            )
        else:
            logger.warning(
                "VLLM_TPU_SEGMENT_TOKENIZE: falling back to full tokenization: %s",
                reason,
            )
        _INSTANCES[key] = seg
        return seg


_STATS_EVERY = 500


class _Engaged:
    """Per-prompt bookkeeping: periodic stats log and optional exactness check."""

    def __init__(self, verify_every: int):
        self._verify_every = verify_every
        self._lock = threading.Lock()
        self._prompts = self._tokens = self._seconds = self._verified = 0

    def encode(self, seg: SegmentTokenizer, text: str, reference) -> list[int]:
        """``seg.encode(text)``; every ``verify_every``-th prompt also runs
        ``reference()`` (the original full encode) and raises on any difference."""
        t0 = time.perf_counter()
        ids = seg.encode(text)
        dt = time.perf_counter() - t0
        with self._lock:
            self._prompts += 1
            self._tokens += len(ids)
            self._seconds += dt
            n = self._prompts
            check = self._verify_every > 0 and n % self._verify_every == 0
        if check:
            want = reference()
            if want != ids:
                first = next(
                    (i for i, (a, b) in enumerate(zip(ids, want)) if a != b),
                    min(len(ids), len(want)),
                )
                raise RuntimeError(
                    "VLLM_TPU_SEGMENT_TOKENIZE: piecewise ids differ from the full "
                    f"encode ({len(ids)} vs {len(want)} ids, first difference at "
                    f"{first})"
                )
            with self._lock:
                self._verified += 1
        if n % _STATS_EVERY == 0:
            logger.info(
                "VLLM_TPU_SEGMENT_TOKENIZE: %d prompts, %.1f M tokens, %.2f ms/prompt "
                "(%.3f us/token), piece cache hits %d / misses %d, %d prompts "
                "verified identical to the full encode.",
                n,
                self._tokens / 1e6,
                1e3 * self._seconds / n,
                1e6 * self._seconds / max(self._tokens, 1),
                seg.hits,
                seg.misses,
                self._verified,
            )
        return ids


def patch_renderer_segment_tokenize() -> None:
    """Route eligible prompt tokenization through the segment cache.

    Must run before the renderer is constructed: ``BaseRenderer.__init__``
    binds ``self._tokenize_prompt`` (and DeepseekV4Renderer
    ``self._apply_chat_template``) into async wrappers.
    """
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_SEGMENT_TOKENIZE:
        return
    max_cached = envs.VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS
    engaged = _Engaged(envs.VLLM_TPU_SEGMENT_TOKENIZE_VERIFY_EVERY)
    _patch_base_renderer(max_cached, engaged)
    _patch_deepseek_v4_renderer(max_cached, engaged)


def _patch_base_renderer(max_cached: int, engaged: _Engaged) -> None:
    from vllm.renderers import base as renderer_base

    cls = renderer_base.BaseRenderer
    if cls.__dict__.get("_tpu_segment_tokenize", False):
        return
    orig = cls._tokenize_prompt

    def _tokenize_prompt(self, prompt, params):
        kwargs = params.get_encode_kwargs()
        if kwargs.get("add_special_tokens") or self._wants_offsets(prompt, params):
            return orig(self, prompt, params)
        seg = get_segment_tokenizer(self.get_tokenizer(), max_cached)
        if seg is None:
            return orig(self, prompt, params)
        ids = engaged.encode(
            seg,
            prompt["prompt"],
            lambda: list(orig(self, dict(prompt), params)["prompt_token_ids"]),
        )
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(ids) > max_length
        ):
            # Over-long prompt: keep the tokenizer's own truncation semantics.
            return orig(self, prompt, params)
        return self._build_tokens_prompt(ids, prompt, offset_mapping=None)

    cls._tokenize_prompt = _tokenize_prompt
    cls._tpu_segment_tokenize = True
    logger.info("VLLM_TPU_SEGMENT_TOKENIZE: patched BaseRenderer._tokenize_prompt")


def _patch_deepseek_v4_renderer(max_cached: int, engaged: _Engaged) -> None:
    from vllm.renderers import deepseek_v4

    cls = deepseek_v4.DeepseekV4Renderer
    if cls.__dict__.get("_tpu_segment_tokenize", False):
        return
    orig = cls._apply_chat_template

    def _apply_chat_template(self, *args, **kwargs):
        # The tokenizer renders the prompt and, unless tokenize=False, returns
        # encode(prompt, add_special_tokens=False) truncated as requested.
        if (
            kwargs.get("tokenize", True) is not True
            or "truncation" in kwargs
            or "max_length" in kwargs
        ):
            return orig(self, *args, **kwargs)
        tokenizer = self.get_tokenizer()
        seg = get_segment_tokenizer(tokenizer, max_cached)
        if seg is None:
            return orig(self, *args, **kwargs)
        text = orig(self, *args, **{**kwargs, "tokenize": False})
        return engaged.encode(
            seg, text, lambda: tokenizer.encode(text, add_special_tokens=False)
        )

    cls._apply_chat_template = _apply_chat_template
    cls._tpu_segment_tokenize = True
    logger.info(
        "VLLM_TPU_SEGMENT_TOKENIZE: patched DeepseekV4Renderer._apply_chat_template"
    )
