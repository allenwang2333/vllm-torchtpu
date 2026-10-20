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
"""Exact segment-cached prompt tokenization for long multi-turn chat prompts.

Opt-in with ``VLLM_TPU_SEGMENT_TOKENIZE=1``.

Why: the API server tokenizes the whole chat-templated prompt on every turn.
HF fast tokenizers run ~1.5-2.5 us/token on one core, so a 500k-token agent
turn spends ~0.75-1.25 s in ``tokenizer(prompt)`` before the request reaches
the engine -- directly on TTFT, with the device idle. Consecutive turns of a
conversation share almost all of that text.

How (and why it is exact): an HF ``tokenizers`` pipeline first splits the
input on added-token occurrences (leftmost-longest), then normalizes,
pre-tokenizes and BPE-encodes each remaining piece independently. When every
added token is ``normalized=False``, ``lstrip=False``, ``rstrip=False`` and
``single_word=False``, and the tokenizer does not encode special tokens as
text, that first split is a pure function of the raw string, so

    encode(text) == concat(encode(piece) for piece in split(text))

holds piece by piece. Tokens with ``normalized=True`` are matched after
normalization, in a second pass over the pieces the first pass left; with no
normalizer (or an empty normalizer ``Sequence``) that is still a function of
the raw string, so such tokens are hard split points too and are handled by
the same two-pass split here. Chat templates put added tokens (``<|user|>``,
``<|assistant|>``, ``<tool_response>``, ...) around every message, so each
message body is one piece. This module performs the same split, serves
pieces already seen from an LRU cache keyed by the piece text, and encodes
only new pieces -- in one ``encode_batch`` call, so a cold long prompt is
also encoded across cores. Tokenizers that do not satisfy the conditions
above make the patch raise at the first request (the flag must not silently
measure the unpatched path); requests that need offsets, special-token
insertion or tokenizer truncation take vLLM's original path unchanged.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Pieces shorter than this are re-encoded every time (cheap) rather than
# cached, so the cache holds message bodies, not template glue.
_MIN_CACHED_CHARS = 64


def _alternation(contents: Iterable[str]) -> re.Pattern | None:
    # Python's alternation takes the first alternative matching at the leftmost
    # position; longest-first order makes that leftmost-longest, the same rule
    # as tokenizers' Aho-Corasick added-token matcher.
    alts = sorted(set(contents), key=len, reverse=True)
    if not alts:
        return None
    return re.compile("|".join(re.escape(a) for a in alts))


def _split_on(pattern: re.Pattern | None, text: str) -> list[tuple[bool, str]]:
    if not text:
        return []
    if pattern is None:
        return [(False, text)]
    pieces: list[tuple[bool, str]] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            pieces.append((False, text[pos : m.start()]))
        pieces.append((True, m.group()))
        pos = m.end()
    if pos < len(text):
        pieces.append((False, text[pos:]))
    return pieces


def _normalizer_is_identity(backend: Any) -> bool:
    normalizer = json.loads(backend.to_str()).get("normalizer")
    return normalizer is None or normalizer == {"type": "Sequence", "normalizers": []}


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
        # tokenizers splits on normalized=False tokens first, then on
        # normalized=True tokens inside the remaining pieces; mirror both passes.
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
        identity_normalizer = _normalizer_is_identity(backend)
        for tok in backend.get_added_tokens_decoder().values():
            if tok.lstrip or tok.rstrip or tok.single_word:
                return f"added token {tok.content!r} is not a hard split point"
            if tok.normalized and not identity_normalizer:
                return (
                    f"added token {tok.content!r} is matched after a "
                    "non-identity normalizer"
                )
        return None

    def split(self, text: str) -> list[tuple[bool, str]]:
        """``(is_added_token, text)`` pieces, as tokenizers' added-token pass."""
        pieces: list[tuple[bool, str]] = []
        for is_added, piece in _split_on(self._split_raw, text):
            if is_added:
                pieces.append((True, piece))
            else:
                pieces.extend(_split_on(self._split_norm, piece))
        return pieces

    def encode(self, text: str) -> list[int]:
        pieces = self.split(text)

        found: dict[str, list[int] | None] = {}
        todo: list[str] = []
        with self._lock:
            for is_added, piece in pieces:
                if is_added or piece in found:
                    continue
                ids = self._cache.get(piece)
                if ids is not None:
                    self._cache.move_to_end(piece)
                    self.hits += 1
                else:
                    todo.append(piece)
                found[piece] = ids
            self.misses += len(todo)
        if todo:
            # Outside the lock: the private backend is never reconfigured, and
            # encode_batch releases the GIL and spreads pieces across cores.
            encs = self._backend.encode_batch(todo, add_special_tokens=False)
            new = [
                (p, list(map(self._ints.__getitem__, e.ids)))
                for p, e in zip(todo, encs)
            ]
            with self._lock:
                for piece, ids in new:
                    found[piece] = ids
                    if len(piece) >= _MIN_CACHED_CHARS:
                        self._insert(piece, ids)

        out: list[int] = []
        for is_added, piece in pieces:
            if is_added:
                out.append(self._added_id[piece])
            else:
                ids = found[piece]
                assert ids is not None
                out.extend(ids)
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


def _backend_of(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or not hasattr(backend, "encode_batch"):
        return None
    return backend


_INSTANCES: dict[int, SegmentTokenizer] = {}
_INSTANCES_LOCK = threading.Lock()


def get_segment_tokenizer(tokenizer: Any, max_cached_tokens: int) -> SegmentTokenizer:
    """One ``SegmentTokenizer`` per vLLM tokenizer object; raises if ineligible."""
    key = id(tokenizer)
    with _INSTANCES_LOCK:
        if key in _INSTANCES:
            return _INSTANCES[key]
        backend = _backend_of(tokenizer)
        if backend is None:
            reason = "no HF tokenizers backend"
        elif getattr(tokenizer, "split_special_tokens", False):
            reason = "split_special_tokens is set"
        else:
            reason = SegmentTokenizer.eligible(backend)
        if reason is not None:
            raise RuntimeError(
                "VLLM_TPU_SEGMENT_TOKENIZE=1 but the tokenizer is not eligible: "
                + reason
            )
        seg = SegmentTokenizer(backend, max_cached_tokens)
        logger.info(
            "VLLM_TPU_SEGMENT_TOKENIZE: piecewise prompt tokenization on "
            "(%d added tokens, cache %d tokens, ~12 B/token).",
            len(seg._added_id),
            max_cached_tokens,
        )
        _INSTANCES[key] = seg
        return seg


def patch_renderer_segment_tokenize() -> None:
    """Route eligible ``BaseRenderer._tokenize_prompt`` calls through the cache.

    Must run before the renderer is constructed: ``BaseRenderer.__init__``
    binds ``self._tokenize_prompt`` into its async wrapper.
    """
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_SEGMENT_TOKENIZE:
        return
    from vllm.renderers import base as renderer_base

    cls = renderer_base.BaseRenderer
    if getattr(cls, "_tpu_segment_tokenize", False):
        return
    orig = cls._tokenize_prompt
    max_cached = envs.VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS
    verify = envs.VLLM_TPU_SEGMENT_TOKENIZE_VERIFY
    stats = _Stats()

    def _tokenize_prompt(self, prompt, params):
        kwargs = params.get_encode_kwargs()
        if kwargs.get("add_special_tokens") or self._wants_offsets(prompt, params):
            return orig(self, prompt, params)
        seg = get_segment_tokenizer(self.get_tokenizer(), max_cached)
        t0 = time.perf_counter()
        ids = seg.encode(prompt["prompt"])
        stats.add(seg, len(ids), time.perf_counter() - t0)
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(ids) > max_length
        ):
            # Over-long prompt: keep the tokenizer's own truncation semantics.
            return orig(self, prompt, params)
        if verify:
            want = orig(self, dict(prompt), params)["prompt_token_ids"]
            if list(want) != ids:
                raise AssertionError(
                    "VLLM_TPU_SEGMENT_TOKENIZE_VERIFY: piecewise ids differ from "
                    f"the full encode ({len(ids)} vs {len(want)} tokens)"
                )
        return self._build_tokens_prompt(ids, prompt, offset_mapping=None)

    cls._tokenize_prompt = _tokenize_prompt
    cls._tpu_segment_tokenize = True
    logger.info(
        "VLLM_TPU_SEGMENT_TOKENIZE: patched BaseRenderer._tokenize_prompt%s",
        " (verifying every prompt against the full encode)" if verify else "",
    )


class _Stats:
    """Logs cumulative piecewise-encode cost every ``_LOG_EVERY`` prompts."""

    _LOG_EVERY = 200

    def __init__(self):
        self._lock = threading.Lock()
        self.prompts = self.tokens = 0
        self.seconds = 0.0

    def add(self, seg: SegmentTokenizer, tokens: int, seconds: float) -> None:
        with self._lock:
            self.prompts += 1
            self.tokens += tokens
            self.seconds += seconds
            if self.prompts % self._LOG_EVERY:
                return
            logger.info(
                "VLLM_TPU_SEGMENT_TOKENIZE: %d prompts, %d tokens, %.2f s encode "
                "(%.3f us/token); pieces %d cached / %d encoded; cache %d tokens.",
                self.prompts,
                self.tokens,
                self.seconds,
                1e6 * self.seconds / max(self.tokens, 1),
                seg.hits,
                seg.misses,
                seg._cached_tokens,
            )
