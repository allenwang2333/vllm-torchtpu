# SPDX-License-Identifier: Apache-2.0
"""Env-gated incremental prompt tokenization for the OpenAI API server.

QX_TOKENIZE_PREFIX_CACHE=1 enables it (default off == stock vLLM behaviour).

Multi-turn agentic chat re-sends the whole conversation every turn, so vLLM
re-renders and re-tokenizes 100k-200k tokens per request in the API server.
The HF fast tokenizer holds the GIL for the whole encode (~1.2 ms per 1k
tokens on the server CPU) and the renderer has a single worker thread by
default, so this cost is paid serially in front of every request.

This module caches token ids of previously tokenized prompts at
``<|im_start|>`` boundaries.  ``<|im_start|>`` is a special added token
(normalized=False, lstrip/rstrip=False), and the HF tokenizer splits the input
on added tokens *before* normalization / pre-tokenization / BPE, so for any
position ``p`` where ``text[p:]`` starts with ``<|im_start|>``::

    encode(text) == encode(text[:p]) + encode(text[p:])

provided the post-processor adds no tokens (checked at install time).  For a
new prompt we find the longest boundary prefix whose bytes hash (blake2b-128
of the exact UTF-8 prefix, plus its length) matches a cached prefix, reuse its
token ids, and tokenize only the suffix.  The k-th ``<|im_start|>`` substring
maps to the k-th ``<|im_start|>`` token id; this is verified on every call
(count check on the encoded suffix) and optionally by full re-tokenization.

Env vars:
  QX_TOKENIZE_PREFIX_CACHE=1               enable
  QX_TOKENIZE_PREFIX_CACHE_ENTRIES=64       max cached prompts (LRU)
  QX_TOKENIZE_PREFIX_CACHE_MAX_TOKENS=16000000  max total cached token ids
  QX_TOKENIZE_PREFIX_CACHE_MIN_TOKENS=2048  only use/cache prompts >= this
  QX_TOKENIZE_PREFIX_CACHE_VERIFY_FIRST=8   verify first N hits by full encode
  QX_TOKENIZE_PREFIX_CACHE_VERIFY_EVERY=0   additionally verify every N-th hit
  QX_TOKENIZE_PREFIX_CACHE_LOG_S=60         stats log period (seconds)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from array import array
from collections import OrderedDict

logger = logging.getLogger("vllm.qx_tokcache")

BOUNDARY = "<|im_start|>"
_BOUNDARY_B = BOUNDARY.encode("utf-8")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class _Entry:
    __slots__ = ("ids", "bounds", "keys", "ntok")

    def __init__(self, ids: array, bounds: list[tuple[int, int]], keys: list[bytes]):
        self.ids = ids          # array('I') of token ids for the full prompt
        self.bounds = bounds    # [(byte_pos, tok_idx)] for each boundary (pos>0)
        self.keys = keys        # prefix keys registered for this entry
        self.ntok = len(ids)


class PrefixTokenCache:
    """Thread-safe LRU of tokenized prompts addressable by boundary prefixes."""

    def __init__(self, encode_fn, boundary_id: int, *, max_entries: int = 64,
                 max_tokens: int = 16_000_000, min_tokens: int = 2048,
                 verify_first: int = 8, verify_every: int = 0,
                 log_period_s: float = 60.0):
        self._encode = encode_fn            # str -> list[int] (full, no truncation)
        self._bid = boundary_id
        self._max_entries = max_entries
        self._max_tokens = max_tokens
        self._min_tokens = min_tokens
        self._verify_first = verify_first
        self._verify_every = verify_every
        self._log_period = log_period_s
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._index: dict[bytes, tuple[int, int, int]] = {}  # key -> (eid, tok_idx, bidx)
        self._next_eid = 0
        self._total_tokens = 0
        self.disabled = False
        # stats
        self.calls = self.hits = self.misses = self.verified = 0
        self.tokens_reused = self.tokens_encoded = 0
        self.time_hit_s = self.time_miss_s = 0.0
        self.miss_tokens = 0
        self._last_log = time.monotonic()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _boundaries(b: bytes) -> list[int]:
        out = []
        find = b.find
        p = find(_BOUNDARY_B, 1)
        while p != -1:
            out.append(p)
            p = find(_BOUNDARY_B, p + len(_BOUNDARY_B))
        return out

    @staticmethod
    def _prefix_keys(b: bytes, bpos: list[int]) -> list[bytes]:
        h = hashlib.blake2b(digest_size=16)
        mv = memoryview(b)
        keys = []
        prev = 0
        for p in bpos:
            h.update(mv[prev:p])
            prev = p
            keys.append(p.to_bytes(6, "little") + h.copy().digest())
        return keys

    # --------------------------------------------------------------- encode
    def encode(self, text: str) -> list[int]:
        if self.disabled:
            return self._encode(text)
        t0 = time.perf_counter()
        self.calls += 1
        b = text.encode("utf-8", "surrogatepass")
        bpos = self._boundaries(b)
        if not bpos:
            return self._encode(text)
        keys = self._prefix_keys(b, bpos)

        hit = None
        with self._lock:
            for i in range(len(keys) - 1, -1, -1):
                v = self._index.get(keys[i])
                if v is not None:
                    e = self._entries.get(v[0])
                    if e is not None:
                        self._entries.move_to_end(v[0])
                        hit = (i, v[1], v[2], e)
                        break
        ids: list[int]
        if hit is not None:
            bi, tok_idx, ebi, e = hit
            p = bpos[bi]
            suffix = b[p:].decode("utf-8", "surrogatepass")
            suf_ids = self._encode(suffix)
            # Boundary <-> token correspondence check on the suffix.
            nb_suffix = len(bpos) - bi
            bt = []
            idx = -1
            bid = self._bid
            try:
                for _ in range(nb_suffix):
                    idx = suf_ids.index(bid, idx + 1)
                    bt.append(idx)
                ok = suf_ids[0] == bid if suf_ids else False
                try:
                    suf_ids.index(bid, idx + 1)
                    ok = False  # more boundary tokens than boundary substrings
                except ValueError:
                    pass
            except ValueError:
                ok = False
            if not ok:
                logger.warning("[QX_TOKCACHE] boundary/token mismatch; full encode")
                return self._encode(text)
            prefix_ids = e.ids[:tok_idx].tolist()
            ids = prefix_ids + suf_ids
            # boundary token offsets: reuse the entry's for the shared prefix
            bounds = [(bpos[j], e.bounds[j][1]) for j in range(bi)]
            bounds += [(bpos[bi + j], tok_idx + bt[j]) for j in range(nb_suffix)]
            self.hits += 1
            self.tokens_reused += tok_idx
            self.tokens_encoded += len(suf_ids)
            if (self.verified < self._verify_first
                    or (self._verify_every and self.hits % self._verify_every == 0)):
                full = self._encode(text)
                self.verified += 1
                if full != ids:
                    self.disabled = True
                    logger.error(
                        "[QX_TOKCACHE] VERIFY MISMATCH (len %d vs %d); disabling "
                        "the tokenize prefix cache permanently", len(full), len(ids))
                    return full
            self.time_hit_s += time.perf_counter() - t0
        else:
            ids = self._encode(text)
            # boundary token offsets via C-level list.index scans; the k-th
            # boundary substring (pos > 0) must be the k-th boundary token
            # after an optional leading one, and there must be no extras.
            bid = self._bid
            idx = 0 if (ids and ids[0] == bid and b.startswith(_BOUNDARY_B)) else -1
            bounds = []
            try:
                for p in bpos:
                    idx = ids.index(bid, idx + 1)
                    bounds.append((p, idx))
                try:
                    ids.index(bid, idx + 1)
                    bounds = None
                except ValueError:
                    pass
            except ValueError:
                bounds = None
            self.misses += 1
            self.miss_tokens += len(ids)
            self.tokens_encoded += len(ids)
            self.time_miss_s += time.perf_counter() - t0
            if bounds is None:
                return ids

        if len(ids) >= self._min_tokens:
            self._insert(ids, bounds, keys)
        self._maybe_log()
        return ids

    def _insert(self, ids: list[int], bounds: list[tuple[int, int]], keys: list[bytes]):
        arr = array("I", ids)
        with self._lock:
            eid = self._next_eid
            self._next_eid += 1
            e = _Entry(arr, bounds, keys)
            self._entries[eid] = e
            self._total_tokens += e.ntok
            idx = self._index
            for j, k in enumerate(keys):
                idx[k] = (eid, bounds[j][1], j)
            while self._entries and (len(self._entries) > self._max_entries
                                     or self._total_tokens > self._max_tokens):
                old_eid, old = self._entries.popitem(last=False)
                self._total_tokens -= old.ntok
                for k in old.keys:
                    v = idx.get(k)
                    if v is not None and v[0] == old_eid:
                        del idx[k]

    def stats(self) -> dict:
        ns_per_tok = (self.time_miss_s / self.miss_tokens * 1e9) if self.miss_tokens else 0.0
        saved = self.tokens_reused * ns_per_tok / 1e9
        return dict(calls=self.calls, hits=self.hits, misses=self.misses,
                    hit_rate=round(self.hits / max(1, self.hits + self.misses), 3),
                    tokens_reused=self.tokens_reused, tokens_encoded=self.tokens_encoded,
                    est_time_saved_s=round(saved, 2),
                    miss_ms_per_1k_tok=round(ns_per_tok / 1e3, 3),
                    hit_time_s=round(self.time_hit_s, 2), entries=len(self._entries),
                    cached_tokens=self._total_tokens, verified=self.verified,
                    disabled=self.disabled)

    def _maybe_log(self):
        now = time.monotonic()
        if now - self._last_log >= self._log_period:
            self._last_log = now
            logger.warning("[QX_TOKCACHE] stats %s", self.stats())


_CACHE: PrefixTokenCache | None = None
_INSTALLED = False


def _build_cache_for(tokenizer) -> PrefixTokenCache | None:
    """Create the cache around a private copy of the Rust tokenizer."""
    try:
        from tokenizers import Tokenizer
    except Exception:  # pragma: no cover
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
    if backend is None or not isinstance(backend, Tokenizer):
        logger.warning("[QX_TOKCACHE] tokenizer %s has no Rust backend; not enabled",
                       type(tokenizer).__name__)
        return None
    boundary_id = tokenizer.convert_tokens_to_ids(BOUNDARY)
    if not isinstance(boundary_id, int) or boundary_id < 0:
        logger.warning("[QX_TOKCACHE] %s not in vocab; not enabled", BOUNDARY)
        return None
    # Per-thread private copies: no truncation/padding state, no RefCell
    # borrow conflicts with the vLLM tokenizer pool.
    blob = backend.to_str()
    local = threading.local()

    def get_tok():
        t = getattr(local, "tok", None)
        if t is None:
            t = Tokenizer.from_str(blob)
            t.no_truncation()
            t.no_padding()
            local.tok = t
        return t

    def encode(s: str) -> list[int]:
        # NB: Tokenizer.encode() holds the GIL for the whole call (measured:
        # a spinning event-loop thread drops to ~1% throughput), whereas
        # encode_batch() runs inside py.allow_threads like vLLM's stock path.
        return get_tok().encode_batch([s], add_special_tokens=False)[0].ids

    # Safety checks: boundary token is atomic and post-processor adds nothing.
    probe = "x<|im_start|>user\nhi<|im_end|>\n"
    t = get_tok()
    if (t.encode("a", add_special_tokens=True).ids != t.encode("a", add_special_tokens=False).ids
            or encode(probe) != tokenizer(probe, add_special_tokens=False)["input_ids"]
            or encode(probe).count(boundary_id) != 1):
        logger.warning("[QX_TOKCACHE] tokenizer safety probe failed; not enabled")
        return None
    return PrefixTokenCache(
        encode, boundary_id,
        max_entries=_env_int("QX_TOKENIZE_PREFIX_CACHE_ENTRIES", 64),
        max_tokens=_env_int("QX_TOKENIZE_PREFIX_CACHE_MAX_TOKENS", 16_000_000),
        min_tokens=_env_int("QX_TOKENIZE_PREFIX_CACHE_MIN_TOKENS", 2048),
        verify_first=_env_int("QX_TOKENIZE_PREFIX_CACHE_VERIFY_FIRST", 8),
        verify_every=_env_int("QX_TOKENIZE_PREFIX_CACHE_VERIFY_EVERY", 0),
        log_period_s=float(_env_int("QX_TOKENIZE_PREFIX_CACHE_LOG_S", 60)),
    )


def maybe_install_tokenize_prefix_cache() -> bool:
    """Monkeypatch BaseRenderer._tokenize_prompt (API server). Idempotent."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("QX_TOKENIZE_PREFIX_CACHE", "0") != "1":
        return False
    try:
        from vllm.renderers import base as rbase
        from vllm.renderers.hf import HfRenderer
    except Exception as e:  # pragma: no cover
        logger.warning("[QX_TOKCACHE] cannot import vLLM renderer (%r); not enabled", e)
        return False

    orig = rbase.BaseRenderer._tokenize_prompt
    cache_lock = threading.Lock()
    state: dict = {"cache": None, "tried": False}

    def _get_cache(renderer):
        if not state["tried"]:
            with cache_lock:
                if not state["tried"]:
                    state["cache"] = _build_cache_for(renderer.get_tokenizer())
                    state["tried"] = True
                    global _CACHE
                    _CACHE = state["cache"]
                    if state["cache"] is not None:
                        logger.warning("[QX_TOKCACHE] active: incremental prompt "
                                       "tokenization at %s boundaries (pid %d)",
                                       BOUNDARY, os.getpid())
        return state["cache"]

    def _tokenize_prompt(self, prompt, params):
        try:
            if (isinstance(self, HfRenderer)
                    and not self._wants_offsets(prompt, params)
                    and not prompt.get("multi_modal_data")
                    and isinstance(prompt.get("prompt"), str)):
                kw = params.get_encode_kwargs()
                if set(kw) <= {"truncation", "max_length", "add_special_tokens"}:
                    cache = _get_cache(self)
                    if cache is not None and not cache.disabled:
                        ids = cache.encode(prompt["prompt"])
                        max_len = kw.get("max_length")
                        if not (kw.get("truncation") and max_len is not None
                                and len(ids) > max_len):
                            return self._build_tokens_prompt(ids, prompt)
        except Exception as e:  # never break serving
            logger.warning("[QX_TOKCACHE] error %r; falling back", e)
        return orig(self, prompt, params)

    _tokenize_prompt.__qx_tokcache__ = True
    rbase.BaseRenderer._tokenize_prompt = _tokenize_prompt
    _INSTALLED = True
    logger.warning("[QX_TOKCACHE] installed BaseRenderer._tokenize_prompt patch "
                   "(QX_TOKENIZE_PREFIX_CACHE=1, pid %d)", os.getpid())
    return True
