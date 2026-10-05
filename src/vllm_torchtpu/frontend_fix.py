# SPDX-License-Identifier: Apache-2.0
"""Env-gated fix: avoid O(n^2) reasoning-token recounting in chat streaming.

QX_LAZY_REASONING_COUNT=1 enables it (default off == stock behaviour).

Problem (found with QX_FE_DIAG): OpenAIServingChat.chat_completion_stream_generator
calls ``parser.count_reasoning_tokens(tuple(generated_token_ids[i]))`` after
*every* streamed delta.  For the Qwen3 engine parsers (DelegatingParser over
``ParserEngineReasoningAdapter``), when the prompt already ends reasoning
(``is_reasoning_end(prompt)``; e.g. agentic turns with an empty/closed think
block) the streaming reasoning phase is skipped, ``_streaming_count_valid`` stays
False, and every call does ``tokenizer.decode(all ids)`` + a full pure-Python
single-pass re-parse of the whole output (incremental_lexer
``_find_content_boundary``).  That is O(output_len) per delta => O(n^2) per
request, all on the API-server event loop.  At c16 it saturated the loop and the
GIL (loop lag p50 ~50 ms, p90 ~110 ms in heavy minutes), which delayed request
acks (render/tokenize executor work starved of the GIL, queueing behind it).

The value is only observed (a) in per-chunk usage when
``stream_options.continuous_usage_stats`` is set and (b) in the final usage /
request metadata after the last delta (``finished=True``).  This patch computes
it exactly (same stock code on the same full token tuple) only when it can be
observed, and returns the previous value otherwise.  The final value is
therefore identical to stock; intermediate values are unobservable.

Scope: only DelegatingParser instances whose reasoning parser is a
``ParserEngineReasoningAdapter`` in the expensive (non-streaming-count) mode,
and only right after a streaming ``parse_delta(finished=False)`` call of a
request without continuous usage stats.  The non-streaming path (no preceding
parse_delta) is untouched.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time

logger = logging.getLogger("vllm.qx_fe_fix")
_P = "[QX_LAZY_REASONING_COUNT]"
_INSTALLED = False


class _Stats:
    skipped = 0
    zero = 0
    computed = 0
    compute_s = 0.0
    last_log = 0.0


_ZERO_CACHE: dict = {}


def _config_never_reasons(cfg) -> bool:
    """True iff the engine state machine can never emit REASONING_CHUNK.

    StreamingParserEngine only sets ``state`` to the config's initial state
    (reset), a transition's ``next_state``, or CONTENT; REASONING_CHUNK events
    come only from ``content_events[state]`` or ``transition.events``.  So if
    no state reachable from {initial, CONTENT} maps to REASONING_CHUNK and no
    reachable transition emits it, ``reasoning_token_count`` is always 0.
    """
    from vllm.parser.engine.events import EventType
    from vllm.parser.engine.parser_engine_config import ParserState

    rc = EventType.REASONING_CHUNK
    reach = {cfg.initial_state, ParserState.CONTENT}
    changed = True
    while changed:
        changed = False
        for (src, _term), tr in cfg.transitions.items():
            if src in reach:
                if rc in tr.events:
                    return False
                if tr.next_state not in reach:
                    reach.add(tr.next_state)
                    changed = True
    return all(cfg.content_events.get(s) != rc for s in reach)


def _provably_zero(rp) -> bool:
    try:
        pe = rp._parser_engine
        from vllm.parser.engine.parser_engine import ParserEngine
        if (type(pe).count_reasoning_tokens is not ParserEngine.count_reasoning_tokens
                or type(pe._engine).__name__ != "StreamingParserEngine"):
            return False
        cfg = pe.parser_engine_config
    except Exception:
        return False
    key = id(cfg)
    v = _ZERO_CACHE.get(key)
    if v is None or v[0] is not cfg:
        try:
            v = (cfg, _config_never_reasons(cfg))
        except Exception:
            v = (cfg, False)
        if len(_ZERO_CACHE) > 256:
            _ZERO_CACHE.clear()
        _ZERO_CACHE[key] = v
    return v[1]


def maybe_install_lazy_reasoning_count() -> bool:
    global _INSTALLED
    if _INSTALLED or os.environ.get("QX_LAZY_REASONING_COUNT", "0") != "1":
        return False
    pname = multiprocessing.current_process().name
    if not (pname == "MainProcess" or pname.startswith("ApiServer")):
        return False
    try:
        from vllm.parser.abstract_parser import DelegatingParser
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
    except Exception as e:
        logger.warning("%s cannot import vLLM parser classes (%r); not enabled", _P, e)
        return False

    orig_parse_delta = DelegatingParser.parse_delta
    orig_count = DelegatingParser.count_reasoning_tokens
    log_s = float(os.environ.get("QX_LAZY_REASONING_COUNT_LOG_S", "60"))
    _Stats.last_log = time.monotonic()

    def parse_delta(self, *args, **kwargs):
        res = orig_parse_delta(self, *args, **kwargs)
        try:
            finished = kwargs.get("finished", True)
            request = kwargs.get("request", args[2] if len(args) > 2 else None)
            so = getattr(request, "stream_options", None)
            cont = bool(so is not None and getattr(so, "continuous_usage_stats", False))
            self._qx_skip_count = (finished is False) and not cont
        except Exception:
            self._qx_skip_count = False
        return res

    def _maybe_log():
        now = time.monotonic()
        if now - _Stats.last_log >= log_s:
            _Stats.last_log = now
            logger.warning("%s stats zero=%d skipped=%d computed=%d compute_s=%.2f", _P,
                           _Stats.zero, _Stats.skipped, _Stats.computed, _Stats.compute_s)

    def count_reasoning_tokens(self, token_ids):
        skip = False
        try:
            skip = self.__dict__.pop("_qx_skip_count", False)
            rp = getattr(self, "_reasoning_parser", None)
            if (isinstance(rp, ParserEngineReasoningAdapter)
                    and not getattr(rp, "_streaming_count_valid", True)):
                if _provably_zero(rp):
                    _Stats.zero += 1
                    _maybe_log()
                    return 0
            else:
                skip = False
        except Exception:
            skip = False
        if skip:
            _Stats.skipped += 1
            return self.__dict__.get("_qx_last_count", 0)
        t0 = time.perf_counter()
        n = orig_count(self, token_ids)
        _Stats.computed += 1
        _Stats.compute_s += time.perf_counter() - t0
        try:
            self._qx_last_count = n
            _maybe_log()
        except Exception:
            pass
        return n

    parse_delta.__qx_lazy__ = True
    count_reasoning_tokens.__qx_lazy__ = True
    DelegatingParser.parse_delta = parse_delta
    DelegatingParser.count_reasoning_tokens = count_reasoning_tokens
    _INSTALLED = True
    logger.warning("%s installed DelegatingParser.parse_delta/count_reasoning_tokens "
                   "patch (pid %d, %s)", _P, os.getpid(), pname)
    return True
