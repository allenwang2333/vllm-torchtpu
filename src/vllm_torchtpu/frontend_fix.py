# SPDX-License-Identifier: Apache-2.0
"""API-server patches for reasoning-token counting and incremental detokenizer priming.

* ``QX_LAZY_REASONING_COUNT=1``: skips per-delta ``DelegatingParser.count_reasoning_tokens``
  recomputation on intermediate streaming chunks when ``continuous_usage_stats`` is
  disabled, and short-circuits to 0 when the ``StreamingParserEngine`` state machine
  has no reachable ``REASONING_CHUNK`` transition or state.
* ``QX_DETOK_PROMPT_TAIL=8``: primes ``FastIncrementalDetokenizer``'s Rust
  ``DecodeStream`` with a UTF-8 boundary-verified prompt suffix instead of decoding
  the full prompt twice on the first emitted token.
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
    if _INSTALLED or os.environ.get("QX_LAZY_REASONING_COUNT", "1") != "1":
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


# ---------------------------------------------------------------------------
# Detokenizer prompt tail (QX_DETOK_PROMPT_TAIL).
#
# vLLM's FastIncrementalDetokenizer primes ``tokenizers.decoders.DecodeStream``
# with the *whole* prompt.  The first ``step()`` then decodes the prompt twice
# (``prefix`` and ``prefix + new``) on the API-server event loop: ~27 ms at
# 94k prompt ids, directly in TTFT.  After that first emitted chunk the stream
# drops every prompt id (``ids.drain(prefix_index..)``), so the prompt only
# matters as left context for the first chunk.
#
# For byte-level decoders the decoded text is ``from_utf8_lossy(concat(bytes))``
# so the left context only matters through UTF-8 synchronization.  If
# ``decode(tail)`` is non-empty and does not end with U+FFFD, the tail ends on
# a character boundary; then ``decode(P + new)[len(decode(P)):]`` equals
# ``decode(T + new)[len(decode(T)):]`` for every continuation, the stream's
# emit/hold decisions are identical, and after the first emitted chunk both
# streams hold exactly ``[new ids]`` (same state).  When the check fails (prompt
# ends inside a multi-byte character, or the tail decodes to "" because it is
# all skipped special tokens) we keep the full prompt, i.e. upstream behaviour.
# ---------------------------------------------------------------------------

_DT = "[QX_DETOK_PROMPT_TAIL]"
_DETOK_INSTALLED = False


class _DetokStats:
    tail = 0
    full_fallback = 0
    short = 0
    last_log = 0.0


def safe_prompt_tail(tokenizer, prompt_ids, skip_special_tokens: bool, tail_n: int):
    """Shortest prompt suffix that primes a DecodeStream identically to the
    full prompt, or ``prompt_ids`` itself when no safe tail exists.

    ``tokenizer`` is a ``tokenizers.Tokenizer`` (rust) instance.
    """
    if prompt_ids is None or tail_n <= 0 or len(prompt_ids) <= tail_n:
        return prompt_ids
    tail = list(prompt_ids[-tail_n:])
    try:
        text = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
    except Exception:
        return prompt_ids
    if not text or text.endswith("\ufffd"):
        return prompt_ids
    return tail


class _PromptTailView:
    """Read-only view of an EngineCoreRequest with a shortened prompt.

    FastIncrementalDetokenizer.__init__ only reads attributes of the request;
    everything except ``prompt_token_ids`` is delegated to the real request.
    """

    __slots__ = ("_qx_req", "prompt_token_ids")

    def __init__(self, req, prompt_token_ids):
        object.__setattr__(self, "_qx_req", req)
        object.__setattr__(self, "prompt_token_ids", prompt_token_ids)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_qx_req"), name)


def maybe_install_detok_prompt_tail() -> bool:
    global _DETOK_INSTALLED
    if _DETOK_INSTALLED:
        return False
    try:
        tail_n = int(os.environ.get("QX_DETOK_PROMPT_TAIL", "8"))
    except ValueError:
        tail_n = 8
    if tail_n <= 0:
        return False
    pname = multiprocessing.current_process().name
    if not (pname == "MainProcess" or pname.startswith("ApiServer")):
        return False
    try:
        import inspect

        from vllm.v1.engine import detokenizer as detok_mod
        cls = detok_mod.FastIncrementalDetokenizer
        src = inspect.getsource(cls.__init__)
    except Exception as e:
        logger.warning("%s cannot import FastIncrementalDetokenizer (%r); not enabled",
                       _DT, e)
        return False
    # Only patch the upstream shape we have proven equivalent: a DecodeStream
    # primed with ``ids=request.prompt_token_ids``.
    if "DecodeStream(" not in src or "ids=request.prompt_token_ids" not in src:
        logger.warning("%s FastIncrementalDetokenizer.__init__ changed upstream; "
                       "not enabled", _DT)
        return False

    orig_init = cls.__init__
    log_s = float(os.environ.get("QX_DETOK_PROMPT_TAIL_LOG_S", "300"))
    _DetokStats.last_log = time.monotonic()

    def __init__(self, tokenizer, request):
        try:
            ids = request.prompt_token_ids
            if ids is not None and len(ids) > tail_n:
                sp = request.sampling_params
                tail = safe_prompt_tail(
                    tokenizer._tokenizer, ids, bool(sp.skip_special_tokens), tail_n
                )
                if tail is not ids:
                    request = _PromptTailView(request, tail)
                    _DetokStats.tail += 1
                else:
                    _DetokStats.full_fallback += 1
            else:
                _DetokStats.short += 1
            now = time.monotonic()
            if now - _DetokStats.last_log >= log_s:
                _DetokStats.last_log = now
                logger.warning("%s stats tail=%d full_fallback=%d short=%d", _DT,
                               _DetokStats.tail, _DetokStats.full_fallback,
                               _DetokStats.short)
        except Exception:
            pass
        orig_init(self, tokenizer, request)

    __init__.__qx_detok_tail__ = True
    __init__.__wrapped__ = orig_init
    cls.__init__ = __init__
    _DETOK_INSTALLED = True
    logger.warning("%s installed FastIncrementalDetokenizer prompt-tail priming "
                   "(tail=%d, pid %d, %s)", _DT, tail_n, os.getpid(), pname)
    return True


# ---------------------------------------------------------------------------
# 5-Point Host TTFT Stage Instrumentation (QX_TTFT_STAGE_LOG_PERIOD_S).
#
# Tracks per-request host-path latencies across:
#   t0 (arrival / render start) -> t1 (render done / tokenize start)
#   -> t2 (tokenize done, recording PrefixTokenCache hit/miss)
#   -> t3 (process_for_engine done / AsyncLLM.add_request start)
#   -> t4 (EngineCore add_request_async dispatch complete)
# When QX_TTFT_STAGE_LOG_PERIOD_S <= 0 (default 0), no hooks are installed and
# runtime overhead is strictly zero.
# ---------------------------------------------------------------------------

_TTFT_P = "[QX_TTFT_STAGES]"
_TTFT_INSTALLED = False


def _p50_p90(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    k = len(s)
    p50 = s[min(int(0.50 * (k - 1)), k - 1)]
    p90 = s[min(int(0.90 * (k - 1)), k - 1)]
    return p50, p90


class _TTFTStageStats:
    n: int = 0
    tok_cache_hits: int = 0
    render_ms: list[float] = []
    tokenize_ms: list[float] = []
    ipc_ms: list[float] = []
    last_log: float = 0.0
    log_period_s: float = 0.0
    MAX_SAMPLES: int = 8192
    MAX_PENDING: int = 1024
    pending_by_id: dict[int, tuple[float, float, float, bool]] = {}
    pending_by_arrival: dict[float, tuple[float, float, float, bool]] = {}

    @classmethod
    def reset(cls, now: float | None = None) -> None:
        cls.n = 0
        cls.tok_cache_hits = 0
        cls.render_ms = []
        cls.tokenize_ms = []
        cls.ipc_ms = []
        cls.last_log = time.monotonic() if now is None else now
        cls.pending_by_id = {}
        cls.pending_by_arrival = {}

    @classmethod
    def put_pending(
        cls,
        eng_prompt,
        t_tok_done: float,
        render_ms: float,
        tokenize_ms: float,
        tok_hit: bool,
    ) -> None:
        if len(cls.pending_by_id) >= cls.MAX_PENDING:
            cls.pending_by_id.clear()
        if len(cls.pending_by_arrival) >= cls.MAX_PENDING:
            cls.pending_by_arrival.clear()
        entry = (t_tok_done, render_ms, tokenize_ms, tok_hit)
        cls.pending_by_id[id(eng_prompt)] = entry
        if isinstance(eng_prompt, dict):
            arr = eng_prompt.get("arrival_time")
            if isinstance(arr, (int, float)):
                cls.pending_by_arrival[float(arr)] = entry

    @classmethod
    def pop_pending(
        cls, prompt, arrival_time: float | None = None
    ) -> tuple[float, float, float, bool] | None:
        entry = cls.pending_by_id.pop(id(prompt), None)
        arr = arrival_time
        if arr is None and isinstance(prompt, dict):
            arr = prompt.get("arrival_time")
        if isinstance(arr, (int, float)):
            arr_entry = cls.pending_by_arrival.pop(float(arr), None)
            if entry is None:
                entry = arr_entry
        return entry

    @classmethod
    def format_summary(cls) -> str:
        hit_pct = (100.0 * cls.tok_cache_hits / cls.n) if cls.n > 0 else 0.0
        r50, r90 = _p50_p90(cls.render_ms)
        t50, t90 = _p50_p90(cls.tokenize_ms)
        i50, i90 = _p50_p90(cls.ipc_ms)
        return (
            f"{_TTFT_P} n={cls.n} "
            f"tok_cache_hit={cls.tok_cache_hits}/{hit_pct:.1f}% "
            f"render_ms p50/p90={r50:.2f}/{r90:.2f} "
            f"tokenize_ms p50/p90={t50:.2f}/{t90:.2f} "
            f"ipc_ms p50/p90={i50:.2f}/{i90:.2f}"
        )

    @classmethod
    def record(
        cls,
        render_ms: float,
        tokenize_ms: float,
        ipc_ms: float,
        tok_cache_hit: bool,
        *,
        now: float | None = None,
    ) -> str | None:
        cls.n += 1
        if tok_cache_hit:
            cls.tok_cache_hits += 1
        if len(cls.render_ms) < cls.MAX_SAMPLES:
            cls.render_ms.append(float(render_ms))
            cls.tokenize_ms.append(float(tokenize_ms))
            cls.ipc_ms.append(float(ipc_ms))
        t_now = time.monotonic() if now is None else now
        if cls.log_period_s > 0 and (t_now - cls.last_log) >= cls.log_period_s:
            summary = cls.format_summary()
            logger.warning("%s", summary)
            cls.n = 0
            cls.tok_cache_hits = 0
            cls.render_ms.clear()
            cls.tokenize_ms.clear()
            cls.ipc_ms.clear()
            cls.last_log = t_now
            return summary
        return None


def _get_tokcache_hits() -> int:
    try:
        from vllm_torchtpu import frontend_tokcache as _tokcache_mod

        cache = getattr(_tokcache_mod, "_CACHE", None)
        return int(getattr(cache, "hits", 0) or 0) if cache is not None else 0
    except Exception:
        return 0


def maybe_install_ttft_stage_tracker() -> bool:
    global _TTFT_INSTALLED
    if _TTFT_INSTALLED:
        return False
    from vllm_torchtpu import envs

    period_s = float(envs.QX_TTFT_STAGE_LOG_PERIOD_S)
    if period_s <= 0:
        return False
    pname = multiprocessing.current_process().name
    if not (pname == "MainProcess" or pname.startswith("ApiServer")):
        return False
    try:
        import asyncio
        from vllm.renderers import base as rbase
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as e:
        logger.warning("%s cannot import vLLM frontend classes (%r); not enabled",
                       _TTFT_P, e)
        return False

    _TTFTStageStats.log_period_s = period_s
    _TTFTStageStats.reset(time.monotonic())

    orig_render_chat_async = rbase.BaseRenderer.render_chat_async
    orig_render_chat = rbase.BaseRenderer.render_chat
    orig_render_cmpl_async = rbase.BaseRenderer.render_cmpl_async
    orig_render_cmpl = rbase.BaseRenderer.render_cmpl
    orig_add_request = AsyncLLM.add_request

    async def render_chat_async(
        self,
        conversations,
        chat_params,
        tok_params=None,
        *,
        prompt_extras=None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()
        t0 = time.perf_counter()
        if tok_params is None:
            tok_params = self.default_chat_tok_params
        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]
        out_conversations = []
        dict_prompts = []
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)
        t1 = time.perf_counter()
        hits0 = _get_tokcache_hits()
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)
        t2 = time.perf_counter()
        tok_hit = _get_tokcache_hits() > hits0
        self._apply_prompt_extras(tok_prompts, prompt_extras)
        eng_prompts = await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )
        render_ms = (t1 - t0) * 1000.0
        tokenize_ms = (t2 - t1) * 1000.0
        for ep in eng_prompts:
            _TTFTStageStats.put_pending(ep, t2, render_ms, tokenize_ms, tok_hit)
        return out_conversations, eng_prompts

    def render_chat(
        self,
        conversations,
        chat_params,
        tok_params=None,
        *,
        prompt_extras=None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()
        t0 = time.perf_counter()
        if tok_params is None:
            tok_params = self.default_chat_tok_params
        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]
        out_conversations = []
        dict_prompts = []
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)
        t1 = time.perf_counter()
        hits0 = _get_tokcache_hits()
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)
        t2 = time.perf_counter()
        tok_hit = _get_tokcache_hits() > hits0
        self._apply_prompt_extras(tok_prompts, prompt_extras)
        eng_prompts = [
            self.process_for_engine(
                prompt, arrival_time, skip_mm_cache=skip_mm_cache
            )
            for prompt in tok_prompts
        ]
        render_ms = (t1 - t0) * 1000.0
        tokenize_ms = (t2 - t1) * 1000.0
        for ep in eng_prompts:
            _TTFTStageStats.put_pending(ep, t2, render_ms, tokenize_ms, tok_hit)
        return out_conversations, eng_prompts

    async def render_cmpl_async(
        self,
        prompts,
        tok_params=None,
        *,
        prompt_extras=None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()
        t0 = time.perf_counter()
        if tok_params is None:
            tok_params = self.default_cmpl_tok_params
        dict_prompts = await self.render_prompts_async(prompts)
        t1 = time.perf_counter()
        hits0 = _get_tokcache_hits()
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)
        t2 = time.perf_counter()
        tok_hit = _get_tokcache_hits() > hits0
        self._apply_prompt_extras(tok_prompts, prompt_extras)
        eng_prompts = await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )
        render_ms = (t1 - t0) * 1000.0
        tokenize_ms = (t2 - t1) * 1000.0
        for ep in eng_prompts:
            _TTFTStageStats.put_pending(ep, t2, render_ms, tokenize_ms, tok_hit)
        return eng_prompts

    def render_cmpl(
        self,
        prompts,
        tok_params=None,
        *,
        prompt_extras=None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()
        t0 = time.perf_counter()
        if tok_params is None:
            tok_params = self.default_cmpl_tok_params
        dict_prompts = self.render_prompts(prompts)
        t1 = time.perf_counter()
        hits0 = _get_tokcache_hits()
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)
        t2 = time.perf_counter()
        tok_hit = _get_tokcache_hits() > hits0
        self._apply_prompt_extras(tok_prompts, prompt_extras)
        eng_prompts = [
            self.process_for_engine(
                prompt, arrival_time, skip_mm_cache=skip_mm_cache
            )
            for prompt in tok_prompts
        ]
        render_ms = (t1 - t0) * 1000.0
        tokenize_ms = (t2 - t1) * 1000.0
        for ep in eng_prompts:
            _TTFTStageStats.put_pending(ep, t2, render_ms, tokenize_ms, tok_hit)
        return eng_prompts

    async def add_request(
        self,
        request_id: str,
        prompt,
        params,
        *args,
        **kwargs,
    ):
        t_add_start = time.perf_counter()
        arrival_time = kwargs.get(
            "arrival_time", args[0] if len(args) > 0 else None
        )
        pending = _TTFTStageStats.pop_pending(prompt, arrival_time=arrival_time)
        out = await orig_add_request(
            self, request_id, prompt, params, *args, **kwargs
        )
        t_add_done = time.perf_counter()
        try:
            if pending is not None:
                t_tok_done, render_ms, tokenize_ms, tok_hit = pending
                ipc_ms = max((t_add_done - t_tok_done) * 1000.0, 0.0)
            else:
                render_ms = 0.0
                tokenize_ms = 0.0
                tok_hit = False
                ipc_ms = max((t_add_done - t_add_start) * 1000.0, 0.0)
            _TTFTStageStats.record(render_ms, tokenize_ms, ipc_ms, tok_hit)
        except Exception:
            pass
        return out

    render_chat_async.__qx_ttft_stages__ = True
    render_chat_async.__wrapped__ = orig_render_chat_async
    render_chat.__qx_ttft_stages__ = True
    render_chat.__wrapped__ = orig_render_chat
    render_cmpl_async.__qx_ttft_stages__ = True
    render_cmpl_async.__wrapped__ = orig_render_cmpl_async
    render_cmpl.__qx_ttft_stages__ = True
    render_cmpl.__wrapped__ = orig_render_cmpl
    add_request.__qx_ttft_stages__ = True
    add_request.__wrapped__ = orig_add_request

    rbase.BaseRenderer.render_chat_async = render_chat_async
    rbase.BaseRenderer.render_chat = render_chat
    rbase.BaseRenderer.render_cmpl_async = render_cmpl_async
    rbase.BaseRenderer.render_cmpl = render_cmpl
    AsyncLLM.add_request = add_request
    _TTFT_INSTALLED = True
    logger.warning(
        "%s installed 5-point host TTFT stage tracker (period=%.1fs, pid %d, %s)",
        _TTFT_P,
        period_s,
        os.getpid(),
        pname,
    )
    return True

