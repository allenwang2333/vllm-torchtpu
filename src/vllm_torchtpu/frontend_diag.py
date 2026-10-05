# SPDX-License-Identifier: Apache-2.0
"""Env-gated, instrumentation-only API-server frontend diagnostics.

QX_FE_DIAG=1 enables it (default off == nothing is patched, stock behaviour).

Goal: explain the bimodal request-ack tail (time from HTTP request start to
response headers) of the OpenAI chat endpoint.  Installed from the general
plugin hook (``layers.register_layers``) in the API-server process only.

What it records (all timing via time.perf_counter):

* Per request (``/...completions`` POSTs), through a FastAPI.__call__ ASGI
  wrapper plus a request-scoped contextvar:
    body   ASGI call start -> last request-body chunk received
    json   starlette Request.json (json.loads) duration
    pyd    json end -> OpenAIServingChat.create_chat_completion entry
           (pydantic validation + FastAPI deps + load_aware/with_cancellation)
    chk    handler entry -> parse_chat_messages start (_check_model, etc.)
    parse  parse_chat_messages_async (runs on the event loop)
    tq/trun  chat-template executor queue wait / run (Jinja, in thread)
    g1     template done (thread) -> tokenize submitted (loop wake-up gap)
    kq/krun  tokenize executor queue wait / run (+ tokcache hit/miss, ISL)
    post   tokenize done -> handler returned (process_for_engine, etc.)
    hdr    handler returned -> http.response.start sent
    fe     ASGI call start -> http.response.start  (== server-side ack)
    addw/add  headers -> AsyncLLM.add_request start / add_request duration
    first  headers -> first non-keepalive body chunk (server-side 1st token)
  One compact line per request (QX_FE_DIAG_PER_REQ=1, default) and
  p50/p90/p99/max per segment every QX_FE_DIAG_LOG_S seconds (default 60).

* Event-loop lag probe: asyncio task sleeping 10 ms, overshoot histogram,
  stall counts (>50/100/300 ms).
* Watchdog thread: when the loop has not ticked for > QX_FE_DIAG_BLOCK_MS
  (default 50) it samples sys._current_frames() every QX_FE_DIAG_SAMPLE_MS
  (default 5) and aggregates collapsed loop-thread stacks, loop leaf frames
  and other threads' (non-idle) leaf frames.  It also counts its own late
  wake-ups (> BLOCK_MS late => GIL held by C code / CPU starvation) and
  records all thread leaf frames at that moment.
* gc.callbacks pause histogram per generation.
* Output path: OutputProcessor.process_outputs (detokenize etc., on the
  loop) and DelegatingParser/ParserEngine.parse_delta per-call histograms.

Every hook is wrapped so that a diagnostics error can never break serving.

Env vars:
  QX_FE_DIAG=1               enable
  QX_FE_DIAG_LOG_S=60        periodic summary period
  QX_FE_DIAG_BLOCK_MS=50     loop-blocked threshold for stack sampling
  QX_FE_DIAG_SAMPLE_MS=5     watchdog sampling period
  QX_FE_DIAG_SLOW_MS=300     fe threshold for the "slow" counters
  QX_FE_DIAG_PER_REQ=1       per-request log line
  QX_FE_DIAG_TOPN=10         stacks / frames per summary
  QX_FE_DIAG_ANYPROC=0       install regardless of process name
"""

from __future__ import annotations

import asyncio
import atexit
import bisect
import contextvars
import functools
import gc
import logging
import math
import multiprocessing
import os
import sys
import threading
import time
from collections import Counter

logger = logging.getLogger("vllm.qx_fe_diag")
_P = "[QX_FE_DIAG]"
perf = time.perf_counter

_INSTALLED = False
_LAZY_DONE = False
_CUR: contextvars.ContextVar = contextvars.ContextVar("qx_fe_diag_rec", default=None)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


LOG_S = _env_float("QX_FE_DIAG_LOG_S", 60.0)
BLOCK_S = _env_float("QX_FE_DIAG_BLOCK_MS", 50.0) / 1e3
SAMPLE_S = _env_float("QX_FE_DIAG_SAMPLE_MS", 5.0) / 1e3
SLOW_S = _env_float("QX_FE_DIAG_SLOW_MS", 300.0) / 1e3
PER_REQ = os.environ.get("QX_FE_DIAG_PER_REQ", "1") == "1"
TOPN = int(_env_float("QX_FE_DIAG_TOPN", 10))
PROBE_S = 0.010
STACK_DEPTH = 24


# ----------------------------------------------------------------- histogram
_EDGES = [1e-6 * (1.1 ** k) for k in range(200)]  # 1 us .. ~190 s


class Hist:
    __slots__ = ("c", "n", "sum", "max")

    def __init__(self):
        self.c = [0] * (len(_EDGES) + 1)
        self.n = 0
        self.sum = 0.0
        self.max = 0.0

    def add(self, v: float) -> None:
        if v < 0:
            v = 0.0
        self.c[bisect.bisect_left(_EDGES, v)] += 1
        self.n += 1
        self.sum += v
        if v > self.max:
            self.max = v

    def q(self, p: float) -> float:
        if not self.n:
            return float("nan")
        k = max(1, math.ceil(p * self.n))
        acc = 0
        for i, c in enumerate(self.c):
            acc += c
            if acc >= k:
                return min(_EDGES[min(i, len(_EDGES) - 1)], self.max)
        return self.max

    def fmt(self) -> str:
        if not self.n:
            return "-"
        return "%.1f/%.1f/%.1f/%.1f" % (self.q(.5) * 1e3, self.q(.9) * 1e3,
                                        self.q(.99) * 1e3, self.max * 1e3)

    def count_over(self, v: float) -> int:
        return sum(self.c[bisect.bisect_left(_EDGES, v) + 1:])


class _Hists(dict):
    def __missing__(self, k):
        h = self[k] = Hist()
        return h


# ------------------------------------------------------------------- state
class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.seg = _Hists()          # per-request segment histograms (period)
        self.seg_hit = _Hists()      # same, split by tokcache hit/miss
        self.glob = _Hists()         # loop lag, gc, output path (period)
        self.nreq = self.nslow = 0
        self.inflight = 0
        self.loop_tid = None
        self.loop = None
        self.probe_task = None
        self.wd_tid = None
        self.last_tick = None
        self.stalls = 0
        self.stall_samples = 0
        self.wd_late = 0
        self.stk = Counter()         # period collapsed loop stacks
        self.leaf = Counter()        # period loop leaf frames
        self.oth = Counter()         # period other-thread leaf frames (stalls)
        self.late = Counter()        # period thread leaf frames at wd late wake
        self.stk_all = Counter()     # cumulative
        self.leaf_all = Counter()
        self.oth_all = Counter()
        self.names = {}
        self.names_t = 0.0
        self.periods = 0
        self.t_start = perf()
        self.active_logged = False
        self.gc_t0 = {}


S = _State()


# ----------------------------------------------------------- frame helpers
_IDLE = {
    ("threading.py", "wait"), ("threading.py", "_wait_for_tstate_lock"),
    ("selectors.py", "select"), ("queue.py", "get"), ("thread.py", "_worker"),
    ("connection.py", "_poll"), ("connection.py", "wait"),
    ("socket.py", "accept"), ("selectors.py", "poll"),
}


def _fr(f, line=True) -> str:
    co = f.f_code
    fn = co.co_filename
    i = fn.rfind("site-packages/")
    if i >= 0:
        fn = fn[i + 14:]
    else:
        fn = os.path.basename(fn)
    return "%s:%s:%d" % (fn, co.co_name, f.f_lineno) if line else "%s:%s" % (fn, co.co_name)


def _collapse(f) -> str:
    out = []
    while f is not None and len(out) < STACK_DEPTH:
        out.append(_fr(f))
        f = f.f_back
    out.reverse()
    return ";".join(out)


def _is_idle(f) -> bool:
    co = f.f_code
    return (os.path.basename(co.co_filename), co.co_name) in _IDLE


def _thread_names():
    now = perf()
    if now - S.names_t > 2.0:
        try:
            S.names = {t.ident: t.name for t in threading.enumerate()}
        except Exception:
            pass
        S.names_t = now
    return S.names


# ------------------------------------------------------------ request rec
class Rec:
    __slots__ = ("t0", "wall", "path", "bytes", "t_body", "j0", "j1", "h",
                 "ps", "pe", "nmsg", "tsub", "t0t", "t1t", "ksub", "k0", "k1",
                 "hit", "isl", "ret", "hdr", "a0", "a1", "first", "end",
                 "status", "infl", "done")

    def __init__(self, path):
        for s in self.__slots__:
            setattr(self, s, None)
        self.t0 = perf()
        self.wall = time.time()
        self.path = path
        self.bytes = 0
        self.done = False


def _d(a, b):
    if a is None or b is None:
        return None
    return b - a


def _finish(r: Rec) -> None:
    if r.done:
        return
    r.done = True
    try:
        segs = {
            "body": _d(r.t0, r.t_body),
            "json": _d(r.j0, r.j1),
            "pyd": _d(r.j1 if r.j1 is not None else r.t_body, r.h),
            "chk": _d(r.h, r.ps),
            "parse": _d(r.ps, r.pe),
            "tq": _d(r.tsub, r.t0t),
            "trun": _d(r.t0t, r.t1t),
            "g1": _d(r.t1t, r.ksub),
            "kq": _d(r.ksub, r.k0),
            "krun": _d(r.k0, r.k1),
            "post": _d(r.k1, r.ret),
            "hdr": _d(r.ret, r.hdr),
            "fe": _d(r.t0, r.hdr),
            "addw": _d(r.hdr, r.a0),
            "add": _d(r.a0, r.a1),
            "first": _d(r.hdr, r.first),
        }
        hk = "hit" if r.hit == 1 else ("miss" if r.hit == 0 else "na")
        with S.lock:
            S.nreq += 1
            fe = segs["fe"]
            if fe is not None and fe > SLOW_S:
                S.nslow += 1
            for k, v in segs.items():
                if v is not None:
                    S.seg[k].add(v)
                    if k in ("krun", "fe", "kq", "trun"):
                        S.seg_hit[hk + "." + k].add(v)
        if PER_REQ:
            def m(v):
                return "-" if v is None else "%.1f" % (v * 1e3)
            logger.warning(
                "%s req t=%.3f isl=%s hit=%s bytes=%d nmsg=%s infl=%s st=%s %s",
                _P, r.wall, r.isl if r.isl is not None else "-",
                "-" if r.hit is None else r.hit, r.bytes or 0,
                r.nmsg if r.nmsg is not None else "-", r.infl, r.status,
                " ".join("%s=%s" % (k, m(v)) for k, v in segs.items()))
    except Exception as e:  # pragma: no cover
        logger.warning("%s finish error %r", _P, e)


# --------------------------------------------------------------- ASGI hook
def _wrap_fastapi_call(orig_call):
    @functools.wraps(orig_call)
    async def __call__(self, scope, receive, send):
        r = None
        try:
            if (scope.get("type") == "http" and scope.get("method") == "POST"
                    and str(scope.get("path", "")).endswith("completions")):
                r = Rec(scope.get("path"))
                _lazy_install()
        except Exception:
            r = None
        if r is None:
            return await orig_call(self, scope, receive, send)

        async def rcv():
            msg = await receive()
            try:
                if msg.get("type") == "http.request":
                    r.bytes += len(msg.get("body", b""))
                    if not msg.get("more_body", False):
                        r.t_body = perf()
            except Exception:
                pass
            return msg

        async def snd(msg):
            try:
                t = msg.get("type")
                if t == "http.response.start":
                    r.hdr = perf()
                    r.status = msg.get("status")
                elif t == "http.response.body":
                    if r.first is None:
                        b = msg.get("body", b"")
                        if b and not b.startswith(b":"):
                            r.first = perf()
                    if not msg.get("more_body", False):
                        r.end = perf()
            except Exception:
                pass
            return await send(msg)

        S.inflight += 1
        r.infl = S.inflight
        tok = _CUR.set(r)
        try:
            return await orig_call(self, scope, rcv, snd)
        finally:
            S.inflight -= 1
            try:
                _CUR.reset(tok)
            except Exception:
                pass
            _finish(r)

    __call__.__qx_fe_diag__ = True
    return __call__


# ---------------------------------------------------------- generic hooks
def _wrap_async_mark(orig, start_attr, end_attr, extra=None):
    @functools.wraps(orig)
    async def w(*a, **kw):
        r = _CUR.get()
        if r is None:
            return await orig(*a, **kw)
        try:
            if getattr(r, start_attr) is None:
                setattr(r, start_attr, perf())
            if extra is not None:
                extra(r, a, kw)
        except Exception:
            pass
        try:
            return await orig(*a, **kw)
        finally:
            if end_attr is not None and getattr(r, end_attr) is None:
                setattr(r, end_attr, perf())

    w.__qx_fe_diag__ = True
    return w


def _nmsg(r, a, kw):
    msgs = a[0] if a else kw.get("messages")
    try:
        r.nmsg = len(msgs)
    except Exception:
        pass


def _tokcache_counts():
    m = sys.modules.get("vllm_torchtpu.frontend_tokcache")
    c = getattr(m, "_CACHE", None) if m is not None else None
    if c is None:
        return None
    return c.hits, c.misses


def _make_async_diag(orig_make_async):
    """Replacement for vllm.utils.async_utils.make_async (same semantics:
    returns a function returning the run_in_executor Future) that records
    executor queue wait / run time for the chat template and tokenizer."""

    def make_async(func, executor=None):
        name = getattr(func, "__name__", type(func).__name__)
        kind = ("t" if "chat_template" in name
                else "k" if name == "_tokenize_prompt" else None)
        if kind is None:
            return orig_make_async(func, executor)

        def _async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            r = _CUR.get()
            if r is None:
                p_func = functools.partial(func, *args, **kwargs)
                return loop.run_in_executor(executor=executor, func=p_func)
            sub = perf()

            def run():
                t0 = perf()
                c0 = _tokcache_counts() if kind == "k" else None
                res = None
                try:
                    res = func(*args, **kwargs)
                    return res
                finally:
                    t1 = perf()
                    try:
                        if kind == "t":
                            if r.tsub is None:
                                r.tsub, r.t0t, r.t1t = sub, t0, t1
                        elif r.ksub is None:
                            r.ksub, r.k0, r.k1 = sub, t0, t1
                            c1 = _tokcache_counts()
                            if c0 is not None and c1 is not None:
                                r.hit = (1 if c1[0] > c0[0]
                                         else 0 if c1[1] > c0[1] else None)
                            if isinstance(res, dict):
                                ids = res.get("prompt_token_ids")
                                if ids is not None:
                                    r.isl = len(ids)
                    except Exception:
                        pass

            return loop.run_in_executor(executor=executor, func=run)

        return _async_wrapper

    make_async.__qx_fe_diag__ = True
    return make_async


def _wrap_sync_hist(orig, key, count_arg=False):
    @functools.wraps(orig)
    def w(*a, **kw):
        t0 = perf()
        try:
            return orig(*a, **kw)
        finally:
            dt = perf() - t0
            S.glob[key].add(dt)
            if count_arg and len(a) > 1:
                try:
                    S.glob[key + "_n"].add(len(a[1]) * 1e-3)  # stored as ms==count
                except Exception:
                    pass

    w.__qx_fe_diag__ = True
    return w


def _patch(obj, attr, factory, *fa):
    cur = getattr(obj, attr)
    if getattr(cur, "__qx_fe_diag__", False):
        return False
    setattr(obj, attr, factory(cur, *fa))
    return True


# ----------------------------------------------------------- lazy install
def _lazy_install():
    """First matching request (on the event loop): patch serving-layer
    classes (all fully imported by now) and start probe + watchdog."""
    global _LAZY_DONE
    if _LAZY_DONE:
        return
    _LAZY_DONE = True
    done = []
    try:
        S.loop_tid = threading.get_ident()
        S.last_tick = perf()
        S.loop = asyncio.get_running_loop()
        S.probe_task = S.loop.create_task(_probe())
        threading.Thread(target=_watchdog, name="qx-fe-diag-wd", daemon=True).start()
        done.append("probe+watchdog")
    except Exception as e:
        logger.warning("%s probe/watchdog start failed %r", _P, e)
    steps = [
        ("vllm.entrypoints.openai.chat_completion.serving", "OpenAIServingChat",
         "create_chat_completion", _wrap_async_mark, ("h", "ret")),
        ("vllm.v1.engine.async_llm", "AsyncLLM", "add_request",
         _wrap_async_mark, ("a0", "a1")),
        ("vllm.renderers.hf", None, "parse_chat_messages_async",
         _wrap_async_mark, ("ps", "pe", _nmsg)),
        ("starlette.requests", "Request", "json", _wrap_async_mark, ("j0", "j1")),
        ("vllm.v1.engine.output_processor", "OutputProcessor", "process_outputs",
         _wrap_sync_hist, ("process_outputs", True)),
        ("vllm.parser.abstract_parser", "DelegatingParser", "parse_delta",
         _wrap_sync_hist, ("parse_delta",)),
        ("vllm.parser.engine.parser_engine", "ParserEngine", "parse_delta",
         _wrap_sync_hist, ("engine_parse_delta",)),
    ]
    for mod, cls, attr, fac, fa in steps:
        try:
            # Only already-imported modules: never import on the event loop
            # (an unimported module's classes are not in use anyway).
            m = sys.modules.get(mod)
            if m is None:
                done.append("skip:%s(not loaded)" % mod)
                continue
            obj = getattr(m, cls) if cls else m
            if _patch(obj, attr, fac, *fa):
                done.append("%s.%s" % (cls or mod.rsplit(".", 1)[-1], attr))
        except Exception as e:
            logger.warning("%s patch %s.%s failed %r", _P, cls or mod, attr, e)
    try:
        gc.callbacks.append(_gc_cb)
        done.append("gc")
    except Exception:
        pass
    logger.warning("%s active (pid %d, loop tid %s): %s", _P, os.getpid(),
                   S.loop_tid, ",".join(done))


# ------------------------------------------------------- probe / watchdog
async def _probe():
    lag_h = "loop_lag"
    while True:
        t = perf()
        S.last_tick = t
        try:
            await asyncio.sleep(PROBE_S)
        except asyncio.CancelledError:
            return
        now = perf()
        S.last_tick = now
        S.glob[lag_h].add(max(0.0, now - t - PROBE_S))


def _gc_cb(phase, info):
    try:
        g = info.get("generation", -1)
        if phase == "start":
            S.gc_t0[g] = perf()
        else:
            t0 = S.gc_t0.pop(g, None)
            if t0 is not None:
                S.glob["gc%d" % g].add(perf() - t0)
    except Exception:
        pass


def _watchdog():
    S.wd_tid = threading.get_ident()
    next_log = perf() + LOG_S
    in_stall = False
    thr = BLOCK_S + PROBE_S
    while True:
        t_sleep = perf()
        time.sleep(SAMPLE_S)
        now = perf()
        try:
            late = now - t_sleep - SAMPLE_S
            frames = None
            if late > BLOCK_S:
                S.wd_late += 1
                frames = sys._current_frames()
                names = _thread_names()
                for tid, f in frames.items():
                    if tid != S.wd_tid and not _is_idle(f):
                        S.late["%s|%s" % (names.get(tid, tid), _fr(f))] += 1
            lt = S.last_tick
            lp = S.loop
            if (lt is not None and now - lt > thr and S.loop_tid is not None
                    and (lp is None or lp.is_running())):
                if not in_stall:
                    in_stall = True
                    S.stalls += 1
                if frames is None:
                    frames = sys._current_frames()
                f = frames.get(S.loop_tid)
                if f is not None:
                    S.stall_samples += 1
                    st = _collapse(f)
                    S.stk[st] += 1
                    lf = _fr(f)
                    S.leaf[lf] += 1
                    names = _thread_names()
                    for tid, of in frames.items():
                        if tid in (S.loop_tid, S.wd_tid) or _is_idle(of):
                            continue
                        S.oth["%s|%s" % (names.get(tid, tid), _fr(of))] += 1
            else:
                in_stall = False
            if now >= next_log:
                next_log = now + LOG_S
                _log_period()
        except Exception as e:  # never die
            try:
                logger.warning("%s watchdog error %r", _P, e)
            except Exception:
                pass


def _top(c: Counter, n: int):
    return c.most_common(n)


def _log_period(final: bool = False) -> None:
    with S.lock:
        seg, S.seg = S.seg, _Hists()
        segh, S.seg_hit = S.seg_hit, _Hists()
        glob, S.glob = S.glob, _Hists()
        nreq, nslow = S.nreq, S.nslow
        S.nreq = S.nslow = 0
    stk, S.stk = S.stk, Counter()
    leaf, S.leaf = S.leaf, Counter()
    oth, S.oth = S.oth, Counter()
    late, S.late = S.late, Counter()
    stalls, S.stalls = S.stalls, 0
    samples, S.stall_samples = S.stall_samples, 0
    wd_late, S.wd_late = S.wd_late, 0
    S.stk_all.update(stk)
    S.leaf_all.update(leaf)
    S.oth_all.update(oth)
    S.periods += 1
    tag = "final" if final else "period"
    order = ["fe", "body", "json", "pyd", "chk", "parse", "tq", "trun", "g1",
             "kq", "krun", "post", "hdr", "addw", "add", "first"]
    logger.warning("%s %s req=%d slow(>%.0fms)=%d seg(ms p50/p90/p99/max) %s",
                   _P, tag, nreq, SLOW_S * 1e3, nslow,
                   " ".join("%s=%s" % (k, seg[k].fmt()) for k in order if k in seg))
    if segh:
        logger.warning("%s %s by_tokcache %s", _P, tag,
                       " ".join("%s=%s(n%d)" % (k, segh[k].fmt(), segh[k].n)
                                for k in sorted(segh)))
    lag = glob.get("loop_lag") or Hist()
    logger.warning(
        "%s %s loop lag(ms p50/p90/p99/max)=%s n=%d >20ms=%d >50ms=%d >100ms=%d "
        ">300ms=%d stalls=%d stall_samples=%d wd_late=%d gc0=%s(n%d) gc1=%s(n%d) "
        "gc2=%s(n%d)", _P, tag, lag.fmt(), lag.n, lag.count_over(.02),
        lag.count_over(.05), lag.count_over(.1), lag.count_over(.3), stalls,
        samples, wd_late,
        *[x for g in range(3) for x in ((glob.get("gc%d" % g) or Hist()).fmt(),
                                         (glob.get("gc%d" % g) or Hist()).n)])
    po = glob.get("process_outputs") or Hist()
    pn = glob.get("process_outputs_n") or Hist()
    pd = glob.get("parse_delta") or Hist()
    pe = glob.get("engine_parse_delta") or Hist()
    logger.warning(
        "%s %s out process_outputs(ms)=%s n=%d sum=%.2fs batch(p50/p90/p99/max)=%s "
        "parse_delta(ms)=%s n=%d sum=%.2fs engine_parse_delta(ms)=%s n=%d sum=%.2fs",
        _P, tag, po.fmt(), po.n, po.sum, pn.fmt(), pd.fmt(), pd.n, pd.sum,
        pe.fmt(), pe.n, pe.sum)
    for k, c in _top(stk, max(1, TOPN // 2)):
        logger.warning("%s %s stk %d %s", _P, tag, c, k)
    for k, c in _top(leaf, TOPN):
        logger.warning("%s %s leaf %d %s", _P, tag, c, k)
    for k, c in _top(oth, TOPN):
        logger.warning("%s %s oth %d %s", _P, tag, c, k)
    for k, c in _top(late, TOPN):
        logger.warning("%s %s late %d %s", _P, tag, c, k)
    if final or S.periods % 5 == 0:
        ctag = "cum_final" if final else "cum"
        for k, c in _top(S.stk_all, TOPN):
            logger.warning("%s %s stk %d %s", _P, ctag, c, k)
        for k, c in _top(S.leaf_all, TOPN):
            logger.warning("%s %s leaf %d %s", _P, ctag, c, k)
        for k, c in _top(S.oth_all, TOPN):
            logger.warning("%s %s oth %d %s", _P, ctag, c, k)
    for cn in (S.stk_all, S.leaf_all, S.oth_all):  # bound memory
        if len(cn) > 20000:
            keep = cn.most_common(5000)
            cn.clear()
            cn.update(dict(keep))


def _atexit():
    if _LAZY_DONE:
        try:
            _log_period(final=True)
        except Exception:
            pass


# ------------------------------------------------------------------ install
def maybe_install_fe_diag() -> bool:
    """Install the early (pre-renderer-init) patches. Idempotent."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("QX_FE_DIAG", "0") != "1":
        return False
    pname = multiprocessing.current_process().name
    if not (pname == "MainProcess" or pname.startswith("ApiServer")
            or os.environ.get("QX_FE_DIAG_ANYPROC", "0") == "1"):
        return False
    done = []
    try:
        import fastapi
        if _patch(fastapi.FastAPI, "__call__", _wrap_fastapi_call):
            done.append("FastAPI.__call__")
    except Exception as e:
        logger.warning("%s FastAPI patch failed %r", _P, e)
        return False
    # Renderers bind make_async(...) at __init__; patch the module globals now
    # (plugin load precedes renderer construction, as for QX_TOKCACHE).
    for mod in ("vllm.renderers.base", "vllm.renderers.hf"):
        try:
            import importlib
            m = importlib.import_module(mod)
            if _patch(m, "make_async", _make_async_diag):
                done.append(mod + ".make_async")
        except Exception as e:
            logger.warning("%s %s.make_async patch failed %r", _P, mod, e)
    atexit.register(_atexit)
    _INSTALLED = True
    logger.warning("%s installed (pid %d, %s): %s; log_s=%.0f block_ms=%.0f "
                   "sample_ms=%.0f", _P, os.getpid(), pname, ",".join(done),
                   LOG_S, BLOCK_S * 1e3, SAMPLE_S * 1e3)
    return True
