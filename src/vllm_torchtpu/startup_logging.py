# SPDX-License-Identifier: Apache-2.0
"""Structured startup phase logging for wrapper-owned vLLM processes."""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

EVENT_PREFIX = "VLLM_STARTUP_EVENT "
SCHEMA_VERSION = 1
_SPAN_SEQUENCE = itertools.count(1)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_TerminalState = Literal["end", "error"]
StartupIdentityResolver = Callable[[Any], tuple[int | None, int | None]]


@dataclass(frozen=True)
class _StartupIdentity:
    startup_id: str
    service_id: str
    role: str
    model_alias: str
    request_id: str
    invocation_id: str
    origin_monotonic_ns: int
    origin_unix_ns: int


@dataclass(frozen=True)
class StartupSpan:
    identity: _StartupIdentity
    phase: str
    span_id: str
    rank: int | None
    world_size: int | None
    detail: str | None
    started_monotonic_ns: int


def _warn_without_raising(message: str, *args: object) -> None:
    try:
        logger.warning(message, *args)
    except Exception:
        pass


def _read_identity() -> _StartupIdentity | None:
    startup_id = os.environ.get("VLLM_STARTUP_ID")
    if startup_id is None:
        return None
    try:
        identity = _StartupIdentity(
            startup_id=startup_id,
            service_id=os.environ["VLLM_SERVICE_ID"],
            role=os.environ["VLLM_ROLE"],
            model_alias=os.environ["VLLM_MODEL_ALIAS"],
            request_id=os.environ["VLLM_REQUEST_ID"],
            invocation_id=os.environ["VLLM_INVOCATION_ID"],
            origin_monotonic_ns=int(
                os.environ["VLLM_STARTUP_ORIGIN_MONOTONIC_NS"]),
            origin_unix_ns=int(os.environ["VLLM_STARTUP_ORIGIN_UNIX_NS"]),
        )
    except (KeyError, ValueError) as error:
        _warn_without_raising(
            "Disabling structured startup logging: invalid wrapper identity: %s",
            error,
        )
        return None
    if (not identity.startup_id or identity.origin_monotonic_ns < 0
            or identity.origin_unix_ns < 0):
        _warn_without_raising(
            "Disabling structured startup logging: invalid startup origin")
        return None
    return identity


def process_startup_identity() -> tuple[int | None, int | None]:
    """Return the slice-global rank identity established by TPU init."""
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None or world_size is None:
        return None, None
    try:
        parsed_rank = int(rank)
        parsed_world_size = int(world_size)
    except ValueError:
        _warn_without_raising(
            "Structured startup rank identity is invalid: RANK=%r WORLD_SIZE=%r",
            rank,
            world_size,
        )
        return None, None
    if (parsed_rank < 0 or parsed_world_size < 1
            or parsed_rank >= parsed_world_size):
        _warn_without_raising(
            "Structured startup rank identity is out of range: "
            "RANK=%d WORLD_SIZE=%d",
            parsed_rank,
            parsed_world_size,
        )
        return None, None
    return parsed_rank, parsed_world_size


def _emit(
    span: StartupSpan,
    state: Literal["start", "end", "error"],
    *,
    now_monotonic_ns: int,
    now_unix_ns: int,
    error_type: str | None = None,
) -> None:
    identity = span.identity
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "startup_id": identity.startup_id,
        "service_id": identity.service_id,
        "role": identity.role,
        "model_alias": identity.model_alias,
        "request_id": identity.request_id,
        "invocation_id": identity.invocation_id,
        "phase": span.phase,
        "state": state,
        "span_id": span.span_id,
        "rank": span.rank,
        "world_size": span.world_size,
        "pid": os.getpid(),
        "origin_unix_ns": identity.origin_unix_ns,
        "ts_unix_ns": now_unix_ns,
        "elapsed_ms": round(
            (now_monotonic_ns - identity.origin_monotonic_ns) / 1_000_000,
            3,
        ),
    }
    if span.detail is not None:
        payload["detail"] = span.detail
    if state != "start":
        payload["duration_ms"] = round(
            (now_monotonic_ns - span.started_monotonic_ns) / 1_000_000,
            3,
        )
    if error_type is not None:
        payload["error_type"] = error_type
    logger.info(
        "%s%s",
        EVENT_PREFIX,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def _emit_without_raising(
    span: StartupSpan,
    state: Literal["start", "end", "error"],
    *,
    error_type: str | None = None,
) -> None:
    try:
        _emit(
            span,
            state,
            now_monotonic_ns=time.monotonic_ns(),
            now_unix_ns=time.time_ns(),
            error_type=error_type,
        )
    except Exception as error:
        _warn_without_raising(
            "Failed to emit structured startup event for %s: %s",
            span.phase,
            error,
        )


def begin_startup_span(
    phase: str,
    *,
    rank: int | None,
    world_size: int | None,
    detail: str | None = None,
) -> StartupSpan | None:
    identity = _read_identity()
    if identity is None:
        return None
    started_monotonic_ns = time.monotonic_ns()
    if started_monotonic_ns < identity.origin_monotonic_ns:
        _warn_without_raising(
            "Disabling structured startup span %s: monotonic origin is in the future",
            phase,
        )
        return None
    span = StartupSpan(
        identity=identity,
        phase=phase,
        span_id=f"{os.getpid()}:{next(_SPAN_SEQUENCE)}",
        rank=rank,
        world_size=world_size,
        detail=detail,
        started_monotonic_ns=started_monotonic_ns,
    )
    _emit_without_raising(span, "start")
    return span


def finish_startup_span(
    span: StartupSpan | None,
    state: _TerminalState = "end",
    *,
    error_type: str | None = None,
) -> None:
    if span is not None:
        _emit_without_raising(span, state, error_type=error_type)


@contextmanager
def startup_span(
    phase: str,
    *,
    rank: int | None,
    world_size: int | None,
    detail: str | None = None,
) -> Iterator[StartupSpan | None]:
    span = begin_startup_span(
        phase,
        rank=rank,
        world_size=world_size,
        detail=detail,
    )
    try:
        yield span
    except BaseException as error:
        finish_startup_span(span, "error", error_type=type(error).__name__)
        raise
    else:
        finish_startup_span(span)


def startup_logged(
    phase: str,
    identity: StartupIdentityResolver,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:

    def decorator(function: Callable[_P, _R]) -> Callable[_P, _R]:

        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if "VLLM_STARTUP_ID" not in os.environ:
                return function(*args, **kwargs)
            try:
                rank, world_size = identity(args[0])
            except Exception as error:
                _warn_without_raising(
                    "Structured startup identity resolution failed for %s: %s",
                    phase,
                    error,
                )
                rank, world_size = None, None
            with startup_span(phase, rank=rank, world_size=world_size):
                return function(*args, **kwargs)

        return wrapped

    return decorator
