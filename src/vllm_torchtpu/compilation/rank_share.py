# SPDX-License-Identifier: Apache-2.0
"""Compile each precompile bucket once per host and share the artifact.

With TPU_PARALLEL_PRECOMPILE every rank compiles every token bucket of every
piecewise subgraph up front (see ``_patch_vllm_compile_all_ranges``), in an
order rotated per rank so that the tier-2 XLA cache hands each XLA compile to
whichever rank reaches it first. aot_autograd and the FX->MLIR lowering are not
cached, though: they still run on every rank for every bucket, and for a large
model they are most of the startup.

What ``TpuCompilerAdaptor.compile`` saves per bucket -- torch's bundled
AOTAutograd entry, wrapper and executable together -- is exactly what a warm
start replays, and nothing in it belongs to the rank that built it as long as
the ranks traced the same graph: each worker drives its own chip as device 0,
and the ranks lower identical programs, which is why the tier-2 cache already
hits across them. So the ranks of a host split the buckets instead: a rank
claims a bucket before compiling it, and replays the claimant's artifact for a
bucket another rank claimed.

Claims go through the default process group's store, so they are scoped to the
run and need no cleanup. They are namespaced by host, because the artifacts are
files on it, and by a fingerprint of the graph -- its generated code, input
signature and constants -- so only ranks that traced the same graph share, and a
rank whose graph differs compiles its own buckets.
"""

import contextlib
import hashlib
import operator
import os
import socket
import time
from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.fx as fx

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Constants are read back to the host to fingerprint them. A graph holding one
# larger than this is compiled by every rank rather than read back.
_MAX_FINGERPRINT_CONSTANT_BYTES = 1 << 20
_POLL_S = 0.1
# A claimant that dies takes the engine down with it, so this only bounds one
# that is alive but stuck; past it the waiting rank compiles the bucket itself.
_WAIT_S = 1800.0


def fingerprint(graph: fx.GraphModule) -> str | None:
    """Identify `graph` across processes, or None if it cannot be pinned down.

    Covers what decides the compiled program: the generated code, the input
    signature, and the values of the constants the graph reads through
    get_attr. Parameters are graph inputs, so their values never enter.
    """
    digest = hashlib.sha256()
    digest.update(repr(shape_variants.graph_signature(graph)).encode())
    digest.update(graph.code.encode())
    for node in graph.graph.nodes:
        if node.op != "get_attr":
            continue
        value = operator.attrgetter(node.target)(graph)
        digest.update(str(node.target).encode())
        if isinstance(value, fx.GraphModule):
            inner = fingerprint(value)
            if inner is None:
                return None
            digest.update(inner.encode())
        elif isinstance(value, torch.Tensor):
            if value.numel() * value.element_size() > _MAX_FINGERPRINT_CONSTANT_BYTES:
                return None
            digest.update(repr((value.dtype, tuple(value.shape))).encode())
            data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(data.numpy().tobytes())
        else:
            return None
    return digest.hexdigest()[:16]


class Session:
    """One rank's view of one subgraph's buckets."""

    def __init__(self, store: Any, namespace: str) -> None:
        self._store = store
        self._namespace = namespace
        self._claimed: set[str] = set()
        self._published: set[str] = set()
        self.replayed = 0

    def _key(self, bucket: str, what: str) -> str:
        return f"{self._namespace}/{bucket}/{what}"

    def claim(self, bucket: str) -> bool:
        """Whether this rank compiles `bucket`; False if a peer claimed it."""
        if self._store.add(self._key(bucket, "claim"), 1) != 1:
            return False
        self._claimed.add(bucket)
        return True

    def publish(self, bucket: str, path: str | None) -> None:
        """Tell the peers where a claimed bucket's artifact is, once.

        None tells them there is none, and they compile the bucket themselves.
        """
        if bucket not in self._claimed or bucket in self._published:
            return
        self._published.add(bucket)
        self._store.set(self._key(bucket, "artifact"), path or "")

    def published(self, bucket: str) -> bool:
        return self._store.check([self._key(bucket, "artifact")])

    def artifact(self, bucket: str) -> str | None:
        """The artifact a peer published for `bucket`, if there is one."""
        if bucket in self._claimed or not self.published(bucket):
            return None
        return self._store.get(self._key(bucket, "artifact")).decode() or None

    @property
    def compiled(self) -> int:
        return len(self._claimed)


def _default_store() -> Any | None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    if dist.get_world_size() < 2:
        return None
    try:
        return dist.distributed_c10d._get_default_store()
    except Exception as e:
        logger.warning("[bucket-share] No process group store, not sharing: %s", e)
        return None


def open_session(graph: fx.GraphModule) -> Session | None:
    """Share `graph`'s buckets with the other ranks, or None to compile all."""
    if not envs.TPU_PRECOMPILE_SHARE_BUCKETS:
        return None
    store = _default_store()
    if store is None:
        return None
    try:
        graph_id = fingerprint(graph)
    except Exception as e:
        logger.warning("[bucket-share] Cannot fingerprint the graph: %s", e)
        graph_id = None
    if graph_id is None:
        logger.info("[bucket-share] Graph not fingerprintable, compiling all buckets")
        return None
    logger.info("[bucket-share] Sharing the buckets of graph %s", graph_id)
    namespace = f"vllm_torchtpu/bucket_share/{socket.gethostname()}/{graph_id}"
    return Session(store, namespace)


_current: tuple[Session, str] | None = None


@contextlib.contextmanager
def compiling(session: Session | None, bucket: str) -> Iterator[None]:
    """Scope one bucket's compile, so the compiler can replay or publish it.

    A claimed bucket is published on the way out whatever happened -- as "no
    artifact" if the compile raised, or never reached the compiler because vLLM
    loaded the bucket from this rank's own cache -- so no peer waits on it.
    """
    global _current
    if session is None:
        yield
        return
    _current = (session, bucket)
    try:
        yield
    finally:
        _current = None
        session.publish(bucket, None)


def peer_artifact() -> str | None:
    """The artifact a peer published for the bucket being compiled, if any."""
    if _current is None:
        return None
    session, bucket = _current
    return session.artifact(bucket)


def publish(path: str | None) -> None:
    """Offer the artifact just saved for the bucket being compiled."""
    if _current is not None:
        session, bucket = _current
        session.publish(bucket, path)


def replayed() -> None:
    """Count a bucket replayed from a peer's artifact."""
    if _current is not None:
        _current[0].replayed += 1


def as_published[T](
    session: Session | None, pending: Iterable[tuple[str, T]]
) -> Iterator[tuple[str, T]]:
    """Yield each pending ``(bucket, item)`` once its bucket is published.

    In the order the peers publish, so this rank replays whatever is ready
    instead of queueing behind the slowest bucket. Past _WAIT_S the rest is
    yielded as is, to be compiled here.
    """
    waiting = list(pending)
    if session is None or not waiting:
        yield from waiting
        return
    deadline = time.monotonic() + _WAIT_S
    while waiting:
        ready = [entry for entry in waiting if session.published(entry[0])]
        if not ready and time.monotonic() >= deadline:
            logger.warning(
                "[bucket-share] Waited %.0fs for peers; compiling %d bucket(s) here",
                _WAIT_S,
                len(waiting),
            )
            ready = list(waiting)
        if not ready:
            time.sleep(_POLL_S)
            continue
        for entry in ready:
            waiting.remove(entry)
            yield entry


def link(src: str, dst: str) -> str:
    """Hard-link a peer's artifact to `dst`, and return the path to record.

    Keeps each rank's cache directory self-contained, as if the rank had saved
    the artifact itself. Where a link cannot be made, the peer's own path is
    recorded instead; it lives under the same cache root.
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    tmp = f"{dst}.{os.getpid()}.link"
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        os.link(src, tmp)
        os.replace(tmp, dst)
        return dst
    except OSError as e:
        logger.info("[bucket-share] Cannot link %s to %s (%s)", src, dst, e)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return src
