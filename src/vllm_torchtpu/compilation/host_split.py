# SPDX-License-Identifier: Apache-2.0
"""Compile one token bucket on all the ranks of a host at once.

Under TPU_PARALLEL_PRECOMPILE with peer replay, each bucket has one owner rank
per host and the others replay its artifact. The biggest bucket then bounds
startup on its own: on GLM-5.2 the 4096-token bucket is one program of ~300k
HLO instructions that takes its owner ~26 minutes (14 of FX->MLIR lowering, 11
of XLA), while every other rank of the host has finished its own buckets and
waits.

With TPU_PRECOMPILE_SPLIT_MIN_TOKENS, such a bucket is instead compiled by
every rank of the host. Each rank traces it through aot_autograd as usual, then
hands it to torch_tpu's SplitCompiler -- which can split a graph into
partitions and compile each into its own executable, and with collective
materialization off (as vllm_torchtpu runs) cuts nowhere by itself -- and cuts
it between decoder layers into one piece per rank (`layer_groups`). Each rank lowers and
compiles only its own piece and saves it next to its vLLM cache entry; it
replays the others' pieces from their ranks' cache dirs. Every rank thus ends
with the same program, a split executable whose pieces run one after another,
and saves it as its cache entry exactly as if it had compiled it alone: a warm
start loads it unchanged.

As with peer replay, the ranks coordinate through nothing but the files they
would write anyway, at a path derived from vLLM's cache key, and a piece is
only replayed if it was built from exactly the graph and inputs this rank
would compile. A piece no peer delivers in time, or delivers for another
graph, is compiled here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pickle
import re
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

logger = init_logger(__name__)

# A decoder layer's index in a node's nn_module_stack path, e.g.
# "model.layers.12.mlp", "L['self'].layers[12]" or "getattr(..layers, '12')".
_LAYER_RE = re.compile(r"(?:^|\W)layers\W+(?:_modules\W+)?(\d+)")
# A partition id handed to split_module is torch_tpu's own id times this, plus
# the piece: still ordered like torch_tpu's ids, and pieces never collide.
_PIECE_STRIDE = 1 << 16
_SUBMOD_RE = re.compile(r"submod_(\d+)$")
_POLL_S = 5.0
# Bumped whenever PieceArtifact changes, so a stale piece is refused.
_PIECE_VERSION = 1

_state = threading.local()


@dataclasses.dataclass
class Plan:
    """This rank's share of the buckets compiled inside `split_across_host`."""

    pieces: int
    piece: int
    wait_s: float


@dataclasses.dataclass
class PieceArtifact:
    """One piece of a split bucket, as its owner saves it for the others.

    `artifact` is the pickled torch_tpu executable; None means the owner
    could not save it, so a peer compiles the piece itself without waiting.
    """

    version: int = -1
    fingerprint: str | None = None
    artifact: bytes | None = None


@contextlib.contextmanager
def split_across_host(pieces: int, piece: int, wait_s: float) -> Iterator[None]:
    """Within this block, a bucket `TpuCompilerAdaptor.compile` builds is cut
    into `pieces` pieces, of which this rank compiles the `piece`-th and waits
    up to `wait_s` for each of the others."""
    _state.plan = Plan(pieces=pieces, piece=piece, wait_s=wait_s)
    try:
        yield
    finally:
        _state.plan = None


def current_plan() -> Plan | None:
    return getattr(_state, "plan", None)


# ---------------------------------------------------------------------------
# Where to cut
# ---------------------------------------------------------------------------


def node_layer(node: fx.Node) -> int | None:
    """The decoder layer `node` was traced in, from its nn_module_stack."""
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None
    for key, value in stack.items():
        path = value[0] if isinstance(value, (tuple, list)) and value else key
        match = _LAYER_RE.search(str(path))
        if match is not None:
            return int(match.group(1))
    return None


def layer_groups(graph: fx.Graph, pieces: int) -> dict[int, int]:
    """Deal the decoder layers of `graph` into at most `pieces` contiguous
    runs of about the same number of nodes: {layer: piece}.

    Empty if there are fewer than two layers to deal, e.g. an MTP head, or
    the graph carries no nn_module_stack.
    """
    counts: dict[int, int] = {}
    for node in graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        layer = node_layer(node)
        if layer is not None:
            counts[layer] = counts.get(layer, 0) + 1
    pieces = min(pieces, len(counts))
    if pieces < 2:
        return {}
    total = sum(counts.values())
    groups, done = {}, 0
    for layer in sorted(counts):
        # The piece the layer's midpoint falls in.
        groups[layer] = min(
            pieces - 1, (2 * done + counts[layer]) * pieces // (2 * total)
        )
        done += counts[layer]
    return groups


def can_split(graph: fx.GraphModule | None, pieces: int) -> bool:
    """True if `graph` has decoder layers to cut into at least two pieces."""
    if graph is None or not isinstance(graph, fx.GraphModule):
        return False
    return len(set(layer_groups(graph.graph, pieces).values())) >= 2


def _pinned_to(node: fx.Node) -> fx.Node | None:
    """The producer a wait_tensor or getitem node has to stay with.

    torch_tpu keeps a collective, its getitems and its wait_tensor in one
    partition; a getitem of anything else may share its producer's too.
    """
    import operator

    target = getattr(node.target, "overloadpacket", node.target)
    wait = getattr(torch.ops._c10d_functional, "wait_tensor", None)
    if node.op != "call_function" or not (
        target is operator.getitem or (wait is not None and target is wait)
    ):
        return None
    producer = node.args[0] if node.args else None
    while (
        isinstance(producer, fx.Node)
        and producer.op == "call_function"
        and producer.target is operator.getitem
        and producer.args
    ):
        producer = producer.args[0]
    return producer if isinstance(producer, fx.Node) else None


def _in_spmd_safe_region(node: fx.Node) -> bool:
    custom = node.meta.get("custom")
    return isinstance(custom, dict) and bool(custom.get("spmd_safe", False))


def piece_partitions(
    gm: fx.GraphModule,
    base: Callable[[fx.Node], int],
    groups: dict[int, int],
) -> dict[fx.Node, int]:
    """Partition ids for split_module: torch_tpu's `base` ids, each further
    cut where a node of the next piece's layers starts.

    The piece only moves forward along the graph, and never inside an SPMD
    safe region; a wait_tensor or getitem stays in its producer's partition.
    """
    ids: dict[fx.Node, int] = {}
    piece = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        producer = _pinned_to(node)
        if producer is not None and producer in ids and base(producer) == base(node):
            ids[node] = ids[producer]
            continue
        layer = node_layer(node)
        if layer in groups and not _in_spmd_safe_region(node):
            piece = max(piece, groups[layer])
        ids[node] = base(node) * _PIECE_STRIDE + piece
    return ids


# ---------------------------------------------------------------------------
# Compiling one's own pieces and replaying the others'
# ---------------------------------------------------------------------------


def _inputs_fingerprint(example_inputs: Any) -> str:
    parts = []
    for value in example_inputs:
        if isinstance(value, torch.Tensor):
            parts.append(
                f"T{value.dtype}{tuple(value.shape)}{tuple(value.stride())}"
                f"{value.storage_offset()}"
            )
        else:
            parts.append(f"{type(value).__qualname__}:{value!r}")
    return "|".join(parts)


def piece_fingerprint(
    gm: fx.GraphModule, example_inputs: Any, graph_fingerprint: str
) -> str:
    """What a piece was built from: the whole bucket's graph, the piece's
    code and its inputs' metadata.

    Constants are taken by dtype and shape only. Their values are pinned by
    `graph_fingerprint`, the fingerprint of the whole graph the pieces are cut
    from, which `TpuCompilerAdaptor.compile` takes before tracing -- reading a
    device tensor here, inside torch_tpu's FX trace, would fail.
    """
    h = hashlib.sha256(graph_fingerprint.encode())
    for name, module in gm.named_modules():
        if not isinstance(module, fx.GraphModule):
            continue
        h.update(f"{name}\n{module.code}".encode())
        for node in module.graph.nodes:
            if node.op != "get_attr":
                continue
            value: Any = module
            for atom in str(node.target).split("."):
                value = getattr(value, atom, None)
            if isinstance(value, torch.Tensor):
                desc = f"{value.dtype}{tuple(value.shape)}"
            else:
                desc = type(value).__qualname__
            h.update(f"{node.target}:{desc}".encode())
    h.update(_inputs_fingerprint(example_inputs).encode())
    return h.hexdigest()


@dataclasses.dataclass
class _Pending:
    index: int
    fingerprint: str
    future: Any
    call: tuple


class _Split:
    """One bucket being compiled across the host (see module docstring)."""

    def __init__(
        self,
        plan: Plan,
        key: str,
        adaptor: TpuCompilerAdaptor,
        label: str,
        graph_fingerprint: str = "",
    ):
        self.plan = plan
        self.key = key
        self.adaptor = adaptor
        self.label = label
        self.graph_fingerprint = graph_fingerprint
        # torch_tpu's eager mode outside the backend: pieces are serialized
        # and loaded in it, as the compile cache does, not inside the FX trace.
        self.outer_mode = _eager_mode()
        # The piece of each partition, in the order torch_tpu compiles them.
        # None until split_module ran; then [] if the graph could not be cut.
        self.partition_pieces: list[int | None] | None = None
        self.calls = 0
        self.pending: list[_Pending] = []
        self.compiled: list[int] = []
        self.replayed: list[int] = []

    # -- split_module --------------------------------------------------------

    def split(self, original: Callable, gm, root, split_callback, *args, **kwargs):
        groups = layer_groups(gm.graph, self.plan.pieces)
        if len(set(groups.values())) < 2:
            logger.warning(
                "[TpuCompilerAdaptor] Range %s has no decoder layers to split "
                "on; compiling it whole on this rank",
                self.label,
            )
            self.partition_pieces = []
            return original(gm, root, split_callback, *args, **kwargs)
        ids = piece_partitions(gm, split_callback, groups)
        split_gm = original(
            gm,
            root,
            lambda node: ids[node]
            if node in ids
            else split_callback(node) * _PIECE_STRIDE,
            *args,
            **kwargs,
        )
        pieces: list[int | None] = []
        for node in split_gm.graph.nodes:
            if node.op != "call_module":
                continue
            match = _SUBMOD_RE.search(str(node.target))
            pieces.append(
                int(match.group(1)) % _PIECE_STRIDE if match is not None else None
            )
        self.partition_pieces = pieces
        logger.info(
            "[TpuCompilerAdaptor] Split range %s into %d partitions over %d "
            "pieces; this rank compiles piece %d",
            self.label,
            len(pieces),
            len({p for p in pieces if p is not None}),
            self.plan.piece,
        )
        return split_gm

    # -- StaticCompiler.__call__ ---------------------------------------------

    def compile_partition(self, original: Callable, compiler, gm, inputs, args, kwargs):
        pieces = self.partition_pieces or []
        index = self.calls
        self.calls += 1
        piece = pieces[index] if index < len(pieces) else None
        fingerprint = (
            piece_fingerprint(gm, inputs, self.graph_fingerprint)
            if piece is not None
            else None
        )
        owner = None if piece is None else piece % self.plan.pieces

        if owner is None or fingerprint is None or owner == self.plan.piece:
            start = time.monotonic()
            artifact = original(compiler, gm, inputs, *args, **kwargs)
            if owner == self.plan.piece and fingerprint is not None:
                self._publish(index, len(pieces), fingerprint, artifact)
                self.compiled.append(index)
                logger.info(
                    "[TpuCompilerAdaptor] Compiled piece %d of range %s "
                    "(partition %d of %d, %d nodes) in %.1f s",
                    piece,
                    self.label,
                    index,
                    len(pieces),
                    len(gm.graph.nodes),
                    time.monotonic() - start,
                )
        else:
            import concurrent.futures

            from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
                AsyncCompiledArtifact,
            )

            future: concurrent.futures.Future = concurrent.futures.Future()
            artifact = AsyncCompiledArtifact(future)
            self.pending.append(
                _Pending(
                    index, fingerprint, future, (compiler, gm, inputs, args, kwargs)
                )
            )
        if index == len(pieces) - 1:
            self.resolve(original)
        return artifact

    @contextlib.contextmanager
    def outside_trace(self) -> Iterator[None]:
        """Run as outside the backend: torch_tpu's eager mode from there, and
        no fake mode, so device tensors can be copied to the host."""
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with contextlib.ExitStack() as stack:
            stack.enter_context(unset_fake_temporarily())
            if self.outer_mode is not None:
                from torch_tpu._internal import execution_mode

                stack.enter_context(execution_mode.set_eager_mode(self.outer_mode))
            yield

    def piece_key(self, index: int, total: int) -> str:
        return f"{self.key}.part{index}of{total}"

    def _publish(self, index: int, total: int, fingerprint: str, artifact) -> None:
        """Save a piece this rank compiled where its peers look for it."""
        assert self.adaptor.cache_dir is not None
        path = os.path.join(self.adaptor.cache_dir, self.piece_key(index, total))
        try:
            with self.outside_trace():
                data = pickle.dumps(artifact)
        except Exception as e:
            # Tell the peers not to wait for it rather than leave them waiting.
            logger.warning(
                "[TpuCompilerAdaptor] Could not serialize piece %d of range %s: %s",
                index,
                self.label,
                e,
            )
            data = None
        payload = PieceArtifact(
            version=_PIECE_VERSION, fingerprint=fingerprint, artifact=data
        )
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("[TpuCompilerAdaptor] Could not save piece %s: %s", path, e)
            with contextlib.suppress(OSError):
                os.remove(tmp)

    def _from_peer(self, pending: _Pending, total: int) -> tuple[str, Any]:
        """('ok', artifact), ('wait', None) if no peer saved it yet, or
        ('compile', None) if what a peer saved cannot be used."""
        paths = self.adaptor._peer_paths(self.piece_key(pending.index, total))
        if not paths:
            return "wait", None
        for path in paths:
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] Could not read piece %s: %s", path, e
                )
                continue
            if (
                not isinstance(payload, PieceArtifact)
                or payload.version != _PIECE_VERSION
                or payload.fingerprint != pending.fingerprint
                or payload.artifact is None
            ):
                logger.warning(
                    "[TpuCompilerAdaptor] Ignoring piece %s: not built from "
                    "this graph or not saved",
                    path,
                )
                continue
            try:
                with self.outside_trace():
                    artifact = pickle.loads(payload.artifact)
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] Could not load piece %s: %s", path, e
                )
                continue
            logger.info(
                "[TpuCompilerAdaptor] Replayed piece %d of range %s from %s",
                pending.index,
                self.label,
                path,
            )
            return "ok", artifact
        return "compile", None

    def resolve(self, original: Callable) -> None:
        """Fill in every piece left to peers: replayed once a peer saves it,
        compiled here if none does in time or what one saved does not fit."""
        total = len(self.partition_pieces or [])
        deadline = time.monotonic() + self.plan.wait_s
        left: list[_Pending] = []
        while self.pending:
            waiting = []
            for pending in self.pending:
                status, artifact = self._from_peer(pending, total)
                if status == "ok":
                    pending.future.set_result(artifact)
                    self.replayed.append(pending.index)
                elif status == "wait":
                    waiting.append(pending)
                else:
                    left.append(pending)
            self.pending = waiting
            if not self.pending or time.monotonic() >= deadline:
                break
            time.sleep(_POLL_S)
        for pending in left + self.pending:
            logger.warning(
                "[TpuCompilerAdaptor] Compiling piece %d of range %s here: no "
                "peer delivered it",
                pending.index,
                self.label,
            )
            compiler, gm, inputs, args, kwargs = pending.call
            try:
                pending.future.set_result(
                    original(compiler, gm, inputs, *args, **kwargs)
                )
            except BaseException as e:
                pending.future.set_exception(e)
                raise
            self.compiled.append(pending.index)
        self.pending = []


def _eager_mode() -> Any:
    try:
        from torch_tpu._internal import execution_mode

        return execution_mode.eager_mode
    except Exception:
        return None


def _active() -> _Split | None:
    return getattr(_state, "split", None)


_install_lock = threading.Lock()
_installed = False
# What `_install` replaced, by name.
_originals: dict[str, Callable] = {}


def _install() -> None:
    """Route torch_tpu's split_module and StaticCompiler through the active
    `_Split`, if any. Without one, both behave exactly as before."""
    global _installed
    with _install_lock:
        if _installed:
            return
        from torch_tpu._internal.compile import _backend, compiler, split_compiler

        original_make = _backend.make_backend_compiler
        original_split = split_compiler.split_module
        original_call = compiler.StaticCompiler.__call__

        def make_backend_compiler(*args, **kwargs):
            # vllm_torchtpu runs with collective materialization off, which
            # leaves torch_tpu's SplitCompiler out. Put it back for a split
            # bucket: with materialization off it cuts nowhere by itself, so
            # the only cuts are the pieces' (see `split_module` below).
            built = original_make(*args, **kwargs)
            if (
                _active() is not None
                and isinstance(built, compiler.StaticCompiler)
                and not isinstance(built, split_compiler.SplitCompiler)
            ):
                built = split_compiler.SplitCompiler(built)
            return built

        def split_module(gm, root, split_callback, *args, **kwargs):
            split = _active()
            if split is None or split.partition_pieces is not None:
                return original_split(gm, root, split_callback, *args, **kwargs)
            return split.split(
                original_split, gm, root, split_callback, *args, **kwargs
            )

        def static_call(self, graph_module, example_inputs, *args, **kwargs):
            split = _active()
            if split is None or not split.partition_pieces:
                return original_call(
                    self, graph_module, example_inputs, *args, **kwargs
                )
            return split.compile_partition(
                original_call, self, graph_module, example_inputs, args, kwargs
            )

        from torch._functorch._aot_autograd import autograd_cache

        _originals["pickle_entry"] = autograd_cache.AOTAutogradCache._pickle_entry

        def pickle_entry(*args, **kwargs):
            # A split executable keeps the constants SplitCompiler sank into
            # its pieces as device tensors, and aot_autograd pickles its cache
            # entry inside torch_tpu's FX trace, where copying a device tensor
            # to the host fails. Pickle it in the eager mode from outside.
            original = _originals["pickle_entry"]
            split = _active()
            if split is None:
                return original(*args, **kwargs)
            with split.outside_trace():
                return original(*args, **kwargs)

        autograd_cache.AOTAutogradCache._pickle_entry = staticmethod(pickle_entry)
        _backend.make_backend_compiler = make_backend_compiler
        split_compiler.split_module = split_module
        compiler.StaticCompiler.__call__ = static_call
        _installed = True


@contextlib.contextmanager
def splitting(
    plan: Plan,
    key: str,
    adaptor: TpuCompilerAdaptor,
    label: str,
    graph_fingerprint: str = "",
) -> Iterator[_Split]:
    """Compile the bucket the backend builds inside this block across the
    host, per `plan`."""
    _install()
    split = _Split(plan, key, adaptor, label, graph_fingerprint)
    _state.split = split
    start = time.monotonic()
    try:
        yield split
        if split.pending:
            # The backend compiled fewer partitions than split_module made.
            split.resolve(_original_static_call())
    finally:
        _state.split = None
    if split.partition_pieces:
        logger.info(
            "[TpuCompilerAdaptor] Range %s: compiled partitions %s here, "
            "replayed %s from peers, in %.1f s",
            label,
            split.compiled,
            split.replayed,
            time.monotonic() - start,
        )


def _original_static_call() -> Callable:
    """StaticCompiler.__call__ as torch_tpu defines it, for a late resolve."""
    from torch_tpu._internal.compile import compiler

    call = compiler.StaticCompiler.__call__

    def original(self, gm, inputs, *args, **kwargs):
        prev = getattr(_state, "split", None)
        _state.split = None
        try:
            return call(self, gm, inputs, *args, **kwargs)
        finally:
            _state.split = prev

    return original
