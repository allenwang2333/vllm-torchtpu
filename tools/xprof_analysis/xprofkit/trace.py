"""Read TPU profiler (xplane) traces into plain Python objects.

This is the only module in the kit that knows what a protobuf is. Everything
above it works on :class:`Capture` / :class:`PhaseTrace` / :class:`RankTrace`,
which are ordinary dataclasses of floats and strings.

Layout it expects
-----------------
A *capture* is one directory holding one subdirectory per inference phase, each
containing ``rank<N>_<host>.xplane.pb`` files (possibly nested under
``plugins/profile/<timestamp>/``)::

    capture/
      decode_only/
        plugins/profile/2026_08_20_19_02_11/
          rank0_t1v-n-abc.xplane.pb
          rank4_t1v-n-abc.xplane.pb
      prefill_heavy/
        ...

That is what a phased server profiler writes. A single flat directory of
``*.xplane.pb`` also works: it is read as one phase named after the directory.

Three corrections applied on the way in
---------------------------------------
These are not refinements. Skip any one of them and the numbers are wrong by
enough to reverse a decision.

1. **A step is not a module event.** The runtime compiles one engine step into
   several jitted graphs -- the decoder plus small side graphs for sampling,
   token scatter and ``as_strided``. A rank that ran 9 steps emits ~61 module
   events. Dividing by all of them shrinks every per-step figure about
   sevenfold. The step is the *dominant* graph, chosen by total duration
   because a recompile renames it.

2. **One phase directory can hold two different captures.** A rank that hits
   its iteration limit early flushes its own timestamped dump. One such
   directory held a 20-step capture averaging 78 ms a step beside a 9-step
   capture averaging 300 ms; averaging them together reported 112% occupancy,
   which is the tell. Only the most complete dump is read; the rest are
   reported as ``dropped``.

3. **Ranks are not interchangeable.** Roughly half the ranks emit a device
   plane, which half is not stable between captures, and they differ by up to
   8% in step time. A comparison over mismatched rank sets partly measures
   which ranks happened to report. Pass ``ranks=`` to pin both sides.

What an op event actually contains
----------------------------------
The event *name* on the ``XLA Ops`` line is not an op name -- it is the whole
HLO instruction::

    %multiply_convert_fusion.44 = bf16[2048,20480]{1,0:T(8,128)(2,1)} fusion(
        f8e4m3fn[2048,4096]{...} %clamp_convert_fusion.134,
        f8e4m3fn[20480,4096]{...} %param.101, ...), kind=kOutput, calls=...

The op name alone is close to useless for classification, and quietly so. That
instruction is a matmul -- fp8 activations by fp8 weights -- but XLA named the
fusion after two of its interior elementwise ops, and there is no ``dot``
anywhere in the module. A name-only taxonomy files 7% of the step under
``elementwise``. Likewise ``reduce-scatter`` carries
``async_execution_thread="sparsecore"``, which is the only way to know 36% of
op time runs on a different unit and needs a different roofline.

So :func:`read_rank` keeps a one-line signature per op alongside its duration
(:attr:`RankTrace.op_hlo`) holding the fields that survive a recompile: fusion
``kind``, ``custom_call_target``, ``async_execution_thread``, and the output
and operand shapes. Taxonomy rules can match on it. What is *not* kept is the
operand *names* (``%param.101``), which are build-local serial numbers and
would make the signature differ between two runs of the same code.

Where the durations come from
-----------------------------
Each event carries a ``device_duration_ps`` stat -- the chip's own accounting,
in picoseconds -- and also an ``event.duration_ns`` attribute. They are close
but not identical: on measured captures the two disagree by about 0.03%, which
is an order of magnitude below the capture-to-capture noise floor and so
changes no conclusion, but it does mean a number from this kit will not match
one computed off ``duration_ns`` to the last digit. The stat is preferred
because it is the device's own figure rather than a host-side timestamp
difference, and because it carries three more significant digits.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import opstats

#: Bump when a change to parsing *logic* would make an old cache entry wrong.
#: Changes to the derivation's *inputs* -- the plumbing list, the signature
#: fields -- are picked up automatically by :func:`_derivation`; this is for
#: the code those inputs feed. 5: the sidecar join resolves a name shared by
#: several HLO modules by weight rather than by file order. 6: that join now
#: refuses a name whose top two callers are tied instead of taking the
#: alphabetically first, so a cached entry written by 5 can hold a source line
#: this version would decline to guess.
CACHE_VERSION = 6

#: Lines of the device plane, and the plane itself. Properties of the profiler
#: rather than of any model, but they do move with runtime versions, and a
#: wrong value here fails silently (empty result, not an error) -- which is why
#: :func:`read_rank` raises instead of returning nothing when the plane is
#: present but the module line is not.
DEVICE_PLANE = "/device:TPU:0"
MODULE_LINE = "XLA Modules"
OP_LINE = "XLA Ops"
ASYNC_LINE = "Async XLA Ops"


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RankTrace:
    """One rank's device plane for one phase, normalised per engine step."""

    rank: str
    path: str
    #: Name of the dominant graph -- the one execution unit that is a step.
    step_module: str
    #: Wall-clock of each execution of that graph, in microseconds. One entry
    #: per step, in trace order, so a caller can see the ramp rather than only
    #: its average.
    step_us: tuple[float, ...]
    #: Device microseconds **per step** for each HLO op, keyed by op name with
    #: the operand list stripped (``%fusion.13 = f32[..] fusion(..)`` ->
    #: ``fusion.13``).
    op_us: dict[str, float] = field(default_factory=dict)
    #: One HLO signature per op in :attr:`op_us` -- see :func:`hlo_signature`.
    #: What the op *is*, as opposed to what the compiler called it.
    op_hlo: dict[str, str] = field(default_factory=dict)
    #: ``file.py:123`` for each op: the innermost stack frame that is not
    #: dispatch plumbing, from the op-stats sidecar rather than the xplane. Not
    #: the innermost frame -- for anything dispatched through Pallas that is the
    #: call wrapper, identically, for half the step. Absent for the ~26% of op
    #: time whose recorded stack is entirely dispatch plumbing -- most of it
    #: collectives, whose stack is the compilation stack and names no part of
    #: the model even when it is there. See :mod:`xprofkit.opstats`.
    op_src: dict[str, str] = field(default_factory=dict)
    #: Device microseconds per step on the ``Async XLA Ops`` line. This work
    #: overlaps the step rather than adding to it, so it is kept apart from
    #: :attr:`op_us` and never summed into :attr:`busy_us`.
    async_us: float = 0.0

    @property
    def steps(self) -> int:
        return len(self.step_us)

    @property
    def mean_us(self) -> float:
        return sum(self.step_us) / len(self.step_us) if self.step_us else 0.0

    @property
    def busy_us(self) -> float:
        """Summed op time per step.

        Against the module wall-clock this lands within a percent on every
        capture measured so far, because the op line is a partition of the
        module's own time and asynchronous work is booked separately. Treat a
        large divergence as a signal that the trace is not what it looks like
        -- see correction 2 above.
        """
        return sum(self.op_us.values())


@dataclass(frozen=True)
class PhaseTrace:
    """One inference phase of one capture, over every rank that reported it."""

    phase: str
    ranks: tuple[RankTrace, ...]
    #: Traces belonging to a less complete dump under the same phase directory.
    dropped: tuple[str, ...] = ()

    def by_rank(self) -> dict[str, RankTrace]:
        return {r.rank: r for r in self.ranks}

    @property
    def rank_ids(self) -> list[str]:
        return sorted((r.rank for r in self.ranks), key=rank_key)

    @property
    def steps(self) -> int:
        return sum(r.steps for r in self.ranks)

    #: Order of the fields :meth:`merge_attributes` merges, and the names the
    #: dispute report uses for them.
    ATTRIBUTE_FIELDS = ("hlo", "src")

    def merge_attributes(
        self,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
        """One reading per op, plus the ops whose ranks disagreed.

        Every rank runs the same program, so these are properties of the op and
        not of the rank -- but they are read per rank, and the readings can
        differ: a name several HLO modules share resolves against whichever
        module dominates *that rank's* sidecar. Classifying per rank then makes
        an op's category depend on which rank the caller happened to iterate
        last, and two commands that iterate differently disagree about the same
        capture. They did.

        So the phase gets one answer per op. Ranks are visited in rank order
        and the first non-empty reading wins; a rank that disagrees drops the
        field to empty rather than overriding it, because a disputed source
        line is not evidence of anything and a wrong one moves real time into
        the wrong category.

        A drop is not free: the op loses the field every ``hlo:`` and ``src:``
        rule matches on, so a stack where the ranks disagree widely would lose
        rule coverage with no other symptom than categories quietly shrinking.
        The dropped ops are returned for that reason, not for the arithmetic.
        """
        out: dict[str, tuple[str, str]] = {}
        disputed: dict[str, list[str]] = {f: [] for f in self.ATTRIBUTE_FIELDS}
        for rank in self.ranks:
            for name in rank.op_us:
                found = (rank.op_hlo.get(name, ""), rank.op_src.get(name, ""))
                if name not in out:
                    out[name] = found
                    continue
                agreed = list(out[name])
                for i, value in enumerate(found):
                    # Not `field`: that is `dataclasses.field`, imported above
                    # and used by every default in this module.
                    kind = self.ATTRIBUTE_FIELDS[i]
                    if value and agreed[i] and value != agreed[i]:
                        disputed[kind].append(name)
                        agreed[i] = ""
                    elif value and not agreed[i] and name not in disputed[kind]:
                        agreed[i] = value
                out[name] = (agreed[0], agreed[1])
        return out, {f: sorted(set(names)) for f, names in disputed.items()}

    def op_attributes(self) -> dict[str, tuple[str, str]]:
        """``{op name: (hlo signature, source line)}`` for the phase as a whole."""
        return self.merge_attributes()[0]

    def mean_over_ranks(self, field_name: str = "mean_us") -> float:
        """Average of a per-rank scalar across the ranks present.

        Per-core, not per-host: under data parallelism every core runs the same
        program on different sequences, so a host total would just be N times
        larger and would move if the sharding did.
        """
        values = [getattr(r, field_name) for r in self.ranks]
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class Capture:
    """One profiler output directory: every phase, every rank."""

    label: str
    root: str
    phases: dict[str, PhaseTrace]

    def phase(self, name: str) -> PhaseTrace | None:
        return self.phases.get(name)

    def __bool__(self) -> bool:
        return bool(self.phases)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def rank_key(rank: str) -> tuple[int, str]:
    """Sort ``2`` before ``10`` while keeping non-numeric names orderable."""
    return (len(rank), rank) if rank.isdigit() else (99, rank)


def rank_of(basename: str) -> str:
    """``rank4_t1v-....xplane.pb`` -> ``4``; anything unexpected keeps its name."""
    match = re.match(r"rank(\d+)_", basename)
    return match.group(1) if match else basename


def phase_dirs(root: str | os.PathLike) -> list[str]:
    """Subdirectories of *root* that contain traces, or ``["."]`` if flat."""
    root = Path(root)
    named = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and any(p.glob("**/*.xplane.pb"))
    )
    if named:
        return named
    return ["."] if any(root.glob("**/*.xplane.pb")) else []


def select_dump(phase_dir: str | os.PathLike) -> tuple[list[str], list[str]]:
    """``(traces of the most complete dump, traces dropped from lesser ones)``.

    "Most complete" is the timestamped directory with the most ranks reporting,
    ties broken by name so the choice is deterministic across runs.
    """
    traces = sorted(
        glob.glob(os.path.join(str(phase_dir), "**", "*.xplane.pb"), recursive=True)
    )
    dumps: dict[str, list[str]] = {}
    for path in traces:
        dumps.setdefault(os.path.dirname(path), []).append(path)
    if len(dumps) <= 1:
        return traces, []
    best = max(dumps, key=lambda d: (len(dumps[d]), d))
    return dumps[best], sorted(p for d, ps in dumps.items() if d != best for p in ps)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _duration_us(event) -> float:
    """One event's device time in microseconds."""
    for name, value in event.stats:
        if name == "device_duration_ps":
            return float(value) / 1e6
    duration_ns = getattr(event, "duration_ns", None)
    if duration_ns is not None:
        return float(duration_ns) / 1e3
    raise RuntimeError(f"no duration on profiler event {event.name!r}")


def _op_name(raw: str) -> str:
    """``%fusion.13 = f32[8]{:T(128)} fusion(...)`` -> ``fusion.13``."""
    return raw.split(" = ", 1)[0].lstrip("%").strip()


#: Matches ``bf16[2048,4096]`` but not the ``{1,0:T(8,128)}`` layout after it.
_SHAPE = re.compile(r"\b([a-z]\w*)\[([\d,]*)\]")
#: How many operand shapes go into a signature. Enough to tell a matmul's two
#: sides apart; not so many that a 30-operand custom call fills the line.
_MAX_OPERANDS = 4
#: The non-shape fields a signature carries, and how each is found. A named
#: constant because it is part of the cache key: adding a field here -- or
#: editing one of these patterns, which is the commoner edit -- changes every
#: stored signature, and a warm capture must not keep serving the old ones
#: while `hlo:` rules are written against the new. Both the key and the pattern
#: text are hashed; see :func:`_derivation`.
_SIGNATURE_FIELDS = (
    ("kind", r"kind=(\w+)"),
    ("target", r'custom_call_target="([^"]*)"'),
    ("thread", r'async_execution_thread="([^"]*)"'),
    # Both spellings XLA prints: the explicit `{{0,1},{2,3}}` list and the
    # iota-and-transpose shorthand `[2,4]<=[4,2]T(1,0)`.
    ("replica_groups", r"replica_groups=(\{\{.*?\}\}|\S+?)(?:,\s|$)"),
)

#: Devices named in a replica-group summary. Membership is what tells one
#: collective's participant set from another's, but the full list for a large
#: mesh is neither readable in a table nor writable in a regex, so the summary
#: names one group and only this far into it.
_MAX_GROUP_MEMBERS = 8

#: Fields whose raw text is summarised before it goes into the signature.
_SIGNATURE_RENDER = {}


def _replica_groups(text: str) -> str:
    """``{{0,1,4,5},{2,3,6,7}}`` -> ``2x4[0,1,4,5]`` -- shape, then one group.

    The participant set is the only thing that tells a tensor-parallel
    collective from an expert-parallel one. They share a name (both are
    ``all-reduce.N``, so :func:`~xprofkit.taxonomy.family` folds them into one
    row), they share a rough shape, and being compiler-generated they carry no
    source line at all -- so name, ``src`` and instance count all fail on them
    at once, and a sharding that collects in two places has its collectives
    booked wholesale to one block. This field is what a ``hlo:`` rule can then
    split them on, from YAML, with no change here.

    Rendered rather than copied. A verbatim group list for a large mesh runs to
    hundreds of characters, which is unusable in both places the signature is
    read -- a table cell and a regex -- while the group count, the group size
    and one group's membership carry what distinguishes the shardings: the
    count and size give the axis's extent, and the membership gives the stride,
    which is what separates two collectives of identical extent over different
    axes. An unrecognised spelling is passed through rather than dropped.
    """
    groups = re.findall(r"\{([^{}]*)\}", text)
    if not groups:
        return text
    members = [group.split(",") if group else [] for group in groups]
    sizes = {len(m) for m in members}
    shape = f"{len(members)}x{sizes.pop() if len(sizes) == 1 else '?'}"
    head = members[0][:_MAX_GROUP_MEMBERS]
    return (f"{shape}[{','.join(head)}"
            f"{'+' if len(members[0]) > _MAX_GROUP_MEMBERS else ''}]")


_SIGNATURE_RENDER["replica_groups"] = _replica_groups


def hlo_signature(raw: str) -> str:
    """A stable one-line summary of what an HLO instruction *is*.

    Keeps only fields that survive a recompile -- shapes, fusion kind, custom
    call target, execution thread, replica groups -- and drops operand names,
    which are build-local serial numbers::

        kind=kOutput out=bf16[2048,17408] in=f8e4m3fn[2048,4096],f8e4m3fn[17408,4096]
        replica_groups=1x8[0,1,4,5,6,7,2,3] out=bf16[16384,512] in=bf16[2048,512]

    Taxonomy rules regex against this, so an op whose *name* says
    ``multiply_convert_fusion`` can still be recognised as the attention QKV
    projection it is -- and two collectives the compiler gave the same name can
    be told apart by who participates in them, which is the only handle a
    collective has (see :func:`_replica_groups`).

    ``channel_id`` and ``dimensions`` are deliberately *not* here.
    ``channel_id`` is a compile-local serial, which is the one class of field
    this summary exists to drop; ``dimensions`` fires on 3,843 non-collective
    instructions in the capture to hand -- every reduce, iota and transpose --
    and would rewrite every signature in the cache to disambiguate collectives
    that their shapes already separate.
    """
    body = raw.split(" = ", 1)[1] if " = " in raw else raw
    shapes = [f"{dtype}[{dims}]" for dtype, dims in _SHAPE.findall(body)]
    parts = []
    for key, pattern in _SIGNATURE_FIELDS:
        found = re.search(pattern, body)
        if found:
            render = _SIGNATURE_RENDER.get(key, lambda text: text)
            parts.append(f"{key}={render(found.group(1))}")
    if shapes:
        parts.append(f"out={shapes[0]}")
        if len(shapes) > 1:
            parts.append("in=" + ",".join(shapes[1:1 + _MAX_OPERANDS]))
    return " ".join(parts)


def _cache_file(path: str) -> Path:
    return Path(path).parent / ".xprofkit" / (Path(path).name + ".json")


def _derivation() -> str:
    """Identity of the configuration the cached columns were derived *from*.

    A cache entry is not just a copy of the trace: ``op_hlo`` and ``op_src``
    are derived by code in this kit, and that code is meant to be edited --
    :data:`~xprofkit.opstats.PLUMBING` is documented as the thing to extend for
    a new stack. Keyed on the files alone, a warm capture answers with the old
    derivation and nothing on screen says so; the failure mode is an hour spent
    on a ``src:`` rule that was never re-evaluated.

    So the *inputs* to that derivation are hashed in: the plumbing and trim
    lists, whether the sidecar reader was importable at all, and the signature
    configuration. Changing the derivation *logic* still needs
    ``CACHE_VERSION``.

    The **regex text** goes in, not just the field name, and that distinction
    was a live bug: this hashed ``[key for key, _ in _SIGNATURE_FIELDS]``, so
    rewriting every pattern in the tuple left the digest unmoved and a warm
    capture kept serving pre-edit signatures. Tightening a pattern is the
    ordinary way to work on this file -- ``replica_groups`` needed three
    attempts -- and it is exactly the edit that produced no invalidation.
    ``_MAX_GROUP_MEMBERS`` and the ``_SIGNATURE_RENDER`` key set are here for
    the same reason: both rewrite stored signatures without touching a name.
    Only the *bodies* of the renderers are still uncovered, and a hash cannot
    see those -- that is what ``CACHE_VERSION`` is for.
    """
    payload = json.dumps({
        "signature": [[key, pattern] for key, pattern in _SIGNATURE_FIELDS],
        "rendered": sorted(_SIGNATURE_RENDER),
        "limits": [_MAX_OPERANDS, _MAX_GROUP_MEMBERS],
        "source": opstats.source_fingerprint(),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _cache_key(path: str) -> dict:
    """What has to be unchanged for a cache entry to still be true.

    The sidecar is in the key as well as the xplane. A capture parsed before
    its ``op_stats_v2.pb`` was written caches an empty ``op_src``, and keying on
    the xplane alone would keep serving that empty column forever -- silently,
    since a missing source line is indistinguishable from an op that has none.
    The same argument covers the derivation: see :func:`_derivation`.
    """
    stat = os.stat(path)
    sidecar = opstats.op_stats_path(path)
    side_stat = os.stat(sidecar) if sidecar else None
    return {
        "version": CACHE_VERSION,
        "derivation": _derivation(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "src_size": side_stat.st_size if side_stat else None,
        "src_mtime_ns": side_stat.st_mtime_ns if side_stat else None,
    }


def _cache_load(path: str) -> dict | None:
    cache = _cache_file(path)
    if not cache.is_file():
        return None
    try:
        payload = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None
    key = _cache_key(path)
    return payload if all(payload.get(k) == v for k, v in key.items()) else None


def _cache_store(path: str, payload: dict) -> None:
    """Best effort. A read-only capture directory is not an error."""
    payload = {**payload, **_cache_key(path)}
    cache = _cache_file(path)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, cache)
    except OSError:
        pass


def read_rank(path: str, *, use_cache: bool = True) -> RankTrace | None:
    """Parse one ``*.xplane.pb``. ``None`` if it holds no device plane.

    Parsing a 28 MB protobuf takes a few seconds and dominates every command in
    this kit, so the result is cached beside the trace under ``.xprofkit/``,
    keyed on size and mtime. Deleting that directory is always safe.
    """
    rank = rank_of(os.path.basename(path))
    if use_cache:
        cached = _cache_load(path)
        if cached is not None:
            return RankTrace(
                rank=rank, path=path,
                step_module=cached["step_module"],
                step_us=tuple(cached["step_us"]),
                op_us=cached["op_us"],
                op_hlo=cached.get("op_hlo", {}),
                op_src=cached.get("op_src", {}),
                async_us=cached["async_us"],
            )

    import jax  # deferred: importing jax costs seconds and may probe a chip

    plane = jax.profiler.ProfileData.from_file(path).find_plane_with_name(DEVICE_PLANE)
    if plane is None:
        return None

    # Collect every module's per-execution durations, then pick the dominant
    # one. Selecting by *total* time rather than by name survives the rename a
    # recompile brings, and rather than by count because the side graphs are
    # frequent and tiny.
    per_module: dict[str, list[float]] = {}
    op_total: dict[str, float] = {}
    op_hlo: dict[str, str] = {}
    async_total = 0.0
    for line in plane.lines:
        if line.name == MODULE_LINE:
            for event in line.events:
                per_module.setdefault(event.name, []).append(_duration_us(event))
        elif line.name == OP_LINE:
            for event in line.events:
                name = _op_name(event.name)
                op_total[name] = op_total.get(name, 0.0) + _duration_us(event)
                # Every execution of one instruction has the same signature,
                # so the first is enough and re-parsing the rest is waste.
                if name not in op_hlo:
                    op_hlo[name] = hlo_signature(event.name)
        elif line.name == ASYNC_LINE:
            for event in line.events:
                async_total += _duration_us(event)
    if not per_module:
        return None

    # Source lines live in the op-stats sidecar the profiler writes beside
    # the xplane, keyed by the same op names. Optional: a capture without one
    # loses the source column and nothing else. `read_callers`, not
    # `read_sources` -- see `opstats.PLUMBING` for why the innermost frame is
    # the wrong one to report.
    sidecar = opstats.op_stats_path(path)
    op_src = opstats.read_callers(sidecar) if sidecar else {}
    op_src = {name: loc for name, loc in op_src.items() if name in op_total}

    module = max(per_module, key=lambda name: sum(per_module[name]))
    steps = len(per_module[module])
    trace = RankTrace(
        rank=rank,
        path=path,
        step_module=module,
        step_us=tuple(per_module[module]),
        op_us={name: us / steps for name, us in op_total.items()},
        op_hlo=op_hlo,
        op_src=op_src,
        async_us=async_total / steps,
    )
    if use_cache:
        _cache_store(path, {
            "step_module": trace.step_module,
            "step_us": list(trace.step_us),
            "op_us": trace.op_us,
            "op_hlo": trace.op_hlo,
            "op_src": trace.op_src,
            "async_us": trace.async_us,
        })
    return trace


def read_phase(
    phase_dir: str | os.PathLike,
    ranks: set[str] | None = None,
    *,
    use_cache: bool = True,
) -> PhaseTrace | None:
    """Every rank of one phase directory. ``None`` if nothing was readable."""
    traces, dropped = select_dump(phase_dir)
    if ranks is not None:
        traces = [p for p in traces if rank_of(os.path.basename(p)) in ranks]
    read = [read_rank(p, use_cache=use_cache) for p in traces]
    kept = tuple(r for r in read if r is not None)
    if not kept:
        return None
    return PhaseTrace(
        phase=os.path.basename(str(phase_dir).rstrip("/")) or ".",
        ranks=tuple(sorted(kept, key=lambda r: rank_key(r.rank))),
        dropped=tuple(
            f"{os.path.basename(p)} @ {os.path.basename(os.path.dirname(p))}"
            for p in dropped
        ),
    )


def load_capture(
    root: str | os.PathLike,
    ranks: set[str] | None = None,
    *,
    label: str | None = None,
    phases: set[str] | None = None,
    use_cache: bool = True,
) -> Capture:
    """Read a whole capture directory.

    ``ranks`` pins the rank set -- always pass it when the result will be
    compared against another capture. ``phases`` restricts which phases are
    parsed, which is worth doing when only one is of interest: each phase costs
    a few seconds per rank on a cold cache.
    """
    root = str(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(root)
    found: dict[str, PhaseTrace] = {}
    for name in phase_dirs(root):
        if phases is not None and name not in phases:
            continue
        data = read_phase(
            root if name == "." else os.path.join(root, name),
            ranks, use_cache=use_cache,
        )
        if data is not None:
            found[name if name != "." else os.path.basename(os.path.normpath(root))] = data
    return Capture(
        label=label or os.path.basename(os.path.normpath(root)),
        root=os.path.abspath(root),
        phases=found,
    )


def shared_ranks(traces: list[PhaseTrace]) -> list[str]:
    """Ranks present in every one of *traces*, in stable order."""
    if not traces:
        return []
    common = set.intersection(*(set(t.by_rank()) for t in traces))
    return sorted(common, key=rank_key)
