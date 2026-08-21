"""xprofkit — read TPU serving profiles and say where the time went.

Importable as a library as well as driven from ``analyze.py``::

    from xprofkit import load_capture, Taxonomy, breakdown

    cap = load_capture("runs/prof/baseline", ranks={"0", "2", "4", "6"})
    result = breakdown.analyse(cap, Taxonomy.load())
    print(result["decode_only"]["categories"]["moe"])

Everything is plain dataclasses and dicts; nothing holds a file handle or a
chip. See the README for the command line and for how to extend the taxonomy.
"""

from . import breakdown, compare, diff, steptime, table
from .taxonomy import Taxonomy, family
from .trace import (
    Capture,
    PhaseTrace,
    RankTrace,
    load_capture,
    phase_dirs,
    read_phase,
    read_rank,
    select_dump,
    shared_ranks,
)

__all__ = [
    "Capture", "PhaseTrace", "RankTrace", "Taxonomy",
    "breakdown", "compare", "diff", "steptime", "table",
    "family", "load_capture", "phase_dirs", "read_phase", "read_rank",
    "select_dump", "shared_ranks",
]
