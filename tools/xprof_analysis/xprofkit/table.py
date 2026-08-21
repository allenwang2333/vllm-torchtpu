"""Fixed-width markdown tables, and the small statistics the reports need.

Output is plain markdown so it pastes into a bug, a doc or a terminal
unchanged. Columns are padded to a constant width rather than to their
contents, so two tables printed one after the other line up.
"""

from __future__ import annotations

import math
import statistics as stats
from collections.abc import Iterable, Sequence


def table(headers: Sequence[str], rows: Iterable[Sequence[object]],
          widths: Sequence[int] | None = None,
          align: str | None = None) -> list[str]:
    """A markdown table as a list of lines.

    ``align`` is one character per column: ``<`` left, ``>`` right. Defaults to
    left for the first column and right for the rest, which is what a table of
    "name, then numbers" wants.
    """
    rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    n = len(headers)
    align = align or ("<" + ">" * (n - 1))
    # A short `align` is a caller that grew a column; pad rather than raise on
    # the last cell of every row.
    align += ">" * (n - len(align))
    if widths is None:
        widths = [
            max(len(str(headers[i])), *(len(r[i]) for r in rows)) if rows
            else len(str(headers[i]))
            for i in range(n)
        ]

    def line(cells: Sequence[str]) -> str:
        # A cell wider than its column widens the row rather than being cut:
        # losing the end of an op name silently is worse than a ragged edge,
        # and callers that can afford to truncate do it themselves.
        return "| " + " | ".join(
            f"{cell:{align[i]}{widths[i]}}" for i, cell in enumerate(cells)
        ) + " |"

    out = [line([str(h) for h in headers])]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    out.extend(line(r) for r in rows)
    return out


def us(value: float | None, places: int = 1) -> str:
    """Microseconds, thousands-separated. Empty for ``None``."""
    return "" if value is None else f"{value:,.{places}f}"


def pct(value: float | None, places: int = 2, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:{'+' if signed else ''}.{places}f}%"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics -- numpy's default rule.

    Truncating the index instead is wrong at the sample sizes this kit sees:
    at n=9, ``int(0.90 * 8)`` is 7, so "p90" was the second largest of nine
    (the 87.5th percentile), and at n<10 "p10" was always the minimum. Reports
    that pair a p90 with a max then showed two different names for what a
    reader takes to be two different numbers.
    """
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def describe(samples: Sequence[float]) -> dict:
    """Location and spread of a sample, with the percentiles reports use."""
    n = len(samples)
    if not n:
        return {"n": 0}
    ordered = sorted(samples)
    mean = stats.mean(samples)
    sd = stats.stdev(samples) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": stats.median(samples),
        "sd": sd,
        "cv_pct": 100.0 * sd / mean if mean else float("nan"),
        "sem": sd / math.sqrt(n) if n > 1 else 0.0,
        "sem_pct": 100.0 * (sd / math.sqrt(n)) / mean if n > 1 and mean else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p10": _percentile(ordered, 0.10),
        "p90": _percentile(ordered, 0.90),
    }


#: Two-sided t critical values at 95%, indexed by degrees of freedom. A lookup
#: rather than a scipy dependency: this kit runs inside a serving venv and is
#: not worth adding a package to. The table runs to 120 df because the fallback
#: below is the normal limit, and at 31 df -- the first df past the old end of
#: the table -- the true value is 2.0395, 4% above 1.96.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t95(df: float) -> float:
    """Two-sided 95% t critical value, rounded up between tabled rows.

    Welch's df is fractional almost always -- three captures against three
    gives values like 2.32 or 3.86 -- so which way the gap is closed is a real
    decision, and it is not symmetric. The critical value falls as df rises, so
    the conservative reading is the *largest tabled df at or below* the real
    one: fewer degrees of freedom, a larger critical value, a wider interval.

    Getting this backwards is not a rounding detail. At df 2.32 the true value
    is ~3.95; taking the df=3 row gives 3.182, an interval ~20% too narrow, and
    a promotion gate then fires on a comparison that does not clear it. Erring
    wide only ever costs another capture.
    """
    if df <= 0:
        return float("inf")
    for key in sorted(_T95, reverse=True):
        if key <= df:
            return _T95[key]
    # Below one degree of freedom there is no row to round down to; the widest
    # one is the honest answer.
    return _T95[min(_T95)]


def welch(a: Sequence[float], b: Sequence[float]) -> dict:
    """Compare two independent samples: *a* against baseline *b*.

    Returns the difference as a percentage of *b*'s mean, its standard error,
    a Welch t statistic and a 95% confidence interval. Welch rather than
    Student because the two arms have no reason to share a variance and often
    do not -- one candidate's captures were three times tighter than the
    baseline's.

    ``t`` is ``None`` and the interval is infinite when the data cannot support
    one. A single capture per arm has no within-arm variance to estimate, so
    the arithmetic gives ``se == 0`` and, taken literally, a zero-width
    interval: two one-shot captures 1.31% apart printed ``CI [+1.31%, +1.31%]``
    and ``t inf``, which reads as certainty and is the exact opposite of what
    n=1 supports. Unbounded is the honest interval -- a run-to-run spread of
    ~1% is normal on this stack, and nothing in a single pair of captures
    measures it. Callers should treat ``t is None`` as "not comparable", not as
    "no difference".
    """
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        return {}
    ma, mb = stats.mean(a), stats.mean(b)
    va = stats.variance(a) if na > 1 else 0.0
    vb = stats.variance(b) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb)
    delta = ma - mb
    out = {
        "mean_a": ma, "mean_b": mb, "n_a": na, "n_b": nb,
        "delta": delta,
        "delta_pct": 100.0 * delta / mb if mb else float("nan"),
        "se_pct": 100.0 * se / mb if mb else float("nan"),
    }
    # `se == 0` with n>1 on both sides is the degenerate case of identical
    # repeats -- three captures reporting the same microsecond. That is not
    # evidence of zero spread either; it is a sample too coarse to show any.
    if min(na, nb) < 2 or se == 0:
        out.update({"t": None, "df": 0.0,
                    "ci_lo_pct": float("-inf"), "ci_hi_pct": float("inf")})
        return out
    denom = ((va / na) ** 2 / max(na - 1, 1)) + ((vb / nb) ** 2 / max(nb - 1, 1))
    df = (va / na + vb / nb) ** 2 / denom if denom else 0.0
    crit = t95(df)
    out.update({
        "t": delta / se,
        "df": df,
        "ci_lo_pct": 100.0 * (delta - crit * se) / mb if mb else float("nan"),
        "ci_hi_pct": 100.0 * (delta + crit * se) / mb if mb else float("nan"),
    })
    return out
