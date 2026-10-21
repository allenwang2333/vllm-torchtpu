# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-server diagnostics of the DeepSeek-V4 megakernel's served outputs.

Both are off unless asked for and leave the kernel program untouched; each
runs its own small compiled graph after the step.

`NonfiniteStats` (`VLLM_TPU_DSV4_MK_NONFINITE_STATS`): every busy megakernel
step counts the NaN, +-inf and +-dtype-max entries of the hidden rows the
runner reads (real rows) and of the bucket's padding rows, plus non-finite
aux rows, into a device accumulator. Every rank logs cumulative totals per
phase (pre-serving warm-up vs served) each stats period.

`ParityProbe` (`VLLM_TPU_DSV4_MK_PARITY*`): on selected megakernel steps the
adapter runs the stock forward first and then the megakernel step on the
same inputs (see `DSv4Megakernel.forward`), and the probe compares the two
outputs: per real row the relative L2 / max abs difference of the hidden
and aux states and, on this rank's vocabulary shard of the lm_head, each
path's top-1 logit and token. Shards are not exchanged (a collective here
would have to be paired by idle DP engines); the global top-1 of a row is
the best of its TP group's per-shard lines.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Callable, Sequence

import torch

PHASES = ("pre-serving", "served")

# NonfiniteStats counter slots
_NF_FIELDS = (
    "nan_real", "inf_real", "sat_real", "bad_rows_real",
    "nan_pad", "inf_pad", "sat_pad", "bad_rows_pad",
    "aux_bad_rows_real", "aux_bad_rows_pad", "steps_bad_real",
)  # fmt: skip
# Drain the int32 device counters before they can overflow: <= 1024 steps of
# rows * hidden entries each (28 * 7168 for the served buckets).
_NF_DRAIN_STEPS = 1024


def _emit(rank: int, msg: str) -> None:
    # stderr: Ray drops worker logger.info lines while serving
    sys.stderr.write(f"[dsv4-megakernel rank {rank}] {msg}\n")
    sys.stderr.flush()


def nonfinite_counts(
    hidden: torch.Tensor,
    aux: Sequence[torch.Tensor],
    request_distribution: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """int32 [len(_NF_FIELDS)]: rows `< request_distribution[2] * q` are real."""
    rows = hidden.shape[0]
    real = torch.arange(rows, device=hidden.device) < request_distribution[2] * q
    pad = ~real
    x = hidden.float()
    nan = torch.isnan(x)
    inf = torch.isinf(x)
    sat = x.abs() == torch.finfo(hidden.dtype).max
    bad_rows = (~torch.isfinite(x)).any(dim=1)
    aux_bad = torch.zeros(rows, dtype=torch.bool, device=hidden.device)
    for a in aux:
        aux_bad = aux_bad | (~torch.isfinite(a.float())).any(dim=1)

    def n(mask):
        return mask.sum(dtype=torch.int32)

    r2, p2 = real[:, None], pad[:, None]
    bad_real = n(bad_rows & real)
    return torch.stack(
        [
            n(nan & r2), n(inf & r2), n(sat & r2), bad_real,
            n(nan & p2), n(inf & p2), n(sat & p2), n(bad_rows & pad),
            n(aux_bad & real), n(aux_bad & pad), (bad_real > 0).to(torch.int32),
        ]
    )  # fmt: skip


class NonfiniteStats:
    def __init__(
        self,
        rank: int,
        q: int,
        device: torch.device,
        period_s: float,
        compile_fn: Callable[[Callable], Callable],
    ) -> None:
        self.rank, self.q, self.period_s = rank, q, period_s
        self.device = device

        def update(acc, hidden, aux, request_distribution):
            return acc + nonfinite_counts(hidden, aux, request_distribution, q)

        self._update = compile_fn(update)
        self._acc = [self._zeros(), self._zeros()]  # per phase, on device
        self._undrained = [0, 0]
        self.totals = [dict.fromkeys(_NF_FIELDS, 0) for _ in PHASES]
        self.busy = [0, 0]
        self.idle = [0, 0]
        self.real_rows = [0, 0]
        self._t0 = time.monotonic()

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(len(_NF_FIELDS), dtype=torch.int32, device=self.device)

    def warm(self, hidden, aux, request_distribution) -> None:
        """Compile the counter for this output shape (startup, all ranks)."""
        self._update(self._zeros(), hidden, list(aux), request_distribution)

    def count(self, phase, hidden, aux, request_distribution, real_rows) -> None:
        self._acc[phase] = self._update(
            self._acc[phase], hidden, list(aux), request_distribution
        )
        self.busy[phase] += 1
        self.real_rows[phase] += real_rows
        self._undrained[phase] += 1
        if self._undrained[phase] >= _NF_DRAIN_STEPS:
            self._drain(phase)

    def count_idle(self, phase: int) -> None:
        self.idle[phase] += 1

    def _drain(self, phase: int) -> None:
        got = self._acc[phase].tolist()  # a device sync
        for k, v in zip(_NF_FIELDS, got):
            self.totals[phase][k] += int(v)
        self._acc[phase] = self._zeros()
        self._undrained[phase] = 0

    def line(self, phase: int) -> str:
        if self._undrained[phase]:
            self._drain(phase)
        t = self.totals[phase]
        return (
            f"nonfinite {PHASES[phase]}: steps busy {self.busy[phase]} idle "
            f"{self.idle[phase]}, steps_with_nonfinite_real {t['steps_bad_real']}; "
            f"real rows {self.real_rows[phase]}: nan {t['nan_real']} inf "
            f"{t['inf_real']} sat {t['sat_real']} bad_rows {t['bad_rows_real']}; "
            f"pad rows: nan {t['nan_pad']} inf {t['inf_pad']} sat {t['sat_pad']} "
            f"bad_rows {t['bad_rows_pad']}; aux bad_rows real "
            f"{t['aux_bad_rows_real']} pad {t['aux_bad_rows_pad']}"
        )

    def maybe_log(self, phase: int) -> None:
        now = time.monotonic()
        if now - self._t0 < self.period_s:
            return
        self._t0 = now
        if self.busy[phase] or self.idle[phase]:
            _emit(self.rank, self.line(phase))

    def log(self, phase: int) -> None:
        _emit(self.rank, self.line(phase))

    def self_check(self) -> None:
        """Log what this device's max-reduction returns over NaN entries: the
        value a `float(x.abs().max())` trace prints for an all-NaN tensor."""
        nan = torch.full((7, 8), float("nan"), dtype=torch.bfloat16)
        mixed = nan.clone()
        mixed[1] = 0
        got = [float(t.to(self.device).float().abs().max()) for t in (nan, mixed)]
        _emit(
            self.rank,
            f"nonfinite self-check: abs().max() over all-NaN = {got[0]:.4g}, "
            f"over NaN with a zero row = {got[1]:.4g}",
        )


@dataclasses.dataclass(frozen=True)
class LocalHead:
    """This rank's vocabulary shard of the lm_head."""

    weight: torch.Tensor  # [rows, hidden]; rows past `num_tokens` are padding
    vocab_start: int  # global id of row 0
    num_tokens: int


@dataclasses.dataclass(frozen=True)
class ParityConfig:
    head: LocalHead
    served_budget: int  # parity calls on served steps
    every: int  # on every `every`-th eligible served step
    warmup_budget: int  # parity calls on pre-serving (warm-up) steps
    tol: float  # rows above this relative L2 are counted as over_tol
    # the two forwards compared, in run order, cycled over the parity calls;
    # ("stock", "mk") is the check, ("mk", "mk") and ("stock", "stock") the
    # method's floor
    pairs: tuple[tuple[str, str], ...] = (("stock", "mk"),)

    def __post_init__(self) -> None:
        for p in self.pairs:
            if len(p) != 2 or not set(p) <= {"stock", "mk"}:
                raise ValueError(f"parity pair must be two of stock / mk, got {p}")
        if not self.pairs:
            raise ValueError("no parity pair")


# ParityProbe per-row float fields; the aux rel-L2 rows follow
_PF_FIELDS = ("rel_l2", "max_abs_diff", "max_abs_stock",
              "s_max", "m_max", "s_at_m", "m_at_s")  # fmt: skip
_PI_FIELDS = ("s_tok", "m_tok", "bad_stock", "bad_mk")


def _row_rel_l2(ref: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    ok = torch.isfinite(ref) & torch.isfinite(new)
    d = torch.where(ok, new - ref, 0.0)
    r = torch.where(ok, ref, 0.0)
    den = torch.sqrt((r * r).sum(dim=1)).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.sqrt((d * d).sum(dim=1)) / den


def parity_rows(
    h_stock: torch.Tensor,
    aux_stock: Sequence[torch.Tensor],
    h_mk: torch.Tensor,
    aux_mk: Sequence[torch.Tensor],
    head_weight: torch.Tensor,
    vocab_start: int,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: float32 [len(_PF_FIELDS) + n_aux, rows], int32
    [len(_PI_FIELDS), rows]. Non-finite entries are left out of the
    differences and counted in bad_*."""
    s, m = h_stock.float(), h_mk.float()
    fin_s, fin_m = torch.isfinite(s), torch.isfinite(m)
    both = fin_s & fin_m
    diff = torch.where(both, (m - s).abs(), 0.0).amax(dim=1)
    ref = torch.where(fin_s, s.abs(), 0.0).amax(dim=1)
    w = head_weight[:num_tokens].float()
    ls, lm = s @ w.T, m @ w.T  # f32 logits of this shard
    s_max, s_arg = ls.max(dim=1)
    m_max, m_arg = lm.max(dim=1)
    s_at_m = ls.gather(1, m_arg[:, None])[:, 0]
    m_at_s = lm.gather(1, s_arg[:, None])[:, 0]
    floats = [_row_rel_l2(s, m), diff, ref, s_max, m_max, s_at_m, m_at_s]
    floats += [_row_rel_l2(a.float(), b.float()) for a, b in zip(aux_stock, aux_mk)]
    ints = [
        s_arg.to(torch.int32) + vocab_start,
        m_arg.to(torch.int32) + vocab_start,
        (~fin_s).sum(dim=1, dtype=torch.int32),
        (~fin_m).sum(dim=1, dtype=torch.int32),
    ]
    return torch.stack(floats), torch.stack(ints)


class ParityProbe:
    """Picks the parity steps (identically on every rank: the inputs are the
    phase and the sequence of megakernel decisions, both rank-invariant) and
    compares their two outputs."""

    def __init__(
        self,
        rank: int,
        cfg: ParityConfig,
        compile_fn: Callable[[Callable], Callable],
    ) -> None:
        if cfg.every < 1:
            raise ValueError(f"parity stride must be >= 1, got {cfg.every}")
        head = cfg.head
        self.rank, self.head, self.tol = rank, head, cfg.tol
        self.pairs = cfg.pairs
        self.pair = cfg.pairs[0]  # of the selected step
        self._selected = 0
        self.budget = (cfg.warmup_budget, cfg.served_budget)
        self.every = (1, cfg.every)
        self.eligible = [0, 0]
        self.calls = [0, 0]
        self.agg = {}  # (phase, pair) -> totals

        def rows(h_s, aux_s, h_m, aux_m, w):
            return parity_rows(h_s, aux_s, h_m, aux_m, w, head.vocab_start,
                               head.num_tokens)  # fmt: skip

        self._rows = compile_fn(rows)

    @staticmethod
    def _new_agg() -> dict:
        return dict(calls=0, compared=0, rows=0, max_rel_l2=0.0, max_abs_diff=0.0,
                    over_tol_rows=0, bad_rows_stock=0, bad_rows_mk=0,
                    shard_top1_agree=0)  # fmt: skip

    def select(self, phase: int) -> bool:
        k = self.eligible[phase]
        self.eligible[phase] += 1
        if self.calls[phase] >= self.budget[phase] or k % self.every[phase]:
            return False
        self.calls[phase] += 1
        self.pair = self.pairs[self._selected % len(self.pairs)]
        self._selected += 1
        return True

    def warm(self, hidden: torch.Tensor, aux: Sequence[torch.Tensor]) -> None:
        """Compile the comparison for this output shape (startup, all ranks)."""
        self._rows(hidden, list(aux), hidden, list(aux), self.head.weight)

    def compare(self, phase, tag, out_stock, out_mk, real_rows: int) -> None:
        """`out_*`: (hidden, aux list | None) as the runner reads them, of the
        pair's first and second forward."""
        (h_s, aux_s), (h_m, aux_m) = out_stock, out_mk
        aux_s, aux_m = list(aux_s or []), list(aux_m or [])
        if len(aux_s) != len(aux_m) or h_s.shape != h_m.shape:
            raise AssertionError(
                f"parity: stock {tuple(h_s.shape)} x{len(aux_s)} aux vs "
                f"megakernel {tuple(h_m.shape)} x{len(aux_m)} aux"
            )
        f, i = self._rows(h_s, aux_s, h_m, aux_m, self.head.weight)
        f, i = f[:, :real_rows].tolist(), i[:, :real_rows].tolist()
        fr = dict(zip(_PF_FIELDS, f))
        ir = dict(zip(_PI_FIELDS, i))
        aux_rel = f[len(_PF_FIELDS) :]
        n = real_rows
        rel = fr["rel_l2"]
        over = sum(r > self.tol for r in rel)
        agree = sum(x == y for x, y in zip(ir["s_tok"], ir["m_tok"]))
        bad_s = sum(x > 0 for x in ir["bad_stock"])
        bad_m = sum(x > 0 for x in ir["bad_mk"])
        a = self.agg.setdefault((phase, self.pair), self._new_agg())
        a["calls"] += 1
        a["compared"] += int(n > 0)
        a["rows"] += n
        a["max_rel_l2"] = max([a["max_rel_l2"], *rel])
        a["max_abs_diff"] = max([a["max_abs_diff"], *fr["max_abs_diff"]])
        a["over_tol_rows"] += over
        a["bad_rows_stock"] += bad_s
        a["bad_rows_mk"] += bad_m
        a["shard_top1_agree"] += agree
        rows = {k: [round(v, 5) for v in vs] for k, vs in fr.items()}
        rows.update(ir)
        rows["aux_rel_l2"] = [[round(v, 5) for v in vs] for vs in aux_rel]
        _emit(
            self.rank,
            f"parity {PHASES[phase]} #{self.calls[phase]} {tag} rows {n} "
            f"{'_'.join(self.pair)}: "
            f"rel_l2 max {max(rel, default=0):.4g} mean "
            f"{sum(rel) / max(n, 1):.4g}; max_abs_diff "
            f"{max(fr['max_abs_diff'], default=0):.4g} (max_abs_stock "
            f"{max(fr['max_abs_stock'], default=0):.4g}); aux rel_l2 max "
            f"{[round(max(v, default=0), 4) for v in aux_rel]}; bad rows stock "
            f"{bad_s} mk {bad_m}; shard top1 agree {agree}/{n}; over_tol {over} "
            f"| {json.dumps(rows, separators=(',', ':'))}",
        )
        _emit(self.rank, self.summary(phase))

    def summary(self, phase: int) -> str:
        a = self.agg.get((phase, self.pair), self._new_agg())
        return (
            f"parity summary {PHASES[phase]} {'_'.join(self.pair)}: calls "
            f"{a['calls']} (with rows {a['compared']}), rows {a['rows']}, max rel_l2 "
            f"{a['max_rel_l2']:.4g}, "
            f"max_abs_diff {a['max_abs_diff']:.4g}, over_tol({self.tol}) rows "
            f"{a['over_tol_rows']}, bad rows stock {a['bad_rows_stock']} mk "
            f"{a['bad_rows_mk']}, shard top1 agree {a['shard_top1_agree']}/{a['rows']}"
        )
