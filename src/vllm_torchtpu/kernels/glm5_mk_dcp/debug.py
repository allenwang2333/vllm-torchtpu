# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""G0 debug instruments for the glm5_mk_dcp MTP acceptance collapse (debug builds only).

Everything here is host-driven and opt-in by env; none of it runs unless one of GLM5_MKD_MODE,
GLM5_MKD_SPECDBG, GLM5_MKD_KVDIFF or GLM5_MKD_IDCHECK is set.

1. Mode schedule (GLM5_MKD_MODE). Each eligible decode/verify step gets a `Plan` that says which
   forward produces each of three things the runner consumes:
     L  the hidden states the step's logits are computed from (and so the sampled tokens),
     K  the KV / rope / indexer-cache rows written for this step's tokens,
     D  the hidden states handed to the MTP drafter.
   `mix:LKD` picks each from S (stock forward) or M (megakernel). When both forwards run, the stock
   forward runs first (it always writes its KV); the megakernel then runs with its writer (K = M:
   it overwrites this step's owned rows with its own bytes) or without it (K = S: read-only probe).
     stock      = mix:SSS (megakernel never runs)
     mk         = mix:MMM (the served megakernel path)
     alt:k      = per eligible step i: stock if i % k == k - 1 else mk (alt:2 == old GLM5_MKD_ALT=1)
     shadow     = mix:SSS plus a read-only megakernel probe on the same inputs: per-step KL, argmax
                  agreement, max |dhidden|; with GLM5_MKD_KVDIFF=1 also the byte diff of its staged
                  KV rows against the rows the stock forward just wrote
     cycle:a/b  = rotate through modes a, b, ... advancing at each NEW request (CONC=1 arms in one
                  boot; separators '/' or ',')
2. SPECDBG (GLM5_MKD_SPECDBG=1): per plan label, from the rejection sampler's own inputs/outputs,
   restricted to real draft slots and real request rows (the old probe averaged over the padded
   draft slots and padded request rows; see `SpecDbg`).
3. KVDIFF helpers: owned-row coordinates and the cache-vs-staging comparison.
4. Metadata identity: p0 = seq_len - nq vs position_ids[t0 + j] - j per request.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Callable

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


# ------------------------------------------------------------------------------------ modes


@dataclasses.dataclass(frozen=True)
class Plan:
    """What one eligible step runs. `label` identifies the arm in logs."""

    label: str
    ret: str = "M"          # L: hidden returned to the runner (logits + sampling): "S" or "M"
    kv: str = "M"           # K: whose bytes land in the caches for this step's rows
    draft: str = "M"        # D: hidden handed to the drafter
    stats: bool = False     # shadow statistics (needs both forwards)

    @property
    def run_stock(self) -> bool:
        return "S" in (self.ret, self.kv, self.draft)

    @property
    def run_mk(self) -> bool:
        return "M" in (self.ret, self.kv, self.draft) or self.stats

    @property
    def mk_writer(self) -> bool:
        return self.kv == "M"

    @property
    def pure_mk(self) -> bool:
        return self.run_mk and not self.run_stock

    @property
    def pure_stock(self) -> bool:
        return self.run_stock and not self.run_mk


STOCK = Plan("stock", "S", "S", "S")
MK = Plan("mk", "M", "M", "M")


def _mix(code: str, label: str | None = None, stats: bool = False) -> Plan:
    code = code.upper()
    if len(code) != 3 or set(code) - {"S", "M"}:
        raise ValueError(f"mix code {code!r}: want three of S/M for (logits, kv, draft)")
    return Plan(label or f"mix:{code}", code[0], code[1], code[2], stats)


class _Alt:
    def __init__(self, k: int):
        if k < 2:
            raise ValueError(f"alt:{k}: period must be >= 2")
        self.k = k
        self.i = 0

    def next(self) -> Plan:
        i, self.i = self.i, self.i + 1
        lab = f"alt:{self.k}"
        return dataclasses.replace(STOCK if i % self.k == self.k - 1 else MK,
                                   label=f"{lab}/{'stock' if i % self.k == self.k - 1 else 'mk'}")


def _parse_one(tok: str):
    tok = tok.strip()
    if tok == "stock":
        return STOCK
    if tok == "mk":
        return MK
    if tok == "shadow":
        return _mix("SSS", "shadow", stats=True)
    if tok == "shadow_draft":                     # stock drives; the drafter consumes the MK hidden
        return _mix("SSM", "shadow_draft", stats=True)
    if tok.startswith("mix:"):
        return _mix(tok[4:], stats=True)
    if tok.startswith("alt:"):
        return _Alt(int(tok[4:]))
    raise ValueError(f"GLM5_MKD_MODE token {tok!r}")


class ModeSchedule:
    """Host-only, rank-uniform (every TP rank sees the same SchedulerOutput) step -> Plan."""

    def __init__(self, spec: str):
        spec = (spec or "mk").strip()
        self.spec = spec
        if spec.startswith("cycle:"):
            toks = [t for t in spec[6:].replace(",", "/").split("/") if t.strip()]
            if not toks:
                raise ValueError("cycle: needs at least one mode")
            self.arms = [_parse_one(t) for t in toks]
            self.arm_names = [t.strip() for t in toks]
            self.cycle = True
        else:
            self.arms = [_parse_one(spec)]
            self.arm_names = [spec]
            self.cycle = False
        self.arm = 0
        self.seen: set[str] = set()
        self.n_new = 0

    @property
    def needs_probe(self) -> bool:
        """A read-only megakernel program is needed (some plan runs the MK without its writer)."""
        for a in self.arms:
            if isinstance(a, Plan) and a.run_mk and not a.mk_writer:
                return True
        return False

    def observe_requests(self, req_ids) -> None:
        """Advance the cycle at each request never seen before (call on every step, eligible or not:
        a request's prefill steps are stock anyway, so its decode steps all fall in one arm)."""
        new = [r for r in req_ids if r not in self.seen]
        if not new:
            return
        self.seen.update(new)
        if self.cycle:
            for _ in new:
                if self.n_new > 0:
                    self.arm = (self.arm + 1) % len(self.arms)
                self.n_new += 1

    def plan(self) -> Plan:
        a = self.arms[self.arm]
        p = a.next() if isinstance(a, _Alt) else a
        if self.cycle:
            p = dataclasses.replace(p, label=f"{self.arm}:{p.label}")
        return p


def schedule_from_env() -> ModeSchedule:
    spec = os.environ.get("GLM5_MKD_MODE", "")
    if not spec:
        spec = "alt:2" if os.environ.get("GLM5_MKD_ALT") == "1" else "mk"
    return ModeSchedule(spec)


# ------------------------------------------------------------------------- spec-decode stats


def valid_draft_slots(segment_ids: torch.Tensor, group_indices: torch.Tensor,
                      draft_lengths: torch.Tensor) -> torch.Tensor:
    """The rejection sampler's own `valid_slot_mask`: slot j is a real draft iff its segment is a
    real request and its group index is below that request's draft count. Padded slots carry the
    out-of-range segment sentinel and, under async scheduling, a REAL token id (the extract re-reads
    input_ids[final_logits_indices[0 + 1]], request 0's first draft), so `draft >= 0` cannot tell
    them apart -- the old SPECDBG counted ~13-29 padded slots per CONC=1 step (n=880 / 50 steps)."""
    P = draft_lengths.shape[0]
    seg = segment_ids.to(torch.int64)
    seg_c = seg.clamp(min=0, max=max(P - 1, 0))
    lim = draft_lengths.to(torch.int64)[seg_c]
    return (seg >= 0) & (seg < P) & (group_indices.to(torch.int64) < lim)


def draft_probs(target_logits: torch.Tensor, draft_ids: torch.Tensor, temps: torch.Tensor,
                top_k: torch.Tensor | None, top_p: torch.Tensor | None,
                topk_candidates: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(p, argmax_ok) per slot, with the rejection sampler's exact transform: logits / (T + 1e-9),
    top-k / top-p masked (`apply_top_k_top_p`), softmax; greedy rows (T == 0) use argmax == draft
    as their acceptance indicator. E[1(u < p)] = E[p], so mean p over position-0 slots must match
    the realised position-0 acceptance -- the positive control. Read together with argmax==draft at
    position 0: low p with HIGH argmax agreement = the target distribution is flat (logits wrong
    beyond argmax); low p with LOW argmax agreement = the drafts are wrong (drafter inputs)."""
    from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p

    tl = target_logits.reshape(-1, target_logits.shape[-1]).float()
    d = draft_ids.reshape(-1).to(torch.int64)
    t = temps.reshape(-1, 1).float()
    scaled = tl / (t + 1e-9)
    if top_k is not None and top_p is not None:
        scaled = apply_top_k_top_p(scaled, top_k.reshape(-1, 1), top_p.reshape(-1, 1).float(),
                                   topk_candidates)
    p = torch.softmax(scaled, -1).gather(-1, d.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    am = tl.argmax(-1) == d
    greedy = t.reshape(-1) == 0
    p = torch.where(greedy, am.float(), p)
    return p, am


class SpecDbg:
    """Per-label spec-decode statistics from the rejection sampler's own tensors (host-synced).

    Fixes vs the original `_specdbg` (d41e70f6d tpu_runner.py:7580-7619), whose stock output
    (p(draft) 0.038, argmax==draft 0.133, accepted_len 1.000 vs server AL 3.2, j-cc66edb3) cannot
    be right:
      - draft slots: only the sampler's valid slots, not every `draft >= 0` slot (padding);
      - p(draft): the sampler's transform incl. top-k / top-p, not a bare softmax;
      - accepted length: real request rows only (`sel[:num_reqs]` with drafts), reported as
        accepted / drafted token totals that are directly comparable with vLLM's
        'SpecDecoding metrics: Accepted: X tokens, Drafted: Y tokens' (sum the windows);
      - a built-in positive control: mean p over position-0 slots vs realised position-0
        acceptance (the same draws decide both), per label.
    Every rank also logs a running hash of the step's real selected tokens (rank divergence)."""

    def __init__(self, every: int = 50):
        self.every = every
        self.st: dict[str, dict] = {}
        self.pending: dict | None = None
        self.hash = 0
        self.nsteps = 0

    def _s(self, label: str) -> dict:
        return self.st.setdefault(label, dict(steps=0, slots=0, p=0.0, am=0, p0=0.0, n0=0, am0=0,
                                              acc0=0, reqs=0, accepted=0, drafted=0))

    @torch.no_grad()
    def on_logits(self, label: str, md, target_logits, req_temperatures, req_top_k, req_top_p,
                  topk_candidates: int, all_greedy: bool) -> None:
        if md.draft_token_ids is None:
            return
        valid = valid_draft_slots(md.segment_ids, md.group_indices, md.draft_lengths)
        n = min(valid.shape[0], target_logits.reshape(-1, target_logits.shape[-1]).shape[0])
        valid = valid[:n]
        seg = md.segment_ids[:n].to(torch.int64)
        if req_temperatures is None or all_greedy:
            temps = torch.zeros(n, device=target_logits.device)
            tk = tp = None
        else:
            P = req_temperatures.shape[0]
            seg_c = seg.clamp(min=0, max=P - 1)
            temps = req_temperatures.reshape(-1)[seg_c]
            tk = req_top_k.reshape(-1)[seg_c] if req_top_k is not None else None
            tp = req_top_p.reshape(-1)[seg_c] if req_top_p is not None else None
        p, am = draft_probs(target_logits.reshape(-1, target_logits.shape[-1])[:n],
                            md.draft_token_ids.reshape(-1)[:n], temps, tk, tp, topk_candidates)
        g0 = valid & (md.group_indices[:n].to(torch.int64) == 0)
        vals = torch.stack([valid.sum().float(), (p * valid).sum(), (am & valid).sum().float(),
                            (p * g0).sum(), g0.sum().float(), (am & g0).sum().float()]).cpu().tolist()
        self.pending = dict(label=label, vals=vals)

    @torch.no_grad()
    def on_output(self, label: str, md, sel: torch.Tensor, rank: int) -> None:
        nreq = len(md.draft_lengths_cpu)
        dl = np.asarray(md.draft_lengths_cpu, dtype=np.int64)
        rows = sel[:nreq].to(torch.int64)
        v = (rows >= 0).sum(-1).cpu().numpy().astype(np.int64)
        has = dl > 0
        s = self._s(label)
        s["steps"] += 1
        s["reqs"] += int(has.sum())
        s["accepted"] += int((v[has] - 1).clip(min=0).sum())
        s["drafted"] += int(dl[has].sum())
        s["acc0"] += int((v[has] >= 2).sum())
        if self.pending is not None and self.pending["label"] == label:
            nv, ps, am, p0, n0, am0 = self.pending["vals"]
            s["slots"] += int(nv)
            s["p"] += ps
            s["am"] += int(am)
            s["p0"] += p0
            s["n0"] += int(n0)
            s["am0"] += int(am0)
        self.pending = None
        # running rank-divergence hash over the real selected tokens (identical on every rank iff
        # the ranks sampled the same tokens)
        flat = rows.reshape(-1).cpu().numpy()
        for x in flat.tolist():
            self.hash = (self.hash * 1000003 + (int(x) & 0xFFFFFFFF)) & 0xFFFFFFFFFFFF
        self.nsteps += 1
        if self.nsteps % self.every == 0:
            logger.info("SPECDBG rankhash rank %d steps %d hash %012x", rank, self.nsteps, self.hash)
        if s["steps"] % self.every == 0 and rank == 0:
            logger.info("SPECDBG %s", self.summary(label))

    def summary(self, label: str) -> str:
        s = self._s(label)
        al = 1.0 + s["accepted"] / max(s["reqs"], 1)
        return (f"{label} steps {s['steps']} AL {al:.3f} accepted {s['accepted']} drafted "
                f"{s['drafted']} verify_reqs {s['reqs']} | mean p(draft) "
                f"{s['p'] / max(s['slots'], 1):.3f} argmax==draft {s['am'] / max(s['slots'], 1):.3f} "
                f"(slots {s['slots']}) | pos0 E[p] {s['p0'] / max(s['n0'], 1):.3f} vs realised "
                f"{s['acc0'] / max(s['reqs'], 1):.3f}, argmax==draft "
                f"{s['am0'] / max(s['n0'], 1):.3f} (n {s['n0']})")


# --------------------------------------------------------------------------- shadow stats


@torch.no_grad()
def shadow_stats(logits_mk: torch.Tensor, logits_stock: torch.Tensor, h_mk: torch.Tensor,
                 h_stock: torch.Tensor) -> dict:
    """KL(p_mk || p_stock) at T = 1 per row, argmax agreement, max |dh| and ||dh|| / ||h||."""
    lm = torch.log_softmax(logits_mk.float(), -1)
    ls = torch.log_softmax(logits_stock.float(), -1)
    kl = (lm.exp() * (lm - ls)).sum(-1)
    agree = (logits_mk.argmax(-1) == logits_stock.argmax(-1)).float()
    dh = (h_mk.float() - h_stock.float())
    rel = dh.norm(dim=-1) / h_stock.float().norm(dim=-1).clamp(min=1e-6)
    out = torch.stack([kl.mean(), kl.max(), agree.mean(), dh.abs().max(), rel.max()]).cpu().tolist()
    return dict(kl_mean=out[0], kl_max=out[1], argmax_agree=out[2], dh_max=out[3], dh_rel_max=out[4],
                rows=int(kl.shape[0]))


# ------------------------------------------------------------------- metadata identity (b)


def metadata_identity(qsl: np.ndarray, seq_lens: np.ndarray, pos: np.ndarray) -> tuple[int, int, list]:
    """Per request b with nq = qsl[b+1] - qsl[b] > 0: the resolve's p0 = seq_lens[b] - nq (K:1010,
    adapter qmeta[:,2]) must equal position_ids[qsl[b] + j] - j for every j < nq (rmeta pos: K:1356
    own-row patch, K:1427 old/new split, K:1434 staging row, K:2303 writer ownership).
    Returns (requests checked, requests violating, first examples)."""
    qsl = np.asarray(qsl, dtype=np.int64)
    seq = np.asarray(seq_lens, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    n_chk = n_bad = 0
    ex = []
    for b in range(min(len(qsl) - 1, len(seq))):
        t0, nq = int(qsl[b]), int(qsl[b + 1] - qsl[b])
        if nq <= 0:
            continue
        n_chk += 1
        p0 = int(seq[b]) - nq
        got = pos[t0:t0 + nq] - np.arange(nq)
        if got.shape[0] != nq or (got != p0).any():
            n_bad += 1
            if len(ex) < 4:
                ex.append(dict(b=b, t0=t0, nq=nq, seq=int(seq[b]), pos=pos[t0:t0 + nq].tolist()))
    return n_chk, n_bad, ex


# ------------------------------------------------------------------------------- KVDIFF (d)


@dataclasses.dataclass
class RowCoords:
    """This rank's owned rows of the step: row index t, (page, slot) of its new KV, rope quarter."""

    t: torch.Tensor
    page: torch.Tensor
    slot: torch.Tensor
    quarter: torch.Tensor
    pos: torch.Tensor


def owned_row_coords(pos: torch.Tensor, req: torch.Tensor, valid: torch.Tensor, bt2: torch.Tensor,
                     nb: int, R: int, ILV: int, pg: int, rank: int) -> RowCoords:
    """Mirror of the writer's `mine` / `where_` (kernel.build_writer): owner(p) = (p % (ILV R)) //
    ILV, local lo = (p // (ILV R)) ILV + p % ILV, (page, slot) = (bt[req, lo // pg] % nb, lo % pg),
    rope quarter = p & 3 (== lo & 3 for ILV % 4 == 0). Also what main's DCP writer computes from
    seq_lens / query_start_loc (kv_cache_utils._get_dcp_page_and_slot) when the identity holds."""
    G = ILV * R
    pos = pos.to(torch.int64)
    own = valid.bool() & (((pos % G) // ILV) == rank)
    t = torch.nonzero(own).reshape(-1)
    p = pos[t]
    lo = (p // G) * ILV + p % ILV
    r = req.to(torch.int64)[t]
    vp = (lo // pg).clamp(max=bt2.shape[1] - 1)
    page = torch.remainder(bt2.to(torch.int64)[r, vp], nb)
    return RowCoords(t=t, page=page, slot=lo % pg, quarter=p & 3, pos=p)


def _bytes_of_words(w: torch.Tensor) -> torch.Tensor:
    """[n, 128] 32-bit words -> [n, 512] bytes in the nope token order (word j = bytes j, 128+j,
    256+j, 384+j, little end first) as int32 0..255."""
    w = w.to(torch.int64) & 0xFFFFFFFF
    return torch.cat([(w >> s) & 0xFF for s in (0, 8, 16, 24)], dim=-1)


def _fp8_decode(b: torch.Tensor) -> torch.Tensor:
    return b.to(torch.uint8).view(torch.float8_e4m3fn).float()


@torch.no_grad()
def compare_staging(cache_of: Callable[[str, int], torch.Tensor | None], L: int,
                    idx_layers: list[int], c: RowCoords, nrow: torch.Tensor, rrow: torch.Tensor,
                    irec: torch.Tensor, idim: int, irec_bytes: int,
                    neighbors: dict | None = None) -> dict:
    """Compare the megakernel's staged rows (nrow / rrow / irec, [L, TMAX, ...], identical on every
    rank) with what the caches hold at this rank's owned (page, slot): after the stock forward wrote
    them (shadow: numerics + format), or after the megakernel's own writer (mk: placement must be
    byte-exact). `cache_of(kind, l)` returns the layer's [nb, ...] cache view (None: no such layer).
    Returns per kind: rows, rows with any differing byte, differing bytes, max |decoded diff|,
    sum |decoded diff| / sum |ref|. With `neighbors` (a `snapshot` taken before the writer ran),
    also counts clobbered bytes in the other quarters / slots of the touched rope rows / idx groups."""
    n = int(c.t.shape[0])
    res = {k: dict(rows=0, bad_rows=0, bad_bytes=0, max_abs=0.0, l1=0.0, ref_l1=0.0)
           for k in ("nope", "rope", "idx")}
    if n == 0:
        if neighbors is not None:
            res["clobber"] = dict(rope=0, idx=0)
        return res
    lanes = (c.quarter.unsqueeze(-1) * 32 + torch.arange(32, device=c.quarter.device)).to(torch.int64)
    acc = {k: [] for k in res}
    for l in range(L):
        cn = cache_of("nope", l)
        if cn is not None:
            got = cn.view(torch.int32)[c.page, c.slot] if cn.element_size() == 4 else cn[c.page, c.slot]
            ref = nrow[l].view(torch.int32)[c.t]
            gb, rb = _bytes_of_words(got), _bytes_of_words(ref)
            acc["nope"].append((gb, rb))
        cr = cache_of("rope", l)
        if cr is not None:
            row = (cr.view(torch.int32) if cr.element_size() == 4 else cr)[c.page, c.slot >> 2]
            got = row.gather(1, lanes) & 0xFFFF
            ref = rrow[l].view(torch.int32)[c.t].gather(1, lanes) & 0xFFFF
            gb = torch.cat([got & 0xFF, (got >> 8) & 0xFF], -1)
            rb = torch.cat([ref & 0xFF, (ref >> 8) & 0xFF], -1)
            acc["rope"].append((gb, rb))
    for l in idx_layers:
        ci = cache_of("idx", l)
        if ci is None:
            continue
        col = ci[c.page, :, :, c.slot]                      # [n, IREC / 4, 4]
        got = col.reshape(n, -1)[:, :irec_bytes].to(torch.int32)
        ref = irec[l][c.t][:, :irec_bytes].to(torch.int32) & 0xFF
        acc["idx"].append((got, ref))
    for k, pairs in acc.items():
        if not pairs:
            continue
        g = torch.stack([a for a, _ in pairs])            # [layers, n, bytes]
        r = torch.stack([b for _, b in pairs])
        diff = g != r
        if k == "idx":
            gv = torch.cat([_fp8_decode(g[..., :idim]), (g[..., idim:idim + 1] - 127).float()], -1)
            rv = torch.cat([_fp8_decode(r[..., :idim]), (r[..., idim:idim + 1] - 127).float()], -1)
        else:
            gv, rv = _fp8_decode(g), _fp8_decode(r)
        d = (gv - rv).abs()
        d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        vals = torch.stack([diff.any(-1).sum().float(), diff.sum().float(), d.max(), d.sum(),
                            rv.abs().where(torch.isfinite(rv), torch.zeros_like(rv)).sum()]).cpu().tolist()
        res[k].update(rows=int(g.shape[0] * g.shape[1]), bad_rows=int(vals[0]), bad_bytes=int(vals[1]),
                      max_abs=vals[2], l1=vals[3], ref_l1=vals[4])
    if neighbors is not None:
        res["clobber"] = _clobber(cache_of, L, idx_layers, c, neighbors)
    return res


@torch.no_grad()
def snapshot(cache_of, L: int, idx_layers: list[int], c: RowCoords) -> dict:
    """Before the writer: the full 4-token rope rows and the 4 idx columns of each owned row's group."""
    snap = {"rope": {}, "idx": {}}
    if int(c.t.shape[0]) == 0:
        return snap
    base = (c.slot >> 2) << 2
    for l in range(L):
        cr = cache_of("rope", l)
        if cr is not None:
            snap["rope"][l] = (cr.view(torch.int32) if cr.element_size() == 4 else cr)[
                c.page, c.slot >> 2].clone()
    for l in idx_layers:
        ci = cache_of("idx", l)
        if ci is not None:
            snap["idx"][l] = torch.stack([ci[c.page, :, :, base + j] for j in range(4)], 1).clone()
    return snap


def _clobber(cache_of, L, idx_layers, c: RowCoords, snap: dict) -> dict:
    """Bytes that changed in a touched rope row / idx group OUTSIDE every owned new token of the step
    (same page and 4-token group). Must be 0: the writer overlays quarters / columns, it must not
    rewrite a neighbour (an accepted token of an earlier step)."""
    n = int(c.t.shape[0])
    out = dict(rope=0, idx=0)
    if n == 0:
        return out
    key = c.page * 1_000_000 + (c.slot >> 2)
    # quarters / slots written this step, per (page, 4-token group)
    newq = torch.zeros((n, 4), dtype=torch.bool, device=c.t.device)
    for j in range(4):
        same = key.unsqueeze(1) == key.unsqueeze(0)              # [n, n]
        newq[:, j] = (same & (c.quarter.unsqueeze(0) == j)).any(1)
    lane_q = torch.arange(128, device=c.t.device) // 32
    keep = ~newq[:, lane_q]                                      # [n, 128] lanes not written
    base = (c.slot >> 2) << 2
    tot_r = tot_i = torch.zeros((), device=c.t.device)
    for l, pre in snap["rope"].items():
        cr = cache_of("rope", l)
        post = (cr.view(torch.int32) if cr.element_size() == 4 else cr)[c.page, c.slot >> 2]
        tot_r = tot_r + ((post != pre) & keep).sum()
    for l, pre in snap["idx"].items():
        ci = cache_of("idx", l)
        post = torch.stack([ci[c.page, :, :, base + j] for j in range(4)], 1)
        k4 = (~newq).reshape(n, 4, 1, 1)
        tot_i = tot_i + ((post != pre) & k4).sum()
    v = torch.stack([tot_r.float(), tot_i.float()]).cpu().tolist()
    out.update(rope=int(v[0]), idx=int(v[1]))
    return out


def fmt_kvdiff(res: dict) -> str:
    parts = []
    for k in ("nope", "rope", "idx"):
        r = res.get(k)
        if not r or not r["rows"]:
            continue
        parts.append(f"{k} rows {r['rows']} bad_rows {r['bad_rows']} bad_bytes {r['bad_bytes']} "
                     f"max|d| {r['max_abs']:.4g} rel_l1 {r['l1'] / max(r['ref_l1'], 1e-12):.3e}")
    if "clobber" in res:
        parts.append(f"clobber rope {res['clobber']['rope']} idx {res['clobber']['idx']}")
    return " | ".join(parts) if parts else "no owned rows"
