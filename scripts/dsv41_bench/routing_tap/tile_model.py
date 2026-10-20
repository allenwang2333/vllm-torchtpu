# SPDX-License-Identifier: Apache-2.0
"""Per-device tile-cost model of the fused-EP v2 MoE kernel at the DSv4.1 C384
decode shape (8 EP devices, 48 local experts, 2304 global rows, top-6).

    kernel_us = K + max_d [ v*visits_d + tau*tiles_d + d64*n64_d + d96*n96_d
                            + d128*n128_d + rho*rows_d ]

An expert's rows are aligned up to ROWBLK (2) per (expert, source rank) run;
tiles = ceil(rows / 128), full tiles run at height 128 and the tail tile at the
smallest rung in (32, 64, 96, 128) that covers it. PARAMS were fitted to 21
kernel-mode slopes of the isolated harness (uniform, skewed, replayed and
relabelled routings, 352-828 us/layer; RMS error 4.4 us) at 00fae88d6.
"""

import numpy as np

EP, E, T, TOPK, TILE, ROWBLK = 8, 384, 2304, 6, 128, 2
G = E // EP
RUNGS = (32, 64, 96, 128)
# K, v, tau, d64, d96, d128, rho
PARAMS = np.array([75.348, -0.670, 6.432, 0.867, 2.369, 2.546, 0.003])


def expert_rows(ids, ep=EP, experts=E):
    """[experts] ROWBLK-aligned rows per expert; token t comes from rank t // (T/ep)."""
    ids = np.asarray(ids).reshape(ids.shape[0], -1)
    tokens, k = ids.shape
    src = np.repeat(np.arange(ep), tokens // ep)[:, None].repeat(k, 1)
    runs = np.zeros((experts, ep), np.int64)
    np.add.at(runs, (ids.ravel(), src.ravel()), 1)
    return ((runs + ROWBLK - 1) // ROWBLK * ROWBLK).sum(1)


def rows_from_counts(counts):
    """[experts] aligned rows from counts [ep(src), experts]."""
    counts = np.asarray(counts)
    return ((counts + ROWBLK - 1) // ROWBLK * ROWBLK).sum(0)


def _tiles(n):
    full, tail = divmod(int(n), TILE)
    return [TILE] * full + ([next(r for r in RUNGS if r >= tail)] if tail else [])


def device_features(rows_e, perm=None):
    """[EP, 6]: visits, tiles, n64, n96, n128, rows; perm[g] = slot (device slot//G)."""
    slot = np.arange(len(rows_e)) if perm is None else np.asarray(perm)
    f = np.zeros((EP, 6))
    for g, n in enumerate(rows_e):
        if n == 0:
            continue
        d = slot[g] // G
        rungs = _tiles(n)
        f[d] += [1, len(rungs), rungs.count(64), rungs.count(96), rungs.count(128), n]
    return f


def device_cost(f, p=PARAMS):
    """Per-device cost [EP] without K."""
    return f @ np.array([p[1], p[2], p[3], p[4], p[5], p[6]])


def predict(rows_e, perm=None, p=PARAMS):
    """(kernel us = K + max device, K + mean device, per-device costs)."""
    c = device_cost(device_features(rows_e, perm), p)
    return p[0] + c.max(), p[0] + c.mean(), c


def expert_cost(n, p=PARAMS):
    if n == 0:
        return 0.0
    rungs = _tiles(n)
    extra = {32: 0.0, 64: p[3], 96: p[4], 128: p[5]}
    return p[1] + len(rungs) * p[2] + sum(extra[r] for r in rungs) + p[6] * n


def balanced_perm(rows_e, cost=expert_cost):
    """Min-makespan placement with G experts per device: LPT, then swaps off the
    most loaded device while they lower the makespan. Returns perm[g] = slot."""
    rows_e = np.asarray(rows_e)
    c = np.array([cost(n) for n in rows_e])
    load = np.zeros(EP)
    members = [[] for _ in range(EP)]
    for g in np.argsort(-c, kind="stable"):
        d = min((d for d in range(EP) if len(members[d]) < G), key=lambda d: load[d])
        members[d].append(g)
        load[d] += c[g]
    while True:
        dmax = int(np.argmax(load))
        best = None
        for dmin in range(EP):
            if dmin == dmax:
                continue
            for a in members[dmax]:
                for b in members[dmin]:
                    delta = c[a] - c[b]
                    if delta > 0 and load[dmin] + delta < load[dmax] - 1e-9:
                        gap = abs(load[dmax] - load[dmin] - 2 * delta)
                        if best is None or gap < best[0]:
                            best = (gap, dmax, dmin, a, b, delta)
        if best is None:
            break
        _, dmax, dmin, a, b, delta = best
        members[dmax].remove(a)
        members[dmin].remove(b)
        members[dmax].append(b)
        members[dmin].append(a)
        load[dmax] -= delta
        load[dmin] += delta
    perm = np.empty(len(rows_e), np.int64)
    for d in range(EP):
        for i, g in enumerate(sorted(members[d])):
            perm[g] = d * G + i
    return perm
