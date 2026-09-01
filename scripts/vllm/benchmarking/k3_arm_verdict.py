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
"""Correctness verdicts over `k3_completion_probe.py` captures.

Two questions, one file, because they answer different failure modes and a
run needs both:

  sanity <captures.json> <arm>
      Is this arm's output degenerate ON ITS OWN? Distinct-token ratio over a
      greedy continuation separates a model emitting text from a model stuck
      in a cycle, with no reference output and no tokenizer.

  agree <root> <arm,arm,...>       (also runs `divergence`)
  divergence <root> <arm,arm,...>
      How far apart are two arms numerically, in nats of top-1 logprob over
      the common token prefix? A boolean "not identical" cannot separate a
      reordered reduction from a miscompile; a magnitude can.

  agree <root> <arm,arm,...>
      Do two arms produce THE SAME greedy tokens? Far more sensitive than the
      ratio: an arm whose collective is miscompiled can still emit fluent
      text. Identical prompts at temperature 0 must agree byte for byte
      unless the arithmetic changed. Reported as agreeing prompts and, for
      the first disagreement, the common prefix length -- a divergence at
      token 0 is a different fault from one at token 40.

Everything goes to stdout: a job that loses its artifacts still has the log.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

# Well under any healthy continuation, well over a model stuck in a short
# cycle. Calibrate against a known-good run before tightening.
DEGENERATE_BELOW = 0.25


def _tokens(cap: dict) -> list:
    return cap.get("tokens") or (cap.get("text") or "").split()


def sanity(path: str, arm: str) -> int:
    caps = json.load(open(path))
    worst = None
    for key, cap in sorted(caps.items()):
        if "error" in cap:
            print(f"K3SANITY arm={arm} prompt={key} error={cap['error'][:70]!r}")
            continue
        toks = _tokens(cap)
        ratio = len(set(toks)) / len(toks) if toks else 0.0
        worst = ratio if worst is None else min(worst, ratio)
    if worst is None:
        print(f"K3SANITY arm={arm} verdict=NO_DATA")
        return 0
    verdict = "OK" if worst >= DEGENERATE_BELOW else "DEGENERATE"
    print(f"K3SANITY arm={arm} worst_distinct_ratio={worst:.3f} verdict={verdict}")
    return 0


def _prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def agree(root: str, arms: list[str]) -> int:
    caps = {}
    for arm in arms:
        path = os.path.join(root, arm, "completions.json")
        if os.path.exists(path):
            caps[arm] = json.load(open(path))
        else:
            print(f"K3AGREE arm={arm} missing {path}")
    for x, y in itertools.combinations(caps, 2):
        same = total = 0
        first_div = None
        for key in sorted(caps[x]):
            cx, cy = caps[x].get(key, {}), caps[y].get(key, {})
            if "error" in cx or "error" in cy:
                continue
            total += 1
            tx, ty = cx.get("text") or "", cy.get("text") or ""
            if tx == ty:
                same += 1
            elif first_div is None:
                first_div = (key, _prefix_len(tx, ty), tx[:60], ty[:60])
        print(f"K3AGREE {x} vs {y}: {same}/{total} prompts byte-identical")
        if first_div:
            key, n, tx, ty = first_div
            print(f"K3AGREE   first divergence prompt={key} common_prefix={n} chars")
            print(f"K3AGREE   {x}: {tx!r}")
            print(f"K3AGREE   {y}: {ty!r}")
    return 0


def divergence(root: str, arms: list[str]) -> int:
    """Magnitude, not a boolean: how far apart are two arms numerically?

    `agree` says whether two arms produced the same tokens. It cannot say
    whether a difference is a reordered reduction (which any change of
    collective implementation produces, and which is harmless) or a
    miscompile. The top-1 logprob at each position is a continuous readout of
    the same forward pass, so |dlogprob| over the COMMON token prefix
    separates the two: bf16-scale reordering perturbs it by ~1e-2 nats,
    corruption by O(1).
    """
    caps = {}
    for arm in arms:
        path = os.path.join(root, arm, "completions.json")
        if os.path.exists(path):
            caps[arm] = json.load(open(path))
    for x, y in itertools.combinations(caps, 2):
        deltas, first_tok_div, npos = [], [], 0
        for key in sorted(caps[x]):
            cx, cy = caps[x].get(key, {}), caps[y].get(key, {})
            if "error" in cx or "error" in cy:
                continue
            tx, ty = _tokens(cx), _tokens(cy)
            lx = cx.get("token_logprobs") or []
            ly = cy.get("token_logprobs") or []
            n = 0
            while n < min(len(tx), len(ty)) and tx[n] == ty[n]:
                n += 1
            first_tok_div.append(n)
            for i in range(min(n, len(lx), len(ly))):
                if lx[i] is not None and ly[i] is not None:
                    deltas.append(abs(lx[i] - ly[i]))
                    npos += 1
        if not deltas:
            print(f"K3DIVERGE {x} vs {y}: no comparable positions")
            continue
        deltas.sort()
        print(f"K3DIVERGE {x} vs {y}: over {npos} matched-token positions "
              f"mean|dlogprob|={sum(deltas)/len(deltas):.4f} "
              f"median={deltas[len(deltas)//2]:.4f} max={deltas[-1]:.4f} nats; "
              f"first token divergence at index "
              f"min={min(first_tok_div)} median="
              f"{sorted(first_tok_div)[len(first_tok_div)//2]} max={max(first_tok_div)}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "sanity":
        return sanity(argv[2], argv[3])
    if len(argv) >= 4 and argv[1] == "agree":
        rc = agree(argv[2], argv[3].split(","))
        return rc or divergence(argv[2], argv[3].split(","))
    if len(argv) >= 4 and argv[1] == "divergence":
        return divergence(argv[2], argv[3].split(","))
    print(f"usage: {argv[0]} sanity <captures.json> <arm>\n"
          f"       {argv[0]} agree <root> <arm,arm,...>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
