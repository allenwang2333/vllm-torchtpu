#!/usr/bin/env python3
"""Diff a decode run's per-step logprobs against the teacher-forced
prefill scoring of the same tokens (chain_prefill_decode.sh legs 1+2).

For every step: the decode run's logprob of its own picked token
([STEP_LOGPROBS]) vs the prefill-scored logprob of that token at the same
context ([TF_LOGPROBS]). A healthy stack agrees within reporting
quantization (~0.125) at every step; the reference stack's own
prefill-vs-decode agreement is 0.01-0.3. Growing drift, or picked tokens
falling outside the prefill state's top-5 ("MISS"), means decode-built
cache state diverges from prefill-built state -- see
docs/dsv4_decode_race_postmortem.md for the bug class this catches.

Usage: diff_teacher_forcing.py <decode.log> <tf.log>
"""
import re
import sys

TOK = re.compile(
    r"(\d+): Logprob\(logprob=([-\d.e+]+), rank=(\d+), decoded_token='(.*?)'\)"
)


def parse_dict(s):
    return {
        int(m.group(1)): (float(m.group(2)), int(m.group(3)), m.group(4))
        for m in TOK.finditer(s)
    }


def parse_step(path, tag):
    steps = {}
    for line in open(path, errors="replace"):
        m = re.search(
            rf"\[{tag}\] (?:file=\S+ )?step=(\d+)"
            r"(?: forced_next=\d+ picked=\d+)? (\{{.*\}})\s*$".replace(
                "{{", "{").replace("}}", "}"), line)
        if m:
            steps[int(m.group(1))] = parse_dict(m.group(2))
    return steps


def main():
    dec = parse_step(sys.argv[1], "STEP_LOGPROBS")
    tf = parse_step(sys.argv[2], "TF_LOGPROBS")
    common = sorted(set(dec) & set(tf))
    if not common:
        sys.exit("no overlapping steps -- check both logs have logprob lines")
    print(
        f"{'step':>4} {'token':>16} {'decode':>8} {'prefill':>8} {'drift':>7}")
    n_miss, worst = 0, 0.0
    for s in common:
        pick = [k for k, v in dec[s].items() if v[1] == 1][0]
        name = dec[s][pick][2]
        dv = dec[s][pick][0]
        tv = tf[s].get(pick, (None, ))[0]
        if tv is None:
            n_miss += 1
            print(f"{s:>4} {name!r:>16} {dv:>8.3f} {'MISS':>8}")
        else:
            drift = abs(tv - dv)
            worst = max(worst, drift)
            print(f"{s:>4} {name!r:>16} {dv:>8.3f} {tv:>8.3f} {drift:>7.3f}")
    print()
    print(f"steps={len(common)} decode-pick-outside-prefill-top5={n_miss} "
          f"max_drift={worst:.3f}")
    if n_miss or worst > 1.0:
        print("VERDICT: decode-built state DIVERGES from prefill-built state "
              "(unhealthy; the reference stays within ~0.3 everywhere)")
    else:
        print("VERDICT: decode matches prefill within noise (healthy)")


if __name__ == "__main__":
    main()
