#!/usr/bin/env python3
"""Parse a phase's rank-0 hlo_stats once and cache it as JSON.

xprof takes ~1-2 minutes to open a capture, and every question asked of a
trace re-opens it. Cache the parsed table next to the trace so the analysis
scripts are instant and identical across arms.

  python3 scripts/k3/hlo_cache.py results/<arm> prefill_only
"""
import glob
import json
import os
import sys


def main():
    results_dir, phase = sys.argv[1], sys.argv[2]
    out = os.path.join(results_dir, f"hlo-{phase}.json")
    if os.path.exists(out) and "--force" not in sys.argv:
        print(f"cached: {out}")
        return 0
    pattern = f"{results_dir}/traces/{phase}/plugins/profile/*/rank0_*.xplane.pb"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no rank0 {phase} trace under {results_dir}")
    from xprof.convert import raw_to_tool_data
    data = raw_to_tool_data.xspace_to_tool_data([paths[0]], "hlo_stats", {})
    if isinstance(data, tuple):
        data = data[0]
    parsed = json.loads(data)
    cols = [c["label"] for c in parsed["cols"]]
    rows = [dict(zip(cols, [c.get("v") for c in r["c"]])) for r in parsed["rows"]]
    with open(out, "w") as fh:
        json.dump({"source": paths[0], "cols": cols, "rows": rows}, fh)
    print(f"wrote {out}: {len(rows)} rows, columns={cols}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
