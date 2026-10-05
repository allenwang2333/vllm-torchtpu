#!/usr/bin/env python3
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
"""Assert that decode pulled only the part of a prompt it did not have cached.

With TPU_RAIDEN_PREFIX_AWARE_LOAD=1, a decode request that hits decode's own
prefix cache skips those tokens and pulls only the rest from prefill (#487).
If the feature is off, or the installed tpu-sync cannot clip a transfer, decode
quietly pulls the whole payload instead. The answers are still right and the
/metrics counters read the same, so this reads decode's own log:

  "Prefix-aware Stage-3 loads are unavailable"   the feature never turned on
  "Stage-3 load ... skip_tokens=N ..."            one line per decode load
  raiden_stage3_transfer_submitted (JSON)         the byte clip actually sent

A load with skip_tokens > 0 must skip whole pages and fill exactly the pages
the rest of the payload needs, and fa_skip_bytes / skip_tokens must be the same
bytes-per-token on every load.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

LOAD_RE = re.compile(
    r"Stage-3 load req_id=\S+ source_req_id=\S+ uuid=\d+ "
    r"num_tokens=(\d+) skip_tokens=(\d+) destination_pages=(\d+)"
)
UNAVAILABLE = "Prefix-aware Stage-3 loads are unavailable"
LOAD_FAILURES = (
    "Prefix hits must be destination-page aligned",
    "Stage-3 FA resharding requires the complete destination page set",
)
SUBMITTED_EVENT = "raiden_stage3_transfer_submitted"

_DECODER = json.JSONDecoder()


def _objects_in(line: str):
    """Yield every JSON object embedded in a log line."""
    start = line.find("{")
    while start >= 0:
        try:
            obj, end = _DECODER.raw_decode(line, start)
        except json.JSONDecodeError:
            start = line.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        start = line.find("{", end)


def _scan(lines, page_tokens: int) -> tuple[int, int, int, int, set[int]]:
    """Return (rc, loads, suffix_loads, submitted_clips, bytes_per_token)."""
    rc = 0
    loads = 0
    suffix_loads = 0
    submitted_clips = 0
    bytes_per_token: set[int] = set()
    # Each decode DP replica logs these once, so count rather than repeat.
    unavailable = 0
    load_failures = dict.fromkeys(LOAD_FAILURES, 0)
    for line in lines:
        if UNAVAILABLE in line:
            unavailable += 1
        for failure in LOAD_FAILURES:
            if failure in line:
                load_failures[failure] += 1
        match = LOAD_RE.search(line)
        if match:
            num_tokens, skip, pages = (int(g) for g in match.groups())
            loads += 1
            if skip > 0:
                suffix_loads += 1
                want = -(-(num_tokens - skip) // page_tokens)
                if skip % page_tokens or pages != want:
                    print(
                        f"PREFIX_AWARE_FAIL plan does not add up: "
                        f"num_tokens={num_tokens} skip_tokens={skip} "
                        f"destination_pages={pages}, expected {want} pages "
                        f"of {page_tokens} tokens after a whole-page skip"
                    )
                    rc = 1
        if SUBMITTED_EVENT not in line:
            continue
        for event in _objects_in(line):
            if event.get("event") != SUBMITTED_EVENT:
                continue
            skip = int(event.get("skip_tokens") or 0)
            if skip <= 0:
                continue
            submitted_clips += 1
            fa_skip_bytes = int(event.get("fa_skip_bytes") or 0)
            if fa_skip_bytes <= 0 or fa_skip_bytes % skip:
                print(
                    f"PREFIX_AWARE_FAIL req={event.get('req_id')} "
                    f"skip_tokens={skip} but fa_skip_bytes={fa_skip_bytes}"
                )
                rc = 1
            else:
                bytes_per_token.add(fa_skip_bytes // skip)
    if unavailable:
        print(
            f"PREFIX_AWARE_FAIL {unavailable} decode engine(s) reported "
            f"prefix-aware loads unavailable at startup: the flag is off or "
            f"tpu-sync lacks dst_skip_bytes support"
        )
        rc = 1
    for failure, count in load_failures.items():
        if count:
            print(f"PREFIX_AWARE_FAIL {count} decode load(s) failed: {failure}")
            rc = 1
    return rc, loads, suffix_loads, submitted_clips, bytes_per_token


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="decode server log")
    ap.add_argument(
        "--page-tokens",
        type=int,
        required=True,
        help="decode scheduler block size in tokens",
    )
    ap.add_argument(
        "--min-suffix-loads",
        type=int,
        default=1,
        help="fail if fewer loads than this skipped a cached prefix",
    )
    args = ap.parse_args()
    if args.page_tokens <= 0:
        print("PREFIX_AWARE_CHECK --page-tokens must be positive")
        return 2

    try:
        with open(args.log, encoding="utf-8", errors="replace") as handle:
            rc, loads, suffix_loads, submitted_clips, bytes_per_token = _scan(
                handle, args.page_tokens
            )
    except OSError as exc:
        print(f"PREFIX_AWARE_CHECK unreadable log {args.log}: {exc}")
        return 2

    print(f"PREFIX_AWARE_LOADS {loads}")
    print(f"PREFIX_AWARE_SUFFIX_LOADS {suffix_loads}")
    print(f"PREFIX_AWARE_SUBMITTED_CLIPS {submitted_clips}")
    print(f"PREFIX_AWARE_BYTES_PER_TOKEN {sorted(bytes_per_token)}")

    if loads == 0:
        print(
            "PREFIX_AWARE_FAIL no Stage-3 loads in the log: wrong log, or "
            "decode never received a transfer"
        )
        rc = 1
    if suffix_loads < args.min_suffix_loads:
        print(
            f"PREFIX_AWARE_FAIL {suffix_loads} load(s) skipped a cached "
            f"prefix, wanted at least {args.min_suffix_loads}: decode pulled "
            f"the whole payload every time"
        )
        rc = 1
    if suffix_loads and submitted_clips == 0:
        print(
            "PREFIX_AWARE_FAIL loads planned a skip but no transfer carried "
            "dst_skip_bytes"
        )
        rc = 1
    if len(bytes_per_token) > 1:
        print(
            f"PREFIX_AWARE_FAIL fa_skip_bytes / skip_tokens differs between "
            f"loads: {sorted(bytes_per_token)}"
        )
        rc = 1

    # 1 for pass, matching the other *_OK markers; the exit code runs the
    # other way.
    print(f"PREFIX_AWARE_CHECK_OK {int(rc == 0)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
