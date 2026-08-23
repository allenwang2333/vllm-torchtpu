#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge full-run DP batch stats, analyze packing, and emit Perfetto JSON."""

import argparse

from vllm_torchtpu.runner.dp_batch_stats import write_converted_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_dir", help="directory containing rank*.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-dp-size", type=int)
    parser.add_argument("--target-bucket", type=int, default=256)
    args = parser.parse_args()
    paths = write_converted_outputs(
        args.stats_dir,
        args.output_dir,
        expected_dp_size=args.expected_dp_size,
        target_bucket=args.target_bucket,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
