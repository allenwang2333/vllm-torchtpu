#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fold structured-decoding metrics into a ``vllm bench serve`` result file.

Called by ``run_benchmarks.sh`` after each profiled point when a config sets
``SD_TRACE_SUMMARY=1``. It adds three things to the measured run's
``isl*_osl*_c*.json`` so ``check_regression.py`` can gate on them and
``upload_results.py`` ships them with everything else:

- ``sd_prepare_input_mean_ms`` / ``sd_mask_logits_mean_ms``: mean duration of
  the ``SD:PrepareInput`` and ``SD:MaskLogits`` scopes the TPU runner records,
  read from the Chrome traces the server's torch profiler wrote.
- ``sd_mask_memory``: bytes per step for the packed int32 bitmask the runner
  ships vs. a dense bool mask and dense logits. A closed-form function of
  batch size and vocab size, recorded for the dashboards rather than gated.
- ``block_efficiency``: tokens emitted per engine step, from the server's
  ``/metrics``. 1.0 without speculative decoding.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# The runner appends "#k=v,...#" metadata to the scope name, so these match as
# prefixes. Result-file key per scope.
SD_SCOPES: dict[str, str] = {
    "SD:PrepareInput": "sd_prepare_input_mean_ms",
    "SD:MaskLogits": "sd_mask_logits_mean_ms",
}


def scope_of(event_name: str) -> str | None:
    """Map a raw trace event name to one of ``SD_SCOPES``, else None."""
    for scope in sorted(SD_SCOPES, key=len, reverse=True):
        if event_name == scope or event_name.startswith(scope + "#"):
            return scope
    return None


# Kineto writes one field per line and copies TPU kernel (HLO) names into
# "name" verbatim, unescaped quotes included (e.g.
# ``custom_call_target="tpu_custom_call"``), which makes the file invalid JSON.
_NAME_LINE = re.compile(r'^(\s*"name": ")(.*)(",?)$', re.MULTILINE)
_UNESCAPED_QUOTE = re.compile(r'(?<!\\)"')


def _repair_name_quotes(text: str) -> str:
    """Escape stray double quotes inside ``"name": "..."`` lines."""
    return _NAME_LINE.sub(
        lambda m: m.group(1) + _UNESCAPED_QUOTE.sub(r'\\"', m.group(2)) + m.group(3),
        text,
    )


def load_events(path: Path) -> list[dict[str, Any]]:
    """``traceEvents`` from a ``.pt.trace.json`` or ``.pt.trace.json.gz``."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_repair_name_quotes(text))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    return events if isinstance(events, list) else []


def scope_durations_ms(events: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-call durations in ms for every complete (``ph == "X"``) SD event."""
    out: dict[str, list[float]] = {scope: [] for scope in SD_SCOPES}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name, dur = event.get("name"), event.get("dur")
        if not isinstance(name, str) or not isinstance(dur, (int, float)):
            continue
        scope = scope_of(name)
        if scope is not None:
            out[scope].append(float(dur) / 1000.0)
    return out


def trace_files(trace_dir: Path) -> list[Path]:
    return sorted(
        p
        for pattern in ("*.pt.trace.json", "*.pt.trace.json.gz")
        for p in trace_dir.rglob(pattern)
    )


def collect_scope_means(trace_dir: Path, exclude: set[Path]) -> dict[str, float]:
    """Mean ms per SD scope across traces in ``trace_dir`` not in ``exclude``.

    Every point profiles into the same directory, so ``exclude`` holds the
    traces earlier points left behind. With TP > 1 each rank writes its own
    trace of the same steps; the rank with the most samples is used rather
    than pooling, which would count every step once per rank.
    """
    best: dict[str, list[float]] = {scope: [] for scope in SD_SCOPES}
    excluded = {p.resolve() for p in exclude}
    for path in trace_files(trace_dir):
        if path.resolve() in excluded:
            continue
        try:
            per_file = scope_durations_ms(load_events(path))
        except (OSError, ValueError, EOFError):
            continue
        for scope, samples in per_file.items():
            if len(samples) > len(best[scope]):
                best[scope] = samples
    return {s: statistics.fmean(v) for s, v in best.items() if v}


def mask_memory(
    num_reqs: int, vocab_size: int, logits_dtype_bytes: int = 2
) -> dict[str, float | int]:
    """Bytes per step: packed int32 bitmask vs dense bool mask vs dense logits.

    The runner ships ``ceil(vocab / 32)`` int32 words per row and unpacks them
    inside the compiled kernel; the dense figures price out materializing the
    mask on the host instead. Logits default to bf16.
    """
    if num_reqs <= 0 or vocab_size <= 0:
        raise ValueError("num_reqs and vocab_size must be positive")
    packed = num_reqs * math.ceil(vocab_size / 32) * 4
    dense_bool = num_reqs * vocab_size
    dense_logits = num_reqs * vocab_size * logits_dtype_bytes
    return {
        "packed_bytes": packed,
        "dense_bool_bytes": dense_bool,
        "dense_logits_bytes": dense_logits,
        "savings_vs_bool": 1.0 - packed / dense_bool,
        "savings_vs_logits": 1.0 - packed / dense_logits,
    }


def block_efficiency(metrics_text: str) -> float:
    """Tokens per engine step from Prometheus text; 1.0 without spec decode.

    Each speculative step emits its accepted drafts plus one bonus token, so
    the mean is ``1 + accepted / drafts`` over vLLM's cumulative counters.
    """
    totals: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            totals[name] = totals.get(name, 0.0) + float(parts[-1])
        except ValueError:
            continue
    drafts = totals.get("vllm:spec_decode_num_drafts_total", 0.0)
    accepted = totals.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    return round(1.0 + accepted / drafts, 4) if drafts > 0 else 1.0


_CELL_RE = re.compile(r"^isl(\d+)_osl(\d+)_c(\d+)\.json$")


def summary_row(config: str, cell: Path) -> dict[str, Any] | None:
    """One concise row for a measured ``isl*_osl*_c*.json`` cell."""
    match = _CELL_RE.match(cell.name)
    if match is None:
        return None
    data = json.loads(cell.read_text(encoding="utf-8"))
    isl, osl, conc = (int(g) for g in match.groups())
    prep = data.get("sd_prepare_input_mean_ms")
    mask = data.get("sd_mask_logits_mean_ms")
    tpot = data.get("mean_tpot_ms")
    memory = data.get("sd_mask_memory") or {}
    per_step = round(prep + mask, 4) if prep is not None and mask is not None else None
    return {
        "config": config,
        "isl": isl,
        "osl": osl,
        "concurrency": conc,
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "output_throughput_tok_s": _round(data.get("output_throughput"), 2),
        "mean_tpot_ms": _round(tpot, 3),
        "p99_tpot_ms": _round(data.get("p99_tpot_ms"), 3),
        "mean_ttft_ms": _round(data.get("mean_ttft_ms"), 3),
        "sd_prepare_input_mean_ms": prep,
        "sd_mask_logits_mean_ms": mask,
        # Host+device ms per step in the two SD scopes. Measured in the profiled
        # run, so not directly comparable to mean_tpot_ms (profiling slows the
        # step, and async scheduling overlaps part of PrepareInput).
        "sd_ms_per_step": per_step,
        "mask_packed_bytes": memory.get("packed_bytes"),
        "mask_savings_vs_bool": memory.get("savings_vs_bool"),
        "block_efficiency": data.get("block_efficiency"),
    }


def _round(value: Any, ndigits: int) -> Any:
    return round(value, ndigits) if isinstance(value, (int, float)) else value


def build_summary(result_dirs: list[Path]) -> dict[str, Any]:
    """``{"metadata": ..., "results": [...]}`` across every config directory."""
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for result_dir in result_dirs:
        config_file = result_dir / "config.json"
        if config_file.is_file() and not metadata:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            metadata = {
                key: cfg.get(key)
                for key in (
                    "model",
                    "tensor_parallelism",
                    "num_prompts",
                    "benchmark_temperature",
                    "timestamp",
                )
            }
        for cell in sorted(result_dir.glob("isl*_osl*_c*.json")):
            row = summary_row(result_dir.name, cell)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (r["config"], r["isl"], r["osl"], r["concurrency"]))
    return {"metadata": metadata, "results": rows}


_TABLE_COLUMNS = (
    ("config", "config"),
    ("concurrency", "conc"),
    ("output_throughput_tok_s", "tok/s"),
    ("mean_tpot_ms", "TPOT ms"),
    ("sd_prepare_input_mean_ms", "PrepareInput ms"),
    ("sd_mask_logits_mean_ms", "MaskLogits ms"),
    ("sd_ms_per_step", "SD ms/step"),
    ("mask_savings_vs_bool", "mask saved vs bool"),
)


def format_table(summary: dict[str, Any]) -> str:
    """Markdown table of the summary rows."""
    header = "| " + " | ".join(title for _, title in _TABLE_COLUMNS) + " |"
    lines = [header, "|" + "---|" * len(_TABLE_COLUMNS)]
    for row in summary["results"]:
        cells = [
            "-" if row.get(key) is None else str(row[key]) for key, _ in _TABLE_COLUMNS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--result-file", type=Path, help="Enrich one measured cell.")
    mode.add_argument(
        "--summarize",
        type=Path,
        nargs="+",
        metavar="RESULTS_DIR",
        help="Condense these config result directories into one summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="With --summarize: where to write the summary JSON.",
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument(
        "--exclude-list",
        type=Path,
        help="File listing traces (one per line) that predate this point.",
    )
    parser.add_argument("--num-reqs", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--metrics-url")
    args = parser.parse_args(argv)

    if args.summarize:
        summary = build_summary(args.summarize)
        if args.output is not None:
            args.output.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote {args.output}")
        print(format_table(summary))
        return 0

    if args.num_reqs is None or args.vocab_size is None:
        parser.error("--result-file needs --num-reqs and --vocab-size")

    data = json.loads(args.result_file.read_text(encoding="utf-8"))

    if args.trace_dir is not None and args.trace_dir.is_dir():
        exclude: set[Path] = set()
        if args.exclude_list is not None and args.exclude_list.is_file():
            exclude = {
                Path(line)
                for line in args.exclude_list.read_text().splitlines()
                if line
            }
        means = collect_scope_means(args.trace_dir, exclude)
        if not means:
            print(f"WARNING: no SD: scopes in new traces under {args.trace_dir}")
        for scope, mean_ms in means.items():
            data[SD_SCOPES[scope]] = round(mean_ms, 4)

    data["sd_mask_memory"] = mask_memory(args.num_reqs, args.vocab_size)

    if args.metrics_url:
        try:
            with urllib.request.urlopen(args.metrics_url, timeout=5) as resp:
                data["block_efficiency"] = block_efficiency(resp.read().decode())
        except OSError as exc:
            print(f"WARNING: could not scrape {args.metrics_url}: {exc}")

    args.result_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
