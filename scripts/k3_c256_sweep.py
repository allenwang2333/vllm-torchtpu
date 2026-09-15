"""Resumable PR898 FP8 local-TP32 geometry sweep on a 32-device slice."""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    rank = int(
        os.getenv("FALCON_JAX_PROCESS_ID", os.getenv("FALCON_RANK", "0")))
    root = Path(os.environ["OUT"]) / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    contexts = (128, 512, 1000, 2000)
    winners = []
    cs = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    buckets = (8, 16, 32, 48, 96, 192, 384, 768, 1536)
    for window in (1, 6):
        for c, bucket in zip(cs, buckets):
            tokens = c if window == 1 else bucket
            tiles = ([(1, 3, 4), (1, 16, 1), (1, 16, 2), (1, 16, 4),
                      (1, 16, 8), (1, 32, 2),
                      (1, 32, 4)] if window == 1 else [(16, 1, 1), (16, 16, 1),
                                                       (8, 16, 1), (32, 16, 1),
                                                       (16, 8, 1), (16, 32, 1),
                                                       (32, 16, 2)])
            valid = []
            for index, tile in enumerate(tiles):
                label = f"w{window}-c{c}-tile" + "-".join(map(str, tile))
                path = root / (label + ".json")
                resume = os.getenv("TUNER_RESUME_ROOT")
                prior = ([
                    Path(resume) / f"rank-{r}/benchmark/{label}.json"
                    for r in range(4)
                ] if resume else [])
                bundles = None
                if prior and all(p.is_file() for p in prior):
                    try:
                        loaded = [json.loads(p.read_text()) for p in prior]
                        if all(
                                len(batch) == len(contexts) and all(
                                    r["status"] == "success"
                                    and r["cache_valid"]
                                    and r["next_output_max_abs_error"] == 0
                                    and r["concurrency"] == c
                                    and r["verify_tokens"] == window and
                                    r["tokens"] == tokens and r["max_num_seqs"]
                                    == 256 and r["pages_per_seq"] == 256
                                    for r in batch) for batch in loaded):
                            bundles = loaded
                    except (ValueError, KeyError, OSError):
                        pass
                # Every host makes the same decision, preserving JAX rendezvous.
                if bundles is not None:
                    path.write_text(json.dumps(bundles[rank]))
                    print("CANDIDATE_RESUMED", label, flush=True)
                else:
                    print("CANDIDATE_START", label, flush=True)
                    result = subprocess.run([
                        sys.executable,
                        str(Path(__file__).with_name("k3_fp8_mla_tuner.py")),
                        "--case",
                        json.dumps([c, window, tokens]), "--tile",
                        json.dumps(tile), "--output",
                        str(path)
                    ],
                                            check=False,
                                            timeout=2400)
                    if result.returncode or not path.exists():
                        if index == 0:
                            raise RuntimeError(
                                f"Baseline failed: {label}, exit={result.returncode}"
                            )
                        print("CANDIDATE_FAILED",
                              label,
                              result.returncode,
                              flush=True)
                        continue
                rows = json.loads(path.read_text())
                assert len(rows) == len(contexts)
                assert all(r["status"] == "success" and r["cache_valid"]
                           and r["next_output_max_abs_error"] == 0
                           for r in rows)
                assert all(
                    r["concurrency"] == c and r["verify_tokens"] == window
                    and r["tokens"] == tokens and r["max_num_seqs"] == 256
                    and r["pages_per_seq"] == 256 for r in rows)
                # Benchmark stays <=1001 KV tokens; 2000 is correctness-only.
                score = statistics.fmean(r["latency_ms"] for r in rows
                                         if r["kv_len"] <= 1000)
                valid.append((score, tile, rows[0]["tuning_key"]))
                with (root / "metrics.jsonl").open("a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                print("CANDIDATE_VALIDATED", label, score, flush=True)
            valid.sort()
            assert valid
            score, tile, key = valid[0]
            winners.append(
                dict(key=key,
                     tile=tile,
                     latency_ms=score,
                     timing="host_dispatch_plus_synchronized_kernel"))
            (root / "winners.partial.json").write_text(
                json.dumps(winners, indent=2))
    (root / "winners.json").write_text(json.dumps(winners, indent=2))
    if rank == 0:
        Path("/coord/winners.json").write_text(json.dumps(winners, indent=2))
    print("TUNE_COMPLETE", len(winners), flush=True)


if __name__ == "__main__":
    main()
