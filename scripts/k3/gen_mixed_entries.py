#!/usr/bin/env python3
"""Emit paste-ready `mixed` TuningKey entries from mla_tune_node.sh winners.

Reads winner-<tokens>.json files from OUT_DIR and prints tuned_params.py
entries for the DP=4 x TP=8 geometry (num_heads=12, max_num_seqs from
--max-num-seqs, pages_per_seq=640). The decode (batched_decode) entries are
already printed paste-ready at the end of decode-sweep.log by
tools/kernel/tuner/mla_kernel_tuner.py; this covers only the mixed case.

Usage: gen_mixed_entries.py OUT_DIR --max-num-seqs 32 --tokens 1,2,4,8,16,32,2048
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("out_dir")
p.add_argument("--max-num-seqs", type=int, required=True)
p.add_argument("--tokens", default="1,2,4,8,16,32,2048")
args = p.parse_args()

TMPL = """\
    TuningKey(
        case="mixed",
        max_num_tokens={tokens},
        actual_num_q_heads=12,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="bfloat16",
        page_size_per_kv_packing=4,
        kv_packing=4,
        max_num_seqs={seqs},
        pages_per_seq=640,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split={q_split},
        num_kv_pages_per_block={bkvp},
        num_queries_per_block={bq},
        vmem_limit_bytes={vmem},
    ),  # median {median:.3f} ms"""

for t in args.tokens.split(","):
    path = os.path.join(args.out_dir, f"winner-{t}.json")
    w = json.load(open(path))
    print(
        TMPL.format(tokens=t,
                    seqs=args.max_num_seqs,
                    q_split=w["q_split"],
                    bkvp=w["bkvp"],
                    bq=w["bq"],
                    vmem=w["vmem_mib"] * 2**20,
                    median=w["median_ms"]))
