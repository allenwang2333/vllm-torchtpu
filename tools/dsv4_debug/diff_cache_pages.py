#!/usr/bin/env python3
"""Byte-level cache differential for chain_cache_diff.sh page dumps.

Usage: diff_cache_pages.py <dump_label>   (dir under $DSV4_SCRATCH/page_dumps)

A. Write fidelity: within ONE call, stored SWA row (dequantized) vs the
   same call's bf16 kv row. Healthy = fp8 nope error ~2-6% rel, bf16 rope
   ~0.4%. This bounds the write path.
B. Prompt rows cross-run: decode-leg prefill wrote positions 0..16; the TF
   control's prefill wrote the same. Byte-identical expected.
C. Decode-written SWA rows vs control's prefill-written rows for the same
   positions: quantifies the frozen forward-path drift.
D. Compressed slots + state rows cross-comparison (correct pages now).
E. L2 kv drift growth curve by decode step (mean metric).
"""
import glob
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.join(
    os.getenv("DSV4_SCRATCH", os.path.expanduser("~/workspace/scratch")),
    "page_dumps", sys.argv[1] if len(sys.argv) > 1 else "cache_diff_L2")
spec = importlib.util.spec_from_file_location(
    "cnr",
    os.path.expanduser("~/workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/"
                       "deepseek_v4/compress_norm_rope.py"))
cnr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnr)
import jax.numpy as jnp  # noqa: E402


def load():
    pre, post = {}, {}
    for f in sorted(glob.glob(ROOT + "/attn_step*.npz")):
        d = dict(np.load(f))
        key = f.replace("_post.npz", ".npz")
        if "_post" in f:
            post[key] = d
        else:
            pre[key] = d
    entries = {}
    for f, d in pre.items():
        p = d["positions"]
        if p.max() == 0:
            continue
        k = ("pf", int(p.max()) + 1) if p[0] == 0 else ("dec", int(p[0]))
        if k not in entries:
            entries[k] = (d, post.get(f))
    return entries


def sw_row(d, postd, pos):
    sw = postd["sw_pages_post"]
    ids = list(postd.get("sw_page_ids_post", d.get("sw_page_ids", [])))
    page = int(d["swa_block_tables"].flatten()[pos // 128])
    if page not in ids:
        return None
    pi = ids.index(page)
    phys = sw.shape[1] * sw.shape[2]
    flat = sw[pi].reshape(phys, sw.shape[-1])
    return flat[pos % 128]


def dequant(row):
    nope, rope, scale = cnr.unpack_sparse_kv_cache(jnp.asarray(row[None]), 448,
                                                   64, 64)
    deq = (np.asarray(nope.astype(jnp.float32)).reshape(7, 64) *
           np.asarray(scale).reshape(7, 1)).reshape(-1)
    ro = np.asarray(rope.astype(jnp.float32)).reshape(-1)
    return deq, ro


def main():
    E = load()

    print("=" * 90)
    print("A. WRITE FIDELITY: stored SWA row vs same call's bf16 kv")
    d, postd = E[("pf", 17)]
    kv = d["kv"].astype(np.float32)
    for pos in (0, 5, 12, 16):
        r = sw_row(d, postd, pos)
        if r is None:
            print(f"  pos{pos}: page not dumped")
            continue
        deq, ro = dequant(r)
        ref = kv[pos]
        e = np.abs(deq - ref[:448])
        er = np.abs(ro - ref[448:])
        print(f"  pos{pos}: nope rel_err mean="
              f"{e.mean()/max(np.abs(ref[:448]).mean(),1e-9):.2%} "
              f"max={e.max():.3g} | rope rel_err mean="
              f"{er.mean()/max(np.abs(ref[448:]).mean(),1e-9):.2%}")

    print()
    print("B. PROMPT ROWS CROSS-RUN (decode-leg prefill vs TF prefill)")
    dd, dp = E[("pf", 17)]
    cd, cp = E[("pf", 30)]
    for pos in (0, 5, 12, 16):
        a = sw_row(dd, dp, pos)
        b = sw_row(cd, cp, pos)
        if a is None or b is None:
            print(f"  pos{pos}: missing page")
            continue
        print(f"  pos{pos}: bytes identical = {bool((a==b).all())} "
              f"({(a==b).mean()*100:.1f}%)")

    print()
    print("C. DECODE-WRITTEN SWA ROWS vs control prefill-written, same pos")
    for pos in (17, 19, 22, 26, 29):
        dk = ("dec", pos)
        if dk not in E or ("pf", 30) not in E:
            continue
        d1, p1 = E[dk]
        a = sw_row(d1, p1, pos)
        b = sw_row(cd, cp, pos)
        if a is None or b is None:
            print(f"  pos{pos}: missing page")
            continue
        da, ra = dequant(a)
        db, rb = dequant(b)
        e = np.abs(da - db)
        print(f"  pos{pos}: bytes same={(a==b).mean()*100:.0f}% "
              f"nope diff mean={e.mean():.4g} "
              f"rel={e.mean()/max(np.abs(db).mean(),1e-9):.1%} "
              f"rope diff mean={np.abs(ra-rb).mean():.4g}")

    print()
    print("D. COMPRESSED SLOTS decode vs control (unpacked values)")

    def slot_rows(d, n):
        pages = d["main_pages"]
        ids = list(d["main_page_ids"])
        page = int(d["main_page_indices"].flatten()[0])
        if page not in ids:
            return None
        flat = pages[ids.index(page)].reshape(-1, pages.shape[-1])
        return flat[:n]

    ddec, _ = E[("dec", 29)]  # kv_len 7: slots 0-6, 4-6 decode-written
    dctl, _ = E[("pf", 30)]
    ra = slot_rows(ddec, 7)
    rb = slot_rows(dctl, 7)
    if ra is not None and rb is not None:
        for s in range(7):
            da, _ = dequant(ra[s])
            db, _ = dequant(rb[s])
            e = np.abs(da - db)
            print(f"  slot{s}: bytes same={(ra[s]==rb[s]).mean()*100:.0f}% "
                  f"nope diff mean={e.mean():.4g} "
                  f"rel={e.mean()/max(np.abs(db).mean(),1e-9):.1%}")

    print()
    print(
        "E. L2 kv drift growth by decode step (vs TF control where ctx matches)"
    )
    for qpos in range(17, 30):
        dk, pk = ("dec", qpos), ("pf", qpos + 1)
        if dk not in E or pk not in E:
            continue
        a = E[dk][0]["kv"][0].astype(np.float32)
        b = E[pk][0]["kv"][qpos].astype(np.float32)
        diff = np.abs(a - b)
        print(f"  qpos={qpos}: mean={diff.mean():.4g} "
              f"frac>0.1={(diff>0.1).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
