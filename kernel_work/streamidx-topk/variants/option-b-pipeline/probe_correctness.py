#!/usr/bin/env python3
"""Option B correctness: identical top-k index *sets* for every G, plus the two
hard invariants from measurement_and_design.md §5.

Four things are checked, at shapes that actually split.  (The repo's own test
file cannot split: every one of its parametrised cases has 1-3 sequences, and
G=4 needs at least 4.  That file is therefore a *transparency* check on the
Option B code path -- run by `option_b_patch.py` -- and this is the *chunking*
check.)

  1. `streamindex_topk_chunked(..., num_groups=G)` for G in {1, 2, 4} returns
     the same index *set* per row as the unmodified incumbent
     `streamindex_topk`.  Sets, not sequences: the kernel documents its output
     as unsorted, so a permutation is not an error.  G=1 is not a free pass --
     it runs Option B's own single-group path (`self_path=True`), not the
     incumbent, so it is a real comparison.

  2. Every G agrees with every other G, checked directly rather than inferred
     from each agreeing with the incumbent.

  3. `_pick_partition` is never called on an Option B path.  Armed with
     `pipeline.p_tripwire`, which replaces the incumbent's chooser with a
     raiser for the duration of tracing.  A run that completes with the
     tripwire armed is the proof.  The tripwire is also *self-tested* here --
     armed around a deliberate `sparsecore_topk` call, it must fire -- because
     a tripwire that cannot fire proves nothing.

  4. `num_waves * 32 == b_pad` for every chunk, reported per (shape, G).

The KV cache is built exactly the way the repo test file builds it -- the same
`quantize_fp8_ue8m0` / `_to_byte_lane` record layout, vectorised over the whole
array instead of looped -- because a cache of random bytes decodes to garbage
(NaN included) and would make top-k tie-breaking, not the chunking, the thing
under test.

Nothing here is a timing.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import option_b_topk as ob  # noqa: E402
import pipeline as P  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    streamindex_topk)

KV_PACKING = 4  # fixed by the cache layout the test file writes
DECODE_REQ_BATCH_SIZE = 4
GS = (1, 2, 4)


# ---------------------------------------------------------------------------
# cache construction, transcribed from tests/kernels/test_dsv4_streamindex_topk
# ---------------------------------------------------------------------------


def _to_byte_lane(x):
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def quantize_fp8_ue8m0(x, block_size: int):
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


def _pack_cache(float32_kv: np.ndarray, D: int) -> jnp.ndarray:
    """(P, page_size, D) f32  ->  (P, page_size//4, 4, width) u8.

    Same record as the test file: `D` fp8 bytes then one ue8m0 scale byte,
    zero-padded to `width`, stored at `[p, s // 4, s % 4]`.
    """
    Pn, page_size, _ = float32_kv.shape
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    q_j, scale_j = quantize_fp8_ue8m0(jnp.asarray(float32_kv), D)
    q_bytes = np.asarray(_to_byte_lane(q_j))                 # (P, ps, D)
    scale_bytes = np.asarray(_to_byte_lane(scale_j))         # (P, ps)
    scale_bytes = scale_bytes.reshape(Pn, page_size, 1)

    record = np.concatenate([q_bytes, scale_bytes], axis=-1)
    record = np.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
    cache = record.reshape(Pn, page_size // KV_PACKING, KV_PACKING, width)
    return jnp.asarray(cache.astype(np.uint8))


# ---------------------------------------------------------------------------
# shapes.  Small enough to run in seconds, large enough that G=4 really splits.
#
# The planner refuses a split that leaves a partial decode batch inside a
# group, so a decode-prefix shape needs seqs/group to be a multiple of
# decode_req_batch_size=4 -- i.e. B >= 16 for G=4.  Prefill-only shapes have no
# decode prefix and are unconstrained.
# ---------------------------------------------------------------------------
SHAPES = [
    dict(name="decode_b16_s512", B=16, T=1, S=512, page_size=64, H=4, D=16,
         k=128, comp=1, bq=1, bkv_p=2, pages_per_seq=8),
    dict(name="decode_b64_s1024", B=64, T=1, S=1024, page_size=128, H=8, D=16,
         k=256, comp=1, bq=1, bkv_p=1, pages_per_seq=8),
    dict(name="prefill_b8_q8_s256", B=8, T=8, S=256, page_size=64, H=4, D=16,
         k=128, comp=1, bq=8, bkv_p=2, pages_per_seq=4),
    dict(name="prefill_b16_q4_s512", B=16, T=4, S=512, page_size=64, H=4, D=16,
         k=128, comp=1, bq=4, bkv_p=2, pages_per_seq=8),
    # mixed: an 8-sequence decode prefix followed by 8 four-token sequences.
    dict(name="mixed_b16_d8", B=16, T=1, T_tail=4, n_decode=8, S=512,
         page_size=64, H=4, D=16, k=128, comp=1, bq=4, bkv_p=2,
         pages_per_seq=8),
    # compression_ratio > 1: the effective row length path differs.
    dict(name="decode_b16_s512_c2", B=16, T=1, S=512, page_size=64, H=4, D=16,
         k=128, comp=2, bq=1, bkv_p=2, pages_per_seq=8),
]


def build(spec, seed=0):
    rng = np.random.default_rng(seed)
    B, S = spec["B"], spec["S"]
    page_size = spec["page_size"]
    pages_per_seq = spec["pages_per_seq"]
    H, D = spec["H"], spec["D"]

    n_decode = spec.get("n_decode", B if spec["T"] == 1 else 0)
    t_tail = spec.get("T_tail", spec["T"])
    t_list = [1] * n_decode + [t_tail] * (B - n_decode)
    if spec.get("n_decode") is None and spec["T"] != 1:
        t_list = [spec["T"]] * B

    num_tokens = int(sum(t_list))
    total_pages = B * pages_per_seq

    q = rng.standard_normal((num_tokens, H, D)).astype(np.float32)
    w = rng.uniform(-1.5, 1.5, size=(num_tokens, H)).astype(np.float32)
    float32_kv = rng.standard_normal(
        (total_pages, page_size, D)).astype(np.float32)
    cache = _pack_cache(float32_kv, D)

    seq_lens = np.full((B, ), S, np.int32)
    page_indices = rng.permutation(total_pages).astype(np.int32)
    cu = np.concatenate([[0], np.cumsum(t_list)]).astype(np.int32)
    distribution = (n_decode, n_decode, B)
    return dict(
        q=jnp.asarray(q), w=jnp.asarray(w), cache=cache,
        seq_lens=jnp.asarray(seq_lens), page_indices=jnp.asarray(page_indices),
        cu=jnp.asarray(cu), distribution=distribution, num_tokens=num_tokens,
        t_list=t_list)


def call_incumbent(spec, inp):
    return streamindex_topk(
        inp["q"], inp["w"], inp["cache"], inp["seq_lens"],
        inp["page_indices"], inp["cu"], inp["distribution"],
        k=spec["k"], compression_ratio=spec["comp"],
        num_kv_pages_per_block=spec["bkv_p"],
        num_queries_per_block=spec["bq"],
        decode_req_batch_size=DECODE_REQ_BATCH_SIZE)


def call_chunked(spec, inp, G, *, force_p1=True, report=None):
    return ob.streamindex_topk_chunked(
        inp["q"], inp["w"], inp["cache"], inp["seq_lens"],
        inp["page_indices"], inp["cu"], inp["distribution"],
        k=spec["k"], compression_ratio=spec["comp"],
        num_kv_pages_per_block=spec["bkv_p"],
        num_queries_per_block=spec["bq"],
        decode_req_batch_size=DECODE_REQ_BATCH_SIZE,
        num_groups=G, force_p1=force_p1, self_path=True, report=report)


def compare_index_sets(ref, got):
    """Row-wise set equality, ignoring order and the -1 padding."""
    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.shape != got.shape:
        return dict(ok=False, why=f"shape {got.shape} != {ref.shape}")
    bad_rows = []
    for r in range(ref.shape[0]):
        a = set(int(x) for x in ref[r] if x >= 0)
        b = set(int(x) for x in got[r] if x >= 0)
        if a != b:
            bad_rows.append(dict(row=r, n_ref=len(a), n_got=len(b),
                                 only_ref=sorted(a - b)[:8],
                                 only_got=sorted(b - a)[:8]))
        if len(bad_rows) >= 5:
            break
    return dict(ok=not bad_rows, bad_rows=bad_rows,
                n_pad_ref=int((ref < 0).sum()), n_pad_got=int((got < 0).sum()))


def tripwire_self_test():
    """A tripwire that cannot fire proves nothing.  Prove it fires."""
    scores = jnp.zeros((16, 1024), jnp.float32)
    lens = jnp.full((16, ), 1024, jnp.int32)
    try:
        with P.p_tripwire(active=True):
            scmod.sparsecore_topk(scores, 128, row_lengths=lens)
    except P.PickPartitionCalled:
        return dict(ok=True, note="tripwire fired on a deliberate "
                                  "sparsecore_topk call, as required")
    return dict(ok=False, note="TRIPWIRE DID NOT FIRE -- it is inert and every "
                               "'never called' result below is vacuous")


def main():
    out = {
        "module_md5s": P.module_md5s(),
        "jax": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "decode_req_batch_size": DECODE_REQ_BATCH_SIZE,
    }
    print(f"[correctness] jax={out['jax']} device={out['device_kind']}")

    st = tripwire_self_test()
    out["tripwire_self_test"] = st
    print(f"[correctness] tripwire self-test: "
          f"{'PASS' if st['ok'] else 'FAIL'} -- {st['note']}")

    results = {}
    all_ok = st["ok"]
    for spec in SHAPES:
        name = spec["name"]
        inp = build(spec, seed=1234 + len(name))
        entry = {"spec": {k: v for k, v in spec.items() if k != "name"},
                 "t_list_head": inp["t_list"][:8],
                 "num_tokens": inp["num_tokens"]}
        try:
            ref = np.asarray(jax.device_get(call_incumbent(spec, inp)))
        except Exception:
            entry["incumbent_error"] = traceback.format_exc(limit=6)
            results[name] = entry
            all_ok = False
            print(f"[correctness] {name}: incumbent FAILED to run")
            print(entry["incumbent_error"])
            continue
        entry["ref_shape"] = list(ref.shape)

        per_g = {}
        for G in GS:
            rep = {}
            key = f"G{G}"
            try:
                # Every Option B path here uses `_sc_topk_direct`, so the
                # tripwire is armed for all of them, G=1 included.
                with P.p_tripwire(active=True):
                    got = np.asarray(jax.device_get(
                        call_chunked(spec, inp, G, report=rep)))
                per_g[G] = got
                cmp = compare_index_sets(ref, got)
                wg = rep.get("waves_guard") or []
                entry[key] = dict(
                    chunked=rep.get("chunked"),
                    reason=rep.get("reason"),
                    n_groups=len(rep.get("groups") or []),
                    waves_guard=wg,
                    waves_guard_ok=(all(w["ok"] for w in wg) if wg else None),
                    pick_partition_called=False,
                    index_sets_match_incumbent=cmp["ok"],
                    compare=cmp,
                )
            except P.PickPartitionCalled as e:
                entry[key] = dict(chunked=rep.get("chunked"),
                                  pick_partition_called=True, error=str(e))
                all_ok = False
            except Exception:
                entry[key] = dict(error=traceback.format_exc(limit=8))
                all_ok = False

            e = entry[key]
            print(f"[correctness] {name} G={G}: "
                  f"chunked={e.get('chunked')} ({e.get('reason')}) "
                  f"groups={e.get('n_groups')} "
                  f"index_sets_match={e.get('index_sets_match_incumbent')} "
                  f"pick_partition_called={e.get('pick_partition_called')} "
                  f"waves_guard_ok={e.get('waves_guard_ok')}")
            if e.get("error"):
                print(e["error"])
            if e.get("index_sets_match_incumbent") is False:
                print("    mismatch detail: "
                      f"{json.dumps(e['compare']['bad_rows'][:2])}")
            all_ok = (all_ok and e.get("index_sets_match_incumbent") is True
                      and not e.get("pick_partition_called"))

        # cross-G identity, checked directly rather than inferred.
        cross = {}
        gs = sorted(per_g)
        for i in range(len(gs) - 1):
            a, b = gs[i], gs[i + 1]
            c = compare_index_sets(per_g[a], per_g[b])
            cross[f"G{a}_vs_G{b}"] = c["ok"]
            all_ok = all_ok and c["ok"]
        entry["cross_G_identical"] = cross
        print(f"[correctness] {name} cross-G: {cross}")

        # was anything actually chunked at G>1?
        chunked_any = any(entry.get(f"G{g}", {}).get("chunked") for g in GS
                          if g > 1)
        entry["chunked_at_G_gt_1"] = bool(chunked_any)
        if not chunked_any:
            print(f"[correctness] {name}: WARNING nothing chunked at G>1 -- "
                  "this shape proves transparency only")
            all_ok = False

        results[name] = entry

    out["shapes"] = results
    out["all_ok"] = all_ok

    print()
    print(f"[correctness] OVERALL: {'PASS' if all_ok else 'FAIL'}")

    d = os.environ.get("CDK_OUTPUT_DIR")
    if d:
        p = pathlib.Path(d) / "probes"
        p.mkdir(parents=True, exist_ok=True)
        (p / "option-b-correctness.json").write_text(
            json.dumps(out, indent=2, default=str))
    print("===== option-b-correctness JSON BEGIN =====")
    print(json.dumps(out, indent=2, default=str))
    print("===== option-b-correctness JSON END =====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
