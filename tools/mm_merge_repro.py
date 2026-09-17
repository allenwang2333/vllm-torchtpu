"""Is torch.where enough to make the multimodal merge shape-static on TPU?

vLLM merges vision embeddings into the token stream with

    inputs_embeds[is_multimodal] = mm_embeds_flat        # models/utils.py

``mm_embeds_flat`` is compacted -- one row per multimodal token in this step --
so its leading dim is the step's mm token total, which changes with every new
image. On a lazy-tensor backend every distinct value of that dim compiles its
own XLA program.

This times nothing; it only counts programs in the torch-tpu cache, isolating
the merge from the work that builds its inputs. Four formulations over a
sequence of mm token counts that never repeats:

  A  masked_setitem   what vLLM core does today
  E  where_cumsum     what vllm_torchtpu already patches in: cumsum index +
                      torch.where, gathering from the compacted tensor
  B  where_devpad     torch.where, mm padded to the token bucket on-device
  C  where_gather     torch.where, mm gathered from a bucket-shaped source
                      through a host-built index of static length
  D  index_copy       C's placement written with index_copy_ instead

The point of B is that `torch.where` on its own changes nothing: padding the
compacted tensor up to the bucket is the same variable-shape op wearing a hat.
What removes the shape axis is never compacting in the first place -- C and D
both read a bucket-shaped source, which is what the encoder cudagraph manager
already produces.
"""

import argparse

import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal import execution_mode, sync

execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE

DEV = torch.device("tpu")
TOKENS = 1024          # the runner's token bucket: static across steps
HIDDEN = 512
BUCKET_FLOOR = 64
# Phase 1: mm token counts a server sees first. Phase 2: counts it has never
# seen that land in buckets phase 1 already used -- the phase that separates a
# program set keyed on the image size from one keyed on the bucket. Phase 3
# repeats phase 1 to show the phase-1 programs were kept, not evicted.
COUNTS = [37, 129, 250, 300, 350, 401, 512, 700]
FRESH = [41, 200, 260, 333, 480, 500, 640, 900]
# Phase 4: same buckets again, but the run no longer starts at token 0 and
# some steps carry two images. Offsets and item counts must arrive as data,
# not as shapes, or the bucket-keyed variants would recompile here too.
PLACED = [(41, 13), (200, 77), (260, 5), (333, 301), (480, 120), (500, 2),
          (640, 64), (900, 31)]
# Phase 5: the same shapes at yet other offsets. Phase 4 may pay once for the
# first non-zero offset it sees; if offsets were a shape axis, phase 5 would
# pay again.
PLACED2 = [(41, 400), (200, 300), (260, 700), (333, 88), (480, 9), (500, 400),
           (640, 200), (900, 100)]


def bucket(n: int) -> int:
    b = BUCKET_FLOOR
    while b < n:
        b *= 2
    return b


def programs() -> int:
    return len(torch.tpu._get_cache_stats().per_entry_stats)


def drain(*tensors):
    sync.synchronize(tensors=[t for t in tensors if t is not None], wait=True)


def make_inputs(n, offset=0):
    """Everything a step has before the merge, drained so it is not counted."""
    emb = torch.randn(TOKENS, HIDDEN).to(DEV)
    is_mm = torch.zeros(TOKENS, dtype=torch.bool)
    offset = min(offset, TOKENS - n)
    is_mm[offset:offset + n] = True
    is_mm = is_mm.to(DEV)
    compact = torch.randn(n, HIDDEN).to(DEV)          # today's encoder cache
    padded = torch.randn(bucket(n), HIDDEN).to(DEV)   # bucket-shaped source
    # Host-built, static length: which source row each token position reads.
    gather = torch.zeros(TOKENS, dtype=torch.long)
    gather[offset:offset + n] = torch.arange(n)
    gather = gather.to(DEV)
    drain(emb, is_mm, compact, padded, gather)
    return emb, is_mm, compact, padded, gather


def masked_setitem(emb, is_mm, compact, padded, gather):
    out = emb.clone()
    out[is_mm] = compact
    return out


def where_devpad(emb, is_mm, compact, padded, gather):
    mm_full = torch.zeros(TOKENS, HIDDEN, dtype=emb.dtype, device=DEV)
    mm_full[:compact.shape[0]] = compact
    return torch.where(is_mm.unsqueeze(1), mm_full, emb)


def where_gather(emb, is_mm, compact, padded, gather):
    mm_full = padded.index_select(0, gather)
    return torch.where(is_mm.unsqueeze(1), mm_full, emb)


def where_cumsum(emb, is_mm, compact, padded, gather):
    """What vllm_torchtpu already installs (_patch_vllm_merge_multimodal_
    embeddings): no nonzero, but the gather still reads the compacted tensor,
    so its height -- this step's mm token total -- keys the program."""
    idx = torch.cumsum(is_mm, dim=0, dtype=torch.int32) - 1
    idx = torch.clamp(idx, 0, compact.shape[0] - 1)
    return torch.where(is_mm.unsqueeze(-1), compact[idx], emb)


def index_copy(emb, is_mm, compact, padded, gather):
    out = emb.clone()
    out.index_copy_(0, gather, padded.index_select(0, gather))
    return out


VARIANTS = [
    ("A masked_setitem", masked_setitem),
    ("B where_devpad", where_devpad),
    ("C where_gather", where_gather),
    ("E where_cumsum", where_cumsum),
    ("D index_copy", index_copy),
]


def run(fn, counts):
    new = 0
    for item in counts:
        n, offset = item if isinstance(item, tuple) else (item, 0)
        inputs = make_inputs(n, offset)
        before = programs()
        out = fn(*inputs)
        drain(out)
        new += programs() - before
    return new


def main():
    argparse.ArgumentParser().parse_args()

    # Warm the allocator and the H2D shapes so the first variant is not
    # charged for them.
    drain(*make_inputs(COUNTS[0]))

    print(f"tokens={TOKENS} hidden={HIDDEN}")
    print(f"phase 1 counts  {COUNTS}")
    print(f"        buckets {[bucket(n) for n in COUNTS]}")
    print(f"phase 2 counts  {FRESH}   (all new, no new buckets)")
    print(f"        buckets {[bucket(n) for n in FRESH]}\n")
    print(f"phase 4 counts  {[n for n, _ in PLACED]}   (offset, not at 0)")
    print(f"        offsets {[o for _, o in PLACED]}\n")
    print(f"{'variant':<18}{'phase1':>8}{'phase2':>8}{'phase3':>8}{'phase4':>8}{'phase5':>8}"
          "   <- new XLA programs")
    for name, fn in VARIANTS:
        p1 = run(fn, COUNTS)
        p2 = run(fn, FRESH)
        p3 = run(fn, COUNTS)
        p4 = run(fn, PLACED)
        p5 = run(fn, PLACED2)
        print(f"{name:<18}{p1:>8}{p2:>8}{p3:>8}{p4:>8}{p5:>8}")


if __name__ == "__main__":
    main()
