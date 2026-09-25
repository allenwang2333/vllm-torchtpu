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
"""Offline-analysis dump of the DSA indexer's real inputs and outputs.

Strictly off unless ``TPU_DSA_DUMP_LAYERS`` is set. Debug tooling only.

The compiled forward cannot call back to the host (the indexer is a jax.export
op inside a torch_tpu executable), so the dump is split in two:

* in-graph (``record``): the chosen indexer layers copy q (fp8 bytes), the
  weights handed to ``streamindex_topk`` and the returned top-k indices into
  persistent device buffers -- the same in-graph buffer-mutation pattern as
  ``topk_indices_buffer``.
* host (``after_forward``, called by the runner right after the compiled
  forward, still inside the forward context): when sequence 0 of this step
  has ``seq_len >= TPU_DSA_DUMP_MIN_SEQLEN``, read those buffers, the step
  metadata and the indexer K cache rows of sequence 0 for ``[0, seq_len)``
  (fp8 bytes + ue8m0 scale byte), and ``np.savez`` them.

Env:
  TPU_DSA_DUMP_LAYERS      comma list of indexer ordinals (construction order,
                           0-based; the prefix is recorded in each file).
  TPU_DSA_DUMP_DIR         output dir (default ``$CDK_OUTPUT_DIR/dsa_dump``).
  TPU_DSA_DUMP_MIN_SEQLEN  default 131072.
  TPU_DSA_DUMP_MAX         dumps per layer, default 3.
  TPU_DSA_DUMP_DP_RANKS    comma list of DP ranks allowed to dump
                           (default: any rank that sees a qualifying step).
"""

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

HEADS, HEAD_DIM = 32, 128
KEY_BYTES = HEAD_DIM + 1  # fp8 key bytes + one e8m0 scale byte


def _int_list(name: str) -> list[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


# Read lazily, not at import: Ray workers can import this module before the
# driver's env vars are applied to their process.
def _layers() -> list[int] | None:
    return _int_list("TPU_DSA_DUMP_LAYERS")


def _min_seqlen() -> int:
    return int(os.environ.get("TPU_DSA_DUMP_MIN_SEQLEN", "131072"))


def _max_dumps() -> int:
    return int(os.environ.get("TPU_DSA_DUMP_MAX", "3"))


_ordinal = 0
_registry: list = []  # modules that carry dump buffers


def enabled() -> bool:
    return _layers() is not None


def active() -> bool:
    """Any layer armed in this process (cheap; checked every forward)."""
    return bool(_registry)


def maybe_register(module, dp_rank: int) -> None:
    """Called from the indexer's __init__; attaches buffers to chosen layers."""
    global _ordinal
    ordinal = _ordinal
    _ordinal += 1
    module._dsa_dump = None
    module._dsa_dump_q = None
    layers = _layers()
    if layers is None or ordinal not in layers:
        return
    dp_ranks = _int_list("TPU_DSA_DUMP_DP_RANKS")
    if dp_ranks is not None and dp_rank not in dp_ranks:
        return
    ref = module.topk_indices_buffer
    rows = ref.shape[0]
    dev = ref.device
    # Host-only bookkeeping; never read while tracing.
    module._dsa_dump = {"ordinal": ordinal, "dp_rank": dp_rank, "count": 0}
    # Plain tensor attributes, like `topk_indices_buffer`, so the compiled
    # graph lifts them as inputs and writes the in-place copies back.
    module._dsa_dump_q = torch.zeros(
        (rows, HEADS * HEAD_DIM), dtype=torch.uint8, device=dev
    )
    module._dsa_dump_w = torch.zeros((rows, HEADS), dtype=torch.bfloat16, device=dev)
    module._dsa_dump_topk = torch.zeros(
        (rows, module.topk_tokens), dtype=torch.int32, device=dev
    )
    _registry.append(module)
    logger.warning(
        "[dsa_dump] armed indexer ordinal %d (%s) dp_rank %d rows %d",
        ordinal, module.k_cache.prefix, dp_rank, rows,
    )


def record(module, q_bytes, weights, topk_indices) -> None:
    """In-graph: copy this step's indexer inputs/outputs into the buffers."""
    qb = module._dsa_dump_q
    if qb is None:
        return
    if q_bytes.shape[1] != HEADS or q_bytes.shape[2] != HEAD_DIM:
        raise ValueError(f"dsa_dump expects q [T,{HEADS},{HEAD_DIM}], got {q_bytes.shape}")
    n = min(topk_indices.shape[0], qb.shape[0])
    qb[:n].copy_(q_bytes[:n].reshape(n, HEADS * HEAD_DIM))
    module._dsa_dump_w[:n].copy_(weights[:n].reshape(n, HEADS))
    module._dsa_dump_topk[:n, : topk_indices.shape[1]].copy_(topk_indices[:n])


def _out_dir() -> str:
    d = os.environ.get("TPU_DSA_DUMP_DIR") or os.path.join(
        os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "dsa_dump"
    )
    os.makedirs(d, exist_ok=True)
    return d


def after_forward(positions=None) -> None:
    """Host: save qualifying steps. Called by the runner after the forward.

    A failing dump is logged and disarms itself; it never takes the server down.
    """
    if not _registry:
        return
    try:
        _after_forward(positions)
    except Exception:  # noqa: BLE001 - debug tooling must not kill serving
        logger.exception("[dsa_dump] dump failed; disarming")
        _registry.clear()


def _after_forward(positions) -> None:
    from vllm.forward_context import get_forward_context

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    try:
        attn = get_forward_context().attn_metadata
    except Exception:  # noqa: BLE001 - no context (e.g. profiling run)
        return
    if not isinstance(attn, dict):
        return
    max_dumps, min_seqlen = _max_dumps(), _min_seqlen()
    for module in _registry:
        d = module._dsa_dump
        if d["count"] >= max_dumps:
            continue
        md = attn.get(module.k_cache.prefix)
        if not isinstance(md, AttentionMetadata):
            continue
        seq_lens = md.seq_lens.cpu().numpy()
        cu = md.query_start_loc.cpu().numpy()
        seq_len = int(seq_lens[0])
        qlen = int(cu[1] - cu[0])
        if seq_len < min_seqlen or qlen <= 0:
            continue
        kv = module.k_cache.kv_cache
        if not isinstance(kv, torch.Tensor) or kv.dim() != 4 or kv.numel() == 0:
            continue  # memory-profiling run: no KV cache allocated yet
        num_pages, spp, pack, width = kv.shape
        page = spp * pack
        page_indices = md.block_tables.cpu().numpy().astype(np.int64)
        pps = page_indices.shape[0] // seq_lens.shape[0]
        npages = -(-seq_len // page)
        assert npages <= pps, (npages, pps)
        pages = page_indices[:npages]
        idx = torch.from_numpy(pages).to(kv.device)
        rows = kv.index_select(0, idx).cpu().numpy().reshape(npages * page, width)
        keys = np.ascontiguousarray(rows[:seq_len, :KEY_BYTES])
        T = int(cu[-1]) if cu.shape[0] > 1 else qlen
        T = max(T, qlen)
        q = module._dsa_dump_q.cpu().numpy()[:T].reshape(T, HEADS, HEAD_DIM)
        w = module._dsa_dump_w.float().cpu().numpy()[:T]
        topk = module._dsa_dump_topk.cpu().numpy()[:T]
        pos = None
        if positions is not None:
            pos = positions[:T].cpu().numpy()
        fname = os.path.join(
            _out_dir(),
            f"dsa_dump_dp{d['dp_rank']}_idx{d['ordinal']}_n{d['count']}_s{seq_len}.npz",
        )
        np.savez(
            fname,
            prefix=np.array(module.k_cache.prefix),
            q_fp8=q,
            weights_f32_from_bf16=w,
            topk_indices=topk,
            keys_fp8_scale=keys,
            seq_lens=seq_lens,
            cu_q_lens=cu,
            positions=pos if pos is not None else np.zeros(0, np.int64),
            page_indices_seq0=pages,
            num_pages_total=np.array(num_pages),
        )
        d["count"] += 1
        logger.warning(
            "[dsa_dump] wrote %s (seq_len %d, qlen %d, T %d)", fname, seq_len, qlen, T
        )
