"""Probe-05 accuracy dump hook (env-gated; never changes numerics).

VLLM_ACCDUMP_DIR        output dir (enables the hook; unset = no-op)
VLLM_ACCDUMP_LAYERS     indexer layer ids to dump, e.g. "10,42,74"
VLLM_ACCDUMP_DEPTHS     chunk-end depths to dump, e.g. "140288,204800,261120"
VLLM_ACCDUMP_PROMPTLEN  only prompts of exactly this length get indexer dumps
VLLM_ACCDUMP_TF_MINLEN  prompts >= this get teacher-forced logprobs on their final chunk
VLLM_ACCDUMP_TOPK       top-k logprobs kept per position (default 1024)
"""

import hashlib
import math
import os
import re
import time

import numpy as np
import torch

_DIR = os.environ.get("VLLM_ACCDUMP_DIR", "")
ENABLED = bool(_DIR)
_LAYERS = {int(x) for x in os.environ.get("VLLM_ACCDUMP_LAYERS", "").split(",") if x}
_DEPTHS = {int(x) for x in os.environ.get("VLLM_ACCDUMP_DEPTHS", "").split(",") if x}
_PROMPTLEN = int(os.environ.get("VLLM_ACCDUMP_PROMPTLEN", "0") or 0)
_TF_MINLEN = int(os.environ.get("VLLM_ACCDUMP_TF_MINLEN", "0") or 0)
_TOPK = int(os.environ.get("VLLM_ACCDUMP_TOPK", "1024"))
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_owners = None


def _log(msg):
    print(f"ACCDUMP pid={os.getpid()} {msg}", flush=True)


def _find_owners(runner):
    global _owners
    if _owners is not None:
        return _owners
    ctx = runner.vllm_config.compilation_config.static_forward_context
    _owners = {}
    for name, mod in ctx.items():
        owner = getattr(mod, "_accdump_owner", None)
        m = _LAYER_RE.search(name)
        if owner is None or m is None:
            continue
        gid = None
        for i, g in enumerate(runner.kv_cache_config.kv_cache_groups):
            if name in g.layer_names:
                gid = i
        _owners[int(m.group(1))] = (name, mod, owner, gid)
    _log(f"indexer layers with dump buffers: {sorted(_owners)}")
    return _owners


def _doc_key(tok):
    return hashlib.md5(np.asarray(tok[:256], dtype=np.int64).tobytes()).hexdigest()[:12]


def _outdir():
    d = os.path.join(_DIR, f"host{os.uname().nodename}")
    os.makedirs(d, exist_ok=True)
    return d


def maybe_dump(runner, scheduler_output, start_index, num_reqs, hidden_states, attn_metadata):
    """Called after forward (hidden_states = finalized, token-ordered)."""
    if not ENABLED or num_reqs != 1:
        return
    req_id = runner.input_batch.req_ids[start_index]
    ri = runner.input_batch.req_id_to_index[req_id]
    n = int(scheduler_output.num_scheduled_tokens[req_id])
    num_computed = int(runner.input_batch.num_computed_tokens_cpu[ri])
    num_prompt = int(runner.input_batch.num_prompt_tokens[ri])
    end = num_computed + n
    if end > num_prompt:
        return
    tok = runner.input_batch.token_ids_cpu[ri, :num_prompt]
    key = _doc_key(tok)

    if _PROMPTLEN and num_prompt == _PROMPTLEN and end in _DEPTHS:
        t0 = time.time()
        for layer, (name, mod, owner, gid) in sorted(_find_owners(runner).items()):
            if layer not in _LAYERS:
                continue
            md = attn_metadata.get(name) if isinstance(attn_metadata, dict) else None
            kv = mod.kv_cache
            tpb = kv.shape[1] * kv.shape[2]
            npages = math.ceil(end / tpb)
            bt_md = md.block_tables[:npages].cpu() if md is not None else None
            if gid is not None:
                bt = runner.input_batch.block_table[gid].get_cpu_tensor()[ri, :npages].clone()
            else:
                bt = bt_md.clone()
            pages = torch.index_select(kv, 0, bt.to(torch.int32).to(kv.device)).cpu()
            q = owner._dump_q[:n].cpu()
            w = owner._dump_w[:n].view(torch.int16).cpu()
            out = dict(
                q_u8=q.numpy(), w_bf16_as_i16=w.numpy(), pages_u8=pages.numpy(),
                seq_len=np.int64(end), n_rows=np.int64(n), block_ids=bt.numpy(),
                block_ids_md=(bt_md.numpy() if bt_md is not None else np.zeros(0)),
                seq_lens_md=(md.seq_lens.cpu().numpy() if md is not None else np.zeros(0)),
                cu_q_md=(md.query_start_loc.cpu().numpy() if md is not None else np.zeros(0)),
                dist_md=(torch.as_tensor(md.request_distribution).cpu().numpy()
                         if md is not None else np.zeros(0)),
            )
            fn = os.path.join(_outdir(), f"idx_{key}_L{layer}_d{end}.npz")
            np.savez(fn, **out)
        _log(f"indexer dump doc={key} end={end} layers={sorted(_LAYERS)} {time.time()-t0:.1f}s")

    if _TF_MINLEN and num_prompt >= _TF_MINLEN and end == num_prompt:
        t0 = time.time()
        idx = torch.arange(n, dtype=torch.int32).to(hidden_states.device)
        logits = runner.compute_selected_logits(hidden_states, idx)
        logits = logits.float().cpu()
        V = runner.model_config.get_vocab_size()
        lp = torch.log_softmax(logits[:, :V].double(), dim=-1)
        tgt = torch.from_numpy(np.asarray(tok[num_computed + 1 : end], dtype=np.int64))
        tgt_lp = lp[torch.arange(n - 1), tgt]
        top_lp, top_ids = lp.topk(_TOPK, dim=-1)
        tail = (1.0 - top_lp.exp().sum(-1)).clamp_min(0.0)
        ent = -(lp.exp() * lp).sum(-1)
        fn = os.path.join(_outdir(), f"tf_{key}_L{num_prompt}.npz")
        np.savez(
            fn, tgt_lp=tgt_lp.numpy(), tgt=tgt.numpy(), top_ids=top_ids.to(torch.int32).numpy(),
            top_lp=top_lp.float().numpy(), tail=tail.numpy(), entropy=ent.numpy(),
            num_prompt=np.int64(num_prompt), start=np.int64(num_computed),
            head_tokens=np.asarray(tok[:32], dtype=np.int64),
        )
        _log(f"tf dump doc={key} L={num_prompt} rows={n} nll={-tgt_lp.mean().item():.5f} {time.time()-t0:.1f}s")
