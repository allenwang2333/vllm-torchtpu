# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Serving glue for the TP x DCP GLM-5.2 decode megakernel (kernels/glm5_mk_dcp).

Enabled by GLM5_MKD=1 on a TP = DCP = world, DP1, EP deployment with MTP (verify rows). The runner
takes the kernel on a step where every scheduled request is in decode/verify (no prefill chunk), at
most QMAX tokens each and TMAX in total, in one chunk; every TP rank sees the same scheduler output, so
the decision is rank-uniform without a collective. Everything else runs stock.

Operands:
- weights: load-time stacked copies [n_layers_of_kind, ...] of every kind except the routed experts
  (~2.8 GB/rank; per-layer operands would add an L-leaf DMA cascade per stream issue site and overflow
  the TensorCore instruction memory: 108.6k bundles vs 49.2k, STRATEGY P5b); the experts alias the
  served tensors per layer;
- caches: the runner's stacked caches (GLM5_MKD_STACK_C=1, runner.glm5_kv_stack), read-only in the
  megakernel and updated in place by the barrier-free writer (donated);
- smalls: one f32 row per layer per rank (norm weights, per-channel scales, router bias; the o_proj,
  shared-down and dense-down scales are per rank: main requantizes them over the local K).
The kernel's rank is Mosaic's device id on a one-axis mesh of all R devices in torch-rank order (the
DCP mesh's order); experts [ER*r, ER*r+ER) live on rank r (linear EP placement, EP rank = TP rank).
"""

import dataclasses
import math
import os
import time
from typing import Any

import jax
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.glm5_mk_dcp import debug as D
from vllm_torchtpu.kernels.glm5_mk_dcp import kernel as K
from vllm_torchtpu.kernels.glm5_mk_dcp.config import FULL
from vllm_torchtpu.kernels.glm5_mk_dcp.layout import rup, smalls_layout
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

AXIS = "mkd"
EXPERT_ALIAS = ("ew13", "ew2")       # per layer, aliased (never copied)


class Unsupported(Exception):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(why)


def enabled() -> bool:
    return os.environ.get("GLM5_MKD", "0") == "1"


def _unwrap(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def _kn(weight: torch.Tensor, k: int, n: int, what: str) -> torch.Tensor:
    if tuple(weight.shape) == (k, n):
        return weight
    _need(tuple(weight.shape) == (n, k), f"{what}: shape {tuple(weight.shape)}, want ({k}, {n})")
    return weight.t().contiguous()


def _pad_n(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    return torch.nn.functional.pad(w, (0, rup(n) - n)) if n % 128 else w


def _scale(layer, n: int, what: str) -> torch.Tensor:
    for name in ("weight_scale", "weight_scale_inv"):
        s = getattr(layer, name, None)
        if s is not None:
            _need(s.numel() == n, f"{what}.{name}: {tuple(s.shape)} is not per-channel [{n}]")
            return s.reshape(-1).float()
    raise Unsupported(f"{what}: no weight_scale")


def _find_backbone(model):
    for m in model.modules():
        if hasattr(m, "layers") and hasattr(m, "embed_tokens") and hasattr(m, "norm"):
            return m
    raise Unsupported("no decoder backbone (layers/embed_tokens/norm) in the model")


def _attn_module(self_attn):
    for m in self_attn.modules():
        if hasattr(m, "W_UK_T") and hasattr(m, "W_UV") and hasattr(m, "kv_cache"):
            return m
    raise Unsupported("no MLA attention module with W_UK_T / W_UV / kv_cache")


def _own(t: torch.Tensor) -> torch.Tensor:
    """A tensor with storage of exactly its own shape (not a view): a view operand of the compiled
    step is materialized by torch_tpu's execute_as_strided on EVERY call (2.2 ms/step of copies of
    the stacked weights and smalls, j-725a55a4)."""
    out = torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
    out.copy_(t)
    if out._base is not None:
        logger.warning("GLM-5.2 TPxDCP megakernel: owned operand %s %s is still a view",
                       tuple(out.shape), out.dtype)
    return out


def _routed(mlp):
    exp = mlp.experts
    return getattr(exp, "routed_experts", exp)


def _expert_scale(routed, name: str, shape: tuple) -> torch.Tensor:
    for suffix in ("_scale_inv", "_scale"):
        s = getattr(routed, name + suffix, None)
        if s is not None:
            _need(s.numel() == math.prod(shape), f"{name}{suffix}: {tuple(s.shape)}, want {shape}")
            return s.reshape(shape).float()
    raise Unsupported(f"routed experts: no {name} scale")


class Glm5MKD:
    """Built at load_model (weights), bound to the stacked caches at capture_model."""

    @classmethod
    def maybe_create(cls, runner) -> "Glm5MKD | None":
        if not enabled():
            return None
        try:
            return cls(runner)
        except Unsupported as e:
            logger.warning("GLM-5.2 TPxDCP megakernel disabled: %s", e)
            return None

    def __init__(self, runner):
        import torch.distributed as dist

        hf = runner.model_config.hf_config
        pc = runner.parallel_config
        _need(type(hf).__name__.startswith("GlmMoeDsa")
              or getattr(hf, "model_type", "") == "glm_moe_dsa", f"model_type {hf.model_type}")
        _need(pc.pipeline_parallel_size == 1, "pipeline parallel")
        _need(getattr(pc, "data_parallel_size", 1) == 1, "data parallel")
        R = int(pc.tensor_parallel_size)
        _need(int(getattr(pc, "decode_context_parallel_size", 1)) == R, "DCP != TP")
        _need(dist.is_initialized() and dist.get_world_size() == R, "world != TP")
        _need(int(getattr(pc, "cp_kv_cache_interleave_size", 1)) == FULL.ILV, "interleave != 4")
        _need(bool(getattr(pc, "enable_expert_parallel", False)), "EP is off")
        _need(os.environ.get("VLLM_DCP_Q_REPLICATE", "0") == "1", "q replicate is off")
        spec = runner.speculative_config
        _need(spec is not None and getattr(spec, "method", "") == "mtp", "MTP is off")
        qmax = 1 + int(spec.num_speculative_tokens)
        _need(qmax <= 4, f"{qmax} rows per request")
        _need(getattr(hf, "scoring_func", None) == "sigmoid", "router scoring")
        _need(getattr(hf, "n_group", 1) == 1 and getattr(hf, "topk_group", 1) == 1, "grouped routing")
        _need(bool(getattr(hf, "norm_topk_prob", True)), "unnormalized top-k")
        _need(bool(getattr(hf, "rope_interleave", False))
              and bool(getattr(hf, "indexer_rope_interleave", False)), "non-interleaved rope")
        rp = getattr(hf, "rope_parameters", None) or {}
        _need(rp.get("rope_type", "default") == "default", f"rope {rp}")
        _need(runner.block_size == FULL.page, f"block_size {runner.block_size}")
        cfg = dataclasses.replace(
            FULL, R=R, L=hf.num_hidden_layers, first_dense=hf.first_k_dense_replace,
            H=hf.hidden_size, NH=hf.num_attention_heads, QL=hf.q_lora_rank,
            idx_freq=hf.index_topk_freq, idx_skip=hf.index_skip_topk_offset,
            rope_theta=float(rp.get("rope_theta", FULL.rope_theta)), eps=hf.rms_norm_eps,
            routed_scale=float(hf.routed_scaling_factor), QMAX=qmax,
            RPR=int(os.environ.get("GLM5_MKD_RPR", "1")))
        # GLM5_MKD_RPR: rows per rank -> TMAX = R * RPR rows a step (MTP 3: conc 4 / 8 / 16 at RPR 1 / 2 / 4).
        # VMEM fit per RPR at R=16, L=78 (AOT): ring banks 11 / 9 / 4, chunked key AG at RPR 4.
        _need(cfg.RPR in (1, 2, 4), f"GLM5_MKD_RPR={cfg.RPR}")
        nbank, kch = {1: (11, 0), 2: (9, 0), 4: (4, 8)}[cfg.RPR]
        # GLM5_MKD_KLOC: local top-k candidates per row (exact; full-KT fallback round when a cut list
        # could hold winners). 1024: -0.3 / -0.77 / -0.62 ms at b1-84k / b2-84k / b2-334k (j-5b5bf421)
        # with headroom over the busiest rank's ~290-620 owned winners per row; 0 = KT.
        self.tiles = K.Tiles(nbank=int(os.environ.get("GLM5_MKD_NBANK", nbank)),
                             kch=int(os.environ.get("GLM5_MKD_KCH", kch)),
                             kloc=int(os.environ.get("GLM5_MKD_KLOC", 1024)))
        for got, want, what in (
                (hf.kv_lora_rank, cfg.KL, "kv_lora_rank"), (hf.qk_nope_head_dim, cfg.DN, "qk_nope"),
                (hf.qk_rope_head_dim, cfg.DR, "qk_rope"), (hf.v_head_dim, cfg.DV, "v_head_dim"),
                (hf.n_routed_experts, cfg.E, "experts"), (hf.num_experts_per_tok, cfg.TOPK, "top-k"),
                (hf.moe_intermediate_size, cfg.I, "moe_intermediate"),
                (hf.moe_intermediate_size * hf.n_shared_experts, cfg.IS, "shared intermediate"),
                (hf.intermediate_size, cfg.ID, "dense intermediate"),
                (hf.index_n_heads, cfg.IH, "index heads"), (hf.index_head_dim, cfg.IDIM, "index dim"),
                (hf.index_topk, cfg.KT, "index_topk")):
            _need(got == want, f"{what} {got} != kernel {want}")
        _need(cfg.NH % R == 0 and cfg.E % R == 0 and cfg.IS % R == 0 and cfg.ID % R == 0,
              f"{R} ranks do not divide heads / experts / intermediates")
        self.cfg = cfg
        self.rank = int(dist.get_rank())
        self.counts = {}
        from torch_tpu._internal.pallas import get_pallas_mesh
        self.mesh = get_pallas_mesh(axis_names=(AXIS,), mesh_shape=(R,))
        bb = _find_backbone(runner.model)
        self.backbone = bb
        _need(len(bb.layers) == cfg.L, f"{len(bb.layers)} layers")
        # local pages a request can span on one rank: a block-table entry is a DCP virtual page of
        # page * R positions (runner.max_num_blocks_per_req is the generic width, R times larger,
        # and sized the key scratch 16x too big: 79.6 MiB VMEM, j-f2171e56)
        mml = int(runner.model_config.max_model_len)
        self.max_pages = -(-mml // (FULL.page * R))
        self.rotary = bb.layers[0].self_attn.rotary_emb
        _need(self.rotary.cos_sin_cache.shape[-1] == cfg.DR, "rope cache width")
        # own row-major copy: gathering 16 rows of the served cache made XLA transpose the whole
        # [max_pos, 64] param every step (246 us, j-725a55a4)
        with torch.no_grad():
            self.cos_sin = _own(self.rotary.cos_sin_cache.detach())
        self.wk = K.weight_kinds(cfg)
        self.stack_w = frozenset(set(self.wk) - set(EXPERT_ALIAS))
        with torch.no_grad():
            self.weights, self.w_names, self.w_specs = self._collect_weights()
            self.smalls = self._build_smalls().detach()
        from vllm_torchtpu.utils import synchronize_tensors
        synchronize_tensors(self.weights + [self.smalls])
        copied = sum(t.numel() * t.element_size() for (k, _), t in zip(self.w_names, self.weights)
                     if k not in EXPERT_ALIAS)
        logger.info("GLM-5.2 TPxDCP megakernel: %d weight operands (%.2f GiB load-time copies), rank %d/%d",
                    len(self.weights), copied / 2 ** 30, self.rank, R)
        self.caches = None
        self._step = None
        self._probe = None
        self.mla_layer_name = _attn_module(bb.layers[0].self_attn).layer_name
        # ---- G0 debug instruments (debug.py); all off unless their env is set
        self.sched = D.schedule_from_env()
        self.kvdiff = D._env_on("GLM5_MKD_KVDIFF")
        self.idcheck = (D._env_on("GLM5_MKD_IDCHECK") or self.kvdiff
                        or os.environ.get("GLM5_MKD_SPECDBG") == "1"
                        or bool(os.environ.get("GLM5_MKD_MODE")))
        self.dbg_every = int(os.environ.get("GLM5_MKD_DBG_EVERY", "50"))
        self.id_tot = [0, 0]                         # requests checked, requests violating
        self.kv_tot: dict[str, dict] = {}
        self.sh_tot: dict[str, list] = {}
        self.n_dbg = 0
        self.idx_layers = [l for l in range(cfg.L) if cfg.full_indexer(l)]
        self.k_scales = self._kv_scales(bb)
        if self.rank == 0:
            logger.info("GLM-5.2 TPxDCP megakernel debug: mode %s (probe %s) kvdiff %s idcheck %s; "
                        "MLA k_scale values %s (the kernel assumes 1.0: K:1660 decodes fp8 raw)",
                        self.sched.spec, self.sched.needs_probe, self.kvdiff, self.idcheck,
                        sorted(set(self.k_scales)))
        _need(all(v == 1.0 for v in self.k_scales),
              f"MLA k_scale {sorted(set(self.k_scales))} != 1.0 (the megakernel ignores k_scale)")

    # ---------------------------------------------------------------- weights

    def _layer(self, l: int) -> dict[str, torch.Tensor]:
        cfg, r = self.cfg, self.rank
        H = cfg.H
        L = self.backbone.layers[l]
        sa = L.self_attn
        at = _attn_module(sa)
        NQB = cfg.HR * cfg.DQ
        d = {
            "qkv_a": _pad_n(_kn(sa.fused_qkv_a_proj.weight, H, cfg.QKVA, "fused_qkv_a_proj")),
            "q_b": _kn(sa.q_b_proj.weight, cfg.QL, cfg.NH * cfg.DQ, "q_b_proj")[:, r * NQB:(r + 1) * NQB],
            "w_uk": at.W_UK_T,
            "w_uv": at.W_UV,
            "o_proj": _kn(sa.o_proj.weight, cfg.KR, H, "o_proj"),
        }
        idx = getattr(sa, "indexer", None)
        _need((idx is not None) == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        if idx is not None:
            d["idx_wq"] = _kn(idx.wq_b.weight, cfg.QL, cfg.IH * cfg.IDIM, "indexer.wq_b")
            d["idx_wk"] = _pad_n(_kn(idx.wk_weights_proj.weight, H, cfg.IDIM + cfg.IH,
                                     "indexer.wk_weights_proj"))
        mlp = L.mlp
        if cfg.is_moe(l):
            d["gate"] = _kn(mlp.gate.weight, H, cfg.E, "gate")
            rt = _routed(mlp)
            d["ew13"] = rt.w13_weight
            d["ew2"] = rt.w2_weight
            d["es13"] = _expert_scale(rt, "w13_weight", (cfg.ER, 2 * cfg.I))
            d["es2"] = _expert_scale(rt, "w2_weight", (cfg.ER, H))
            se = mlp.shared_experts
            d["sw13"] = _kn(se.gate_up_proj.weight, H, 2 * cfg.ISR, "shared.gate_up_proj")
            d["sw2"] = _kn(se.down_proj.weight, cfg.ISR, H, "shared.down_proj")
        else:
            d["w13"] = _kn(mlp.gate_up_proj.weight, H, 2 * cfg.IDR, "dense.gate_up_proj")
            d["w2"] = _kn(mlp.down_proj.weight, cfg.IDR, H, "dense.down_proj")
        return d

    def _collect_weights(self):
        """Operands in kernel.build's w_names order: one stacked copy per kind in stack_w, else the
        served per-layer tensors."""
        per_layer = {}
        out, names, specs = [], [], []
        for kind, (layers, shape, dtype) in self.wk.items():
            if not layers:                       # a kind no layer has (kernel.build drops it too)
                continue
            want = {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16,
                    "float32": torch.float32}[jax.numpy.dtype(dtype).name]
            ts = []
            for l in layers:
                if l not in per_layer:
                    per_layer[l] = self._layer(l)
                t = per_layer[l][kind]
                _need(tuple(t.shape) == tuple(shape), f"{kind}[{l}]: {tuple(t.shape)} != {shape}")
                _need(t.dtype == want, f"{kind}[{l}]: {t.dtype} != {want}")
                ts.append(t.detach())
            if kind in self.stack_w:
                out.append(_own(torch.stack(ts)))
                names.append((kind, "stk"))
                specs.append(P(None, AXIS))
            else:
                for l, t in zip(layers, ts):
                    out.append(t)
                    names.append((kind, l))
                    specs.append(P(AXIS))
        return out, names, specs

    def _build_smalls(self) -> torch.Tensor:
        cfg, r = self.cfg, self.rank
        lay, S = smalls_layout(cfg)
        rows = []
        for l in range(cfg.L + 1):
            f = self._small_fields(l) if l < cfg.L else {"in_ln": self.backbone.norm.weight}
            dev = next(iter(f.values())).device
            parts = []
            for name, (_, n) in lay.items():
                v = f.get(name)
                v = torch.zeros(n, dtype=torch.float32, device=dev) if v is None else v.reshape(-1).float()
                _need(v.numel() == n, f"smalls {name}[{l}]: {v.numel()} != {n}")
                parts.append(torch.nn.functional.pad(v, (0, rup(n) - n)))
            rows.append(torch.cat(parts))
        return _own(torch.stack(rows).reshape(cfg.L + 1, 1, S))

    def _small_fields(self, l: int) -> dict[str, torch.Tensor]:
        cfg, r = self.cfg, self.rank
        L = self.backbone.layers[l]
        sa = L.self_attn
        NQB = cfg.HR * cfg.DQ
        qkv_s = _scale(sa.fused_qkv_a_proj, cfg.QKVA, "fused_qkv_a_proj")
        f = {
            "in_ln": L.input_layernorm.weight,
            "post_ln": L.post_attention_layernorm.weight,
            "q_a_ln": sa.q_a_layernorm.weight,
            "kv_a_ln": sa.kv_a_layernorm.weight,
            "qkv_a_s": torch.nn.functional.pad(qkv_s, (0, rup(cfg.QKVA) - cfg.QKVA)),
            "q_b_s": _scale(sa.q_b_proj, cfg.NH * cfg.DQ, "q_b_proj")[r * NQB:(r + 1) * NQB],
            "o_proj_s": _scale(sa.o_proj, cfg.H, "o_proj"),
        }
        for norm in (L.input_layernorm, L.post_attention_layernorm, sa.q_a_layernorm, sa.kv_a_layernorm):
            _need(abs(float(getattr(norm, "variance_epsilon", cfg.eps)) - cfg.eps) < 1e-12,
                  f"layer {l}: rmsnorm eps")
        idx = getattr(sa, "indexer", None)
        if idx is not None:
            f["idx_wq_s"] = _scale(idx.wq_b, cfg.IH * cfg.IDIM, "indexer.wq_b")
            f["idx_kn_w"] = idx.k_norm.weight
            f["idx_kn_b"] = idx.k_norm.bias
        mlp = L.mlp
        if cfg.is_moe(l):
            f["gate_bias"] = mlp.gate.e_score_correction_bias.float()
            f["sw13_s"] = _scale(mlp.shared_experts.gate_up_proj, 2 * cfg.ISR, "shared.gate_up_proj")
            f["sw2_s"] = _scale(mlp.shared_experts.down_proj, cfg.H, "shared.down_proj")
        else:
            f["w13_s"] = _scale(mlp.gate_up_proj, 2 * cfg.IDR, "dense.gate_up_proj")
            f["w2_s"] = _scale(mlp.down_proj, cfg.H, "dense.down_proj")
        return f

    # ----------------------------------------------------------------- caches

    def bind_caches(self, runner) -> None:
        """Stacked caches (GLM5_MKD_STACK_C=1: runner.glm5_kv_stack) when the runner allocated them,
        else the served per-layer caches (one operand per (kind, layer): a larger program, ~81k
        bundles vs ~49k, but the stock path is untouched)."""
        cfg = self.cfg
        grp = getattr(runner, "glm5_kv_stack_groups", None) or {}
        if not grp:                               # older runner contract: one tensor per kind
            stk = getattr(runner, "glm5_kv_stack", None) or {}
            grp = {k: [v] for k, v in stk.items()}
        caches = []
        cmap = {}
        if set(grp) == {"nope", "rope", "idx"}:
            nb = int(grp["nope"][0][0].shape[1])
            ck = K.cache_kinds(cfg, nb)
            G = {}
            for kind, (layers, shape, dtype) in ck.items():
                groups = grp[kind]
                gs = len(layers) // len(groups)
                _need(gs * len(groups) == len(layers), f"{kind}: {len(groups)} groups of {len(layers)} layers")
                for gi, (t, names) in enumerate(groups):
                    _need(tuple(t.shape) == (gs,) + tuple(shape),
                          f"stacked cache {kind}[{gi}]: {tuple(t.shape)} != {(gs,) + tuple(shape)}")
                    caches.append(t)
                for i, l in enumerate(layers):           # KVDIFF view: layer -> (group tensor, row)
                    cmap[(kind, l)] = (groups[i // gs][0], i % gs)
                G[kind] = len(groups)
            _need(G["nope"] == G["rope"], "nope / rope grouped differently")
            self.stack_c = G
            self.c_specs = [P(None, AXIS)] * len(caches)
        else:
            _need(not grp, f"partially stacked caches {sorted(grp)}")
            per = {}
            for l in range(cfg.L):
                sa = self.backbone.layers[l].self_attn
                kv = _unwrap(_attn_module(sa).kv_cache)
                _need(isinstance(kv, (tuple, list)) and len(kv) == 2,
                      f"layer {l}: MLA cache is not the (nope, rope) SparseCore pair")
                per[("nope", l)], per[("rope", l)] = kv[0], kv[1]
                idx = getattr(sa, "indexer", None)
                if idx is not None:
                    per[("idx", l)] = _unwrap(idx.k_cache.kv_cache)
            nb = int(per[("nope", 0)].shape[0])
            ck = K.cache_kinds(cfg, nb)
            for kind, (layers, shape, dtype) in ck.items():
                for l in layers:
                    t = per[(kind, l)]
                    _need(tuple(t.shape) == tuple(shape), f"cache {kind}[{l}]: {tuple(t.shape)} != {shape}")
                    caches.append(t)
                    cmap[(kind, l)] = (t, None)
            self.stack_c = frozenset()
            self.c_specs = [P(AXIS)] * len(caches)
        logger.info("GLM-5.2 TPxDCP megakernel: %s caches, %d operands, %d blocks",
                    f"stacked {self.stack_c}" if self.stack_c else "per-layer", len(caches), nb)
        self.caches = caches
        self.nb = nb
        self._cmap = cmap
        self._step_staging = self.kvdiff or self.sched.any_flag("kvdiff")
        self._step = self._build_op(writer=True, staging=self._step_staging)
        if self.sched.needs_probe or self._step_staging:
            # read-only megakernel (no writer, caches not donated) returning its staged rows
            self._probe = self._build_op(writer=False, staging=True)

    def _cache_of(self, kind: str, l: int):
        e = self._cmap.get((kind, l))
        if e is None:
            return None
        t, row = e
        return t if row is None else t[row]

    def _kv_scales(self, bb) -> list[float]:
        """Every backbone MLA layer's k_scale as the stock sparse-MLA op reads it
        (`_get_kv_scales`: `_k_scale_float`, else a 0-d `_k_scale`). The megakernel ignores it."""
        out = []
        for l in range(self.cfg.L):
            at = _attn_module(bb.layers[l].self_attn)
            v = getattr(at, "_k_scale_float", None)
            if v is None:
                t = getattr(at, "_k_scale", None)
                if isinstance(t, torch.Tensor) and t.numel() == 1:
                    v = float(t.detach().float().cpu())
                elif t is not None:
                    v = float(t)
            out.append(1.0 if v is None else float(v))
        return out

    def _build_op(self, writer: bool = True, staging: bool = False):
        """The compiled step. writer=True: megakernel + barrier-free writer (caches donated, updated
        in place: the served path). writer=False: the megakernel alone, caches read-only (a probe
        for the G0 modes). staging=True: also return the staged rows nrow / rrow / irec
        ([L, TMAX, 128] u32, [L, TMAX, 128] u32, [L, TMAX, 256] i32, identical on every rank)."""
        cfg = self.cfg
        build_kw = dict(nb=self.nb, max_pages=self.max_pages, bmax=cfg.TMAX,
                        stack_w=self.stack_w, stack_c=self.stack_c)
        # GLM5_MKD_DEBUG=1: also return the per-layer owner rows (dbg_h) and owned winner lists
        # (dbg_own) per rank -> self.last_dbg (parity diagnostics only)
        self.debug = os.environ.get("GLM5_MKD_DEBUG", "0") == "1"
        call, info = K.build(cfg, axis_name=AXIS, debug=self.debug, dbg_part=False, tiles=self.tiles,
                             **build_kw)
        wfn, winfo = K.build_writer(cfg, axis_name=AXIS, nb=self.nb, max_pages=self.max_pages,
                                    bmax=cfg.TMAX, stack_c=build_kw["stack_c"])
        assert list(info["w_names"]) == self.w_names, "weight operand order"
        assert len(info["c_names"]) == len(self.caches) == len(winfo["c_names"])
        self.info = info
        n_w, n_c = len(self.weights), len(self.caches)
        rep = P()
        in_specs = (rep,) * 7 + (P(AXIS),) + tuple(self.w_specs) + tuple(self.c_specs)
        n_dbg = 2 if self.debug else 0
        n_stg = 3 if staging else 0
        out_specs = ((P(AXIS),) + (tuple(self.c_specs) if writer else ()) + (P(AXIS),) * n_stg
                     + (P(AXIS),) * n_dbg)

        def body(*a):
            scal, rmeta, qmeta, bt, rvec, h0, tab, smalls = a[:8]
            caches = a[8 + n_w:]
            outs = call(*a)
            out, nrow, rrow, irec = outs[:4]
            res = [out]
            if writer:
                res += list(wfn(scal, rmeta, bt, nrow, rrow, irec, *caches))
            if staging:
                res += [nrow, rrow, irec]
            return tuple(res + list(outs[4:4 + n_dbg]))

        smap = jax.shard_map(body, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
                             check_vma=False)
        n_in = 8 + n_w + n_c
        args = [f"x{i}" for i in range(n_in)]
        ret = "tuple[" + ", ".join(["jax.Array"] * len(out_specs)) + "]"
        src = (f"def glm5_mkd({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
               f"    return _smap({', '.join(args)})\n")
        ns: dict[str, Any] = {"jax": jax, "_smap": smap}
        exec(compile(src, "<glm5_mkd>", "exec"), ns)  # noqa: S102 - arity is data
        name = "glm5_mkd::" + ("step" if writer else "probe") + ("_stg" if staging else "")
        # unregistered callable: ~175 operands exceed a torch op schema's 64 (j-a67fe755); the op
        # below takes the weights and caches as two Tensor[] arguments
        flat = sharded_jax_op(name + "_flat", ns["glm5_mkd"], mesh=self.mesh,
                              input_partition_specs=in_specs, output_partition_specs=out_specs,
                              donate_argnums=tuple(range(n_in - n_c, n_in)) if writer else (),
                              register=False)

        @torch.library.custom_op(name, mutates_args=())
        def op(scal: torch.Tensor, rmeta: torch.Tensor, qmeta: torch.Tensor, bt: torch.Tensor,
               rvec: torch.Tensor, h0: torch.Tensor, tab: torch.Tensor, smalls: torch.Tensor,
               weights: list[torch.Tensor], caches: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(flat(scal, rmeta, qmeta, bt, rvec, h0, tab, smalls, *weights, *caches))

        @op.register_fake
        def _(scal, rmeta, qmeta, bt, rvec, h0, tab, smalls, weights, caches):
            M = info["M"]
            dev = h0.device
            stg = ([torch.empty((cfg.L, cfg.TMAX, 128), dtype=torch.uint32, device=dev),
                    torch.empty((cfg.L, cfg.TMAX, 128), dtype=torch.uint32, device=dev),
                    torch.empty((cfg.L, cfg.TMAX, 256), dtype=torch.int32, device=dev)]
                   if staging else [])
            dbg = ([torch.empty((cfg.L, 2, cfg.RPR, cfg.H), dtype=torch.float32, device=dev),
                    torch.empty((cfg.L, cfg.TMAX, cfg.KT // 128, 128), dtype=torch.int32, device=dev)]
                   if n_dbg else [])
            return ([torch.empty((M, cfg.H), dtype=torch.bfloat16, device=dev)]
                    + ([torch.empty_like(c) for c in caches] if writer else []) + stg + dbg)

        TMAX, M, DR = cfg.TMAX, info["M"], cfg.DR
        nb = int(self.nb)
        BMAX = cfg.TMAX
        max_pages = self.max_pages

        def step(ids, pos, qsl, seq_lens, bt_flat, cos_sin, smalls, weights, caches):
            dev = ids.device
            i32 = torch.int32
            nq_tab = qsl.shape[0] - 1
            bt2 = bt_flat.reshape(nq_tab, -1).to(i32)
            # pad / truncate the request axis to BMAX and the page axis to max_pages
            def fit(x, n, axis=0):
                k = x.shape[axis]
                if k >= n:
                    return x.narrow(axis, 0, n)
                pad = [0, 0] * (x.dim() - 1 - axis) + [0, n - k]
                return torch.nn.functional.pad(x, pad)
            qsl_ = qsl.to(i32)                        # padded entries repeat the total (runner)
            last = qsl_[-1:]
            qsl_b = torch.cat([qsl_, last.expand(BMAX + 1)])[: BMAX + 1]
            T = qsl_[-1]
            seq_b = fit(seq_lens.to(i32), BMAX)
            # DCP block tables hold VIRTUAL page ordinals; every main reader / writer resolves them on
            # this rank as `virtual % num_pages` (kernels/mla/kv_cache_utils.py:583, :713). Unresolved,
            # ids >= nb made the megakernel read (and its writer write) out of bounds -- the
            # real-weight parity failure (bounds checks are off in the kernel)
            bt_b = torch.remainder(fit(fit(bt2, BMAX, 0), max_pages, 1), nb)
            t = torch.arange(TMAX, device=dev, dtype=i32)
            valid = (t < T).to(i32)
            req = (qsl_b[1:BMAX + 1][None, :] <= t[:, None]).sum(1).to(i32)
            req = torch.clamp(req, max=BMAX - 1)
            p = fit(pos.to(i32), TMAX) * valid
            seq_t = seq_b[req] * valid
            z = torch.zeros(TMAX, dtype=i32, device=dev)
            rmeta = torch.stack([valid, p, req * valid, seq_t, z, z, z, z], 1)
            t0 = qsl_b[:BMAX]
            nq = qsl_b[1:BMAX + 1] - t0
            qmeta = torch.stack([t0, nq, seq_b, (nq > 0).to(i32)], 1)
            B = (nq > 0).sum().to(i32)
            scal = torch.cat([torch.stack([T, B]), torch.zeros(6, dtype=i32, device=dev)])
            rvec = torch.zeros((M, 128), dtype=i32, device=dev)
            rvec[:TMAX, 0] = valid
            rvec[:TMAX, 1] = p
            # rope tables: GPT-J pairs in lanes 0..DR-1 (cos on both, -sin / +sin), rolled to
            # lanes 128-DR..127; identity elsewhere; bf16 values (main's cos/sin cache dtype)
            cs = cos_sin[p.long()].to(torch.bfloat16).float()                 # [TMAX, DR]
            c = cs[:, : DR // 2].repeat_interleave(2, dim=1)
            s = torch.stack([-cs[:, DR // 2:], cs[:, DR // 2:]], -1).reshape(TMAX, DR)
            ones = torch.ones((TMAX, 128 - DR), dtype=torch.float32, device=dev)
            zeros = torch.zeros((TMAX, 128 - DR), dtype=torch.float32, device=dev)
            cos_lo, sin_lo = torch.cat([c, ones], 1), torch.cat([s, zeros], 1)
            cos_hi, sin_hi = torch.cat([ones, c], 1), torch.cat([zeros, s], 1)
            tab = torch.stack([cos_lo, sin_lo, cos_hi, sin_hi])
            tab = torch.nn.functional.pad(tab, (0, 0, 0, M - TMAX), value=0.0)
            tab[:, TMAX:, :] = torch.tensor([1.0, 0.0, 1.0, 0.0], device=dev)[:, None, None]
            h0 = self.backbone.embed_tokens(fit(ids, TMAX)).float()
            outs = op(scal, rmeta, qmeta, bt_b, rvec, h0, tab, smalls, list(weights), list(caches))
            k = 1
            if writer:
                for dst, src in zip(caches, outs[1:1 + len(caches)]):
                    dst.copy_(src)
                k += len(caches)
            # what the kernel was handed (rows: valid, pos, req, seq; requests: t0, nq, seq, valid),
            # for the in-graph metadata check (debug; two small extra outputs)
            return (outs[0][:TMAX], rmeta[:, :4], qmeta, *outs[k:])

        return torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)

    # ------------------------------------------------------------------- run

    def step_plan(self, runner, scheduler_output, num_chunks: int) -> "D.Plan | None":
        """Rank-uniform host decision for this step: None (not eligible) or the `debug.Plan` the
        runner's forward_model executes (GLM5_MKD_MODE; default `mk`). A `pure_stock` plan runs the
        stock forward alone; it is kept (not None) so SPECDBG can label it (alt:k/stock, cycle arms)."""
        self.sched.observe_requests(scheduler_output.num_scheduled_tokens.keys())
        ok, why = self._step_ok(runner, scheduler_output, num_chunks)
        plan = self.sched.plan() if ok else None
        if plan is not None:
            why = plan.label
        # engagement counters (rank 0 logs every 500 steps): steps per plan label, decode-only steps
        # the megakernel declined (too many tokens), and steps with a prefill (never taken)
        c = self.counts
        c[why] = c.get(why, 0) + 1
        c["n"] = c.get("n", 0) + 1
        if c["n"] % 500 == 0 and self.rank == 0:
            logger.info("GLM-5.2 TPxDCP megakernel engagement: %s", {k: v for k, v in c.items() if k != "n"})
        return plan

    def step_ok(self, runner, scheduler_output, num_chunks: int) -> bool:
        plan = self.step_plan(runner, scheduler_output, num_chunks)
        return plan is not None and not plan.pure_stock

    def _step_ok(self, runner, scheduler_output, num_chunks: int):
        cfg = self.cfg
        if self._step is None or num_chunks != 1 or scheduler_output.scheduled_encoder_inputs:
            return False, "other"
        ns = scheduler_output.num_scheduled_tokens
        ib = runner.input_batch
        for req_id, n in ns.items():
            i = ib.req_id_to_index[req_id]
            if n > cfg.QMAX or int(ib.num_computed_tokens_cpu[i]) < int(ib.num_prompt_tokens[i]):
                return False, "prefill"
        if scheduler_output.total_num_scheduled_tokens > cfg.TMAX or len(ns) > cfg.TMAX:
            return False, "decode_over_tmax"
        return True, "mk"

    def _metadata(self) -> AttentionMetadata:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(self.mla_layer_name)
        assert isinstance(md, AttentionMetadata), type(md)
        return md

    def _run(self, fn, md, input_ids, positions):
        """One compiled call -> (out [n, H] padded like the runner's bucket, staging or None)."""
        res = fn(input_ids, positions, md.query_start_loc, md.seq_lens, md.block_tables,
                 self.cos_sin, self.smalls, self.weights, self.caches)
        out = res[0]
        self._seen = (res[1], res[2])                # rmeta[:, :4], qmeta as consumed by the kernel
        res = (out, *res[3:])
        has = fn is self._probe or (fn is self._step and self._step_staging)
        stg = tuple(res[1:4]) if has and len(res) >= 4 else None
        rest = res[4:] if stg is not None else res[1:]
        if len(rest) > 0:
            self.last_dbg = tuple(rest)
        n = input_ids.shape[0]
        out = out[:n] if n <= out.shape[0] else torch.nn.functional.pad(out, (0, 0, 0, n - out.shape[0]))
        return out, stg

    @torch.no_grad()
    def __call__(self, runner, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The served path (plan `mk`): megakernel + writer."""
        if self._step is None:               # capture_model was skipped (e.g. a microbenchmark)
            self.bind_caches(runner)
        md = self._metadata()
        plan = getattr(runner, "_mkd_plan", None)
        kv = self._kv_on(plan)
        coords = snap = None
        if kv:
            coords, snap = self._dbg_safe(lambda: self._coords_snapshot(md, positions), (None, None))
        out, stg = self._run(self._step, md, input_ids, positions)
        if kv and stg is not None and coords is not None:
            # after the megakernel's own writer: placement must be byte-exact, neighbours untouched
            self._dbg_safe(lambda: self._kv_report("writer", coords, stg, snap))
        self._after_step(md, positions, self._id_on(plan))
        return out

    @torch.no_grad()
    def aux(self, runner, plan, input_ids: torch.Tensor, positions: torch.Tensor,
            stock_out: torch.Tensor) -> torch.Tensor:
        """A mixed plan, after the stock forward ran (it wrote its KV for this step's rows): run the
        megakernel with its writer (plan.kv == 'M': overwrites the owned rows with its own bytes) or
        read-only (plan.kv == 'S'), then hand back the hidden the plan selects; the drafter's hidden
        goes through runner._mkd_draft_hidden."""
        md = self._metadata()
        kv = self._kv_on(plan)
        if plan.mk_writer:
            coords = snap = None
            if kv:
                coords, snap = self._dbg_safe(lambda: self._coords_snapshot(md, positions), (None, None))
            out, stg = self._run(self._step, md, input_ids, positions)
            if kv and stg is not None and coords is not None:
                self._dbg_safe(lambda: self._kv_report("writer", coords, stg, snap))
        else:
            out, stg = self._run(self._probe, md, input_ids, positions)
            if kv and stg is not None:
                # the caches hold what the STOCK forward just wrote for these rows: numerics/format
                self._dbg_safe(lambda: self._kv_report(
                    "numerics", self._coords_snapshot(md, positions, snap=False)[0], stg, None))
        if plan.stats:
            self._dbg_safe(lambda: self._shadow(runner, plan, md, out, stock_out))
        self._after_step(md, positions, self._id_on(plan))
        ret = out if plan.ret == "M" else stock_out
        drafted = out if plan.draft == "M" else stock_out
        runner._mkd_draft_hidden = None if plan.draft == plan.ret else drafted
        return ret

    # ------------------------------------------------------------- debug helpers

    def _kv_on(self, plan) -> bool:
        return self.kvdiff if plan is None or plan.kvdiff is None else plan.kvdiff

    def _id_on(self, plan) -> bool:
        return self.idcheck if plan is None or plan.idcheck is None else plan.idcheck

    def _dbg_safe(self, fn, default=None):
        try:
            return fn()
        except Exception as e:                   # noqa: BLE001 - debug must never take serving down
            logger.warning("GLM-5.2 TPxDCP megakernel debug failed: %r", e)
            return default

    def _coords_snapshot(self, md, positions, snap: bool = True):
        cfg = self.cfg
        TMAX = cfg.TMAX
        qsl = md.query_start_loc.to(torch.int64)
        T = qsl[-1]
        npos = min(TMAX, positions.shape[0])
        pos = torch.zeros(TMAX, dtype=torch.int64, device=positions.device)
        pos[:npos] = positions[:npos].to(torch.int64)
        t = torch.arange(TMAX, device=positions.device)
        valid = t < T
        req = (qsl[1:][None, :] <= t[:, None]).sum(1)
        nq_tab = md.query_start_loc.shape[0] - 1
        bt2 = md.block_tables.reshape(nq_tab, -1)
        req = req.clamp(max=nq_tab - 1)
        c = D.owned_row_coords(pos, req, valid, bt2, self.nb, cfg.R, cfg.ILV, cfg.page, self.rank)
        s = D.snapshot(self._cache_of, cfg.L, self.idx_layers, c) if snap else None
        return c, s

    def _kv_report(self, kind: str, coords, stg, snap) -> None:
        nrow, rrow, irec = stg
        res = D.compare_staging(self._cache_of, self.cfg.L, self.idx_layers, coords, nrow, rrow,
                                irec, self.cfg.IDIM, self.cfg.IDIM + 1, neighbors=snap)
        tot = self.kv_tot.setdefault(kind, {})
        for k, r in res.items():
            a = tot.setdefault(k, {})
            for f, v in r.items():
                a[f] = max(a.get(f, 0), v) if f == "max_abs" else a.get(f, 0) + v
        bad = any(r.get("bad_rows", 0) for k, r in res.items() if k != "clobber") or any(
            res.get("clobber", {}).values())
        if bad and kind == "writer":
            logger.warning("KVDIFF %s rank %d step mismatch: %s", kind, self.rank, D.fmt_kvdiff(res))

    def _shadow(self, runner, plan, md, out_mk, out_stock) -> None:
        T = int(md.query_start_loc[-1].cpu())
        if T <= 0:
            return
        lm = runner.model.compute_logits(out_mk[:T])
        ls = runner.model.compute_logits(out_stock[:T])
        st = D.shadow_stats(lm, ls, out_mk[:T], out_stock[:T])
        a = self.sh_tot.setdefault(plan.label, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a[0] += 1
        a[1] += st["kl_mean"]
        a[2] = max(a[2], st["kl_max"])
        a[3] += st["argmax_agree"]
        a[4] = max(a[4], st["dh_max"])
        a[5] = max(a[5], st["dh_rel_max"])

    def _after_step(self, md, positions, idcheck: bool) -> None:
        self.n_dbg += 1
        if self.sched.spec != "mk" or self.idcheck:
            # in-graph check, batched: keep device refs, sync once per dbg_every steps
            q = self.__dict__.setdefault("_seen_q", [])
            q.append((self._seen[0], self._seen[1], positions, md.seq_lens, md.query_start_loc))
            if len(q) >= self.dbg_every:
                self._dbg_safe(self._seen_check)
                q.clear()
        if idcheck:
            def chk():
                qsl = md.query_start_loc.cpu().numpy()
                seq = md.seq_lens.cpu().numpy()
                pos = positions[: int(qsl[-1])].cpu().numpy()
                n, bad, ex = D.metadata_identity(qsl, seq, pos)
                self.id_tot[0] += n
                self.id_tot[1] += bad
                if bad:
                    logger.warning("IDCHECK rank %d: p0 = seq_len - nq != position_ids[t0] on %d of %d "
                                   "requests: %s", self.rank, bad, n, ex)
            self._dbg_safe(chk)
        if self.n_dbg % self.dbg_every == 0:
            if self.id_tot[0] and self.rank == 0:
                logger.info("IDCHECK steps %d requests %d violations %d", self.n_dbg, *self.id_tot)
            for kind, tot in self.kv_tot.items():
                logger.info("KVDIFF %s rank %d steps %d: %s", kind, self.rank, self.n_dbg,
                            D.fmt_kvdiff(tot))
            if self.rank == 0:
                for lab, a in self.sh_tot.items():
                    logger.info("SHADOW %s steps %d mean KL(p_mk||p_stock) %.4g max %.4g argmax "
                                "agree %.4f max|dh| %.4g max rel|dh| %.4g", lab, a[0], a[1] / a[0],
                                a[2], a[3] / a[0], a[4], a[5])

    def _seen_check(self) -> None:
        """For every queued step: the rows the KERNEL got (rmeta pos / seq, qmeta seq / nq, built
        inside the compiled step) must satisfy p0 = seq - nq == pos[t0] (the identity, (b)) and must
        equal the runner's corrected position_ids / seq_lens read now. A mismatch with the second
        while the first holds would mean the step consumed its inputs before an async correction
        (subtract_num_rejected_tokens) landed: an ordering hazard, invisible to a host read."""
        n_id = n_ord = n_tot = 0
        ex = []
        for rmeta, qmeta, pos, seq, qsl in self._seen_q:
            rm = rmeta.cpu().numpy()
            qm = qmeta.cpu().numpy()
            qs = qsl.cpu().numpy()
            T = int(qs[-1])
            ps = pos[:T].cpu().numpy()
            sq = seq.cpu().numpy()
            n_tot += 1
            bad_id = bad_ord = False
            for b in range(qm.shape[0]):
                t0, nq, s_k, v = (int(x) for x in qm[b])
                if not v:
                    continue
                if any(int(rm[t0 + j, 1]) != s_k - nq + j for j in range(nq)):
                    bad_id = True
                if b < len(sq) and int(sq[b]) != s_k:
                    bad_ord = True
            if T and (rm[:T, 1] != ps).any():
                bad_ord = True
            n_id += bad_id
            n_ord += bad_ord
            if (bad_id or bad_ord) and len(ex) < 2:
                ex.append(dict(kernel_pos=rm[:T, 1].tolist(), runner_pos=ps.tolist(),
                               kernel_seq=qm[:, 2][qm[:, 3] > 0].tolist(), runner_seq=sq[:4].tolist()))
        if n_id or n_ord or self.rank == 0:
            logger.info("SEENCHECK rank %d: %d steps, identity violations %d, kernel-vs-runner "
                        "mismatches %d %s", self.rank, n_tot, n_id, n_ord, ex if ex else "")

    def warmup(self, runner) -> None:
        """Compile and run once: one request of QMAX rows at positions 0..QMAX-1 on block 0 (the
        null block; nothing is served from it before the first step)."""
        from vllm_torchtpu.utils import synchronize_tensors

        cfg = self.cfg
        dev = runner.device
        T = cfg.QMAX
        ids = torch.zeros(cfg.TMAX, dtype=torch.int32, device=dev)
        pos = torch.arange(cfg.TMAX, dtype=torch.int32, device=dev)
        qsl = torch.tensor([0] + [T] * cfg.TMAX, dtype=torch.int32, device=dev)
        seq = torch.tensor([T] + [0] * (cfg.TMAX - 1), dtype=torch.int32, device=dev)
        bt = torch.zeros(cfg.TMAX * self.max_pages, dtype=torch.int32, device=dev)
        for fn, what in ((self._step, "step"), (self._probe, "probe")):
            if fn is None:
                continue
            t0 = time.monotonic()
            out = fn(ids, pos, qsl, seq, bt, self.cos_sin, self.smalls, self.weights, self.caches)[0]
            synchronize_tensors([out])
            logger.info("GLM-5.2 TPxDCP megakernel %s compiled and warmed in %.1f s (%d weight operands)",
                        what, time.monotonic() - t0, len(self.weights))
