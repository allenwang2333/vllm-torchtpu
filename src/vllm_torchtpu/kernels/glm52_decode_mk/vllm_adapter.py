# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Serving glue for the stage-A GLM-5.2 decode megakernel (kernels/glm52_decode_mk). UNWIRED in stage A.

Port of glm5_mk_dcp's adapter (vllm_adapter.py @ d41e70f6d) to this kernel's operand plan and deployment:
TP = DCP = world, DP 1, **EP off** (TP16 MoE: the served routed experts are main's moe_tp_decode TP layout
w13 [E, H, 2I/R], w2 [E, I/R, H], s13 [E, 1, 1, 2I/R], s2 [E, 1, 1, H], passed per layer, zero-copy), **qrep 0**
(the served q_b_proj is column-parallel: this rank's HR heads and their scale slice are taken as they are; a
replicated q_b (qrep 1) is sliced to this rank's heads), MTP verify rows. Enabled by GLM52_DECODE_MK=1.

The runner hook (TPUModelRunner calls `maybe_create` at load_model, `bind_caches` at capture_model, `step_ok`
per scheduled step and `__call__` in place of the backbone) lives on the glm5_mk_dcp branch's tpu_runner.py; it is
not on main and is not part of stage A (WHITEBOARD.v1 §13: A3 builds the twin harness extension first). This
module is import-clean and its host-side metadata / rope packing is the same arithmetic as wrapper.metadata /
wrapper.rope_table (the instruments-tested path); the torch operand collection is untested on hardware.
"""

import dataclasses
import os

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

from . import kernel as K
from . import layout as LY
from .config import FULL, rup
from .writer import build_writer

logger = init_logger(__name__)

AXIS = "mk52"


class Unsupported(Exception):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(why)


def enabled() -> bool:
    return os.environ.get("GLM52_DECODE_MK", "0") == "1"


def _unwrap(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def _kn(weight: torch.Tensor, k: int, n: int, what: str) -> torch.Tensor:
    if tuple(weight.shape) == (k, n):
        return weight
    _need(tuple(weight.shape) == (n, k), f"{what}: shape {tuple(weight.shape)}, want ({k}, {n})")
    return weight.t().contiguous()


def _pad_n(w: torch.Tensor, n: int) -> torch.Tensor:
    k = w.shape[-1]
    return torch.nn.functional.pad(w, (0, n - k)) if k != n else w


def _scale(layer, n: int, what: str) -> torch.Tensor:
    for name in ("weight_scale", "weight_scale_inv"):
        s = getattr(layer, name, None)
        if s is not None:
            _need(s.numel() == n, f"{what}.{name}: {tuple(s.shape)} is not per-channel [{n}]")
            return s.reshape(-1).float()
    raise Unsupported(f"{what}: no weight_scale")


def _own(t: torch.Tensor) -> torch.Tensor:
    """An operand with storage of exactly its own shape: a view operand is re-materialized by torch_tpu on every
    call (2.2 ms/step of copies, j-725a55a4)."""
    out = torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


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


def _routed(mlp):
    exp = mlp.experts
    return getattr(exp, "routed_experts", exp)


def q_b_local(weight_kn: torch.Tensor, scale: torch.Tensor, cfg, rank: int):
    """qrep 0: the served q_b_proj holds this rank's HR heads ([QL, HR*DQ], scale [HR*DQ]) -> as is; a replicated
    q_b (qrep 1: [QL, NH*DQ]) -> this rank's columns. The glm5_mk_dcp adapter declined qrep 0 (AD:159)."""
    nqb = cfg.HR * cfg.DQ
    n = weight_kn.shape[1]
    if n == nqb:
        _need(scale.numel() == nqb, f"q_b scale {tuple(scale.shape)} != local [{nqb}]")
        return weight_kn, scale
    _need(n == cfg.NH * cfg.DQ, f"q_b_proj N {n}: neither local {nqb} nor replicated {cfg.NH * cfg.DQ}")
    return weight_kn[:, rank * nqb:(rank + 1) * nqb], scale.reshape(-1)[rank * nqb:(rank + 1) * nqb]


class Glm52DecodeMK:
    """Built at load_model (weights), bound to the caches at capture_model."""

    @classmethod
    def maybe_create(cls, runner) -> "Glm52DecodeMK | None":
        if not enabled():
            return None
        try:
            return cls(runner)
        except Unsupported as e:
            logger.warning("GLM-5.2 decode megakernel disabled: %s", e)
            return None

    def __init__(self, runner):
        import torch.distributed as dist

        hf = runner.model_config.hf_config
        pc = runner.parallel_config
        _need(type(hf).__name__.startswith("GlmMoeDsa") or getattr(hf, "model_type", "") == "glm_moe_dsa",
              f"model_type {hf.model_type}")
        _need(pc.pipeline_parallel_size == 1, "pipeline parallel")
        _need(getattr(pc, "data_parallel_size", 1) == 1, "data parallel")
        R = int(pc.tensor_parallel_size)
        _need(int(getattr(pc, "decode_context_parallel_size", 1)) == R, "DCP != TP")
        _need(dist.is_initialized() and dist.get_world_size() == R, "world != TP")
        _need(int(getattr(pc, "cp_kv_cache_interleave_size", 1)) == FULL.ILV, "interleave != 4")
        _need(not bool(getattr(pc, "enable_expert_parallel", False)), "EP is on (stage A is TP16 MoE, EP off)")
        spec = runner.speculative_config
        _need(spec is not None and getattr(spec, "method", "") == "mtp", "MTP is off")
        qmax = 1 + int(spec.num_speculative_tokens)
        _need(qmax <= 4, f"{qmax} rows per request")
        _need(getattr(hf, "scoring_func", None) == "sigmoid", "router scoring")
        _need(getattr(hf, "n_group", 1) == 1 and getattr(hf, "topk_group", 1) == 1, "grouped routing")
        _need(bool(getattr(hf, "norm_topk_prob", True)), "unnormalized top-k")
        _need(bool(getattr(hf, "rope_interleave", False)) and bool(getattr(hf, "indexer_rope_interleave", False)),
              "non-interleaved rope")
        rp = getattr(hf, "rope_parameters", None) or {}
        _need(rp.get("rope_type", "default") == "default", f"rope {rp}")
        _need(runner.block_size == FULL.page, f"block_size {runner.block_size}")
        tmax = int(os.environ.get("GLM52_DECODE_MK_ROWS", "16"))   # stage-A scope 16 (WHITEBOARD.v1 §16)
        _need(tmax in (16, 32), f"GLM52_DECODE_MK_ROWS={tmax}")
        cfg = dataclasses.replace(
            FULL, R=R, L=hf.num_hidden_layers, first_dense=hf.first_k_dense_replace, H=hf.hidden_size,
            NH=hf.num_attention_heads, QL=hf.q_lora_rank, idx_freq=hf.index_topk_freq,
            idx_skip=hf.index_skip_topk_offset, eps=hf.rms_norm_eps, routed_scale=float(hf.routed_scaling_factor),
            QMAX=qmax, TMAX=tmax, BM=int(runner.scheduler_config.max_num_seqs))
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
        try:
            cfg.check()
        except AssertionError as e:
            raise Unsupported(f"admission: {e}") from e
        self.cfg = cfg
        self.rank = int(dist.get_rank())
        self.counts = {}
        from torch_tpu._internal.pallas import get_pallas_mesh
        self.mesh = get_pallas_mesh(axis_names=(AXIS,), mesh_shape=(R,))
        bb = _find_backbone(runner.model)
        self.backbone = bb
        _need(len(bb.layers) == cfg.L, f"{len(bb.layers)} layers")
        mml = int(runner.model_config.max_model_len)
        self.max_pages = -(-mml // (cfg.page * R))
        rot = bb.layers[0].self_attn.rotary_emb
        _need(rot.cos_sin_cache.shape[-1] == cfg.DR, "rope cache width")
        with torch.no_grad():
            self.cos_sin = _own(rot.cos_sin_cache.detach())
            self.k_scales = [float(getattr(_attn_module(bb.layers[l].self_attn), "_k_scale_float", 1.0))
                             for l in range(cfg.L)]
            self.dense, self.experts = self._collect_weights()
            self.smalls = self._build_smalls().detach()
        self.tiles = K.Tiles()
        self.caches = None
        self._step = None
        self.mla_layer_name = _attn_module(bb.layers[0].self_attn).layer_name

    # ---------------------------------------------------------------- weights

    def _layer(self, l: int) -> dict:
        cfg, r = self.cfg, self.rank
        H = cfg.H
        Ly = self.backbone.layers[l]
        sa = Ly.self_attn
        at = _attn_module(sa)
        qb_w, qb_s = q_b_local(_kn(sa.q_b_proj.weight, cfg.QL, sa.q_b_proj.weight.numel() // cfg.QL, "q_b_proj"),
                               _scale(sa.q_b_proj, sa.q_b_proj.weight.numel() // cfg.QL, "q_b_proj"), cfg, r)
        d = {"qkva": _pad_n(_kn(sa.fused_qkv_a_proj.weight, H, cfg.QKVA, "fused_qkv_a_proj"), cfg.QKVP),
             "qb": qb_w, "q_b_s": qb_s, "wuk": at.W_UK_T, "wuv": at.W_UV,
             "wo": _kn(sa.o_proj.weight, cfg.KR, H, "o_proj")}
        idx = getattr(sa, "indexer", None)
        _need((idx is not None) == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        if idx is not None:
            d["iwq"] = _kn(idx.wq_b.weight, cfg.QL, cfg.IH * cfg.IDIM, "indexer.wq_b")
            d["iwk"] = _pad_n(_kn(idx.wk_weights_proj.weight, H, cfg.IDIM + cfg.IH, "indexer.wk_weights_proj"),
                              cfg.IKW)
        mlp = Ly.mlp
        if cfg.is_moe(l):
            d["gate"] = _pad_n(_kn(mlp.gate.weight, H, cfg.E, "gate"), cfg.EPAD)
            rt = _routed(mlp)
            d["ew13"], d["ew2"] = rt.w13_weight, rt.w2_weight
            d["es13"], d["es2"] = rt.w13_weight_scale_inv, rt.w2_weight_scale_inv
            for k_, shp in (("ew13", (cfg.E, H, 2 * cfg.IR)), ("ew2", (cfg.E, cfg.IR, H)),
                            ("es13", (cfg.E, 1, 1, 2 * cfg.IR)), ("es2", (cfg.E, 1, 1, H))):
                _need(tuple(d[k_].shape) == shp, f"layer {l} {k_}: {tuple(d[k_].shape)} != {shp} (TP16 layout)")
            se = mlp.shared_experts
            d["sh13"] = _kn(se.gate_up_proj.weight, H, 2 * cfg.ISR, "shared.gate_up_proj")
            d["sh2"] = _kn(se.down_proj.weight, cfg.ISR, H, "shared.down_proj")
        else:
            d["dw13"] = _kn(mlp.gate_up_proj.weight, H, 2 * cfg.IDR, "dense.gate_up_proj")
            d["dw2"] = _kn(mlp.down_proj.weight, cfg.IDR, H, "dense.down_proj")
        return d

    def _collect_weights(self):
        cfg = self.cfg
        per = {l: self._layer(l) for l in range(cfg.L)}
        self._per_layer = per
        dsh = LY.dense_shapes(cfg)
        dense = []
        for k in LY.DKINDS:
            ls = LY.layers_of(cfg, k)
            if not ls:
                continue
            ts = [per[l][k].detach() for l in ls]
            for l, t in zip(ls, ts):
                _need(tuple(t.shape) == tuple(dsh[k][0]), f"{k}[{l}]: {tuple(t.shape)} != {dsh[k][0]}")
            dense.append(_own(torch.stack(ts)))                       # load-time stacked copy
        experts = []
        for l in LY.layers_of(cfg, "gate"):                           # served tensors, per layer, zero-copy
            experts += [per[l]["ew13"], per[l]["ew2"], per[l]["es13"], per[l]["es2"]]
        return dense, experts

    def _build_smalls(self) -> torch.Tensor:
        cfg = self.cfg
        lay, S = LY.smalls_layout(cfg)
        rows = []
        for l in range(cfg.L + 1):
            f = self._small_fields(l) if l < cfg.L else {"in_ln": self.backbone.norm.weight}
            dev = next(iter(f.values())).device
            parts = []
            for name, (_, n) in lay.items():
                v = f.get(name)
                v = torch.zeros(n, dtype=torch.float32, device=dev) if v is None else v.reshape(-1).float()
                if name == "k_scale":
                    v = v[:1].expand(n)
                _need(v.numel() == n, f"smalls {name}[{l}]: {v.numel()} != {n}")
                parts.append(torch.nn.functional.pad(v, (0, rup(n) - n)))
            rows.append(torch.cat(parts))
        return _own(torch.stack(rows).reshape(cfg.L + 1, 1, S))

    def _small_fields(self, l: int) -> dict:
        cfg = self.cfg
        Ly = self.backbone.layers[l]
        sa = Ly.self_attn
        dev = Ly.input_layernorm.weight.device
        f = {"in_ln": Ly.input_layernorm.weight, "post_ln": Ly.post_attention_layernorm.weight,
             "q_a_ln": sa.q_a_layernorm.weight, "kv_a_ln": sa.kv_a_layernorm.weight,
             "qkv_a_s": torch.nn.functional.pad(_scale(sa.fused_qkv_a_proj, cfg.QKVA, "fused_qkv_a_proj"),
                                                (0, cfg.QKVP - cfg.QKVA)),
             "q_b_s": self._per_layer[l]["q_b_s"], "o_proj_s": _scale(sa.o_proj, cfg.H, "o_proj"),
             "k_scale": torch.full((1,), self.k_scales[l], dtype=torch.float32, device=dev)}
        idx = getattr(sa, "indexer", None)
        if idx is not None:
            f["idx_wq_s"] = _scale(idx.wq_b, cfg.IH * cfg.IDIM, "indexer.wq_b")
            f["idx_kn_w"] = idx.k_norm.weight
            f["idx_kn_b"] = idx.k_norm.bias
        mlp = Ly.mlp
        if cfg.is_moe(l):
            gb = torch.full((cfg.EPAD,), -1e30, dtype=torch.float32, device=dev)
            gb[:cfg.E] = mlp.gate.e_score_correction_bias.float()
            f["gate_bias"] = gb
            f["sw13_s"] = _scale(mlp.shared_experts.gate_up_proj, 2 * cfg.ISR, "shared.gate_up_proj")
            f["sw2_s"] = _scale(mlp.shared_experts.down_proj, cfg.H, "shared.down_proj")
        else:
            f["w13_s"] = _scale(mlp.gate_up_proj, 2 * cfg.IDR, "dense.gate_up_proj")
            f["w2_s"] = _scale(mlp.down_proj, cfg.H, "dense.down_proj")
        return f

    # ----------------------------------------------------------------- caches

    def bind_caches(self, runner) -> None:
        """Stacked cache groups (runner.glm5_kv_stack_groups, mk2 / glm5_mk_dcp runner contract) when allocated,
        else the served per-layer caches (one group per layer)."""
        cfg = self.cfg
        grp = getattr(runner, "glm5_kv_stack_groups", None) or {}
        caches, G = [], {}
        if set(grp) == {"nope", "rope", "idx"}:
            nb = int(grp["nope"][0][0].shape[1])
            nbi = int(grp["idx"][0][0].shape[1])
            for kind in ("nope", "rope", "idx"):
                G[kind] = len(grp[kind])
                caches += [t for t, _names in grp[kind]]
        else:
            per = {}
            for l in range(cfg.L):
                sa = self.backbone.layers[l].self_attn
                kv = _unwrap(_attn_module(sa).kv_cache)
                _need(isinstance(kv, (tuple, list)) and len(kv) == 2, f"layer {l}: MLA cache is not (nope, rope)")
                per[("nope", l)], per[("rope", l)] = kv[0], kv[1]
                idx = getattr(sa, "indexer", None)
                if idx is not None:
                    per[("idx", l)] = _unwrap(idx.k_cache.kv_cache)
            nb = int(per[("nope", 0)].shape[0])
            nbi = int(per[("idx", 0)].shape[0])
            for kind in ("nope", "rope", "idx"):
                ls = LY.layers_of(cfg, "iwq") if kind == "idx" else list(range(cfg.L))
                G[kind] = len(ls)
                caches += [per[(kind, l)][None] for l in ls]      # [1, ...] groups of one layer
        ck = LY.cache_kinds(cfg, nb, nbi)
        for (kind, gi), t in zip(K.cache_names(ck, G), caches):
            gs = len(ck[kind][0]) // G[kind]
            _need(tuple(t.shape) == (gs,) + tuple(ck[kind][1]), f"cache {kind}[{gi}]: {tuple(t.shape)}")
        self.caches, self.groups, self.nb, self.nbi = caches, G, nb, nbi
        self._step = self._build_op()

    def _build_op(self):
        """The per-rank step (megakernel + writer) over this adapter's operands. Wrapping it into the torch op
        (sharded_jax_op with Tensor[] weight / cache arguments, as glm5_mk_dcp's `_build_op`) belongs to the runner
        integration, which stage A does not wire (module docstring)."""
        cfg = self.cfg
        call, info = K.build(cfg, max_pages=self.max_pages, nb=self.nb, nbi=self.nbi, tiles=self.tiles,
                             stack_e=False, groups=self.groups)
        writer, _ = build_writer(cfg, max_pages=self.max_pages, nb=self.nb, nbi=self.nbi, groups=self.groups)
        self.info = info
        n_d, n_e = len(self.dense), len(self.experts)
        logger.info("GLM-5.2 decode megakernel: %d dense + %d expert + %d cache operands, %d rows",
                    n_d, n_e, len(self.caches), cfg.TMAX)

        def per_rank(scal, rmeta, qmeta, bt, bti, ltab, dtab, rvec, h0, tab, smalls, *ops):
            caches = ops[n_d + n_e:]
            outs = call(scal, rmeta, qmeta, bt, bti, ltab, dtab, rvec, h0, tab, smalls, *ops)
            new = writer(scal, rmeta, bt, bti, outs[1], outs[2], outs[3], *caches)
            return (outs[0], *new)
        return per_rank

    def step_ok(self, runner, scheduler_output, num_chunks: int) -> bool:
        """Rank-uniform host predicate (WHITEBOARD.v1 §2, §16 W4): every request in verify with 2..QMAX tokens,
        <= TMAX tokens, one chunk, no encoder inputs; q_len 1 (plain decode) and mixed steps decline."""
        cfg = self.cfg
        if self._step is None or num_chunks != 1 or scheduler_output.scheduled_encoder_inputs:
            return False
        ns = scheduler_output.num_scheduled_tokens
        ib = runner.input_batch
        for req_id, n in ns.items():
            i = ib.req_id_to_index[req_id]
            if not 2 <= n <= cfg.QMAX or int(ib.num_computed_tokens_cpu[i]) < int(ib.num_prompt_tokens[i]):
                return False
        return scheduler_output.total_num_scheduled_tokens <= cfg.TMAX and len(ns) <= cfg.BMAX
