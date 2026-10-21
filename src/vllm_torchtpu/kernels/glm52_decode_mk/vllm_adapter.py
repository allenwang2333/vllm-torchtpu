# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Serving glue for the GLM-5.2 decode megakernel (kernels/glm52_decode_mk), wired into TPUModelRunner (stage-B lane
L5: the same-harness twin A/B; WHITEBOARD.v3 §6.5).

Port of glm5_mk_dcp's adapter (vllm_adapter.py @ d41e70f6d; its rebased copy ran on main 9ffdb1eb4 in G0, j-31f97434)
to this kernel's operand plan and deployment: TP = DCP = world, DP 1, **EP off** (TP16 MoE: the served routed experts are
main's moe_tp_decode TP layout w13 [E, H, 2I/R], w2 [E, I/R, H], s13 [E, 1, 1, 2I/R], s2 [E, 1, 1, H], passed per layer,
zero-copy = the served operand form), **qrep 0** (the served q_b_proj is column-parallel: this rank's HR heads and their
scale slice are taken as they are), MTP verify rows. Enabled by GLM52_DECODE_MK=1.

Runner contract (runner/tpu_runner.py): `maybe_create` at load_model, `bind_caches` + `warmup` at capture_model,
`step_ok` per scheduled step (sets runner._glm52_mk_step), `__call__` in place of the backbone inside forward_model when
`enabled and runner._glm52_mk_step`. `enabled` is the runtime A/B toggle (twin harness R8: flipped by collective_rpc
between interleaved stock / megakernel batches); `repack()` re-reads the router weight and bias after the harness
re-draws them (R3).

Env:
- GLM52_DECODE_MK=1            build the megakernel at load (else the runner never sees it);
- GLM52_DECODE_MK_ROWS=16,32   one compiled step per row bucket; a step takes the smallest bucket >= its tokens;
- GLM52_DECODE_MK_TRACE=1      trace build (named scopes 'L_norm' ... per layer; A1-N3: +0.05-0.14% time);
- GLM52_DECODE_MK_OFF=1        built and warmed, but `enabled` starts False (the harness turns it on per arm).
"""

import dataclasses
import os
import time
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

from . import kernel as K
from . import layout as LY
from . import presets as PR
from .config import FULL, rup

logger = init_logger(__name__)

AXIS = "mk52"


class Unsupported(Exception):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(why)


def enabled() -> bool:
    return os.environ.get("GLM52_DECODE_MK", "0") == "1"


def _rows_env() -> list[int]:
    v = os.environ.get("GLM52_DECODE_MK_ROWS", "16")
    rows = sorted({int(x) for x in v.split(",") if x.strip()})
    _need(bool(rows) and all(r in (16, 32) for r in rows), f"GLM52_DECODE_MK_ROWS={v}")
    return rows


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


def _owns_storage(t: torch.Tensor) -> bool:
    try:
        return (t.storage_offset() == 0 and t.is_contiguous()
                and t.untyped_storage().nbytes() == t.numel() * t.element_size())
    except Exception:                                    # noqa: BLE001
        return False


def _dt(x) -> str:
    """dtype name of a torch tensor / torch dtype / jnp dtype ('float8_e4m3fn', 'bfloat16', 'uint32', ...)."""
    d = x.dtype if isinstance(x, torch.Tensor) else x
    if isinstance(d, torch.dtype):
        return str(d).replace("torch.", "")
    import numpy as np
    return np.dtype(d).name


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


def index_pattern(has_indexer: list[bool], freq0: int, skip0: int) -> tuple[int, int]:
    """(idx_freq, idx_skip) reproducing the model's ACTUAL per-layer fresh-indexer pattern under the kernel's rule
    (config.full_indexer: max(l - skip + 1, 0) % freq == 0), the hf (freq, skip) first. R1 (twin-plan §5.1): the twin
    sets `index_topk_pattern` [F, F, S], which the hf freq/skip (4, 3) do not describe; (4, 2) does."""
    L = len(has_indexer)

    def pat(fr, sk):
        return [max(l - sk + 1, 0) % fr == 0 for l in range(L)]

    cands = ([(freq0, skip0)] + [(freq0, sk) for sk in range(0, L + 1)]
             + [(fr, sk) for fr in range(1, max(L, 4) + 1) for sk in range(0, L + 1)])
    for fr, sk in cands:
        if fr >= 1 and pat(fr, sk) == list(has_indexer):
            return fr, sk
    raise Unsupported(f"indexer pattern {has_indexer} is not max(l - skip + 1, 0) % freq == 0 for any (freq, skip)")


N_FIX = 10     # pos [TMAX], seq [S], qsl [S+1], bt [S, W], cs rows [TMAX, DR], h0 [M, H], ltab, dtab (replicated),
               # rank i32[1], smalls (per rank)


def build_spmd(cfg, mesh, *, max_pages, nb, nbi, tiles, groups):
    """The SPMD step program, jax only (also lowered by the lane's AOT script, stageB/L5-twin-harness/aot_l5.py):
    shard_map over the one-axis R-rank mesh of (metadata + rope table from wrapper.py -> megakernel -> writer).
    Inputs: N_FIX fixed (8 replicated + smalls per rank), then the dense stacks, the per-layer served expert tensors
    and the per-layer caches [nb, ...] (each reshaped to the kernel's one-layer group [1, ...] inside the body, a
    bitcast), all sharded on dim 0 (= this rank's own tensors). Outputs: out [M, H] bf16 and the written caches.
    -> (smap, in_specs, out_specs, info)."""
    import jax
    from jax.sharding import PartitionSpec as P

    from . import wrapper as W
    from .writer import build_writer

    call, info = K.build(cfg, max_pages=max_pages, nb=nb, nbi=nbi, tiles=tiles, stack_e=False, groups=groups)
    writer, winfo = build_writer(cfg, max_pages=max_pages, nb=nb, nbi=nbi, groups=groups)
    assert len(info["c_names"]) == len(winfo["c_names"])
    n_d, n_e, n_c = len(info["dkinds"]), len(info["e_names"]), len(info["c_names"])
    TMAX = cfg.TMAX
    rep, sh = P(), P(AXIS)
    in_specs = (rep,) * (N_FIX - 2) + (sh, sh) + (sh,) * (n_d + n_e) + (sh,) * n_c
    out_specs = (sh,) + (sh,) * n_c

    def body(*a):
        pos, seq, qsl, btv, csr, h0, ltab_, dtab_, rank, smalls = a[:N_FIX]
        dw = a[N_FIX:N_FIX + n_d]
        ew = a[N_FIX + n_d:N_FIX + n_d + n_e]
        cs = a[N_FIX + n_d + n_e:]
        r = rank[0]          # an operand, not lax.axis_index: partition-id does not lower in torch_tpu's export
        scal, rmeta, qmeta, bt, bti, rvec = W.metadata(cfg, TMAX, pos, seq, qsl, btv, nb, nbi, max_pages, r)
        tab = W.rope_table(cfg, csr)
        c1 = [c[None] for c in cs]                       # [1, ...]: the kernel's one-layer cache group
        outs = call(scal, rmeta, qmeta, bt, bti, ltab_, dtab_, rvec, h0, tab, smalls, *dw, *ew, *c1)
        out, nrow, rrow, irec = outs[:4]
        new = writer(scal, rmeta, bt, bti, nrow, rrow, irec, *c1)
        return (out, *[c[0] for c in new])

    smap = jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    return smap, in_specs, out_specs, info


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
        _need(qmax == 4, f"{qmax} rows per request")
        _need(getattr(hf, "scoring_func", None) == "sigmoid", "router scoring")
        _need(getattr(hf, "n_group", 1) in (1, None) and getattr(hf, "topk_group", 1) in (1, None), "grouped routing")
        _need(bool(getattr(hf, "norm_topk_prob", True)), "unnormalized top-k")
        _need(bool(getattr(hf, "rope_interleave", False)) and bool(getattr(hf, "indexer_rope_interleave", False)),
              "non-interleaved rope")
        rp = getattr(hf, "rope_parameters", None) or {}
        _need(rp.get("rope_type", "default") == "default", f"rope {rp}")
        _need(runner.block_size == FULL.page, f"block_size {runner.block_size}")
        self.rows = _rows_env()
        bb = _find_backbone(runner.model)
        self.backbone = bb
        L = int(hf.num_hidden_layers)
        _need(len(bb.layers) == L, f"{len(bb.layers)} layers")
        has_idx = [getattr(bb.layers[l].self_attn, "indexer", None) is not None for l in range(L)]
        freq, skip = index_pattern(has_idx, int(hf.index_topk_freq), int(hf.index_skip_topk_offset))
        base = dataclasses.replace(
            FULL, R=R, L=L, first_dense=hf.first_k_dense_replace, H=hf.hidden_size,
            NH=hf.num_attention_heads, QL=hf.q_lora_rank, idx_freq=freq, idx_skip=skip, eps=hf.rms_norm_eps,
            routed_scale=float(hf.routed_scaling_factor), QMAX=qmax, TMAX=self.rows[-1],
            BM=int(runner.scheduler_config.max_num_seqs))
        for got, want, what in (
                (hf.kv_lora_rank, base.KL, "kv_lora_rank"), (hf.qk_nope_head_dim, base.DN, "qk_nope"),
                (hf.qk_rope_head_dim, base.DR, "qk_rope"), (hf.v_head_dim, base.DV, "v_head_dim"),
                (hf.n_routed_experts, base.E, "experts"), (hf.num_experts_per_tok, base.TOPK, "top-k"),
                (hf.moe_intermediate_size, base.I, "moe_intermediate"),
                (hf.moe_intermediate_size * hf.n_shared_experts, base.IS, "shared intermediate"),
                (hf.intermediate_size, base.ID, "dense intermediate"),
                (hf.index_n_heads, base.IH, "index heads"), (hf.index_head_dim, base.IDIM, "index dim"),
                (hf.index_topk, base.KT, "index_topk")):
            _need(got == want, f"{what} {got} != kernel {want}")
        self.cfgs = {}
        for t in self.rows:
            c = dataclasses.replace(base, TMAX=t)
            try:
                c.check()
            except AssertionError as e:
                raise Unsupported(f"admission (TMAX {t}): {e}") from e
            self.cfgs[t] = c
        cfg = self.cfg = self.cfgs[self.rows[-1]]
        for l in range(L):
            _need(has_idx[l] == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        self.rank = int(dist.get_rank())
        self.counts = {}
        self.enabled = os.environ.get("GLM52_DECODE_MK_OFF", "0") != "1"
        self.calls = 0
        from torch_tpu._internal.pallas import get_pallas_mesh
        self.mesh = get_pallas_mesh(axis_names=(AXIS,), mesh_shape=(R,))
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
        self.tiles = K.Tiles(trace=os.environ.get("GLM52_DECODE_MK_TRACE", "0") == "1")
        # stage-B levers (presets.py): GLM52_DECODE_MK_LEVERS = "" / "A" (stage A) | "B" (all) | "L4,L1,..."
        self.levers = PR.parse(os.environ.get("GLM52_DECODE_MK_LEVERS", ""))
        self.tiles_by_rows = {t: dataclasses.replace(self.tiles, **PR.tiles_kwargs(t, self.levers)) for t in self.rows}
        self.caches = None
        self._steps = {}
        self.mla_layer_name = _attn_module(bb.layers[0].self_attn).layer_name
        logger.info("GLM-5.2 decode megakernel: built for rows %s, L %d (fresh indexer layers %s, freq %d skip %d), "
                    "rope cache rows %d (max_model_len %d), k_scales %s, trace %s, enabled %s, levers %s", self.rows, cfg.L,
                    LY.layers_of(cfg, "iwq"), freq, skip, int(self.cos_sin.shape[0]), mml, sorted(set(self.k_scales)),
                    self.tiles.trace, self.enabled, ",".join(self.levers) or "stage A")

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
            d["gate"] = self._gate_kn(l)
            rt = _routed(mlp)
            d["ew13"], d["ew2"] = rt.w13_weight, rt.w2_weight
            d["es13"], d["es2"] = rt.w13_weight_scale_inv, rt.w2_weight_scale_inv
            for k_, shp in (("ew13", (cfg.E, H, 2 * cfg.IR)), ("ew2", (cfg.E, cfg.IR, H)),
                            ("es13", (cfg.E, 1, 1, 2 * cfg.IR)), ("es2", (cfg.E, 1, 1, H))):
                _need(tuple(d[k_].shape) == shp, f"layer {l} {k_}: {tuple(d[k_].shape)} != {shp} (TP16 layout)")
            for k_ in ("ew13", "ew2", "es13", "es2"):
                _need(_owns_storage(d[k_]), f"layer {l} {k_}: a view operand (would be re-materialized per call)")
            se = mlp.shared_experts
            d["sh13"] = _kn(se.gate_up_proj.weight, H, 2 * cfg.ISR, "shared.gate_up_proj")
            d["sh2"] = _kn(se.down_proj.weight, cfg.ISR, H, "shared.down_proj")
        else:
            d["dw13"] = _kn(mlp.gate_up_proj.weight, H, 2 * cfg.IDR, "dense.gate_up_proj")
            d["dw2"] = _kn(mlp.down_proj.weight, cfg.IDR, H, "dense.down_proj")
        return d

    def _gate_kn(self, l: int) -> torch.Tensor:
        cfg = self.cfg
        return _pad_n(_kn(self.backbone.layers[l].mlp.gate.weight, cfg.H, cfg.E, "gate"), cfg.EPAD)

    def _collect_weights(self):
        cfg = self.cfg
        per = {l: self._layer(l) for l in range(cfg.L)}
        self._per_layer = per
        dsh = LY.dense_shapes(cfg)
        dense, self._dense_kind = [], {}
        for k in LY.DKINDS:
            ls = LY.layers_of(cfg, k)
            if not ls:
                continue
            ts = [per[l][k].detach() for l in ls]
            for l, t in zip(ls, ts):
                _need(tuple(t.shape) == tuple(dsh[k][0]), f"{k}[{l}]: {tuple(t.shape)} != {dsh[k][0]}")
                _need(_dt(t) == _dt(dsh[k][1]), f"{k}[{l}]: dtype {t.dtype} != {dsh[k][1]}")
            self._dense_kind[k] = len(dense)
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

    @torch.no_grad()
    def repack(self) -> dict:
        """R3: after the router weights / bias are re-drawn in place (twin harness `_randomize_router`), copy them
        into the megakernel's operands (the stacked `gate` kind and the smalls rows), in place, then check on device
        that every operand equals its module. Returns the max |diff| per check (0.0 = by-reference equivalent)."""
        cfg = self.cfg
        gi = self._dense_kind["gate"]
        g = self.dense[gi]
        for j, l in enumerate(LY.layers_of(cfg, "gate")):
            g[j].copy_(self._gate_kn(l))
        self.smalls.copy_(self._build_smalls())
        lay, _ = LY.smalls_layout(cfg)
        off = 0
        o_gb = None
        for name, (_, n) in lay.items():
            if name == "gate_bias":
                o_gb = (off, n)
            off += rup(n)
        gate_d = bias_d = 0.0
        for j, l in enumerate(LY.layers_of(cfg, "gate")):
            gate_d = max(gate_d, float((g[j].float() - self._gate_kn(l).float()).abs().max().cpu()))
            if o_gb is not None:
                b = self.smalls[l, 0, o_gb[0]:o_gb[0] + cfg.E]
                bias_d = max(bias_d, float((b - self.backbone.layers[l].mlp.gate.e_score_correction_bias.float())
                                           .abs().max().cpu()))
        return {"gate_maxdiff": gate_d, "bias_maxdiff": bias_d, "moe_layers": len(LY.layers_of(cfg, "gate"))}

    # ----------------------------------------------------------------- caches

    def bind_caches(self, runner) -> None:
        """The served per-layer caches, one operand per (kind, layer), passed as the runner's own tensors (no view,
        no stack); the jax body reshapes each to the kernel's [1, ...] group of one layer (a bitcast)."""
        cfg = self.cfg
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
        caches, G, views = [], {}, []
        ck = LY.cache_kinds(cfg, nb, nbi)
        for kind in ("nope", "rope", "idx"):
            ls = ck[kind][0]
            G[kind] = len(ls)
            for l in ls:
                t = per[(kind, l)]
                _need(tuple(t.shape) == tuple(ck[kind][1]), f"cache {kind}[{l}]: {tuple(t.shape)} != {ck[kind][1]}")
                _need(_dt(t) == _dt(ck[kind][2]), f"cache {kind}[{l}]: dtype {t.dtype} != {ck[kind][2]}")
                if not _owns_storage(t):
                    views.append(f"{kind}{l}")
                caches.append(t)
        self.caches, self.groups, self.nb, self.nbi = caches, G, nb, nbi
        self.cache_views = views
        logger.info("GLM-5.2 decode megakernel: %d dense + %d expert + %d cache operands (groups %s), %d blocks "
                    "(idx %d), max_pages %d; view caches %s", len(self.dense), len(self.experts), len(caches), G, nb,
                    nbi, self.max_pages, views or "none")
        for t in self.rows:
            self._steps[t] = self._build_op(self.cfgs[t])

    def _build_op(self, cfg):
        """The compiled per-step torch function for row bucket cfg.TMAX: embed (torch, vocab-parallel) -> one SPMD
        program shard_map(metadata + rope table (wrapper.py, the instruments-tested arithmetic) -> megakernel ->
        writer) over the R-rank mesh -> caches copied back (donated, aliased in place)."""
        import jax

        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

        n_d, n_e, n_c = len(self.dense), len(self.experts), len(self.caches)
        smap, in_specs, out_specs, info = build_spmd(cfg, self.mesh, max_pages=self.max_pages, nb=self.nb,
                                                     nbi=self.nbi, tiles=self.tiles_by_rows[cfg.TMAX],
                                                     groups=self.groups)
        assert list(info["dkinds"]) == list(self._dense_kind), (info["dkinds"], list(self._dense_kind))
        assert len(info["e_names"]) == n_e, (len(info["e_names"]), n_e)
        assert len(info["c_names"]) == n_c
        TMAX, M = cfg.TMAX, cfg.M
        dev = self.smalls.device
        ltab = torch.tensor(info["ltab"], dtype=torch.int32).to(dev)
        dtab = torch.tensor(info["dtab"], dtype=torch.int32).to(dev)
        rank_t = torch.tensor([self.rank], dtype=torch.int32).to(dev)
        n_fix = N_FIX
        n_in = n_fix + n_d + n_e + n_c
        args = [f"x{i}" for i in range(n_in)]
        ret = "tuple[" + ", ".join(["jax.Array"] * len(out_specs)) + "]"
        fname = f"glm52_decode_mk_b{TMAX}"
        src = (f"def {fname}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
               f"    return _smap({', '.join(args)})\n")
        ns: dict[str, Any] = {"jax": jax, "_smap": smap}
        exec(compile(src, f"<{fname}>", "exec"), ns)  # noqa: S102 - arity is data
        name = f"glm52_decode_mk::step_b{TMAX}"
        # unregistered callable (sharded_jax_op register=False): the operand count exceeds a torch op schema's 64;
        # the op below takes the weights and caches as Tensor[] arguments (glm5_mk_dcp)
        flat = sharded_jax_op(name + "_flat", ns[fname], mesh=self.mesh, input_partition_specs=in_specs,
                              output_partition_specs=out_specs, donate_argnums=tuple(range(n_in - n_c, n_in)),
                              register=False)

        @torch.library.custom_op(name, mutates_args=())
        def op(pos: torch.Tensor, seq: torch.Tensor, qsl: torch.Tensor, btv: torch.Tensor, csr: torch.Tensor,
               h0: torch.Tensor, ltab_: torch.Tensor, dtab_: torch.Tensor, rank: torch.Tensor, smalls: torch.Tensor,
               weights: list[torch.Tensor], caches: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(flat(pos, seq, qsl, btv, csr, h0, ltab_, dtab_, rank, smalls, *weights, *caches))

        @op.register_fake
        def _(pos, seq, qsl, btv, csr, h0, ltab_, dtab_, rank, smalls, weights, caches):
            return ([torch.empty((M, cfg.H), dtype=torch.bfloat16, device=h0.device)]
                    + [torch.empty_like(c) for c in caches])

        def fit(x, n):
            k = x.shape[0]
            if k >= n:
                return x[:n]
            return torch.nn.functional.pad(x, [0, 0] * (x.dim() - 1) + [0, n - k])

        cos_sin = self.cos_sin
        n_pos = int(cos_sin.shape[0])

        def step(ids, positions, qsl, seq_lens, bt_flat, smalls, weights, caches):
            i32 = torch.int32
            nq = qsl.shape[0] - 1
            btv = bt_flat.reshape(nq, -1).to(i32)
            p = fit(positions.reshape(-1).to(i32), TMAX)
            T = qsl[-1].to(i32)
            valid = torch.arange(TMAX, device=p.device, dtype=i32) < T
            pc = torch.clamp(p * valid.to(i32), 0, n_pos - 1)
            csr = cos_sin[pc.long()].to(torch.bfloat16).float()                  # [TMAX, DR], bf16 values
            h0 = self.backbone.embed_tokens(fit(ids.reshape(-1), TMAX)).float()
            outs = op(p, seq_lens.to(i32), qsl.to(i32), btv, csr, h0, ltab, dtab, rank_t, smalls, list(weights),
                      list(caches))
            for dst, src in zip(caches, outs[1:1 + len(caches)]):
                dst.copy_(src)
            return outs[0][:TMAX]

        logger.info("GLM-5.2 decode megakernel: step b%d built (%d operands)", TMAX, n_in)
        return torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)

    # ------------------------------------------------------------------- run

    def bucket_for(self, num_tokens: int) -> int | None:
        for t in self.rows:
            if num_tokens <= t:
                return t
        return None

    def step_ok(self, runner, scheduler_output, num_chunks: int) -> bool:
        """Rank-uniform host predicate (WHITEBOARD.v1 §2, §16 W4): every request in verify with 2..QMAX tokens,
        <= the largest row bucket, one chunk, no encoder inputs; q_len 1 (plain decode) and mixed steps decline."""
        ok, why = self._step_ok(runner, scheduler_output, num_chunks)
        c = self.counts
        c[why] = c.get(why, 0) + 1
        return ok

    def _step_ok(self, runner, scheduler_output, num_chunks: int):
        cfg = self.cfg
        if not self.enabled or not self._steps:
            return False, "off"
        if num_chunks != 1 or scheduler_output.scheduled_encoder_inputs:
            return False, "other"
        ns = scheduler_output.num_scheduled_tokens
        ib = runner.input_batch
        for req_id, n in ns.items():
            i = ib.req_id_to_index[req_id]
            if not 2 <= n <= cfg.QMAX or int(ib.num_computed_tokens_cpu[i]) < int(ib.num_prompt_tokens[i]):
                return False, "not_verify"
        if scheduler_output.total_num_scheduled_tokens > self.rows[-1] or len(ns) > cfg.BMAX:
            return False, "over_bucket"
        return True, "mk"

    def rows_ok(self, q_lens: list[int], num_tokens: int) -> bool:
        """The same predicate from host-known rows (the twin harness, which calls forward_model directly)."""
        cfg = self.cfg
        return (self.enabled and bool(self._steps) and bool(q_lens) and all(2 <= q <= cfg.QMAX for q in q_lens)
                and sum(q_lens) <= num_tokens and len(q_lens) <= cfg.BMAX and num_tokens in self._steps)

    def _metadata(self):
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(self.mla_layer_name)
        assert md is not None and hasattr(md, "query_start_loc"), type(md)
        return md

    @torch.no_grad()
    def __call__(self, runner, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.caches is None:               # capture_model was skipped
            self.bind_caches(runner)
        n = input_ids.shape[0]
        t = self.bucket_for(n)
        assert t is not None, (n, self.rows)
        md = self._metadata()
        out = self._steps[t](input_ids, positions, md.query_start_loc, md.seq_lens, md.block_tables, self.smalls,
                             self.dense + self.experts, self.caches)
        self.calls += 1
        if n <= out.shape[0]:
            return out[:n]
        return torch.nn.functional.pad(out, (0, 0, 0, n - out.shape[0]))

    def status(self) -> dict:
        return {"rank": self.rank, "enabled": self.enabled, "rows": self.rows, "built": sorted(self._steps),
                "L": self.cfg.L, "fresh_layers": LY.layers_of(self.cfg, "iwq"),
                "idx_freq_skip": [self.cfg.idx_freq, self.cfg.idx_skip], "n_dense": len(self.dense),
                "n_experts": len(self.experts), "n_caches": len(self.caches or []), "nb": getattr(self, "nb", None),
                "max_pages": self.max_pages, "trace": self.tiles.trace,
                "levers": list(self.levers), "calls": self.calls,
                "cache_views": getattr(self, "cache_views", None), "k_scales": sorted(set(self.k_scales)),
                "counts": dict(self.counts)}

    def warmup(self, runner) -> None:
        """Compile and run each bucket once: one request of QMAX rows at positions 0..QMAX-1 on block 0 (the null
        block; nothing is served from it before the first step), inside a forward context the step reads."""
        from vllm_torchtpu.utils import synchronize_tensors

        cfg = self.cfg
        dev = self.smalls.device
        for t in self.rows:
            ids = torch.zeros(t, dtype=torch.int32, device=dev)
            pos = torch.arange(t, dtype=torch.int32, device=dev) % cfg.QMAX
            S = cfg.BMAX
            qsl = torch.tensor([0] + [cfg.QMAX] * S, dtype=torch.int32, device=dev)
            seq = torch.tensor([cfg.QMAX] + [0] * (S - 1), dtype=torch.int32, device=dev)
            bt = torch.zeros(S * self.max_pages, dtype=torch.int32, device=dev)
            t0 = time.monotonic()
            out = self._steps[t](ids, pos, qsl, seq, bt, self.smalls, self.dense + self.experts, self.caches)
            synchronize_tensors([out])
            logger.info("GLM-5.2 decode megakernel: b%d compiled and warmed in %.1f s", t, time.monotonic() - t0)
