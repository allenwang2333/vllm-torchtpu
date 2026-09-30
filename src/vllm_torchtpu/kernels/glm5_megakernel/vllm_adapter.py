# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Serving glue for the GLM-5.2 decode megakernel.

The kernel runs one decode token per DP rank through every decoder layer, with
its own all-gather / reduce-scatter between ranks around the routed experts. So
it is all-or-nothing across the DP group: the runner may take it only on a step
where *every* rank has at most one token (decided off the DP all-reduce the
runner already does), busy ranks from `execute_model` and idle ranks from the
DP dummy forward with `seq_len` 0, which reads and writes nothing.

Operands are the served tensors themselves -- weights as loaded, the MLA and
indexer caches updated in place through donation -- plus three load-time
copies the kernel's layout needs: the router weight and the indexer's
`wk_weights_proj` transposed to (K, N), and one f32 row per layer packing the
norm weights, per-channel scales and the router bias (`layout.smalls_layout`).

Expert relabel: the kernel owns global expert j on mesh index j // ER, while
the served shard at mesh index i holds the experts of EP rank
`ep_rank_order()[i]`. Permuting the router's columns and bias by
perm[j] = order[j // ER] * ER + j % ER makes the two agree without moving any
expert weight.
"""

import dataclasses
import math
import sys
import time
from typing import Any

import jax
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_callable
from vllm_torchtpu.kernels.glm5_megakernel import kernel as K
from vllm_torchtpu.kernels.glm5_megakernel import replicas as REP
from vllm_torchtpu.kernels.glm5_megakernel.config import FULL
from vllm_torchtpu.kernels.glm5_megakernel.layout import rup, smalls_layout
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

EXPERT_KINDS = ("ew13", "ew2", "es13", "es2")
# operands sharded over the EP axis (the local tensor is this rank's shard): experts, and
# the per-head projections that are TP over heads: o_proj's rows, q_b's columns, W_UK's heads
# decode replica slots (GLM5_MK_DECODE_REPLICAS): kind -> the expert kind it copies
REPLICA_KINDS = {"rw13": "ew13", "rw2": "ew2", "rs13": "es13", "rs2": "es2"}
SHARDED_KINDS = EXPERT_KINDS + ("o_proj", "q_b", "w_uk") + tuple(REPLICA_KINDS)
# GLM5_MK_STACK_W: the kinds whose operand is this adapter's own load-time buffer (a copy, clone,
# column permutation, pad or zeros), so one [n_layers, ...] tensor per kind replaces the per-layer
# tensors byte for byte and the KV pool is sized against the same HBM. qkv_a is one: _kn_pad pads
# N = 2624 to 2688, so it was always a copy (1.2 GiB/rank already duplicated per layer).
OWNED_KINDS = ("qkv_a", "q_b", "w_uk", "o_proj", "idx_wk", "gate") + tuple(REPLICA_KINDS)
# Kinds that alias a stock parameter but are stacked with NO duplicate: the stack replaces the
# stock parameter: the stock layer gets layer.W_UV_stack = (stack, row) and reads the row inside
# one jax op (attention._w_uv_bmm_stacked: a static slice feeding its bmm). Every other kind (idx_wq, the experts
# and their scales, the shared and dense MLPs) aliases a stock parameter and stays per layer.
VIEW_KINDS = ("w_uv",)


class Unsupported(Exception):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(why)


def _unwrap(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def _kn(weight: torch.Tensor, k: int, n: int, what: str) -> torch.Tensor:
    """A linear weight as (K, N), whichever way the loader left it."""
    if tuple(weight.shape) == (k, n):
        return weight
    _need(tuple(weight.shape) == (n, k), f"{what}: shape {tuple(weight.shape)}, want ({k}, {n})")
    return weight.t().contiguous()


def _kn_pad(weight: torch.Tensor, k: int, n: int, what: str) -> torch.Tensor:
    """_kn, N zero-padded to a multiple of 128: an unaligned minor dim gets a
    non-default XLA layout, i.e. a relayout copy in front of every step."""
    w = _kn(weight, k, n, what)
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
        if hasattr(m, "W_UK_T") and hasattr(m, "kv_cache"):
            return m
    raise Unsupported("no MLA attention module with W_UK_T and kv_cache")


def _routed(mlp):
    exp = mlp.experts
    return getattr(exp, "routed_experts", exp)


def _expert_scale(routed, name: str, shape: tuple) -> torch.Tensor:
    for suffix in ("_scale_inv", "_scale"):
        s = getattr(routed, name + suffix, None)
        if s is not None:
            _need(s.numel() == math.prod(shape), f"{name}{suffix}: {tuple(s.shape)}, want {shape}")
            return s.reshape(shape)
    raise Unsupported(f"routed experts: no {name} scale")


class Glm5Megakernel:
    """Built at `load_model` (weights), bound to the caches after KV init."""

    @classmethod
    def maybe_create(cls, runner) -> "Glm5Megakernel | None":
        if not envs.USE_GLM5_MEGAKERNEL:
            return None
        try:
            return cls(runner)
        except Unsupported as e:
            logger.warning("GLM-5.2 megakernel disabled: %s", e)
            return None

    def __init__(self, runner):
        hf = runner.model_config.hf_config
        pc = runner.parallel_config
        _need(type(hf).__name__.startswith("GlmMoeDsa") or getattr(hf, "model_type", "")
              == "glm_moe_dsa", f"model_type {getattr(hf, 'model_type', None)}")
        _need(pc.pipeline_parallel_size == 1, "pipeline parallel")
        _need(pc.tensor_parallel_size == 1, "tensor parallel within a DP rank")
        _need(runner.speculative_config is None, "speculative decoding")
        _need(runner._dp_lockstep_enabled(), "DP lockstep is off")
        _need(getattr(hf, "scoring_func", None) == "sigmoid", "router scoring is not sigmoid")
        _need(getattr(hf, "topk_method", None) == "noaux_tc", "router is not noaux_tc")
        _need(getattr(hf, "n_group", 1) == 1 and getattr(hf, "topk_group", 1) == 1, "grouped routing")
        _need(bool(getattr(hf, "norm_topk_prob", True)), "unnormalized top-k")
        _need(bool(getattr(hf, "rope_interleave", False))
              and bool(getattr(hf, "indexer_rope_interleave", False)), "non-interleaved rope")
        rp = getattr(hf, "rope_parameters", None) or {}
        _need(rp.get("rope_type", "default") == "default", f"rope {rp}")
        _need(getattr(hf, "index_topk_pattern", None) is None, "explicit index_topk_pattern")

        self.mesh = build_ep_mesh()
        _need(self.mesh is not None, "EP is off")
        R = self.mesh.devices.size
        cfg = dataclasses.replace(
            FULL, R=R, L=hf.num_hidden_layers, first_dense=hf.first_k_dense_replace,
            H=hf.hidden_size, NH=hf.num_attention_heads, QL=hf.q_lora_rank,
            idx_freq=hf.index_topk_freq, idx_skip=hf.index_skip_topk_offset,
            rope_theta=float(rp.get("rope_theta", FULL.rope_theta)), eps=hf.rms_norm_eps,
            routed_scale=float(hf.routed_scaling_factor),
            KREP=int(envs.GLM5_MK_DECODE_REPLICAS))
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
        _need(cfg.NH % R == 0 and cfg.E % R == 0, f"{R} ranks do not divide heads/experts")
        _need(cfg.KREP >= 0 and R * cfg.KREP <= 64 and cfg.ER + cfg.KREP <= 128,
              f"GLM5_MK_DECODE_REPLICAS={cfg.KREP} at R={R}")
        self.cfg = cfg
        self.rank = ep_mesh_index()
        order = ep_rank_order()
        ER = cfg.ER
        self.perm = torch.tensor([order[j // ER] * ER + j % ER for j in range(cfg.E)],
                                 dtype=torch.long)
        # Online EPLB (distributed/tpu_eplb.py) moves experts between ranks; a
        # moved layer's router columns and bias follow the layer's own table.
        self.layer_perm: dict[int, torch.Tensor] = {}
        # Decode replicas: per MoE layer the LOGICAL expert load that places them (EPLB's window
        # once it has rebalanced, else the offline profile, else none = uniform), and the table
        # in kernel expert space that load gives under the layer's current permutation.
        self.krep = cfg.KREP
        self.rep_load: dict[int, Any] = {}
        self.rep_entries: dict[int, list] = {}
        self._gather_ops: dict = {}
        if self.krep:
            prof = envs.GLM5_MK_DECODE_REPLICAS_PROFILE
            if prof:
                self.rep_load = REP.load_profile(prof)
            for l in range(cfg.L):
                if cfg.is_moe(l):
                    self.rep_entries[l] = self._rep_place(l)
            logger.info("GLM-5.2 megakernel: %d decode replica slots/rank/MoE layer, placement %s",
                        self.krep, f"profile {prof}" if prof else "uniform (no load yet)")

        bb = _find_backbone(runner.model)
        self.backbone = bb
        _need(len(bb.layers) == cfg.L, f"{len(bb.layers)} layers")
        # Inference-only operands: detached (shared storage) so the tpu backend's
        # aot_autograd never builds a backward through the custom op.
        self.embed = bb.embed_tokens.weight.detach()
        self.page = cfg.page
        self.max_pages = runner.max_num_blocks_per_req
        _need(runner.block_size == cfg.page, f"block_size {runner.block_size} != {cfg.page}")
        self.rotary = bb.layers[0].self_attn.rotary_emb
        _need(self.rotary.cos_sin_cache.shape[-1] == cfg.DR, "rope cache width")

        self.wk = K.weight_kinds(cfg)
        self.stack_w = self._stack_knobs()
        self._put_ops: dict = {}
        with torch.no_grad():
            self.weights = [w.detach() for w in self._collect_weights()]
            self._rebind_views()
            if self.krep:
                # allocated here, at load_model, so profile_run's HBM accounting sees them before
                # the KV pool is sized: k * (ew13 + ew2 + scales) per MoE layer per rank
                self._gather_replicas(sorted(self.rep_entries))
                rb = sum(t.numel() * t.element_size() for (k_, _), t in
                         zip(self._w_index, self.weights) if k_ in REPLICA_KINDS)
                logger.info("GLM-5.2 megakernel: decode replicas hold %.3f GiB per rank",
                            rb / 2 ** 30)
            self.smalls = self._build_smalls().detach()
            # a runtime input, not a graph constant: see _build_op
            self.rank_t = torch.tensor([self.rank], dtype=torch.int32, device=self.smalls.device)
        # Debug: per-layer comparison against the stock layers (VERIFY + LAYERDIFF).
        self.layerdiff = bool(envs.GLM5_MEGAKERNEL_VERIFY and envs.GLM5_MEGAKERNEL_LAYERDIFF)
        self.dbg = None
        if self.layerdiff:
            self._install_layer_taps()
        self.op = None
        self.caches = None
        self.stack_c = frozenset()
        self.cache_rows: dict = {}
        self.rope_packed = False
        self._step = None
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(self.weights + [self.smalls])
        logger.info("GLM-5.2 megakernel: %d weight operands, mesh index %d of %d",
                    len(self.weights), self.rank, R)

    # ---------------------------------------------------------------- weights

    def _stack_knobs(self) -> frozenset:
        """GLM5_MK_STACK_W (kernel.stack_kinds syntax: 1 or a kind list) -> the weight kinds
        served stacked: OWNED_KINDS in place, VIEW_KINDS by rebinding the stock parameter to its
        row of the stack (_rebind_views). GLM5_MK_STACK_C is the runner's: it allocates the
        cache kinds stacked (kv_cache_manager) and bind_caches passes those stacks through."""
        # the replica kinds are valid knob values at GLM5_MK_DECODE_REPLICAS=0 too (no operand)
        sw = K.stack_kinds(None, "GLM5_MK_STACK_W", set(self.wk) | set(REPLICA_KINDS)) & set(self.wk)
        K.stack_kinds(None, "GLM5_MK_STACK_C", K.cache_kinds(self.cfg, 1, True))  # validate
        alias = sorted(sw - set(OWNED_KINDS) - set(VIEW_KINDS))
        if alias:
            logger.warning("GLM5_MK_STACK_W: %s alias stock parameters; served per layer", alias)
        sw = frozenset(sw & (set(OWNED_KINDS) | set(VIEW_KINDS)))
        if sw:
            logger.info("GLM-5.2 megakernel: stacked weight operands %s", sorted(sw))
        return sw

    def _put_op(self, kind: str):
        """(stack [n, ...], x, [row]) -> stack with row `row` = x; the stack is donated (an in-place
        DUS: no second copy of the stack at load or at an EPLB rebalance)."""
        op = self._put_ops.get(kind)
        if op is None:
            sh = kind in SHARDED_KINDS

            def put(stk: jax.Array, x: jax.Array, row: jax.Array) -> jax.Array:
                return jax.lax.dynamic_update_index_in_dim(stk, x, row[0], 0)

            op = sharded_jax_callable(
                f"glm5_megakernel::stack_put_{kind}", put, mesh=self.mesh,
                input_partition_specs=(P(None, EP_AXIS_NAME) if sh else P(),
                                       P(EP_AXIS_NAME) if sh else P(), P()),
                output_partition_specs=P(None, EP_AXIS_NAME) if sh else P(),
                donate_argnums=(0,))
            self._put_ops[kind] = op
        return op

    def _row_t(self, row: int, dev) -> torch.Tensor:
        return torch.tensor([row], dtype=torch.int32, device=dev)

    def _set_weight(self, kind: str, l: int, x: torch.Tensor) -> None:
        """Replace layer l's `kind` operand: the per-layer tensor, or its row of the stack."""
        i, row = self._w_pos[(kind, l)]
        if row is None:
            self.weights[i] = x.detach()
        else:
            self.weights[i] = self._put_op(kind)(self.weights[i], x,
                                                 self._row_t(row, x.device)).detach()

    def _layer_weights(self, l: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        H = cfg.H
        L = self.backbone.layers[l]
        sa = L.self_attn
        at = _attn_module(sa)
        d = {
            "qkv_a": _kn_pad(sa.fused_qkv_a_proj.weight, H, cfg.QKVA, "fused_qkv_a_proj"),
            # q_b and W_UK are TP over heads (heads [rank * HR, (rank + 1) * HR)): q_b's columns
            # are head-major, so this rank's are one contiguous column range
            "q_b": _kn(sa.q_b_proj.weight, cfg.QL, cfg.NH * cfg.DQ, "q_b_proj")[
                :, self.rank * cfg.HR * cfg.DQ:(self.rank + 1) * cfg.HR * cfg.DQ].contiguous(),
            "w_uk": at.W_UK_T[self.rank * cfg.HR:(self.rank + 1) * cfg.HR].clone(),
            "w_uv": at.W_UV,
            # o_proj is TP over heads: this rank's rows, heads [rank * HR, (rank + 1) * HR).
            # clone, not contiguous(): the row slice is already contiguous, so contiguous() kept a
            # view at storage offset rank * 6 MiB -- the only per-rank operand difference
            # (j-45afbf20; the exported module is identical on all 16 ranks)
            "o_proj": _kn(sa.o_proj.weight, cfg.NH * cfg.DV, H, "o_proj")[
                self.rank * cfg.HR * cfg.DV:(self.rank + 1) * cfg.HR * cfg.DV].clone(),
        }
        idx = getattr(sa, "indexer", None)
        _need((idx is not None) == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        if idx is not None:
            d["idx_wq"] = _kn(idx.wq_b.weight, cfg.QL, cfg.IH * cfg.IDIM, "indexer.wq_b")
            d["idx_wk"] = _kn_pad(idx.wk_weights_proj.weight, H, cfg.IDIM + cfg.IH,
                                  "indexer.wk_weights_proj")
        mlp = L.mlp
        if cfg.is_moe(l):
            g = _kn(mlp.gate.weight, H, cfg.E, "gate")
            d["gate"] = g[:, self._perm(l).to(g.device)].contiguous()
            r = _routed(mlp)
            d["ew13"] = r.w13_weight
            d["ew2"] = r.w2_weight
            d["es13"] = _expert_scale(r, "w13_weight", (cfg.ER, 2 * cfg.I))
            d["es2"] = _expert_scale(r, "w2_weight", (cfg.ER, H))
            if cfg.KREP:                 # filled by _gather_replicas (stacked: zeroed whole, below)
                for k_, src in REPLICA_KINDS.items():
                    if k_ in self.stack_w:
                        continue
                    shp = self.wk[k_][1]
                    d[k_] = torch.zeros(shp, dtype=d[src].dtype, device=d[src].device)
            se = mlp.shared_experts
            d["sw13"] = _kn(se.gate_up_proj.weight, H, 2 * cfg.IS, "shared.gate_up_proj")
            d["sw2"] = _kn(se.down_proj.weight, cfg.IS, H, "shared.down_proj")
        else:
            d["w13"] = _kn(mlp.gate_up_proj.weight, H, 2 * cfg.ID, "dense.gate_up_proj")
            d["w13_s"] = _scale(mlp.gate_up_proj, 2 * cfg.ID, "dense.gate_up_proj").reshape(1, -1)
            d["w2"] = _kn(mlp.down_proj.weight, cfg.ID, H, "dense.down_proj")
        return d

    def _perm(self, l: int) -> torch.Tensor:
        """Kernel expert slot j -> logical expert, for layer l."""
        return self.layer_perm.get(l, self.perm)

    @torch.no_grad()
    def set_expert_placement(self, phys2log_mesh: dict) -> None:
        """Follow an online EPLB rebalance: {layer: logical expert at mesh slot q}.

        The expert tensors were rewritten in place by the rebalancer, so only
        the router columns and bias -- the kernel's own load-time copies --
        change. Kernel slot j is mesh slot j (both are j // ER on mesh index
        j // ER), so the new permutation is the table itself. Every replaced
        operand keeps its shape, dtype and layout: a runtime input of the
        compiled step, not a recompile.
        """
        import time

        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        for l, p2l in phys2log_mesh.items():
            self.layer_perm[int(l)] = torch.as_tensor(p2l, dtype=torch.long)
        changed = {int(l) for l in phys2log_mesh}
        for (kind, l), (i, row) in list(self._w_pos.items()):
            if l not in changed:
                continue
            if kind == "gate":
                L = self.backbone.layers[l]
                g = _kn(L.mlp.gate.weight, self.cfg.H, self.cfg.E, "gate")
                self._set_weight("gate", l, g[:, self._perm(l).to(g.device)].contiguous())
            elif kind in EXPERT_KINDS:      # never stacked (aliases, not in OWNED_KINDS)
                assert row is None, kind
                r = _routed(self.backbone.layers[l].mlp)
                new = {"ew13": lambda: r.w13_weight, "ew2": lambda: r.w2_weight,
                       "es13": lambda: _expert_scale(r, "w13_weight", (self.cfg.ER, 2 * self.cfg.I)),
                       "es2": lambda: _expert_scale(r, "w2_weight", (self.cfg.ER, self.cfg.H))}[kind]()
                self.weights[i] = new.detach()
        if self.krep:
            # kernel slot j now holds another logical expert: re-place from the logical load and
            # re-copy the slots (the owners' rows moved even where the table did not)
            for l in sorted(changed):
                if l in self.rep_entries:
                    self.rep_entries[l] = self._rep_place(l)
            self._gather_replicas(sorted(changed & set(self.rep_entries)))
        self.smalls = self._build_smalls().detach()
        t1 = time.perf_counter()
        synchronize_tensors(self.weights + [self.smalls])
        # (host rebuild s, device sync s), read by the EPLB stall trace.
        self.last_placement_timing = (t1 - t0, time.perf_counter() - t1)

    # ------------------------------------------------------------ replicas

    def _rep_place(self, l: int) -> list:
        """Replica entries of layer l from its logical load, mapped to kernel expert space."""
        cfg = self.cfg
        ll = self.rep_load.get(l)
        kl = None if ll is None else np.asarray(ll, np.float64)[self._perm(l).numpy()]
        return REP.place(REP.hit_probability(kl, cfg.E, cfg.R), cfg.R, cfg.ER, cfg.KREP)

    @torch.no_grad()
    def refresh_replicas(self, logical_load: dict) -> int:
        """Online EPLB, at every rebalance (after set_expert_placement), on every rank: re-place
        from the window's logical load {layer: [E]} and re-copy the layers whose table changed.
        Collective. Returns the number of layers re-copied."""
        if not self.krep:
            return 0
        from vllm_torchtpu.utils import synchronize_tensors

        redo = []
        for l, ll in logical_load.items():
            l = int(l)
            if l not in self.rep_entries:
                continue
            self.rep_load[l] = np.asarray(ll, np.float64)
            new = self._rep_place(l)
            if new != self.rep_entries[l]:
                self.rep_entries[l] = new
                redo.append(l)
        if redo:
            self._gather_replicas(redo)
            self.smalls = self._build_smalls().detach()
            synchronize_tensors(self.weights + [self.smalls])
        return len(redo)

    def _gather_op(self, src_shape, dst_shape, dtype):
        key = (tuple(src_shape), tuple(dst_shape), str(dtype))
        op = self._gather_ops.get(key)
        if op is None:
            spec = P(EP_AXIS_NAME)
            op = sharded_jax_callable(
                f"glm5_megakernel::replica_gather_{len(self._gather_ops)}",
                REP.make_gather_fn(self.mesh, EP_AXIS_NAME), mesh=self.mesh,
                input_partition_specs=(spec,) * 6, output_partition_specs=spec)
            self._gather_ops[key] = op
        return op

    @torch.no_grad()
    def _gather_replicas(self, layers) -> None:
        """Copy each slot's expert rows from its owner's shard (collective, same on every rank)."""
        cfg = self.cfg
        dev = self.weights[0].device
        for l in layers:
            plans = REP.gather_plans(self.rep_entries[l], cfg.R, cfg.ER, cfg.KREP)
            ops = [[torch.tensor(a[self.rank:self.rank + 1], dtype=torch.int32, device=dev)
                    for a in p] for p in plans]
            for k_, src in REPLICA_KINDS.items():
                si, srow = self._w_pos[(src, l)]
                assert srow is None, src           # expert kinds alias: never stacked
                w = self.weights[si]
                i, row = self._w_pos[(k_, l)]
                out = self.weights[i]
                if row is None:
                    op = self._gather_op(w.shape, out.shape, w.dtype)
                    for q_ops in ops:
                        out = op(w, out, *q_ops)
                else:          # the slot rows of layer l inside the donated [n, ...] stack
                    op = self._gather_stk_op(w.shape, out.shape, w.dtype)
                    rt = self._row_t(row, dev)
                    for q_ops in ops:
                        out = op(out, w, rt, *q_ops)
                self.weights[i] = out.detach()

    def _gather_stk_op(self, src_shape, stk_shape, dtype):
        key = ("stk", tuple(src_shape), tuple(stk_shape), str(dtype))
        op = self._gather_ops.get(key)
        if op is None:
            spec, sspec = P(EP_AXIS_NAME), P(None, EP_AXIS_NAME)
            op = sharded_jax_callable(
                f"glm5_megakernel::replica_gather_stk_{len(self._gather_ops)}",
                REP.make_gather_stk_fn(self.mesh, EP_AXIS_NAME), mesh=self.mesh,
                input_partition_specs=(sspec, spec, P()) + (spec,) * 4,
                output_partition_specs=sspec, donate_argnums=(0,))
            self._gather_ops[key] = op
        return op

    def _collect_weights(self) -> list[torch.Tensor]:
        """Operands in kernel.build's w_names order: per layer, or one [n_layers, ...] tensor for a
        kind in self.stack_w. _w_pos[(kind, l)] = (operand index, row in the stack or None)."""
        per_layer = {}
        out = []
        self.w_specs = []
        self._w_index = []
        self._w_pos = {}
        for kind, (layers, shape, dtype) in self.wk.items():
            want = {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16,
                    "float32": torch.float32}[jax.numpy.dtype(dtype).name]
            spec = P(EP_AXIS_NAME) if kind in SHARDED_KINDS else P()
            stk = None
            if kind in self.stack_w and kind in REPLICA_KINDS:   # zeros; _gather_replicas fills
                stk = torch.zeros((len(layers),) + tuple(shape), dtype=want,
                                  device=self.embed.device)
            for row, l in enumerate(layers):
                if kind in self.stack_w:
                    self._w_pos[(kind, l)] = (len(out), row)
                    if kind in REPLICA_KINDS:
                        continue
                else:
                    self._w_pos[(kind, l)] = (len(out), None)
                    self._w_index.append((kind, l))
                if l not in per_layer:
                    per_layer[l] = self._layer_weights(l)
                # pop: a stacked kind's per-layer copy is freed once its row is written
                t = per_layer[l].pop(kind)
                _need(tuple(t.shape) == tuple(shape), f"{kind}[{l}]: {tuple(t.shape)} != {shape}")
                _need(t.dtype == want, f"{kind}[{l}]: {t.dtype} != {want}")
                if kind in self.stack_w:
                    if stk is None:
                        stk = torch.zeros((len(layers),) + tuple(shape), dtype=want, device=t.device)
                    stk = self._put_op(kind)(stk, t, self._row_t(row, t.device))
                    del t
                else:
                    out.append(t)
                    self.w_specs.append(spec)
            if kind in self.stack_w:
                self._w_index.append((kind, "stk"))
                out.append(stk)
                self.w_specs.append(P(None, *spec) if kind in SHARDED_KINDS else P())
        return out

    def _rebind_views(self) -> None:
        """VIEW_KINDS in stack_w: the stock parameter becomes its row of the stack, so the per-layer
        tensor is freed (before profile_run sizes the KV pool) and no byte is duplicated."""
        from vllm_torchtpu.layers.adapter.attention import _build_w_uv_stack_op

        for kind in sorted(set(VIEW_KINDS) & self.stack_w):
            assert kind == "w_uv", kind
            layers = self.wk[kind][0]
            i = self._w_index.index((kind, "stk"))
            stk = self.weights[i]
            for row, l in enumerate(layers):
                at = _attn_module(self.backbone.layers[l].self_attn)
                old = at.W_UV
                _need(tuple(old.shape) == tuple(stk.shape[1:]) and old.dtype == stk.dtype,
                      f"W_UV[{l}] {tuple(old.shape)} {old.dtype} vs stack row {tuple(stk.shape[1:])}")
                # the stock layer reads (stack, row) in one jax op (_w_uv_bmm_stacked); W_UV
                # stays as a row view for shape / VERIFY readers only: a view passed to an op is
                # materialized by torch_tpu as a copy of stack[:row + 1] (probe 35, j-1f77eaac)
                at.W_UV = torch.nn.Parameter(stk[row], requires_grad=False)
                at.W_UV_stack = (stk, row)
                _build_w_uv_stack_op(at)
            logger.info("GLM-5.2 megakernel: %s stacked %s; stock W_UV reads its row of the stack",
                        kind, tuple(stk.shape))

    def _build_smalls(self) -> torch.Tensor:
        cfg = self.cfg
        lay, S = smalls_layout(cfg)
        rows = []
        for l in range(cfg.L + 1):
            fields = self._small_fields(l) if l < cfg.L else {
                "in_ln": self.backbone.norm.weight}
            dev = next(iter(fields.values())).device
            parts = []
            for name, (_, n) in lay.items():
                v = fields.get(name)
                if v is None:
                    v = torch.zeros(n, dtype=torch.float32, device=dev)
                v = v.reshape(-1).float()
                _need(v.numel() == n, f"smalls {name}[{l}]: {v.numel()} != {n}")
                parts.append(torch.nn.functional.pad(v, (0, rup(n) - n)))
            rows.append(torch.cat(parts))
        return torch.stack(rows).reshape(cfg.L + 1, 1, S).contiguous()

    def _small_fields(self, l: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        L = self.backbone.layers[l]
        sa = L.self_attn
        f = {
            "in_ln": L.input_layernorm.weight,
            "post_ln": L.post_attention_layernorm.weight,
            "q_a_ln": sa.q_a_layernorm.weight,
            "kv_a_ln": sa.kv_a_layernorm.weight,
            "qkv_a_s": _scale(sa.fused_qkv_a_proj, cfg.QKVA, "fused_qkv_a_proj"),
            "q_b_s": _scale(sa.q_b_proj, cfg.NH * cfg.DQ, "q_b_proj"),
            "o_proj_s": _scale(sa.o_proj, cfg.H, "o_proj"),
        }
        for norm in (L.input_layernorm, L.post_attention_layernorm, sa.q_a_layernorm,
                     sa.kv_a_layernorm):
            _need(abs(float(getattr(norm, "variance_epsilon", cfg.eps)) - cfg.eps) < 1e-12,
                  f"layer {l}: rmsnorm eps")
        idx = getattr(sa, "indexer", None)
        if idx is not None:
            f["idx_wq_s"] = _scale(idx.wq_b, cfg.IH * cfg.IDIM, "indexer.wq_b")
            f["idx_kn_w"] = idx.k_norm.weight
            f["idx_kn_b"] = idx.k_norm.bias
        mlp = L.mlp
        if cfg.is_moe(l):
            b = mlp.gate.e_score_correction_bias
            f["gate_bias"] = b.float()[self._perm(l).to(b.device)]
            f["sw13_s"] = _scale(mlp.shared_experts.gate_up_proj, 2 * cfg.IS, "shared.gate_up_proj")
            f["sw2_s"] = _scale(mlp.shared_experts.down_proj, cfg.H, "shared.down_proj")
            if cfg.KREP:
                rep_e, rep_s = REP.table(self.rep_entries[l])
                f["rep_e"] = torch.from_numpy(rep_e).to(b.device)
                f["rep_s"] = torch.from_numpy(rep_s).to(b.device)
        else:
            f["w2_s"] = _scale(mlp.down_proj, cfg.H, "dense.down_proj")
        return f

    def _install_layer_taps(self) -> None:
        """Wrap each stock decoder layer (and indexer) so the stock forward copies row 0 of its
        (hidden, residual) output and its top-k set into fixed buffers. Installed at model load,
        before any compile, so the copies are traced into the stock graph."""
        cfg = self.cfg
        dev = self.embed.device
        ref_s = torch.zeros(cfg.L, 2, cfg.H, dtype=torch.float32, device=dev)
        ref_tk = torch.full((cfg.L, cfg.KT), -1, dtype=torch.int32, device=dev)
        self.ref_s, self.ref_tk = ref_s, ref_tk
        for l in range(cfg.L):
            layer = self.backbone.layers[l]

            def fwd(*a, _o=layer.forward, _l=l, **kw):
                out = _o(*a, **kw)
                ref_s[_l, 0].copy_(out[0][0].float())
                ref_s[_l, 1].copy_(out[1][0].float())
                return out

            layer.forward = fwd
            idx = getattr(layer.self_attn, "indexer", None)
            if idx is not None:
                def ifwd(*a, _o=idx.forward, _l=l, **kw):
                    out = _o(*a, **kw)
                    ref_tk[_l].copy_(out[0, : cfg.KT].to(torch.int32))
                    return out

                idx.forward = ifwd
        logger.info("GLM-5.2 megakernel: layer-diff taps on %d layers", cfg.L)

    # ----------------------------------------------------------------- caches

    def bind_caches(self, runner) -> None:
        """After KV init: collect the caches in the kernel's order and build the op."""
        cfg = self.cfg
        groups = runner.kv_cache_config.kv_cache_groups
        group_of = {n: g for g, grp in enumerate(groups) for n in grp.layer_names}
        # GLM5_MK_STACK_C: the runner allocated these kinds as one [n_layers, nb, ...] tensor
        # each (kv_cache_manager._allocate_glm5_stacked_kv) and bound it to every layer of the
        # kind with module.kv_stack = (row, nb); the kernel takes the stack as one operand
        stk = getattr(runner, "glm5_kv_stack", None) or {}
        self.stack_c = frozenset(stk)
        if self.stack_c:
            logger.info("GLM-5.2 megakernel: stacked cache operands %s", sorted(self.stack_c))
        per_layer = []
        names = set()
        lname: dict[str, dict[int, str]] = {"nope": {}, "rope": {}, "idx": {}}
        rows: dict[str, dict[int, int]] = {"nope": {}, "rope": {}, "idx": {}}
        shapes = []
        for l in range(cfg.L):
            sa = self.backbone.layers[l].self_attn
            at = _attn_module(sa)
            kv = _unwrap(at.kv_cache)
            _need(isinstance(kv, (tuple, list)) and len(kv) == 2,
                  f"layer {l}: MLA cache is not the (nope, rope) SPARSECORE pair")
            d = {"nope": kv[0], "rope": kv[1]}
            sh = {"nope": tuple(kv[0].shape), "rope": tuple(kv[1].shape)}
            lname["nope"][l] = lname["rope"][l] = at.layer_name
            if "nope" in self.stack_c:
                row = getattr(at, "kv_stack", (None,))[0]
                rows["nope"][l] = rows["rope"][l] = row
                sh = {"nope": sh["nope"][1:], "rope": sh["rope"][1:]}
            names.add(at.layer_name)
            idx = getattr(sa, "indexer", None)
            if idx is not None:
                d["idx"] = _unwrap(idx.k_cache.kv_cache)
                sh["idx"] = tuple(d["idx"].shape)
                lname["idx"][l] = idx.k_cache.prefix
                if "idx" in self.stack_c:
                    rows["idx"][l] = getattr(idx.k_cache, "kv_stack", (None,))[0]
                    sh["idx"] = sh["idx"][1:]
                names.add(idx.k_cache.prefix)
            per_layer.append(d)
            shapes.append(sh)
        gids = {group_of.get(n) for n in names}
        _need(len(gids) == 1 and None not in gids,
              f"MLA and indexer caches span KV cache groups {gids}; the kernel takes one block table")
        self.mla_layer_name = _attn_module(self.backbone.layers[0].self_attn).layer_name
        nb = shapes[0]["nope"][0]
        # ROPE cache width follows the allocation (TPU_SPARSE_MLA_ROPE_PACKED): 64 B/token
        # packed, [nb, page/8, 128], or 128 B/token padded, [nb, page/4, 128].
        rshape = tuple(shapes[0]["rope"])
        self.rope_packed = rshape == tuple(K.cache_kinds(cfg, nb, True)["rope"][1])
        ck = K.cache_kinds(cfg, nb, self.rope_packed)
        logger.info("GLM-5.2 megakernel: %s RoPE cache %s", "PACKED" if self.rope_packed
                    else "padded", rshape)
        caches = []
        self.cache_rows = {}
        for kind, (layers, shape, dtype) in ck.items():
            if kind in self.stack_c:
                # the runner's stacked tensor itself (never a view): row i is layers[i]
                t, row_names = stk[kind]
                _need(tuple(t.shape) == (len(layers),) + tuple(shape),
                      f"stacked cache {kind}: {tuple(t.shape)} != {(len(layers),) + tuple(shape)}")
                _need(list(row_names) == [lname[kind][l] for l in layers],
                      f"stacked cache {kind}: row order {row_names[:4]}... is not the kernel's")
                for i, l in enumerate(layers):
                    _need(per_layer[l][kind] is t and rows[kind][l] == i,
                          f"layer {l}: {kind} is not bound to row {i} of the stack")
                    self.cache_rows[(kind, l)] = (len(caches), i)
                caches.append(t)
                continue
            for l in layers:
                t = per_layer[l][kind]
                _need(tuple(t.shape) == tuple(shape), f"cache {kind}[{l}]: {tuple(t.shape)} != {shape}")
                self.cache_rows[(kind, l)] = (len(caches), None)
                caches.append(t)
        self.caches = caches
        self.nb = nb
        self._step = self._build_op()

    def _build_op(self, skip: frozenset = frozenset(), tag: str = ""):
        """The compiled step; `skip` (kernel.ABLATIONS) and `tag` are for GLM5_MEGAKERNEL_BISECT."""
        cfg = self.cfg
        dbg = self.layerdiff and not skip
        call, info = K.build(cfg, nb=self.nb, max_pages=self.max_pages, tiles=K.Tiles(trace=envs.GLM5_MEGAKERNEL_TRACE),
                             debug=dbg, skip=skip, rope_packed=self.rope_packed,
                             # the served kinds: owned weights (_stack_knobs), and the caches
                             # the runner allocated stacked (bind_caches)
                             stack_w=self.stack_w, stack_c=self.stack_c)
        self.info = info
        assert list(info["w_names"]) == self._w_index, "weight operand order"
        n_w = len(self.weights)
        n_c = len(self.caches)
        assert n_w == len(info["w_names"]) and n_c == len(info["c_names"])
        A = P(EP_AXIS_NAME)
        # Per-rank operands are concatenated along axis 0 globally, so each
        # rank's torch tensor is its shard unmodified and the body sees the
        # kernel's per-rank shapes directly.
        in_specs = (A, A, A, A, P()) + tuple(self.w_specs) + (A,) * n_c
        out_specs = (A,) * (1 + n_c + (2 if dbg else 0))

        def body(*args):
            return tuple(call(*args))

        smap = jax.shard_map(body, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
                             check_vma=False)
        n_in = 5 + n_w + n_c
        args = [f"x{i}" for i in range(n_in)]
        ret = "tuple[" + ", ".join(["jax.Array"] * len(out_specs)) + "]"
        src = (f"def glm5_megakernel({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
               f"    return _smap({', '.join(args)})\n")
        ns: dict[str, Any] = {"jax": jax, "_smap": smap}
        exec(compile(src, "<glm5_megakernel>", "exec"), ns)  # noqa: S102 - arity is data
        name = "glm5_megakernel::step" + tag
        flat = sharded_jax_callable(
            name, ns["glm5_megakernel"], mesh=self.mesh,
            input_partition_specs=in_specs, output_partition_specs=out_specs,
            donate_argnums=tuple(range(n_in - n_c, n_in)))

        # The dispatcher caps a schema at 64 arguments and this has ~1150, so
        # the op takes the weights and caches as two Tensor[] arguments and
        # hands the flat callable its operands positionally.
        @torch.library.custom_op(name, mutates_args=())
        def op(meta: torch.Tensor, bt: torch.Tensor, h0: torch.Tensor, rt: torch.Tensor,
               smalls: torch.Tensor, weights: list[torch.Tensor],
               caches: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(flat(meta, bt, h0, rt, smalls, *weights, *caches))

        @op.register_fake
        def _(meta, bt, h0, rt, smalls, weights, caches):
            extra = []
            if dbg:
                extra = [torch.empty((2 * cfg.L, cfg.H), dtype=torch.float32, device=h0.device),
                         torch.empty((cfg.L, cfg.KT // 128, 128), dtype=torch.int32,
                                     device=h0.device)]
            return [torch.empty_like(h0)] + [torch.empty_like(c) for c in caches] + extra

        if not skip:
            self.op = op
        if tag == "_bisect0":
            self._fp = (flat, in_specs)

        page = self.page
        max_pages = self.max_pages
        DN, DR = cfg.DN, cfg.DR

        # Every input is sliced to a fixed shape before this is called, so the
        # (minutes-long) compile happens once, at warmup, whatever bucket the
        # runner padded the step to.
        def step(ids, pos, busy, bt, rank_t, embed, cos_sin, smalls, weights, caches):
            pos = pos.to(torch.int32)
            seq = (pos + 1) * busy
            npages = (seq + (page - 1)) // page
            # The rank comes in as a tensor: a rank constant in this graph made the ranks'
            # executables differ (j-c4e6dce4: rank 1 halted at its own PC), and a remote DMA names
            # the receiver's semaphore by the sender's allocation, so peers' all-gather counts
            # landed on other ranks' pool_sem / g_sem. (lax.axis_index in the body instead lowers
            # to partition-id, which the SPMD partitioner rejects: j-a28e43a5.)
            meta = torch.cat([rank_t, pos, seq, npages])
            h0 = torch.nn.functional.pad(embed[ids.long()], (0, 0, 0, K.M - 1))
            cs = cos_sin[pos.long()].reshape(-1).float()
            c = cs[: DR // 2].repeat_interleave(2)
            s = torch.stack([-cs[DR // 2:], cs[DR // 2:]], -1).reshape(-1)
            one = torch.ones(256, dtype=torch.float32, device=cs.device)
            zero = torch.zeros(256, dtype=torch.float32, device=cs.device)
            rt = torch.stack([
                torch.cat([one[:DN], c, one[DN + DR:]]),
                torch.cat([zero[:DN], s, zero[DN + DR:]]),
                torch.cat([c, one[DR:]]),
                torch.cat([s, zero[DR:]]),
                zero, zero, zero, zero])
            outs = op(meta, bt, h0, rt, smalls, list(weights), list(caches))
            for dst, src in zip(caches, outs[1:1 + n_c]):
                dst.copy_(src)
            if dbg:
                return outs[0], outs[1 + n_c], outs[2 + n_c]
            return outs[0]

        return torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)

    # ------------------------------------------------------------------- run

    def _metadata(self):
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(self.mla_layer_name)
        assert isinstance(md, AttentionMetadata), type(md)
        return md

    @torch.no_grad()
    def _run(self, ids, pos, busy, bt, step=None):
        return (step or self._step)(ids, pos, busy, bt, self.rank_t, self.embed,
                                    self.rotary.cos_sin_cache.detach(), self.smalls,
                                    self.weights, self.caches)

    def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor, idle: bool,
                 num_tokens: int) -> torch.Tensor:
        """Hidden states [num_tokens, H] after the final norm; row 0 is this rank's token."""
        md = self._metadata()
        bt = md.block_tables.reshape(md.seq_lens.shape[0], -1)[0]
        if bt.shape[0] >= self.max_pages:
            bt = bt[: self.max_pages]
        else:
            bt = torch.nn.functional.pad(bt, (0, self.max_pages - bt.shape[0]))
        busy = torch.tensor([0 if idle else 1], dtype=torch.int32).to(input_ids.device)
        out = self._run(input_ids[0:1], positions[0:1], busy, bt.to(torch.int32))
        if self.layerdiff:
            out, *self.dbg = out
        if num_tokens <= K.M:
            return out[:num_tokens]
        return torch.nn.functional.pad(out, (0, 0, 0, num_tokens - K.M))

    def _row_block(self, pos: int):
        md = self._metadata()
        bt = md.block_tables.reshape(md.seq_lens.shape[0], -1)[0]
        return int(bt[pos // self.cfg.page].item()), pos % self.cfg.page

    def _layer_caches(self):
        """[((kind, layer), that layer's cache)] in c_names order, a stacked kind expanded to its
        rows (VERIFY only: row views, never passed to a compiled graph)."""
        out = []
        for (kind, l), (i, row) in self.cache_rows.items():
            c = self.caches[i]
            out.append(((kind, l), c if row is None else c[row]))
        return out

    @torch.no_grad()
    def snapshot_block(self, pos: int) -> list[torch.Tensor]:
        """VERIFY: every cache's block holding `pos`, after the stock forward wrote row `pos`."""
        blk, _ = self._row_block(pos)
        return [c[blk].clone() for _, c in self._layer_caches()]

    @torch.no_grad()
    def diff_block(self, snap: list[torch.Tensor], pos: int) -> str:
        """VERIFY: compare the kernel's rewrite of row `pos` with the stock write in `snap`.

        The stock forward writes row pos first and the kernel rewrites it, so both forwards read the
        kernel's rows afterwards and the hidden-state check is blind to a wrong cache value. This
        reads the rows directly: per kind, the byte-mismatch fraction and dequantized rel error of
        this token's rows, the worst layer, and whether ANY other row of the block changed."""
        blk, off = self._row_block(pos)
        g = off // 4
        rseg = 8 if self.rope_packed else 4
        rg, rw = off // rseg, 128 // rseg
        acc = {}
        for ((kind, l), c), s in zip(self._layer_caches(), snap):
            now = c[blk]
            if kind == "nope":
                a, b = now[off], s[off]
                other = torch.cat([now[:off], now[off + 1:]]).ne(
                    torch.cat([s[:off], s[off + 1:]])).sum()
            elif kind == "rope":
                sl_ = slice((off % rseg) * rw, (off % rseg + 1) * rw)
                a, b = now[rg, sl_], s[rg, sl_]
                other = now.ne(s).sum() - a.ne(b).sum()
            else:  # idx: bytes only, the packed layout mixes fp8 and scales
                a, b = now[g, off % 4], s[g, off % 4]
                other = now.ne(s).sum() - a.ne(b).sum()
            # fp8 bytes in a band-permuted order: the norm of a difference does not care
            a, b = a.cpu().contiguous(), b.cpu().contiguous()
            ba, bb = (x.view(torch.uint8) if x.dtype != torch.uint8 else x for x in (a, b))
            mis = float(ba.ne(bb).float().mean())
            if kind == "idx":
                rel = 0.0
            else:
                fa, fb = (x.view(torch.float8_e4m3fn).float() for x in (ba, bb))
                rel = float((fa - fb).norm() / fb.norm().clamp_min(1e-30))
            st = [mis, rel, int(other.item())]
            e = acc.setdefault(kind, [0, 0.0, 0.0, -1, 0])
            e[0] += 1
            e[1] = max(e[1], st[0])
            if st[1] > e[2]:
                e[2], e[3] = st[1], l
            e[4] += int(st[2])
        return " ".join(f"{k}:n={n},max_mis={m:.4f},max_rel={r:.5f}@L{wl},other_rows_changed={o}"
                        for k, (n, m, r, wl, o) in acc.items())

    def _program_fingerprint(self) -> None:
        """Serving-hang diag (j-c4e6dce4, j-bf0ef082: one rank halts at its own PC, i.e. runs a
        different executable). Per rank: the XLA-relevant environment, and a hash of every dumped
        HLO module holding a Mosaic call (whole file, and the custom-call lines alone), so ranks
        can be compared line by line in the log. Needs --xla_dump_to in LIBTPU_INIT_ARGS or XLA_FLAGS."""
        import hashlib
        import os
        import re

        def h(b):
            return hashlib.sha256(b).hexdigest()[:12]
        tag = f"GLM5_MEGAKERNEL_FP rank {self.rank}"
        for k in sorted(os.environ):
            if re.match(r"(TPU|XLA|LIBTPU|JAX|MEGASCALE|PJRT|TORCH_TPU|CLOUD_TPU)", k):
                print(f"{tag} env {k}={os.environ[k][:300]}", file=sys.stderr, flush=True)
        # operand signature per weight kind / cache kind: shape, dtype, stride, storage offset,
        # device -- a per-rank layout difference would compile a different executable
        groups: dict[str, list[str]] = {}

        def sig(t):
            return f"{tuple(t.shape)}|{t.dtype}|{t.stride()}|{t.storage_offset()}|{t.device}"
        for (k, l), t in zip(self.info["w_names"], self.weights):
            groups.setdefault(f"w:{k}", []).append(f"{l}:{sig(t)}")
        for (k, l), t in zip(self.info["c_names"], self.caches):
            groups.setdefault(f"c:{k}", []).append(f"{l}:{sig(t)}")
        for nm, t in (("smalls", self.smalls), ("embed", self.embed), ("rank_t", self.rank_t),
                      ("cos_sin", self.rotary.cos_sin_cache)):
            groups[nm] = [sig(t)]
        for g, v in groups.items():
            print(f"{tag} sig {g} n {len(v)} {h(chr(10).join(v).encode())} first {v[0][:160]}",
                  file=sys.stderr, flush=True)
        print(f"{tag} sig ALL {h(repr(sorted(groups.items())).encode())} nb {self.nb} "
              f"max_pages {self.max_pages} specs {h(repr(self.w_specs).encode())}",
              file=sys.stderr, flush=True)
        # the exact served module (SPMD, Mosaic payload included), exported like the op's fake impl
        try:
            from torch_tpu._internal.pallas import pallas as pallas_impl
            flat, in_specs = self._fp
            dev = self.smalls.device
            small = [torch.zeros(4, dtype=torch.int32, device=dev),
                     torch.zeros(self.max_pages, dtype=torch.int32, device=dev),
                     torch.zeros(K.M, self.cfg.H, dtype=self.embed.dtype, device=dev),
                     torch.zeros(8, 256, dtype=torch.float32, device=dev)]
            jax_args = pallas_impl.jax_placeholders(
                small + [self.smalls, *self.weights, *self.caches], mesh=self.mesh,
                partition_specs=in_specs)
            with jax._src.config.export_ignore_forward_compatibility(True):
                txt = flat.exported(*jax_args).mlir_module()
            cc = [ln for ln in txt.split("\n") if "tpu_custom_call" in ln]
            print(f"{tag} exported {h(txt.encode())} len {len(txt)} cc {h(chr(10).join(cc).encode())} "
                  f"n_cc {len(cc)}", file=sys.stderr, flush=True)
            for i, ln in enumerate(cc[:4]):
                print(f"{tag} cc{i} len {len(ln)} {h(ln.encode())}", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001 - diag only
            print(f"{tag} exported FAILED {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
        m = re.search(r"--xla_dump_to=(\S+)", os.environ.get("LIBTPU_INIT_ARGS", "") + " " + os.environ.get("XLA_FLAGS", ""))
        if not m or not os.path.isdir(m.group(1)):
            print(f"{tag} no dump dir", file=sys.stderr, flush=True)
            return
        for root, _, files in os.walk(m.group(1)):
            for f in sorted(files):
                path = os.path.join(root, f)
                try:
                    data = open(path, "rb").read()
                except OSError:
                    continue
                if b"tpu_custom_call" not in data:
                    continue
                cc = b"\n".join(ln for ln in data.split(b"\n") if b"tpu_custom_call" in ln)
                print(f"{tag} file {f} size {len(data)} all {h(data)} cc {h(cc)} n_cc {cc.count(b'tpu_custom_call')}",
                      file=sys.stderr, flush=True)

    def warmup(self, runner) -> None:
        """Compile and run once as an idle step on every rank (seq_len 0 touches no cache)."""
        from vllm_torchtpu.utils import synchronize_tensors

        z = torch.zeros(self.max_pages, dtype=torch.int32).to(runner.device)
        for i, sk in enumerate(s for s in envs.GLM5_MEGAKERNEL_BISECT.split("/") if s):
            skip = frozenset(set(K.ABLATIONS) - {"agdiag"} if sk == "all" else sk.split(","))
            # stderr: Ray drops worker logger.info lines
            print(f"GLM5_MEGAKERNEL_BISECT rank {self.rank} skip={sk} start", file=sys.stderr,
                  flush=True)
            t0 = time.monotonic()
            step = self._build_op(skip, f"_bisect{i}")
            if i == 0:     # before the run: a halt must not hide it
                self._program_fingerprint()
            out = self._run(z[0:1], z[0:1], z[0:1], z, step=step)
            synchronize_tensors([out])
            print(f"GLM5_MEGAKERNEL_BISECT rank {self.rank} skip={sk} returned in "
                  f"{time.monotonic() - t0:.1f} s", file=sys.stderr, flush=True)
            if "agdiag" in skip:   # out row 1: see kernel.ABLATIONS
                d = out[1, :K.AGDIAG_TAG0 + self.cfg.R].float().cpu().tolist()
                print(f"GLM5_MEGAKERNEL_AGDIAG rank {self.rank} {[int(v) for v in d]}",
                      file=sys.stderr, flush=True)
        out = self._run(z[0:1], z[0:1], z[0:1], z)
        if self.layerdiff:
            out = out[0]
        synchronize_tensors([out])
        logger.info("GLM-5.2 megakernel compiled and warmed (%d operands)",
                    5 + len(self.weights) + len(self.caches))
