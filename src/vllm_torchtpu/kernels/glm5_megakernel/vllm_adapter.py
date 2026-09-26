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
from vllm_torchtpu.kernels.glm5_megakernel.config import FULL
from vllm_torchtpu.kernels.glm5_megakernel.layout import rup, smalls_layout
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

EXPERT_KINDS = ("ew13", "ew2", "es13", "es2")
# operands sharded over the EP axis (the local tensor is this rank's shard): experts, and
# o_proj's head rows (TP over heads)
SHARDED_KINDS = EXPERT_KINDS + ("o_proj",)


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
            routed_scale=float(hf.routed_scaling_factor))
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
        self.cfg = cfg
        self.rank = ep_mesh_index()
        order = ep_rank_order()
        ER = cfg.ER
        self.perm = torch.tensor([order[j // ER] * ER + j % ER for j in range(cfg.E)],
                                 dtype=torch.long)

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
        with torch.no_grad():
            self.weights = [w.detach() for w in self._collect_weights()]
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
        self._step = None
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(self.weights + [self.smalls])
        logger.info("GLM-5.2 megakernel: %d weight operands, mesh index %d of %d",
                    len(self.weights), self.rank, R)

    # ---------------------------------------------------------------- weights

    def _layer_weights(self, l: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        H = cfg.H
        L = self.backbone.layers[l]
        sa = L.self_attn
        at = _attn_module(sa)
        d = {
            "qkv_a": _kn_pad(sa.fused_qkv_a_proj.weight, H, cfg.QKVA, "fused_qkv_a_proj"),
            "q_b": _kn(sa.q_b_proj.weight, cfg.QL, cfg.NH * cfg.DQ, "q_b_proj"),
            "w_uk": at.W_UK_T,
            "w_uv": at.W_UV,
            # o_proj is TP over heads: this rank's rows, heads [rank * HR, (rank + 1) * HR)
            "o_proj": _kn(sa.o_proj.weight, cfg.NH * cfg.DV, H, "o_proj")[
                self.rank * cfg.HR * cfg.DV:(self.rank + 1) * cfg.HR * cfg.DV].contiguous(),
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
            d["gate"] = g[:, self.perm.to(g.device)].contiguous()
            r = _routed(mlp)
            d["ew13"] = r.w13_weight
            d["ew2"] = r.w2_weight
            d["es13"] = _expert_scale(r, "w13_weight", (cfg.ER, 2 * cfg.I))
            d["es2"] = _expert_scale(r, "w2_weight", (cfg.ER, H))
            se = mlp.shared_experts
            d["sw13"] = _kn(se.gate_up_proj.weight, H, 2 * cfg.IS, "shared.gate_up_proj")
            d["sw2"] = _kn(se.down_proj.weight, cfg.IS, H, "shared.down_proj")
        else:
            d["w13"] = _kn(mlp.gate_up_proj.weight, H, 2 * cfg.ID, "dense.gate_up_proj")
            d["w13_s"] = _scale(mlp.gate_up_proj, 2 * cfg.ID, "dense.gate_up_proj").reshape(1, -1)
            d["w2"] = _kn(mlp.down_proj.weight, cfg.ID, H, "dense.down_proj")
        return d

    def _collect_weights(self) -> list[torch.Tensor]:
        per_layer = {}
        out = []
        self.w_specs = []
        for kind, (layers, shape, dtype) in self.wk.items():
            for l in layers:
                if l not in per_layer:
                    per_layer[l] = self._layer_weights(l)
                t = per_layer[l][kind]
                _need(tuple(t.shape) == tuple(shape), f"{kind}[{l}]: {tuple(t.shape)} != {shape}")
                want = {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16,
                        "float32": torch.float32}[jax.numpy.dtype(dtype).name]
                _need(t.dtype == want, f"{kind}[{l}]: {t.dtype} != {want}")
                out.append(t)
                self.w_specs.append(P(EP_AXIS_NAME) if kind in SHARDED_KINDS else P())
        return out

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
            f["gate_bias"] = b.float()[self.perm.to(b.device)]
            f["sw13_s"] = _scale(mlp.shared_experts.gate_up_proj, 2 * cfg.IS, "shared.gate_up_proj")
            f["sw2_s"] = _scale(mlp.shared_experts.down_proj, cfg.H, "shared.down_proj")
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
        per_layer = []
        names = set()
        for l in range(cfg.L):
            sa = self.backbone.layers[l].self_attn
            at = _attn_module(sa)
            kv = _unwrap(at.kv_cache)
            _need(isinstance(kv, (tuple, list)) and len(kv) == 2,
                  f"layer {l}: MLA cache is not the (nope, rope) SPARSECORE pair")
            d = {"nope": kv[0], "rope": kv[1]}
            names.add(at.layer_name)
            idx = getattr(sa, "indexer", None)
            if idx is not None:
                d["idx"] = _unwrap(idx.k_cache.kv_cache)
                names.add(idx.k_cache.prefix)
            per_layer.append(d)
        gids = {group_of.get(n) for n in names}
        _need(len(gids) == 1 and None not in gids,
              f"MLA and indexer caches span KV cache groups {gids}; the kernel takes one block table")
        self.mla_layer_name = _attn_module(self.backbone.layers[0].self_attn).layer_name
        nb = per_layer[0]["nope"].shape[0]
        ck = K.cache_kinds(cfg, nb)
        caches = []
        for kind, (layers, shape, dtype) in ck.items():
            for l in layers:
                t = per_layer[l][kind]
                _need(tuple(t.shape) == tuple(shape), f"cache {kind}[{l}]: {tuple(t.shape)} != {shape}")
                caches.append(t)
        self.caches = caches
        self.nb = nb
        self._step = self._build_op()

    def _build_op(self, skip: frozenset = frozenset(), tag: str = ""):
        """The compiled step; `skip` (kernel.ABLATIONS) and `tag` are for GLM5_MEGAKERNEL_BISECT."""
        cfg = self.cfg
        dbg = self.layerdiff and not skip
        call, info = K.build(cfg, nb=self.nb, max_pages=self.max_pages, tiles=K.Tiles(),
                             debug=dbg, skip=skip)
        self.info = info
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
