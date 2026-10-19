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
"""vLLM TPU adapter for the Qwen3.5 prefill megakernel at a pipeline stage (TP1).

VLLM_TPU_QWEN35_PREFILL_MK=1 routes every prefill-only step of a pipeline stage to one
pallas_call that runs the stage's layers (megakernel.build_stage): the runner's packed tokens
are gathered into the kernel's padded segment layout, the kernel reads and writes the KV pages
and GDN states of the stage's region pools in place, and the outputs are gathered back into
the runner's layout as the stage's IntermediateTensors (or, on the last stage, the final-norm
hidden states). Steps it cannot take (decodes, a context that is not a multiple of 128, more
tokens than the largest bucket) run the stock model.

VLLM_TPU_QWEN35_PREFILL_MK_BUCKETS: padded-token buckets (default "4096,8192,12288").
VLLM_TPU_QWEN35_PREFILL_MK_MAX_SEQS: sequences per step (default 16).
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_callable
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import (
    _build_tp_rank_mesh,
    _find_backbone,
    _fresh,
    extract_kn_fp8_weight,
)
from vllm_torchtpu.kernels.qwen35_prefill_mk import megakernel as MK
from vllm_torchtpu.kernels.qwen35_prefill_mk import phases as P
from vllm_torchtpu.kernels.qwen35_prefill_mk import pool_plan
from vllm_torchtpu.kernels.qwen35_prefill_mk.reference import StageConfig
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PAGE = P.PAGE


def _is_missing(layer) -> bool:
    return type(layer).__name__ == "PPMissingLayer"


def _t(x) -> torch.Tensor:
    return _fresh(x.detach().contiguous())


class Qwen35PrefillMegakernel:
    """Prefill steps of one pipeline stage through the stage megakernel."""

    @classmethod
    def maybe_create(cls, runner: Any) -> Qwen35PrefillMegakernel | None:
        if os.environ.get("VLLM_TPU_QWEN35_PREFILL_MK", "0") != "1":
            return None
        if int(runner.parallel_config.tensor_parallel_size) != 1:
            logger.warning("Qwen3.5 prefill megakernel disabled: needs TP1 (pipeline stages)")
            return None
        try:
            backbone = _find_backbone(runner.model)
        except ValueError as exc:
            logger.warning("Qwen3.5 prefill megakernel disabled: %s", exc)
            return None
        local = [i for i, layer in enumerate(backbone.layers) if not _is_missing(layer)]
        if not local or local != list(range(local[0], local[-1] + 1)):
            logger.warning("Qwen3.5 prefill megakernel disabled: no contiguous local layers")
            return None
        return cls(runner, backbone, local)

    def __init__(self, runner: Any, backbone: Any, local: list[int]):
        self.backbone = backbone
        self.local = local
        self.first = local[0] == 0
        self.last = local[-1] == len(backbone.layers) - 1
        self.layers = [backbone.layers[i] for i in local]
        self.types = tuple("attn" if getattr(l, "layer_type", "") == "full_attention" else "gdn"
                           for l in self.layers)
        self.c = StageConfig(layer_types=self.types)
        self.buckets = tuple(sorted(int(b) for b in os.environ.get(
            "VLLM_TPU_QWEN35_PREFILL_MK_BUCKETS", "4096,8192,12288").split(",") if b))
        self.max_seqs = int(os.environ.get("VLLM_TPU_QWEN35_PREFILL_MK_MAX_SEQS", "16"))
        self.mesh = _build_tp_rank_mesh(1)
        self._steps: dict[int, Any] = {}
        self.bound = False
        self.n_run = self.n_fallback = 0
        logger.info("Qwen3.5 prefill megakernel: stage layers %d..%d (%s), buckets %s",
                    local[0], local[-1], ",".join(self.types), self.buckets)

    # ------------------------------------------------------------------ binding
    def bind_caches(self, runner: Any) -> None:
        """Capture the stage's weights in the kernel layouts and its region pools."""
        c = self.c
        regions, layer_region, layer_group, self.pools = {}, [], [], []
        k_scales, v_scales = [], []
        self.weights = []
        attn_names, mamba_names = [], {}
        for li, (gi, layer) in enumerate(zip(self.local, self.layers)):
            region = gi // 4
            if region not in regions:
                regions[region] = len(regions)
                kv = (layer.self_attn.attn.kv_cache if self.types[li] == "attn"
                      else layer.linear_attn.kv_cache)
                pool = kv[0] if isinstance(kv, (list, tuple)) else kv
                self.pools.append(pool)
            layer_region.append(regions[region])
            layer_group.append(gi % 4 if self.types[li] == "gdn" else 0)
            if self.types[li] == "attn":
                attn = layer.self_attn
                k_scales.append(float(getattr(attn.attn, "_k_scale_float", 1.0) or 1.0))
                v_scales.append(float(getattr(attn.attn, "_v_scale_float", 1.0) or 1.0))
                attn_names.append(attn.attn.layer_name)
            else:
                mamba_names.setdefault(gi % 4, layer.linear_attn.prefix)
            self.weights.append(self._layer_weights(layer, self.types[li]))
        self.layout = MK.StageLayout(self.types, tuple(layer_region), tuple(layer_group),
                                     tuple(k_scales), tuple(v_scales))
        self.attn_name = attn_names[0]
        self.mamba_names = [mamba_names.get(g, next(iter(mamba_names.values())))
                            for g in range(3)]
        pool0 = self.pools[0]
        self.pool_shape = tuple(pool0.shape)
        self.block_tokens = int(runner.cache_config.block_size)
        jdt = jnp.float8_e4m3fn if pool0.dtype == torch.float8_e4m3fn else jnp.dtype(str(pool0.dtype).split(".")[-1])
        self.plan, self.stride = pool_plan.build_gdn_plan(
            self.pool_shape, jdt, conv_dim=c.QKV, n_v=c.VH, d_k=c.KD, d_v=c.VD,
            kernel_size=c.CONV, pool_block_tokens=self.block_tokens)
        self.pps = int(runner.model_config.max_model_len + PAGE - 1) // PAGE
        rope = MK.rope_table(c, self.pps * PAGE)
        self.rope = _t(torch.from_numpy(np.asarray(rope)).to(pool0.device))
        if self.first:
            self.embed = self.backbone.embed_tokens.weight
        if self.last:
            self.final_norm = self.backbone.norm.weight
        self.w_flat = [w[n] for li, w in enumerate(self.weights)
                       for n in ((MK.GDN_W if self.types[li] == "gdn" else MK.ATT_W) + MK.MOE_W)]
        self.bound = True
        logger.info("Qwen3.5 prefill megakernel bound: %d regions, pool %s, block %d tokens, "
                    "plan stride %d, pps %d", len(self.pools), self.pool_shape, self.block_tokens,
                    self.stride, self.pps)

    def _layer_weights(self, layer, lt):
        c, H = self.c, self.c.H
        w = {"in_norm": _t(layer.input_layernorm.weight.reshape(1, H)),
             "post_norm": _t(layer.post_attention_layernorm.weight.reshape(1, H))}
        if lt == "gdn":
            la = layer.linear_attn
            qw, qs = extract_kn_fp8_weight(la.in_proj_qkvz, H, c.QKV + c.Z)
            ow, os_ = extract_kn_fp8_weight(la.out_proj, c.Z, H)
            ba = la.in_proj_ba.weight.data
            ba = ba if tuple(ba.shape) == (H, 2 * c.VH) else ba.transpose(0, 1)
            w.update(qkvz_w=_t(qw), qkvz_s=_t(qs.reshape(1, -1)), ba_w=_t(ba.to(torch.bfloat16)),
                     conv_w=_t(la.conv1d.weight.data.reshape(c.QKV, c.CONV).transpose(0, 1)
                               .to(torch.bfloat16)),
                     a_log=_t(la.A_log.data.float().reshape(1, c.VH)),
                     dt_bias=_t(la.dt_bias.data.float().reshape(1, c.VH)),
                     gnorm_w=_t(la.norm.weight.data.float().reshape(1, c.VD)),
                     out_w=_t(ow), out_s=_t(os_.reshape(1, -1)))
        else:
            sa = layer.self_attn
            qw, qs = extract_kn_fp8_weight(sa.qkv_proj, H, c.ATT_QKV)
            ow, os_ = extract_kn_fp8_weight(sa.o_proj, c.NH * c.HD, H)
            w.update(qkv_w=_t(qw), qkv_s=_t(qs.reshape(1, -1)),
                     q_norm=_t(sa.q_norm.weight.reshape(1, c.HD)),
                     k_norm=_t(sa.k_norm.weight.reshape(1, c.HD)),
                     o_w=_t(ow), o_s=_t(os_.reshape(1, -1)))
        mlp = layer.mlp
        ex = getattr(mlp.experts, "routed_experts", mlp.experts)
        s13 = getattr(ex, "w13_weight_scale", None)
        s13 = s13 if s13 is not None else ex.w13_weight_scale_inv
        s2 = getattr(ex, "w2_weight_scale", None)
        s2 = s2 if s2 is not None else ex.w2_weight_scale_inv
        sd = mlp._fused_se_down_fp8_weight
        sd = sd if tuple(sd.shape) == (c.IS, H) else sd.transpose(0, 1)
        sg = mlp._fused_se_gate_weight
        sg = sg if tuple(sg.shape) == (H, 1) else sg.reshape(1, H).transpose(0, 1)
        w.update(gate_gu_w=_t(mlp._fused_gate_gu_fp8_weight),
                 gate_gu_s=_t(mlp._fused_gate_gu_fp8_scale.float().reshape(1, -1)),
                 sd_w=_t(sd), sd_s=_t(mlp._fused_se_down_fp8_scale.float().reshape(1, -1)),
                 sgate_w=_t(sg.to(torch.bfloat16)),
                 w13=ex.w13_weight.data, w13_s=_t(s13.float().reshape(c.E, 1, 2 * c.I)),
                 w2=ex.w2_weight.data, w2_s=_t(s2.float().reshape(c.E, 1, H)))
        return w

    # ------------------------------------------------------------------ steps
    def _bucket(self, padded: int) -> int | None:
        for b in self.buckets:
            if padded <= b:
                return b
        return None

    def _build_step(self, bucket: int):
        c, S, pps = self.c, self.max_seqs, self.pps
        step = MK.build_stage(c, self.layout, bucket, max_seqs=S, pps=pps,
                              pool_shape=self.pool_shape, plan=self.plan, stride=self.stride)
        n_w, n_p = len(self.w_flat), len(self.pools)
        w_names = [(MK.GDN_W if lt == "gdn" else MK.ATT_W) + MK.MOE_W for lt in self.types]
        first, last, eps = self.first, self.last, c.eps

        def fn(qsl, seq_lens, n_seq, bt, s0, s1, s2, hidden, residual, aux, norm, rope, *rest):
            """Runner layout -> padded segment layout -> megakernel -> runner layout."""
            ws_flat, pools = rest[:n_w], rest[n_w:]
            n_tok = hidden.shape[0] if not first else aux.shape[0]
            sidx = jnp.arange(S)
            live = sidx < n_seq[0]
            q_len = jnp.where(live, qsl[1:S + 1] - qsl[:S], 0)
            ctx = jnp.where(live, seq_lens[:S] - q_len, 0)
            padded = (q_len + P.SEG_ALIGN - 1) // P.SEG_ALIGN * P.SEG_ALIGN
            start = jnp.cumsum(padded) - padded
            row = jnp.arange(bucket)
            inside = ((row[:, None] >= start[None, :]) & (row[:, None] < (start + padded)[None, :])
                      & live[None, :])
            seg = jnp.any(inside, 1)
            sid = jnp.argmax(inside, 1)
            local = row - start[sid]
            valid = seg & (local < q_len[sid])
            src = jnp.where(valid, qsl[sid] + local, 0)
            tile_seg = jnp.where(seg, sid, -1).astype(jnp.int32)[::P.BLK]
            meta = {"tile_seg": tile_seg, "seg_start": start.astype(jnp.int32),
                    "seg_len": q_len.astype(jnp.int32), "seg_ctx": ctx.astype(jnp.int32),
                    "bt": bt.reshape(-1)[:S * pps].astype(jnp.int32),
                    "gslot0": s0[:S].astype(jnp.int32), "gslot1": s1[:S].astype(jnp.int32),
                    "gslot2": s2[:S].astype(jnp.int32)}
            if first:                              # aux = input ids, hidden = embedding table
                h_in = jnp.take(hidden, jnp.take(aux, src), axis=0).astype(jnp.bfloat16)
                r_in = jnp.zeros_like(h_in)
            else:
                h_in, r_in = jnp.take(hidden, src, axis=0), jnp.take(residual, src, axis=0)
            wl, k = [], 0
            for names in w_names:
                wl.append(dict(zip(names, ws_flat[k:k + len(names)])))
                k += len(names)
            h, r, new_pools = step(meta, h_in, r_in, wl, list(pools), rope)
            # padded -> runner rows: token j of sequence s sits at start[s] + j - qsl[s].
            tok = jnp.arange(n_tok)
            ts = jnp.clip(jnp.searchsorted(qsl[1:S + 1], tok, side="right"), 0, S - 1)
            back = jnp.clip(start[ts] + tok - qsl[ts], 0, bucket - 1)
            h_out, r_out = jnp.take(h, back, axis=0), jnp.take(r, back, axis=0)
            if last:                               # model.norm(hidden, residual)
                x = h_out.astype(jnp.float32) + r_out.astype(jnp.float32)
                x = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)
                h_out = (x * (1.0 + norm.reshape(1, -1).astype(jnp.float32))).astype(jnp.bfloat16)
            return (h_out, r_out, *new_pools)

        n_fixed = 12
        n_in = n_fixed + n_w + n_p
        name = f"qwen35_prefill_mk_stage{self.local[0]}_{self.local[-1]}_t{bucket}"
        # sharded_jax_callable reads the signature: every argument a jax.Array, positional.
        args = [f"x{i}" for i in range(n_in)]
        src = (f"def {name}({', '.join(a + ': jax.Array' for a in args)}) -> "
               f"tuple[{', '.join(['jax.Array'] * (2 + n_p))}]:\n"
               f"    return _fn({', '.join(args)})\n")
        ns = {"jax": jax, "_fn": fn}
        exec(compile(src, f"<{name}>", "exec"), ns)  # noqa: S102
        spec = jax.sharding.PartitionSpec()
        return sharded_jax_callable(
            f"qwen35_prefill_mk::stage{self.local[0]}_{self.local[-1]}_t{bucket}", ns[name],
            mesh=self.mesh, input_partition_specs=(spec,) * n_in,
            output_partition_specs=(spec,) * (2 + n_p),
            donate_argnums=tuple(range(n_fixed + n_w, n_in)))

    def can_run(self, input_ids, positions, intermediate_tensors=None, runner=None, **_) -> bool:
        if not self.bound:
            return False
        md = get_forward_context().attn_metadata
        if not md or self.attn_name not in md:
            return False
        attn_md = md[self.attn_name]
        del attn_md
        n = int(runner.input_batch.num_reqs)
        if n == 0 or n > self.max_seqs:
            return self._fallback("seqs")
        qsl = np.asarray(runner.query_start_loc_np[:n + 1])
        seq_lens = np.asarray(runner.seq_lens_np[:n])
        q_len = np.diff(qsl)
        ctx = seq_lens - q_len
        if (q_len <= 1).any() or (ctx % PAGE).any():
            return self._fallback("decode-or-unaligned")
        padded = int(((q_len + P.SEG_ALIGN - 1) // P.SEG_ALIGN * P.SEG_ALIGN).sum())
        if self._bucket(padded) is None:
            return self._fallback("tokens")
        return True

    def _fallback(self, why: str) -> bool:
        self.n_fallback += 1
        if self.n_fallback in (1, 10, 100, 1000):
            logger.info("Qwen3.5 prefill megakernel: stock model for a step (%s); runs %d, "
                        "fallbacks %d", why, self.n_run, self.n_fallback)
        return False

    def __call__(self, input_ids, positions, intermediate_tensors=None, runner=None):
        md = get_forward_context().attn_metadata
        attn_md = md[self.attn_name]
        n = int(runner.input_batch.num_reqs)
        qsl_np = np.asarray(runner.query_start_loc_np[:n + 1])
        q_len = np.diff(qsl_np)
        padded = int(((q_len + P.SEG_ALIGN - 1) // P.SEG_ALIGN * P.SEG_ALIGN).sum())
        bucket = self._bucket(padded)
        step = self._steps.get(bucket)
        if step is None:
            step = self._steps[bucket] = self._build_step(bucket)
        dev = self.rope.device
        S = self.max_seqs
        qsl = torch.zeros(S + 1, dtype=torch.int32, device=dev)
        qsl[:n + 1] = attn_md.query_start_loc[:n + 1].to(torch.int32)
        qsl[n + 1:] = qsl[n]
        seq_lens = torch.zeros(S, dtype=torch.int32, device=dev)
        seq_lens[:n] = attn_md.seq_lens[:n].to(torch.int32)
        n_seq = torch.tensor([n], dtype=torch.int32, device=dev)
        bt = attn_md.block_tables.reshape(-1).to(torch.int32)
        if bt.numel() < S * self.pps:
            bt = torch.nn.functional.pad(bt, (0, S * self.pps - bt.numel()))
        slots = []
        for name in self.mamba_names:
            si = md[name].mamba_state_indices.to(torch.int32).reshape(-1)
            slots.append(torch.nn.functional.pad(si, (0, max(0, S - si.numel())))[:S])
        if self.first:
            hidden, residual, aux = self.embed, self.embed[:1], input_ids.to(torch.int32)
        else:
            hidden = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            aux = n_seq
        norm = self.final_norm if self.last else self.rope[:1, :1]
        outs = step(qsl, seq_lens, n_seq, bt, *slots, hidden, residual, aux, norm, self.rope,
                    *self.w_flat, *self.pools)
        h, r = outs[0], outs[1]
        for dst, src in zip(self.pools, outs[2:]):
            dst.set_(src)
        self.n_run += 1
        if self.n_run in (1, 10, 100, 1000):
            logger.info("Qwen3.5 prefill megakernel: runs %d, fallbacks %d (last bucket %d)",
                        self.n_run, self.n_fallback, bucket)
        if self.last:
            return h
        return IntermediateTensors({"hidden_states": h, "residual": r})

