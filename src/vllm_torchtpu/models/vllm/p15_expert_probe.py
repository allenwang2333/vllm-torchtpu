# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""P15 debug probe (throwaway): how predictable is layer L's expert set from
layer L-1's states, on real served decode traffic.

TPU_P15_PROBE=1 replaces DeepseekV2Model.forward with a copy of the v0.29.0
forward that, for every MoE layer L (with a layer L-1 before it), computes in
the compiled graph, on the first TPU_P15_MAX_ROWS token rows:

  act  = top-8  of router_L(x_L)                      (x_L: layer L's MoE input)
  a    = top-32 of router_L(x_{L-1})                  (layer L-1's MoE input, its own norm)
  a2   = top-32 of router_L(post_attn_norm_L(r_mid_{L-1}))   (same residual, L's norm)
  b    = top-32 of router_L(post_attn_norm_L(r_in_L))  (r_in_L = layer L's input residual,
                                                        i.e. layer L-1's output, before L's attention)

router_L is production's: GateLinear (bf16-rounded logits) -> f32 sigmoid ->
+ e_score_correction_bias -> top-k (ungrouped; GLM-5.2 has n_group 1). The ids
go into a persistent [MAX_ROWS, n_slots, 104] int32 tensor by copy_ (the
KV-cache writeback pattern), with the positions beside them. The runner's
forward_model is wrapped so that TP rank 0 reads them back on sampled forwards
and writes npz chunks to $CDK_OUTPUT_DIR/p15. Changes no served output.
"""

import os
import time
from itertools import islice

import numpy as np
import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models import deepseek_v2
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("TPU_P15_PROBE", "0") == "1"
MAX_ROWS = int(os.environ.get("TPU_P15_MAX_ROWS", "64"))
K_PRED = 32
TOPK = 8
SKIP = int(os.environ.get("TPU_P15_SKIP", "200"))
BURST = int(os.environ.get("TPU_P15_BURST", "40"))
PERIOD = int(os.environ.get("TPU_P15_PERIOD", "300"))
MAX_STEPS = int(os.environ.get("TPU_P15_MAX_STEPS", "2500"))
CHUNK = int(os.environ.get("TPU_P15_CHUNK", "250"))
WIDTH = TOPK + 3 * K_PRED  # act8 | a32 | a2_32 | b32

UPSTREAM_MODEL_FORWARD = deepseek_v2.DeepseekV2Model.forward


def _route_ids(layer, x, k):
    """Production routing for MoE layer `layer` on rows `x`: top-k expert ids."""
    gate = layer.mlp.gate
    logits, _ = gate(x)
    scores = logits.float().sigmoid()
    biased = scores + gate.e_score_correction_bias.float().view(1, -1)
    return torch.topk(biased, k, dim=-1).indices.to(torch.int32)


def _first_rows(t):
    """t's first MAX_ROWS rows, zero-padded, with a STATIC row count: slicing
    a symbolic token count directly (t[:MAX_ROWS]) adds a Min() SymInt that
    differs between shape variants (ShapeSpecializationError)."""
    pad = t.new_zeros((MAX_ROWS,) + tuple(t.shape[1:]))
    return torch.cat([t, pad], dim=0)[:MAX_ROWS]


def _layer_forward(self, positions, hidden_states, residual, llama_4_scaling):
    """DeepseekV2DecoderLayer.forward (v0.29.0) that also returns the MoE input
    and the post-attention residual."""
    full_num_tokens = positions.shape[0]
    input_is_sequence_parallel = (
        self.use_sequence_parallel_moe
        and residual is not None
        and hidden_states.shape[0] != full_num_tokens
    )
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]
    if self.use_mha:
        hidden_states = self.self_attn(positions, hidden_states)
    else:
        hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
    if (
        not isinstance(self.self_attn, deepseek_v2.DeepseekAttention)
        and hidden_states.dtype == torch.float16
    ):
        hidden_states *= 1.0 / self.routed_scaling_factor
        if self.layer_idx == 0:
            residual *= 1.0 / self.routed_scaling_factor
    if self.use_sequence_parallel_moe:
        raise NotImplementedError("P15 probe: sequence-parallel MoE not supported")
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    mlp_in = hidden_states
    r_mid = residual
    hidden_states = self.mlp(hidden_states)
    if (
        isinstance(self.mlp, deepseek_v2.DeepseekV2MLP)
        and hidden_states.dtype == torch.float16
    ):
        hidden_states *= 1.0 / self.routed_scaling_factor
    return hidden_states, residual, mlp_in, r_mid


def _is_moe(layer) -> bool:
    return isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE)


def patched_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds=None,
):
    ids_buf = getattr(self, "_p15_ids", None)
    if ids_buf is None:
        return UPSTREAM_MODEL_FORWARD(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    pos_buf = self._p15_pos
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    if llama_4_scaling_config is not None:
        llama_4_scaling = deepseek_v2._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    slots = []
    prev = None  # (layer L-1's mlp_in, r_mid) on the first MAX_ROWS rows
    for idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        if (
            hidden_states.shape[0] != positions.shape[0]
            and not layer.use_sequence_parallel_moe
        ):
            combined_states = torch.cat([hidden_states, residual], dim=-1)
            combined_states = tensor_model_parallel_all_gather(combined_states, 0)
            combined_states = combined_states[: positions.shape[0]]
            hidden_states, residual = combined_states.split(
                [self.hidden_size, self.hidden_size], dim=-1
            )
        if idx in self.aux_hidden_state_layers:
            aux_hidden_state = hidden_states + residual
            aux_hidden_states.append(aux_hidden_state)
        r_in = None
        if prev is not None and _is_moe(layer):
            # layer L's input residual = layer L-1's output + its residual
            r_in = _first_rows(hidden_states) + _first_rows(residual)
        hidden_states, residual, mlp_in, r_mid = _layer_forward(
            layer, positions, hidden_states, residual, llama_4_scaling
        )
        if _is_moe(layer) and prev is not None:
            p_mlp_in, p_r_mid = prev
            norm = layer.post_attention_layernorm
            act = _route_ids(layer, _first_rows(mlp_in), TOPK)
            a = _route_ids(layer, p_mlp_in, K_PRED)
            a2 = _route_ids(layer, norm(p_r_mid), K_PRED)
            b = _route_ids(layer, norm(r_in), K_PRED)
            slots.append(torch.cat([act, a, a2, b], dim=-1))
        prev = (_first_rows(mlp_in), _first_rows(r_mid))

    if slots:
        ids_buf.copy_(torch.stack(slots, dim=1))  # [MAX_ROWS, n_slots, WIDTH]
        pos_buf.copy_(_first_rows(positions.to(torch.int32) + 1) - 1)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    if hidden_states.shape[0] != positions.shape[0]:
        combined_states = torch.cat([hidden_states, residual], dim=-1)
        combined_states = tensor_model_parallel_all_gather(combined_states, 0)
        combined_states = combined_states[: positions.shape[0]]
        hidden_states, residual = combined_states.split(
            [self.hidden_size, self.hidden_size], dim=-1
        )
    if self.end_layer in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


# ---------------------------------------------------------------- host side


def _target_models(model):
    return [m for m in model.modules() if isinstance(m, deepseek_v2.DeepseekV2Model)]


def _ensure_buffers(runner, device=None) -> None:
    if getattr(runner, "_p15_ready", False):
        return
    model = getattr(runner, "model", None)
    if model is None:
        return
    targets = _target_models(model)
    if not targets:
        logger.warning("P15 probe: no DeepseekV2Model in the runner's model")
        runner._p15_ready = True
        return
    for m in targets:
        layers = list(islice(m.layers, m.start_layer, m.end_layer))
        slot_layers = [
            m.start_layer + i
            for i, layer in enumerate(layers)
            if i > 0 and _is_moe(layer)
        ]
        dev = device if device is not None else next(m.parameters()).device
        m._p15_ids = torch.zeros(
            (MAX_ROWS, len(slot_layers), WIDTH), dtype=torch.int32, device=dev
        )
        m._p15_pos = torch.full((MAX_ROWS,), -1, dtype=torch.int32, device=dev)
        m._p15_slot_layers = slot_layers
        runner._p15_model = m
        logger.info(
            "P15 probe: armed %d slots (layers %d..%d), MAX_ROWS %d, device %s",
            len(slot_layers),
            slot_layers[0] if slot_layers else -1,
            slot_layers[-1] if slot_layers else -1,
            MAX_ROWS,
            dev,
        )
    runner._p15_ready = True


def _is_rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001
        return False


class _Recorder:
    def __init__(self, slot_layers):
        base = os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output")
        self.dir = os.path.join(base, "p15")
        os.makedirs(self.dir, exist_ok=True)
        self.slot_layers = np.asarray(slot_layers, dtype=np.int32)
        self.n_fwd = 0  # real forwards seen
        self.n_cap = 0
        self.n_stale = 0
        self.chunk = 0
        self.reset()
        logger.info("P15 probe: rank 0 recording to %s", self.dir)

    def reset(self):
        self.rows_ids = []
        self.rows_meta = []  # per row: cap_idx, fwd_idx, row, pos, req_hash, ncomp, nsched, rowinreq, tpad
        self.t0 = time.time()

    def want(self) -> bool:
        if self.n_cap >= MAX_STEPS:
            return False
        k = self.n_fwd - SKIP
        return k >= 0 and (k % PERIOD) < BURST

    def record(self, runner, ids, pos, tpad):
        so = runner._p15_sched
        ib = runner.input_batch
        reqs = []
        for i, rid in enumerate(ib.req_ids):
            if rid is None:
                continue
            n = int(so.num_scheduled_tokens.get(rid, 0))
            if n <= 0:
                continue
            reqs.append((rid, int(ib.num_computed_tokens_cpu[i]), n))
        row = 0
        ok = True
        out_rows = []
        for rid, ncomp, n in reqs:
            h = hash(rid) & 0x7FFFFFFF
            for j in range(n):
                if row >= min(tpad, MAX_ROWS):
                    break
                if int(pos[row]) != ncomp + j:
                    ok = False
                out_rows.append((row, int(pos[row]), h, ncomp, n, j))
                row += 1
        if not ok or not out_rows:
            self.n_stale += 1
            if self.n_stale <= 20 or self.n_stale % 500 == 0:
                logger.warning(
                    "P15 probe: positions mismatch/empty (stale=%d) pos=%s reqs=%s",
                    self.n_stale,
                    pos[: min(tpad, 16)].tolist(),
                    [(r[1], r[2]) for r in reqs[:8]],
                )
            if not out_rows:
                return
        for r, p, h, ncomp, n, j in out_rows:
            self.rows_ids.append(ids[r].astype(np.uint8))
            self.rows_meta.append(
                (self.n_cap, self.n_fwd, r, p, h, ncomp, n, j, tpad, int(ok))
            )
        self.n_cap += 1
        if self.n_cap % CHUNK == 0 or self.n_cap >= MAX_STEPS:
            self.flush()

    def flush(self):
        if not self.rows_ids:
            return
        ids = np.stack(self.rows_ids)
        meta = np.asarray(self.rows_meta, dtype=np.int64)
        path = os.path.join(self.dir, f"p15_chunk_{self.chunk:04d}.npz")
        np.savez_compressed(path, ids=ids, meta=meta, slot_layers=self.slot_layers)
        try:
            logger.info("P15 probe: %s; %s", path, _online_summary(ids, meta))
        except Exception as e:  # noqa: BLE001
            logger.warning("P15 probe: summary failed: %r", e)
        self.chunk += 1
        self.reset()


def _online_summary(ids, meta) -> str:
    """Quick recall numbers for the log (decode rows: nsched <= 4)."""
    dec = meta[:, 6] <= 4
    ids = ids[dec].astype(np.int32)
    if len(ids) == 0:
        return "no decode rows"
    act = ids[:, :, :TOPK]
    out = [f"decode_rows={len(ids)}"]
    for name, off in (("a", TOPK), ("a2", TOPK + K_PRED), ("b", TOPK + 2 * K_PRED)):
        for k in (8, 16):
            pred = ids[:, :, off : off + k]
            hit = (act[:, :, :, None] == pred[:, :, None, :]).any(-1).mean()
            out.append(f"recall_{name}@{k}={hit:.3f}")
    m = meta[dec]
    key = m[:, 0] * (1 << 32) + m[:, 4]
    uni = []
    for kv in np.unique(key):
        sel = act[key == kv]
        if len(sel) == 4:
            uni.append(
                np.mean([len(np.unique(sel[:, s])) for s in range(sel.shape[1])])
            )
    if uni:
        out.append(f"mtp4_union_mean={np.mean(uni):.2f}(n={len(uni)})")
    return " ".join(out)


def _wrap_runner() -> None:
    from vllm_torchtpu.runner import tpu_runner

    cls = tpu_runner.TPUModelRunner
    if getattr(cls, "_p15_wrapped", False):
        return
    orig_exec = cls.execute_model
    orig_fwd = cls.forward_model
    orig_load = cls.load_model

    def load_model(self, *a, **kw):
        out = orig_load(self, *a, **kw)
        _ensure_buffers(self)
        return out

    def execute_model(self, scheduler_output, *a, **kw):
        self._p15_sched = scheduler_output
        try:
            return orig_exec(self, scheduler_output, *a, **kw)
        finally:
            self._p15_sched = None

    def forward_model(self, input_ids, positions, *a, **kw):
        _ensure_buffers(self)
        out = orig_fwd(self, input_ids, positions, *a, **kw)
        if getattr(self, "_p15_sched", None) is None:
            return out
        m = getattr(self, "_p15_model", None)
        if m is None or not _is_rank0():
            return out
        rec = getattr(self, "_p15_rec", None)
        if rec is None:
            rec = self._p15_rec = _Recorder(m._p15_slot_layers)
        try:
            if rec.want():
                tpad = int(input_ids.shape[0]) if input_ids is not None else int(
                    positions.shape[0]
                )
                ids = m._p15_ids.cpu().numpy()
                pos = m._p15_pos.cpu().numpy()
                rec.record(self, ids, pos, tpad)
            rec.n_fwd += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("P15 probe: record failed: %r", e)
        return out

    cls.load_model = load_model
    cls.execute_model = execute_model
    cls.forward_model = forward_model
    cls._p15_wrapped = True


def maybe_install(*args: object, **kwargs: object) -> None:
    if not ENABLED:
        return None
    import vllm_torchtpu.envs as envs

    if envs.TPU_SEGMENT_MEGAKERNEL:
        raise RuntimeError("P15 probe needs TPU_SEGMENT_MEGAKERNEL=0")
    deepseek_v2.DeepseekV2Model.forward = patched_model_forward
    import sys

    # Wrap the runner only in a worker (its module is imported by then, at
    # model_load); elsewhere ask the registry to retry at a later stage.
    if "vllm_torchtpu.runner.tpu_runner" not in sys.modules:
        logger.info("P15 probe: forward patched; runner hooks deferred")
        return False
    _wrap_runner()
    logger.info("P15 probe: installed (forward + runner hooks)")
    return None
