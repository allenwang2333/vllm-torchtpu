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
"""Per-(layer, expert) router load histograms, for EP load-imbalance analysis.

Off unless ``TPU_MOE_EXPERT_LOAD_DIR`` names a directory. When off, no buffer
is registered and :func:`record` returns on a ``getattr`` -- the traced graph
is byte-identical to a build that does not contain this file.

Why the counts accumulate on device. Routing runs inside the ``torch.compile``d
forward, so Python in the router body executes at *trace* time, not per step: a
``.cpu()`` there would be one device sync per MoE layer per step (92 per step
on Kimi-K3), which is not instrumentation, it is a different workload. Instead
each layer accumulates into a persistent ``int32[slots, num_experts]`` buffer
that the host reads back a handful of times, from outside the compiled region.

Why it is never on during a profiled capture. Accumulating adds HLO, and a
capture whose op count differs from the one it is compared against is not
comparable to it. The pod job runs the capture and the dump in separate server
lifetimes for exactly this reason.

What lands in the npz, per rank:

    hist_all[L, S, E]   top-k selections, all rows including padding
    hist_real[L, S, E]  ... restricted to rows with a non-zero routing weight
    rows[L, S, 2]       (all, real) (token, expert) row counts
    calls[L, S]         forward calls that landed in the slot
    layer_names[L]      module path, so L is interpretable
    experts_start[R]    first global expert id owned by each EP rank

``hist_all - hist_real`` is where the padded rows went, which at concurrency 1
(1 real token padded to 16) is most of the mass.
"""

import os

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.vllm import token_padding
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Resolved once, at import: the compiled forward never reads the environment.
_DIR = envs.TPU_MOE_EXPERT_LOAD_DIR
_SLOTS = envs.TPU_MOE_EXPERT_LOAD_SLOTS
_BUCKET = envs.TPU_MOE_EXPERT_LOAD_BUCKET
# Only instrument the small token buckets. The bucket size is static per
# compiled graph, so this is a trace-time branch: the prefill graph gets no
# instrumentation at all, and a prefill step cannot swamp slot 0.
_MAX_TOKENS = envs.TPU_MOE_EXPERT_LOAD_MAX_TOKENS
_EVERY = envs.TPU_MOE_EXPERT_LOAD_EVERY
_RANKS = frozenset(
    int(r) for r in envs.TPU_MOE_EXPERT_LOAD_RANKS.split(",") if r.strip())

_BUFFERS = ("_eld_hist_all", "_eld_hist_real", "_eld_rows", "_eld_calls",
            "_eld_step")


def enabled() -> bool:
    return bool(_DIR)


def register_buffers(layer, *, device: torch.device) -> None:
    """Give one MoE layer its histogram buffers. No-op unless enabled.

    Called from ``moe_routing.register_experts_start_buffer``, which every TPU
    MoE quant method already invokes once per layer at weight load, so this
    inherits its single-call guarantee and its device.
    """
    if not enabled():
        return
    num_experts = int(layer.global_num_experts)

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int32, device=device)

    layer.register_buffer("_eld_hist_all",
                          zeros(_SLOTS, num_experts),
                          persistent=False)
    layer.register_buffer("_eld_hist_real",
                          zeros(_SLOTS, num_experts),
                          persistent=False)
    layer.register_buffer("_eld_rows", zeros(_SLOTS, 2), persistent=False)
    layer.register_buffer("_eld_calls", zeros(_SLOTS), persistent=False)
    layer.register_buffer("_eld_step", zeros(1), persistent=False)


def record(layer, topk_ids: torch.Tensor,
           topk_weights: torch.Tensor) -> None:
    """Accumulate one forward's expert selection. Traced; no host sync.

    ``topk_ids`` are GLOBAL expert ids and identical on every TP rank -- the
    hidden states are replicated after the attention all-reduce and the gate is
    replicated, and the rank split happens later, inside the kernel
    (``fused_moe_func``: ``local_ids = topk_ids - experts_start``). So one
    rank's histogram is every rank's, and per-rank load is a partial sum of it.
    """
    hist_all = getattr(layer, "_eld_hist_all", None)
    if hist_all is None or topk_ids.shape[0] > _MAX_TOKENS:
        return

    slots, num_experts = hist_all.shape
    onehot = torch.nn.functional.one_hot(topk_ids.reshape(-1).long(),
                                         num_experts).to(torch.int32)
    counts_all = onehot.sum(0)
    is_real = _real_row_mask(topk_ids, topk_weights).to(torch.int32)
    counts_real = (onehot * is_real.unsqueeze(1)).sum(0)

    step = layer._eld_step
    slot = torch.clamp(step // _BUCKET, max=slots - 1)
    # A [slots] selector instead of an index_add: the slot index is device
    # data, and a dense mask keeps the update a plain elementwise add.
    sel = (torch.arange(slots, device=step.device, dtype=step.dtype)
           == slot).to(torch.int32)
    col = sel.unsqueeze(1)

    hist_all.copy_(hist_all + col * counts_all.unsqueeze(0))
    layer._eld_hist_real.copy_(layer._eld_hist_real +
                               col * counts_real.unsqueeze(0))
    layer._eld_rows.copy_(
        layer._eld_rows +
        col * torch.stack([counts_all.sum(), counts_real.sum()]).unsqueeze(0))
    layer._eld_calls.copy_(layer._eld_calls + sel)
    step.copy_(step + 1)


def _real_row_mask(topk_ids: torch.Tensor,
                   topk_weights: torch.Tensor) -> torch.Tensor:
    """Flat bool mask over (token, expert) rows: True where the token is real.

    Two independent tests, because neither alone covers every quant method.
    ``zero_routing_weights_for_padding`` gives a padded row weight 0, but it
    runs *after* routing and only on the mxfp4 path -- so on that path, which
    is the one Kimi-K3 takes, the weights this hook sees are all non-zero and
    the padding mask is the only signal. On paths that do not track padding the
    mask is absent and the zero weight is.
    """
    num_tokens, topk = topk_ids.shape
    real = topk_weights.reshape(-1) != 0
    is_padding = token_padding.get_local_padding_mask(num_tokens)
    # A DP-gathered routing tensor is longer than the local mask; fall back to
    # the weight test rather than broadcast a mask that means something else.
    if is_padding is not None and is_padding.shape[0] == num_tokens:
        real = real & ~is_padding.unsqueeze(1).expand(num_tokens,
                                                      topk).reshape(-1)
    return real


def _instrumented_layers(model):
    return [(name, m) for name, m in model.named_modules()
            if getattr(m, "_eld_hist_all", None) is not None]


def reset(model) -> None:
    """Zero every histogram. Host-side."""
    for _, layer in _instrumented_layers(model):
        for name in _BUFFERS:
            getattr(layer, name).zero_()


def _experts_start_table(layer) -> list[int]:
    """First global expert id of each EP rank, under linear placement.

    Same formula as ``moe_routing.get_experts_start``, evaluated for every
    rank, so the dump is self-describing: a reader can turn per-expert counts
    into per-rank counts without knowing the topology.
    """
    cfg = layer.moe_config
    if not cfg.moe_parallel_config.use_ep:
        return []
    total, size = int(layer.global_num_experts), int(cfg.ep_size)
    base, rem = divmod(total, size)
    return [r * base + min(r, rem) for r in range(size)] + [total]


def dump(model, rank: int) -> str | None:
    """Write this rank's histograms. Host-side; returns the path or None."""
    if not enabled() or rank not in _RANKS:
        return None
    layers = _instrumented_layers(model)
    if not layers:
        logger.warning("expert-load dump is on but no MoE layer carries a "
                       "histogram buffer -- nothing written")
        return None
    import numpy as np

    def stack(attr):
        return np.stack(
            [getattr(m, attr).detach().cpu().numpy() for _, m in layers])

    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, f"expert_load_rank{rank}.npz")
    tmp = path + ".tmp"
    # A file object, not a name: np.savez_compressed appends '.npz' to a name
    # that lacks it, which would defeat the atomic rename below.
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            hist_all=stack("_eld_hist_all"),
            hist_real=stack("_eld_hist_real"),
            rows=stack("_eld_rows"),
            calls=stack("_eld_calls"),
            steps=stack("_eld_step"),
            layer_names=np.array([n for n, _ in layers]),
            experts_start=np.array(_experts_start_table(layers[0][1]),
                                   dtype=np.int64),
            bucket=np.int64(_BUCKET),
            rank=np.int64(rank),
        )
    # The dump is overwritten as the run proceeds and the job may be killed
    # mid-write, so the file that exists must always be a complete one.
    os.replace(tmp, path)
    return path


# Dump early, then often. The early ones prove the format end to end while the
# run is young; a run cut short must still leave something readable.
_EAGER_DUMPS = (8, 64, 512)


class DumpDriver:
    """Counts host-side steps and dumps on a schedule. One per worker."""

    def __init__(self, rank: int):
        self.rank = rank
        self.calls = 0

    def step(self, runner) -> None:
        if not enabled():
            return
        model = runner.get_model()
        self.calls += 1
        if self.calls == 1:
            # Compile warm-up and dummy runs went through the same forward and
            # would otherwise own the first slots.
            reset(model)
            return
        if self.calls in _EAGER_DUMPS or self.calls % _EVERY == 0:
            path = dump(model, self.rank)
            if path:
                logger.info("expert-load dump at step %d -> %s", self.calls,
                            path)
