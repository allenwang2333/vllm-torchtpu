# SPDX-License-Identifier: Apache-2.0
"""Opt-in tap of CSA2 top-k records from a serving step (DSV41_TOPK_TAP=<dir>).

Instrumentation only: every tapped layer copies its [tokens, 512] main records
into a persistent device buffer inside the compiled graph, and the recorder
reads the buffers back (a device sync) for a few steps at the scenario shape.
The dumps feed the union-size analysis of a request's queries; timing of a
tapped run is not a measurement.
"""

import functools
import os
from pathlib import Path
from typing import Any

# One layer per index group plus a second layer of two groups, to check that a
# group's layers really receive identical records.
TAP_LAYERS = (2, 3, 8, 14, 20, 21, 24, 28, 32, 36)
MAX_TOKENS = 2048
MAX_STARTS = 256
TOPK = 512

_state: dict[str, Any] = {"layer": -1}
_buffers: dict[str, Any] = {}


def enabled() -> bool:
    return bool(os.environ.get("DSV41_TOPK_TAP"))


def _allocate() -> None:
    import torch

    device = torch.device("tpu")
    n = len(TAP_LAYERS)
    _buffers["records"] = torch.full(
        (n, MAX_TOKENS, TOPK), -2, dtype=torch.int32, device=device
    )
    _buffers["positions"] = torch.full(
        (n, MAX_TOKENS), -2, dtype=torch.int32, device=device
    )
    _buffers["starts"] = torch.full(
        (n, MAX_STARTS), -2, dtype=torch.int32, device=device
    )


def install() -> None:
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as attn_module
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter

    init_ops = csa2_adapter.initialize_csa2_ops

    @functools.wraps(init_ops)
    def initialize_csa2_ops() -> None:
        init_ops()
        if not _buffers:
            _allocate()

    csa2_adapter.initialize_csa2_ops = initialize_csa2_ops

    cls = attn_module.DeepseekV41Attention
    forward = cls.forward

    @functools.wraps(forward)
    def tapped_forward(self, *args, **kwargs):
        # A trace-time side effect: Dynamo folds the layer id into the graph.
        _state["layer"] = self.layer_id
        return forward(self, *args, **kwargs)

    cls.forward = tapped_forward

    attention = csa2_adapter.attention

    @functools.wraps(attention)
    def tapped_attention(
        q, swa, main, records, pages, main_starts, starts, positions, *rest
    ):
        layer = _state["layer"]
        if layer in TAP_LAYERS:
            slot = TAP_LAYERS.index(layer)
            tokens = records.shape[0]
            width = min(records.shape[1], TOPK)
            _buffers["records"][slot, :tokens, :width].copy_(
                records[:, :width].to(_buffers["records"].dtype)
            )
            _buffers["positions"][slot, :tokens].copy_(
                positions.to(_buffers["positions"].dtype)
            )
            _buffers["starts"][slot, : starts.shape[0]].copy_(
                starts.to(_buffers["starts"].dtype)
            )
        return attention(
            q, swa, main, records, pages, main_starts, starts, positions, *rest
        )

    csa2_adapter.attention = tapped_attention


class Dumper:
    """Write the tapped buffers for steps at the scenario's steady shape."""

    def __init__(self, rank: int, scenario: Any) -> None:
        self.rank = rank
        self.scenario = scenario
        self.root = Path(os.environ["DSV41_TOPK_TAP"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_dumps = int(os.environ.get("DSV41_TOPK_TAP_DUMPS", "4"))
        self.stride = int(os.environ.get("DSV41_TOPK_TAP_STRIDE", "25"))
        self.seen = 0
        self.dumped = 0

    def after_step(self, runner: Any, tokens: int) -> None:
        import numpy as np

        if self.dumped >= self.max_dumps or not _buffers:
            return
        n = runner.input_batch.num_reqs
        contexts = [int(v) for v in runner.input_batch.num_computed_tokens_cpu[:n]]
        decode = self.scenario.name == "decode"
        steady = (
            tokens
            == self.scenario.local_concurrency * (self.scenario.dspark_tokens + 1)
            and n == self.scenario.local_concurrency
            if decode
            else tokens == self.scenario.chunk_tokens
            and min(contexts, default=0) >= self.scenario.prefix_tokens
        )
        if not steady:
            return
        self.seen += 1
        if (self.seen - 1) % self.stride:
            return
        records = _buffers["records"][:, :tokens].cpu().numpy()
        positions = _buffers["positions"][:, :tokens].cpu().numpy()
        starts = _buffers["starts"].cpu().numpy()
        np.savez_compressed(
            self.root / f"rank{self.rank}-{self.scenario.name}-dump{self.dumped}.npz",
            layers=np.array(TAP_LAYERS, dtype=np.int32),
            records=records,
            positions=positions,
            starts=starts,
            contexts=np.array(contexts, dtype=np.int64),
            req_ids=np.array(list(runner.input_batch.req_ids[:n])),
            tokens=np.int64(tokens),
        )
        self.dumped += 1
