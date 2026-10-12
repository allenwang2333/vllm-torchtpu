# SPDX-License-Identifier: Apache-2.0
"""100% statement & branch CTracer coverage tests for spec_decode.mtp_fused_loop."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class _StubMeta(type):
    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()


class _StubModule(ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "support_torch_compile":
            return lambda *a, **kw: (lambda cls: cls)
        if name.isupper():
            return None
        if name[:1].isupper():
            return _StubMeta(name, (), {"__init__": lambda self, *a, **k: None})
        return MagicMock()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in {"vllm", "torch_tpu"}:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        return None


try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    sys.meta_path.insert(0, _StubFinder())

import vllm_torchtpu.spec_decode.mtp_fused_loop as mod  # noqa: E402


class _FakeTPGroup:
    """Multi-rank TP collective emulator supporting bit-exact rank simulation."""

    def __init__(
        self, world_size: int, rank: int, all_rank_tensors: list | None = None
    ):
        self.world_size = world_size
        self.rank_in_group = rank
        self._peers = all_rank_tensors

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert self._peers is not None
        parts = [fn(tensor) if callable(fn) else fn for fn in self._peers]
        parts[self.rank_in_group] = tensor
        return torch.cat(parts, dim=dim)


def _reference_compute_logits(
    full_weight: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.linear(hidden, full_weight)


def test_resolve_fast_lm_head_rejection_and_valid_paths(monkeypatch):
    # 1. Missing lm_head or non-2D weight -> None
    assert mod._resolve_fast_lm_head(SimpleNamespace()) is None
    assert (
        mod._resolve_fast_lm_head(
            SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(8)))
        )
        is None
    )

    # 2. Non-None bias -> None
    assert (
        mod._resolve_fast_lm_head(
            SimpleNamespace(
                lm_head=SimpleNamespace(
                    weight=torch.ones((8, 4)),
                    bias=torch.zeros(8),
                )
            )
        )
        is None
    )

    # 3. Unsupported quant_method class -> None
    class CustomFp8QuantMethod:
        pass

    assert (
        mod._resolve_fast_lm_head(
            SimpleNamespace(
                lm_head=SimpleNamespace(
                    weight=torch.ones((8, 4)),
                    bias=None,
                    quant_method=CustomFp8QuantMethod(),
                )
            )
        )
        is None
    )

    # 4. Softcap or non-unit logit scale -> None
    assert (
        mod._resolve_fast_lm_head(
            SimpleNamespace(
                lm_head=SimpleNamespace(weight=torch.ones((8, 4)), bias=None),
                logits_processor=SimpleNamespace(softcap=30.0, scale=1.0),
            )
        )
        is None
    )
    assert (
        mod._resolve_fast_lm_head(
            SimpleNamespace(
                lm_head=SimpleNamespace(weight=torch.ones((8, 4)), bias=None),
                logits_processor=SimpleNamespace(softcap=None, scale=0.5),
            )
        )
        is None
    )

    # 5. Valid UnquantizedEmbeddingMethod + TP=1 + org_vocab_size slicing vs full
    class UnquantizedEmbeddingMethod:
        pass

    monkeypatch.setattr(
        mod, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    w_padded = torch.randn((16, 4))
    res_sliced = mod._resolve_fast_lm_head(
        SimpleNamespace(
            lm_head=SimpleNamespace(
                weight=w_padded,
                bias=None,
                quant_method=UnquantizedEmbeddingMethod(),
                org_vocab_size=12,
            ),
            logits_processor=SimpleNamespace(softcap=None, scale=1.0),
        )
    )
    assert res_sliced is not None
    eff_w, tp_grp, offset = res_sliced
    assert eff_w.shape == (12, 4)
    assert tp_grp is None
    assert offset == 0

    res_unsliced = mod._resolve_fast_lm_head(
        SimpleNamespace(
            lm_head=SimpleNamespace(
                weight=w_padded,
                bias=None,
                quant_method=None,
            )
        )
    )
    assert res_unsliced is not None
    eff_w2, tp_grp2, offset2 = res_unsliced
    assert eff_w2 is w_padded
    assert tp_grp2 is None
    assert offset2 == 0


def test_mtp_fused_loop_tp_sharded_bit_exact_and_ties(monkeypatch):
    torch.manual_seed(42)
    tp_size = 4
    shard_vocab = 32
    full_vocab = tp_size * shard_vocab
    hidden_size = 16
    batch = 5
    full_weight = torch.randn((full_vocab, hidden_size), dtype=torch.bfloat16)

    # Inject explicit cross-rank exact tie on row 0 between rank 1 and rank 3
    full_weight[:, :] *= 0.1
    full_weight[1 * shard_vocab + 5] = full_weight[3 * shard_vocab + 9] = torch.ones(
        hidden_size, dtype=torch.bfloat16
    )

    class _DraftTP(nn.Module):
        def __init__(self, rank: int):
            super().__init__()
            self.lm_head = SimpleNamespace(
                weight=full_weight[rank * shard_vocab : (rank + 1) * shard_vocab],
                bias=None,
                quant_method=None,
                org_vocab_size=full_vocab - 3,
            )

        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors=None,
            inputs_embeds=None,
            spec_step_idx: int = 0,
        ):
            step_delta = (input_ids.unsqueeze(-1).to(hidden_states.dtype) % 5) * 0.05
            pos_term = (positions.unsqueeze(-1) * 0.01).to(hidden_states.dtype)
            h_next = hidden_states + step_delta + pos_term
            if spec_step_idx == 1:
                return (h_next,)
            if spec_step_idx == 2:
                return h_next, h_next
            return h_next

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return _reference_compute_logits(
                full_weight[: full_vocab - 3], hidden_states
            )

    input_ids = torch.arange(batch, dtype=torch.int32)
    positions = torch.arange(10, 10 + batch, dtype=torch.int32)
    hidden = torch.ones((batch, hidden_size), dtype=torch.bfloat16)

    # Reference fallback run (fast path disabled via bias flag -> calls compute_logits)
    ref_draft = _DraftTP(0)
    ref_draft.lm_head.bias = torch.zeros(1)
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    ref_loop = mod._MtpFusedLoop(
        vllm_config=SimpleNamespace(),
        prefix="ref",
        mtp=ref_draft,
        num_steps=3,
    )
    seq_lens_ref = torch.full((batch,), 20, dtype=torch.int32)
    ref_tokens = ref_loop(
        input_ids.clone(), positions.clone(), hidden.clone(), seq_lens_ref
    )
    assert seq_lens_ref.tolist() == [22] * batch

    # Verify TP=4 sharded fast path matches reference bit-for-bit across all steps
    def _run_all_ranks_step(rank_idx: int):
        cur_ids = [input_ids.clone()]
        cur_pos = [positions.clone()]
        cur_h = [hidden.clone()]

        drafts = [_DraftTP(r) for r in range(tp_size)]
        per_rank_step_packed: list[list[torch.Tensor]] = []
        ids_s = input_ids.clone()
        pos_s = positions.clone()
        h_s = hidden.clone()
        for step_i in range(3):
            if step_i > 0:
                pos_s = pos_s + 1
            out_s = _DraftTP.forward(
                drafts[0], ids_s, pos_s, h_s, None, None, spec_step_idx=step_i + 1
            )
            if isinstance(out_s, tuple) and len(out_s) == 1:
                out_s = out_s[0]
            last_h = out_s[0] if isinstance(out_s, tuple) else out_s
            h_s = out_s[1] if isinstance(out_s, tuple) else out_s
            rank_packs = []
            for r in range(tp_size):
                w_r = drafts[r].lm_head.weight
                valid = min(shard_vocab, max(0, (full_vocab - 3) - r * shard_vocab))
                w_eff = w_r[:valid] if valid < shard_vocab else w_r
                sl = torch.nn.functional.linear(last_h, w_eff)
                mx, ix = sl.max(dim=-1)
                gi = ix.to(torch.int32) + r * shard_vocab
                pk = torch.stack((mx.to(torch.float32), gi.to(torch.float32)), dim=-1)
                rank_packs.append(pk.unsqueeze(0))
            per_rank_step_packed.append(rank_packs)
            ids_s = ref_tokens[step_i]

        call_idx = [0]

        def _fake_peer_fn(r_peer: int):
            def _get(_self_t):
                c = call_idx[0]
                return per_rank_step_packed[c][r_peer]

            return _get

        class _StepTrackingTPGroup:
            world_size = tp_size
            rank_in_group = rank_idx

            def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
                c = call_idx[0]
                call_idx[0] += 1
                parts = list(per_rank_step_packed[c])
                parts[rank_idx] = tensor
                return torch.cat(parts, dim=dim)

        monkeypatch.setattr(mod, "get_tp_group", lambda: _StepTrackingTPGroup())
        loop_tp = mod.build_mtp_fused_loop(
            vllm_config=SimpleNamespace(),
            prefix=f"tp_rank_{rank_idx}",
            mtp=_DraftTP(rank_idx),
            num_steps=3,
            dynamic_arg_dims={"input_ids": 0, "positions": 0},
        )
        # Exercise warm _FUSED_LOOP_CLS_CACHE hit on second call
        loop_tp_warm = mod.build_mtp_fused_loop(
            vllm_config=SimpleNamespace(),
            prefix=f"tp_rank_{rank_idx}_warm",
            mtp=_DraftTP(rank_idx),
            num_steps=3,
            dynamic_arg_dims={"positions": 0, "input_ids": 0},
        )
        assert type(loop_tp) is type(loop_tp_warm)
        seq_lens_tp = torch.full((batch,), 20, dtype=torch.int32)
        out_tp = loop_tp(cur_ids[0], cur_pos[0], cur_h[0], seq_lens_tp)
        for s in range(3):
            assert torch.equal(out_tp[s], ref_tokens[s])

    _run_all_ranks_step(0)
    _run_all_ranks_step(3)


def test_mtp_fused_loop_tp1_fast_path_and_int32_passthrough(monkeypatch):
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    weight = torch.randn((24, 8), dtype=torch.float32)

    class _DraftTP1(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = SimpleNamespace(
                weight=weight, bias=None, quant_method=None, org_vocab_size=24
            )

        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors=None,
            inputs_embeds=None,
            spec_step_idx: int = 0,
        ):
            return hidden_states + 0.1

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(hidden_states, weight)

    draft = _DraftTP1()
    loop_fast = mod._MtpFusedLoop(
        vllm_config=SimpleNamespace(), prefix="tp1", mtp=draft, num_steps=2
    )
    ids = torch.zeros(3, dtype=torch.int32)
    pos = torch.arange(3, dtype=torch.int32)
    h = torch.randn((3, 8), dtype=torch.float32)
    sl = torch.ones(3, dtype=torch.int32)
    res_fast = loop_fast(ids.clone(), pos.clone(), h.clone(), sl)

    # Also cover pre-int32 argmax return type on fallback compute_logits path
    class _Int32LogitsTensor:
        def argmax(self, dim: int = -1):
            return torch.tensor([1, 2, 3], dtype=torch.int32)

    class _DraftInt32Argmax:
        def forward(self, *args, **kwargs):
            return args[2]

        def compute_logits(self, hidden_states):
            return _Int32LogitsTensor()

    loop_i32 = mod._MtpFusedLoop(
        vllm_config=SimpleNamespace(),
        prefix="i32",
        mtp=_DraftInt32Argmax(),  # type: ignore[arg-type]
        num_steps=1,
    )
    (tok_i32,) = loop_i32(ids, pos, h, sl)
    assert tok_i32.dtype == torch.int32
    assert tok_i32.tolist() == [1, 2, 3]
    assert len(res_fast) == 2
