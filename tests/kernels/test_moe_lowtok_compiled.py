"""Compiled MoE parity with primary and replicated expert placements."""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_a_tpu():
    import jax
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")


def _fp8_weight(rng, shape):
    """Build one expert at a time to bound host temporaries at served shapes."""
    weight = torch.empty(shape, dtype=torch.float8_e4m3fn)
    for expert in range(shape[0]):
        values = rng.standard_normal(shape[1:], dtype=np.float32)
        values *= 40
        np.clip(values, -448, 448, out=values)
        weight[expert].copy_(torch.from_numpy(values))
    return weight


@pytest.mark.parametrize(
    "rank_source,replicas,tokens",
    [
        # The rank reaches the kernel as a scalar-prefetch operand and must be a
        # stored tensor: the compile flow cannot specialise one the traced forward
        # computes, and the attempt aborts the process rather than raising. The
        # served layer passes the expert-start buffer registered at load, which is
        # the "experts_start" case below; "buffer" covers the same shape of input.
        ("buffer", False, 4),
        ("experts_start", False, 4),
        ("experts_start", True, 4),
        ("experts_start", True, 32),
    ])
def test_scores_op_compiled_matches_eager(device, rank_source, replicas,
                                          tokens):
    if device.type != "tpu":
        pytest.skip("The compiled scores op requires a TPU")

    from vllm_torchtpu.layers.vllm import moe_lowtok_scores_op as scores_op

    rank, topk, ranks, local = 3, 10, 8, 64
    hidden, intermediate = 4096, 1024
    rng = np.random.default_rng(5)
    owned = local if rank_source == "experts_start" else 1
    scores_op.prebuild(topk, ranks, owned)

    class ScoresModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.register_buffer("experts_start",
                                 torch.tensor(rank * local, dtype=torch.int32))
            self.register_buffer("rank_id",
                                 torch.tensor(rank, dtype=torch.int32))
            gid = torch.arange(rank * local, (rank + 1) * local,
                               dtype=torch.int32)
            primary = torch.arange(local * ranks, dtype=torch.int32) // local
            copies = torch.full_like(primary, -1)
            if replicas:
                # Sixteen experts from each rank have a copy on its successor.
                # This gives all ranks the served 64-primary + 16-copy layout.
                for owner in range(ranks):
                    copies[owner * local:owner * local +
                           16] = (owner + 1) % ranks
                donor = (rank - 1) % ranks
                gid = torch.cat([
                    gid,
                    torch.arange(donor * local,
                                 donor * local + 16,
                                 dtype=torch.int32)
                ])
            self.register_buffer("gid", gid)
            self.register_buffer("primary", primary)
            self.register_buffer("copies", copies)
            from vllm_torchtpu.kernels.moe_lowtok import (hot_onehot,
                                                          hot_tables,
                                                          slot_table)
            order = torch.nonzero(copies >= 0).flatten().tolist()
            hot, hprim, hcopy = hot_tables(primary.numpy(),
                                           copies.numpy(),
                                           order=order)
            for name, value in (("hot", hot), ("hprim", hprim),
                                ("hcopy",
                                 hcopy), ("slot",
                                          slot_table(self.gid.numpy(),
                                                     self.gid.numel()))):
                self.register_buffer(
                    name, torch.tensor(np.asarray(value), dtype=torch.int32))
            self.register_buffer(
                "hot1h",
                torch.tensor(np.asarray(hot_onehot(hot)).astype(np.float32),
                             dtype=torch.bfloat16))
            for name, shape, width in (
                ("w13", (self.gid.numel(), hidden, 2 * intermediate), hidden),
                ("w2", (self.gid.numel(), intermediate, hidden), intermediate),
            ):
                self.register_buffer(name, _fp8_weight(rng, shape))
                scales = rng.random((shape[0], 1, 1, shape[-1]),
                                    dtype=np.float32) / (40 * np.sqrt(width))
                self.register_buffer(name + "_scale",
                                     torch.tensor(scales, dtype=torch.float32))

        def forward(self, x, logits):
            rank_value = self.rank_id
            if rank_source == "computed":
                rank_value = self.experts_start // local
            elif rank_source == "experts_start":
                rank_value = self.experts_start
            return scores_op.moe_lowtok_scores_op(x,
                                                  torch.softmax(logits.float(),
                                                                dim=-1),
                                                  rank_value,
                                                  self.gid,
                                                  self.primary,
                                                  self.copies,
                                                  self.hot,
                                                  self.hprim,
                                                  self.hcopy,
                                                  self.hot1h,
                                                  self.slot,
                                                  self.w13,
                                                  self.w13_scale,
                                                  self.w2,
                                                  self.w2_scale,
                                                  topk=topk,
                                                  ranks=ranks,
                                                  owned=owned)

    model = ScoresModule().to(device)
    x = torch.tensor(rng.standard_normal((tokens, hidden)),
                     dtype=torch.bfloat16,
                     device=device)
    if replicas:
        # No selected expert is a primary on this rank. Its nonzero output
        # must come from active replica slots assigned by the in-kernel loop.
        logits = np.full((tokens, local * ranks), -30.0, np.float32)
        donor = (rank - 1) % ranks
        selected = [
            donor * local + i for i in (0, 1, 2, 3, 4, 5, 16, 17, 18, 19)
        ]
        logits[:, selected] = np.linspace(6.0, 5.0, topk)
    else:
        logits = rng.standard_normal((tokens, local * ranks)) * 2
        logits[:, rank * local:(rank + 1) * local] += 1.5
    logits = torch.tensor(logits, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        eager = model(x, logits).cpu()
        compiled = torch.compile(model,
                                 backend="tpu",
                                 fullgraph=True,
                                 dynamic=False)
        actual = compiled(x, logits).cpu()
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    if replicas:
        assert actual.abs().max(
        ) > 0, "The active copied experts contributed nothing"


@pytest.mark.parametrize("tokens", [64, 512])
def test_grouped_fallback_with_copy_slots(device, monkeypatch, tokens):
    """The compiled fallback ignores replica slots beyond its primary interval."""
    if device.type != "tpu":
        pytest.skip("The compiled grouped fallback requires a TPU")

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                     prebuild_fused_moe_kernel)

    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 0)
    monkeypatch.setattr(envs, "USE_MOE_COUNTING_SORT", 1)
    owned, copies, hidden, inter, topk, start = 8, 2, 512, 256, 5, 24
    slots = owned + copies
    rng = np.random.default_rng(19)
    w13 = _fp8_weight(rng, (slots, hidden, 2 * inter)).to(device)
    w2 = _fp8_weight(rng, (slots, inter, hidden)).to(device)
    s13 = torch.full((slots, 1, 1, 2 * inter),
                     1 / (40 * np.sqrt(hidden)),
                     dtype=torch.float32,
                     device=device)
    s2 = torch.full((slots, 1, 1, hidden),
                    1 / (40 * np.sqrt(inter)),
                    dtype=torch.float32,
                    device=device)
    experts_start = torch.tensor(start, dtype=torch.int32, device=device)
    x = torch.tensor(rng.standard_normal((tokens, hidden)),
                     dtype=torch.bfloat16,
                     device=device)
    # The two ids immediately beyond the primary interval would incorrectly
    # read copy slots if the fallback used the physical weight count as its bound.
    ids = torch.tensor(
        [start, start + owned - 1, start + owned, start + owned + 1, -1],
        dtype=torch.int32).repeat(tokens, 1)
    ids[-2] = torch.arange(topk, dtype=torch.int32)  # no local experts
    ids = ids.to(device)
    weights = torch.full((tokens, topk), 1 / topk, dtype=torch.bfloat16)
    weights[-1] = 0  # padded row
    weights = weights.to(device)
    opts = dict(topk=topk,
                activation="silu",
                use_ep=True,
                skip_padded_tokens=True)
    prebuild_fused_moe_kernel(**opts)
    prebuild_fused_moe_kernel(**opts, num_primary_experts=owned)

    def run(h, routing_ids, routing_weights):
        return fused_moe_gmm(h,
                             w13,
                             w2,
                             s13,
                             s2,
                             None,
                             None,
                             routing_weights,
                             routing_ids,
                             experts_start,
                             topk,
                             "silu",
                             skip_padded_tokens=True,
                             num_primary_experts=owned)

    with torch.no_grad():
        expected = fused_moe_gmm(x,
                                 w13[:owned],
                                 w2[:owned],
                                 s13[:owned],
                                 s2[:owned],
                                 None,
                                 None,
                                 weights,
                                 ids,
                                 experts_start,
                                 topk,
                                 "silu",
                                 skip_padded_tokens=True).float().cpu()
        actual = torch.compile(run,
                               backend="tpu",
                               fullgraph=True,
                               dynamic=False)(x, ids, weights).float().cpu()
    assert torch.isfinite(actual).all()
    assert expected.abs().max() > 0
    scale = expected.abs().max() + 1e-6
    torch.testing.assert_close(actual / scale,
                               expected / scale,
                               rtol=0,
                               atol=0.06)
    assert torch.linalg.vector_norm(actual - expected) / (
        torch.linalg.vector_norm(expected) + 1e-9) < 0.01
    assert torch.count_nonzero(actual[-2:]) == 0
