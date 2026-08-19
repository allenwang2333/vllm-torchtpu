# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the fp8 MoE dispatch all-gather (TPU_MOE_DISPATCH_FP8)."""

import pytest
import torch

from vllm_torchtpu.distributed import fp8_wire, tpu_communicator

DP = 4
HIDDEN = 2048
BLOCK = fp8_wire._block_for(HIDDEN)  # the whole row unless WIRE_BLOCK is set
NUM_BLOCKS = HIDDEN // BLOCK


class _FakeDpGroup:
    """Concat-style all-gather that records what crossed the wire."""

    world_size = DP

    def __init__(self):
        self.gathered: list[tuple[torch.dtype, tuple[int, ...]]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0):
        self.gathered.append((tensor.dtype, tuple(tensor.shape)))
        return torch.cat([tensor] * self.world_size, dim=dim)

    def wire_bytes(self) -> int:
        return sum(
            torch.empty(0, dtype=dt).element_size() *
            int(torch.tensor(shape).prod()) for dt, shape in self.gathered)


def _communicator(dispatch_fp8: bool, monkeypatch) -> tuple:
    comm = object.__new__(tpu_communicator.TpuDeviceCommunicator)
    comm.dispatch_fp8 = dispatch_fp8
    comm.packed_wire = fp8_wire.packed_wire()
    group = _FakeDpGroup()
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: group)
    return comm, group


def test_wire_block_is_the_row_or_exactly_the_kernel_block():
    # A partial block width would be the worst of both: gmm_v2 re-derives its
    # lhs scale over LHS_QUANT_BLOCK_SIZE-wide k slices, so anything other than
    # "one scale per row" or "exactly that block" costs a second rounding for
    # no bytes saved. Per-row is the shipped choice (it fuses; the block form
    # materialises 480 MB on the receive side), so the contract is a disjunction.
    from vllm_torchtpu.kernels.megablox import gmm_v2

    assert (fp8_wire.WIRE_BLOCK is None
            or fp8_wire.WIRE_BLOCK == gmm_v2.LHS_QUANT_BLOCK_SIZE)


def test_default_wire_is_two_async_collectives_not_one_packed_buffer(
        monkeypatch):
    """The property the -60.8%/-61.5% device runs lacked.

    A uint8 payload compiles to a synchronous all-gather; float8_e4m3fn and
    bf16 get the async, sparsecore-offloaded pair. Assert what must be PRESENT:
    two gathers, the rows in fp8 and the scales in fp32, and no uint8 anywhere.
    """
    comm, group = _communicator(True, monkeypatch)
    comm.dispatch_hidden_states(torch.randn(8, HIDDEN, dtype=torch.bfloat16))

    dtypes = [dt for dt, _ in group.gathered]
    assert dtypes == [torch.float8_e4m3fn, torch.float32]
    assert torch.uint8 not in dtypes
    assert group.gathered[0][1] == (8, HIDDEN)
    assert group.gathered[1][1] == (8, NUM_BLOCKS)


def test_packed_wire_is_reachable_and_is_the_single_uint8_buffer(monkeypatch):
    monkeypatch.setattr(fp8_wire.envs, "TPU_MOE_DISPATCH_FP8_WIRE", "uint8")
    comm, group = _communicator(True, monkeypatch)
    comm.packed_wire = fp8_wire.packed_wire()
    comm.dispatch_hidden_states(torch.randn(8, HIDDEN, dtype=torch.bfloat16))

    assert [dt for dt, _ in group.gathered] == [torch.uint8]


def test_quantize_rows_round_trip_is_within_one_fp8_step():
    torch.manual_seed(0)
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16)

    q, scale = fp8_wire.quantize_rows(x)
    back = fp8_wire.dequantize_rows(q, scale, x.dtype).float()

    assert q.dtype is torch.float8_e4m3fn
    assert scale.shape == (64, NUM_BLOCKS) and scale.dtype is torch.float32
    # e4m3 keeps 3 mantissa bits, so a value is at worst 2^-4 relative to its
    # block max away from the nearest representable point.
    blocks = x.float().unflatten(-1, (NUM_BLOCKS, BLOCK))
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    err = (back - x.float()).unflatten(-1, (NUM_BLOCKS, BLOCK))
    assert torch.all(err.abs() <= block_max * 2**-4)


def test_block_max_comes_back_saturated_so_the_kernel_requant_is_a_no_op():
    torch.manual_seed(0)
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16)

    q, _ = fp8_wire.quantize_rows(x)

    per_block = q.float().unflatten(-1, (NUM_BLOCKS, BLOCK))
    assert torch.all(per_block.abs().amax(dim=-1) == fp8_wire.WIRE_MAX)


def test_zero_rows_survive_the_wire_exactly():
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16)
    x[3:] = 0  # padded tokens
    x[0, :BLOCK] = 0  # a dead block inside a live row

    q, scale = fp8_wire.quantize_rows(x)
    back = fp8_wire.dequantize_rows(q, scale, x.dtype)

    assert torch.all(back[3:] == 0)
    assert torch.all(torch.isfinite(back))


def test_encode_decode_round_trips_through_one_uint8_buffer():
    torch.manual_seed(0)
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16)

    payload = fp8_wire.encode(x)
    back = fp8_wire.decode(payload, HIDDEN, x.dtype)

    assert payload.dtype is torch.uint8
    assert payload.shape == (8, HIDDEN + NUM_BLOCKS * 4)
    q, scale = fp8_wire.quantize_rows(x)
    assert torch.equal(back, fp8_wire.dequantize_rows(q, scale, x.dtype))


def test_dispatch_router_logits_halves_the_hidden_payload(monkeypatch):
    torch.manual_seed(0)
    hidden = torch.randn(32, HIDDEN, dtype=torch.bfloat16)
    logits = torch.randn(32, 64, dtype=torch.bfloat16)

    off, off_group = _communicator(False, monkeypatch)
    ref, ref_logits = off.dispatch_router_logits(hidden, logits)
    on, on_group = _communicator(True, monkeypatch)
    got, got_logits = on.dispatch_router_logits(hidden, logits)

    # Rows and scales are separate collectives so both keep the async,
    # sparsecore-offloaded form; the router logits stay bf16 (rounding them
    # moves top-k membership, not just magnitudes).
    assert [dt for dt, _ in on_group.gathered
            ] == [torch.float8_e4m3fn, torch.float32, torch.bfloat16]
    assert got.dtype is torch.bfloat16 and got.shape == ref.shape
    assert torch.equal(got_logits, ref_logits)
    # 1 B/element + 4 B per hidden block instead of 2 B/element; logits as-is.
    assert on_group.wire_bytes() == (off_group.wire_bytes() - 32 * HIDDEN +
                                     32 * NUM_BLOCKS * 4)
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.03


def test_dispatch_keeps_routing_tensors_intact(monkeypatch):
    torch.manual_seed(0)
    hidden = torch.randn(16, HIDDEN, dtype=torch.bfloat16)
    weights = torch.rand(16, 4, dtype=torch.bfloat16)
    ids = torch.randint(0, 64, (16, 4), dtype=torch.int32)

    comm, group = _communicator(True, monkeypatch)
    _, got_w, got_ids = comm.dispatch(hidden, weights, ids)

    assert got_ids.dtype is torch.int32
    assert torch.equal(got_ids, torch.cat([ids] * DP))
    assert torch.equal(got_w, torch.cat([weights] * DP))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dequantized_rows_keep_the_input_dtype(dtype, monkeypatch):
    comm, _ = _communicator(True, monkeypatch)
    out = comm.dispatch_hidden_states(torch.randn(8, HIDDEN, dtype=dtype))
    assert out.dtype is dtype


def test_dispatch_hidden_states_is_a_no_op_without_dp(monkeypatch):
    # Callers outside this class (a patched MoERunner._maybe_dispatch) route
    # the activation gather through here unconditionally.
    comm, group = _communicator(True, monkeypatch)
    monkeypatch.setattr(group.__class__, "world_size", 1)
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16)

    assert comm.dispatch_hidden_states(x) is x
    assert group.gathered == []
