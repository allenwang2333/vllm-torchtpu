# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the fp8 MoE dispatch all-gather (TPU_MOE_DISPATCH_FP8)."""

import pytest
import torch

from vllm_torchtpu.distributed import fp8_wire, tpu_communicator

DP = 4
HIDDEN = 2048
# kernels.megablox.gmm_v2.LHS_QUANT_BLOCK_SIZE, spelled out so this file stays
# importable without jax; the contract test below pins the two together.
LHS_QUANT_BLOCK_SIZE = 512
BLOCK = fp8_wire._block_for(HIDDEN)  # the whole row unless the knob is set
NUM_BLOCKS = HIDDEN // BLOCK


@pytest.fixture
def kernel_block_wire(monkeypatch):
    """TPU_MOE_DISPATCH_FP8_BLOCK=512: the wire on the kernel's own grid."""
    monkeypatch.setattr(fp8_wire.envs, "TPU_MOE_DISPATCH_FP8_BLOCK",
                        LHS_QUANT_BLOCK_SIZE)
    return LHS_QUANT_BLOCK_SIZE


def _kernel_requantize(x: torch.Tensor):
    """gmm_v2's lhs quantizer, gmm_v2.py:544-559, in the kernel's dtypes.

    The lhs tile arrives in the activation dtype, so the amax, the scale and
    the reciprocal are all computed there -- reproducing that is the whole
    point, because a model of it in fp32 would hide the rounding under test.
    """
    blocks = x.unflatten(-1, (-1, LHS_QUANT_BLOCK_SIZE))
    scale = blocks.abs().amax(dim=-1, keepdim=True) / fp8_wire.WIRE_MAX
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1 / scale)
    q = (blocks * inv).to(fp8_wire.WIRE_DTYPE)
    return q.flatten(-2), scale.squeeze(-1)


def _codes(q: torch.Tensor) -> torch.Tensor:
    return q.view(torch.uint8)


class _FakeDpGroup:
    """Concat-style all-gather that records what crossed the wire."""

    def __init__(self, world_size: int = DP):
        self.world_size = world_size
        self.gathered: list[tuple[torch.dtype, tuple[int, ...]]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0):
        self.gathered.append((tensor.dtype, tuple(tensor.shape)))
        return torch.cat([tensor] * self.world_size, dim=dim)

    def wire_bytes(self) -> int:
        return sum(
            torch.empty(0, dtype=dt).element_size() *
            int(torch.tensor(shape).prod()) for dt, shape in self.gathered)


def _communicator(dispatch_fp8: bool,
                  monkeypatch,
                  world_size: int = DP,
                  offload_threshold: int | None = None) -> tuple:
    comm = object.__new__(tpu_communicator.TpuDeviceCommunicator)
    comm.dispatch_fp8 = dispatch_fp8
    comm.packed_wire = fp8_wire.packed_wire()
    comm.offload_threshold = offload_threshold
    group = _FakeDpGroup(world_size)
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: group)
    return comm, group


@pytest.mark.parametrize("env_value,expected", [(0, None), (512, 512)])
def test_wire_block_is_the_row_or_exactly_the_kernel_block(
        monkeypatch, env_value, expected):
    # A partial block width would be the worst of both: gmm_v2 re-derives its
    # lhs scale over LHS_QUANT_BLOCK_SIZE-wide k slices, so anything other than
    # "one scale per row" or "exactly that block" costs a second rounding for
    # no bytes saved. Assert every value the env var can take, not just the
    # default -- a disjunction that only the default satisfies tests nothing.
    from vllm_torchtpu.kernels.megablox import gmm_v2

    monkeypatch.setattr(fp8_wire.envs, "TPU_MOE_DISPATCH_FP8_BLOCK", env_value)

    assert LHS_QUANT_BLOCK_SIZE == gmm_v2.LHS_QUANT_BLOCK_SIZE
    assert fp8_wire.wire_block() == expected
    assert fp8_wire.wire_block() in (None, gmm_v2.LHS_QUANT_BLOCK_SIZE)


def test_a_hidden_size_the_kernel_block_does_not_divide_falls_back_to_the_row(
        kernel_block_wire):
    # Mismatched blocks are the worst case (two roundings on two grids), so an
    # H the kernel's block does not tile must take the row form instead.
    assert fp8_wire._block_for(3 * LHS_QUANT_BLOCK_SIZE // 2) == 768
    assert fp8_wire._block_for(4 *
                               LHS_QUANT_BLOCK_SIZE) == LHS_QUANT_BLOCK_SIZE


def test_default_wire_is_two_async_collectives_not_one_packed_buffer(
        monkeypatch):
    """The property the two packed-uint8 device runs lacked.

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


def test_block_max_comes_back_saturated_so_the_kernel_requant_is_a_no_op(
        kernel_block_wire):
    torch.manual_seed(0)
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16)

    q, _ = fp8_wire.quantize_rows(x)

    per_block = q.float().unflatten(-1, (-1, kernel_block_wire))
    assert per_block.shape[1] > 1, "a single block per row tests nothing"
    assert torch.all(per_block.abs().amax(dim=-1) == fp8_wire.WIRE_MAX)


def test_the_saturation_test_has_teeth_a_row_scale_does_not_saturate_blocks():
    # Negative control for the test above: with one scale per row only the
    # block holding the row max comes back at WIRE_MAX, which is why the
    # kernel re-derives seven other scales and rounds the payload again.
    torch.manual_seed(0)
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16)

    q, _ = fp8_wire.quantize_rows(x)

    per_block = q.float().unflatten(-1, (-1, LHS_QUANT_BLOCK_SIZE))
    assert not torch.all(per_block.abs().amax(dim=-1) == fp8_wire.WIRE_MAX)


def test_the_kernel_requant_returns_the_wire_codes_and_the_bf16_scale(
        kernel_block_wire):
    """The property the whole block form exists for.

    On the kernel's grid the wire hands gmm_v2 exactly the lhs it would have
    computed from the bf16 rows: the same fp8 codes and the same block scale,
    bit for bit. The layer's output is therefore identical to the bf16
    dispatch's -- the fp8 wire costs bytes on ICI and nothing in accuracy.
    """
    torch.manual_seed(0)
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16)

    q, scale = fp8_wire.quantize_rows(x)
    got_q, got_scale = _kernel_requantize(
        fp8_wire.dequantize_rows(q, scale, x.dtype))
    want_q, want_scale = _kernel_requantize(x)  # what the bf16 wire hands it

    assert torch.equal(_codes(got_q), _codes(q))
    assert torch.equal(_codes(got_q), _codes(want_q))
    assert torch.equal(got_scale, want_scale)


def test_a_row_scale_makes_the_kernel_round_the_payload_a_second_time():
    # Negative control: the same comparison under the shipped default. The
    # kernel keeps neither the wire's codes nor the bf16 path's, which is the
    # 1.38x error amplification the row form pays.
    torch.manual_seed(0)
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16)

    q, scale = fp8_wire.quantize_rows(x)
    got_q, _ = _kernel_requantize(fp8_wire.dequantize_rows(q, scale, x.dtype))
    want_q, _ = _kernel_requantize(x)

    assert not torch.equal(_codes(got_q), _codes(q))
    assert not torch.equal(_codes(got_q), _codes(want_q))


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
@pytest.mark.parametrize("block", [0, LHS_QUANT_BLOCK_SIZE])
def test_dequantized_rows_keep_the_input_dtype(dtype, block, monkeypatch):
    # The kernel-grid path scales by a reciprocal in the payload's dtype, so
    # it has to survive every dtype this entry point advertises.
    monkeypatch.setattr(fp8_wire.envs, "TPU_MOE_DISPATCH_FP8_BLOCK", block)
    comm, _ = _communicator(True, monkeypatch)

    out = comm.dispatch_hidden_states(torch.randn(8, HIDDEN, dtype=dtype))

    assert out.dtype is dtype
    assert torch.all(torch.isfinite(out))


def test_dispatch_hidden_states_is_a_no_op_without_dp(monkeypatch):
    # Callers outside this class (a patched MoERunner._maybe_dispatch) route
    # the activation gather through here unconditionally.
    comm, group = _communicator(True, monkeypatch, world_size=1)
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16)

    assert comm.dispatch_hidden_states(x) is x
    assert group.gathered == []


def test_the_kernel_block_wire_sends_one_scale_per_block(
        monkeypatch, kernel_block_wire):
    comm, group = _communicator(True, monkeypatch)
    comm.dispatch_hidden_states(torch.randn(8, HIDDEN, dtype=torch.bfloat16))

    blocks = HIDDEN // kernel_block_wire
    assert [dt for dt, _ in group.gathered
            ] == [torch.float8_e4m3fn, torch.float32]
    assert group.gathered[0][1] == (8, HIDDEN)
    assert group.gathered[1][1] == (8, blocks)


# The SparseCore all-gather offload threshold on tpu7x: env_override sets the
# flag to the chip's per-tensor-core VMEM, and the predicate is >= on the
# gathered output buffer.
TPU7X_OFFLOAD_BYTES = 64 * 2**20
H_QWEN = 4096


@pytest.mark.parametrize(
    "dp,rows,fp8",
    [
        # DP8_EP, mbt=2048: only the full-prefill bucket clears the threshold,
        # and it clears it by exactly 0 bytes (16384 x 4096 x 1 B = 64 MiB).
        (8, 2048, True),
        (8, 1024, False),  # gathered 8192: the bucket that flips async->sync
        (8, 64, False),  # decode at conc 512
        (8, 8, False),  # decode at conc 64
        # DP4TP2_EP, mbt=4096: same gathered shapes, half the ranks.
        (4, 4096, True),
        (4, 2048, False),
        (4, 16, False),
        # No DP: dispatch_hidden_states is a no-op before the gate is reached.
        (1, 4096, False),
    ])
def test_the_offload_gate_takes_fp8_only_where_the_gather_stays_async(
        monkeypatch, dp, rows, fp8):
    comm, group = _communicator(True,
                                monkeypatch,
                                world_size=dp,
                                offload_threshold=TPU7X_OFFLOAD_BYTES)
    comm.dispatch_hidden_states(torch.randn(rows, H_QWEN,
                                            dtype=torch.bfloat16))

    sent = [dt for dt, _ in group.gathered]
    assert sent == ([torch.float8_e4m3fn, torch.float32]
                    if fp8 else [torch.bfloat16] if dp > 1 else [])


def test_the_offload_gate_is_off_unless_asked_for(monkeypatch):
    # Default: no threshold read, so TPU_MOE_DISPATCH_FP8=1 alone puts the fp8
    # wire on at every bucket, decode included.
    comm, group = _communicator(True, monkeypatch)
    comm.dispatch_hidden_states(torch.randn(8, HIDDEN, dtype=torch.bfloat16))

    assert [dt for dt, _ in group.gathered
            ] == [torch.float8_e4m3fn, torch.float32]


def test_the_offload_threshold_is_read_from_the_flag_xla_will_see(monkeypatch):
    from vllm_torchtpu import env_override

    flag = env_override.SC_ALL_GATHER_OFFLOAD_FLAG
    monkeypatch.setenv(
        "LIBTPU_INIT_ARGS",
        f"--xla_tpu_use_dynamic_smem_negotiation=true {flag}=1234")
    assert env_override.sc_all_gather_offload_threshold_bytes() == 1234

    # No flag (SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES=0, or an unknown chip
    # family): libtpu's own default applies and we must not guess it.
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_tpu_scoped_vmem_limit_kib=1")
    assert env_override.sc_all_gather_offload_threshold_bytes() is None


class _RecordingLogger:
    """Captures the boot-time markers a configuration is read back from."""

    def __init__(self):
        self.lines: list[str] = []

    def info_once(self, msg, *args, **kwargs):
        self.lines.append(msg)

    warning_once = info_once


# The four knob combinations worth booting, and the marker that must be in the
# server log for each. `forbid` is the point: a marker that cannot appear when
# the knob is off is what makes the log unambiguous.
ARMS = [
    ("ctl", False, 0, False, [], ["[fp8wire]"]),
    ("c9", True, 0, False, ["block=row",
                            "offload gate=off"], ["block=512", "gate=on"]),
    # The gate is refused on this runner (see _offload_gate_is_traceable), so
    # a run that asks for it must say REFUSED rather than look engaged.
    ("c9AB", True, LHS_QUANT_BLOCK_SIZE, True,
     ["block=512", "offload gate=REFUSED"], ["block=row",
                                             "gate=on threshold"]),
    ("c9B", True, 0, True, ["block=row", "offload gate=REFUSED"],
     ["block=512", "gate=on threshold"]),
]


@pytest.mark.parametrize("arm,fp8,block,gate,want,forbid", ARMS)
def test_the_boot_log_names_which_arm_is_live(monkeypatch, arm, fp8, block,
                                              gate, want, forbid):
    monkeypatch.setattr(fp8_wire.envs, "TPU_MOE_DISPATCH_FP8_BLOCK", block)
    monkeypatch.setattr(tpu_communicator.envs,
                        "TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD", gate)
    recorder = _RecordingLogger()
    monkeypatch.setattr(tpu_communicator, "logger", recorder)
    comm, _ = _communicator(
        fp8,
        monkeypatch,
        offload_threshold=TPU7X_OFFLOAD_BYTES if gate else None)

    comm._log_engagement()

    log = "\n".join(recorder.lines)
    assert all(s in log for s in want), (arm, log)
    assert not any(s in log for s in forbid), (arm, log)


def test_the_gate_marker_says_so_when_it_cannot_resolve_a_threshold(
        monkeypatch):
    # A gate whose flag is absent is inert; it must not read as engaged.
    monkeypatch.setattr(tpu_communicator.envs,
                        "TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD", True)
    monkeypatch.setattr(tpu_communicator, "_offload_gate_is_traceable",
                        lambda: True)
    recorder = _RecordingLogger()
    monkeypatch.setattr(tpu_communicator, "logger", recorder)
    comm, _ = _communicator(True, monkeypatch, offload_threshold=None)

    comm._log_engagement()

    assert "offload gate=on but INERT" in "\n".join(recorder.lines)


def _compile_dispatch(comm,
                      rows: int,
                      hidden: int = HIDDEN,
                      max_rows: int | None = None):
    """Trace dispatch_hidden_states the way the runner traces the backbone.

    vLLM marks the token dim dynamic before compiling, so anything that turns
    the row count into a Python constant raises ConstraintViolationError at
    engine init rather than silently re-specializing. Reproducing that here is
    the check that was missing when the offload gate shipped: on device, every
    server with the gate on died in exactly this way.
    """
    torch._dynamo.reset()
    x = torch.randn(rows, hidden, dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x, 0)

    def backbone(t):
        if max_rows is not None:
            # The runner's largest bucket: the upper bound that turns the
            # gate's ">= threshold" into an equality on the token dim.
            torch._check(t.size(0) <= max_rows)
        return comm.dispatch_hidden_states(t)

    return torch.compile(backbone, backend="eager", fullgraph=True)(x)


@pytest.mark.parametrize("block", [0, LHS_QUANT_BLOCK_SIZE])
def test_the_wire_traces_with_the_token_dim_dynamic(monkeypatch, block):
    # Knob A is legal in a dynamic-token-dim graph because every decision it
    # makes is on the hidden width, which is static: _block_for(x.shape[-1]),
    # the slice bounds, and the scale count. Two row counts, one graph.
    monkeypatch.setattr(fp8_wire.envs, "TPU_MOE_DISPATCH_FP8_BLOCK", block)
    comm, group = _communicator(True, monkeypatch)

    for rows in (128, 2048):
        out = _compile_dispatch(comm, rows)
        assert out.shape == (rows * DP, HIDDEN)
        assert out.dtype is torch.bfloat16


def test_a_wire_chosen_per_bucket_cannot_be_traced(monkeypatch):
    """The falsification, pinned: this is why the gate is refused.

    Forcing a threshold past the refusal reproduces the device failure on
    CPU in a second. If this test ever stops raising, the runner has gained
    per-bucket retracing and the gate can be re-enabled.
    """
    from torch.fx.experimental.symbolic_shapes import ConstraintViolationError

    comm, _ = _communicator(True,
                            monkeypatch,
                            offload_threshold=TPU7X_OFFLOAD_BYTES)

    # DP4TP2's top bucket: 4096 rows x dp4 x 4096 x 1 B is the threshold to
    # the byte, so ">= threshold" collapses the dim's range to the point 4096.
    # That zero-byte margin is the same one #499 depends on for its win.
    with pytest.raises(ConstraintViolationError) as excinfo:
        _compile_dispatch(comm, 4096, hidden=H_QWEN, max_rows=4096)
    assert "specialized it to be a constant" in str(excinfo.value)


def test_the_gate_env_var_is_refused_and_says_so(monkeypatch):
    # Belt and braces: with the knob set, the communicator resolves no
    # threshold, so the traced path is the ungated one and the server boots.
    monkeypatch.setattr(tpu_communicator.envs,
                        "TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD", True)
    monkeypatch.setattr(tpu_communicator, "_offload_gate_is_traceable",
                        lambda: False)
    recorder = _RecordingLogger()
    monkeypatch.setattr(tpu_communicator, "logger", recorder)
    comm, _ = _communicator(True, monkeypatch)
    comm.offload_threshold = None

    comm._log_engagement()
    out = _compile_dispatch(comm, 2048, hidden=H_QWEN)

    assert "offload gate=REFUSED" in "\n".join(recorder.lines)
    assert out.shape == (2048 * DP, H_QWEN)
