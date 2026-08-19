# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the fp8 MoE dispatch all-gather (TPU_MOE_DISPATCH_FP8)."""

import ast
import multiprocessing
import os
import pathlib
import queue as queue_module
import traceback

import pytest
import torch
import torch.distributed as dist

from vllm_torchtpu.distributed import fp8_wire, tpu_communicator

DP = 4
HIDDEN = 2048
# kernels.megablox.gmm_v2.LHS_QUANT_BLOCK_SIZE, spelled out so this file stays
# importable without jax; the contract test below pins the two together.
LHS_QUANT_BLOCK_SIZE = 512
BLOCK = fp8_wire._block_for(HIDDEN)  # the whole row unless the knob is set
NUM_BLOCKS = HIDDEN // BLOCK


def _cpu_rng(seed: int = 0) -> torch.Generator:
    """One CPU generator, instead of seeding every device.

    ``torch.manual_seed`` also seeds each registered device backend, so with
    torch_tpu loaded it calls ``torch.tpu.manual_seed_all`` -- which builds the
    PjRt client inside this process and takes /tmp/libtpu_lockfile away from
    every isolated-subprocess TPU test that runs after this file. Nothing here
    needs a device, so nothing here touches one.
    """
    return torch.Generator().manual_seed(seed)


# Two calls open the device runtime from whatever process makes them, and
# neither looks like it touches a device:
#
#   torch.manual_seed -> torch.random._seed_custom_device -> tpu.manual_seed_all
#   torch.compile     -> dynamo's SymbolicStreamState -> current_stream() ->
#                        torch._C._accelerator_getDeviceIndex()
#
# In the pytest parent either one creates the PjRt client and takes
# /tmp/libtpu_lockfile for the rest of the session, and what then fails is not
# this file but the next isolated-subprocess TPU test -- with a lockfile error
# that reads like a dirty machine. So: seed a local generator, and trace only
# inside a `_child_` function that `_run_in_subprocess` runs elsewhere.
_FORBIDDEN_ANYWHERE = ("torch.manual_seed", "torch.seed")
_TRACING_CALLS = {
    "torch.compile": ("_compile_dispatch", ),
    "_compile_dispatch": None
}  # None = only `_child_` callers


def test_this_file_never_touches_the_device_from_the_pytest_process():
    """The rule above, pinned, because breaking it fails somewhere else."""
    tree = ast.parse(pathlib.Path(__file__).read_text())
    scopes = [(node.name, node) for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)]
    offences = []
    for scope, node in scopes:
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            called = ast.unparse(call.func)
            if called in _FORBIDDEN_ANYWHERE:
                offences.append(f"{scope} calls {called}")
            if called in _TRACING_CALLS:
                allowed = _TRACING_CALLS[called]
                if (scope not in (allowed or ())
                        and not scope.startswith("_child_")):
                    offences.append(f"{scope} traces via {called}")

    assert not offences, offences


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
                  monkeypatch=None,
                  world_size: int = DP,
                  offload_threshold: int | None = None) -> tuple:
    comm = object.__new__(tpu_communicator.TpuDeviceCommunicator)
    comm.dispatch_fp8 = dispatch_fp8
    comm.packed_wire = fp8_wire.packed_wire()
    comm.offload_threshold = offload_threshold
    group = _FakeDpGroup(world_size)
    if monkeypatch is None:  # a spawned child (below) has no monkeypatch, and
        tpu_communicator.get_dp_group = lambda: group  # nothing runs after it
    else:
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
    rng = _cpu_rng()
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, generator=rng)

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
    rng = _cpu_rng()
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16, generator=rng)

    q, _ = fp8_wire.quantize_rows(x)

    per_block = q.float().unflatten(-1, (-1, kernel_block_wire))
    assert per_block.shape[1] > 1, "a single block per row tests nothing"
    assert torch.all(per_block.abs().amax(dim=-1) == fp8_wire.WIRE_MAX)


def test_the_saturation_test_has_teeth_a_row_scale_does_not_saturate_blocks():
    # Negative control for the test above: with one scale per row only the
    # block holding the row max comes back at WIRE_MAX, which is why the
    # kernel re-derives seven other scales and rounds the payload again.
    rng = _cpu_rng()
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16, generator=rng)

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
    rng = _cpu_rng()
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, generator=rng)

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
    rng = _cpu_rng()
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, generator=rng)

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
    rng = _cpu_rng()
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, generator=rng)

    payload = fp8_wire.encode(x)
    back = fp8_wire.decode(payload, HIDDEN, x.dtype)

    assert payload.dtype is torch.uint8
    assert payload.shape == (8, HIDDEN + NUM_BLOCKS * 4)
    q, scale = fp8_wire.quantize_rows(x)
    assert torch.equal(back, fp8_wire.dequantize_rows(q, scale, x.dtype))


def test_dispatch_router_logits_halves_the_hidden_payload(monkeypatch):
    rng = _cpu_rng()
    hidden = torch.randn(32, HIDDEN, dtype=torch.bfloat16, generator=rng)
    logits = torch.randn(32, 64, dtype=torch.bfloat16, generator=rng)

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
    rng = _cpu_rng()
    hidden = torch.randn(16, HIDDEN, dtype=torch.bfloat16, generator=rng)
    weights = torch.rand(16, 4, dtype=torch.bfloat16, generator=rng)
    ids = torch.randint(0, 64, (16, 4), dtype=torch.int32, generator=rng)

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


def _tracing_worker_loop(jobs, results):
    """Run ``(name, args)`` jobs from this module until told to stop."""
    while True:
        job = jobs.get()
        if job is None:
            return
        name, args = job
        try:
            globals()[name](*args)
            results.put(("OK", None))
        except BaseException:  # reported to the parent, which re-raises it
            results.put(("ERROR", traceback.format_exc()))


class _TracingWorker:
    """A spawned process that does this file's tracing, so we do not.

    Dynamo builds a ``SymbolicStreamState`` for every frame it traces --
    ``torch.compile`` and ``dynamo.explain`` alike -- and that asks
    ``torch.accelerator.current_stream()`` for the current device index. Where
    torch_tpu is the registered accelerator, that call creates the PjRt client
    and takes /tmp/libtpu_lockfile for the life of the process;
    ``backend="eager"`` does not help, because the door is in the tracer and
    not in the backend. Left in the pytest process, the lock then breaks the
    next isolated-subprocess TPU test in the session -- which is what
    tests/distributed/test_kv_scatter.py, three tests later, needs.

    One worker rather than one process per case: each spawn pays ~19 s of
    interpreter and vLLM import. It exits with the module, releasing the lock.
    """

    def __init__(self, ctx):
        self._jobs, self._results = ctx.Queue(), ctx.Queue()
        self._proc = ctx.Process(target=_tracing_worker_loop,
                                 args=(self._jobs, self._results),
                                 daemon=True)
        self._proc.start()

    def run(self, target, *args):
        """Run a top-level ``_child_`` function there, re-raising failures."""
        self._jobs.put((target.__name__, args))
        try:
            status, payload = self._results.get(timeout=300)
        except queue_module.Empty:
            raise AssertionError("the tracing worker never answered") from None
        if status == "ERROR":
            raise AssertionError(f"tracing worker failed:\n{payload}")

    def stop(self):
        self._jobs.put(None)
        self._proc.join(timeout=60)
        if self._proc.is_alive():
            self._proc.kill()


@pytest.fixture(scope="module")
def traced_elsewhere():
    worker = _TracingWorker(multiprocessing.get_context("spawn"))
    try:
        yield worker
    finally:
        worker.stop()


def _child_the_wire_traces_with_the_token_dim_dynamic(block):
    # Knob A is legal in a dynamic-token-dim graph because every decision it
    # makes is on the hidden width, which is static: _block_for(x.shape[-1]),
    # the slice bounds, and the scale count. Two row counts, one graph.
    os.environ["TPU_MOE_DISPATCH_FP8_BLOCK"] = str(block)
    comm, _ = _communicator(True)

    for rows in (128, 2048):
        out = _compile_dispatch(comm, rows)
        assert out.shape == (rows * DP, HIDDEN)
        assert out.dtype is torch.bfloat16


def _child_a_wire_chosen_per_bucket_cannot_be_traced():
    from torch.fx.experimental.symbolic_shapes import ConstraintViolationError

    comm, _ = _communicator(True, offload_threshold=TPU7X_OFFLOAD_BYTES)

    # DP4TP2's top bucket: 4096 rows x dp4 x 4096 x 1 B is the threshold to
    # the byte, so ">= threshold" collapses the dim's range to the point 4096.
    # That zero-byte margin is the same one #499 depends on for its win.
    try:
        _compile_dispatch(comm, 4096, hidden=H_QWEN, max_rows=4096)
    except ConstraintViolationError as exc:
        assert "specialized it to be a constant" in str(exc)
    else:
        raise AssertionError(
            "the per-bucket branch traced: the runner has gained what the "
            "offload gate needs, and the refusal can be revisited")


def _child_the_gate_env_var_is_refused_and_says_so():
    # Belt and braces: with the knob set, the communicator resolves no
    # threshold, so the traced path is the ungated one and the server boots.
    os.environ["TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD"] = "1"
    recorder = _RecordingLogger()
    tpu_communicator.logger = recorder
    tpu_communicator._offload_gate_is_traceable = lambda: False
    comm, _ = _communicator(True)
    comm.offload_threshold = None

    comm._log_engagement()
    out = _compile_dispatch(comm, 2048, hidden=H_QWEN)

    assert "offload gate=REFUSED" in "\n".join(recorder.lines)
    assert out.shape == (2048 * DP, H_QWEN)


@pytest.mark.parametrize("block", [0, LHS_QUANT_BLOCK_SIZE])
def test_the_wire_traces_with_the_token_dim_dynamic(block, traced_elsewhere):
    traced_elsewhere.run(_child_the_wire_traces_with_the_token_dim_dynamic,
                         block)


def test_a_wire_chosen_per_bucket_cannot_be_traced(traced_elsewhere):
    """The falsification, pinned: this is why the gate is refused.

    Forcing a threshold past the refusal reproduces the device failure on
    CPU in a second. If this test ever stops raising, the runner has gained
    per-bucket retracing and the gate can be re-enabled.
    """
    traced_elsewhere.run(_child_a_wire_chosen_per_bucket_cannot_be_traced)


def test_the_gate_env_var_is_refused_and_says_so(traced_elsewhere):
    traced_elsewhere.run(_child_the_gate_env_var_is_refused_and_says_so)


# --- Against the shipped object, not a stand-in ------------------------------
#
# Every test above builds the communicator with ``object.__new__`` and assigns
# its three fields, so none of them can catch an env var that never reaches
# them. The rest of the file builds the real thing: TpuDeviceCommunicator over
# a real torch.distributed group, configured only by the environment.


@pytest.fixture
def solo_world():
    """A real single-rank gloo group: what a dp_size == 1 rank really has."""
    dist.init_process_group(backend="gloo",
                            store=dist.HashStore(),
                            rank=0,
                            world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.fixture
def dp4_world():
    """Four ranks without four processes.

    torch's ``fake`` backend gives real ProcessGroup plumbing and real
    collective shapes; the contents of a gathered buffer are undefined, so
    tests on this fixture assert what crossed the wire and never its values.
    """
    dist.init_process_group(backend="fake",
                            store=dist.HashStore(),
                            rank=0,
                            world_size=DP)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def _shipped_communicator(process_group):
    """The communicator vLLM builds, through its own __init__."""
    return tpu_communicator.TpuDeviceCommunicator(cpu_group=process_group,
                                                  device=torch.device("cpu"),
                                                  device_group=process_group,
                                                  unique_name="ep")


def _only_the_wire_flag(monkeypatch):
    """TPU_MOE_DISPATCH_FP8=1 and no other knob, as an operator would set it."""
    monkeypatch.setenv("TPU_MOE_DISPATCH_FP8", "1")
    for knob in ("TPU_MOE_DISPATCH_FP8_WIRE", "TPU_MOE_DISPATCH_FP8_BLOCK",
                 "TPU_MOE_DISPATCH_FP8_GATE_OFFLOAD"):
        monkeypatch.delenv(knob, raising=False)


class _TorchDpGroup:
    """The two members of a DP group coordinator that the wire calls.

    vLLM's own GroupCoordinator cannot be built for dp>1 on a CPU host (its
    per-rank subgroups need the ranks), so the collective here is torch's, on
    the fixture's process group -- not a torch.cat standing in for one.
    """

    def __init__(self, process_group):
        self._group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.gathered: list[tuple[torch.dtype, tuple[int, ...]]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0):
        assert dim == 0, "the MoE dispatch only ever gathers on the token dim"
        self.gathered.append((tensor.dtype, tuple(tensor.shape)))
        out = torch.zeros(
            (tensor.shape[0] * self.world_size, *tensor.shape[1:]),
            dtype=tensor.dtype)
        dist.all_gather_into_tensor(out,
                                    tensor.contiguous(),
                                    group=self._group)
        return out


def test_the_shipped_default_wires_fp8_rows_and_an_fp32_scale(
        monkeypatch, dp4_world):
    """What an operator gets from ``TPU_MOE_DISPATCH_FP8=1`` and nothing else.

    The knobs are read by the real __init__ from the real environment, so this
    fails if an env var is renamed, mistyped in envs.py, or stops being read at
    construction -- none of which the hand-built communicators above can see.
    """
    _only_the_wire_flag(monkeypatch)

    comm = _shipped_communicator(dp4_world)

    assert comm.dispatch_fp8 is True
    assert comm.packed_wire is False  # two collectives, not one uint8 buffer
    assert comm.offload_threshold is None  # the gate is opt-in and refused
    assert fp8_wire.wire_block() is None  # one scale per row, the shipped grid

    group = _TorchDpGroup(dp4_world)
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: group)
    out = comm.dispatch_hidden_states(
        torch.randn(8, HIDDEN, dtype=torch.bfloat16))

    assert [dt for dt, _ in group.gathered
            ] == [torch.float8_e4m3fn, torch.float32]
    assert [shape for _, shape in group.gathered] == [(8, HIDDEN), (8, 1)]
    assert out.shape == (8 * DP, HIDDEN) and out.dtype is torch.bfloat16


def test_the_wire_is_inert_at_dp_size_1(monkeypatch, solo_world):
    """Ten of the fourteen InferenceX sweep points run TP8_EP, i.e. dp_size 1.

    Against a real vLLM GroupCoordinator, because "the flag does nothing here"
    is a claim about vLLM's group, not about our arithmetic.
    """
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.platforms import current_platform

    _only_the_wire_flag(monkeypatch)
    # GroupCoordinator resolves torch.device(f"{device_name}:{local_rank}") for
    # an out-of-tree platform, and "tpu" is not a torch device on a CPU host.
    # Nothing else in this test depends on the name.
    monkeypatch.setattr(type(current_platform), "device_name", "cpu")
    dp_group = GroupCoordinator(group_ranks=[[0]],
                                local_rank=0,
                                torch_distributed_backend="gloo",
                                use_device_communicator=False,
                                group_name="dp")
    try:
        monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: dp_group)
        comm = _shipped_communicator(solo_world)

        rng = _cpu_rng()
        x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, generator=rng)
        logits = torch.randn(8, 16, dtype=torch.bfloat16, generator=rng)

        assert dp_group.world_size == 1
        assert comm.dispatch_hidden_states(x) is x
        assert comm.dispatch_router_logits(x, logits) == (x, logits)
        assert comm.combine(x) is x
    finally:
        # A GroupCoordinator registers itself process-wide and owns two torch
        # subgroups; leaving it behind is state the next test file inherits.
        dp_group.destroy()


# --- Which activation gathers the wire owns, per the installed vLLM ----------
#
# dispatch_hidden_states is only reached if upstream's MoERunner routes through
# it, and it is not the only MoE activation gather in a step. These run
# upstream's own _maybe_dispatch, so the docstring's account of which leg is
# ours cannot drift away from the vLLM we pin.


@pytest.fixture
def patched_pcp_collectives():
    """Apply the TPU PCP patch for one test, then put the class back.

    ``_patch_moe_explicit_pcp_collectives`` rewrites a property on an upstream
    class process-wide. Production applies it once at startup; a test must not
    leave it applied for whatever runs next in the same process.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig

    from vllm_torchtpu import _patch_moe_explicit_pcp_collectives

    original = FusedMoEParallelConfig.use_all2all_kernels
    marked = getattr(FusedMoEParallelConfig,
                     "_tpu_explicit_pcp_collectives_patch", False)
    _patch_moe_explicit_pcp_collectives()
    try:
        yield
    finally:
        FusedMoEParallelConfig.use_all2all_kernels = original
        FusedMoEParallelConfig._tpu_explicit_pcp_collectives_patch = marked


def _moe_config(dp: int, pcp: int, use_ep: bool):
    """A real FusedMoEConfig, so use_all2all_kernels is upstream's own rule."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig)
    from vllm.model_executor.layers.fused_moe.router.router_factory import \
        RoutingMethodType

    # The `patched_pcp_collectives` fixture has applied the TPU patch that
    # keeps the explicit PCP collectives alive for our monolithic kernels --
    # part of what these cases test -- and undoes it afterwards.
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=pcp,
        dp_size=dp,
        ep_size=dp if use_ep else 1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False)
    return FusedMoEConfig(num_experts=8,
                          experts_per_token=2,
                          hidden_dim=HIDDEN,
                          intermediate_size=HIDDEN,
                          num_local_experts=8,
                          num_logical_experts=8,
                          activation=MoEActivation.SILU,
                          device=torch.device("cpu"),
                          routing_method=RoutingMethodType.Default,
                          moe_parallel_config=parallel,
                          in_dtype=torch.bfloat16)


def _moe_runner(dp: int, pcp: int, use_ep: bool):
    """Upstream's MoERunner with only the state _maybe_dispatch reads."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import \
        MoERunner

    class _Runner(MoERunner):

        def __init__(self):  # MoERunner.__init__ builds a whole layer
            self.moe_config = _moe_config(dp, pcp, use_ep)
            # supports_internal_mk is False for every TPU quant method: they
            # are all monolithic, which is what puts us on the naive DP path.
            self.routed_experts = type(
                "_Experts", (), {
                    "quant_method":
                    type("_Method", (), {"supports_internal_mk": False})()
                })()

    return _Runner()


# (dp, pcp, EP) -> what the DP leg puts on the wire, what the PCP leg does.
DISPATCH_LEGS = [
    # Plain DP+EP: our wire, and it is the only activation gather.
    (4, 1, True, "fp8", None),
    # DP+EP with PCP: upstream skips its PCP leg once use_all2all_kernels is
    # true, which dp>1 makes it. Still only our gather.
    (4, 4, True, "fp8", None),
    # The prefill server of examples/disagg/launch_qwen35_pcp4_dp4.sh: our
    # patch keeps the PCP leg explicit, and at dp==1 the wire is inert -- so
    # every activation byte crosses in bf16 and the flag cannot reach it.
    (1, 4, True, None, "bf16"),
    # Both legs, the only configuration that has them: the PCP gather is dp
    # times the DP gather's rows, i.e. the fp8 wire covers the smaller one.
    (4, 4, False, "fp8", "bf16"),
]


@pytest.mark.parametrize("dp,pcp,use_ep,dp_leg,pcp_leg", DISPATCH_LEGS)
def test_which_dispatch_legs_the_fp8_wire_owns(monkeypatch, dp4_world,
                                               patched_pcp_collectives, dp,
                                               pcp, use_ep, dp_leg, pcp_leg):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    _only_the_wire_flag(monkeypatch)
    comm = _shipped_communicator(dp4_world)
    dp_group, pcp_group = _FakeDpGroup(dp), _FakeDpGroup(pcp)
    monkeypatch.setattr(tpu_communicator, "get_dp_group", lambda: dp_group)
    # A GroupCoordinator forwards dispatch_router_logits to its device
    # communicator; handing over the communicator drops that one indirection.
    monkeypatch.setattr(moe_runner, "get_ep_group", lambda: comm)
    monkeypatch.setattr(moe_runner, "get_pcp_group", lambda: pcp_group)

    rows = 8
    hidden = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    logits = torch.randn(rows, 16, dtype=torch.bfloat16)
    out, _ = _moe_runner(dp, pcp, use_ep)._maybe_dispatch(hidden, logits)

    # The DP leg carries rows, scales and the (always bf16) router logits.
    assert [dt for dt, _ in dp_group.gathered
            ] == ([torch.float8_e4m3fn, torch.float32, torch.bfloat16]
                  if dp_leg else [])
    assert [dt for dt, _ in pcp_group.gathered
            ] == ([torch.bfloat16] * 2 if pcp_leg else [])
    assert out.shape == (rows * (dp if dp_leg else 1) *
                         (pcp if pcp_leg else 1), HIDDEN)
    if dp_leg and pcp_leg:
        assert pcp_group.gathered[0][1][0] == dp * dp_group.gathered[0][1][0]


# --- The offload-threshold margin, checked at boot ---------------------------
#
# The wire wins only while the gathered fp8 buffer stays at or above the
# SparseCore all-gather offload threshold, and at the shipped geometry it
# clears that threshold by zero bytes. The gate that would have decided this
# per bucket is refused (above), so what is left is to say so at init.

# The shipped serving geometry: Qwen3.5 at hidden 4096, DP8_EP,
# max_num_batched_tokens 2048 -> buckets 16..2048, top gather 64 MiB of fp8.
QWEN_BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048]


class _Config:
    """The fields _geometry reads out of a VllmConfig."""

    def __init__(self, dp, hidden, buckets, max_num_batched_tokens=0):
        self.parallel_config = type("_P", (), {"data_parallel_size": dp})()
        self.model_config = type("_M", (),
                                 {"get_hidden_size": lambda self: hidden})()
        self.compilation_config = type("_C", (), {"compile_sizes": buckets})()
        self.scheduler_config = type(
            "_S", (), {"max_num_batched_tokens": max_num_batched_tokens})()


def _boot_log(monkeypatch,
              *,
              dp,
              hidden,
              buckets,
              threshold,
              packed=False,
              max_num_batched_tokens=0):
    """The [fp8wire] lines a rank with this geometry prints at init."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(tpu_communicator, "logger", recorder)
    monkeypatch.setattr(
        tpu_communicator, "get_current_vllm_config_or_none",
        lambda: _Config(dp, hidden, buckets, max_num_batched_tokens))
    monkeypatch.setattr(tpu_communicator,
                        "sc_all_gather_offload_threshold_bytes",
                        lambda: threshold)
    comm, _ = _communicator(True, monkeypatch)
    comm.packed_wire = packed

    comm._log_engagement()

    return "\n".join(recorder.lines)


def test_the_shipped_geometry_clears_the_offload_threshold_by_nothing():
    # 2048 x dp8 x 4096 == 67108864 == the tpu7x threshold, on a >=. The whole
    # measured win rests on that equality; the warning below exists for it.
    assert 2048 * 8 * H_QWEN == TPU7X_OFFLOAD_BYTES


@pytest.mark.parametrize(
    "dp,hidden,buckets,threshold,warning",
    [
        # The shipped geometry: margin 0 B.
        (8, H_QWEN, QWEN_BUCKETS, TPU7X_OFFLOAD_BYTES, "thin offload margin"),
        # DP4TP2_EP, mbt 4096: same gathered shape, same zero margin.
        (4, H_QWEN, QWEN_BUCKETS + [4096], TPU7X_OFFLOAD_BYTES,
         "thin offload margin"),
        # One halving of the gathered rows -- a smaller batch, two
        # microbatches -- and every bucket is under the threshold.
        (8, H_QWEN, QWEN_BUCKETS[:-1], TPU7X_OFFLOAD_BYTES,
         "no offload margin at all"),
        # A whole bucket-halving of slack is enough to stop warning.
        (8, H_QWEN, QWEN_BUCKETS + [4096], TPU7X_OFFLOAD_BYTES, None),
        # dp 1: the wire is inert, so the threshold cannot matter.
        (1, H_QWEN, QWEN_BUCKETS, TPU7X_OFFLOAD_BYTES, None),
    ])
def test_the_boot_log_warns_exactly_when_the_margin_is_under_one_bucket(
        monkeypatch, dp, hidden, buckets, threshold, warning):
    log = _boot_log(monkeypatch,
                    dp=dp,
                    hidden=hidden,
                    buckets=buckets,
                    threshold=threshold)

    assert ("offload margin" in log) is (warning is not None)
    if warning:
        assert warning in log
        # A warning an operator cannot act on is noise: name the variable and
        # a value, and make that value buy a bucket rather than another zero.
        suggested = max(buckets) * dp * hidden // 2
        assert f"SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES={suggested}" in log
        assert f"{max(buckets)} x dp{dp} x {hidden}" in log


def test_the_margin_warning_says_when_it_cannot_read_a_threshold(monkeypatch):
    # No flag in LIBTPU_INIT_ARGS: libtpu's own default applies and we must not
    # guess it, but the payload it would be compared against is still worth
    # printing -- the operator is the only one who can close that gap.
    log = _boot_log(monkeypatch,
                    dp=8,
                    hidden=H_QWEN,
                    buckets=QWEN_BUCKETS,
                    threshold=None)

    assert "offload margin unknown" in log
    assert "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES=33554432" in log


def test_the_margin_warning_is_silent_for_the_packed_wire(monkeypatch):
    # A uint8 gather is synchronous at every size, so there is no margin to
    # lose and the advice would be wrong.
    log = _boot_log(monkeypatch,
                    dp=8,
                    hidden=H_QWEN,
                    buckets=QWEN_BUCKETS,
                    threshold=TPU7X_OFFLOAD_BYTES,
                    packed=True)

    assert "offload margin" not in log


def test_the_margin_check_falls_back_to_the_token_budget(monkeypatch):
    # compile_sizes is filled from the bucket ladder by the platform, but a
    # caller can pass its own; the largest bucket is the token budget either
    # way, and the check must not go quiet because one field is empty.
    log = _boot_log(monkeypatch,
                    dp=8,
                    hidden=H_QWEN,
                    buckets=[],
                    threshold=TPU7X_OFFLOAD_BYTES,
                    max_num_batched_tokens=2048)

    assert "thin offload margin" in log
    assert "2048 x dp8 x 4096 = 67108864 B" in log


# --- The companion-tensor channel we used to refuse --------------------------
#
# extra_tensors is how vLLM ships an activation scale beside the payload
# (prepare_finalize/naive_dp_ep.py), i.e. the contract the fp8 wire above
# re-implements privately. Nothing on the TPU path passes it -- every TPU quant
# method is monolithic, so the runner dispatches through us rather than through
# a prepare/finalize -- but refusing the channel we rebuilt is the wrong shape
# to leave in the tree.


@pytest.mark.parametrize("entry", ["dispatch_router_logits", "dispatch"])
def test_companion_tensors_are_gathered_beside_the_payload(monkeypatch, entry):
    comm, group = _communicator(True, monkeypatch)
    rows = 8
    hidden = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    routing = torch.randn(rows, 4, dtype=torch.bfloat16)
    scale = torch.rand(rows, 1, dtype=torch.float32)  # what upstream sends

    args = ((hidden, routing) if entry == "dispatch_router_logits" else
            (hidden, routing, torch.zeros(rows, 4, dtype=torch.int32)))
    result = getattr(comm, entry)(*args, extra_tensors=[scale])

    # One more slot than without extras, and the companion is gathered on the
    # token axis in its own dtype -- it is not put through the fp8 wire.
    assert len(result) == len(args) + 1
    gathered_scale, = result[-1]
    assert torch.equal(gathered_scale, torch.cat([scale] * DP))
    assert (torch.float32, (rows, 1)) in group.gathered


def test_companion_tensors_pass_straight_through_without_dp(monkeypatch):
    comm, group = _communicator(True, monkeypatch, world_size=1)
    hidden = torch.randn(8, HIDDEN, dtype=torch.bfloat16)
    logits = torch.randn(8, 4, dtype=torch.bfloat16)
    scale = torch.rand(8, 1, dtype=torch.float32)

    out = comm.dispatch_router_logits(hidden, logits, extra_tensors=[scale])

    assert out == (hidden, logits, [scale])
    assert group.gathered == []
