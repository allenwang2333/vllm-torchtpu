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
"""Load and trace GLM-5.2-FP8's MoE layer through the fused EP v2 bridge.

WHAT THIS IS FOR. Every gate on this path has so far been found by starting a
server: the operand-arity mismatch cost two ~50-minute starts on its own, and
behind it sat the whole-expert VMEM overage. This file asks the questions the
first minute of a server start asks -- does this layer ARM, does the op it
builds accept the operands the call site passes, does the program TRACE, and
does it COMPILE for the device -- with no model, no server and no checkpoint.

THE LAYER IS GLM-5.2-FP8'S MoE BLOCK: hidden 6144, `moe_intermediate_size`
2048, top-8, fp8 `e4m3` weights block-scaled [128, 128], `sigmoid` scoring
with a real per-global-expert `e_score_correction_bias` (`noaux_tc`)
(`knowledge/models/GLM-5.2-FP8/refs/model.md`, U-annyan-260909-01, -06, -08;
`config.json` @ GCS generation `1785565055944226`).

THE EP DEGREE DIFFERS FROM PRODUCTION, DELIBERATELY, and the same way
`tests/kernels/fused_moe/test_fused_ep_moe_v2_kblocked.py` differs: GLM-5.2
serves EP16 on a 2x2x2 slice, 256 global experts over 16 shards, `g_local`
16. The probe pod is one 2x2x1 node -- 8 cores -- so this runs **EP8 with 128
GLOBAL experts**, which is the same 16 local experts per core. `g_local`,
hidden and inter are what the weight buffers, the VMEM budget, the SMEM
bound and the operand shapes are functions of; only the number of shards a
routed row can travel between is halved.

WHAT IS AND IS NOT EXERCISED.

* `process_weights_after_loading` is the real fp8 one, including its
  dequantize/requantize of block-scaled checkpoint weights into the
  per-output-channel form the kernel requires, and the real arming sequence
  against a real eight-device mesh.
* The forward is traced, not executed. The torch op's operands are this
  RANK's shards and `pallas.jax_op` reconstitutes the global array from the
  mesh, which in a single-process test has no other ranks to reconstitute
  from. What the trace does still cover is exactly where both failures
  landed: a torch operator schema mismatch is raised by the argument parser
  before any device work, and Dynamo is the frame the server's `fullgraph`
  backbone compiles in.
* The program is COMPILED for the device from the op's own closure --
  `jax.jit(...).lower(...).compile()` over the same mesh and the same
  partition specs `_build_op` gave `shard_map`, at the operand shapes the
  call site passed. That is the XLA compile, so it is where a Mosaic VMEM
  overage refuses; it stops short of execution.

`_max_node_tokens` reads the served scheduler's cap, so it is supplied
rather than discovered: 1024 per engine (U-annyan-260909-17) times EP8.
"""

import contextlib
from types import SimpleNamespace

import jax
import pytest
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import ep_mesh
from vllm_torchtpu.kernels.fused_moe.v2 import host
from vllm_torchtpu.kernels.fused_moe.v2 import kernel as kernel_mod
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge

pytestmark = pytest.mark.multichip

EP = 8
GLOBAL_EXPERTS = 128
LOCAL_EXPERTS = GLOBAL_EXPERTS // EP  # 16, production's g_local
HIDDEN = 6144
INTER = 2048
TOPK = 8
CKPT_BLOCK = 128
# The served per-engine cap (U-annyan-260909-17) over this probe's EP width.
NODE_TOKENS = 1024 * EP
TOKENS = 512

# What `_build_op` closes over as the kernel's capacity, and therefore part
# of the built program's scope name.
CAPACITY = bridge._TILE_M
# The name the K-blocked residency -- the only one GLM's expert fits in VMEM
# (`1da98e252`) -- is built under. Assembled from the same constants the
# builder reads, so a retuned block size moves both together.
GLM_KERNEL_NAME = (f"fused_ep_moe_v2_kb{host.KBLOCK_HIDDEN}"
                   f"x{host.KBLOCK_INTER}_g{LOCAL_EXPERTS}_c{CAPACITY}"
                   f"_nb{host.NBUF}")


class _RecordingPallas:
    """A stand-in for `kernel.pl` that records every `pallas_call` name.

    The name is built inside `_build_fused_ep_moe_kernel.make_call` and never
    leaves it, so proxying the module the builder calls through is the only
    way to see it. Everything else is delegated untouched. Copied from
    `tests/kernels/fused_moe/test_fused_ep_moe_v2_kblocked.py`, which
    introduced it for the same purpose one layer lower down.
    """

    def __init__(self, real, names):
        self._real = real
        self.names = names

    def __getattr__(self, attr):
        return getattr(self._real, attr)

    def pallas_call(self, *args, **kwargs):
        self.names.append(kwargs.get("name"))
        return self._real.pallas_call(*args, **kwargs)


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip unless this host really has the eight-way slice.

    A fixture rather than a `skipif` for the reason the kernel suites give: a
    `skipif` is evaluated while pytest IMPORTS the module, and
    `jax.device_count()` builds the PJRT client to answer it.
    """
    if jax.device_count() < EP:
        pytest.skip(f"the fused EP MoE bridge needs an {EP}-way expert-"
                    f"parallel mesh; this host reports {jax.device_count()} "
                    "JAX devices")


def _glm_layer():
    """GLM-5.2-FP8's MoE block as the fp8 loader receives it.

    Checkpoint layout, not runtime layout: `[E, 2I, H]` and `[E, H, I]` with
    block-[128, 128] inverse scales, which is what
    `process_weights_after_loading` transposes and requantizes. Only this
    rank's 16 experts are present, which is what EP weight loading gives it.
    """
    layer = torch.nn.Module()
    gen = torch.Generator().manual_seed(11)

    def fp8(*shape):
        # A range e4m3 represents well enough that the requantize round trip
        # stays finite; the numerics are not what is gated here.
        return (torch.rand(shape, generator=gen) * 2.0 - 1.0).to(
            torch.float8_e4m3fn)

    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(fp8(LOCAL_EXPERTS, 2 * INTER, HIDDEN),
                           requires_grad=False))
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(fp8(LOCAL_EXPERTS, HIDDEN, INTER),
                           requires_grad=False))
    layer.register_parameter(
        "w13_weight_scale_inv",
        torch.nn.Parameter(torch.full(
            (LOCAL_EXPERTS, 2 * INTER // CKPT_BLOCK, HIDDEN // CKPT_BLOCK),
            0.5),
                           requires_grad=False))
    layer.register_parameter(
        "w2_weight_scale_inv",
        torch.nn.Parameter(torch.full(
            (LOCAL_EXPERTS, HIDDEN // CKPT_BLOCK, INTER // CKPT_BLOCK), 0.5),
                           requires_grad=False))
    # `noaux_tc`: a sigmoid score with a per-global-expert correction added
    # for SELECTION only. This is the field that decides the op's arity.
    layer.register_parameter(
        "e_score_correction_bias",
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, GLOBAL_EXPERTS),
                           requires_grad=False))
    layer.scoring_func = "sigmoid"
    # Unpinned for this checkpoint (no ref in knowledge/) and inert for
    # everything this file gates: it is closed into the op, not read by any
    # admission check.
    layer.routed_scaling_factor = 1.0
    layer.custom_routing_function = None
    layer.use_grouped_topk = False
    layer.num_expert_group = None
    layer.topk_group = None
    layer.global_num_experts = GLOBAL_EXPERTS
    layer.renormalize = True
    layer.activation = "silu"
    layer.moe_config = SimpleNamespace(
        has_bias=False,
        is_act_and_mul=True,
        experts_per_token=TOPK,
        intermediate_size_per_partition=INTER,
        intermediate_size_per_partition_unpadded=INTER,
        tp_size=1,
        tp_rank=0,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        moe_parallel_config=SimpleNamespace(use_ep=True, pcp_size=1),
    )
    return layer


class _OpProxy:
    """The built op, with the operands the call site passed recorded.

    Installed in place of the op on the quant method, so `fused_moe_ep`
    reaches the REAL operator -- schema check included -- and this file can
    still see what it was handed.
    """

    def __init__(self, op):
        self.op = op
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.op(*args, **kwargs)


@pytest.fixture(scope="module")
def loaded():
    """One real fp8 weight load against a real eight-device EP mesh.

    Module-scoped: the load requantizes ~600 MB of expert weights and the
    build is a Mosaic lowering, and `_OPS` memoizes the op anyway, so a
    second load would re-answer nothing.
    """
    from vllm_torchtpu.layers.adapter import fused_moe_rs
    from vllm_torchtpu.layers.adapter.quantization import fp8

    layer = _glm_layer()
    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method.moe = layer.moe_config
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_block_size = [CKPT_BLOCK, CKPT_BLOCK]
    method.block_quant = True
    method.weight_scale_name = "weight_scale_inv"

    device_ids = tuple(int(d.id) for d in jax.devices()[:EP])
    log = []
    built = []
    real_sharded_jax_op = bridge.sharded_jax_op

    def recording_sharded_jax_op(name, fn, **kwargs):
        """The real registration, with the closure kept for the compile test.

        Reading the function and the specs off the build is what keeps the
        compile below on the op's own closure rather than on a second copy
        of `_build_op`'s argument list written out here.
        """
        built.append(SimpleNamespace(name=name, fn=fn, **kwargs))
        return real_sharded_jax_op(name, fn, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        # A real mesh over the real slice; only the EP group's collective --
        # which needs a torch.distributed group this test has no use for --
        # is replaced.
        mp.setattr(ep_mesh, "_MESH_CACHE", {})
        mp.setattr(ep_mesh, "ep_device_ids", lambda: device_ids)
        mp.setattr(bridge, "ep_mesh_index", lambda: 0)
        mp.setattr(bridge, "ep_rank_order", lambda: None)
        mp.setattr(bridge, "_max_node_tokens", lambda *a: NODE_TOKENS)
        mp.setattr(bridge, "_OPS", {})
        mp.setattr(bridge, "sharded_jax_op", recording_sharded_jax_op)
        # The refusal path logs and returns None; capture the reason so a
        # layer that does not arm says WHY rather than just failing.
        mp.setattr(bridge.logger, "warning_once",
                   lambda msg, *a: log.append(msg % a if a else msg))
        mp.setattr(bridge.logger, "info_once",
                   lambda msg, *a: log.append(msg % a if a else msg))
        mp.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
        mp.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024)
        mp.setattr(envs, "MOE_FUSED_EP_V2_SHARDED_PLAN", False)
        # Everything the GMM fallback registers wants a distributed group or
        # a served config; none of it is on the path under test.
        mp.setattr(fp8, "RoutedExperts", torch.nn.Module)
        mp.setattr(fp8, "prebuild_fused_moe_kernel", lambda **kw: None)
        mp.setattr(fp8.moe_routing, "register_experts_start_buffer",
                   lambda *a, **kw: None)
        mp.setattr(fp8.moe_routing, "validate_linear_ep_placement",
                   lambda *a: None)
        mp.setattr(fused_moe_rs, "prebuild_fused_moe_rs",
                   lambda *a, **kw: None)
        method.process_weights_after_loading(layer)

    op = getattr(method, bridge.FUSED_MOE_EP_OP_ATTR, None)
    proxy = _OpProxy(op) if op is not None else None
    if proxy is not None:
        setattr(method, bridge.FUSED_MOE_EP_OP_ATTR, proxy)
    return SimpleNamespace(layer=layer,
                           method=method,
                           log=log,
                           built=built[0] if built else None,
                           proxy=proxy)


def test_the_glm_layer_arms_the_fused_ep_path(loaded):
    """Gate 1: does this shape arm at all, and if not, which refusal."""
    assert loaded.proxy is not None, (
        "GLM-5.2-FP8's MoE layer did not arm the fused EP v2 path at EP8; "
        "prebuild's log said: " + " | ".join(loaded.log))
    assert bridge.fused_moe_ep_supported(loaded.method)


def test_the_weights_reached_the_form_the_kernel_takes(loaded):
    """Gate 2: the real requantization produced the kernel's layout.

    Block-[128, 128] on disk, per-output-channel in the op. Read off the
    tensors the loader installed, because `prebuild` admitted the layer on
    exactly these and the forward hands them over unchanged.
    """
    layer = loaded.layer
    assert layer.w13_weight.shape == (LOCAL_EXPERTS, HIDDEN, 2 * INTER)
    assert layer.w2_weight.shape == (LOCAL_EXPERTS, INTER, HIDDEN)
    assert layer.w13_weight.dtype is torch.float8_e4m3fn
    assert layer.w2_weight.dtype is torch.float8_e4m3fn
    assert bridge._squeeze_channel_scale(layer.w13_weight_scale_inv,
                                         LOCAL_EXPERTS, 2 * INTER) is not None
    assert bridge._squeeze_channel_scale(layer.w2_weight_scale_inv,
                                         LOCAL_EXPERTS, HIDDEN) is not None


def test_the_routing_bias_is_staged_for_the_call_site(loaded):
    """Gate 3: the bug that cost two server starts, at GLM's own shapes."""
    staged = bridge.score_bias_operand(loaded.layer)
    assert staged is not None, (
        "the op is built with the score_bias operand and the call site has "
        f"nothing to pass: {bridge.FUSED_MOE_EP_SCORE_BIAS_ATTR} is unset")
    assert staged.shape == (GLOBAL_EXPERTS, )
    assert staged.dtype is torch.float32


def _inputs():
    return (torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16),
            torch.ones(TOKENS, GLOBAL_EXPERTS, dtype=torch.bfloat16))


def test_the_forward_reaches_the_operator_with_the_operands_it_takes(loaded):
    """Gate 4: the call the server makes, taken as far as no model allows.

    Under `FakeTensorMode` the operator's argument parser still checks the
    call against the schema, which is what rejected the seven-operand call
    with `pallas::fused_moe_ep() is missing value for argument 'score_bias'`.
    Device execution is what a single process cannot do: the op is handed
    this rank's shards and there are no other ranks here to supply theirs.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    x, router_logits = _inputs()
    with FakeTensorMode(allow_non_fake_inputs=True):
        out = loaded.method.apply_monolithic(loaded.layer, x, router_logits)
    assert tuple(out.shape) == (TOKENS, HIDDEN)
    assert loaded.proxy.calls, "the forward did not reach the built op"
    args, kwargs = loaded.proxy.calls[-1]
    assert not kwargs
    assert len(args) == len(loaded.built.input_partition_specs), (
        f"the op was built for {len(loaded.built.input_partition_specs)} "
        f"operands and the call site passed {len(args)}")
    # Not just "the counts match": the last operand has to be the staged
    # bias itself, which is what makes the count eight rather than seven.
    assert args[-1] is bridge.score_bias_operand(loaded.layer)
    print("\nGLM-5.2 fused EP operands:",
          [(tuple(a.shape), str(a.dtype)) for a in args])


_JAX_DTYPE = {
    torch.float8_e4m3fn: jax.numpy.float8_e4m3fn,
    torch.bfloat16: jax.numpy.bfloat16,
    torch.float32: jax.numpy.float32,
    torch.int32: jax.numpy.int32,
}


def _global_aval(tensor, spec, mesh):
    """This rank's shard, described as the global array the op declares.

    `sharded_jax_op` documents the convention: the torch operands are one
    rank's shards and the global shape is reconstituted from the mesh. Only
    axis 0 is ever partitioned here, so that is the only one scaled.
    """
    shape = tuple(tensor.shape)
    if spec and spec[0] is not None:
        shape = (shape[0] * mesh.shape[bridge.EP_AXIS_NAME], ) + shape[1:]
    dtype = _JAX_DTYPE.get(tensor.dtype)
    assert dtype is not None, f"unmapped operand dtype {tensor.dtype}"
    return jax.ShapeDtypeStruct(shape,
                                dtype,
                                sharding=NamedSharding(mesh, spec))


def test_the_program_compiles_for_the_device_at_glm_shapes(loaded):
    """Gate 5: the XLA + Mosaic compile, which is where VMEM refuses.

    Runs the op's own closure -- the function `_build_op` registered, its
    mesh and its partition specs -- at the operand shapes the call site
    passed, and compiles it. No execution, so no other ranks are needed;
    everything a compile decides, including the kernel's weight residency
    and its VMEM budget, is decided here.
    """
    assert loaded.proxy.calls, (
        "this test reuses the operands recorded by the forward above; run "
        "the whole module")
    args, _ = loaded.proxy.calls[-1]
    built = loaded.built
    mesh = built.mesh
    avals = [
        _global_aval(t, spec, mesh)
        for t, spec in zip(args, built.input_partition_specs)
    ]
    out_sharding = NamedSharding(mesh, P(bridge.EP_AXIS_NAME))
    # One recorder around BOTH traces: `kernel._BUILD_CACHE` memoizes, so
    # whichever of the two traces first is the only one that reaches
    # `make_call`, and a recorder around the second alone would see nothing.
    names = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kernel_mod, "pl", _RecordingPallas(kernel_mod.pl, names))
        # The program returns the rows it was given: the global array is
        # every rank's tokens, and each rank gets its own back.
        out = jax.eval_shape(built.fn, *avals)
        assert tuple(
            out.shape) == (TOKENS * EP,
                           HIDDEN), (f"global output shape {out.shape}")
        compiled = jax.jit(built.fn,
                           out_shardings=out_sharding).lower(*avals).compile()
    assert compiled is not None

    # WHICH kernel compiled, not merely that one did. The scope name is
    # emitted inside `_build_fused_ep_moe_kernel.make_call`, once per build,
    # and is the only artifact that distinguishes the residencies -- so this
    # is also the proof that the trace above was not served from
    # `kernel._BUILD_CACHE`, which would make a green compile say nothing.
    ffn = [n for n in names if n and n.startswith("fused_ep_moe_v2")]
    assert ffn, ("no fused_ep_moe_v2 pallas_call was traced, so the compile "
                 "reused a cached program and proves nothing")
    assert set(ffn) == {
        GLM_KERNEL_NAME
    }, (f"the adapter's GLM build(s) were named {sorted(set(ffn))}, expected "
        f"[{GLM_KERNEL_NAME!r}]; a name without the "
        f"_kb{host.KBLOCK_HIDDEN}x{host.KBLOCK_INTER} tag is the whole-expert "
        "residency, which is the one that did not fit")
    print("\nGLM-5.2 fused EP kernel built through the adapter:",
          sorted(set(ffn)))
    with contextlib.suppress(Exception):
        print("compiled memory analysis:", compiled.memory_analysis())


class _TracedFarEnough(Exception):
    """Raised by the test backend once Dynamo has produced a whole graph."""


def _caused_by(exc, kind):
    while exc is not None:
        if isinstance(exc, kind):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def test_the_forward_traces_under_dynamo(loaded):
    """Gate 6: the same call inside a compiled region, where it failed.

    vLLM compiles the backbone with `fullgraph=True`, and both failures this
    file exists for surfaced as Dynamo errors rather than as exceptions from
    the kernel. The backend refuses to run the graph it is given, because
    executing it is the part a single process cannot do; reaching the
    backend at all with a whole graph is the assertion.
    """
    import torch._dynamo as dynamo

    dynamo.reset()
    traced = []

    def backend(gm, example_inputs):
        traced.append(gm)
        raise _TracedFarEnough()

    def forward(x, router_logits):
        return loaded.method.apply_monolithic(loaded.layer, x, router_logits)

    compiled = torch.compile(forward, fullgraph=True, backend=backend)
    x, router_logits = _inputs()
    error = None
    try:
        compiled(x, router_logits)
    except Exception as exc:  # noqa: BLE001 - the backend always raises
        error = exc
    assert traced, (
        "Dynamo did not reach the backend with a whole graph for the fused "
        f"EP forward; it raised {error!r}")
    assert _caused_by(error, _TracedFarEnough), (
        f"the graph was traced but the compile then failed: {error!r}")
    ops = {
        str(node.target)
        for node in traced[0].graph.nodes if node.op == "call_function"
    }
    assert any("fused_moe_ep" in op for op in ops), (
        f"the compiled graph does not contain the fused EP op; it has {ops}")
