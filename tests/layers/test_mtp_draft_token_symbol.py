"""The MTP draft forward traced by Dynamo on the CPU, with no chips: the real
Qwen3.5 predictor forward, with and without the plugin's token-count patch, on
a stub predictor whose decoder layer forks on the row count the way the real
head's MoE block does -- one side of the fork meets positions' length in a
shape-checked op, the other never does.

It checks what compilation/shape_variants needs to hand a token bucket from one
draft trace to the next: the trace groups lift the same SymInt inputs.
"""
import inspect
import types

import pytest
import torch

pytestmark = pytest.mark.cpu_test

HIDDEN, ROWS_SMALL, ROWS_LARGE, FORK_ROWS = 16, 4, 48, 33

# One pipeline stage. vLLM's real group needs the platform's device type,
# which a host with no chips does not register.
_PP_GROUP = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)


@pytest.fixture
def forwards(monkeypatch):
    """(unpatched, patched) Qwen3.5 predictor forwards; the classes are
    restored afterwards."""
    import vllm_torchtpu
    from vllm.distributed import parallel_state
    from vllm.model_executor.models import qwen3_5_mtp, qwen3_next_mtp
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: _PP_GROUP)
    for module in (qwen3_5_mtp, qwen3_next_mtp):
        monkeypatch.setattr(module, "get_pp_group", lambda: _PP_GROUP)
    for cls in (qwen3_5_mtp.Qwen3_5MultiTokenPredictor,
                qwen3_next_mtp.Qwen3NextMultiTokenPredictor):
        monkeypatch.setattr(cls, "forward", inspect.unwrap(cls.forward))
        monkeypatch.setattr(cls,
                            "_tpu_draft_token_count_patch",
                            False,
                            raising=False)
    cls = qwen3_5_mtp.Qwen3_5MultiTokenPredictor
    original = cls.forward
    vllm_torchtpu._patch_mtp_draft_token_count_symbol()
    return original, cls.forward


class _Norm(torch.nn.Module):

    def forward(self, x, residual=None):
        if residual is None:
            return x * 0.5
        y = x + residual
        return y * 0.5, y


class _Layer(torch.nn.Module):
    use_attn_reduce_scatter_for_moe = False

    def forward(self, positions, hidden_states, residual):
        residual = hidden_states
        rows = hidden_states.shape[0]
        # every path reads positions, as the layer's attention does
        hidden_states = hidden_states + positions[-1].sum().to(
            hidden_states.dtype) * 0
        if rows >= FORK_ROWS:
            # one side of the row-count fork broadcasts positions' length
            # against the rows, which makes Dynamo equate the two sizes
            hidden_states = hidden_states + positions[-1].to(
                hidden_states.dtype)[:, None] * 0
        # a size the compiled graph keeps as an input
        hidden_states = hidden_states.reshape(rows, 2, -1).reshape(rows, -1)
        return hidden_states, residual


def _predictor():
    """What Qwen3_5MultiTokenPredictor.forward reads off its module."""
    stub = torch.nn.Module()
    stub.register_buffer("emb", torch.randn(64, HIDDEN))
    stub.embed_input_ids = types.MethodType(lambda self, ids: self.emb[ids],
                                            stub)
    stub.pre_fc_norm_embedding = _Norm()
    stub.pre_fc_norm_hidden = _Norm()
    stub.fc = torch.nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
    stub.num_mtp_layers = 1
    stub.layers = torch.nn.ModuleList([_Layer()])
    stub.norm = _Norm()
    stub.config = types.SimpleNamespace(hidden_size=HIDDEN)
    return stub


def _lifted_symints(forward, rows):
    """The SymInt inputs of the graph Dynamo captures for one trace, compiled
    as vLLM does: guards dropped, and each argument's token dim marked dynamic
    on its own (dynamic_arg_dims; positions mrope-shaped)."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    stub = _predictor()
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda i, p, h: forward(stub, i, p, h),
        fullgraph=True,
        dynamic=False,
        backend=backend,
        options={"guard_filter_fn": lambda guards: [False] * len(guards)})
    input_ids = torch.zeros(rows, dtype=torch.int64)
    positions = torch.zeros(3, rows, dtype=torch.int64)
    hidden_states = torch.zeros(rows, HIDDEN)
    torch._dynamo.mark_dynamic(input_ids, 0)
    torch._dynamo.mark_dynamic(positions, 1)
    torch._dynamo.mark_dynamic(hidden_states, 0)
    compiled(input_ids, positions, hidden_states)
    names = []
    for node in graphs[-1].graph.nodes:
        if node.op != "placeholder":
            break
        value = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(value, torch.SymInt):
            names.append(node.name)
    return sorted(names)


def test_draft_trace_groups_lift_the_same_token_symbols(forwards):
    unpatched, patched = forwards
    # the failure the patch removes: the side of the fork that equates the
    # sizes drops one symbol, so the traces cannot hand buckets over
    small = _lifted_symints(unpatched, ROWS_SMALL)
    large = _lifted_symints(unpatched, ROWS_LARGE)
    assert small != large, "the stub no longer reproduces the fork"
    small = _lifted_symints(patched, ROWS_SMALL)
    large = _lifted_symints(patched, ROWS_LARGE)
    assert small == large, (
        f"lifted SymInt inputs differ between the draft trace groups: "
        f"{set(small) ^ set(large)}")


def test_disagreeing_token_counts_raise(forwards):
    _, patched = forwards
    with pytest.raises(ValueError, match="disagree on the token count"):
        patched(_predictor(), torch.zeros(4, dtype=torch.int64),
                torch.zeros(3, 5, dtype=torch.int64), torch.zeros(4, HIDDEN))


def test_patch_wraps_once_and_keeps_the_signature(forwards):
    import vllm_torchtpu
    from vllm.model_executor.models import qwen3_5_mtp
    unpatched, patched = forwards
    vllm_torchtpu._patch_mtp_draft_token_count_symbol()
    cls = qwen3_5_mtp.Qwen3_5MultiTokenPredictor
    assert cls.forward is patched
    assert inspect.unwrap(cls.forward) is unpatched
    assert inspect.signature(cls.forward) == inspect.signature(unpatched)
