"""Every torch op wrapper's fake implementation accepts exactly the operands
the real call passes. Dynamo traces the served forward against the fake
implementation, so a signature that drifts from the real one fails the whole
compile at startup. Runs on the CPU, no device needed."""
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import vllm_torchtpu.layers.vllm.fused_qk_norm_rope_tpu as qk
import vllm_torchtpu.layers.vllm.moe_lowtok_scores_op as scores_op


def test_fused_qk_norm_rope_gate_fake_matches_the_call():
    heads, kv_heads, dim, rot = 8, 1, 256, 128
    qk.prebuild(heads, kv_heads, dim, rot, 1e-6)
    with FakeTensorMode():
        q_gate = torch.empty(4, heads * dim * 2, dtype=torch.bfloat16)
        k = torch.empty(4, kv_heads * dim, dtype=torch.bfloat16)
        q_w = torch.empty(dim, dtype=torch.bfloat16)
        k_w = torch.empty(dim, dtype=torch.bfloat16)
        cache = torch.empty(9236, rot, dtype=torch.bfloat16)
        positions = torch.zeros(4, dtype=torch.int64)
        q, kk, gate = qk.fused_qk_rmsnorm_rope_gate(q_gate, k, q_w, k_w, cache,
                                                    positions, 1e-6, heads,
                                                    kv_heads, dim, rot)
    assert tuple(q.shape) == (4, heads * dim)
    assert tuple(kk.shape) == (4, kv_heads * dim)
    assert tuple(gate.shape) == (4, heads * dim)


def test_moe_lowtok_scores_fake_matches_the_call():
    topk, ranks, local, hidden, inter = 10, 8, 64, 4096, 1024
    scores_op.prebuild(topk, ranks, local)
    with FakeTensorMode():
        x = torch.empty(4, hidden, dtype=torch.bfloat16)
        scores = torch.empty(4, 512, dtype=torch.float32)
        tables = torch.zeros(1024, dtype=torch.int32)
        prim = torch.zeros(512, dtype=torch.int32)
        copy = torch.full((512, ), -1, dtype=torch.int32)
        hot1h = torch.zeros(512, 128, dtype=torch.bfloat16)
        w13 = torch.empty(local, hidden, 2 * inter, dtype=torch.float8_e4m3fn)
        s13 = torch.empty(local, 1, 1, 2 * inter)
        w2 = torch.empty(local, inter, hidden, dtype=torch.float8_e4m3fn)
        s2 = torch.empty(local, 1, 1, hidden)
        y = scores_op.moe_lowtok_scores_op(x,
                                           scores,
                                           tables,
                                           prim,
                                           copy,
                                           hot1h,
                                           w13,
                                           s13,
                                           w2,
                                           s2,
                                           topk=topk,
                                           ranks=ranks,
                                           owned=local)
    assert tuple(y.shape) == (4, hidden) and y.dtype == torch.bfloat16
