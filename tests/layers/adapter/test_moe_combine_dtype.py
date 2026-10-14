# SPDX-License-Identifier: Apache-2.0
"""Outer MoE collective precision preserves routed and shared output contracts."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from vllm.model_executor.layers.fused_moe.runner import moe_runner as upstream

from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as candidate


class Subject(candidate.VllmMoERunner):
    @property
    def do_naive_dispatch_combine(self):
        return self.naive

    @property
    def shared_experts(self):
        return self.shared


def runner(dtype=torch.float32, shared=None, naive=True, pcp=1):
    r = object.__new__(Subject)
    torch.nn.Module.__init__(r)
    r.combine_dtype = dtype
    r.naive = naive
    r.shared = shared
    r.moe_config = NS(
        pcp_size=pcp,
        is_sequence_parallel=False,
        moe_parallel_config=NS(use_all2all_kernels=False),
    )
    return r


class CombineDtypeTests(unittest.TestCase):
    def test_exact_old_candidate_collective_and_cast_order(self):
        for shared in [None, torch.tensor([[0.123]], dtype=torch.bfloat16)]:
            for dtype in [torch.bfloat16, torch.float32]:
                x = torch.tensor([[1.0], [0.00390625]] * 8, dtype=dtype)
                group = Mock(world_size=8)
                seen = []

                def reduce(value, dim, seen=seen):
                    seen.append(value.clone())
                    return value.reshape(8, 2, 1).sum(0)

                group.reduce_scatter.side_effect = reduce
                communicator = object.__new__(comm.TpuDeviceCommunicator)
                ep = NS(combine=communicator.combine)
                with (
                    patch.object(comm, "get_dp_group", return_value=group),
                    patch.object(upstream, "get_ep_group", return_value=ep),
                ):
                    out = runner(shared=shared)._maybe_combine(shared, x)
                routed = out[1] if shared is not None else out
                reference = reduce(x.float(), 0).to(dtype)
                self.assertTrue(torch.equal(routed, reference))
                self.assertEqual(routed.dtype, dtype)
                self.assertEqual(group.reduce_scatter.call_count, 1)
                self.assertEqual(seen[0].dtype, torch.float32)
                self.assertTrue(torch.equal(seen[0], x.float()))
                if shared is not None:
                    self.assertIs(out[0], shared)

    def test_default_preserves_dtype_and_identity(self):
        x = torch.ones(2, 2, dtype=torch.bfloat16)
        shared = torch.zeros_like(x)
        for s in [None, shared]:
            ep = NS(combine=Mock(return_value=x))
            with patch.object(upstream, "get_ep_group", return_value=ep):
                out = runner(dtype=None, shared=s)._maybe_combine(s, x)
            self.assertIs(out if s is None else out[1], x)
            self.assertIs(ep.combine.call_args.args[0], x)
            self.assertEqual(ep.combine.call_count, 1)

    def test_unsupported_collective_ownership_fails_before_communication(self):
        with patch.object(upstream, "get_ep_group") as group:
            for r, x in [
                (runner(naive=False), torch.ones(1)),
                (runner(pcp=2), torch.ones(1)),
                (runner(), object()),
            ]:
                with self.assertRaises(NotImplementedError):
                    r._maybe_combine(None, x)
            group.assert_not_called()
