# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import \
    promote_tpu_gdn_state_dtypes


def test_conv_state_promotes_to_fp32():
    """The page-size derivation and the runtime GDN op must agree on the
    conv state dtype; an unpromoted bf16 conv in the derivation under-
    budgets the pooled page and asserts at startup (page_size_padded)."""
    conv, temporal = promote_tpu_gdn_state_dtypes(torch.bfloat16,
                                                  torch.float32)
    assert conv == torch.float32
    assert temporal == torch.float32
    conv, temporal = promote_tpu_gdn_state_dtypes(torch.float32,
                                                  torch.bfloat16)
    assert conv == torch.float32
    assert temporal == torch.bfloat16
