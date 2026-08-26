# SPDX-License-Identifier: Apache-2.0
"""Pool-tag vocabulary shared with the Raiden controller.

These strings are wire-visible: they name pools in manifests and byte-span
registrations on both sides of a disagg pair, and the live-pair acceptance
gate builds its expected tag list from them. The values are frozen; changing
one is a coupled two-repo protocol change, not a refactor.
"""
TAG_FA = "fa"
TAG_GDN_CONV = "gdn.conv"
TAG_GDN_SSM = "gdn.ssm"
TAG_DSA_IDX = "dsa.idx"

# DeepSeek-V4. Every DSv4 tag carries a ``.g{ordinal}`` suffix naming its
# kv-cache-group, because DSv4 is the first admitted model whose logical caches
# alias one array at the same byte offset and are told apart only by block ID.
# A tag that did not name the group would address the wrong caches' pages.
TAG_DSV4_CSA_NOPE = "dsv4.csa.nope"
TAG_DSV4_CSA_ROPE = "dsv4.csa.rope"
TAG_DSV4_IDX = "dsv4.idx"
TAG_DSV4_HCA = "dsv4.hca"
TAG_DSV4_SWA = "dsv4.swa"
# The compressor's paged staging buffer, one tag per compressed-KV role since
# their windows (and so their per-request block counts) differ.
TAG_DSV4_STATE_CSA = "dsv4.state.csa"
TAG_DSV4_STATE_IDX = "dsv4.state.idx"
TAG_DSV4_STATE_HCA = "dsv4.state.hca"


def dsv4_group_tag(base_tag: str, group_ordinal: int) -> str:
    """``dsv4.swa`` + group 3 -> ``dsv4.swa.g3``."""
    return f"{base_tag}.g{int(group_ordinal)}"
