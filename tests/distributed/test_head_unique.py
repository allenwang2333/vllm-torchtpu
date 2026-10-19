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
"""Head-unique Stage-3 transfer: owner rule, layout admission, gather rows."""

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden.head_unique import (
    GATHER_BLOCK_BUCKETS,
    gather_bucket,
    gather_passes,
    gather_plan,
    head_unique_group_size,
    load_group_size,
    owned_page_indices,
    padded_copy_rows,
    padded_copy_rows_to,
    pages_per_replica,
)


class TestGroupSize:
    def test_qwen35_397b_tp8_has_groups_of_four(self):
        assert (
            head_unique_group_size(
                tp_size=8, total_num_kv_heads=2, dcp_size=1, dst_shards=8
            )
            == 4
        )

    def test_qwen35_35b_tp8_has_groups_of_two(self):
        assert (
            head_unique_group_size(
                tp_size=8, total_num_kv_heads=4, dcp_size=1, dst_shards=8
            )
            == 2
        )

    def test_refuses_dcp(self):
        with pytest.raises(ValueError, match="decode_context_parallel_size=1"):
            head_unique_group_size(
                tp_size=8, total_num_kv_heads=2, dcp_size=4, dst_shards=8
            )

    def test_refuses_dst_shards_other_than_tp(self):
        with pytest.raises(ValueError, match="TPU_RAIDEN_DST_SHARDS"):
            head_unique_group_size(
                tp_size=8, total_num_kv_heads=2, dcp_size=1, dst_shards=4
            )

    def test_refuses_a_layout_without_a_replicated_head(self):
        with pytest.raises(ValueError, match="replicated across TP ranks"):
            head_unique_group_size(
                tp_size=4, total_num_kv_heads=4, dcp_size=1, dst_shards=4
            )

    def test_refuses_heads_that_do_not_divide_tp(self):
        with pytest.raises(ValueError, match="divide the TP size"):
            head_unique_group_size(
                tp_size=8, total_num_kv_heads=3, dcp_size=1, dst_shards=8
            )


class TestOwnerRule:
    def test_short_loads_stay_replicated(self):
        assert load_group_size(3, 4) == 1
        assert load_group_size(4, 4) == 4
        assert load_group_size(150, 4) == 4
        assert load_group_size(150, 1) == 1

    def test_pages_are_dealt_round_robin(self):
        assert owned_page_indices(10, 4, 0) == [0, 4, 8]
        assert owned_page_indices(10, 4, 1) == [1, 5, 9]
        assert owned_page_indices(10, 4, 2) == [2, 6]
        assert owned_page_indices(10, 4, 3) == [3, 7]
        assert sorted(
            page for r in range(4) for page in owned_page_indices(10, 4, r)
        ) == list(range(10))

    def test_replica_share_is_the_largest_and_bucketed(self):
        assert pages_per_replica(10, 4) == 3
        assert pages_per_replica(8, 4) == 2
        assert pages_per_replica(10, 4, block_bucket=8) == 8
        assert pages_per_replica(150, 4, block_bucket=8) == 40


class TestGatherPlan:
    def test_every_page_lands_on_every_replica_once(self):
        block_ids = [30, 10, 20, 40, 50, 60, 70]
        kps = 5
        plans = {
            replica: gather_plan(
                block_ids=block_ids,
                group_size=4,
                replica=replica,
                kernel_pages_per_block=kps,
                block_row_stride=kps,
            )
            for replica in range(4)
        }
        pool_rows = {
            page: list(range(block * kps, block * kps + kps))
            for page, block in enumerate(block_ids)
        }
        for replica, plan in plans.items():
            own = owned_page_indices(len(block_ids), 4, replica)
            # Pack reads exactly the own pages, in page order, into buffer
            # blocks 0, 1, ...
            assert list(plan.pack_src_rows) == [
                row for page in own for row in pool_rows[page]
            ]
            assert list(plan.pack_dst_rows) == list(range(len(own) * kps))
            # Unpack writes every page this replica does not own, once.
            written = list(plan.unpack_dst_rows)
            expected = sorted(
                row
                for page in range(len(block_ids))
                if page not in own
                for row in pool_rows[page]
            )
            assert sorted(written) == expected
            assert len(set(written)) == len(written)
            assert plan.buffer_rows == pages_per_replica(len(block_ids), 4) * kps

    def test_unpack_reads_the_owner_slot_of_the_gathered_buffer(self):
        block_ids = [3, 4, 5, 6, 7]
        plan = gather_plan(
            block_ids=block_ids,
            group_size=2,
            replica=0,
            kernel_pages_per_block=2,
            block_row_stride=2,
        )
        # Replica 1 owns pages 1 and 3 (blocks 4 and 6); its send buffer is
        # the second half of the gathered buffer (3 blocks of 2 rows each).
        per_replica_rows = pages_per_replica(5, 2) * 2
        assert plan.unpack_src_rows == tuple(per_replica_rows + r for r in range(4))
        assert plan.unpack_dst_rows == (8, 9, 12, 13)

    def test_base_row_and_stride_address_the_storage(self):
        plan = gather_plan(
            block_ids=[2, 3],
            group_size=2,
            replica=1,
            kernel_pages_per_block=1,
            block_row_stride=4,
            base_row=100,
        )
        assert plan.pack_src_rows == (100 + 3 * 4,)
        assert plan.unpack_dst_rows == (100 + 2 * 4,)

    def test_bucket_fixes_the_buffer_and_pads_the_copies(self):
        plan = gather_plan(
            block_ids=list(range(10)),
            group_size=4,
            replica=2,
            kernel_pages_per_block=5,
            block_row_stride=5,
            buffer_blocks=8,
        )
        assert plan.buffer_rows == 8 * 5
        # Pack copies are padded to the bucket's rows with repeats of the
        # last pair; the real copies come first.
        assert len(plan.pack_src_rows) == 8 * 5
        assert plan.pack_src_rows[: 2 * 5] == tuple(
            r for b in (2, 6) for r in range(b * 5, b * 5 + 5)
        )
        assert plan.pack_src_rows[2 * 5 :] == (plan.pack_src_rows[2 * 5 - 1],) * 30
        assert len(plan.unpack_src_rows) == 3 * 8 * 5
        # Other replicas' slots start at their bucketed offset.
        assert plan.unpack_src_rows[0] == 0
        assert plan.unpack_src_rows[3 * 5] == 8 * 5

    def test_bucket_smaller_than_the_share_is_refused(self):
        with pytest.raises(ValueError, match="smaller than the replica share"):
            gather_plan(
                block_ids=list(range(40)),
                group_size=4,
                replica=0,
                kernel_pages_per_block=5,
                block_row_stride=5,
                buffer_blocks=8,
            )

    @pytest.mark.parametrize("pages", [1, 2, 3])
    def test_a_short_pass_pads_the_replicas_without_a_page(self, pages):
        # A tail pass after a 192-page pass: replicas >= pages own nothing.
        block_ids = list(range(500, 500 + pages))
        for replica in range(4):
            plan = gather_plan(
                block_ids=block_ids,
                group_size=4,
                replica=replica,
                kernel_pages_per_block=5,
                block_row_stride=5,
                buffer_blocks=8,
            )
            assert len(plan.pack_src_rows) == 8 * 5
            assert len(plan.unpack_src_rows) == 3 * 8 * 5
            if replica < pages:
                assert plan.pack_src_rows[0] == block_ids[replica] * 5
            else:
                # Owns nothing: the send buffer is padded from a read of page 0.
                assert plan.pack_src_rows == (block_ids[0] * 5,) * 40
                assert plan.pack_dst_rows == (0,) * 40
            # Every write lands on a row of a page this replica does not own,
            # or on its own page with its own bytes.
            own_rows = {
                block_ids[p] * 5 + k for p in range(replica, pages, 4) for k in range(5)
            }
            for src, dst in zip(plan.unpack_src_rows, plan.unpack_dst_rows):
                if dst in own_rows:
                    assert src == replica * 8 * 5 + (dst - block_ids[replica] * 5)
                    assert pages == 1
        assert gather_passes(list(range(192 + pages)), 4)[1] == list(
            range(192, 192 + pages)
        )

    def test_a_one_page_pass_self_copies_on_its_owner(self):
        plan = gather_plan(
            block_ids=[9],
            group_size=4,
            replica=0,
            kernel_pages_per_block=5,
            block_row_stride=5,
            buffer_blocks=8,
        )
        # Nobody else owns a page: the unpack is a self-copy of page 0's row.
        assert plan.unpack_src_rows == (0,) * 3 * 40
        assert plan.unpack_dst_rows == (9 * 5,) * 3 * 40
        assert plan.pack_src_rows[:5] == tuple(range(45, 50))

    def test_refuses_a_group_of_one(self):
        with pytest.raises(ValueError):
            gather_plan(
                block_ids=[1],
                group_size=1,
                replica=0,
                kernel_pages_per_block=1,
                block_row_stride=1,
            )


class TestBuckets:
    def test_bucket_is_the_smallest_that_holds_the_share(self):
        assert gather_bucket(4, 4) == 8
        assert gather_bucket(32, 4) == 8
        assert gather_bucket(33, 4) == 16
        assert gather_bucket(150, 4) == 48
        assert gather_bucket(192, 4) == 48
        with pytest.raises(ValueError, match="exceed the largest gather pass"):
            gather_bucket(193, 4)

    def test_passes_split_at_a_multiple_of_the_group(self):
        span = GATHER_BLOCK_BUCKETS[-1] * 4
        blocks = list(range(1000, 1000 + span + 10))
        passes = gather_passes(blocks, 4)
        assert [len(p) for p in passes] == [span, 10]
        assert passes[0] + passes[1] == blocks
        assert span % 4 == 0
        assert gather_passes(blocks[:150], 4) == [blocks[:150]]

    def test_owner_rule_survives_a_pass_boundary(self):
        # Page p of the whole load and page p' of its pass share p mod G.
        span = GATHER_BLOCK_BUCKETS[-1] * 4
        blocks = list(range(span + 20))
        for offset, pass_blocks in zip(
            range(0, len(blocks), span), gather_passes(blocks, 4)
        ):
            for local, block in enumerate(pass_blocks):
                assert (offset + local) % 4 == local % 4
                assert block == offset + local


class TestPaddedCopyRows:
    def test_pads_to_an_exact_length(self):
        assert padded_copy_rows_to((1, 2), (5, 6), 4) == ((1, 2, 2, 2), (5, 6, 6, 6))
        assert padded_copy_rows_to((1,), (5,), 1) == ((1,), (5,))
        with pytest.raises(ValueError, match="exceed the bucket"):
            padded_copy_rows_to((1, 2, 3), (4, 5, 6), 2)
        with pytest.raises(ValueError, match="empty"):
            padded_copy_rows_to((), (), 2)

    def test_pads_by_repeating_the_last_pair(self):
        src, dst = padded_copy_rows((1, 2, 3), (7, 8, 9), 4)
        assert src == (1, 2, 3, 3)
        assert dst == (7, 8, 9, 9)

    def test_exact_multiples_and_empty_lists_are_unchanged(self):
        assert padded_copy_rows((1, 2), (3, 4), 2) == ((1, 2), (3, 4))
        assert padded_copy_rows((), (), 8) == ((), ())

    def test_refuses_mismatched_lists(self):
        with pytest.raises(ValueError):
            padded_copy_rows((1,), (1, 2), 2)
