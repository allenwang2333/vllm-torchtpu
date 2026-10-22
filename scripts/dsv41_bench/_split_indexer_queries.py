# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark independent query chunks under one shared short-context guard."""

from unittest import mock

import jax
import jax.numpy as jnp


def split_score_and_select(original, chunks):
    condition = jax.lax.cond

    def sparse_condition(pred, true_fun, false_fun, *operands, **kwargs):
        # Hoist only the model's identical whole-batch guard. Kernel DMA and
        # arithmetic conditions still use the original control flow.
        name = getattr(false_fun, "__qualname__", "")
        if name == "_score_and_select.<locals>.body.<locals>.sparse":
            return false_fun(*operands)
        return condition(pred, true_fun, false_fun, *operands, **kwargs)

    def split(q, banks, records, weights, lengths, candidates, requests, *settings):
        batch_max, _, topk, block_size, write, filtering, _ = settings
        if (
            write
            or not filtering
            or candidates.shape[1] * block_size >= records.shape[1]
        ):
            return original(
                q, banks, records, weights, lengths, candidates, requests, *settings
            )
        rows = q.shape[0] // chunks
        assert q.shape[0] % chunks == 0 and rows % 6 == 0

        def full():
            return original(
                q, banks, records, weights, lengths, candidates, requests, *settings
            )

        def sparse():
            results = []
            with mock.patch.object(jax.lax, "cond", sparse_condition):
                for start in range(0, q.shape[0], rows):
                    sl = slice(start, start + rows)
                    results.append(
                        original(
                            q[sl],
                            banks,
                            records,
                            weights[sl],
                            lengths[sl],
                            candidates[sl],
                            requests[sl],
                            *settings,
                        )
                    )
            return tuple(jnp.concatenate(parts, axis=0) for parts in zip(*results))

        return condition(
            jnp.minimum(batch_max.max(), records.shape[1]) <= topk, full, sparse
        )

    return split
