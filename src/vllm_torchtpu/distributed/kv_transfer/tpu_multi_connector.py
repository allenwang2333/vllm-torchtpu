"""TPU Multi Connector wrapper for combining disagg and offloading."""

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector


class TPUMultiConnector(MultiConnector):
    """MultiConnector combining TPURaidenConnector (disaggregation) with
    TPURaidenOffloadingConnector (offloading).

    Children must be listed explicitly in
    ``kv_connector_extra_config["connectors"]``. No defaults are synthesized:
    upstream classmethods (``get_required_kvcache_layout``,
    ``requires_piecewise_for_cudagraph``) parse the original config before
    the connector is instantiated, so children injected at init time would
    be invisible to them.
    """

    def register_runner(self, runner) -> None:
        # torchtpu extension unknown to upstream MultiConnector: forward it to
        # the children, or the Raiden child never gets a runner and fails with
        # "register_runner must be called before transfer" on the first load.
        for c in self._connectors:
            fn = getattr(c, "register_runner", None)
            if fn is not None:
                fn(runner)

    def get_block_ids_with_load_errors_group_index(self) -> int | None:
        """Return the KV cache group shared by all children's load errors.

        Failed block ids are unioned by upstream
        ``get_block_ids_with_load_errors``; this index only tells the
        scheduler which group's block table those ids resolve against.
        One scalar cannot scope two groups, so conflicting child indices
        raise instead of mis-scoping.
        """
        index: int | None = None
        for c in self._connectors:
            fn = getattr(c, "get_block_ids_with_load_errors_group_index", None)
            if fn is None:
                continue
            child_index = fn()
            if child_index is None:
                continue
            if index is not None and child_index != index:
                raise RuntimeError(
                    "Conflicting KV-load-error group indices across child "
                    f"connectors: {index} vs {child_index}; cannot scope "
                    "invalid blocks to a single cache group"
                )
            index = child_index
        return index

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return any(c.supports_divergent_local_hybrid_hits for c in self._connectors)

    def poll_and_complete_transfers(
        self, scheduler
    ) -> tuple[int, int, int, bool]:
        total_ns = 0
        total_early = 0
        total_waited = 0
        any_waited = False
        for c in self._connectors:
            fn = getattr(c, "poll_and_complete_transfers", None)
            if callable(fn):
                ns, early, waited_reqs, waited = fn(scheduler)
                total_ns += ns
                total_early += early
                total_waited += waited_reqs
                any_waited = any_waited or waited
        return total_ns, total_early, total_waited, any_waited

    def resolve_divergent_local_hit_tokens(
        self,
        request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        for c in self._connectors:
            fn = getattr(c, "resolve_divergent_local_hit_tokens", None)
            if fn is not None:
                res = fn(request, num_local, num_reconciled, block_size)
                if res is not None:
                    return res
        return None

    def estimate_external_hit_tokens(
        self,
        request,
        num_local_tokens: int,
    ) -> int:
        best = 0
        for c in self._connectors:
            fn = getattr(c, "estimate_external_hit_tokens", None)
            if fn is not None:
                res = fn(request, num_local_tokens)
                if res and res > best:
                    best = res
        return best
