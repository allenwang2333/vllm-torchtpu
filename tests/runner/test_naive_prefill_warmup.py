"""Warmup must cover every shape a staged bundle can launch with.

A bundle's compiled program is selected by its request dimension as well as
its token count. Warming only the all-dummy shape leaves the request shapes a
*real* slot produces uncompiled, and the first real staged step then compiles
while its peers wait inside the collectives they already entered.
"""
import inspect


def test_warmup_covers_the_request_shapes_a_real_slot_produces():
    """A real slot holding n requests pads to `num_reqs_paddings`; an idle
    rank's dummy uses `num_reqs_max_model_len`. Both have to be warmed."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    class FakeRunner:
        num_reqs_paddings = [8, 16]
        num_reqs_max_model_len = 16

    counts = TPUModelRunner._staged_warmup_req_counts(FakeRunner())
    assert counts == (8, 16)
    # MIN_NUM_SEQS is what a single-request slot pads to, and it is the shape
    # the previous warmup missed entirely.
    assert 8 in counts

    class NarrowRunner:
        """A cap below some paddings must not warm shapes that cannot occur."""
        num_reqs_paddings = [8, 16, 32]
        num_reqs_max_model_len = 8

    assert TPUModelRunner._staged_warmup_req_counts(NarrowRunner()) == (8, )

    class OffGridRunner:
        """The idle-rank shape need not be one of the paddings.

        `num_reqs_max_model_len` is capped by the Pallas backend's SMEM limit,
        so it can land between padding buckets. It still has to be warmed --
        it is the shape every idle rank launches.
        """
        num_reqs_paddings = [8, 16, 32]
        num_reqs_max_model_len = 12

    assert TPUModelRunner._staged_warmup_req_counts(OffGridRunner()) == (8, 12)


def test_warmup_request_shapes_are_deduplicated_and_sorted():
    """The idle shape is usually also a padding; it must be warmed once."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    class FakeRunner:
        num_reqs_paddings = [8, 16]
        num_reqs_max_model_len = 16

    counts = TPUModelRunner._staged_warmup_req_counts(FakeRunner())
    assert len(counts) == len(set(counts))
    assert sorted(counts) == list(counts)


def test_a_dummy_bundle_takes_a_request_count_and_defaults_to_the_idle_shape():
    """Warmup drives the request dimension directly; the live idle path must
    keep getting the shape its peers expect when it passes nothing."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    signature = inspect.signature(
        TPUModelRunner._prepare_dummy_prefill_microbatch)
    assert "num_reqs" in signature.parameters
    assert signature.parameters["num_reqs"].default is None
    # Checked against the live signature rather than a fake: a fake that
    # defines the interface it stands in for proves nothing.
    idle = inspect.signature(TPUModelRunner._run_idle_two_microbatch_prefill)
    assert "num_reqs" in idle.parameters
    assert idle.parameters["num_reqs"].default is None
