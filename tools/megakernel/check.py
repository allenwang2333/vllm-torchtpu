# SPDX-License-Identifier: Apache-2.0
"""Numerics checks of the K3 programs (``check spec`` / ``check handoff`` /
``check greedy``).

**spec** drives the decoder (``KimiDecoder``) as the runner does, over stand-in
unified pools, and checks greedy DSpark against plain greedy decode (the streams
agree until a near-tie), requests joining and leaving across the single and
batched speculation widths (each stream against its plain reference), seeded
sampling (one seed reproduces, another differs) and that no step after warmup
compiles; it reports tokens per step and pipelined step times.

**handoff** runs the prefill program over chunked prompts and prefix-cache hits
into stand-in pools, binds the decode program to the same pools and decodes
each request from the prefill's first token, against a decode-only replay: the
first tokens must match (later ones may part at a near-tie). With ``--dspark``
each imported draft ring must hold the prompt's last positions, and decoding
from it is compared with a cold ring in tokens per step.

**greedy** replays the fixed prompt and four short prompts on the decoder, then
writes each request's greedy tokens and text to ``--output`` (JSON): two kernel
configs' files compare a layout change (e.g. attention DP) on real text.
"""

from __future__ import annotations

import dataclasses
import gc
import time

import numpy as np

from tools.megakernel import harness, k3


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="check", required=True)
    spec = sub.add_parser(
        "spec", help="DSpark against plain decode, batch transitions, seeded sampling"
    )
    spec.add_argument("--prompt-tokens", type=int, default=4000)
    spec.add_argument("--generate", type=int, default=200)
    spec.add_argument("--single-drafts", type=int, choices=(3, 7), default=7)
    spec.add_argument("--small-batch-drafts", type=int, choices=(3, 7), default=7)
    spec.add_argument(
        "--dspark", required=True, help="the DSpark speculator checkpoint"
    )
    handoff = sub.add_parser(
        "handoff", help="prefill -> pools -> decode against a decode-only replay"
    )
    handoff.add_argument("--generate", type=int, default=32)
    handoff.add_argument("--single-drafts", type=int, choices=(3, 7), default=7)
    handoff.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: hand the draft ring over too",
    )
    handoff.add_argument("--speculate-steps", type=int, default=64)
    greedy = sub.add_parser(
        "greedy", help="real prompts replayed on the decoder, then greedy tokens"
    )
    greedy.add_argument("--prompt-tokens", type=int, default=5000)
    greedy.add_argument("--generate", type=int, default=128)
    greedy.add_argument("--output", required=True)
    for check in (spec, handoff, greedy):
        check.add_argument(
            "--model",
            required=True,
            help="the Kimi-K3 checkpoint (the tokenizer and the fixed prompt's text)",
        )


def run(args, records: harness.Records) -> None:
    {"spec": run_spec, "handoff": run_handoff, "greedy": run_greedy}[args.check](
        args, records
    )


def run_greedy(args, records: harness.Records) -> None:
    import json

    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps

    all_ids, tokenizer = k3.fixed_prompt(args.prompt_tokens, args.model)
    prompts = [np.asarray(all_ids, np.int32)] + [
        np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)
        for text in (
            "Write a short story about a robot who learns to paint.",
            "Explain, step by step, how a transistor amplifies a signal.",
            "List the prime numbers below fifty and say why each is prime.",
            "Once upon a time, in a small village by the sea,",
        )
    ]
    pages = -(-(args.prompt_tokens + args.generate + 64) // k3.logical()) + 1
    # One batch size for every config (under attention DP the five prompts
    # need B8): the replay's single rows and the batch run the same kernel.
    ops = DecodeOps(args.presharded, num_slots=2 * 8 + 1, batch_sizes=(8,))
    pools = k3.Pools(len(prompts) * pages + 1, pages, 1)
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=8,
        max_model_len=pages * k3.logical(),
    )
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]

    def step(requests):
        return decoder.step(mk.StepInput(requests)).tokens

    started = time.perf_counter()
    outs = [
        step([k3.request(f"r{n}", ids, tables[n]) for n, ids in enumerate(prompts)])
    ]
    harness.log(f"prompts replayed in {time.perf_counter() - started:.1f} s")
    for _ in range(args.generate - 1):
        outs.append(
            step([k3.request(f"r{n}", [0], tables[n]) for n in range(len(prompts))])
        )
    streams = torch.stack([o[: len(prompts), 0] for o in outs]).cpu().numpy().T
    result = [
        {
            "prompt_tokens": len(ids),
            "tokens": stream.tolist(),
            "text": tokenizer.decode(stream.tolist()),
        }
        for ids, stream in zip(prompts, streams, strict=True)
    ]
    if harness.rank() == 0:
        with open(args.output, "w") as stream:
            json.dump(result, stream, indent=1)
    for n, item in enumerate(result):
        harness.log(
            f"r{n} ({item['prompt_tokens']} prompt tokens): {item['text'][:160]!r}"
        )
    records.add(check="greedy", requests=len(prompts), generate=args.generate)


def _agree(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def run_spec(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import Control, DecodeOps
    from vllm_torchtpu.megakernel.kimi_k3.draft import DsparkDraft

    count, generate, rows = args.prompt_tokens, args.generate, args.single_drafts + 1
    all_ids, tokenizer = k3.fixed_prompt(count + 4000, args.model)
    ids = np.asarray(all_ids[:count], np.int32)

    def encode(text):
        return np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)

    prompts = {
        "A": ids,
        "B": encode("The history of the lighthouse begins"),
        "C": np.asarray(all_ids[count : count + 2000], np.int32),  # the text after A's
        "D": encode("Write a short story about a robot who learns to paint."),
        "E": encode("Explain, step by step, how a transistor amplifies a signal."),
        "F": np.asarray(all_ids[count + 2000 :], np.int32),  # the text after C's
        "G": encode("List the prime numbers below fifty and say why each is prime."),
        "H": encode("Once upon a time, in a small village by the sea,"),
    }
    # C and F retain their 2000-token fixtures even when --prompt-tokens
    # shortens A. All fixtures must fit both the pool and the draft's RoPE.
    longest_prompt = max(map(len, prompts.values()))
    pages = -(-(longest_prompt + 4 * generate + 64) // k3.LOGICAL) + 1
    # One copy of the weights for the three decoders below (max_num_seqs 8).
    ops = DecodeOps(
        args.presharded, num_slots=Control.total_slots(17), batch_sizes=(1, 2, 4, 8)
    )
    pools = k3.Pools(10 * pages + 1, pages, 1)
    draft = DsparkDraft(
        max_position=longest_prompt + 8 * generate + 64,
        checkpoint=args.dspark,
        presharded=args.presharded,
    )

    def decoder(schedule=None):
        return k3.decoder(
            ops,
            pools,
            max_num_seqs=8,
            max_model_len=pages * k3.LOGICAL,
            draft=draft,
            schedule=schedule,
        )

    # Single-request speculation at the requested width; plain batched steps.
    single = decoder(((1, 1, args.single_drafts), (2, 8, 0)))
    plain = decoder()
    # Select both singleton and batched widths, including their transitions.
    batched = decoder(
        ((1, 1, args.single_drafts), (2, 4, args.small_batch_drafts), (5, 8, 3))
    )
    harness.sync()
    tables = {
        name: np.arange(1 + n * pages, 1 + (n + 1) * pages, dtype=np.int32)
        for n, name in enumerate(("A", "B", "ref", "C", "D", "E", "F", "G", "H"))
    }

    def tokens_of(emitted):
        return [int(t) for t in torch.cat(emitted).cpu().reshape(-1).tolist() if t >= 0]

    def row(req, tokens, table, sampling=(0.0, 1.0, 0, 0)):
        return k3.request(req, tokens, table, sampling)

    def decode(decoder, requests, finished=()):
        return decoder.step(mk.StepInput(requests, finished)).tokens

    def plain_reference(prompt, n):
        outs = [decode(plain, [row("ref", prompt, tables["ref"])])[:1]]
        for _ in range(n - 1):
            outs.append(decode(plain, [row("ref", [0], tables["ref"])])[:1])
        plain.forget(("ref",))
        return tokens_of(outs)

    single.warmup()
    plain.warmup()
    reference = plain_reference(ids, generate)
    b_reference = plain_reference(prompts["B"], 24)
    harness.log("references done")

    def run_alone(req, sampling, n_steps, first=True):
        outs = []
        if first:
            outs.append(decode(single, [row(req, ids, tables["A"], sampling)])[:1])
        for _ in range(n_steps):
            outs.append(
                decode(
                    single,
                    [row(req, np.zeros(rows), tables["A"], sampling)],
                )[:1]
            )
        return outs

    a_tokens = tokens_of(run_alone("a", (0.0, 1.0, 0, 0), 60))
    same = _agree(a_tokens, reference)
    per_step = (len(a_tokens) - 1) / 60
    harness.log(
        f"greedy DSpark vs plain greedy: first {same} of "
        f"{min(len(a_tokens), len(reference))} agree; "
        f"{per_step:.2f} tokens per speculative step"
    )

    outs_a, outs_b = [], []
    out = decode(
        single,
        [
            row("a", [0], tables["A"]),
            row("b", prompts["B"], tables["B"]),
        ],
    )
    outs_a.append(out[:1])
    outs_b.append(out[1:2])
    for _ in range(23):
        out = decode(
            single,
            [row("a", [0], tables["A"]), row("b", [0], tables["B"])],
        )
        outs_a.append(out[:1])
        outs_b.append(out[1:2])
    b_tokens = tokens_of(outs_b)
    outs_a.append(decode(single, [row("a", np.zeros(rows), tables["A"])], ("b",))[:1])
    outs_a += run_alone("a", (0.0, 1.0, 0, 0), 20, first=False)
    a_more = tokens_of(outs_a)
    same_b = _agree(b_tokens, b_reference)
    harness.log(
        "B (batch 2 with A) vs its plain reference: "
        f"first {same_b} of {len(b_tokens)} agree"
    )
    harness.log(f"A with B, then alone: {tokenizer.decode(a_more[:120])!r}")
    harness.log(f"B: {tokenizer.decode(b_tokens)!r}")
    single.forget(("a",))

    runs = []
    for req, seed in (("s1", 7), ("s2", 7), ("s3", 8)):
        runs.append(tokens_of(run_alone(req, (0.7, 0.9, 50, seed), 12)))
        single.forget((req,))
    reproducible, varies = runs[0] == runs[1], runs[0] != runs[2]
    harness.log(
        f"sampled: same seed identical {reproducible}, other seed differs {varies}; "
        f"{tokenizer.decode(runs[0][:60])!r}"
    )

    run_alone("t", (0.0, 1.0, 0, 0), 4)
    bridge.materialize(single.control)
    before = dict(single.stats)
    started, host = time.perf_counter(), 0.0
    for _ in range(40):
        t0 = time.perf_counter()
        out = decode(single, [row("t", np.zeros(rows), tables["A"])])
        host += time.perf_counter() - t0
    bridge.materialize(out)
    step_ms = 1e3 * (time.perf_counter() - started) / 40
    harness.log(
        f"pipelined greedy DSpark step: {step_ms:.2f} ms "
        f"(host {1e3 * host / 40:.2f} ms per step); "
        f"ops over 40 steps: { {k: single.stats[k] - before[k] for k in before} }"
    )
    single.forget(("t",))

    # ---------------- 5. batched speculation
    batched.warmup()

    def verify_rows(requests):
        return batched.spec_tokens[requests] + 1

    references = {
        n: reference if n == "A" else plain_reference(prompts[n], generate)
        for n in prompts
    }

    def churn():
        """Requests joining and leaving across the single and batched paths
        right after warmup, timing every step: a draft state carries the
        buffer layouts its producer gave it, and a batched op compiles per
        layout mix of its states, so a step that compiles takes seconds.
        Returns the slowest join step and the slowest steady step (ms)."""
        short = {n: prompts[n][:24] for n in prompts}
        order = [
            ("A",),
            ("A", "B"),
            tuple("ABCD"),
            tuple("ABCDEFGH"),
            tuple("ABC"),
            ("C",),
            ("C", "D"),
            tuple("CDEFG"),
            tuple("ABCDEFGH"),
            ("H",),
        ]
        live, worst_join, worst_steady, log = set(), 0.0, 0.0, []
        for names in order:
            leaving = tuple("c" + n for n in sorted(live - set(names)))
            times = []
            for step in range(4):
                group = []
                for n in names:
                    if n in live:
                        q = verify_rows(len(names))
                        group.append(row("c" + n, np.zeros(q), tables[n]))
                    else:
                        group.append(row("c" + n, short[n], tables[n]))
                t0 = time.perf_counter()
                out = decode(batched, group, leaving if step == 0 else ())
                bridge.materialize(out)
                times.append(1e3 * (time.perf_counter() - t0))
                live = set(names)
            worst_join = max(worst_join, times[0])
            worst_steady = max(worst_steady, *times[1:])
            log.append(f"{''.join(names)} " + "/".join(f"{t:.0f}" for t in times))
        batched.forget(tuple("c" + n for n in live))
        harness.log(
            "churn after warmup, ms per step (join/steady...): " + "; ".join(log)
        )
        return worst_join, worst_steady

    def run_group(names, tag, sampling, steps):
        """The group's streams and tokens per step: their prompts, then
        ``steps`` batched steps of ``k`` scheduled tokens each."""
        first = [row(tag + n, prompts[n], tables[n], sampling) for n in names]
        k = verify_rows(len(names))
        following = [row(tag + n, np.zeros(k), tables[n], sampling) for n in names]
        count_ = len(names)
        outs = [decode(batched, first)[:count_]]
        for _ in range(steps):
            outs.append(decode(batched, following)[:count_])
        stacked = torch.stack(outs).cpu().numpy()  # [steps + 1, rows, 8]
        streams = [
            [int(t) for t in stacked[:, r].reshape(-1) if t >= 0] for r in range(count_)
        ]
        rates = [float((stacked[1:, r] >= 0).sum(axis=1).mean()) for r in range(count_)]
        return streams, rates

    def report(label, names, streams, rates):
        agree = [
            _agree(st, references[n]) for n, st in zip(names, streams, strict=True)
        ]
        harness.log(
            f"batched greedy, {label}: "
            + "; ".join(
                f"{n} agrees for {a} of {min(len(st), len(references[n]))} tokens at "
                f"{r:.2f} tokens per step"
                for n, a, st, r in zip(names, agree, streams, rates, strict=True)
            )
        )
        return agree

    churn_join, churn_steady = churn()
    steps = 48
    pair_streams, pair_rates = run_group(("A", "B"), "p", (0.0, 1.0, 0, 0), steps)
    decode(batched, [], ("pB",))
    alone = [
        decode(batched, [row("pA", np.zeros(rows), tables["A"])])[:1] for _ in range(20)
    ]
    pa_more = pair_streams[0] + tokens_of(alone)
    batched.forget(("pA",))
    same_pa, same_pb = report(
        f"2 x {verify_rows(2)} rows", ("A", "B"), pair_streams, pair_rates
    )
    same_more = _agree(pa_more, reference)
    harness.log(f"A alone again after the pair: {same_more} agree")
    four = ("A", "B", "C", "D")
    four_streams, four_rates = run_group(four, "f", (0.0, 1.0, 0, 0), 32)
    batched.forget(tuple("f" + n for n in four))
    four_agree = report(f"4 x {verify_rows(4)} rows", four, four_streams, four_rates)

    def plain_group(names, streams, n=112):
        """The group in plain steps: how far batch composition alone moves
        each greedy stream from its single-request reference, and how far
        the batched speculative ``streams`` follow these."""
        first = [row("g" + m, prompts[m], tables[m]) for m in names]
        following = [row("g" + m, [0], tables[m]) for m in names]
        outs = [decode(plain, first)[: len(names)]]
        for _ in range(n - 1):
            outs.append(decode(plain, following)[: len(names)])
        plain.forget(tuple("g" + m for m in names))
        stacked = torch.stack(outs).cpu().numpy()
        tokens = [[int(t) for t in stacked[:, r, 0]] for r in range(len(names))]
        agree = [_agree(t, references[m]) for t, m in zip(tokens, names, strict=True)]
        follow = [_agree(st, t) for st, t in zip(streams, tokens, strict=True)]
        harness.log(
            f"plain batch-{len(names)} greedy: "
            + "; ".join(
                f"{m} agrees for {a} of {n} (the batched stream follows it for "
                f"{f} of {min(n, len(st))})"
                for m, a, f, st in zip(names, agree, follow, streams, strict=True)
            )
        )
        return agree, follow

    plain_four, follow_four = plain_group(four, four_streams)
    eight = tuple("ABCDEFGH")
    eight_streams, eight_rates = run_group(eight, "e", (0.0, 1.0, 0, 0), 32)
    batched.forget(tuple("e" + n for n in eight))
    eight_agree = report(
        f"8 x {verify_rows(8)} rows", eight, eight_streams, eight_rates
    )
    plain_eight, follow_eight = plain_group(eight, eight_streams)
    five = tuple("ABCDE")
    five_streams, five_rates = run_group(five, "v", (0.0, 1.0, 0, 0), 16)
    batched.forget(tuple("v" + n for n in five))
    five_agree = report(
        f"5 (padded to 8) x {verify_rows(5)} rows", five, five_streams, five_rates
    )
    three = ("A", "B", "C")
    three_streams, three_rates = run_group(three, "t", (0.0, 1.0, 0, 0), 16)
    batched.forget(tuple("t" + n for n in three))
    three_agree = report(
        f"3 (padded to 4) x {verify_rows(3)} rows", three, three_streams, three_rates
    )
    sampled_runs = []
    for tag, seed in (("x", 7), ("y", 7), ("z", 8)):
        streams, _ = run_group(("A", "B"), tag, (0.7, 0.9, 50, seed), 12)
        batched.forget((f"{tag}A", f"{tag}B"))
        sampled_runs.append(streams)
    b_reproducible = sampled_runs[0] == sampled_runs[1]
    b_varies = sampled_runs[0] != sampled_runs[2]
    harness.log(
        f"batched sampled: same seed identical {b_reproducible}, other seed "
        f"differs {b_varies}; A {tokenizer.decode(sampled_runs[0][0][:40])!r}"
    )

    def pipelined(decoder, requests, query, n_steps=40):
        """ms per step of the group's first ``requests`` prompts (distinct
        tokens per row, as serving routes them; the rates above are theirs)."""
        names = eight[:requests]
        first = [row(f"q{r}", prompts[n], tables[n]) for r, n in enumerate(names)]
        decode(decoder, first)
        following = [
            row(f"q{r}", np.zeros(query), tables[n]) for r, n in enumerate(names)
        ]
        for _ in range(4):
            out = decode(decoder, following)
        bridge.materialize(out)
        started = time.perf_counter()
        for _ in range(n_steps):
            out = decode(decoder, following)
        bridge.materialize(out)
        decoder.forget(tuple(f"q{r}" for r in range(requests)))
        return 1e3 * (time.perf_counter() - started) / n_steps

    timing = {}
    for requests, rates in ((2, pair_rates), (4, four_rates), (8, eight_rates)):
        k = verify_rows(requests)
        spec_ms, plain_ms = (
            pipelined(batched, requests, k),
            pipelined(plain, requests, 1),
        )
        timing[requests] = (round(spec_ms, 3), round(plain_ms, 3))
        harness.log(
            f"pipelined batched DSpark step ({requests} x {k} rows): {spec_ms:.2f} ms; "
            f"plain batch-{requests} step {plain_ms:.2f} ms; tokens per ms at the "
            f"greedy rates above {sum(rates) / spec_ms:.3f} vs "
            f"{requests / plain_ms:.3f}"
        )
    records.add(
        case="spec",
        single_drafts=args.single_drafts,
        small_batch_drafts=args.small_batch_drafts,
        dspark_vs_plain=[same, min(len(a_tokens), len(reference))],
        tokens_per_step=per_step,
        batch2_vs_plain=[same_b, len(b_tokens)],
        seed_reproducible=reproducible,
        seed_varies=varies,
        pipelined_ms=round(step_ms, 3),
        batched_pair_agree=[same_pa, same_pb],
        batched_pair_tokens_per_step=[round(r, 2) for r in pair_rates],
        batched_then_alone=same_more,
        batched_four_agree=four_agree,
        plain_four_agree=plain_four,
        batched_four_follow_plain=follow_four,
        batched_four_tokens_per_step=[round(r, 2) for r in four_rates],
        batched_three_agree=three_agree,
        batched_eight_agree=eight_agree,
        plain_eight_agree=plain_eight,
        batched_eight_follow_plain=follow_eight,
        batched_eight_tokens_per_step=[round(r, 2) for r in eight_rates],
        batched_five_agree=five_agree,
        batched_seed_reproducible=b_reproducible,
        batched_seed_varies=b_varies,
        batched_ms={str(b): t for b, t in timing.items()},
        churn_worst_ms=[round(churn_join), round(churn_steady)],
    )
    ok = reproducible and varies and same > 0
    ok = ok and b_reproducible and b_varies and same_pa > 0 and same_pb > 0
    ok = ok and all(a > 0 for a in four_agree + three_agree + eight_agree + five_agree)
    if churn_steady > 1000 or churn_join > 1000:
        harness.log("a step after warmup compiled (churn above)")
        ok = False
    if not ok:
        harness.fail("spec check")


def run_handoff(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program

    generate, max_model_len = args.generate, 32768
    # The kernel rank stands in for the TP / DCP rank, so the prefiller writes
    # the pools in the mesh order the decoder binds (as the KV transfer does).
    rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
    ids, tokenizer = k3.fixed_prompt(24000, args.model)
    lengths = {
        "A": k3.LOGICAL + 3808,
        "B": k3.LOGICAL + 1500,
        "C": 700,
        "D": k3.PAGE,
        "E": 100,
        "F": k3.LOGICAL + 400,
    }
    hit = {"B": ("A", k3.LOGICAL), "F": ("A", k3.LOGICAL)}
    prompts, at = {}, 0
    for req, n in lengths.items():
        if req in hit:
            donor, cached = hit[req]
            prompts[req] = prompts[donor][:cached] + ids[at : at + n - cached]
            at += n - cached
        else:
            prompts[req] = ids[at : at + n]
            at += n
    schedule = [
        [("A", 4096), ("C", 700)],
        *[[("A", end)] for end in range(8192, k3.LOGICAL + 1, 4096)],
        [("A", lengths["A"])],
        [("B", lengths["B"]), ("D", k3.PAGE)],
        [("E", 100)],
        [("F", lengths["F"])],
    ]
    decode_pages = max_model_len // k3.LOGICAL + 1
    pools = k3.Pools(
        1
        + 3 * decode_pages
        + sum(-(-n // k3.LOGICAL) + 3 * -(-n // k3.PAGE) for n in lengths.values())
        + 8,
        decode_pages,
        max_model_len // k3.PAGE,
    )
    logical, states = {}, {}
    for req, n in lengths.items():
        blocks = -(-n // k3.LOGICAL)
        if req in hit:
            donor, cached = hit[req]
            logical[req] = logical[donor][: cached // k3.LOGICAL] + pools.fresh(
                blocks - cached // k3.LOGICAL
            )
        else:
            logical[req] = pools.fresh(blocks)
        states[req] = [pools.fresh(-(-n // k3.PAGE)) for _ in range(3)]
    for req, (donor, cached) in hit.items():
        for g in range(3):
            states[req][g][cached // k3.PAGE - 1] = states[donor][g][
                cached // k3.PAGE - 1
            ]
    # The decoder's KDA tables also cover the token it decodes next, as vLLM
    # allocates them: D (256 tokens) gets a second column the prefill never
    # wrote; the transferred state is in the first.
    decode_states = {
        req: [s + pools.fresh(-(-(n + 1) // k3.PAGE) - len(s)) for s in states[req]]
        for req, n in lengths.items()
    }
    reference_tables = {req: pools.fresh(decode_pages) for req in lengths}

    def row(req_id, computed, scheduled, tokens, prompt_len, mla_table, kda_tables):
        tokens = np.asarray(tokens, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=scheduled,
            token_ids=tokens,
            num_tokens=len(tokens),
            num_prompt_tokens=prompt_len,
            block_tables=pools.tables(mla_table, kda_tables),
        )

    from pathlib import Path

    draft = Path(args.dspark) if args.dspark else None
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=4,
            max_model_len=max_model_len,
            max_num_batched_tokens=4096,
        ),
        args.presharded,
        draft,
        tp_rank=rank,
    )
    prefiller.load_weights()
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()
    done = {req: hit[req][1] if req in hit else 0 for req in lengths}
    first_token = {}
    for step in schedule:
        rows = [
            row(
                req,
                done[req],
                total - done[req],
                prompts[req],
                len(prompts[req]),
                logical[req],
                states[req],
            )
            for req, total in step
        ]
        tokens = (
            prefiller.step(mk.StepInput(rows)).tokens[: len(rows), 0].cpu().tolist()
        )
        for (req, total), token in zip(step, tokens, strict=True):
            done[req] = total
            if total == len(prompts[req]):
                first_token[req] = int(token)
    bridge.materialize(pools.tensors[0])
    harness.log(f"prefill done; first tokens {first_token}")
    # The registered ops keep the prefill program alive: drop its weights by hand.
    prefiller.program.weights.clear()
    if prefiller.ring is not None:
        prefiller.ring.weights = ()
    del prefiller
    gc.collect()
    harness.sync()

    decoder = k3_program.build(
        k3.engine_config(
            "decode",
            max_num_seqs=2,
            max_model_len=max_model_len,
            max_num_batched_tokens=512,
            num_speculative_tokens=args.single_drafts if draft else 0,
            # The check decodes one request at a time, at the requested width.
            num_speculative_tokens_per_batch_size=(
                (1, 1, args.single_drafts),
                (2, 2, 0),
            )
            if draft
            else None,
        ),
        args.presharded,
        draft,
        tp_rank=rank,
    )
    decoder.load_weights()
    decoder.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()

    def continue_from(req_id, first_row):
        outs = [decoder.step(mk.StepInput([first_row])).tokens[:1]]
        computed = first_row.end
        for _ in range(generate - 1):
            following = mk.RequestInput(
                req_id=req_id,
                computed=computed,
                scheduled=1,
                token_ids=np.zeros(computed + 1, np.int32),
                num_tokens=computed + 1,
                num_prompt_tokens=first_row.num_prompt_tokens,
                block_tables=first_row.block_tables,
            )
            outs.append(decoder.step(mk.StepInput([following])).tokens[:1])
            computed += 1
        decoder.step(mk.StepInput([], (req_id,)))
        return [int(t) for t in torch.cat(outs).cpu().reshape(-1).tolist() if t >= 0]

    all_first_match = True
    for req, n in lengths.items():
        pd = [first_token[req]] + continue_from(
            f"pd-{req}",
            row(
                f"pd-{req}",
                n,
                1,
                prompts[req] + [first_token[req]],
                n,
                logical[req],
                decode_states[req],
            ),
        )[: generate - 1]
        ref = continue_from(
            f"ref-{req}",
            row(
                f"ref-{req}", 0, n, prompts[req], n, reference_tables[req], [[], [], []]
            ),
        )
        same = _agree(pd, ref)
        all_first_match &= pd[0] == ref[0]
        harness.log(
            f"{req} ({n} tokens{', hit ' + str(hit[req][1]) if req in hit else ''}): "
            f"first token {'matches' if pd[0] == ref[0] else 'DIFFERS'}; "
            f"first {same} of {len(ref)} tokens agree; "
            f"P/D {tokenizer.decode(pd[:24])!r}; "
            f"reference {tokenizer.decode(ref[:24])!r}"
        )
        records.add(
            case=f"handoff-{req}",
            prompt_tokens=n,
            first_token_matches=pd[0] == ref[0],
            agree=[same, len(ref)],
        )
    rings_ok = True
    if draft is not None:
        rings_ok = _handoff_rings(
            args,
            records,
            decoder,
            lengths,
            row,
            prompts,
            first_token,
            logical,
            decode_states,
        )
    if not all_first_match:
        harness.fail("a prefill first token differs from the decode-only reference")
    if not rings_ok:
        harness.fail("an imported draft ring does not hold the prompt's last positions")


def _handoff_rings(
    args, records, decoder, lengths, row, prompts, first_token, logical, states
) -> bool:
    """The DSpark half of ``check handoff`` (see the module docstring)."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.pools import kda_blocks_at_length

    window, steps, ok = decoder.draft.config.window, args.speculate_steps, True
    for req, n in lengths.items():
        first = row(
            f"spec-{req}",
            n,
            1,
            prompts[req] + [first_token[req]],
            n,
            logical[req],
            states[req],
        )
        blocks = kda_blocks_at_length(first.block_tables, decoder._kda_groups, n)
        _, positions = decoder._importer.import_ring(
            decoder.draft.config, torch.from_numpy(blocks).to("tpu"), n
        )
        held = int((positions >= 0).sum().cpu())
        ok &= held == min(n, window)
        counts = {}
        for arm, ring in (("ring", True), ("cold", False)):
            decoder._ring = ring
            req_id = f"spec-{arm}-{req}"
            outs = [
                decoder.step(
                    mk.StepInput([dataclasses.replace(first, req_id=req_id)])
                ).tokens[:1]
            ]
            following = mk.RequestInput(
                req_id=req_id,
                computed=n + 1,
                scheduled=args.single_drafts + 1,
                token_ids=np.zeros(n + args.single_drafts + 2, np.int32),
                num_tokens=n + args.single_drafts + 2,
                num_prompt_tokens=n,
                block_tables=first.block_tables,
            )
            for _ in range(steps - 1):
                outs.append(decoder.step(mk.StepInput([following])).tokens[:1])
            decoder.step(mk.StepInput([], (req_id,)))
            emitted = torch.cat(outs).cpu().numpy()
            counts[arm] = (emitted >= 0).sum(axis=1)
        decoder._ring = True
        windows = [w for w in (16, 64) if w < steps] + [steps]
        rates = {
            arm: [round(float(counts[arm][:w].mean()), 2) for w in windows]
            for arm in counts
        }
        harness.log(
            f"{req} ({n} tokens): imported ring holds {held} positions "
            f"(expected {min(n, window)}); DSpark tokens per step ring / cold over "
            "the first "
            + "; ".join(
                f"{w} steps: {rates['ring'][i]} / {rates['cold'][i]}"
                for i, w in enumerate(windows)
            )
        )
        records.add(
            case=f"handoff-ring-{req}",
            prompt_tokens=n,
            ring_positions=held,
            tokens_per_step_ring=rates["ring"],
            tokens_per_step_cold=rates["cold"],
        )
    return ok
