"""CPU checks for prefix_routing_hook.py against a stub DPLBAsyncMPClient.

    python3 test_prefix_routing_hook.py [--old OLD_HOOK.py]
    python3 test_prefix_routing_hook.py --embed   # after editing the hook

1. The recipe embeds exactly prefix_routing_hook.py.
2. With --old (the previous hook, decoded from the recipe's base64), one API server
   routes a randomized request stream identically to it, for each env-knob setting.
3. With N API server processes sharing the table and each request sent to a random
   process, every request of a session goes to one engine, and a subagent session
   joins the engine that owns its parent's prefix.
"""

import argparse
import base64
import collections
import multiprocessing
import os
import random
import re
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
HOOK = os.path.join(HERE, "prefix_routing_hook.py")
RECIPE = os.path.join(HERE, "infx-deepseek-v4-agentx-evolve.jsonnet")
NUM_ENGINES = 16
KNOBS = ("ENABLE_PREFIX_ROUTING", "PREFIX_ROUTING_NEW_SESSION_AFFINITY",
         "PREFIX_ROUTING_AFFINITY_GUARD", "PREFIX_ROUTING_CONTEXT_LOAD")


class StubDPLBAsyncMPClient:
    """The attributes and methods of vLLM 0.29's DPLBAsyncMPClient the hook touches."""

    def __init__(self, client_count=1, client_index=0):
        self.client_count = client_count
        self.client_index = client_index
        self.core_engines = [i.to_bytes(2, "little") for i in range(NUM_ENGINES)]
        self.lb_engines = [[0, 0, 0.0] for _ in range(NUM_ENGINES)]
        self.engine_inflight = collections.Counter()
        self.reqs_in_flight = {}
        self.eng_start_index = (NUM_ENGINES * client_index) // client_count

    def get_core_engine_for_request(self, request):
        if (idx := request.data_parallel_rank) is None:
            best = None
            for i in range(NUM_ENGINES):
                j = (self.eng_start_index + i) % NUM_ENGINES
                w, r, _ = self.lb_engines[j]
                score = max(self.client_count * self.engine_inflight[self.core_engines[j]], w + r)
                if best is None or score < best:
                    best, idx = score, j
            self.lb_engines[idx][0] += self.client_count
            self.eng_start_index = (self.eng_start_index + 1) % NUM_ENGINES
        chosen = self.core_engines[idx]
        self.reqs_in_flight[request.request_id] = chosen
        self.engine_inflight[chosen] += 1
        return chosen

    @staticmethod
    async def process_engine_outputs(self, outputs):
        for req_id in outputs.finished_requests or ():
            if (engine := self.reqs_in_flight.pop(req_id, None)) is not None:
                self.engine_inflight[engine] -= 1


def install(hook_src, env):
    """Exec a hook against a fresh stub class under `env`; returns the patched class."""
    cls = type("DPLBAsyncMPClient", (StubDPLBAsyncMPClient,), {})
    mod = types.ModuleType("vllm.v1.engine.core_client")
    mod.DPLBAsyncMPClient = cls
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.v1.engine.core_client"] = mod
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update(env)
    exec(compile(hook_src, "<hook>", "exec"), {"__name__": "hook"})
    return cls


def drive(coro):
    try:
        coro.send(None)
    except StopIteration:
        return
    raise AssertionError("process_engine_outputs awaited something")


def finish(cls, client, req_ids):
    drive(cls.process_engine_outputs(client, types.SimpleNamespace(finished_requests=req_ids)))


def workload(seed, n_events=3000):
    """A replayable event stream: requests (sessions, subagents, sessionless, pinned),
    completions, coordinator load snapshots and clock advances."""
    rng = random.Random(seed)
    toks = lambda n: [rng.randrange(1 << 17) for _ in range(n)]
    sessions = {}  # sid -> current prompt
    parent_of = {}  # subagent sid -> (parent sid, shared tokens)
    inflight, events, n_req = [], [], 0
    for _ in range(n_events):
        x = rng.random()
        if x < 0.25 and inflight:
            done = rng.sample(inflight, rng.randint(1, min(4, len(inflight))))
            for r in done:
                inflight.remove(r)
            events.append(("finish", done))
            continue
        if x < 0.30:
            events.append(("stats", [[rng.randint(0, 6), rng.randint(0, 8), rng.random()]
                                     for _ in range(NUM_ENGINES)]))
            continue
        if x < 0.33:
            events.append(("tick", rng.choice([0.0, 0.001, 0.5])))
            continue
        n_req += 1
        rid = "req-%d" % n_req
        sid, rank, prompt = None, None, None
        y = rng.random()
        if y < 0.15 or not sessions:  # new top-level session
            sid = "sess-%d" % n_req
            prompt = toks(rng.randint(2000, 30000))
        elif y < 0.30:  # subagent: a new session sharing a parent's prefix
            parent = rng.choice(sorted(sessions))
            base = sessions[parent]
            shared = min(len(base), rng.choice([512 * rng.randint(1, 11), 512 * rng.randint(12, 40) + rng.randint(0, 511)]))
            sid = "sub-%d" % n_req
            prompt = base[:shared] + toks(rng.randint(100, 4000))
            parent_of[sid] = (parent, shared)
        elif y < 0.85:  # next turn of an existing session
            sid = rng.choice(sorted(sessions))
            prompt = sessions[sid] + toks(rng.randint(50, 3000))
        elif y < 0.95:  # sessionless, long or short, maybe sharing a prefix
            base = sessions[rng.choice(sorted(sessions))] if rng.random() < 0.5 else []
            prompt = base[:rng.randint(0, len(base))] + toks(rng.randint(100, 9000))
        else:  # pinned by a router header
            sid = rng.choice(sorted(sessions)) if rng.random() < 0.5 else None
            rank = rng.randrange(NUM_ENGINES)
            prompt = toks(rng.randint(100, 3000))
        if sid is not None and rank is None:
            sessions[sid] = prompt
        inflight.append(rid)
        events.append(("req", types.SimpleNamespace(request_id=rid, session_id=sid,
                                                    data_parallel_rank=rank, prompt_token_ids=prompt)))
    return events, parent_of


class FakeClock:
    t = 1000.0

    def __call__(self):
        return self.t


def replay(cls, client, events, clock):
    """Route every request; returns [(request_id, session_id, engine index)]."""
    out = []
    for kind, arg in events:
        if kind == "finish":
            finish(cls, client, arg)
        elif kind == "stats":
            client.lb_engines = [list(v) for v in arg]
        elif kind == "tick":
            clock.t += arg
        else:
            eng = cls.get_core_engine_for_request(client, arg)
            out.append((arg.request_id, arg.session_id, client.core_engines.index(eng)))
    return out


ENVS = [
    {},
    {"PREFIX_ROUTING_NEW_SESSION_AFFINITY": "1"},
    {"PREFIX_ROUTING_NEW_SESSION_AFFINITY": "1", "PREFIX_ROUTING_AFFINITY_GUARD": "0"},
    {"PREFIX_ROUTING_NEW_SESSION_AFFINITY": "1", "PREFIX_ROUTING_CONTEXT_LOAD": "1"},
    {"PREFIX_ROUTING_CONTEXT_LOAD": "true"},
    {"ENABLE_PREFIX_ROUTING": "0"},
]


EMBED_RE = re.compile(r'(echo ")([A-Za-z0-9+/=]+)(" \| base64 -d >> "\$CORE_CLIENT_PATH")')


def embed():
    recipe = open(RECIPE).read()
    b64 = base64.b64encode(open(HOOK).read().encode()).decode()
    out, n = EMBED_RE.subn(lambda m: m.group(1) + b64 + m.group(3), recipe)
    assert n == 1, n
    open(RECIPE, "w").write(out)


def check_embedded():
    src = open(HOOK).read()
    m = EMBED_RE.search(open(RECIPE).read())
    assert m, "no hook in the recipe"
    assert base64.b64decode(m.group(2)).decode() == src, "recipe hook != prefix_routing_hook.py; re-embed it"
    print("embedded hook matches prefix_routing_hook.py")


def check_single_server_matches(old_src, seeds):
    new_src = open(HOOK).read()
    clock = FakeClock()
    time.monotonic = clock  # both hooks read the clock through the time module
    for env in ENVS:
        for seed in seeds:
            events, _ = workload(seed)
            routes = []
            for src in (old_src, new_src):
                clock.t = 1000.0
                cls = install(src, env)
                routes.append(replay(cls, cls(), events, clock))
            diff = [(a, b) for a, b in zip(*routes) if a != b]
            assert len(routes[0]) == len(routes[1]) and not diff, (env, seed, diff[:5])
            n_sess = sum(1 for _, s, _ in routes[0] if s)
            print("N=1 %-90s seed %d: %d requests (%d with a session) routed identically, %d engines used"
                  % (env, seed, len(routes[0]), n_sess, len({e for _, _, e in routes[0]})))


def worker(conn, client_index, client_count, env):
    cls = install(open(HOOK).read(), env)
    client = cls(client_count=client_count, client_index=client_index)
    while (msg := conn.recv()) is not None:
        kind, arg = msg
        if kind == "finish":
            finish(cls, client, arg)
        elif kind == "stats":
            client.lb_engines = [list(v) for v in arg]
        else:
            conn.send(client.core_engines.index(cls.get_core_engine_for_request(client, arg)))
    conn.close()


def check_shared(n_procs, seed, env):
    ctx = multiprocessing.get_context("spawn")  # as vLLM's APIServerProcessManager
    pipes, procs = [], []
    for k in range(n_procs):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, k, n_procs, env))
        p.start()
        pipes.append(a)
        procs.append(p)
    rng = random.Random(seed)
    events, parent_of = workload(seed, n_events=4000)
    owner_proc, routed = {}, []
    for kind, arg in events:
        if kind == "req":
            k = rng.randrange(n_procs)
            owner_proc[arg.request_id] = k
            pipes[k].send((kind, arg))
            routed.append((arg, pipes[k].recv(), k))
        elif kind == "finish":
            by_proc = collections.defaultdict(list)
            for r in arg:
                by_proc[owner_proc[r]].append(r)
            for k, rs in by_proc.items():
                pipes[k].send((kind, rs))
        elif kind == "stats":
            for p in pipes:
                p.send((kind, arg))
    for p in pipes:
        p.send(None)
    for p in procs:
        p.join(30)
        assert p.exitcode == 0, p.exitcode
    stale = [f for f in os.listdir("/dev/shm") if f.startswith("vllm_prefix_routing_%d_" % os.getpid())]
    assert not stale, stale  # the creator unlinks the table at exit

    engines, procs_of = collections.defaultdict(set), collections.defaultdict(set)
    first = {}
    for req, eng, k in routed:
        if req.session_id and req.data_parallel_rank is None:
            engines[req.session_id].add(eng)
            procs_of[req.session_id].add(k)
            first.setdefault(req.session_id, eng)
    split = {s: e for s, e in engines.items() if len(e) > 1}
    assert not split, split
    multi = sum(1 for s in procs_of if len(procs_of[s]) > 1)
    joined = [first[s] == first[p] for s, (p, shared) in parent_of.items() if shared >= 12 * 512]
    msg = ("N=%d %s seed %d: %d sessions (%d served by >1 process) each on one engine"
           % (n_procs, env, seed, len(engines), multi))
    if env.get("PREFIX_ROUTING_AFFINITY_GUARD") == "1e9":
        assert all(joined), joined.count(False)
        msg += "; %d/%d subagents joined their parent's engine across processes" % (sum(joined), len(joined))
    print(msg + "; %d engines used" % len({e for _, e, _ in routed}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", help="previous hook source, for the N=1 equivalence check")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--embed", action="store_true", help="write the hook into the recipe and exit")
    args = ap.parse_args()
    if args.embed:
        return embed()
    check_embedded()
    if args.old:
        check_single_server_matches(open(args.old).read(), range(args.seeds))
    for n in (2, 4):
        for env in ({"PREFIX_ROUTING_NEW_SESSION_AFFINITY": "1"},
                    {"PREFIX_ROUTING_NEW_SESSION_AFFINITY": "1", "PREFIX_ROUTING_AFFINITY_GUARD": "1e9"},
                    {"PREFIX_ROUTING_NEW_SESSION_AFFINITY": "1", "PREFIX_ROUTING_CONTEXT_LOAD": "1"}):
            check_shared(n, 100 + n, env)
    print("all checks passed")


if __name__ == "__main__":
    main()
