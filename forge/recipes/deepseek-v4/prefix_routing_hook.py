def _install_prefix_aware_routing():
    import os, time, array, zlib, mmap, hashlib, atexit
    if os.environ.get("ENABLE_PREFIX_ROUTING", "1") not in ("1", "true", "True"):
        return

    from vllm.v1.engine.core_client import DPLBAsyncMPClient
    orig_get = DPLBAsyncMPClient.get_core_engine_for_request

    _BLOCK_SIZE = 512
    _MIN_MATCH_BLOCKS = 12
    # A new session whose prompt shares >= _MIN_MATCH_BLOCKS blocks with a prompt
    # already routed (a subagent stream and its parent) joins that engine, unless
    # the engine has more than this many requests above the mean in flight.
    _affinity = os.environ.get("PREFIX_ROUTING_NEW_SESSION_AFFINITY", "0") in ("1", "true", "True")
    _affinity_guard = float(os.environ.get("PREFIX_ROUTING_AFFINITY_GUARD", "4"))
    # A new session goes to the engine with the fewest prompt tokens in flight
    # (waiting and running) instead of the fewest requests: under DP lockstep a
    # rank holding several long-context prompts sets every step's attention time.
    _context_load = os.environ.get("PREFIX_ROUTING_CONTEXT_LOAD", "0") in ("1", "true", "True")
    _req_ctx = {}

    # Routing state is one mmap shared by every API server process of a `vllm serve`
    # (--api-server-count > 1 spawns them as children of one launcher), so a session
    # keeps its engine whichever process accepts its next connection. One API server
    # keeps it in anonymous memory. Hash tables use linear probing over uint64 entries
    # (key << 8 | engine + 1, 0 = empty), written with single aligned 8-byte stores:
    # a race between processes loses at most one entry.
    _MAX_ENGINES = 255
    _MAX_CLIENTS = 64
    _SESSION_SLOTS = 1 << 18
    _BLOCK_SLOTS = 1 << 23
    _PROBES = 32
    _N_WORDS = 2 * _MAX_ENGINES + _MAX_CLIENTS * _MAX_ENGINES + _SESSION_SLOTS + _BLOCK_SLOTS
    _st = {}

    def _open_table(client_count):
        size = 8 * _N_WORDS
        if client_count == 1:
            return mmap.mmap(-1, size), "private"
        ppid = os.getppid()
        with open("/proc/%d/stat" % ppid) as f:
            start_ticks = f.read().rsplit(")", 1)[1].split()[19]
        path = "/dev/shm/vllm_prefix_routing_%d_%s" % (ppid, start_ticks)
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.ftruncate(fd, size)
            atexit.register(os.unlink, path)
            how = "created"
        except FileExistsError:
            fd = os.open(path, os.O_RDWR)
            deadline = time.monotonic() + 30
            while os.fstat(fd).st_size < size:
                if time.monotonic() > deadline:
                    raise RuntimeError("prefix routing table %s never reached %d bytes" % (path, size))
                time.sleep(0.01)
            how = "attached"
        buf = mmap.mmap(fd, size)
        os.close(fd)
        return buf, "%s %s" % (how, path)

    def _state(self):
        if _st:
            return _st
        assert len(self.core_engines) <= _MAX_ENGINES and self.client_count <= _MAX_CLIENTS
        buf, where = _open_table(self.client_count)
        mv = memoryview(buf)
        e, c = 8 * _MAX_ENGINES, 8 * _MAX_ENGINES * _MAX_CLIENTS
        o_sess = 2 * e + c
        o_blk = o_sess + 8 * _SESSION_SLOTS
        _st.update(
            buf=buf,
            used=mv[0:e].cast("Q"),
            last=mv[e:2 * e].cast("d"),
            ctx=mv[2 * e:2 * e + c].cast("q"),
            sess=mv[o_sess:o_blk].cast("Q"),
            blk=mv[o_blk:].cast("Q"),
            row=self.client_index * _MAX_ENGINES,
            clients=self.client_count,
        )
        print("[prefix_routing] pid %d API server %d/%d routing table %s"
              % (os.getpid(), self.client_index, self.client_count, where), flush=True)
        return _st

    def _get(tab, key):
        mask = len(tab) - 1
        i = key & mask
        for _ in range(_PROBES):
            ent = tab[i]
            if ent == 0:
                return None
            if ent >> 8 == key:
                return (ent & 0xFF) - 1
            i = (i + 1) & mask
        return None

    def _put(tab, key, val):
        mask = len(tab) - 1
        new = (key << 8) | (val + 1)
        i = key & mask
        for _ in range(_PROBES):
            ent = tab[i]
            if ent == 0 or ent >> 8 == key:
                if ent != new:
                    tab[i] = new
                return
            i = (i + 1) & mask
        tab[key & mask] = new  # neighbourhood full: evict the home slot

    def _session_key(sid):
        return int.from_bytes(hashlib.blake2b(str(sid).encode(), digest_size=7).digest(), "little")

    def _engine_ctx(st, i):
        ctx = st["ctx"]
        return sum(ctx[r * _MAX_ENGINES + i] for r in range(st["clients"]))

    def _active(self, i):
        eng = self.core_engines[i]
        waiting, running, kv_usage = self.lb_engines[i]
        return max(self.client_count * self.engine_inflight[eng], waiting + running)

    def _least_loaded(self, num_engines):
        st = _state(self)
        used, last = st["used"], st["last"]

        def _engine_key(i):
            waiting, running, kv_usage = self.lb_engines[i]
            active = _active(self, i)
            if _context_load:
                return (_engine_ctx(st, i), active, used[i], last[i], kv_usage, i)
            return (active, used[i], last[i], kv_usage, i)

        idx = min(range(num_engines), key=_engine_key)
        if not used[idx]:
            used[idx] = 1
        return idx

    def _prefix_hashes(tokens):
        n_tok = min(len(tokens), 262144)
        num_blocks = n_tok // _BLOCK_SIZE
        raw = array.array("I", tokens[:num_blocks * _BLOCK_SIZE]).tobytes()
        b_hashes = []
        h = 0
        block_bytes = _BLOCK_SIZE * 4
        for i in range(num_blocks):
            h = zlib.crc32(raw[i * block_bytes : (i + 1) * block_bytes], h)
            b_hashes.append(h)
        return b_hashes

    def _prefix_owner(self, b_hashes, num_engines):
        blk = _state(self)["blk"]
        for i in range(len(b_hashes) - 1, _MIN_MATCH_BLOCKS - 2, -1):
            owner = _get(blk, b_hashes[i])
            if owner is not None:
                return owner % num_engines
        return None

    def _register(self, b_hashes, idx):
        blk = _state(self)["blk"]
        for bh in b_hashes:
            _put(blk, bh, idx)

    def _track_context(self, request, chosen):
        if not _context_load:
            return
        st = _state(self)
        idx = self.core_engines.index(chosen)
        n = len(getattr(request, "prompt_token_ids", None) or ())
        _req_ctx[request.request_id] = (idx, n)
        st["ctx"][st["row"] + idx] += n

    def prefix_aware_get_core_engine(self, request):
        chosen = _route(self, request)
        _track_context(self, request, chosen)
        return chosen

    orig_process_outputs = DPLBAsyncMPClient.process_engine_outputs

    async def context_load_process_outputs(self, outputs):
        for req_id in outputs.finished_requests or ():
            entry = _req_ctx.pop(req_id, None)
            if entry is not None:
                st = _state(self)
                st["ctx"][st["row"] + entry[0]] -= entry[1]
        await orig_process_outputs(self, outputs)

    def _dispatch(self, request, idx, now):
        _state(self)["last"][idx] = now
        chosen = self.core_engines[idx]
        self.lb_engines[idx][0] += self.client_count
        self.reqs_in_flight[request.request_id] = chosen
        self.engine_inflight[chosen] += 1
        return chosen

    def _route(self, request):
        if getattr(request, "data_parallel_rank", None) is not None:
            return orig_get(self, request)

        num_engines = len(self.core_engines)
        if num_engines <= 1:
            return orig_get(self, request)

        now = time.monotonic()
        tokens = getattr(request, "prompt_token_ids", None)

        # Strategy 1: Session ID affinity (X-Session-ID header / AIPERF session affinity)
        sid = getattr(request, "session_id", None)
        if sid:
            sess = _state(self)["sess"]
            skey = _session_key(sid)
            b_hashes = _prefix_hashes(tokens) if (_affinity and tokens) else []
            candidate_idx = _get(sess, skey)
            if candidate_idx is not None:
                candidate_idx %= num_engines
            else:
                if b_hashes:
                    owner = _prefix_owner(self, b_hashes, num_engines)
                    if owner is not None:
                        mean = sum(_active(self, i) for i in range(num_engines)) / num_engines
                        if _active(self, owner) <= mean + _affinity_guard:
                            candidate_idx = owner
                if candidate_idx is None:
                    candidate_idx = _least_loaded(self, num_engines)
                _put(sess, skey, candidate_idx)
            if b_hashes:
                _register(self, b_hashes, candidate_idx)
            return _dispatch(self, request, candidate_idx, now)

        # Strategy 2: Cumulative Fixed-Block Radix Prefix Routing (Token-level prefix match)
        if tokens and len(tokens) >= (_BLOCK_SIZE * _MIN_MATCH_BLOCKS):
            b_hashes = _prefix_hashes(tokens)
            candidate_idx = _prefix_owner(self, b_hashes, num_engines)
            if candidate_idx is None:
                candidate_idx = _least_loaded(self, num_engines)
            _register(self, b_hashes, candidate_idx)
            return _dispatch(self, request, candidate_idx, now)

        return orig_get(self, request)

    DPLBAsyncMPClient.get_core_engine_for_request = prefix_aware_get_core_engine
    if _context_load:
        DPLBAsyncMPClient.process_engine_outputs = staticmethod(context_load_process_outputs)
    print("[prefix_routing] Enabled Trajectory & Prefix-Aware DP Router (SessionAffinity + Block-Radix)"
          + (" with new-session prefix affinity (guard %g)" % _affinity_guard if _affinity else "")
          + (", new sessions by in-flight prompt tokens" if _context_load else "")
          + "; routing state shared across API servers (pid %d)" % os.getpid(), flush=True)

_install_prefix_aware_routing()
