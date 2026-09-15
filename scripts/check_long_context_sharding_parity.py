#!/usr/bin/env python3
"""Long-context greedy generation-equivalence probe.

Runs entirely client-side against an OpenAI-compatible vLLM server.
Deterministic by construction: prompts depend only on SEED and on the
server's tokenizer (identical across arms), never on wall clock,
dict order, or scheduling.

Emits one `GENCHK_RESULT <json>` line per completed request.

Why this exists
---------------
Accuracy evals are not enough to clear a change to the KV page layout.
lm-eval's MMLU tasks cap the context at 2047 tokens, which at
``--block-size=256`` touches at most 8 KV blocks; a serving benchmark at
``--random-input-len=8192`` touches 32. A page-table striding bug corrupts
*old* tokens specifically, so it is invisible in the first regime and fatal
in the second. This probe plants facts at known depths in an ~8k-token
context and checks they are still retrievable with 24-31 later blocks
resident.

Do not read the exact-match rate as the result
----------------------------------------------
Measured on tpu7x, this server is not run-to-run reproducible at fixed
sharding: re-running one unchanged configuration at a different concurrency
decorrelates the token sequence as completely as changing the sharding does
(0/32 exact match within-arm, against 0-1/32 cross-arm). Exact token equality
therefore fails against the reference itself and cannot discriminate. Always
run the within-arm control before reading a cross-arm number, and rest any
verdict on planted-fact retrieval and coherence instead.

Validated 2026-09-15 on Qwen3-Coder-30B-A3B-Instruct-FP8, CUSTOM
(batched-RPA) backend, fp8 KV, block_size 256, comparing TP=8/DP=1 against
TP=2/DP=4 -- jobs j-59e9c39c-3888-4c3a-8436 and j-47ba1b1e-0a5e-45a3-8889,
both Succeeded, 100% planted-fact retrieval on both arms. See
gs://vllm-opt-experiments/Qwen3-Coder-30B-A3B-Instruct-FP8/
20260915-vdziuba-fp8-kv-layout-seq-along-lane/experiment.md
"""
import argparse, hashlib, json, random, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

SEED = 20260915
TARGET_RAW_TOKENS = 8060      # raw text; chat template adds ~20 more
TOL = 90
POOL_LINES = 620
MAX_TOKENS = 192
REQ_TIMEOUT = 1200

# ------------------------------------------------------------------ http

def post(url, payload, timeout=REQ_TIMEOUT):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

# ------------------------------------------------------- prompt material

NOUNS = ["turbine","aquifer","lattice","beacon","conduit","ledger","cistern",
         "gantry","paddock","reservoir","manifold","trellis","bastion","culvert",
         "spillway","armature","granary","pylon","caisson","bulwark"]
ADJS = ["corroded","luminous","dormant","brackish","temperate","fractured",
        "reinforced","submerged","weathered","calibrated","insulated","derelict",
        "provisional","saturated","magnetised","annotated"]
VERBS = ["surveyed","realigned","catalogued","flushed","re-seated","annotated",
         "decommissioned","inspected","re-tensioned","sealed","mapped","purged"]
PLACES = ["North Traverse","Kestrel Hollow","Vantry Flats","Elder Shoal",
          "Marrow Ridge","Cold Fenn","Dunmere Basin","Saltpan Reach",
          "Ashgrove Cut","Pell Narrows","Harrow Deep","Brindle Wash"]
WEATHER = ["low fog","sleet","still air","a steady crosswind","light drizzle",
           "hard frost","overcast haze","gusting squalls"]

SENSORS = ["VX-ORIOLE","TR-MARLIN","QD-HALCYON","ZP-CONDOR","LM-KESTREL",
           "NF-BITTERN","GS-PLOVER","RA-SHRIKE","BC-AVOCET","WT-CURLEW",
           "DH-GANNET","XE-MERLIN"]


def make_line(rnd, i):
    return ("[E%04d] Survey team %s the %s %s at %s under %s; "
            "drift %d.%02d mm, bay %d, shift %s." % (
                i, rnd.choice(VERBS), rnd.choice(ADJS), rnd.choice(NOUNS),
                rnd.choice(PLACES), rnd.choice(WEATHER),
                rnd.randrange(0, 40), rnd.randrange(0, 100),
                rnd.randrange(1, 60), rnd.choice("ABCD")))


def needle_sentence(sensor, value):
    return ("[CALIBRATION NOTICE] The calibration constant for sensor %s "
            "is %s. This value supersedes all earlier notices." % (sensor, value))


def build_pool(rnd):
    return [make_line(rnd, i + 1) for i in range(POOL_LINES)]


def assemble(pool, k, needles, tail):
    """needles: list of (depth_fraction, sentence). Deterministic ordering."""
    lines = list(pool[:k])
    for depth, sent in sorted(needles, key=lambda t: -t[0]):
        pos = min(len(lines), max(0, int(round(depth * k))))
        lines.insert(pos, sent)
    head = ("STATION DOSSIER -- CONSOLIDATED FIELD LOG\n"
            "The following entries are an unabridged archive. Read all of it.\n\n")
    return head + "\n".join(lines) + "\n\n" + tail

# ------------------------------------------------------ length calibration

def raw_token_count(base, model, text):
    r = post(base + "/tokenize", {"model": model, "prompt": text}, timeout=300)
    return int(r["count"])


def calibrate(base, model, pool, needles, tail):
    """Deterministic bisection on line count k to hit TARGET_RAW_TOKENS."""
    lo, hi = 1, POOL_LINES
    best_k, best_n = None, None
    for _ in range(11):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        n = raw_token_count(base, model, assemble(pool, mid, needles, tail))
        if best_k is None or abs(n - TARGET_RAW_TOKENS) < abs(best_n - TARGET_RAW_TOKENS):
            best_k, best_n = mid, n
        if n < TARGET_RAW_TOKENS:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_k, best_n

# ----------------------------------------------------------- prompt specs

def build_specs():
    rnd = random.Random(SEED)
    specs = []

    # 12 x single needle in the FIRST QUARTER of the context
    for i in range(12):
        depth = 0.03 + 0.018 * i          # 0.030 .. 0.228
        sensor = SENSORS[i % len(SENSORS)]
        value = "K-%d-%s%s" % (rnd.randrange(1000, 9999),
                               rnd.choice("QRTVXZ"), rnd.choice("BDFHLN"))
        tail = ("QUESTION: What is the calibration constant for sensor %s?\n"
                "Answer with the exact constant on the first line. Then, in two or "
                "three sentences, describe where in the dossier the calibration "
                "notice appeared and summarise the entry that immediately "
                "followed it." % sensor)
        specs.append(dict(id="P%02d" % len(specs), kind="needle_early",
                          depths=[round(depth, 3)],
                          needles=[[sensor, value, round(depth, 3)]],
                          tail=tail, rnd_seed=rnd.randrange(1 << 30)))

    # 6 x two needles: one early, one late -- tests retention of OLD pages
    for i in range(6):
        d_early = 0.05 + 0.03 * i
        d_late = 0.80 + 0.025 * i
        s1 = SENSORS[(i + 3) % len(SENSORS)]
        s2 = SENSORS[(i + 8) % len(SENSORS)]
        v1 = "K-%d-%s%s" % (rnd.randrange(1000, 9999), rnd.choice("QRTVXZ"), rnd.choice("BDFHLN"))
        v2 = "K-%d-%s%s" % (rnd.randrange(1000, 9999), rnd.choice("QRTVXZ"), rnd.choice("BDFHLN"))
        tail = ("QUESTION: Two calibration notices appear in this dossier.\n"
                "Line 1: the calibration constant for sensor %s.\n"
                "Line 2: the calibration constant for sensor %s.\n"
                "Then state which of the two notices appeared earlier in the "
                "dossier and explain briefly how you can tell." % (s1, s2))
        specs.append(dict(id="P%02d" % len(specs), kind="needle_pair",
                          depths=[round(d_early, 3), round(d_late, 3)],
                          needles=[[s1, v1, round(d_early, 3)],
                                   [s2, v2, round(d_late, 3)]],
                          tail=tail, rnd_seed=rnd.randrange(1 << 30)))

    # 6 x control needles at mid / late depth
    for i in range(6):
        depth = [0.45, 0.55, 0.65, 0.75, 0.88, 0.95][i]
        sensor = SENSORS[(i + 5) % len(SENSORS)]
        value = "K-%d-%s%s" % (rnd.randrange(1000, 9999), rnd.choice("QRTVXZ"), rnd.choice("BDFHLN"))
        tail = ("QUESTION: What is the calibration constant for sensor %s?\n"
                "Answer with the exact constant on the first line, then describe "
                "in two or three sentences the entries surrounding the notice."
                % sensor)
        specs.append(dict(id="P%02d" % len(specs), kind="needle_late",
                          depths=[depth], needles=[[sensor, value, depth]],
                          tail=tail, rnd_seed=rnd.randrange(1 << 30)))

    # 8 x free-form, no planted fact
    freeform = [
        "TASK: Write a concise operational summary of the dossier above in one paragraph.",
        "TASK: List the five field conditions that recur most often in the entries above.",
        "TASK: Continue the field log with three further entries in exactly the same format.",
        "TASK: Draft a short handover note for the incoming survey team.",
        "TASK: Describe the earliest entries in the dossier and how they differ from the latest ones.",
        "TASK: Identify any structural hazards implied by the entries and justify each.",
        "TASK: Write a two-paragraph narrative account of a single shift at this station.",
        "TASK: Propose a maintenance schedule derived from the entries above.",
    ]
    for i, tail in enumerate(freeform):
        specs.append(dict(id="P%02d" % len(specs), kind="freeform",
                          depths=[], needles=[], tail=tail,
                          rnd_seed=rnd.randrange(1 << 30)))
    return specs

# --------------------------------------------------------------- requests

VARIANTS = [
    ("full",      dict(logprobs=True, top_logprobs=0, seed=1234, ignore_eos=False)),
    ("no_toplp",  dict(logprobs=True, seed=1234, ignore_eos=False)),
    ("no_lp",     dict(seed=1234, ignore_eos=False)),
    ("bare",      dict(seed=1234)),
    ("minimal",   dict()),
]
VARIANT = {"name": None, "extra": None}


def chat_payload(model, prompt, extra):
    p = {"model": model,
         "messages": [{"role": "user", "content": prompt}],
         "max_tokens": MAX_TOKENS,
         "temperature": 0.0, "top_p": 1.0, "stream": False}
    p.update(extra)
    return p


def select_variant(base, model):
    """Pick the richest payload the server accepts. Done once, before any real
    request, so both arms provably use the same request shape."""
    for name, extra in VARIANTS:
        try:
            r = post(base + "/v1/chat/completions",
                     chat_payload(model, "Reply with the single word: ok.", extra),
                     timeout=300)
            _ = r["choices"][0]["message"]["content"]
            VARIANT["name"], VARIANT["extra"] = name, extra
            has_lp = bool((r["choices"][0].get("logprobs") or {}).get("content"))
            print("GENCHK_INFO payload_variant=%s logprobs_returned=%s extra=%s"
                  % (name, has_lp, json.dumps(extra, sort_keys=True)), flush=True)
            return
        except urllib.error.HTTPError as e:
            print("GENCHK_INFO variant %s rejected: HTTP %s %s"
                  % (name, e.code, e.read().decode()[:200]), flush=True)
        except Exception as e:
            print("GENCHK_INFO variant %s failed: %s: %s" % (name, type(e).__name__, e), flush=True)
    raise SystemExit("no usable chat/completions payload variant")


def run_one(base, model, spec, prompt, raw_n, pass_name):
    t0 = time.time()
    last = None
    for attempt in (1, 2):
        try:
            r = post(base + "/v1/chat/completions",
                     chat_payload(model, prompt, VARIANT["extra"]))
            last = None
            break
        except urllib.error.HTTPError as e:
            last = "HTTP %s: %s" % (e.code, e.read().decode()[:600])
            break                      # a 4xx will not fix itself
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
            time.sleep(5)
    if last is not None:
        return dict(id=spec["id"], pass_name=pass_name, error=last)
    ch = r["choices"][0]
    text = ch["message"]["content"] or ""
    toks, lps = [], []
    lp = ch.get("logprobs") or {}
    for item in (lp.get("content") or []):
        toks.append(item.get("token"))
        v = item.get("logprob")
        lps.append(None if v is None else round(float(v), 3))
    usage = r.get("usage", {})
    found = []
    for sensor, value, depth in spec["needles"]:
        found.append([sensor, value, depth, value in text])
    return dict(id=spec["id"], pass_name=pass_name, kind=spec["kind"],
                depths=spec["depths"],
                prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                prompt_chars=len(prompt), raw_tokens=raw_n,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=ch.get("finish_reason"),
                needles=found, n_tokens=len(toks), tokens=toks, logprobs=lps,
                text=text, wall_s=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--arm", default="unknown")
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    base = "http://%s:%s" % (a.host, a.port)

    models = get(base + "/v1/models")
    model = models["data"][0]["id"]
    print("GENCHK_INFO arm=%s base=%s model=%s" % (a.arm, base, model), flush=True)
    select_variant(base, model)

    specs = build_specs()
    print("GENCHK_INFO n_specs=%d target_raw_tokens=%d max_tokens=%d seed=%d"
          % (len(specs), TARGET_RAW_TOKENS, MAX_TOKENS, SEED), flush=True)

    # ---- build every prompt up front (deterministic, server-tokenizer based)
    built = []
    for spec in specs:
        rnd = random.Random(spec["rnd_seed"])
        pool = build_pool(rnd)
        needles = [(d, needle_sentence(s, v)) for s, v, d in spec["needles"]]
        k, n = calibrate(base, model, pool, needles, spec["tail"])
        prompt = assemble(pool, k, needles, spec["tail"])
        n_exact = raw_token_count(base, model, prompt)
        built.append((spec, prompt, n_exact))
        print("GENCHK_PROMPT %s" % json.dumps(dict(
            id=spec["id"], kind=spec["kind"], k_lines=k, raw_tokens=n_exact,
            chars=len(prompt), depths=spec["depths"],
            sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            in_tolerance=abs(n_exact - TARGET_RAW_TOKENS) <= TOL)), flush=True)

    allsha = hashlib.sha256("".join(p for _, p, _ in built).encode()).hexdigest()
    print("GENCHK_INFO all_prompts_sha256=%s" % allsha, flush=True)

    # ---- pass A: strictly sequential (concurrency 1)
    for pass_name, conc in (("seq", 1), ("conc%d" % a.conc, a.conc)):
        print("GENCHK_INFO pass=%s concurrency=%d start" % (pass_name, conc), flush=True)
        t0 = time.time()
        if conc == 1:
            results = [run_one(base, model, s, p, n, pass_name) for s, p, n in built]
        else:
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [ex.submit(run_one, base, model, s, p, n, pass_name)
                        for s, p, n in built]
                results = [f.result() for f in futs]
        for r in results:
            print("GENCHK_RESULT " + json.dumps(r, ensure_ascii=False), flush=True)
        nerr = sum(1 for r in results if "error" in r)
        print("GENCHK_INFO pass=%s done wall_s=%.1f errors=%d"
              % (pass_name, time.time() - t0, nerr), flush=True)
        if nerr:
            print("GENCHK_INFO FATAL pass=%s had %d errors" % (pass_name, nerr), flush=True)
            return 1
    print("GENCHK_INFO all passes complete arm=%s" % a.arm, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
