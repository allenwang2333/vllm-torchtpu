#!/bin/bash
# Bring a multimodal Kimi-K3 up and put one image through it.
#
# Nothing in the tree covers K3's vision path end to end: every K3 recipe
# serves --language-model-only. This starts a server that does NOT, then posts
# a synthetic image to /v1/chat/completions and reports whether the encoder,
# projector and scatter survived the round trip. It does not judge the text --
# with the sliced checkpoint there is no text worth judging.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="kimi-k3-sliced8-tp8-mm"; RESULTS_DIR=""; PORT=8000
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)      CONFIG_NAME="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
RESULTS_DIR="${RESULTS_DIR:-/tmp/k3_mm_$CONFIG_NAME}"
mkdir -p "$RESULTS_DIR"

trap 'echo "--- cleanup"; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM
bash "$SCRIPT_DIR/cleanup_server.sh"

echo "--- server + text rows ($CONFIG_NAME)"
"$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" --results-dir "$RESULTS_DIR" \
  --keep-alive --port "$PORT" || { echo "K3MM stage=serve status=FAIL"; exit 1; }

MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$RESULTS_DIR/config.json")

python3 - "$MODEL" "$PORT" "$RESULTS_DIR" <<'PY'
import base64, io, json, sys, time, urllib.request

model, port, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def png(w, h):
    from PIL import Image
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def ask(sizes, max_tokens=16):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + png(*s)}}
               for s in sizes]
    content.append({"type": "text", "text": "Describe the image."})
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read()), time.time() - t0

fail = 0
# One image, then two, then a different aspect ratio: NaViT buckets on size,
# so a single shape proves less than it looks.
for label, sizes in (("one_224", [(224, 224)]),
                     ("two_224", [(224, 224), (224, 224)]),
                     ("one_448x336", [(448, 336)])):
    try:
        resp, dt = ask(sizes)
        usage = resp.get("usage", {})
        text = resp["choices"][0]["message"].get("content") or ""
        print(f"K3MM case={label} status=OK latency_s={dt:.2f} "
              f"prompt_tokens={usage.get('prompt_tokens')} "
              f"completion_tokens={usage.get('completion_tokens')} "
              f"text={text[:60]!r}")
        json.dump(resp, open(f"{out_dir}/mm_{label}.json", "w"), indent=2)
    except Exception as e:
        fail = 1
        body = getattr(e, "read", lambda: b"")()
        print(f"K3MM case={label} status=FAIL err={type(e).__name__}: {e} "
              f"body={body[:400]!r}")
sys.exit(fail)
PY
