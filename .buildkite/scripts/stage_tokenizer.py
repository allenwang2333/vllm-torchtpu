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
"""Copy a checkpoint's tokenizer files out of GCS into a local directory.

`vllm serve` accepts a gs:// model path (runai_streamer reads it directly), but
`vllm bench serve` does not: its `--tokenizer` goes through huggingface_hub,
which rejects anything that is not a `namespace/repo` id --

    HFValidationError: Repo id must be in the form 'repo_name' or
    'namespace/repo_name': 'gs://tpu-commons-ci/moonshootai/kimi/linear-48b'

and `--model` is only the served-name alias, which need not exist on the Hub at
all. So the benchmark client needs the tokenizer on local disk. Everything
except the weights is a couple of MB.

Usage:
    python3 .buildkite/scripts/stage_tokenizer.py gs://bucket/prefix /tmp/tok
"""

import os
import sys

# Weights are what make a checkpoint big and the tokenizer does not need them.
_SKIP_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".npz", ".gguf")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} gs://bucket/prefix /local/dir", file=sys.stderr)
        return 2

    src, dest = argv[1], argv[2]
    if not src.startswith("gs://"):
        print(f"[stage-tokenizer] ERROR: source must be a gs:// URI, got {src!r}",
              file=sys.stderr)
        return 2

    bucket_name, _, prefix = src[len("gs://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"

    from google.cloud import storage

    os.makedirs(dest, exist_ok=True)
    client = storage.Client()

    copied = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        name = blob.name[len(prefix):]
        # Only the top level: nested dirs are checkpoint shards, not tokenizer.
        if not name or "/" in name or name.endswith(_SKIP_SUFFIXES):
            continue
        target = os.path.join(dest, name)
        blob.download_to_filename(target)
        print(f"[stage-tokenizer] {blob.name} -> {target} "
              f"({blob.size or 0} bytes)")
        copied += 1

    if copied == 0:
        print(f"[stage-tokenizer] ERROR: no tokenizer files found under {src}. "
              f"Checked bucket={bucket_name!r} prefix={prefix!r}; every object "
              f"was either nested or a weights file {_SKIP_SUFFIXES}.",
              file=sys.stderr)
        return 1

    print(f"[stage-tokenizer] staged {copied} file(s) into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
