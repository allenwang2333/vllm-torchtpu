#!/usr/bin/env bash
# Run the GDN v3 validity checks on one arm: validity.sh <arm-root>
# Prints perfkit's one JSON value on stdout. The pinned jax/libtpu venv is
# built once and cached; rebuilt when the pins change.
set -euo pipefail

arm=${1:?usage: validity.sh <arm-root>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pins=(jax==0.10.2 jaxlib==0.10.2 libtpu==0.0.47 numpy==2.5.4 ml_dtypes==0.6.0 scipy==1.18.1)
venv=${XDG_CACHE_HOME:-$HOME/.cache}/qwen35-gdn-validity-venv

mkdir -p "$(dirname "$venv")"
exec 9>"$venv.lock"
flock 9
if [[ "$(cat "$venv/.pins" 2>/dev/null)" != "${pins[*]}" ]]; then
  rm -rf "$venv"
  python3 -m venv "$venv" >&2
  "$venv/bin/pip" install --quiet "${pins[@]}" >&2
  echo "${pins[*]}" > "$venv/.pins"
fi
flock -u 9

exec perfkit validity run --adapter "$here/gdn_v3_validity.py" --arm "$arm" \
  --python "$venv/bin/python"
