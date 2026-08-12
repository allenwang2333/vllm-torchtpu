import os
import re
from typing import Any, Dict, Tuple

_STANDARD_KEYS = frozenset({
    "host_tracer_level",
    "device_tracer_level",
    "python_tracer_level",
})

_ALLOWED_VALUE_CHARS = re.compile(r'^[a-zA-Z0-9_./,:\-\s]*$')
_OPTION_PATTERN = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$')


def _parse_option_value(key: str, val: str) -> Any:
    val_lower = val.lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    try:
        return int(val)
    except ValueError:
        if not _ALLOWED_VALUE_CHARS.match(val):
            raise ValueError(f"Invalid characters in option value '{val}' "
                             f"for key '{key}'")
        return val


def parse_profile_options(
        profile_prefix: str | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not profile_prefix or not profile_prefix.strip():
        return {}, {}

    if not profile_prefix.isascii():
        raise ValueError("profile_prefix contains non-ASCII characters")

    standard_opts = {}
    advanced_opts = {}

    parts = profile_prefix.split(';')

    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = _OPTION_PATTERN.match(part)
        if not match:
            raise ValueError(f"Invalid profile option format in '{part}'. "
                             "Expected 'key:value' format.")
        key = match.group(1)
        val = match.group(2)

        parsed_val = _parse_option_value(key, val)

        if key in _STANDARD_KEYS:
            standard_opts[key] = parsed_val
        else:
            advanced_opts[key] = parsed_val

    return standard_opts, advanced_opts


def resolve_profile_dir_and_opts(
        base_dir: str, profile_prefix: str | None
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Resolves the profile_dir and extracts standard and advanced tracing options."""
    standard_opts, advanced_opts = {}, {}
    profile_dir = base_dir

    if profile_prefix:
        if ":" in profile_prefix or ";" in profile_prefix:
            standard_opts, advanced_opts = parse_profile_options(
                profile_prefix)
        else:
            profile_dir = os.path.join(base_dir, profile_prefix)

    return profile_dir, standard_opts, advanced_opts
