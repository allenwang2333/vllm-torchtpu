"""Minimal text-only Kimi-K3 XTML prompt rendering."""

THINKING_EFFORT = (
    "`thinking_effort` guides on how much to think in your thinking channel "
    "(not including the response channel), supported values include `low`, "
    "`medium`, `high`, and `max`.\n")


def render_prompt(content: str, thinking_effort: str = "low") -> str:
    if thinking_effort not in {"low", "medium", "high", "max"}:
        raise ValueError(f"unsupported thinking effort: {thinking_effort}")
    effort_message = (
        THINKING_EFFORT +
        f"Now the system is invoked with `thinking_effort={thinking_effort}`.")
    return ('<|open|>message role="system" type="thinking-effort"<|sep|>' +
            effort_message + "<|close|>message<|sep|><|end_of_msg|>" +
            '<|open|>message role="user"<|sep|>' + content +
            "<|close|>message<|sep|><|end_of_msg|>" +
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')
