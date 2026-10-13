"""A tool protocol carried in ordinary message text.

The upstream chat template is not involved: the tool list is rendered into the
last user turn, and a call comes back as a JSON object at the end of the
assistant's text. One renderer serves both the corpus builder and the server,
so the string trained on and the string served are the same string.

Use it when the upstream's own tool template would inject definitions the
model never saw while training. The alternative is to declare tools upstream
and let the template render them, which is the default.
"""

from __future__ import annotations

import json
from typing import Any, Optional

RULE = (
    "To call tools, end your reply with a single JSON object on its own line:\n"
    '{"calls":[{"name":"<tool>","input":{...}}]}\n'
    "Use only the tools listed in the message, and only their declared "
    "arguments. Say what you are doing in one sentence before the object. To "
    "answer without calling anything, write the answer and no object."
)

HEADING = "Tools you may call:"

TYPES = {
    "string": "str",
    "integer": "int",
    "number": "num",
    "boolean": "bool",
    "array": "list",
    "object": "obj",
}

MARKER = '{"calls"'


def render_result(body: str, *, failed: bool = False) -> str:
    """A tool result as the history carries it, in place of a ``tool`` turn."""
    return f"<tool_response>{' (error)' if failed else ''}\n{body}\n</tool_response>"


def _signature(schema: dict) -> str:
    """Required arguments first, so what must be filled is read first."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    args = []
    for name in sorted(properties, key=lambda n: (n not in required, n)):
        kind = TYPES.get((properties[name] or {}).get("type") or "string", "str")
        args.append(f"{name}:{kind}" + ("" if name in required else "?"))
    return ", ".join(args)


def render_catalogue(tools: list[dict], *, about_chars: int = 80) -> str:
    """One line per tool: name, argument names with types, first sentence."""
    lines = [HEADING]
    for tool in sorted(tools, key=lambda t: t.get("name") or ""):
        name = tool.get("name")
        if not name:
            continue
        schema = tool.get("input_schema") or tool.get("parameters") or {}
        about = (tool.get("description") or "").split(".")[0].strip()[:about_chars]
        lines.append(f"{name}({_signature(schema)}) - {about}")
    return "\n".join(lines)


def with_catalogue(text: str, catalogue: str) -> str:
    """The catalogue sits above the turn it applies to."""
    return f"{catalogue}\n\n{text}" if text else catalogue


def render_calls(calls: list[dict]) -> str:
    """The object an assistant turn ends with, as the model must emit it."""
    body = [{"name": c.get("name"), "input": c.get("input") or {}} for c in calls]
    return json.dumps({"calls": body}, ensure_ascii=False)


def parse_calls(text: str) -> tuple[str, Optional[list[dict]], Optional[str]]:
    """Prose, calls and the reason there are none.

    The reason is ``None`` when the answer carries no call object at all, and
    a message when it carries one that could not be read — a grader needs to
    tell a plain answer from a broken call, where a server does not.
    """
    at = text.rfind(MARKER)
    if at < 0:
        return text, None, None
    try:
        payload = json.loads(text[at:].strip())
    except json.JSONDecodeError as exc:
        return text, None, f"call object is not JSON: {exc}"
    calls = payload.get("calls") if isinstance(payload, dict) else None
    if not isinstance(calls, list):
        return text, None, "`calls` is not a list"
    if not calls:
        return text, None, "`calls` is empty"
    clean: list[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            return text, None, "a call is not an object"
        if not call.get("name"):
            return text, None, "a call has no `name`"
        clean.append({"name": str(call["name"]), "input": call.get("input") or {}})
    return text[:at].rstrip(), clean, None


def split_calls(text: str) -> tuple[str, Optional[list[dict]]]:
    """Prose and calls from one assistant answer.

    Returns ``(text, None)`` when the answer carries no readable call object,
    so a plain answer and a malformed object are the same case downstream.
    """
    said, calls, _ = parse_calls(text)
    return said, calls


def inline_request(body: dict) -> dict:
    """Move ``tools`` out of the request and into the last user turn."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return body
    catalogue = render_catalogue(tools)

    out: dict[str, Any] = {k: v for k, v in body.items() if k != "tools"}
    messages = [dict(m) for m in out.get("messages") or []]
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        message["content"] = _prepend(message.get("content"), catalogue)
        break
    else:
        messages.insert(0, {"role": "user", "content": catalogue})
    out["messages"] = messages
    out["system"] = _append_rule(out.get("system"))
    return out


def _prepend(content: Any, catalogue: str) -> Any:
    if isinstance(content, str):
        return with_catalogue(content, catalogue)
    if isinstance(content, list):
        return [{"type": "text", "text": catalogue}, *content]
    return catalogue


def _append_rule(system: Any) -> Any:
    if isinstance(system, list):
        return [*system, {"type": "text", "text": RULE}]
    if isinstance(system, str) and system:
        return f"{system}\n\n{RULE}"
    return RULE
