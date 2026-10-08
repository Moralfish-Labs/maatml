"""Chat rows with tools: message normalization, rendering, span location.

A chat row is ``messages`` plus optional ``tools`` (OpenAI-style function
specs, passed to the chat template as ``tools=``) and ``chat_template_kwargs``
(template variables such as ``enable_thinking``). Assistant turns may carry
``tool_calls``; ``tool`` messages carry the results.

The supervised span of an assistant turn is located on one full render of the
conversation. Templates do not render a turn the same way in every position:
Qwen3 writes an empty ``<think></think>`` block for the final assistant turn
and omits it for earlier ones, so subtracting independently rendered prefixes
shifts every non-final span. :func:`assistant_char_spans` instead renders each
turn alone to learn its body (content, tool-call blocks, closing tokens) and
then finds that body in the full render right after the turn header, where the
header is the generation prompt or a prefix of it. Anything that does not line
up raises :class:`ChatSpanError` instead of training on shifted labels.

Torch-free: the predictor and the prepare-time validator import it too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

CHAT_ROLES = ("system", "user", "assistant", "tool")

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}

# Template variables maatml passes unless a row overrides them. Thinking off is
# what the causal SFT trainers have always rendered with.
DEFAULT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

_TOOL_CALL_RX = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_EMPTY_THINK_RX = re.compile(r"^\s*<think>\s*</think>\s*", re.S)


class ChatSpanError(ValueError):
    """An assistant turn could not be located exactly in the full render."""


@dataclass(frozen=True)
class AssistantSpan:
    """Character span of one assistant turn's supervised body in the render."""

    message_index: int
    header_start: int
    start: int
    end: int


@dataclass
class ParsedAssistant:
    """A generated assistant turn split into prose and structured calls."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Row shape
# ---------------------------------------------------------------------------


def _normalize_tool_call(call: Any) -> Optional[dict[str, Any]]:
    if not isinstance(call, dict):
        return None
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = fn.get("arguments", {})
    out: dict[str, Any] = {
        "type": call.get("type") or "function",
        "function": {"name": name, "arguments": arguments},
    }
    if call.get("id") is not None:
        out["id"] = call["id"]
    return out


def normalize_messages(raw: Any) -> list[dict[str, Any]]:
    """Canonical messages for the chat template.

    Plain turns keep exactly ``{"role", "content"}`` (what the trainer always
    rendered). Assistant turns keep ``tool_calls`` and ``reasoning_content``
    when present, and ``tool`` turns keep ``name`` and ``tool_call_id``.
    Unknown roles are skipped, as before.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = _ROLE_ALIASES.get(str(msg.get("role") or msg.get("from") or ""))
        if role is None:
            continue
        content = msg.get("content", msg.get("value", ""))
        item: dict[str, Any] = {"role": role, "content": "" if content is None else str(content)}
        if role == "assistant":
            calls = [c for c in map(_normalize_tool_call, msg.get("tool_calls") or []) if c]
            if calls:
                item["tool_calls"] = calls
            if isinstance(msg.get("reasoning_content"), str):
                item["reasoning_content"] = msg["reasoning_content"]
        elif role == "tool":
            for key in ("name", "tool_call_id"):
                if msg.get(key) is not None:
                    item[key] = str(msg[key])
        out.append(item)
    return out


def chat_template_options(sample: dict) -> tuple[Optional[list[dict]], dict[str, Any]]:
    """``(tools, template_kwargs)`` for a row: defaults overridden per row."""
    tools = sample.get("tools")
    tools = list(tools) if isinstance(tools, list) and tools else None
    kwargs = dict(DEFAULT_TEMPLATE_KWARGS)
    row_kwargs = sample.get("chat_template_kwargs")
    if isinstance(row_kwargs, dict):
        kwargs.update(row_kwargs)
    return tools, kwargs


def has_tool_features(messages: list[dict[str, Any]], tools: Optional[list] = None) -> bool:
    return bool(tools) or any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)


def validate_chat_row(row: dict) -> list[str]:
    """Structural problems in a chat row's tool fields (empty list = fine).

    Only the tool-call extension is checked, so rows that trained before keep
    preparing exactly as they did.
    """
    problems: list[str] = []
    tools = row.get("tools")
    tool_names: set[str] = set()
    if tools is not None:
        if not isinstance(tools, list):
            problems.append("tools must be a list")
        else:
            for i, spec in enumerate(tools):
                fn = spec.get("function") if isinstance(spec, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    problems.append(f"tools[{i}] must be {{type: function, function: {{name}}}}")
                    continue
                tool_names.add(fn["name"])
                params = fn.get("parameters")
                if params is not None and not isinstance(params, dict):
                    problems.append(f"tools[{i}].function.parameters must be an object")
    kwargs = row.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        problems.append("chat_template_kwargs must be an object")
    for key in ("family", "surface", "category", "scenario"):
        if row.get(key) is not None and not isinstance(row[key], str):
            problems.append(f"{key} must be a string")

    messages = row.get("messages")
    if messages is None:
        return problems
    if not isinstance(messages, list):
        return [*problems, "messages must be a list"]
    open_ids: set[str] = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = _ROLE_ALIASES.get(str(msg.get("role") or msg.get("from") or ""))
        calls = msg.get("tool_calls")
        if calls is not None and role != "assistant":
            problems.append(f"messages[{i}]: only assistant turns carry tool_calls")
        if role == "assistant" and calls is not None:
            if not isinstance(calls, list):
                problems.append(f"messages[{i}].tool_calls must be a list")
                continue
            for j, call in enumerate(calls):
                where = f"messages[{i}].tool_calls[{j}]"
                norm = _normalize_tool_call(call)
                if norm is None:
                    problems.append(f"{where} has no function name")
                    continue
                name = norm["function"]["name"]
                if tool_names and name not in tool_names:
                    problems.append(f"{where} calls {name!r}, which tools does not declare")
                args = norm["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except ValueError:
                        problems.append(f"{where}.arguments is not JSON")
                        parsed = {}
                    if not isinstance(parsed, dict):
                        problems.append(f"{where}.arguments must encode an object")
                elif not isinstance(args, dict):
                    problems.append(f"{where}.arguments must be an object or JSON string")
                if norm.get("id") is not None:
                    open_ids.add(str(norm["id"]))
        if role == "tool":
            if not isinstance(msg.get("content"), str):
                problems.append(f"messages[{i}]: tool content must be a string")
            cid = msg.get("tool_call_id")
            if cid is not None and open_ids and str(cid) not in open_ids:
                problems.append(f"messages[{i}]: tool_call_id {cid!r} answers no earlier call")
    return problems


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_chat(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
    tools: Optional[list[dict]] = None,
    template_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """Render to text with ``tools`` and template variables.

    Template variables are dropped on ``TypeError`` (older transformers that
    reject them), as the trainer always did. ``tools`` are never dropped: a
    tool row rendered without its tools would teach calls to tools the prompt
    never shows.
    """
    if not messages and not add_generation_prompt:
        return ""
    kwargs = dict(DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs)
    base: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if tools:
        base["tools"] = tools
    try:
        text = tokenizer.apply_chat_template(messages, **base, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **base)
    return text or ""


def _render_or_empty(messages, tokenizer, **kwargs) -> str:
    if not messages:
        try:
            return render_chat(messages, tokenizer, **kwargs)
        except Exception:  # noqa: BLE001  templates may reject an empty conversation
            return ""
    return render_chat(messages, tokenizer, **kwargs)


def assistant_char_spans(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    full_text: str,
    tools: Optional[list[dict]] = None,
    template_kwargs: Optional[dict[str, Any]] = None,
) -> list[AssistantSpan]:
    """Locate every assistant turn's body in ``full_text`` (see module doc).

    For turn ``i``: ``before`` is the render of ``messages[:i]`` and must be a
    prefix of the full render; ``gen`` is the generation prompt that follows
    it; ``body`` is what a render ending at turn ``i`` adds after it. In the
    full render the turn is ``gen + body`` with at most one contiguous piece
    left out before the turn's first content: Qwen3 with thinking off drops
    the empty think block from the generation prompt of a non-final turn, and
    with thinking on drops it from the body. The smallest omission wins, and
    the supervised span is whatever of ``body`` remains.
    """
    opts = {"tools": tools, "template_kwargs": template_kwargs}
    spans: list[AssistantSpan] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        before = _render_or_empty(messages[:i], tokenizer, add_generation_prompt=False, **opts)
        prompt = _render_or_empty(messages[:i], tokenizer, add_generation_prompt=True, **opts)
        through = render_chat(messages[: i + 1], tokenizer, add_generation_prompt=False, **opts)
        if not full_text.startswith(before):
            raise ChatSpanError(f"turn {i}: the render of the turns before it is not a prefix")
        if not prompt.startswith(before):
            raise ChatSpanError(f"turn {i}: the generation prompt does not extend the prefix")
        if not through.startswith(prompt):
            raise ChatSpanError(f"turn {i}: the turn render does not extend the generation prompt")
        turn = through[len(before) :]
        gen_len = len(prompt) - len(before)
        # An omission may not reach into the turn's own content or calls.
        anchors = [turn.find(f, gen_len) for f in expected_fragments(msg)]
        limit = min([a for a in anchors if a >= 0], default=gen_len)
        at = len(before)
        span = _match_turn(full_text, at, turn, gen_len, max(limit, gen_len))
        if span is None:
            raise ChatSpanError(f"turn {i}: its body is not where the full render puts the turn")
        header_len, start_in_turn = span
        start = at + header_len
        spans.append(AssistantSpan(i, at, start, start + len(turn) - start_in_turn))
    return spans


def _match_turn(
    full_text: str, at: int, turn: str, gen_len: int, limit: int
) -> Optional[tuple[int, int]]:
    """Find ``turn[:a] + turn[k:]`` at ``at`` with ``a <= gen_len`` and
    ``a <= k <= limit``.

    Candidates whose omitted window touches the boundary between generation
    prompt and body (``k == gen_len``: thinking off; ``a == gen_len``: thinking
    on) are tried first, then the smallest omission. The window can otherwise
    slide where the characters around it coincide (``<t`` opens both
    ``<think>`` and ``<tool_call>``), which would clip or pad the target.

    Returns ``(header_chars_in_full, body_start_in_turn)``.
    """
    candidates = sorted(
        (
            (k != gen_len and a != gen_len, k - a, -a, a, k)
            for a in range(gen_len + 1)
            for k in range(a, limit + 1)
        ),
    )
    for _unaligned, _omitted, _neg, a, k in candidates:
        if full_text.startswith(turn[:a], at) and full_text.startswith(turn[k:], at + a):
            body_start = max(k, gen_len)
            return a + (body_start - k), body_start
    return None


def expected_fragments(message: dict[str, Any]) -> list[str]:
    """Text an assistant turn's supervised span must contain verbatim."""
    out: list[str] = []
    content = str(message.get("content") or "").strip()
    if content:
        out.append(content)
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name"):
            out.append(str(fn["name"]))
        if isinstance(fn.get("arguments"), str) and fn["arguments"]:
            out.append(fn["arguments"])
    return out


# ---------------------------------------------------------------------------
# Generated output
# ---------------------------------------------------------------------------


def parse_tool_calls(text: str) -> ParsedAssistant:
    """Split generated text into prose and ``<tool_call>`` JSON blocks.

    Each block is ``{"name": ..., "arguments": {...}}`` (Hermes / Qwen form).
    Calls come back in the message shape rows use, with ``arguments`` as a
    parsed object. A leading empty think block is ignored; an unclosed block or
    one that is not a JSON object is reported in ``errors``.
    """
    text = _EMPTY_THINK_RX.sub("", text or "")
    parsed = ParsedAssistant(content="")
    for n, block in enumerate(_TOOL_CALL_RX.findall(text)):
        try:
            obj = json.loads(block)
        except ValueError as exc:
            parsed.errors.append(f"tool_call[{n}] is not JSON: {exc}")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            parsed.errors.append(f"tool_call[{n}] has no name")
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                parsed.errors.append(f"tool_call[{n}] arguments are not JSON")
        parsed.tool_calls.append(
            {"type": "function", "function": {"name": obj["name"], "arguments": args}}
        )
    rest = _TOOL_CALL_RX.sub("", text)
    if "<tool_call>" in rest or "</tool_call>" in rest:
        parsed.errors.append("unterminated <tool_call> block")
    parsed.content = rest.strip()
    return parsed
