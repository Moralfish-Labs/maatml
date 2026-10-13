"""Serve a local model behind Anthropic's Messages API.

A translating proxy: it accepts ``POST /v1/messages`` in Anthropic's shape and
forwards to an OpenAI-compatible upstream — normally ``llama-server --jinja``
holding a GGUF — then turns the reply back into Anthropic's SSE frames.

    llama-server --jinja -m model.gguf --host 127.0.0.1 --port 8081 &
    maatml serve <model-dir> --server anthropic \\
        --server-option upstream=http://127.0.0.1:8081 --port 8080

Constraints the client end imposes:

* Only ``data:`` lines are read; ``event:`` lines are skipped.
* A missing ``stop_reason`` is read as truncation, so every stream carries one.
* Content blocks are positioned by ``index``, not by arrival order.
* ``cache_control`` and ``output_config`` arrive on every request and are
  dropped rather than refused.

Thinking blocks are never emitted: Anthropic rejects one handed back without
the signature it issued, and a local server has no signature to give.

``tool_style`` selects who renders tool definitions. ``native`` forwards them
to the upstream's own chat template. ``inline`` keeps them out of the request
and carries them in message text instead — see :mod:`maatml.wire.inline_tools`.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional

from . import inline_tools

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM = "http://127.0.0.1:8081"
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024  # whole-file rewrites travel in tool args
TOOL_STYLES = ("native", "inline")

# OpenAI's reason for stopping, in Anthropic's vocabulary.
STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


# ------------------------------------------------------------------ request


def _text_of(content: Any) -> str:
    """The text of a system field that may be a string or a block array."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def to_openai_messages(body: dict, tool_style: str = "native") -> list[dict]:
    """Anthropic's messages (plus its top-level system) as OpenAI chat turns.

    Under ``inline`` the history is written the way the corpus writes it: calls
    and results are text, so no upstream template renders them.
    """
    out: list[dict] = []
    inline = tool_style == "inline"

    system = _text_of(body.get("system"))
    if system:
        out.append({"role": "system", "content": system})

    for message in body.get("messages") or []:
        role = message.get("role")
        blocks = _blocks(message.get("content"))

        results = [b for b in blocks if b.get("type") == "tool_result"]
        said = "\n".join(b.get("text") or "" for b in blocks if b.get("type") == "text").strip()
        calls = [b for b in blocks if b.get("type") == "tool_use"]

        if results and inline:
            rendered = [
                inline_tools.render_result(
                    _text_of(b.get("content"))
                    if not isinstance(b.get("content"), str)
                    else b["content"],
                    failed=bool(b.get("is_error")),
                )
                for b in results
            ]
            said = "\n".join([p for p in [said, *rendered] if p])
        elif results:
            # A user turn carrying tool_result becomes one `tool` message per
            # result — OpenAI has no block array, and pairing is by id.
            for block in results:
                payload = block.get("content")
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": _text_of(payload) if not isinstance(payload, str) else payload,
                    }
                )

        if role == "assistant" and calls and inline:
            body_text = "\n".join([p for p in [said, inline_tools.render_calls(calls)] if p])
            out.append({"role": "assistant", "content": body_text})
        elif role == "assistant" and calls:
            out.append(
                {
                    "role": "assistant",
                    "content": said or None,
                    "tool_calls": [
                        {
                            "id": c.get("id"),
                            "type": "function",
                            "function": {
                                "name": c.get("name"),
                                "arguments": json.dumps(c.get("input") or {}, ensure_ascii=False),
                            },
                        }
                        for c in calls
                    ],
                }
            )
        elif said:
            out.append({"role": role or "user", "content": said})

    return out


def to_openai_tools(tools: Any) -> list[dict]:
    """`input_schema` is Anthropic's name for what OpenAI calls `parameters`."""
    if not isinstance(tools, list):
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
        if isinstance(t, dict) and t.get("name")
    ]


def to_openai(body: dict, model: str, tool_style: str = "native") -> dict:
    """The upstream request. Unknown Anthropic keys are dropped, never refused."""
    if tool_style == "inline":
        body = inline_tools.inline_request(body)
    out: dict[str, Any] = {
        "model": model,
        "messages": to_openai_messages(body, tool_style),
        "stream": True,
        # Without this a streaming reply carries no token counts, and the
        # client's context gauge reads zero.
        "stream_options": {"include_usage": True},
        # These fine-tunes are trained with thinking stripped, so a thinking
        # model is not the model that was measured. Said twice because no one
        # key reaches every upstream: llama.cpp reads the template kwarg, and
        # LM Studio ignores it and honours `reasoning_effort`. Both are dropped
        # where unrecognised, so sending both costs nothing and leaving either
        # out serves a model nobody gated.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }
    if isinstance(body.get("max_tokens"), int):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    tools = to_openai_tools(body.get("tools"))
    if tools:
        out["tools"] = tools
    # Ignored: cache_control, output_config.effort, metadata, and system beyond
    # its text. Dropping beats 400ing on a key the client always sends.
    return out


# ----------------------------------------------------------------- response


def _frame(kind: str, payload: dict) -> bytes:
    """One SSE frame, carrying both an `event:` and a `data:` line."""
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _usage(counts: dict) -> dict:
    """Anthropic's four names. The cache pair is always zero locally."""
    return {
        "input_tokens": int(counts.get("prompt_tokens") or 0),
        "output_tokens": int(counts.get("completion_tokens") or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


class Translator:
    """OpenAI streaming deltas in, Anthropic frames out.

    Holds the block cursor: text is block 0 when present, and each upstream
    tool call takes the next index. `index` is how the client positions
    blocks, so it has to be stable for the life of one reply.

    Under ``inline`` the call object arrives at the end of the text, so text is
    buffered and every block is emitted from :meth:`finish`.
    """

    def __init__(self, model: str, tool_style: str = "native") -> None:
        self.model = model
        self.inline = tool_style == "inline"
        self.opened: dict[int, str] = {}  # anthropic index -> kind
        self.tool_at: dict[int, int] = {}  # openai tool index -> anthropic index
        self.next_index = 0
        self.stop: Optional[str] = None
        self.counts: dict = {}
        self.buffer: list[str] = []
        self.thought = False

    def start(self) -> bytes:
        return _frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": _usage({}),
                },
            },
        )

    def _open(self, index: int, block: dict) -> bytes:
        return _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            },
        )

    def delta(self, chunk: dict) -> Iterator[bytes]:
        if isinstance(chunk.get("usage"), dict):
            self.counts.update(chunk["usage"])
        # llama.cpp reports its own counts under `timings` in the final chunk,
        # under different names, and only sends `usage` when asked. Read both,
        # so the gauge is right whichever upstream is in front.
        timings = chunk.get("timings")
        if isinstance(timings, dict):
            if timings.get("prompt_n") is not None:
                self.counts.setdefault("prompt_tokens", timings["prompt_n"])
            if timings.get("predicted_n") is not None:
                self.counts.setdefault("completion_tokens", timings["predicted_n"])

        for choice in chunk.get("choices") or []:
            done = choice.get("finish_reason")
            if done:
                self.stop = STOP.get(done, "end_turn")

            delta = choice.get("delta") or {}

            # Thinking is never forwarded — a thinking block needs a signature
            # this wire has none of — but it is counted, because a reply that
            # is *only* thinking means the upstream ignored every key that
            # switches it off, and that is worth saying rather than serving as
            # an empty turn.
            if delta.get("reasoning_content"):
                self.thought = True

            said = delta.get("content")
            if said and self.inline:
                self.buffer.append(said)
            elif said:
                if 0 not in self.opened:
                    self.opened[0] = "text"
                    self.next_index = max(self.next_index, 1)
                    yield self._open(0, {"type": "text", "text": ""})
                yield _frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": said},
                    },
                )

            for call in delta.get("tool_calls") or []:
                at = call.get("index", 0)
                if at not in self.tool_at:
                    index = self.next_index
                    self.next_index += 1
                    self.tool_at[at] = index
                    self.opened[index] = "tool_use"
                    fn = call.get("function") or {}
                    yield self._open(
                        index,
                        {
                            "type": "tool_use",
                            "id": call.get("id") or f"toolu_local_{index}",
                            "name": fn.get("name") or "",
                            "input": {},
                        },
                    )
                fragment = (call.get("function") or {}).get("arguments")
                if fragment:
                    yield _frame(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self.tool_at[at],
                            "delta": {"type": "input_json_delta", "partial_json": fragment},
                        },
                    )

    def _inline_blocks(self) -> Iterator[bytes]:
        said, calls = inline_tools.split_calls("".join(self.buffer))
        if said:
            index = self.next_index
            self.next_index += 1
            self.opened[index] = "text"
            yield self._open(index, {"type": "text", "text": ""})
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": said},
                },
            )
        for at, call in enumerate(calls or []):
            index = self.next_index
            self.next_index += 1
            self.opened[index] = "tool_use"
            yield self._open(
                index,
                {
                    "type": "tool_use",
                    "id": f"toolu_local_{index}",
                    "name": call["name"],
                    "input": {},
                },
            )
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(call["input"], ensure_ascii=False),
                    },
                },
            )
            self.tool_at[at] = index
        # A truncated reply keeps its own reason; the calls it carries are the
        # ones that survived, not a complete turn.
        if calls and self.stop != "max_tokens":
            self.stop = "tool_use"

    def finish(self) -> Iterator[bytes]:
        if self.inline:
            yield from self._inline_blocks()
        if not self.opened:
            # An empty turn is the one outcome a client cannot act on: NOON
            # drops an answer with no text, so the ask reads as never asked.
            # Say which of the two it was.
            why = (
                "The upstream replied with reasoning only and no content, so "
                "thinking is still on there; these adapters were gated with it "
                "off. Nothing was run and nothing was written."
                if self.thought
                else "The upstream replied with nothing at all. Nothing was run "
                "and nothing was written."
            )
            self.opened[0] = "text"
            self.next_index = max(self.next_index, 1)
            yield self._open(0, {"type": "text", "text": ""})
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": why},
                },
            )
        for index in sorted(self.opened):
            yield _frame("content_block_stop", {"type": "content_block_stop", "index": index})
        # A reply with no stop_reason reads as truncation on the client.
        stop = self.stop or ("tool_use" if self.tool_at else "end_turn")
        yield _frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": _usage(self.counts),
            },
        )
        yield _frame("message_stop", {"type": "message_stop"})


NUDGE = (
    "That reply carried no call object. Answer again, and end with a single "
    '{"calls":[{"name":"<tool>","input":{...}}]} object on its own line.'
)


def needs_call(body: dict) -> bool:
    """Whether this turn has to act rather than speak.

    A turn answering a tool result may summarise it: that is how an agent loop
    ends, and forbidding it would leave the loop unable to stop. A turn
    answering the user has no result to summarise, so prose there is the model
    speaking where it was asked to act — and prose raises no tool call, so a
    client that gates writes behind approval never sees it. Work never done
    reads as work done.

    A request offering no tools owes nothing either: there is no call to make,
    so insisting on one asks the model for something the request did not make
    available.
    """
    if not (body.get("tools") or []):
        return False
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return not any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content
            )
        return True
    return True


def _nudged(body: dict) -> dict:
    """The same ask with the rule restated as a trailing user turn."""
    asked = dict(body)
    asked["messages"] = list(body.get("messages") or []) + [{"role": "user", "content": NUDGE}]
    return asked


REPAIR = (
    "The validator rejected that reply: {errors}. Answer again and correct "
    "exactly what it names; the contract has not changed."
)


def _last_user_text(body: dict) -> Optional[str]:
    """The request the validator judges the reply against, if any."""
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        said = _text_of(message.get("content"))
        return said or None
    return None


def _corrected(body: dict, raw: str, errors: list) -> dict:
    """The same ask, the failed reply on the record, and the verdict after it.

    The reply goes back as the assistant turn it was, because a correction only
    reads as a correction next to what it corrects — an error message alone
    asks the model to fix something it can no longer see.
    """
    listed = (
        "; ".join(f"{getattr(e, 'code', 'invalid')}: {getattr(e, 'message', e)}" for e in errors)
        or "invalid output"
    )
    asked = dict(body)
    asked["messages"] = list(body.get("messages") or []) + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": REPAIR.format(errors=listed)},
    ]
    return asked


def _repairing(
    upstream: str,
    body: dict,
    model: str,
    timeout: float,
    tool_style: str,
    retries: int,
    validate: Any,
    strict: bool,
) -> Iterator[bytes]:
    """Ask until the validator accepts, then say plainly that it never did.

    Buffered like ``_insisting`` and for the same reason: whether a reply is
    valid is only known once it ends, and a reply already on the wire cannot be
    taken back. The feedback loop is the decline-with-note flow a person runs
    by hand — quote the fault, ask again — with the validator writing the note.
    """
    prompt = _last_user_text(body)
    asked = body
    verdict = None
    collected: list[bytes] = []
    for _ in range(max(retries, 0) + 1):
        talk = Translator(model, tool_style)
        collected = list(_once(upstream, asked, model, timeout, tool_style, talk))
        raw = "".join(talk.buffer)
        verdict = validate(raw, prompt)
        if verdict.ok:
            yield from collected
            return
        asked = _corrected(asked, raw, verdict.errors)

    if not strict:
        # Annotate-only mode has nowhere to put an annotation on this wire, so
        # the last reply stands: the retries were still worth their asks.
        yield from collected
        return

    # Said rather than passed through: under --enforce an invalid reply must
    # not reach a client that would act on it, and an empty answer would read
    # as one that quietly succeeded.
    first = verdict.errors[0] if verdict is not None and verdict.errors else None
    named = f"{first.code}: {first.message}" if first is not None else "invalid output"
    talk = Translator(model, tool_style)
    yield talk.start()
    yield talk._open(0, {"type": "text", "text": ""})
    yield _frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": f"The validator refused every reply after "
                f"{max(retries, 0) + 1} attempts ({named}); "
                f"nothing was run and nothing was written.",
            },
        },
    )
    yield _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": _usage({}),
        },
    )
    yield _frame("message_stop", {"type": "message_stop"})


def relay(
    upstream: str,
    body: dict,
    model: str,
    timeout: float,
    tool_style: str = "native",
    call_retries: Optional[int] = None,
    validate: Any = None,
    validate_retries: int = 0,
    strict: bool = False,
) -> Iterator[bytes]:
    """Ask the upstream and translate its stream.

    With ``call_retries`` set and this turn owing a call, the reply is collected
    before any of it is sent: whether a call arrived is only known once the text
    ends, and a reply already on the wire cannot be taken back. With ``validate``
    set the same collection happens on every turn, the validator judges the
    reply, and a rejection is fed back and re-asked — which covers the owed-call
    case too, so ``call_retries`` is not consulted while a validator gates.
    """
    # Only a turn that owes a call is judged. The validators gate corpus rows
    # where an action was expected, so on a turn answering a tool result they
    # report the summary as a missing call — and refusing that is refusing the
    # loop its ending. Same rule `call_retries` follows, and for the same
    # reason: prose is wrong where an action was asked for, and right where a
    # result is being reported.
    if validate is not None and tool_style == "inline" and needs_call(body):
        yield from _repairing(
            upstream, body, model, timeout, tool_style, validate_retries, validate, strict
        )
        return
    if call_retries is not None and tool_style == "inline" and needs_call(body):
        yield from _insisting(upstream, body, model, timeout, tool_style, call_retries)
        return
    yield from _once(upstream, body, model, timeout, tool_style, Translator(model, tool_style))


def _insisting(
    upstream: str, body: dict, model: str, timeout: float, tool_style: str, retries: int
) -> Iterator[bytes]:
    """Ask until a call comes back, then say plainly that none did."""
    asked = body
    for _ in range(max(retries, 0) + 1):
        talk = Translator(model, tool_style)
        frames = list(_once(upstream, asked, model, timeout, tool_style, talk))
        if talk.tool_at:
            yield from frames
            return
        asked = _nudged(asked)

    # Said rather than silently passed through: the caller asked for an action,
    # and an empty reply would read as one that quietly succeeded.
    talk = Translator(model, tool_style)
    yield talk.start()
    yield talk._open(0, {"type": "text", "text": ""})
    yield _frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": f"No tool call after {max(retries, 0) + 1} attempts; "
                f"nothing was run and nothing was written.",
            },
        },
    )
    yield _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": _usage({}),
        },
    )
    yield _frame("message_stop", {"type": "message_stop"})


def _once(
    upstream: str, body: dict, model: str, timeout: float, tool_style: str, talk: "Translator"
) -> Iterator[bytes]:
    request = urllib.request.Request(
        f"{upstream.rstrip('/')}/v1/chat/completions",
        data=json.dumps(to_openai(body, model, tool_style)).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    yield talk.start()
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        for raw in reply:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            yield from talk.delta(chunk)
    yield from talk.finish()


# -------------------------------------------------------------------- server


def _handler(
    upstream: str,
    model: str,
    max_body: int,
    timeout: float,
    token: Optional[str],
    tool_style: str,
    call_retries: Optional[int] = None,
    validate: Any = None,
    validate_retries: int = 0,
    strict: bool = False,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "maatml-anthropic/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _fail(self, code: int, kind: str, message: str) -> None:
            # Anthropic's envelope exactly: clients parse {"error":{"message"}}.
            envelope = {"type": "error", "error": {"type": kind, "message": message}}
            body = json.dumps(envelope).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/health", ""):
                body = json.dumps({"status": "ok", "upstream": upstream, "model": model}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._fail(404, "not_found_error", f"no route {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/messages":
                self._fail(404, "not_found_error", f"no route {self.path}")
                return
            if token and self.headers.get("x-api-key") != token:
                self._fail(401, "authentication_error", "bad or missing x-api-key")
                return

            size = int(self.headers.get("content-length") or 0)
            if size > max_body:
                self._fail(413, "request_too_large", f"body over {max_body} bytes")
                return
            try:
                body = json.loads(self.rfile.read(size) or b"{}")
            except json.JSONDecodeError as exc:
                self._fail(400, "invalid_request_error", f"body is not JSON: {exc}")
                return

            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            try:
                stream = relay(
                    upstream,
                    body,
                    model,
                    timeout,
                    tool_style,
                    call_retries,
                    validate,
                    validate_retries,
                    strict,
                )
                for frame in stream:
                    self.wfile.write(frame)
                    self.wfile.flush()
            except urllib.error.URLError as exc:
                # Mid-stream, the status line is long gone; Anthropic's own
                # answer to that is an `error` frame inside the 200, which the
                # client surfaces rather than treating as a broken reply.
                self.wfile.write(
                    _frame(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"upstream {upstream}: {exc}",
                            },
                        },
                    )
                )
                self.wfile.flush()

    return Handler


def build(
    *,
    upstream: str = DEFAULT_UPSTREAM,
    model: str = "maatml-local",
    host: str = "127.0.0.1",
    port: int = 8080,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    timeout: float = 600.0,
    auth_token: Optional[str] = None,
    tool_style: str = "native",
    call_retries: Optional[int] = None,
    validate: Any = None,
    validate_retries: int = 0,
    strict: bool = False,
) -> ThreadingHTTPServer:
    """A ready server. The caller runs it, so a test can use an ephemeral port.

    ``validate`` is a ``(raw_text, user_prompt) -> ValidationResult`` callable;
    with it set every reply is collected, judged, and re-asked on rejection up
    to ``validate_retries`` times. ``strict`` decides what an exhausted retry
    budget serves: a plain statement that nothing passed, or the last reply.
    """
    if tool_style not in TOOL_STYLES:
        raise ValueError(f"tool_style must be one of {TOOL_STYLES}; got {tool_style!r}")
    if validate is not None and tool_style != "inline":
        # The raw text the validator judges is the inline transcript; under
        # native tools the call never exists as text, so there is nothing to
        # hand a corpus-trained validator.
        raise ValueError("validate requires tool_style=inline")
    return ThreadingHTTPServer(
        (host, port),
        _handler(
            upstream,
            model,
            max_body_bytes,
            timeout,
            auth_token,
            tool_style,
            call_retries,
            validate,
            validate_retries,
            strict,
        ),
    )
