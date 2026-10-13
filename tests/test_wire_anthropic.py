"""The Anthropic wire, checked against what a real client actually sends.

The client of record is NOON, whose reader (`crates/noon-core/src/stream.rs`)
is stricter than the spec: it reads only `data:` lines, treats a missing
`stop_reason` as truncation, and positions blocks by `index`. These tests
encode those three, plus the two keys NOON always sends that must be tolerated
rather than refused.
"""

from __future__ import annotations

import json

import pytest

from maatml.validation.base import ValidationError, ValidationResult
from maatml.wire import anthropic as wire
from maatml.wire.anthropic import Translator, to_openai, to_openai_messages, to_openai_tools


def frames(raw: list[bytes]) -> list[dict]:
    """Parse the way stream.rs does: `data:` lines only, everything else skipped."""
    out = []
    for chunk in raw:
        for line in chunk.decode().splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


# ------------------------------------------------------------------ request


def test_system_blocks_become_a_leading_message():
    body = {
        "system": [
            {
                "type": "text",
                "text": "you are NOON",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [{"role": "user", "content": "hello"}],
    }
    out = to_openai_messages(body)
    assert out[0] == {"role": "system", "content": "you are NOON"}
    assert out[1] == {"role": "user", "content": "hello"}


def test_cache_control_and_output_config_are_dropped_not_refused():
    """NOON sends both on every request. A 400 here breaks every ask."""
    body = {
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": "medium"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                ],
            }
        ],
        "max_tokens": 32000,
    }
    out = to_openai(body, "local")
    assert "cache_control" not in json.dumps(out)
    assert "output_config" not in out
    assert out["max_tokens"] == 32000


def test_tool_use_and_tool_result_round_trip_to_openai_shapes():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "reading it"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "docs/shots.json"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "{...}"},
                ],
            },
        ]
    }
    out = to_openai_messages(body)
    assert out[0]["tool_calls"][0]["id"] == "toolu_1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "read_file"
    args = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "docs/shots.json"}
    assert out[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "{...}"}


def test_tool_result_content_may_be_a_block_array():
    """NOON stores it as a bare string in 164 cases and an array in 96."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                ],
            }
        ]
    }
    assert to_openai_messages(body)[0]["content"] == "ok"


def test_input_schema_becomes_parameters():
    tools = [
        {
            "name": "set_shot_cell",
            "description": "d",
            "input_schema": {"type": "object", "properties": {"shot": {"type": "string"}}},
        }
    ]
    out = to_openai_tools(tools)
    assert out[0]["function"]["parameters"]["properties"]["shot"]["type"] == "string"
    assert "input_schema" not in json.dumps(out)


def test_thinking_is_switched_off_at_the_template():
    """The corpus is trained with thinking stripped; a server that reasons
    first is a different model than the one measured."""
    out = to_openai({"messages": []}, "local")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


# ----------------------------------------------------------------- response


def test_text_streams_as_one_indexed_block_and_always_stops():
    talk = Translator("local")
    raw = [talk.start()]
    raw += list(talk.delta({"choices": [{"delta": {"content": "he"}}]}))
    raw += list(talk.delta({"choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}]}))
    raw += list(talk.finish())

    seen = frames(raw)
    kinds = [f["type"] for f in seen]
    assert kinds[0] == "message_start"
    assert "content_block_start" in kinds
    assert kinds[-1] == "message_stop"

    said = "".join(f["delta"]["text"] for f in seen if f["type"] == "content_block_delta")
    assert said == "hello"
    assert all(f["index"] == 0 for f in seen if "index" in f)

    ending = [f for f in seen if f["type"] == "message_delta"]
    assert ending and ending[0]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.parametrize(
    "upstream,expected",
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
    ],
)
def test_finish_reason_maps_to_anthropic(upstream, expected):
    talk = Translator("local")
    list(talk.delta({"choices": [{"delta": {"content": "x"}, "finish_reason": upstream}]}))
    ending = [f for f in frames(list(talk.finish())) if f["type"] == "message_delta"]
    assert ending[0]["delta"]["stop_reason"] == expected


def test_a_stream_that_never_said_why_still_carries_a_stop_reason():
    """stream.rs treats a missing stop_reason as truncation and errors. An
    upstream that closes without one must not become a broken reply."""
    talk = Translator("local")
    list(talk.delta({"choices": [{"delta": {"content": "x"}}]}))
    ending = [f for f in frames(list(talk.finish())) if f["type"] == "message_delta"]
    assert ending[0]["delta"]["stop_reason"] == "end_turn"


def test_tool_calls_get_their_own_block_index_and_json_deltas():
    talk = Translator("local")
    list(talk.delta({"choices": [{"delta": {"content": "calling"}}]}))
    out = list(
        talk.delta(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "edit_file", "arguments": '{"pa'},
                                },
                            ]
                        }
                    }
                ]
            }
        )
    )
    out += list(
        talk.delta(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'th":"a.py"}'}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    )

    seen = frames(out)
    opened = [f for f in seen if f["type"] == "content_block_start"]
    assert opened[0]["index"] == 1, "text took block 0, so the call takes 1"
    assert opened[0]["content_block"]["name"] == "edit_file"

    args = "".join(
        f["delta"]["partial_json"]
        for f in seen
        if f["type"] == "content_block_delta" and f["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(args) == {"path": "a.py"}


def test_two_tool_calls_do_not_share_an_index():
    talk = Translator("local")
    out = list(
        talk.delta(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "function": {"name": "read_file", "arguments": "{}"},
                                },
                                {
                                    "index": 1,
                                    "id": "b",
                                    "function": {"name": "get_state", "arguments": "{}"},
                                },
                            ]
                        }
                    }
                ]
            }
        )
    )
    opened = [f for f in frames(out) if f["type"] == "content_block_start"]
    assert [f["index"] for f in opened] == [0, 1]


def test_usage_uses_anthropics_four_names():
    talk = Translator("local")
    list(
        talk.delta(
            {
                "usage": {"prompt_tokens": 120, "completion_tokens": 8},
                "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}],
            }
        )
    )
    ending = [f for f in frames(list(talk.finish())) if f["type"] == "message_delta"][0]
    assert ending["usage"] == {
        "input_tokens": 120,
        "output_tokens": 8,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_no_thinking_block_is_ever_emitted():
    """Anthropic rejects a thinking block handed back without its signature,
    and a local server has none to give."""
    talk = Translator("local")
    raw = [talk.start()]
    raw += list(talk.delta({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}))
    raw += list(talk.finish())
    assert "thinking" not in json.dumps(frames(raw))


# ------------------------------------------------------- the validator gates


def _scripted_upstream(monkeypatch, replies: list[str], record: list[dict]) -> None:
    """Play canned replies through the real Translator, one per ask."""
    queue = list(replies)

    def fake_once(upstream, asked, model, timeout, tool_style, talk):
        record.append(asked)
        said = queue.pop(0)
        yield talk.start()
        yield from talk.delta({"choices": [{"delta": {"content": said}, "finish_reason": "stop"}]})
        yield from talk.finish()

    monkeypatch.setattr(wire, "_once", fake_once)


def _accepting(raw: str, prompt=None) -> ValidationResult:
    return ValidationResult(raw_output=raw, passed_layers={1})


def _rejecting(raw: str, prompt=None) -> ValidationResult:
    return ValidationResult(
        raw_output=raw,
        errors=[ValidationError(layer=1, code="bad_json", message="no call object parsed")],
    )


def _texts(raw_frames: list[bytes]) -> str:
    return "".join(
        f["delta"]["text"]
        for f in frames(raw_frames)
        if f.get("type") == "content_block_delta" and f["delta"].get("type") == "text_delta"
    )


def test_a_rejected_reply_is_re_asked_with_the_error_on_the_record(monkeypatch):
    record: list[dict] = []
    good = '{"calls":[{"name":"set_tracker_cell","input":{"unit":7}}]}'
    _scripted_upstream(monkeypatch, ["prose, not a call", good], record)
    verdicts = iter([_rejecting("prose, not a call"), _accepting(good)])

    out = list(
        wire.relay(
            "up",
            {
                "tools": [{"name": "set_tracker_cell", "input_schema": {}}],
                "messages": [{"role": "user", "content": "flip the cell"}],
            },
            "m",
            1.0,
            "inline",
            validate=lambda raw, prompt: next(verdicts),
            validate_retries=2,
            strict=True,
        )
    )

    assert len(record) == 2, "the rejection was re-asked exactly once"
    retried = record[1]["messages"]
    assert retried[-2] == {"role": "assistant", "content": "prose, not a call"}, (
        "the failed reply goes back as the turn it was, or the model corrects blind"
    )
    assert "bad_json: no call object parsed" in retried[-1]["content"]
    kinds = [
        f["content_block"]["type"] for f in frames(out) if f.get("type") == "content_block_start"
    ]
    assert "tool_use" in kinds, "the accepted reply is lifted into a call block"
    assert "prose, not a call" not in _texts(out), "no rejected reply reaches the client"


def test_enforce_replaces_an_exhausted_retry_budget_with_a_plain_refusal(monkeypatch):
    record: list[dict] = []
    _scripted_upstream(monkeypatch, ["wrong", "wrong again"], record)

    out = list(
        wire.relay(
            "up",
            {
                "tools": [{"name": "set_tracker_cell", "input_schema": {}}],
                "messages": [{"role": "user", "content": "flip the cell"}],
            },
            "m",
            1.0,
            "inline",
            validate=_rejecting,
            validate_retries=1,
            strict=True,
        )
    )

    assert len(record) == 2
    said = _texts(out)
    assert "nothing was run and nothing was written" in said
    assert "bad_json" in said, "the refusal names what the validator refused"
    assert "wrong" not in said, "no rejected reply reaches the client under enforce"


def test_without_enforce_the_last_reply_stands_after_the_retries(monkeypatch):
    record: list[dict] = []
    _scripted_upstream(monkeypatch, ["wrong", "still wrong"], record)

    out = list(
        wire.relay(
            "up",
            {
                "tools": [{"name": "set_tracker_cell", "input_schema": {}}],
                "messages": [{"role": "user", "content": "flip the cell"}],
            },
            "m",
            1.0,
            "inline",
            validate=_rejecting,
            validate_retries=1,
            strict=False,
        )
    )

    assert len(record) == 2, "the retries still ran"
    assert _texts(out) == "still wrong", "annotate-only mode serves the last reply"


def test_a_validator_needs_the_inline_transcript_to_judge():
    import pytest

    with pytest.raises(ValueError, match="inline"):
        wire.build(port=0, tool_style="native", validate=_accepting)


def test_a_turn_answering_a_tool_result_is_not_asked_for_a_call(monkeypatch):
    """The summary that ends a loop must not be refused for carrying no call.

    The validators gate rows where an action was expected; on a reporting turn
    they read the summary as a missing call. Enforcing there would mean no
    conversation could ever end.
    """
    record: list[dict] = []
    _scripted_upstream(monkeypatch, ["Set Camera Dir to done."], record)

    out = list(
        wire.relay(
            "up",
            {
                "tools": [{"name": "set_tracker_cell", "input_schema": {}}],
                "messages": [
                    {"role": "user", "content": "flip the cell"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "set_tracker_cell",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
                        ],
                    },
                ],
            },
            "m",
            1.0,
            "inline",
            validate=_rejecting,
            validate_retries=2,
            strict=True,
        )
    )

    assert len(record) == 1, "a reporting turn is asked once and not re-asked"
    assert _texts(out) == "Set Camera Dir to done.", "the summary reaches the client"


def test_a_request_offering_no_tools_owes_no_call(monkeypatch):
    """Nothing is callable, so nothing can be insisted on or refused."""
    record: list[dict] = []
    _scripted_upstream(monkeypatch, ["ready"], record)

    out = list(
        wire.relay(
            "up",
            {"messages": [{"role": "user", "content": "say ready"}]},
            "m",
            1.0,
            "inline",
            validate=_rejecting,
            validate_retries=2,
            strict=True,
        )
    )

    assert len(record) == 1
    assert _texts(out) == "ready"


def test_thinking_is_switched_off_by_every_key_the_upstreams_read():
    """One key does not reach every upstream: llama.cpp reads the template
    kwarg, LM Studio reads reasoning_effort. Serving with either missing is
    serving a model nobody gated."""
    out = wire.to_openai({"messages": [{"role": "user", "content": "hi"}]}, "m")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}
    assert out["reasoning_effort"] == "none"


def test_a_reply_that_was_all_thinking_says_so_instead_of_arriving_empty():
    """An empty turn is dropped by the client, so the ask reads as never asked."""
    talk = wire.Translator("m", "inline")
    list(
        talk.delta(
            {
                "choices": [
                    {"delta": {"reasoning_content": "Okay, the user..."}, "finish_reason": "stop"}
                ]
            }
        )
    )
    said = "".join(
        f["delta"]["text"]
        for f in frames(list(talk.finish()))
        if f.get("type") == "content_block_delta"
    )
    assert "thinking is still on" in said
    assert "nothing was written" in said


def test_an_upstream_that_said_nothing_at_all_says_that_too():
    talk = wire.Translator("m", "inline")
    list(talk.delta({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    said = "".join(
        f["delta"]["text"]
        for f in frames(list(talk.finish()))
        if f.get("type") == "content_block_delta"
    )
    assert "nothing at all" in said
