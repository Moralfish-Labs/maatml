"""Tools carried in message text: one renderer, one parser, both directions."""

from __future__ import annotations

import json

from maatml.wire import inline_tools
from maatml.wire.anthropic import Translator, to_openai

TOOLS = [
    {
        "name": "set_tracker_cell",
        "description": "Set one tracker cell's status. Writes the tracker JSON.",
        "input_schema": {
            "type": "object",
            "properties": {
                "unit": {"type": "integer"},
                "stage": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["unit", "stage"],
        },
    },
    {
        "name": "get_state",
        "description": "Full production state.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def test_the_catalogue_names_every_argument_and_its_type() -> None:
    rendered = inline_tools.render_catalogue(TOOLS)
    assert "set_tracker_cell(stage:str, unit:int, note:str?)" in rendered
    assert "get_state()" in rendered
    # Sorted, so a reordered schema export does not rewrite every row.
    assert rendered.index("get_state") < rendered.index("set_tracker_cell")


def test_an_optional_argument_is_marked_and_a_required_one_is_not() -> None:
    line = inline_tools.render_catalogue(TOOLS).splitlines()[-1]
    assert "unit:int," in line and "unit:int?" not in line
    assert "note:str?" in line


def test_a_call_object_round_trips() -> None:
    calls = [{"name": "get_state", "input": {"project": "bdb"}}]
    answer = "Reading the project.\n" + inline_tools.render_calls(calls)
    said, back = inline_tools.split_calls(answer)
    assert said == "Reading the project."
    assert back == calls


def test_an_answer_with_no_object_is_all_text() -> None:
    said, calls = inline_tools.split_calls("There is nothing to do.")
    assert calls is None
    assert said == "There is nothing to do."


def test_a_malformed_object_stays_text_rather_than_becoming_a_call() -> None:
    said, calls = inline_tools.split_calls('Doing it.\n{"calls":[{"name":')
    assert calls is None
    assert said.endswith('{"calls":[{"name":')


def test_a_call_without_a_name_is_refused() -> None:
    _, calls = inline_tools.split_calls('{"calls":[{"input":{}}]}')
    assert calls is None


def test_parallel_calls_survive_the_round_trip() -> None:
    calls = [
        {"name": "get_state", "input": {}},
        {"name": "get_gates", "input": {"project": "bdb"}},
    ]
    _, back = inline_tools.split_calls(inline_tools.render_calls(calls))
    assert back == calls


def test_prose_that_mentions_the_marker_keeps_the_last_object() -> None:
    text = 'The format is {"calls":[...]}. Here it is:\n' + inline_tools.render_calls(
        [{"name": "get_state", "input": {}}]
    )
    _, calls = inline_tools.split_calls(text)
    assert calls == [{"name": "get_state", "input": {}}]


def test_inline_moves_tools_out_of_the_request_and_into_the_last_user_turn() -> None:
    body = {
        "system": "You are NOON.",
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "what is next?"},
        ],
    }
    out = inline_tools.inline_request(body)
    assert "tools" not in out
    assert out["messages"][0]["content"] == "hello"
    assert out["messages"][2]["content"].startswith(inline_tools.HEADING)
    assert out["messages"][2]["content"].endswith("what is next?")
    assert inline_tools.RULE in out["system"]


def test_inline_prepends_a_catalogue_block_to_a_tool_result_turn() -> None:
    body = {
        "tools": TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                ],
            },
        ],
    }
    blocks = inline_tools.inline_request(body)["messages"][0]["content"]
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"].startswith(inline_tools.HEADING)
    assert blocks[1]["type"] == "tool_result"


def test_a_request_with_no_tools_is_left_alone() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert inline_tools.inline_request(body) == body


def test_the_upstream_request_declares_no_tools_under_inline() -> None:
    body = {"tools": TOOLS, "messages": [{"role": "user", "content": "what is next?"}]}
    native = to_openai(body, "m", "native")
    inline = to_openai(body, "m", "inline")
    assert [t["function"]["name"] for t in native["tools"]] == ["set_tracker_cell", "get_state"]
    assert "tools" not in inline
    assert inline_tools.HEADING in inline["messages"][-1]["content"]


def test_inline_history_writes_past_calls_and_results_as_text() -> None:
    body = {
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "what is next?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the project."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "get_state",
                        "input": {"project": "bdb"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "shot 3 is blocked"},
                ],
            },
        ],
    }
    turns = to_openai(body, "m", "inline")["messages"]
    assert all("tool_calls" not in t for t in turns)
    assert all(t["role"] != "tool" for t in turns)
    assert turns[-2]["content"] == (
        'Reading the project.\n{"calls": [{"name": "get_state", "input": {"project": "bdb"}}]}'
    )
    assert turns[-1]["content"].endswith("<tool_response>\nshot 3 is blocked\n</tool_response>")


def test_native_history_still_uses_openai_tool_turns() -> None:
    body = {
        "tools": TOOLS,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "get_state", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                ],
            },
        ],
    }
    turns = to_openai(body, "m", "native")["messages"]
    assert turns[0]["tool_calls"][0]["function"]["name"] == "get_state"
    assert turns[1] == {"role": "tool", "tool_call_id": "t1", "content": "ok"}


def test_a_failed_result_is_marked_in_the_history() -> None:
    body = {
        "tools": TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "boom",
                        "is_error": True,
                    },
                ],
            }
        ],
    }
    assert "<tool_response> (error)" in to_openai(body, "m", "inline")["messages"][-1]["content"]


def _frames(chunks: list[dict], tool_style: str) -> list[dict]:
    talk = Translator("m", tool_style)
    out = [talk.start()]
    for chunk in chunks:
        out.extend(talk.delta(chunk))
    out.extend(talk.finish())
    return [json.loads(f.decode().split("data: ", 1)[1]) for f in out]


def _stream(text: str) -> list[dict]:
    return [{"choices": [{"delta": {"content": piece}}]} for piece in text] + [
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    ]


def test_a_buffered_call_object_becomes_a_tool_use_block() -> None:
    said = 'Setting the cell.\n{"calls":[{"name":"get_state","input":{"project":"bdb"}}]}'
    frames = _frames(_stream(said), "inline")
    kinds = [
        f.get("content_block", {}).get("type") for f in frames if f["type"] == "content_block_start"
    ]
    assert kinds == ["text", "tool_use"]

    text = "".join(
        f["delta"]["text"]
        for f in frames
        if f["type"] == "content_block_delta" and f["delta"]["type"] == "text_delta"
    )
    assert text == "Setting the cell."

    args = [
        f["delta"]["partial_json"]
        for f in frames
        if f["type"] == "content_block_delta" and f["delta"]["type"] == "input_json_delta"
    ]
    assert json.loads(args[0]) == {"project": "bdb"}
    assert frames[-2]["delta"]["stop_reason"] == "tool_use"


def test_an_inline_answer_with_no_call_ends_the_turn() -> None:
    frames = _frames(_stream("Nothing is blocked."), "inline")
    assert frames[-2]["delta"]["stop_reason"] == "end_turn"
    assert [f.get("index") for f in frames if f["type"] == "content_block_start"] == [0]


def test_a_truncated_inline_reply_keeps_max_tokens_over_tool_use() -> None:
    chunks = _stream('Doing it.\n{"calls":[{"name":"get_state","input":{}}]}')
    chunks[-1] = {"choices": [{"delta": {}, "finish_reason": "length"}]}
    assert _frames(chunks, "inline")[-2]["delta"]["stop_reason"] == "max_tokens"


def test_native_streaming_is_unchanged_by_the_option() -> None:
    frames = _frames(_stream("hello"), "native")
    text = "".join(
        f["delta"]["text"]
        for f in frames
        if f["type"] == "content_block_delta" and f["delta"]["type"] == "text_delta"
    )
    assert text == "hello"
    assert frames[-2]["delta"]["stop_reason"] == "end_turn"
