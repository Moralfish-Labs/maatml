"""Tool-call chat rows: message building, span masking, predictor, prepare.

A character tokenizer with a template shaped like Qwen3's (tools preamble in
the system turn, an empty think block on the final assistant turn only, tool
results as user turns) keeps the unit tests hermetic. Tests marked ``qwen3``
use a real Qwen3 tokenizer from the local Hugging Face cache and skip cleanly
when none is cached; nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maatml.config import load_model_def  # noqa: E402
from maatml.data.formats import is_degenerate  # noqa: E402
from maatml.data.pipeline import prepare_rows  # noqa: E402
from maatml.training.sft_base import (  # noqa: E402
    _messages_from_sample,
    _render_then_tokenize,
    build_chat_example,
    render_chat_prompt,
)
from maatml.utils.chat import (  # noqa: E402
    ChatSpanError,
    parse_tool_calls,
    validate_chat_row,
)

SPEC = {"system": "S", "user_template": "U:<<USER_REQUEST>>"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a fact.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search stored facts.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]

REMEMBER_ARGS = '{"text": "host is HERC03"}'
SEARCH_ARGS = '{"query": "host"}'

TOOL_ROW = {
    "sample_id": "tc-1",
    "family": "tool_call",
    "surface": "assist_conversation",
    "category": "memory_remember",
    "chat_template_kwargs": {"enable_thinking": False},
    "tools": TOOLS,
    "request": "Remember: host is HERC03",
    "expected": {"steps": [["remember"], "answer"]},
    "messages": [
        {"role": "system", "content": "You are Neyru."},
        {"role": "user", "content": "Remember: host is HERC03"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "remember", "arguments": REMEMBER_ARGS},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "remember", "content": '{"ok":true}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search_memory", "arguments": SEARCH_ARGS},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "search_memory", "content": "[]"},
        {"role": "assistant", "content": "Saved: host is HERC03."},
    ],
}

PLAIN_ROW = {
    "sample_id": "cv-1",
    "family": "conversation",
    "request": "hi",
    "expected": "Hello.",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello."},
    ],
}

ROUTING_ROW = {
    "sample_id": "rt-1",
    "family": "routing",
    "request": "Remember the host",
    "expected": {"kind": "conversation"},
    "messages": [
        {"role": "system", "content": "Route the request. Reply with JSON only."},
        {"role": "user", "content": "Remember the host"},
        {"role": "assistant", "content": '{"kind": "conversation"}'},
    ],
}


class _QwenLikeTokenizer:
    """Character tokenizer (id == ord) with a Qwen3-shaped chat template."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, *, render_tools: bool = True) -> None:
        self.render_tools = render_tools
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        enable_thinking=True,
        **kwargs,
    ):
        del tokenize, kwargs
        self.calls.append({"tools": tools, "enable_thinking": enable_thinking})
        users = [i for i, m in enumerate(messages) if m["role"] == "user"]
        last_user = users[-1] if users else -1
        out = []
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "system":
                text = m["content"]
                if tools and self.render_tools:
                    text += "\n<tools>" + "".join(json.dumps(t) for t in tools) + "</tools>"
                out.append(f"<s>system\n{text}</s>\n")
            elif role == "user":
                out.append(f"<s>user\n{m['content']}</s>\n")
            elif role == "tool":
                out.append(f"<s>user\n<tool_response>{m['content']}</tool_response></s>\n")
            else:
                head = "<s>assistant\n"
                if i > last_user and i == len(messages) - 1:
                    head += "<think></think>\n"
                body = m["content"]
                for call in m.get("tool_calls") or []:
                    fn = call["function"]
                    args = fn["arguments"]
                    args = args if isinstance(args, str) else json.dumps(args)
                    call_json = f'{{"name": "{fn["name"]}", "arguments": {args}}}'
                    body += f"<tool_call>{call_json}</tool_call>"
                out.append(head + body + "</s>\n")
        if add_generation_prompt:
            out.append("<s>assistant\n" + ("" if enable_thinking else "<think></think>\n"))
        return "".join(out)

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        del add_special_tokens, return_tensors, kwargs
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i) for i in ids)


def _legacy_build(sample: dict, tokenizer, *, target_field: str = "expected") -> dict:
    """The masking ``build_chat_example`` used before span location (frozen)."""
    from maatml.training.sft_config import render_assistant_target

    rendered = []
    for msg in sample.get("messages") or []:
        role = {"human": "user", "user": "user", "gpt": "assistant"}.get(msg["role"], msg["role"])
        if role in ("system", "user", "assistant"):
            rendered.append({"role": role, "content": str(msg.get("content", ""))})
    if not rendered:
        rendered = [
            {"role": "system", "content": SPEC["system"]},
            {
                "role": "user",
                "content": SPEC["user_template"].replace("<<USER_REQUEST>>", sample["request"]),
            },
            {"role": "assistant", "content": render_assistant_target(sample, target_field)},
        ]
    full_ids = _render_then_tokenize(rendered, tokenizer, add_generation_prompt=False)
    labels = [-100] * len(full_ids)
    for i, msg in enumerate(rendered):
        if msg["role"] != "assistant":
            continue
        prefix = _render_then_tokenize(rendered[:i], tokenizer, add_generation_prompt=True)
        through = _render_then_tokenize(rendered[: i + 1], tokenizer, add_generation_prompt=False)
        for j in range(len(prefix), min(len(through), len(full_ids))):
            labels[j] = full_ids[j]
    return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


def _supervised_runs(example: dict, decode) -> list[str]:
    runs, cur = [], []
    for tok, lab in zip(example["input_ids"], example["labels"], strict=True):
        if lab != -100:
            assert lab == tok
            cur.append(tok)
        elif cur:
            runs.append(decode(cur))
            cur = []
    if cur:
        runs.append(decode(cur))
    return runs


def _build(row: dict, tokenizer, **kwargs) -> dict:
    return build_chat_example(
        row, SPEC, tokenizer, max_length=10**6, target_field="expected", **kwargs
    )


# --- message building -----------------------------------------------------


def test_messages_keep_tool_calls_and_tool_turns() -> None:
    msgs = _messages_from_sample(
        TOOL_ROW,
        SPEC,
        target_field="expected",
        request_field="request",
        user_placeholder="<<USER_REQUEST>>",
    )
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    call = msgs[2]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"] == {"name": "remember", "arguments": REMEMBER_ARGS}
    assert msgs[3] == {
        "role": "tool",
        "content": '{"ok":true}',
        "name": "remember",
        "tool_call_id": "call_1",
    }
    # Plain turns keep exactly role + content, as before.
    assert msgs[0] == {"role": "system", "content": "You are Neyru."}
    assert msgs[6] == {"role": "assistant", "content": "Saved: host is HERC03."}


def test_tools_and_template_kwargs_reach_the_template() -> None:
    tok = _QwenLikeTokenizer()
    _build(TOOL_ROW, tok)
    # Every render carries the tools except the one check that the template
    # actually renders them (same conversation, tools left out).
    assert [c["tools"] for c in tok.calls].count(None) == 1
    assert all(c["tools"] in (TOOLS, None) for c in tok.calls)
    assert all(c["enable_thinking"] is False for c in tok.calls)

    tok = _QwenLikeTokenizer()
    _build({**TOOL_ROW, "chat_template_kwargs": {"enable_thinking": True}}, tok)
    assert all(c["enable_thinking"] is True for c in tok.calls)


# --- masking --------------------------------------------------------------


def test_multi_turn_tool_row_masks_exactly_the_assistant_spans() -> None:
    tok = _QwenLikeTokenizer()
    ex = _build(TOOL_ROW, tok, return_spans=True)
    assert _supervised_runs(ex, tok.decode) == [
        f'<tool_call>{{"name": "remember", "arguments": {REMEMBER_ARGS}}}</tool_call></s>\n',
        f'<tool_call>{{"name": "search_memory", "arguments": {SEARCH_ARGS}}}</tool_call></s>\n',
        "Saved: host is HERC03.</s>\n",
    ]
    assert [idx for idx, _, _ in ex["spans"]] == [2, 4, 6]
    # Headers, the think block, tool results and the tools preamble stay masked.
    full = tok.decode(ex["input_ids"])
    assert "<tools>" in full and "<tool_response>" in full


def test_multi_turn_spans_do_not_shift_when_non_final_turns_drop_think() -> None:
    """The bug this replaces: prefix-length masking against one full render
    started every non-final span late and ran it into the next turn."""
    row = {
        "messages": [
            {"role": "user", "content": "Pick a colour."},
            {"role": "assistant", "content": "Noted: teal."},
            {"role": "user", "content": "Which colour?"},
            {"role": "assistant", "content": "Teal."},
        ]
    }
    tok = _QwenLikeTokenizer()
    old = _supervised_runs(_legacy_build(row, tok), tok.decode)
    assert any("<s>user" in run for run in old)
    new = _supervised_runs(_build(row, tok), tok.decode)
    assert new == ["Noted: teal.</s>\n", "Teal.</s>\n"]


def test_thinking_on_rows_drop_the_think_block_from_non_final_targets() -> None:
    tok = _QwenLikeTokenizer()
    row = {**TOOL_ROW, "chat_template_kwargs": {"enable_thinking": True}}
    runs = _supervised_runs(_build(row, tok), tok.decode)
    assert runs[0].startswith("<tool_call>")
    assert runs[-1] == "<think></think>\nSaved: host is HERC03.</s>\n"


def test_decision_only_row_trains_the_final_call() -> None:
    row = {**TOOL_ROW, "messages": TOOL_ROW["messages"][:3]}
    tok = _QwenLikeTokenizer()
    runs = _supervised_runs(_build(row, tok), tok.decode)
    assert runs == [
        f'<tool_call>{{"name": "remember", "arguments": {REMEMBER_ARGS}}}</tool_call></s>\n'
    ]


def test_template_without_tool_support_is_refused() -> None:
    with pytest.raises(ChatSpanError, match="did not render tools"):
        _build(TOOL_ROW, _QwenLikeTokenizer(render_tools=False))


def test_unlocatable_tool_span_raises_instead_of_mislabeling() -> None:
    class _Rewriting(_QwenLikeTokenizer):
        """Renders tool calls differently in non-final position."""

        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            calls = [i for i, m in enumerate(messages) if m.get("tool_calls")]
            if calls and calls[-1] != len(messages) - 1:
                text = text.replace("<tool_call>", "<call>", 1)
            return text

    with pytest.raises(ChatSpanError, match="tc-1"):
        _build(TOOL_ROW, _Rewriting())


@pytest.mark.parametrize("row", [PLAIN_ROW, ROUTING_ROW], ids=["plain", "routing"])
def test_plain_and_routing_rows_are_unchanged(row: dict) -> None:
    tok = _QwenLikeTokenizer()
    new = _build(row, tok)
    assert new == _legacy_build(row, tok)
    prompt = "<s>system\n" + row["messages"][0]["content"] + "</s>\n<s>user\n"
    prompt += row["messages"][1]["content"] + "</s>\n<s>assistant\n<think></think>\n"
    assert _supervised_runs(new, tok.decode) == [row["messages"][2]["content"] + "</s>\n"]
    assert new["input_ids"][: len(prompt)] == [ord(c) for c in prompt]


def test_prompt_spec_row_is_unchanged() -> None:
    row = {"request": "req", "expected": {"ok": True}}
    tok = _QwenLikeTokenizer()
    assert _build(row, tok) == _legacy_build(row, tok)


# --- real Qwen3 tokenizer (local cache only) ------------------------------


@pytest.fixture(scope="module")
def qwen3():
    transformers = pytest.importorskip("transformers")
    for name in ("Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"):
        try:
            return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:  # noqa: BLE001
            continue
    pytest.skip("no Qwen3 tokenizer in the local Hugging Face cache")


def _qwen_decode(tok):
    return lambda ids: tok.decode(ids, skip_special_tokens=False)


def test_qwen3_tool_row_spans_decode_to_the_assistant_turns(qwen3) -> None:
    ex = _build(TOOL_ROW, qwen3, return_spans=True)
    runs = _supervised_runs(ex, _qwen_decode(qwen3))
    assert runs == [
        f'<tool_call>\n{{"name": "remember", "arguments": {REMEMBER_ARGS}}}\n'
        "</tool_call><|im_end|>\n",
        f'<tool_call>\n{{"name": "search_memory", "arguments": {SEARCH_ARGS}}}\n'
        "</tool_call><|im_end|>\n",
        "Saved: host is HERC03.<|im_end|>\n",
    ]
    full = qwen3.decode(ex["input_ids"], skip_special_tokens=False)
    assert "<tools>" in full and '"name": "search_memory"' in full
    # The final turn's empty think block is prompt, not target.
    assert full.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\nSaved: host is HERC03.<|im_end|>\n"
    )


@pytest.mark.parametrize("row", [PLAIN_ROW, ROUTING_ROW], ids=["plain", "routing"])
def test_qwen3_plain_and_routing_rows_are_unchanged(qwen3, row: dict) -> None:
    new = _build(row, qwen3)
    assert new == _legacy_build(row, qwen3)
    assert _supervised_runs(new, _qwen_decode(qwen3)) == [
        row["messages"][2]["content"] + "<|im_end|>\n"
    ]


def test_qwen3_prompt_matches_the_training_prefix(qwen3) -> None:
    """The eval prompt for the final turn is the token prefix training used."""
    row = {**TOOL_ROW, "messages": TOOL_ROW["messages"][:4]}
    ids = render_chat_prompt(row, qwen3)
    text = qwen3.decode(ids, skip_special_tokens=False)
    assert "<tools>" in text and text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    final = {**row, "messages": [*row["messages"], {"role": "assistant", "content": "ok"}]}
    ex = _build(final, qwen3, return_spans=True)
    ((_, start, _end),) = [s for s in ex["spans"] if s[0] == 4]
    assert ex["input_ids"][:start] == ids


# --- predictor ------------------------------------------------------------


class _CannedModel:
    def __init__(self, reply: str) -> None:
        self.reply = [ord(c) for c in reply]

    def generate(self, input_ids, **kwargs):
        del kwargs
        tail = torch.tensor([self.reply], dtype=torch.long)
        return torch.cat([input_ids, tail], dim=-1)


def _predictor(tok, reply: str):
    from maatml.evaluation.predictors import CausalSFTPredictor
    from maatml.training.sft_base import render_inference_prompt

    pred = CausalSFTPredictor()
    pred._torch = torch
    pred._tokenizer = tok
    pred._model = _CannedModel(reply)
    pred._device = torch.device("cpu")
    pred._render = render_inference_prompt
    pred._render_chat = render_chat_prompt
    pred._prompt_spec = SPEC
    return pred


def test_predictor_passes_tools_and_parses_tool_calls() -> None:
    tok = _QwenLikeTokenizer()
    reply = f'<tool_call>{{"name": "remember", "arguments": {REMEMBER_ARGS}}}</tool_call>'
    pred = _predictor(tok, reply)
    row = {**TOOL_ROW, "messages": TOOL_ROW["messages"][:3]}

    prompt = "".join(chr(i) for i in pred.prompt_ids(row))
    assert "<tools>" in prompt and "<tool_call>" not in prompt
    assert prompt.endswith("<s>user\nRemember: host is HERC03</s>\n<s>assistant\n<think></think>\n")
    assert all(c["tools"] == TOOLS for c in tok.calls)

    out = pred.predict_structured(row)
    assert out["raw"] == reply
    assert out["content"] == ""
    assert out["errors"] == []
    assert out["tool_calls"] == [
        {
            "type": "function",
            "function": {"name": "remember", "arguments": {"text": "host is HERC03"}},
        }
    ]


def test_predictor_keeps_prompt_spec_path_for_request_rows() -> None:
    tok = _QwenLikeTokenizer()
    pred = _predictor(tok, "x")
    prompt = "".join(chr(i) for i in pred.prompt_ids({"request": "req"}))
    assert prompt == "<s>system\nS</s>\n<s>user\nU:req</s>\n<s>assistant\n<think></think>\n"


def test_parse_tool_calls_shapes() -> None:
    parsed = parse_tool_calls(
        '<think>\n\n</think>\n\nLet me check.\n<tool_call>\n{"name": "a", "arguments": {"x": 1}}'
        '\n</tool_call>\n<tool_call>{"name": "b", "arguments": "{\\"y\\": 2}"}</tool_call>'
    )
    assert parsed.content == "Let me check."
    assert [c["function"] for c in parsed.tool_calls] == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    assert parsed.errors == []

    bad = parse_tool_calls('<tool_call>{not json}</tool_call><tool_call>{"name": "c"')
    assert bad.tool_calls == []
    assert len(bad.errors) == 2

    assert parse_tool_calls("Plain answer.").content == "Plain answer."


# --- prepare / validation -------------------------------------------------


def test_validate_chat_row_accepts_tool_rows_and_flags_malformed_ones() -> None:
    assert validate_chat_row(TOOL_ROW) == []
    assert validate_chat_row(PLAIN_ROW) == []
    assert validate_chat_row(ROUTING_ROW) == []

    broken = json.loads(json.dumps(TOOL_ROW))
    broken["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{oops"
    broken["messages"][4]["tool_calls"][0]["function"]["name"] = "forget"
    broken["messages"][5]["tool_call_id"] = "call_9"
    broken["chat_template_kwargs"] = "thinking off"
    problems = validate_chat_row(broken)
    assert any("not JSON" in p for p in problems)
    assert any("'forget'" in p for p in problems)
    assert any("call_9" in p for p in problems)
    assert any("chat_template_kwargs" in p for p in problems)


def test_tool_call_only_turn_is_not_degenerate() -> None:
    row = {**TOOL_ROW, "messages": TOOL_ROW["messages"][:3]}
    assert not is_degenerate(row)


def _model_dir(tmp_path: Path):
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        "name: tools\nmodel_id: tools\narchitecture: causal_sft\nversion: 0.1.0\n"
        "dataset:\n  seed_samples: seeds.jsonl\n  group_by: scenario\n",
        encoding="utf-8",
    )
    return load_model_def(mdir)


def test_prepare_keeps_tool_fields_and_refuses_malformed_rows(tmp_path: Path) -> None:
    md = _model_dir(tmp_path)
    rows = [{**TOOL_ROW, "sample_id": f"tc-{i}", "scenario": f"s{i}"} for i in range(6)] + [
        {**PLAIN_ROW, "sample_id": f"cv-{i}", "scenario": f"p{i}"} for i in range(6)
    ]
    summary = prepare_rows(md, rows, out_dir=tmp_path / "out")
    written = [
        json.loads(line)
        for path in summary["splits"].values()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    tool_rows = [r for r in written if r["family"] == "tool_call"]
    assert len(tool_rows) == 6
    assert all(r["tools"] == TOOLS and r["messages"] == TOOL_ROW["messages"] for r in tool_rows)
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in tool_rows)

    bad = json.loads(json.dumps(rows[0]))
    bad["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{oops"
    with pytest.raises(ValueError, match="malformed chat rows"):
        prepare_rows(md, [bad, *rows[1:]], out_dir=tmp_path / "out2")
