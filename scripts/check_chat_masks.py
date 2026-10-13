"""Dry-run the causal SFT renderer and loss masks over chat rows.

Renders every row with a real tokenizer exactly as ``train_sft`` would
(``build_chat_example``: the row's ``tools`` and ``chat_template_kwargs``, one
full render, assistant spans located and decoded back) and reports counts,
lengths, and every row whose supervised spans do not match. Nothing is trained
and nothing is written unless ``--json`` is given.

    python scripts/check_chat_masks.py ROWS.jsonl [ROWS.jsonl ...] \
        --tokenizer Qwen/Qwen3-4B --max-length 8192 [--compare-legacy]

``--compare-legacy`` also builds every plain row (no tools, no tool turns)
with the masking the trainer used before span location and counts rows whose
ids or labels changed. The tokenizer is loaded from the local cache only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from maatml.training.sft_base import (
    _render_then_tokenize,
    build_chat_example,
)
from maatml.utils.chat import (
    ChatSpanError,
    assistant_char_spans,
    chat_template_options,
    has_tool_features,
    normalize_messages,
    render_chat,
)

SPEC = {"system": "", "user_template": "<<USER_REQUEST>>"}


def legacy_example(row: dict, tokenizer: Any) -> dict[str, list[int]]:
    """The pre-span masking (maatml main before this change), frozen here."""
    rendered = [
        {"role": m["role"], "content": m["content"]}
        for m in normalize_messages(row.get("messages"))
        if m["role"] in ("system", "user", "assistant")
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
    return {"input_ids": full_ids, "labels": labels}


def header_parity(row: dict, tokenizer: Any) -> tuple[int, int]:
    """``(turns, turns_without_full_generation_prompt)`` for one row.

    A turn whose header in the full render is shorter than the generation
    prompt (Qwen3: no empty think block on non-final turns) is trained in a
    context the model never sees at inference, where the prompt always ends
    with the full generation prompt.
    """
    messages = normalize_messages(row.get("messages"))
    tools, kwargs = chat_template_options(row)
    text = render_chat(
        messages, tokenizer, add_generation_prompt=False, tools=tools, template_kwargs=kwargs
    )
    spans = assistant_char_spans(
        messages, tokenizer, full_text=text, tools=tools, template_kwargs=kwargs
    )
    short = 0
    for span in spans:
        before = render_chat(
            messages[: span.message_index],
            tokenizer,
            add_generation_prompt=False,
            tools=tools,
            template_kwargs=kwargs,
        )
        prompt = render_chat(
            messages[: span.message_index],
            tokenizer,
            add_generation_prompt=True,
            tools=tools,
            template_kwargs=kwargs,
        )
        if span.start - span.header_start < len(prompt) - len(before):
            short += 1
    return len(spans), short


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="+", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report: dict[str, Any] = {"tokenizer": args.tokenizer, "files": {}}
    exit_code = 0
    for path in args.rows:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stats: Counter[str] = Counter()
        lengths: list[int] = []
        failures: list[str] = []
        legacy_changed: list[str] = []
        for row in rows:
            sid = str(row.get("sample_id"))
            messages = normalize_messages(row.get("messages"))
            tools, _ = chat_template_options(row)
            tool_row = has_tool_features(messages, tools)
            stats["tool_rows" if tool_row else "plain_rows"] += 1
            try:
                ex = build_chat_example(
                    row,
                    SPEC,
                    tokenizer,
                    max_length=10**9,
                    target_field="expected",
                    return_spans=True,
                )
            except ChatSpanError as exc:
                failures.append(f"{sid}: {exc}")
                continue
            n = ex["length"]
            lengths.append(n)
            stats["over_max_length"] += n > args.max_length
            want = sum(m["role"] == "assistant" for m in messages)
            got = len(ex["spans"])
            stats["assistant_turns"] += want
            stats["supervised_spans"] += got
            stats["supervised_tokens"] += sum(lab != -100 for lab in ex["labels"])
            stats["tool_call_turns"] += sum(bool(m.get("tool_calls")) for m in messages)
            if got != want:
                failures.append(f"{sid}: {got} spans located for {want} assistant turns")
                continue
            # Every supervised token belongs to a located span and nothing else.
            inside = {j for _, s, e in ex["spans"] for j in range(s, e)}
            stray = [j for j, lab in enumerate(ex["labels"]) if (lab != -100) != (j in inside)]
            if stray:
                failures.append(f"{sid}: {len(stray)} labels outside the located spans")
            for idx, s, e in ex["spans"]:
                piece = tokenizer.decode(ex["input_ids"][s:e], skip_special_tokens=False)
                msg = messages[idx]
                if msg.get("tool_calls") and piece.count("<tool_call>") != len(msg["tool_calls"]):
                    failures.append(f"{sid} turn {idx}: tool_call blocks do not match the calls")
                if "<|im_start|>" in piece or not piece.endswith("<|im_end|>\n"):
                    failures.append(f"{sid} turn {idx}: span leaks past its turn")
            turns, short = header_parity(row, tokenizer)
            stats["turns_without_full_generation_prompt"] += short
            stats["rows_with_several_assistant_turns"] += turns > 1
            if args.compare_legacy and not tool_row:
                old = legacy_example(row, tokenizer)
                same = old["input_ids"] == ex["input_ids"] and old["labels"] == ex["labels"]
                stats["legacy_identical" if same else "legacy_changed"] += 1
                if not same:
                    legacy_changed.append(sid)
        lengths.sort()
        file_report = {
            "rows": len(rows),
            **dict(stats),
            "max_tokens": lengths[-1] if lengths else 0,
            "p50_tokens": lengths[len(lengths) // 2] if lengths else 0,
            "span_failures": len(failures),
            "failures": failures,
            "legacy_changed_rows": legacy_changed,
        }
        report["files"][str(path)] = file_report
        if failures:
            exit_code = 1
        summary = {k: v for k, v in file_report.items() if k not in ("failures",)}
        if len(legacy_changed) > 10:
            summary["legacy_changed_rows"] = legacy_changed[:10] + ["..."]
        print(f"{path}:")
        print(json.dumps(summary, indent=2))
        for line in failures[:20]:
            print(f"  FAIL {line}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
