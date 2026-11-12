"""Torch-gated trainer unit tests: tokenization, label masking, collation.

Skipped on the torch-free matrix; the ml CI job runs them. Stub tokenizers
keep the tests hermetic (no hub download), so what is under test is maatml's
masking and tensor assembly, not a specific tokenizer's vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maatml.training.multi_head import HeadSpec  # noqa: E402
from maatml.training.multi_head import _build_dataset as build_head_dataset  # noqa: E402
from maatml.training.seq2seq import _build_dataset as build_seq2seq_dataset  # noqa: E402


class _CharTokenizer:
    """Character-level stand-in: token id == ord(char), so spans are readable."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        del tokenize, kwargs
        text = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        return text

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        del add_special_tokens, return_tensors, kwargs
        return {"input_ids": [ord(c) for c in text]}


class _PaddingTokenizer:
    """Pads to max_length and reports offsets, like a fast tokenizer would."""

    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length=8,
        padding=None,
        truncation=None,
        return_offsets_mapping=False,
        return_tensors=None,
        **kwargs,
    ):
        del padding, truncation, return_tensors, kwargs
        ids = [ord(c) for c in text][:max_length]
        offsets = [(i, i + 1) for i in range(len(ids))]
        attention = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attention.append(0)
            offsets.append((0, 0))
        out = {"input_ids": ids, "attention_mask": attention}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# --- causal SFT: loss masking ---------------------------------------------


def test_build_chat_example_masks_prompt_and_unmasks_assistant() -> None:
    from maatml.training.sft_base import build_chat_example

    tokenizer = _CharTokenizer()
    spec = {"system": "S", "user_template": "U:<<USER_REQUEST>>"}
    example = build_chat_example(
        {"request": "req", "expected": {"ok": True}},
        spec,
        tokenizer,
        max_length=512,
        target_field="expected",
        request_field="request",
    )

    text = "[system]S[user]U:req[assistant]" + '{"ok":true}'
    assert example["input_ids"] == [ord(c) for c in text]

    prompt_len = len("[system]S[user]U:req[assistant]")
    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert example["labels"][prompt_len:] == example["input_ids"][prompt_len:]
    assert example["length"] == len(example["input_ids"])


def test_build_chat_example_unmasks_every_assistant_turn() -> None:
    from maatml.training.sft_base import build_chat_example

    example = build_chat_example(
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ]
        },
        {"system": "S", "user_template": "U"},
        _CharTokenizer(),
        max_length=512,
        target_field="unused",
    )
    unmasked = "".join(
        chr(tok)
        for tok, lab in zip(example["input_ids"], example["labels"], strict=False)
        if lab != -100
    )
    # Both assistant turns contribute to the loss; the user turns do not.
    assert unmasked == "bd"


def test_build_chat_example_refuses_a_row_longer_than_max_length() -> None:
    """A long row is an error, never a left-truncated prompt: cutting from the
    left drops the system turn the model is always served."""
    from maatml.training.sft_base import RowTooLongError, build_chat_example

    row = {"sample_id": "long-row", "request": "req", "expected": {"ok": True}}
    spec = {"system": "S", "user_template": "U:<<USER_REQUEST>>"}
    kwargs = {"target_field": "expected", "request_field": "request"}
    length = len("[system]S[user]U:req[assistant]" + '{"ok":true}')

    fits = build_chat_example(row, spec, _CharTokenizer(), max_length=length, **kwargs)
    assert fits["length"] == length
    assert fits["input_ids"][0] == ord("[")

    with pytest.raises(RowTooLongError, match=rf"row long-row: {length} tokens exceed") as raised:
        build_chat_example(row, spec, _CharTokenizer(), max_length=length - 1, **kwargs)
    assert isinstance(raised.value, ValueError)
    assert f"max_input_tokens={length - 1}" in str(raised.value)


def test_build_chat_example_refuses_a_long_messages_row() -> None:
    from maatml.training.sft_base import RowTooLongError, build_chat_example

    row = {
        "sample_id": "chat-row",
        "messages": [
            {"role": "system", "content": "always served"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
    }
    with pytest.raises(RowTooLongError, match="row chat-row"):
        build_chat_example(row, {}, _CharTokenizer(), max_length=8, target_field="unused")


def test_sft_collator_pads_and_masks_padding() -> None:
    from maatml.training.sft_base import SFTDataCollator

    collator = SFTDataCollator(
        _CharTokenizer(),
        {"system": "S", "user_template": "U"},
        max_length=16,
        target_field="expected",
        pretokenized=True,
    )
    batch = collator(
        [
            {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
            {"input_ids": [8], "labels": [8]},
        ]
    )
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"][1].tolist() == [1, 0, 0]
    assert batch["labels"][1].tolist() == [8, -100, -100]


# --- seq2seq: label padding ------------------------------------------------


def test_seq2seq_dataset_masks_pad_tokens_in_labels() -> None:
    ds = build_seq2seq_dataset(
        [{"request": "ab", "target": {"k": 1}}],
        _PaddingTokenizer(),
        source_max_len=6,
        target_max_len=12,
        source_prefix="p:",
        request_field="request",
        target_field="target",
    )
    item = ds[0]
    assert item["input_ids"].tolist()[:4] == [ord(c) for c in "p:ab"]
    labels = item["labels"].tolist()
    assert labels[: len('{"k":1}')] == [ord(c) for c in '{"k":1}']
    # Padding is -100 so it never contributes to the loss.
    assert set(labels[len('{"k":1}') :]) == {-100}


def test_seq2seq_dataset_refuses_an_empty_target() -> None:
    ds = build_seq2seq_dataset(
        [{"request": "ab", "target": {}}],
        _PaddingTokenizer(),
        source_max_len=6,
        target_max_len=6,
        request_field="request",
        target_field="target",
    )
    with pytest.raises(ValueError, match="target is empty"):
        ds[0]


# --- multi_head: per-head targets -----------------------------------------


def test_multi_head_dataset_builds_class_and_line_targets() -> None:
    heads = [
        HeadSpec(
            name="validity", kind="classification", labels=["invalid", "valid"], target_path="valid"
        ),
        HeadSpec(name="line", kind="line_pointer", target_path="errors[0].line"),
    ]
    ds = build_head_dataset(
        [{"request": "a\nb", "target": {"valid": False, "errors": [{"line": 2}]}}],
        _PaddingTokenizer(),
        8,
        heads,
        request_field="request",
        target_field="target",
        text_transform=None,
    )
    item = ds[0]
    assert item["validity_label"].item() == 0
    line_labels = item["line_labels"].tolist()
    # Line 2 starts after the newline: char offset 2.
    assert line_labels[:2] == [0, 0]
    assert line_labels[2] == 1
    # Padding positions stay ignored.
    assert line_labels[3:] == [-100] * (len(line_labels) - 3)


def test_multi_head_dataset_rejects_unknown_gold_labels() -> None:
    from maatml.training.multi_head import UnknownLabelError

    heads = [HeadSpec(name="code", kind="classification", labels=["a", "none"], target_path="code")]
    ds = build_head_dataset(
        [{"request": "x", "target": {"code": "unknown"}}],
        _PaddingTokenizer(),
        4,
        heads,
        request_field="request",
        target_field="target",
        text_transform=None,
    )
    with pytest.raises(UnknownLabelError):
        ds[0]


# --- preference rows -------------------------------------------------------


def test_preference_rows_load_as_json_and_count_identical_pairs(tmp_path: Path) -> None:
    from maatml.training.preference import _load_preference_rows
    from maatml.utils.io import write_jsonl

    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            {"prompt": "p", "chosen": {"ok": True}, "rejected": "no"},
            {"prompt": "q", "chosen": "same", "rejected": "same"},
        ],
    )
    rows, identical = _load_preference_rows(path, None)
    assert rows[0]["chosen"] == '{"ok":true}'
    assert identical == 1
