"""Loss masking for reasoning spans (<think>, <thought>, <|thought|>, Gemma 4 channel)."""

from __future__ import annotations

import pytest

from maatml.training.sft_config import SFTTrainConfig
from maatml.utils.chat import reasoning_char_spans


def test_reasoning_char_spans_finds_all_supported_delimiters_and_ignores_unclosed() -> None:
    text = (
        "prefix "
        "<think>step 1</think> mid "
        "<thought>step 2</thought> "
        "<|thought|>step 3<|/thought|> "
        "<|channel>thought\nstep 4<channel|> "
        "<think>unclosed tail"
    )
    spans = reasoning_char_spans(text)
    extracted = [text[a:b] for a, b in spans]
    assert extracted == [
        "<think>step 1</think>",
        "<thought>step 2</thought>",
        "<|thought|>step 3<|/thought|>",
        "<|channel>thought\nstep 4<channel|>",
    ]


def test_sft_config_validates_reasoning_loss_modes() -> None:
    assert SFTTrainConfig().reasoning_loss == "include"
    assert SFTTrainConfig(reasoning_loss="mask").reasoning_loss == "mask"
    assert SFTTrainConfig(reasoning_loss="reasoning_only").reasoning_loss == "reasoning_only"
    with pytest.raises(ValueError, match="reasoning_loss"):
        SFTTrainConfig(reasoning_loss="invalid")


class _CharTokenizer:
    """Minimal character tokenizer with an incremental chat template."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs
    ):
        del tokenize, tools, kwargs
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text: str, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        del add_special_tokens, kwargs
        ids = [ord(ch) + 10 for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(i - 10) for i in ids)


@pytest.mark.parametrize(
    ("thought_block", "answer"),
    [
        ("<think>compute two plus two</think>", "ANSWER_4"),
        ("<thought>check tool</thought>", '{"ok":true}'),
        ("<|thought|>plan first<|/thought|>", "done"),
        ("<|channel>thought\nplan<channel|>", "final answer"),
    ],
)
def test_build_chat_example_applies_reasoning_loss_policies(
    thought_block: str, answer: str
) -> None:
    pytest.importorskip("torch")
    from maatml.training.sft_base import build_chat_example

    tok = _CharTokenizer()
    spec = {"system": "S", "user_template": "<<USER_REQUEST>>"}
    row = {
        "sample_id": "r-1",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": f"{thought_block}\n{answer}"},
        ],
    }

    inc = build_chat_example(
        row, spec, tok, max_length=512, target_field="expected", return_spans=True
    )
    masked = build_chat_example(
        row,
        spec,
        tok,
        max_length=512,
        target_field="expected",
        return_spans=True,
        reasoning_loss="mask",
    )
    only = build_chat_example(
        row,
        spec,
        tok,
        max_length=512,
        target_field="expected",
        return_spans=True,
        reasoning_loss="reasoning_only",
    )

    def _supervised_text(ex: dict) -> str:
        kept = [lid for lid in ex["labels"] if lid != -100]
        return tok.decode(kept)

    assert thought_block in _supervised_text(inc)
    assert answer in _supervised_text(inc)
    assert inc["reasoning_tokens"] == len(thought_block)

    masked_text = _supervised_text(masked)
    assert thought_block not in masked_text
    assert answer in masked_text
    assert masked["reasoning_tokens"] == len(thought_block)

    only_text = _supervised_text(only)
    assert only_text == thought_block
    assert answer not in only_text


def test_reasoning_loss_mask_keeps_non_reasoning_rows_unchanged() -> None:
    pytest.importorskip("torch")
    from maatml.training.sft_base import build_chat_example

    tok = _CharTokenizer()
    spec = {"system": "S", "user_template": "<<USER_REQUEST>>"}
    row = {
        "sample_id": "plain-1",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    inc = build_chat_example(
        row, spec, tok, max_length=256, target_field="expected", reasoning_loss="include"
    )
    masked = build_chat_example(
        row, spec, tok, max_length=256, target_field="expected", reasoning_loss="mask"
    )
    assert inc["labels"] == masked["labels"]
