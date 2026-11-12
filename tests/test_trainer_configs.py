"""Config parsing and gold-label mapping for the trainers (no torch needed).

seq2seq / multi_head keep their torch imports inside functions, so their
config surface is unit-testable on the torch-free matrix. The tokenization
and label-masking tests live in test_trainers_torch.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from maatml.data.preference import as_completion_text, normalize_preference
from maatml.training.multi_head import (
    HeadSpec,
    MultiHeadConfig,
    UnknownLabelError,
    _label_index,
    parse_heads,
    scan_label_coverage,
)
from maatml.training.seq2seq import (
    Seq2SeqConfig,
    _drop_targetless,
    _serialise_target,
    has_target,
)
from maatml.training.sft_config import SFTTrainConfig, validate_precision

# --- precision -------------------------------------------------------------


def test_precision_validated_at_parse_time() -> None:
    assert validate_precision("bf16") == "bf16"
    for cfg_cls, kwargs in (
        (Seq2SeqConfig.from_dict, {"precision": "bfloat16"}),
        (MultiHeadConfig.from_dict, {"precision": "float16"}),
    ):
        with pytest.raises(ValueError, match="training.precision must be one of"):
            cfg_cls(kwargs)
    with pytest.raises(ValueError, match="training.precision must be one of"):
        SFTTrainConfig(precision="fp8")


# --- fractional epochs -----------------------------------------------------


def test_fractional_epochs_survive_config_parse() -> None:
    assert Seq2SeqConfig.from_dict({"epochs": 0.5}).epochs == 0.5
    assert (
        MultiHeadConfig.from_dict(
            {"epochs": 0.25, "heads": [{"name": "h", "labels": ["a", "b"]}]}
        ).epochs
        == 0.25
    )
    # Parity with the SFT config, which already modelled epochs as a float.
    assert SFTTrainConfig(epochs=0.5).epochs == 0.5


# --- multi_head heads ------------------------------------------------------


def test_absent_or_malformed_heads_is_an_error() -> None:
    for training in ({}, {"heads": []}, {"heads": "validity"}, {"heads": {}}):
        with pytest.raises(ValueError, match="training.heads must be"):
            parse_heads(training)


def test_legacy_jcl_head_shape_still_parses_when_its_keys_are_present() -> None:
    heads = parse_heads({"heads": {"error_codes": ["missing_dd", "none"]}})
    assert [h.name for h in heads] == ["validity", "error_code", "severity", "line"]
    assert parse_heads({"head_loss_weights": {"validity": 2.0}})[0].loss_weight == 2.0


def test_bool_gold_honours_declared_label_order() -> None:
    assert _label_index(True, ["invalid", "valid"]) == 1
    assert _label_index(False, ["invalid", "valid"]) == 0
    # Reversed declaration: True must still mean "valid", not index 1.
    assert _label_index(True, ["valid", "invalid"]) == 0
    assert _label_index(False, ["valid", "invalid"]) == 1
    # Two labels with no boolean-ish names fall back to False→0, True→1.
    assert _label_index(True, ["off", "on"]) == 1


def test_unknown_gold_label_raises_instead_of_mapping_to_none() -> None:
    with pytest.raises(UnknownLabelError):
        _label_index("not_a_declared_code", ["missing_dd", "none"])
    with pytest.raises(UnknownLabelError):
        _label_index(None, ["a", "b"])
    assert _label_index(None, ["a", "none"]) == 1


def test_scan_label_coverage_counts_unknown_gold() -> None:
    heads = [
        HeadSpec(name="code", kind="classification", labels=["a", "none"], target_path="code"),
        HeadSpec(name="line", kind="line_pointer", target_path="line"),
    ]
    rows = [
        {"target": {"code": "a"}},
        {"target": {"code": "zzz"}},
        {"target": {"code": "zzz"}},
    ]
    assert scan_label_coverage(rows, heads, target_field="target") == {"code": {"'zzz'": 2}}
    assert scan_label_coverage(rows[:1], heads, target_field="target") == {}


# --- seq2seq targets -------------------------------------------------------


def test_falsy_targets_are_dropped_not_serialised_to_braces() -> None:
    rows = [
        {"target": {"a": 1}},
        {"target": {}},
        {"target": None},
        {"target": ""},
        {},
    ]
    kept, dropped = _drop_targetless(rows, "target")
    assert kept == [{"target": {"a": 1}}]
    assert dropped == 4
    assert has_target({"target": "text"}, "target") is True
    with pytest.raises(ValueError, match="target is empty"):
        _serialise_target({})


def test_serialise_target_honours_key_order() -> None:
    assert _serialise_target({"b": 2, "a": 1}, key_order=["a", "b"]) == '{"a":1,"b":2}'


# --- preference rows -------------------------------------------------------


def test_structured_completions_serialise_as_json_not_repr() -> None:
    assert as_completion_text({"a": 1, "b": None}) == '{"a":1,"b":null}'
    assert as_completion_text(["x"]) == '["x"]'
    assert as_completion_text("already text") == "already text"
    row = normalize_preference({"prompt": "p", "chosen": {"ok": True}, "rejected": {"ok": False}})
    assert row["chosen"] == '{"ok":true}'
    assert "'" not in row["chosen"]


def test_identical_chosen_and_rejected_warns() -> None:
    with pytest.warns(UserWarning, match="identical chosen and rejected"):
        normalize_preference({"prompt": "p", "chosen": "same", "rejected": "same"})


def test_lora_typo_is_rejected_not_silently_dropped() -> None:
    """LoraSettings is strict like its siblings: a typo under training.lora
    would otherwise train at the default while the run looks legitimately fresh
    (training_config is hashed into the lifecycle fingerprint)."""
    with pytest.raises(ValidationError):
        SFTTrainConfig(lora={"enabled": True, "alpah": 32})
    # The correctly spelled key still parses.
    assert SFTTrainConfig(lora={"enabled": True, "alpha": 32}).lora.alpha == 32


def test_lora_target_modules_take_a_list_or_one_regular_expression() -> None:
    """A list names suffixes or full module names; a string is one expression
    for the whole name, which is what a base with shared projection names in
    its other towers needs."""
    expression = (
        r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    )
    assert SFTTrainConfig(lora={"target_modules": expression}).lora.target_modules == expression
    assert SFTTrainConfig().lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert SFTTrainConfig(lora={"target_modules": ["q_proj"]}).lora.target_modules == ["q_proj"]
    with pytest.raises(ValidationError, match="not a regular expression"):
        SFTTrainConfig(lora={"target_modules": "layers.(q_proj"})
    with pytest.raises(ValidationError, match="empty expression"):
        SFTTrainConfig(lora={"target_modules": " "})
    with pytest.raises(ValidationError):
        SFTTrainConfig(lora={"target_modules": 7})


# --- shared schedule / precision helpers ------------------------------------


def test_total_steps_divides_by_world_size() -> None:
    """Each rank sees len(dataset)/world_size rows. Sizing the schedule as
    though the run were single-process made a 0.06 warmup ratio cover ~48% of
    an 8-GPU schedule."""
    from maatml.training.schedule import total_training_steps, warmup_steps

    kwargs = {"batch_size": 2, "grad_accum": 8, "epochs": 3.0}
    single = total_training_steps(1600, processes=1, **kwargs)
    eight = total_training_steps(1600, processes=8, **kwargs)
    assert single == 300
    assert eight == 37, "8 ranks each see an eighth of the corpus"
    # The warmup ratio now lands on the real schedule instead of 8x it.
    assert warmup_steps(eight, 0.06) == 2
    assert warmup_steps(single, 0.06) == 18


def test_total_steps_honours_explicit_max_steps() -> None:
    from maatml.training.schedule import total_training_steps

    assert total_training_steps(10_000, batch_size=2, grad_accum=8, epochs=3.0, max_steps=50) == 50


@pytest.mark.parametrize("device", ["cpu", "cpu:0", None])
def test_precision_flags_off_without_accelerator(device) -> None:
    """transformers rejects bf16=True on an unsupported CPU. seq2seq and
    multi_head passed it unguarded, so the same config trained for causal_sft
    and hard-failed for them on the same host."""
    from maatml.training.schedule import precision_flags

    assert precision_flags("bf16", device=device) == (False, False)
    assert precision_flags("fp16", device=device) == (False, False)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps"])
def test_precision_flags_on_for_accelerators(device) -> None:
    from maatml.training.schedule import precision_flags

    assert precision_flags("bf16", device=device) == (True, False)
    assert precision_flags("fp16", device=device) == (False, True)


def test_precision_flags_allowed_when_distributed() -> None:
    """Under torchrun the Trainer owns placement, so the device string is not
    the deciding factor."""
    from maatml.training.schedule import precision_flags

    assert precision_flags("bf16", device="cpu", distributed=True) == (True, False)


def test_all_trainers_share_one_precision_derivation() -> None:
    """Regression guard for the drift itself: no trainer may re-derive the
    mixed-precision flags locally."""
    import pathlib

    for name in ("sft_base", "seq2seq", "multi_head", "preference"):
        src = pathlib.Path(f"src/maatml/training/{name}.py").read_text()
        assert "precision_flags(" in src, f"{name} does not use the shared helper"
        assert 'cfg.precision == "bf16"' not in src, f"{name} re-derives use_bf16"


def test_dpo_and_orpo_share_one_config_body() -> None:
    """DPO and ORPO previously duplicated a 42-line config block that differed
    only in two class names, so a one-sided edit changed one method silently.

    Structural guard: the trl config is constructed once. This path needs the
    [pref] extra to execute, so nothing runs it in the offline suite.
    """
    import pathlib

    src = pathlib.Path("src/maatml/training/preference.py").read_text()
    assert src.count("PreferenceConfig(") == 1, "config built in more than one place"
    assert "DPOConfig(" not in src and "ORPOConfig(" not in src
    # Both methods must still be reachable.
    assert "DPOConfig as PreferenceConfig" in src
    assert "ORPOConfig as PreferenceConfig" in src


def test_seq2seq_and_multi_head_reject_unknown_training_keys() -> None:
    """Parity with the pydantic configs' extra="forbid". These two build
    themselves with d.get(...), so a typo silently trained at the built-in
    default while the run still looked fresh."""
    from maatml.training.multi_head import MultiHeadConfig
    from maatml.training.seq2seq import Seq2SeqConfig

    with pytest.raises(ValueError, match="learning_rat"):
        Seq2SeqConfig.from_dict({"learning_rat": 3e-5})
    with pytest.raises(ValueError, match="learning_rat"):
        MultiHeadConfig.from_dict({"learning_rat": 3e-5, "heads": [{"name": "x"}]})

    # Correct spelling still parses.
    assert Seq2SeqConfig.from_dict({"learning_rate": 3e-5}).learning_rate == 3e-5


def test_shipped_seq2seq_config_survives_the_strict_key_check() -> None:
    """The key set is hand-maintained, so pin it against the real model:
    a missing entry would reject a config that is actually valid."""
    from maatml.config import load_model_def
    from maatml.training.seq2seq import Seq2SeqConfig

    md = load_model_def("examples/vision-describer")
    Seq2SeqConfig.from_dict(dict(md.training))
    # smoke maps base_model -> model_id and drops smoke-only keys.
    Seq2SeqConfig.from_dict(md.merged_smoke())


def test_trainer_and_predictor_resolve_identical_special_tokens(tmp_path) -> None:
    """Train/eval tokenization skew: the trainer rebound pad_token to the
    tokenizer file's name while the predictor kept the built-in <PAD>, so a
    tokenizer declaring [PAD] trained and evaluated with different pad tokens."""
    import json

    from maatml.evaluation.predictors import _tokenizer_specials
    from maatml.training.multi_head import _default_special_tokens
    from maatml.utils.tokenizers import resolve_special_tokens

    def _tok(names):
        path = tmp_path / f"tok_{abs(hash(tuple(names)))}.json"
        path.write_text(
            json.dumps({"added_tokens": [{"content": n, "special": True} for n in names]}),
            encoding="utf-8",
        )
        return path

    for names in (
        ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<COL1>", "<CONT>"],
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        [],
    ):
        path = _tok(names)
        assert _default_special_tokens(path) == _tokenizer_specials(path)

    # The square-bracket tokenizer is the case that used to diverge.
    square = resolve_special_tokens(_tok(["[PAD]", "[UNK]"]))
    assert square["pad_token"] == "[PAD]"
    assert square["unk_token"] == "[UNK]"

    # A missing or unreadable tokenizer falls back rather than raising.
    assert resolve_special_tokens(None)["pad_token"] == "<PAD>"
    assert resolve_special_tokens(tmp_path / "nope.json")["pad_token"] == "<PAD>"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert resolve_special_tokens(bad)["pad_token"] == "<PAD>"
