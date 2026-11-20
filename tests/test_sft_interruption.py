"""An interrupted native SFT run must not remain eligible as a live run."""

from __future__ import annotations

import json

import pytest

from maatml.config import ModelDefinition
from maatml.runs import latest_completed_run, list_runs


@pytest.mark.parametrize("phase", ["splits", "tokenizer"])
@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("load failed")])
def test_sft_records_abort_and_reraises(tmp_path, monkeypatch, phase, failure):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from maatml.training import sft_base

    model = ModelDefinition(
        name="interrupted-sft",
        model_id="interrupted-sft",
        architecture="causal_sft",
        version="0.1.0",
        training={"model_id": "local-test-model"},
        dataset={"prompt_spec": "prompt.json"},
    )
    object.__setattr__(model, "model_dir", tmp_path)
    (tmp_path / "prompt.json").write_text(json.dumps({"system": "Test"}))
    model.prepared_dir.mkdir(parents=True)
    (model.prepared_dir / "train.jsonl").write_text('{"request":"a","expected":"b"}\n')
    (model.prepared_dir / "val.jsonl").write_text("")

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise failure

    if phase == "splits":
        monkeypatch.setattr(sft_base, "iter_jsonl", interrupt)
    else:
        monkeypatch.setattr(sft_base.AutoTokenizer, "from_pretrained", interrupt)

    with pytest.raises(type(failure)) as raised:
        sft_base.train_sft(model, device="cpu")
    assert raised.value is failure
    records = list_runs(model)
    assert len(records) == 1
    assert records[0].status == "aborted"
    assert records[0].error == (str(failure) or type(failure).__name__)
    assert records[0].finished_at is not None
    assert latest_completed_run(model) is None
    assert not list(model.checkpoints_dir.rglob("*.safetensors"))


def test_sft_aborts_before_training_on_a_row_longer_than_max_input_tokens(tmp_path, monkeypatch):
    """The whole split is tokenized before the first step, so one long row
    stops the run there: no optimisation, no checkpoint, an aborted record."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer

    from maatml.training import sft_base

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", bos_token="a", eos_token="b"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tiny = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=16))
    monkeypatch.setattr(sft_base.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(sft_base.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: tiny)

    def no_training(self, *args, **kwargs):
        raise AssertionError("training started on a split with an over-long row")

    monkeypatch.setattr(Trainer, "train", no_training)

    model = ModelDefinition(
        name="long-row-sft",
        model_id="long-row-sft",
        architecture="causal_sft",
        version="0.1.0",
        training={
            "model_id": "local-test-model",
            "max_input_tokens": 6,
            "precision": "fp32",
            "lora": {"enabled": False},
        },
        dataset={"prompt_spec": "prompt.json"},
    )
    object.__setattr__(model, "model_dir", tmp_path)
    (tmp_path / "prompt.json").write_text(
        json.dumps({"system": "a", "user_template": "<<USER_REQUEST>>"})
    )
    model.prepared_dir.mkdir(parents=True)
    rows = [
        {"sample_id": "short", "request": "a b", "expected_output": "b"},
        {"sample_id": "long", "request": "a b a b a b", "expected_output": "b"},
    ]
    (model.prepared_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (model.prepared_dir / "val.jsonl").write_text("")

    with pytest.raises(
        sft_base.RowTooLongError, match="row long: 8 tokens exceed max_input_tokens=6"
    ):
        sft_base.train_sft(model, device="cpu")
    records = list_runs(model)
    assert [r.status for r in records] == ["aborted"]
    assert "row long" in records[0].error
    assert latest_completed_run(model) is None
    assert not list(model.checkpoints_dir.rglob("*.safetensors"))
