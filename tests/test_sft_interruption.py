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
