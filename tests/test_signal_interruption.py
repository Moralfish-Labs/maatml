"""Signal-driven graceful stop and `"interrupted"` run status."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from maatml.cli import app
from maatml.config import ModelDefinition
from maatml.runs import finish_run, list_runs, start_run
from maatml.training.signals import (
    TrainingInterrupted,
    TrainingStopController,
    make_signal_stop_callback,
)


def _model_def(tmp_path: Path, *, arch: str = "causal_sft") -> ModelDefinition:
    tmp_path.mkdir(parents=True, exist_ok=True)
    md = ModelDefinition(
        name="sig-test",
        model_id="sig-test",
        architecture=arch,
        version="0.1.0",
        training={
            "model_id": "local-test-model",
            **({"heads": [{"name": "h", "labels": ["a", "b"]}]} if arch == "classifier" else {}),
        },
        dataset={"prompt_spec": "prompt.json"},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    (tmp_path / "prompt.json").write_text('{"system": "S", "user_template": "<<USER_REQUEST>>"}')
    md.prepared_dir.mkdir(parents=True, exist_ok=True)
    (md.prepared_dir / "train.jsonl").write_text('{"request":"a","expected_output":"b"}\n')
    (md.prepared_dir / "val.jsonl").write_text("")
    return md


def test_stop_controller_defers_first_signal_in_train_loop_and_restores_handler() -> None:
    transformers = pytest.importorskip("transformers")
    prev_term = signal.getsignal(signal.SIGTERM)
    ctrl = TrainingStopController()
    cb = make_signal_stop_callback(ctrl)
    control = transformers.TrainerControl()

    with ctrl.installed():
        assert signal.getsignal(signal.SIGTERM) != prev_term
        with ctrl.training_loop():
            os.kill(os.getpid(), signal.SIGTERM)
            assert ctrl.stop_requested is True
            assert ctrl.signum == signal.SIGTERM
            cb.on_step_end(None, None, control)
            assert control.should_save is True
            assert control.should_training_stop is True
            # Second signal inside training loop raises immediately.
            with pytest.raises(TrainingInterrupted) as exc_info:
                os.kill(os.getpid(), signal.SIGTERM)
            assert exc_info.value.signum == signal.SIGTERM

    assert signal.getsignal(signal.SIGTERM) == prev_term
    with pytest.raises(TrainingInterrupted):
        ctrl.raise_if_interrupted()


def test_stop_controller_raises_immediately_outside_training_loop() -> None:
    ctrl = TrainingStopController()
    with ctrl.installed(), pytest.raises(TrainingInterrupted) as exc_info:
        os.kill(os.getpid(), signal.SIGTERM)
    assert exc_info.value.signum == signal.SIGTERM


def test_finish_run_interrupted_is_resumable(tmp_path: Path) -> None:
    from maatml.runs import latest_incomplete_run

    md = _model_def(tmp_path)
    rec = start_run(md, smoke=False, device="cpu", profile="cpu")
    (Path(rec.out_dir) / "checkpoint-10").mkdir(parents=True)
    updated = finish_run(md, rec.run_id, "interrupted", error="Interrupted by SIGTERM (15)")
    assert updated is not None and updated.status == "interrupted"

    runs = list_runs(md)
    assert [r.status for r in runs] == ["interrupted"]
    resumable = latest_incomplete_run(md)
    assert resumable is not None and resumable.run_id == rec.run_id


def test_trainers_mark_run_interrupted_on_training_interrupted(tmp_path: Path, monkeypatch) -> None:
    from maatml.training import multi_head, preference, seq2seq, sft_base

    # 1. sft_base
    md_sft = _model_def(tmp_path / "sft")
    monkeypatch.setattr(
        sft_base,
        "iter_jsonl",
        lambda *_a, **_kw: (_ for _ in ()).throw(TrainingInterrupted(signal.SIGTERM)),
    )
    with pytest.raises(TrainingInterrupted):
        sft_base.train_sft(md_sft, device="cpu")
    assert [r.status for r in list_runs(md_sft)] == ["interrupted"]

    # 2. seq2seq
    md_seq = _model_def(tmp_path / "seq", arch="seq2seq")
    monkeypatch.setattr(
        seq2seq,
        "iter_jsonl",
        lambda *_a, **_kw: (_ for _ in ()).throw(TrainingInterrupted(signal.SIGTERM)),
    )
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **kw: None),
    )
    with pytest.raises(TrainingInterrupted):
        seq2seq.train_seq2seq_model(md_seq, device="cpu")
    assert [r.status for r in list_runs(md_seq)] == ["interrupted"]

    # 3. multi_head
    md_mh = _model_def(tmp_path / "mh", arch="classifier")
    monkeypatch.setattr(
        multi_head,
        "iter_jsonl",
        lambda *_a, **_kw: (_ for _ in ()).throw(TrainingInterrupted(signal.SIGTERM)),
    )
    with pytest.raises(TrainingInterrupted):
        multi_head.train_multi_head(md_mh, device="cpu")
    assert [r.status for r in list_runs(md_mh)] == ["interrupted"]

    # 4. preference
    md_pref = _model_def(tmp_path / "pref", arch="dpo")
    monkeypatch.setattr(preference, "_require_trl", lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "datasets", SimpleNamespace(Dataset=object))
    monkeypatch.setitem(
        __import__("sys").modules, "peft", SimpleNamespace(LoraConfig=object, TaskType=object)
    )
    monkeypatch.setattr(
        preference,
        "_load_preference_rows",
        lambda *_a, **_kw: (_ for _ in ()).throw(TrainingInterrupted(signal.SIGTERM)),
    )
    with pytest.raises(TrainingInterrupted):
        preference.train_dpo(md_pref, device="cpu")
    assert [r.status for r in list_runs(md_pref)] == ["interrupted"]


def test_cli_train_exits_with_signal_code_on_training_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    from maatml.registry import TRAINERS

    (tmp_path / "model.yml").write_text(
        "name: sig\nmodel_id: sig\narchitecture: sig_test_arch\nversion: 0.1.0\n",
        encoding="utf-8",
    )

    def _interrupted_trainer(*args, **kwargs):
        del args, kwargs
        raise TrainingInterrupted(signal.SIGTERM, "SIGTERM")

    TRAINERS.register("sig_test_arch", _interrupted_trainer, source="test")
    try:
        result = CliRunner().invoke(app, ["train", str(tmp_path)])
    finally:
        TRAINERS.unregister("sig_test_arch")

    assert result.exit_code == 128 + int(signal.SIGTERM)
    assert "interrupted" in result.output.lower()


def test_real_cpu_sft_sigterm_flushes_checkpoint_and_marks_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from maatml.runs import latest_incomplete_run
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

    md = _model_def(tmp_path)
    md.training.update(
        {
            "max_input_tokens": 16,
            "precision": "fp32",
            "batch_size": 1,
            "grad_accum": 1,
            "max_steps": 4,
            "save_steps": 100,
            "disable_tqdm": True,
            "lora": {"enabled": False},
        }
    )
    (md.prepared_dir / "train.jsonl").write_text(
        "".join(f'{{"sample_id":"r{i}","request":"a","expected_output":"b"}}\n' for i in range(4))
    )

    # Fire SIGTERM after step 1 completes so SignalStopCallback flushes checkpoint-1
    # and stops before step 4 without waiting until end of training.
    orig_step = sft_base.Trainer.training_step
    fired = False

    def _step_then_sigterm(self, *args, **kwargs):
        nonlocal fired
        loss = orig_step(self, *args, **kwargs)
        if not fired:
            fired = True
            os.kill(os.getpid(), signal.SIGTERM)
        return loss

    monkeypatch.setattr(sft_base.Trainer, "training_step", _step_then_sigterm)

    with pytest.raises(TrainingInterrupted):
        sft_base.train_sft(md, device="cpu")

    runs = list_runs(md)
    assert [r.status for r in runs] == ["interrupted"]
    ckpt_dir = Path(runs[0].out_dir) / "checkpoint-1"
    assert ckpt_dir.is_dir()
    assert not (Path(runs[0].out_dir) / "run_metadata.json").exists()
    resumable = latest_incomplete_run(md)
    assert resumable is not None and resumable.run_id == runs[0].run_id
