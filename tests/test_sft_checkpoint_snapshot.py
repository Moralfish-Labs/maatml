"""SFT readiness is published after saves and checked across snapshot copies."""

import json
import os
import tarfile
from pathlib import Path

import pytest

from maatml.portable import read_bundle_manifest
from maatml.training.checkpoint_snapshot import (
    READY_FILE,
    make_snapshot_ready_callback,
    publish_sft_checkpoint,
    verify_sft_checkpoint,
)
from maatml.training.sft_config import SFTTrainConfig

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX writer-lock protocol")


def checkpoint(tmp_path):
    path = tmp_path / "checkpoint-2"
    path.mkdir()
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (path / name).write_bytes(b"synthetic " + name.encode())
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    return path


def test_opt_in_default_and_complete_marker(tmp_path):
    assert SFTTrainConfig().checkpoint_snapshots is False
    path = checkpoint(tmp_path)
    publish_sft_checkpoint(path, 2, "recipe")
    verify_sft_checkpoint(path, 2, "recipe")
    publish_sft_checkpoint(path, 2, "recipe")  # Identical completion is idempotent.
    assert not (path / (READY_FILE + ".tmp")).exists()


@pytest.mark.parametrize("fault", ["optimizer", "weights", "step", "symlink"])
def test_incomplete_save_never_publishes(tmp_path, fault):
    path = checkpoint(tmp_path)
    if fault == "optimizer":
        (path / "optimizer.pt").unlink()
    elif fault == "weights":
        (path / "adapter_model.safetensors").unlink()
    elif fault == "step":
        (path / "trainer_state.json").write_text('{"global_step":1}')
    else:
        (path / "extra").symlink_to(path / "optimizer.pt")
    with pytest.raises(ValueError):
        publish_sft_checkpoint(path, 2, "recipe")
    assert not (path / READY_FILE).exists()


@pytest.mark.parametrize("fault", ["bytes", "extra", "recipe", "marker_symlink"])
def test_changed_completion_is_rejected(tmp_path, fault):
    path = checkpoint(tmp_path)
    publish_sft_checkpoint(path, 2, "recipe")
    if fault == "bytes":
        (path / "optimizer.pt").write_bytes(b"changed")
    elif fault == "extra":
        (path / "unapproved.bin").write_bytes(b"extra")
    elif fault == "marker_symlink":
        old = path / READY_FILE
        renamed = tmp_path / "outside.json"
        old.rename(renamed)
        old.symlink_to(renamed)
    with pytest.raises(ValueError):
        verify_sft_checkpoint(path, 2, "wrong" if fault == "recipe" else "recipe")


@posix_only
def test_native_snapshot_preserves_adapter_optimizer_and_completion(tmp_path, monkeypatch):
    from test_snapshot import call, fixture

    md, marker, run, old, record = fixture(tmp_path, monkeypatch)
    for path in old.iterdir():
        path.unlink()
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (old / name).write_bytes(name.encode())
    (old / "trainer_state.json").write_text('{"global_step":500}')
    publish_sft_checkpoint(old, 500, record.spec_hash)
    archive = call(md, marker, tmp_path / "sft.tar.gz")
    assert read_bundle_manifest(archive)["snapshot_step"] == 500
    with tarfile.open(archive) as bundle:
        assert "run/checkpoint-500/optimizer.pt" in bundle.getnames()
        assert "run/checkpoint-500/" + READY_FILE in bundle.getnames()


def test_real_cpu_trainer_publishes_only_completed_resumable_saves(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
        )
    )
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=2,
        per_device_train_batch_size=1,
        save_strategy="steps",
        save_steps=1,
        report_to="none",
        disable_tqdm=True,
    )
    data = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}]
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=data,
        callbacks=[make_snapshot_ready_callback("real-cpu-recipe")],
    )
    trainer.train()
    for step in (1, 2):
        path = Path(tmp_path) / f"checkpoint-{step}"
        verify_sft_checkpoint(path, step, "real-cpu-recipe")
        optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        assert optimizer["state"]
    resumed_args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "resumed"),
        use_cpu=True,
        max_steps=3,
        per_device_train_batch_size=1,
        save_strategy="steps",
        save_steps=1,
        report_to="none",
        disable_tqdm=True,
    )
    resumed = transformers.Trainer(
        model=transformers.GPT2LMHeadModel(model.config),
        args=resumed_args,
        train_dataset=data,
        callbacks=[make_snapshot_ready_callback("real-cpu-recipe")],
    )
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    assert resumed.state.global_step == 3
    restored = tmp_path / "resumed/checkpoint-3"
    verify_sft_checkpoint(restored, 3, "real-cpu-recipe")
    optimizer = torch.load(restored / "optimizer.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 3 for value in optimizer["state"].values())


@posix_only
@pytest.mark.parametrize("sft", [False, True])
def test_prune_before_enumeration_preserves_previous_backup(tmp_path, monkeypatch, sft):
    import shutil

    from test_snapshot import call, fixture

    import maatml.snapshot as snapshot
    from maatml.portable import BundleError

    md, marker, run, path, record = fixture(tmp_path, monkeypatch)
    if sft:
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (path / name).write_bytes(b"synthetic state")
        (path / "trainer_state.json").write_text('{"global_step":500}')
        publish_sft_checkpoint(path, 500, record.spec_hash)
    original = snapshot._publish

    def prune_then_publish(*args):
        shutil.rmtree(path)
        return original(*args)

    monkeypatch.setattr(snapshot, "_publish", prune_then_publish)
    out = tmp_path / "backup.tar.gz"
    out.write_bytes(b"previous verified backup")
    with pytest.raises(BundleError, match="trainer stopped"):
        call(md, marker, out)
    assert out.read_bytes() == b"previous verified backup"


def test_sharded_sft_checkpoint_publishes_and_verifies(tmp_path):
    path = tmp_path / "checkpoint-2"
    path.mkdir()
    for name in ("config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(b"synthetic " + name.encode())
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    (path / "model-00001-of-00002.safetensors").write_bytes(b"shard1")
    (path / "model-00002-of-00002.safetensors").write_bytes(b"shard2")
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "w1": "model-00001-of-00002.safetensors",
                    "w2": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    publish_sft_checkpoint(path, 2, "recipe")
    verify_sft_checkpoint(path, 2, "recipe")

    # Missing shard is rejected before publishing.
    (path / READY_FILE).unlink()
    (path / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="optimizer and weights"):
        publish_sft_checkpoint(path, 2, "recipe")

    # Path traversal in weight_map is rejected.
    (path / "model-00002-of-00002.safetensors").write_bytes(b"shard2")
    index_path.write_text(json.dumps({"weight_map": {"w1": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="optimizer and weights"):
        publish_sft_checkpoint(path, 2, "recipe")


def test_sft_snapshots_reject_no_save_and_forward_training_args(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from maatml.config import ModelDefinition
    from maatml.training import sft_base

    for bad_opts in (
        {"save_strategy": "no"},
        {"save_only_model": True},
    ):
        md = ModelDefinition(
            name="snap-guard",
            model_id="snap-guard",
            architecture="causal_sft",
            version="0.1.0",
            training={
                "model_id": "toy",
                "checkpoint_snapshots": True,
                **bad_opts,
            },
            dataset={"prompt_spec": "prompt.json"},
        )
        object.__setattr__(md, "model_dir", tmp_path)
        (tmp_path / "prompt.json").write_text('{"system":"S","user_template":"<<USER_REQUEST>>"}')
        with pytest.raises(ValueError, match="checkpoint snapshots require"):
            sft_base.train_sft(md, device="cpu")

    # Verify TrainingArguments receives custom optim/scheduler/clip/save fields.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", bos_token="a", eos_token="b"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tiny = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=16))
    monkeypatch.setattr(sft_base.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(sft_base.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: tiny)

    captured_args = []
    real_trainer = sft_base.Trainer

    class _SpyTrainer(real_trainer):
        def __init__(self, *args, **kwargs):
            captured_args.append(kwargs.get("args") or args[1])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(sft_base, "Trainer", _SpyTrainer)
    md2 = ModelDefinition(
        name="args-pass",
        model_id="args-pass",
        architecture="causal_sft",
        version="0.1.0",
        training={
            "model_id": "toy",
            "max_input_tokens": 16,
            "precision": "fp32",
            "batch_size": 1,
            "grad_accum": 1,
            "max_steps": 1,
            "disable_tqdm": True,
            "lora": {"enabled": False},
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "lr_scheduler_kwargs": {"num_cycles": 0.5},
            "max_grad_norm": 0.25,
            "save_strategy": "no",
            "save_only_model": True,
        },
        dataset={"prompt_spec": "prompt.json"},
    )
    object.__setattr__(md2, "model_dir", tmp_path / "m2")
    md2.model_dir.mkdir()
    (md2.model_dir / "prompt.json").write_text('{"system":"S","user_template":"<<USER_REQUEST>>"}')
    md2.prepared_dir.mkdir(parents=True)
    (md2.prepared_dir / "train.jsonl").write_text('{"request":"a","expected_output":"b"}\n')
    (md2.prepared_dir / "val.jsonl").write_text("")

    sft_base.train_sft(md2, device="cpu")
    args = captured_args[0]
    assert args.optim == "adamw_torch"
    assert args.lr_scheduler_type == "cosine"
    assert args.lr_scheduler_kwargs == {"num_cycles": 0.5}
    assert args.max_grad_norm == 0.25
    assert args.save_strategy == "no"
    assert args.save_only_model is True
