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
