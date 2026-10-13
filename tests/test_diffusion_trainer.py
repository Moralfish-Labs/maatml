"""The diffusion_lora trainer: command construction, dataset layout, records.

No torch and no kohya here — the subprocess seam takes a double, because what
this module owns is the contract around the subprocess, not the optimizer.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.data.image_folder import prepare_image_caption_folder
from maatml.runs import list_runs
from maatml.training.diffusion import (
    DiffusionLoraConfig,
    build_command,
    train_diffusion_lora,
)


def _md(tmp_path: Path, *, training: dict, dataset: dict | None = None) -> ModelDefinition:
    md = ModelDefinition(
        name="dingdong-style",
        model_id="dingdong-style",
        architecture="diffusion_lora",
        version="0.1.0",
        training=training,
        dataset=dataset or {},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def _training(tmp_path: Path) -> dict:
    scripts = tmp_path / "sd-scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "sdxl_train_network.py").write_text("# kohya stand-in\n")
    return {
        "base_model": str(tmp_path / "base.safetensors"),
        "sd_scripts": str(scripts),
        "network_dim": 32,
        "network_alpha": 16,
        "batch_size": 4,
        "epochs": 8,
        "unet_lr": 1e-4,
        "text_encoder_lr": 5e-5,
    }


def _prepared_rows(tmp_path: Path, md: ModelDefinition, n: int = 4) -> None:
    md.prepared_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        img = tmp_path / f"img_{i}.png"
        img.write_bytes(b"\x89PNG-fake")
        rows.append({"image": str(img), "caption": f"anime frame {i}"})
    with open(md.prepared_dir / "train.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_command_holds_the_whole_config(tmp_path: Path) -> None:
    cfg = DiffusionLoraConfig(**_training(tmp_path))
    cmd = build_command(
        cfg,
        script_dir=Path(cfg.sd_scripts),
        dataset_root=tmp_path / "ds",
        out_dir=tmp_path / "out",
        output_name="dingdong-style",
        seed=None,
    )
    text = " ".join(cmd)
    assert "sdxl_train_network.py" in text
    assert "--network_dim 32" in text
    assert "--unet_lr 0.0001" in text
    assert "--text_encoder_lr 5e-05" in text
    assert "--enable_bucket" in text
    assert "--seed 42" in text


def test_cli_seed_overrides_config_seed(tmp_path: Path) -> None:
    cfg = DiffusionLoraConfig(**_training(tmp_path))
    cmd = build_command(
        cfg,
        script_dir=Path(cfg.sd_scripts),
        dataset_root=tmp_path / "ds",
        out_dir=tmp_path / "out",
        output_name="x",
        seed=7,
    )
    assert "--seed 7" in " ".join(cmd)


def test_unknown_training_key_is_rejected(tmp_path: Path) -> None:
    training = _training(tmp_path) | {"learning_rte": 1e-4}
    with pytest.raises(Exception, match="learning_rte"):
        DiffusionLoraConfig(**training)


def test_train_records_a_completed_run_with_parsed_metrics(tmp_path: Path) -> None:
    md = _md(tmp_path, training=_training(tmp_path))
    _prepared_rows(tmp_path, md)

    def runner(command):
        out_dir = Path(command[command.index("--output_dir") + 1])
        dataset_root = Path(command[command.index("--train_data_dir") + 1])
        # The materialized kohya layout exists before the subprocess starts.
        folder = dataset_root / "10_maatml"
        assert sorted(p.suffix for p in folder.glob("*.txt"))
        (out_dir / "dingdong-style.safetensors").write_bytes(b"weights")
        yield "steps:  12%| 460/3680 [10:00<1:00:00, 2.1s/it, avr_loss=0.0912]"
        yield "steps: 100%| 3680/3680 [2:00:00<00:00, 2.0s/it, avr_loss=0.0451]"

    result = train_diffusion_lora(md, runner=runner)
    assert result.metrics == {"steps": 3680.0, "avr_loss": 0.0451}
    runs = list_runs(md)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].metrics == result.metrics


def test_a_failing_subprocess_aborts_the_run(tmp_path: Path) -> None:
    md = _md(tmp_path, training=_training(tmp_path))
    _prepared_rows(tmp_path, md)

    def runner(command):
        raise RuntimeError("CUDA out of memory")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="out of memory"):
        train_diffusion_lora(md, runner=runner)
    runs = list_runs(md)
    assert runs[0].status == "aborted"
    assert "out of memory" in (runs[0].error or "")


def test_missing_kohya_checkout_is_an_actionable_error(tmp_path: Path) -> None:
    training = _training(tmp_path)
    training["sd_scripts"] = str(tmp_path / "nowhere")
    md = _md(tmp_path, training=training)
    with pytest.raises(FileNotFoundError, match="kohya-ss/sd-scripts"):
        train_diffusion_lora(md, runner=lambda cmd: iter(()))


def test_prepare_splits_an_image_caption_folder(tmp_path: Path) -> None:
    source = tmp_path / "frames"
    source.mkdir()
    for i in range(40):
        (source / f"frame_{i:03d}.png").write_bytes(b"\x89PNG-fake")
        (source / f"frame_{i:03d}.txt").write_text(f"caption {i}")
    (source / "no_caption.png").write_bytes(b"\x89PNG-fake")

    md = _md(
        tmp_path,
        training=_training(tmp_path),
        dataset={"format": "image_caption_folder", "source_dir": str(source)},
    )
    summary = prepare_image_caption_folder(md)
    counts = summary["split_counts"]
    assert sum(counts.values()) == 40  # the captionless image stayed out
    assert counts["val"] >= 1 and counts["test"] >= 1
    row = json.loads((md.prepared_dir / "train.jsonl").read_text().splitlines()[0])
    assert set(row) == {"sample_id", "image", "caption"}

    # Deterministic: a second prepare lands every row in the same split.
    assert prepare_image_caption_folder(md)["split_counts"] == counts


def test_the_model_folder_picks_the_subprocess_interpreter(tmp_path: Path) -> None:
    training = _training(tmp_path) | {"python": "/opt/kohya/.venv/bin/python"}
    cfg = DiffusionLoraConfig(**training)
    cmd = build_command(
        cfg,
        script_dir=Path(cfg.sd_scripts),
        dataset_root=tmp_path / "ds",
        out_dir=tmp_path / "out",
        output_name="x",
        seed=None,
    )
    assert cmd[0] == "/opt/kohya/.venv/bin/python"


def test_resume_is_an_honest_not_yet(tmp_path: Path) -> None:
    md = _md(tmp_path, training=_training(tmp_path))
    with pytest.raises(NotImplementedError, match="network_weights"):
        train_diffusion_lora(md, resume="latest", runner=lambda cmd: iter(()))


def test_a_step_budget_replaces_the_epoch_count(tmp_path: Path) -> None:
    """kohya recomputes steps from --max_train_epochs, so sending both would
    make the step cap a flag that changes nothing."""
    cfg = DiffusionLoraConfig(**(_training(tmp_path) | {"max_steps": 8}))
    text = " ".join(
        build_command(
            cfg,
            script_dir=Path(cfg.sd_scripts),
            dataset_root=tmp_path / "ds",
            out_dir=tmp_path / "out",
            output_name="x",
            seed=None,
        )
    )
    assert "--max_train_steps 8" in text
    assert "--max_train_epochs" not in text


def test_a_signal_stops_the_child_and_records_the_abort(tmp_path: Path, monkeypatch) -> None:
    """Killing maatml must not leave kohya holding the GPU under a run record
    that still says `running`."""
    from maatml.training import diffusion

    md = _md(tmp_path, training=_training(tmp_path))
    _prepared_rows(tmp_path, md)
    killed: list[int] = []

    class FakeProc:
        pid = 4242

        def __init__(self) -> None:
            self.stdout = iter(["steps:  1%| 1/3360 [00:10<10:00:00, 10.8s/it, avr_loss=0.11]"])

        def poll(self):
            return None if not killed else 0

        def wait(self):  # pragma: no cover  the signal fires first
            return 0

    monkeypatch.setattr(diffusion.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(diffusion, "_kill_process_group", lambda pid: killed.append(pid))

    # The handler is what a `kill <maatml pid>` reaches; call it as the OS would.
    def fire(*_args, **_kwargs):
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(FakeProc, "wait", fire)

    with pytest.raises(KeyboardInterrupt):
        train_diffusion_lora(md)

    assert killed == [FakeProc.pid], "the kohya process group was not signalled"
    runs = list_runs(md)
    assert runs[-1].status == "aborted"
    assert "signal" in (runs[-1].error or "")


def test_kill_process_group_falls_back_to_os_kill_without_killpg(monkeypatch) -> None:
    """Windows has no getpgid/killpg; the child itself is what we can stop."""
    from maatml.training import diffusion

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(diffusion.os, "killpg", None, raising=False)
    monkeypatch.setattr(diffusion.os, "getpgid", None, raising=False)
    monkeypatch.setattr(diffusion.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    diffusion._kill_process_group(4242)
    assert sent == [(4242, signal.SIGTERM)]
