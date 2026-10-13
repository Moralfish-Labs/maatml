"""Image-diffusion LoRA training by driving kohya's sd-scripts.

The trainer owns the lifecycle contract — run records, checkpoint layout,
``model.yml`` as the single home of every knob — and delegates the actual
optimization to ``sdxl_train_network.py`` in a kohya sd-scripts checkout,
which is the reference implementation for SDXL LoRA and not worth
reimplementing here.

Prepared rows come from ``prepare`` as ``{image, caption}`` jsonl (see
``maatml.data.image_folder``); this trainer materializes them into the
``<repeats>_<name>`` folder layout kohya reads, launches training as a
subprocess, and records what happened in ``runs.jsonl`` exactly as the text
trainers do. No torch import happens in this module: the subprocess owns the
GPU, which also means the CLI can plan and validate a diffusion model folder
on a machine that could never train it.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from ..config import ModelDefinition
from ..runs import begin_training_run, finish_run
from ..utils.io import iter_jsonl

console = Console()

# kohya prints ``steps:  12%|█▎  | 460/3680 [.., avr_loss=0.0912]`` — the two
# numbers a run record can honestly report without parsing the whole bar.
_PROGRESS_RX = re.compile(r"(\d+)/(\d+)\s*\[.*?avr_loss=([0-9.eE+-]+)")


class DiffusionLoraConfig(BaseModel):
    """The ``training:`` section for ``architecture: diffusion_lora``.

    Every knob the subprocess receives lives here, so the run record and the
    train fingerprint can report the whole of what trained the checkpoint.
    ``extra_args`` is the escape hatch for kohya flags not modeled; it is part
    of the config, so using it still fingerprints.
    """

    model_config = ConfigDict(extra="forbid")

    # What trains. ``base_model`` is a local ``.safetensors`` path or an HF id
    # kohya can resolve; ``sd_scripts`` points at a checkout because pinning a
    # trainer version is the model folder's decision, not this module's.
    base_model: str
    sd_scripts: str = "sd-scripts"
    script: str = "sdxl_train_network.py"
    # Which interpreter runs kohya. sd-scripts pins transformers 4.x while
    # maatml's own environment rides the 5.x line, so the subprocess usually
    # needs its own venv; None means this process's interpreter.
    python: Optional[str] = None

    # Dataset shaping (kohya's folder-repeats convention).
    repeats: int = 10
    resolution: str = "1024,1024"
    enable_bucket: bool = True
    min_bucket_reso: int = 640
    max_bucket_reso: int = 1536
    caption_extension: str = ".txt"

    # LoRA shape.
    network_module: str = "networks.lora"
    network_dim: int = 32
    network_alpha: int = 16

    # Optimization.
    batch_size: int = 4
    epochs: int = 8
    # A step budget, for smoke tiers. kohya recomputes its step count from
    # ``--max_train_epochs`` whenever that flag is present, so a step cap can
    # only be honoured by sending it *instead* of the epoch count — passing
    # both is a flag that silently changes nothing.
    max_steps: Optional[int] = None
    learning_rate: float = 1e-4
    unet_lr: Optional[float] = None
    text_encoder_lr: Optional[float] = None
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 100
    optimizer: str = "AdamW8bit"
    precision: str = "bf16"
    save_precision: str = "fp16"
    seed: int = 42
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    sdpa: bool = True
    no_half_vae: bool = True
    max_data_loader_n_workers: int = 4
    save_every_n_epochs: int = 2

    # Passed through to kohya verbatim, after the modeled flags.
    extra_args: list[str] = Field(default_factory=list)


class DiffusionTrainResult(BaseModel):
    """What ``_step_train`` reads back: where the weights are, what happened."""

    out_dir: str
    metrics: dict[str, float]


def _materialize_dataset(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    *,
    repeats: int,
    caption_extension: str,
    resolve: Any,
) -> int:
    """Write the ``<repeats>_<name>`` folder kohya trains from.

    Rows are ``{image, caption}``; images are linked rather than copied when
    the filesystem allows it, because a training set can be gigabytes and the
    prepared jsonl already owns the canonical copy decision.
    """
    folder = dataset_root / f"{repeats}_maatml"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    folder.mkdir(parents=True)
    written = 0
    for i, row in enumerate(rows):
        image = row.get("image")
        if not image:
            raise ValueError(f"prepared row {i} has no 'image' field: {row}")
        src = Path(resolve(image))
        if not src.is_file():
            raise FileNotFoundError(f"prepared row {i} names a missing image: {src}")
        dst = folder / f"{i:05d}{src.suffix.lower()}"
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        dst.with_suffix(caption_extension).write_text(
            str(row.get("caption", "")).strip() + "\n", encoding="utf-8"
        )
        written += 1
    if not written:
        raise ValueError("no rows to train on")
    return written


def _kill_process_group(pid: int) -> None:
    """Stop the training child and anything it spawned.

    POSIX: kill the process group created by ``start_new_session``. Windows
    has no ``getpgid`` / ``killpg``; SIGTERM on the child is what we can
    actually deliver there.
    """
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is not None and getpgid is not None:
        killpg(getpgid(pid), signal.SIGTERM)
        return
    os.kill(pid, signal.SIGTERM)


@contextmanager
def _stopping(proc: "subprocess.Popen[str]") -> Any:
    """Pass SIGTERM/SIGINT on to the training subprocess, then re-raise.

    Two things go wrong without this. The subprocess outlives the CLI that
    started it, holding the GPU and writing checkpoints nobody is recording;
    and the run stays ``running`` in ``runs.jsonl`` forever, because a default
    SIGTERM ends the process without unwinding to the handler that would mark
    it aborted. Raising ``KeyboardInterrupt`` from the handler turns a signal
    back into an exception, which is the path that already records the abort.
    """
    previous: dict[int, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        with contextlib.suppress(ProcessLookupError, OSError):
            _kill_process_group(proc.pid)
        raise KeyboardInterrupt(f"stopped by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):  # not the main thread
            previous[sig] = signal.signal(sig, stop)
    try:
        yield
    finally:
        for saved, handler in previous.items():
            with contextlib.suppress(ValueError):
                signal.signal(saved, handler)
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                _kill_process_group(proc.pid)


def build_command(
    cfg: DiffusionLoraConfig,
    *,
    script_dir: Path,
    dataset_root: Path,
    out_dir: Path,
    output_name: str,
    seed: Optional[int],
) -> list[str]:
    """The exact subprocess argv, pure so a test can hold it to the config."""
    cmd = [
        cfg.python or sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        cfg.precision,
        str(script_dir / cfg.script),
        "--pretrained_model_name_or_path",
        cfg.base_model,
        "--train_data_dir",
        str(dataset_root),
        "--output_dir",
        str(out_dir),
        "--output_name",
        output_name,
        "--resolution",
        cfg.resolution,
        *(
            ["--max_train_steps", str(cfg.max_steps)]
            if cfg.max_steps is not None
            else ["--max_train_epochs", str(cfg.epochs)]
        ),
        "--network_module",
        cfg.network_module,
        "--network_dim",
        str(cfg.network_dim),
        "--network_alpha",
        str(cfg.network_alpha),
        "--train_batch_size",
        str(cfg.batch_size),
        "--learning_rate",
        str(cfg.learning_rate),
        "--lr_scheduler",
        cfg.lr_scheduler,
        "--lr_warmup_steps",
        str(cfg.lr_warmup_steps),
        "--optimizer_type",
        cfg.optimizer,
        "--mixed_precision",
        cfg.precision,
        "--save_precision",
        cfg.save_precision,
        "--caption_extension",
        cfg.caption_extension,
        "--save_every_n_epochs",
        str(cfg.save_every_n_epochs),
        "--seed",
        str(cfg.seed if seed is None else seed),
        "--max_data_loader_n_workers",
        str(cfg.max_data_loader_n_workers),
    ]
    if cfg.unet_lr is not None:
        cmd += ["--unet_lr", str(cfg.unet_lr)]
    if cfg.text_encoder_lr is not None:
        cmd += ["--text_encoder_lr", str(cfg.text_encoder_lr)]
    if cfg.enable_bucket:
        cmd += [
            "--enable_bucket",
            "--min_bucket_reso",
            str(cfg.min_bucket_reso),
            "--max_bucket_reso",
            str(cfg.max_bucket_reso),
        ]
    if cfg.cache_latents:
        cmd.append("--cache_latents")
    if cfg.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if cfg.sdpa:
        cmd.append("--sdpa")
    if cfg.no_half_vae:
        cmd.append("--no_half_vae")
    cmd += list(cfg.extra_args)
    return cmd


def train_diffusion_lora(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    runner: Any = None,
) -> DiffusionTrainResult:
    """Train an image LoRA from the prepared splits via kohya sd-scripts.

    ``runner`` is the subprocess seam (tests pass a double); production leaves
    it None and gets ``subprocess.Popen`` streaming through our console so the
    kohya progress bar stays visible.
    """
    if resume is not None:
        raise NotImplementedError(
            "diffusion_lora does not resume yet. Continuing a LoRA maps to kohya's "
            "--network_weights; pass it via training.extra_args until resume is wired."
        )
    training = dict(model_def.merged_smoke() if smoke else model_def.training)
    cfg = DiffusionLoraConfig(**training)

    script_dir = Path(model_def.resolve(cfg.sd_scripts))
    if not (script_dir / cfg.script).is_file():
        raise FileNotFoundError(
            f"kohya script not found: {script_dir / cfg.script}. Point training.sd_scripts "
            "at a kohya-ss/sd-scripts checkout (git clone --depth 1 "
            "https://github.com/kohya-ss/sd-scripts)."
        )

    rows = list(iter_jsonl(model_def.prepared_dir / "train.jsonl"))
    if limit is not None:
        rows = rows[:limit]

    run, out_dir, _resume = begin_training_run(
        model_def, smoke=smoke, device=device, profile="subprocess"
    )
    dataset_root = out_dir / "kohya_dataset"
    try:
        count = _materialize_dataset(
            rows,
            dataset_root,
            repeats=cfg.repeats,
            caption_extension=cfg.caption_extension,
            resolve=model_def.resolve,
        )
        command = build_command(
            cfg,
            script_dir=script_dir,
            dataset_root=dataset_root,
            out_dir=out_dir,
            output_name=model_def.identity,
            seed=seed,
        )
        console.print(
            f"[cyan]diffusion_lora train[/]: run={run.run_id} rows={count} "
            f"base={Path(cfg.base_model).name} dim={cfg.network_dim}"
        )

        last_step, total_steps, last_loss = 0, 0, float("nan")
        if runner is not None:
            for line in runner(command):
                match = _PROGRESS_RX.search(line)
                if match:
                    last_step, total_steps = int(match.group(1)), int(match.group(2))
                    last_loss = float(match.group(3))
        else:
            # New process group, so a signal that stops maatml can be passed on
            # to kohya. Without it, killing the CLI leaves the trainer running:
            # an orphan that still holds the GPU, still writes checkpoints, and
            # is no longer attached to any run record.
            proc = subprocess.Popen(
                command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with _stopping(proc):
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")  # noqa: T201  training progress goes to stdout
                    match = _PROGRESS_RX.search(line)
                    if match:
                        last_step, total_steps = int(match.group(1)), int(match.group(2))
                        last_loss = float(match.group(3))
                code = proc.wait()
            if code != 0:
                raise RuntimeError(f"kohya exited {code}; last step {last_step}/{total_steps}")

        weights = sorted(out_dir.glob("*.safetensors"))
        if not weights:
            raise RuntimeError(f"training subprocess wrote no .safetensors under {out_dir}")

        metrics = {
            "steps": float(last_step),
            "avr_loss": last_loss,
        }
        finish_run(model_def, run.run_id, "completed", metrics=metrics)
        return DiffusionTrainResult(out_dir=str(out_dir), metrics=metrics)
    except BaseException as exc:
        finish_run(model_def, run.run_id, "aborted", error=f"{type(exc).__name__}: {exc}")
        raise
