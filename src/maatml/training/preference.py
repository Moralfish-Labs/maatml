"""DPO / ORPO preference trainers via TRL (optional ``maatml[pref]`` extra).

Registered as architectures ``dpo`` and ``orpo``. Requires ``trl>=0.9``;
missing TRL raises a clear install hint.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import ModelDefinition
from ..data.preference import as_completion_text
from ..device import (
    effective_dataloader_workers,
    resolve_training_placement,
)
from ..progress import JobProgressReporter, make_progress_callback, resolve_progress_file
from ..runs import begin_training_run, finish_run, normalize_report_to
from ..utils.io import iter_jsonl
from .guards import make_nan_guard_callback, write_run_metadata
from .load import from_pretrained_kwargs, maybe_prepare_kbit
from .schedule import precision_flags
from .sft_base import _maybe_attach_lora, _refuse_quantized_merge, _save_sft_artifacts
from .sft_config import LoraSettings, QuantizationSettings, validate_precision
from .signals import TrainingInterrupted, TrainingStopController, make_signal_stop_callback


class PreferenceTrainConfig(BaseModel):
    """Shared training knobs for DPO / ORPO."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-0.6B"
    max_input_tokens: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accum: int = Field(default=8, gt=0)
    learning_rate: float = 5e-5
    epochs: float = 1.0
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    seed: int = 7331
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 200
    logging_steps: int = 10
    max_steps: int = -1
    beta: float = 0.1  # DPO / ORPO beta
    lora: LoraSettings = Field(default_factory=LoraSettings)
    quantization: Optional[QuantizationSettings] = None
    attn_implementation: Optional[str] = None
    dataloader_workers: Optional[int] = None
    report_to: Any = None
    model_revision: Optional[str] = None

    @field_validator("precision")
    @classmethod
    def _check_precision(cls, v: str) -> str:
        return validate_precision(v)


@dataclass
class PreferenceTrainResult:
    out_dir: Path
    metrics: dict[str, float]
    train_runtime: float


def _require_trl():
    try:
        import trl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Preference training (DPO/ORPO) requires TRL; install maatml[pref]"
        ) from exc


def _load_preference_rows(path: Path, limit: Optional[int]) -> tuple[list[dict], int]:
    """Load ``{prompt, chosen, rejected}`` rows; returns (rows, n_identical).

    Structured completions are JSON-serialised, not ``str()``-ed: a dict
    completion used to reach the trainer as its Python ``repr``
    (``{'a': 1}``), teaching the model single quotes and ``None``.
    """
    rows = list(iter_jsonl(path))
    if limit is not None:
        rows = rows[:limit]
    cleaned: list[dict] = []
    identical = 0
    for row in rows:
        chosen = as_completion_text(row["chosen"])
        rejected = as_completion_text(row["rejected"])
        if chosen == rejected:
            identical += 1
        cleaned.append(
            {
                "prompt": as_completion_text(row["prompt"]),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return cleaned, identical


def train_preference(
    model_def: ModelDefinition,
    *,
    method: str = "dpo",
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
    progress_file: Optional[str | Path] = None,
) -> PreferenceTrainResult:
    """Train with TRL ``DPOTrainer`` or ``ORPOTrainer``."""
    _require_trl()
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    method = method.lower().strip()
    if method not in ("dpo", "orpo"):
        raise ValueError(f"method must be 'dpo' or 'orpo'; got {method!r}")

    training_dict = dict(model_def.merged_smoke() if smoke else model_def.training)
    for drop in ("generation", "heads", "head_loss_weights", "embedding_strategy"):
        training_dict.pop(drop, None)
    cfg = PreferenceTrainConfig(**training_dict)
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    quant = cfg.quantization if cfg.quantization and cfg.quantization.enabled() else None
    _refuse_quantized_merge(
        cfg.lora.save_mode,
        lora_enabled=cfg.lora.enabled,
        quantized=quant is not None,
    )
    target_device, profile, distributed = resolve_training_placement(device)
    run, out_dir, resume_path = begin_training_run(
        model_def,
        smoke=smoke,
        device=str(target_device),
        profile=profile.name,
        out_dir=out_dir,
        resume=resume,
        trial=trial,
    )
    reporter = JobProgressReporter(
        resolve_progress_file(progress_file),
        run_id=run.run_id,
        artifact_uri=str(out_dir),
    )
    stop_controller = TrainingStopController()

    # The run record exists from here on, so row loading failures abort/interrupt it too.
    try:
        with stop_controller.installed():
            reporter.update(stage="train")
            train_rows, identical_train = _load_preference_rows(dataset_dir / "train.jsonl", limit)
            val_limit = None if limit is None else max(2, limit // 4)
            val_rows, identical_val = _load_preference_rows(dataset_dir / "val.jsonl", val_limit)
            if not train_rows:
                raise ValueError(f"No training rows in {dataset_dir / 'train.jsonl'}")
            if identical_train or identical_val:
                warnings.warn(
                    f"{identical_train} train / {identical_val} val preference rows have "
                    "identical chosen and rejected completions; they carry no preference "
                    "signal",
                    stacklevel=2,
                )

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs = from_pretrained_kwargs(
                profile,
                precision=cfg.precision,
                attn_implementation=cfg.attn_implementation,
                quantization=quant,
                revision=cfg.model_revision,
                distributed=distributed,
            )
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
            model = maybe_prepare_kbit(model, quant)

            peft_config = None
            if cfg.lora.enabled:
                peft_config = LoraConfig(
                    r=cfg.lora.r,
                    lora_alpha=cfg.lora.alpha,
                    lora_dropout=cfg.lora.dropout,
                    target_modules=cfg.lora.target_modules,
                    task_type=TaskType.CAUSAL_LM,
                )
                if quant is None:
                    model = _maybe_attach_lora(model, cfg.lora)
                    peft_config = None  # already attached

            train_ds = Dataset.from_list(train_rows)
            eval_ds = Dataset.from_list(val_rows) if val_rows else None

            use_bf16, use_fp16 = precision_flags(
                cfg.precision, device=target_device, distributed=distributed
            )
            num_workers = effective_dataloader_workers(profile, cfg.dataloader_workers)
            report_to = normalize_report_to(cfg.report_to)

            # DPO and ORPO differ only in which TRL config/trainer pair they use.
            # Keeping one body means a change to the argument set cannot land for
            # one method and silently miss the other.
            if method == "dpo":
                from trl import DPOConfig as PreferenceConfig
                from trl import DPOTrainer as PreferenceTrainer
            else:
                from trl import ORPOConfig as PreferenceConfig
                from trl import ORPOTrainer as PreferenceTrainer

            args = PreferenceConfig(
                output_dir=str(out_dir),
                run_name=run.run_id,
                per_device_train_batch_size=cfg.batch_size,
                per_device_eval_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.grad_accum,
                learning_rate=cfg.learning_rate,
                num_train_epochs=cfg.epochs,
                weight_decay=cfg.weight_decay,
                warmup_ratio=cfg.warmup_ratio,
                logging_steps=cfg.logging_steps,
                save_steps=cfg.save_steps,
                save_total_limit=2,
                seed=cfg.seed,
                bf16=use_bf16,
                fp16=use_fp16,
                gradient_checkpointing=bool(cfg.grad_checkpointing)
                and profile.allow_grad_checkpointing,
                dataloader_num_workers=num_workers,
                report_to=report_to,
                max_steps=cfg.max_steps,
                beta=cfg.beta,
                max_length=cfg.max_input_tokens,
                max_prompt_length=cfg.max_input_tokens // 2,
                remove_unused_columns=False,
                use_cpu=(not distributed) and target_device.type == "cpu",
            )
            callbacks = [make_nan_guard_callback(), make_signal_stop_callback(stop_controller)]
            if reporter.enabled:
                callbacks.append(make_progress_callback(reporter))
            trainer_kwargs: dict[str, Any] = {
                "model": model,
                "args": args,
                "train_dataset": train_ds,
                "eval_dataset": eval_ds,
                "processing_class": tokenizer,
                "callbacks": callbacks,
            }
            if peft_config is not None:
                trainer_kwargs["peft_config"] = peft_config
            try:
                trainer = PreferenceTrainer(**trainer_kwargs)
            except TypeError:
                # Older TRL takes `tokenizer` rather than `processing_class`.
                trainer_kwargs.pop("processing_class", None)
                trainer_kwargs["tokenizer"] = tokenizer
                trainer = PreferenceTrainer(**trainer_kwargs)

            with stop_controller.training_loop():
                train_output = trainer.train(
                    resume_from_checkpoint=str(resume_path) if resume_path else None
                )
            stop_controller.raise_if_interrupted()
            metrics_out: dict[str, float] = {}
            train_metrics = getattr(train_output, "metrics", None) or {}
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    metrics_out[str(k)] = float(v)

            # Same saver as SFT, so training.lora.save_mode (merged | adapter |
            # both) means the same thing for DPO/ORPO as it does for causal SFT.
            # TRL may have attached the adapter itself when peft_config was passed.
            trained_model = getattr(trainer, "model", model)
            save_meta = _save_sft_artifacts(
                trained_model,
                tokenizer,
                out_dir,
                save_mode=cfg.lora.save_mode if cfg.lora.enabled else "merged",
                base_model_id=cfg.model_id,
                quantized=quant is not None,
            )

            write_run_metadata(
                out_dir,
                model_def,
                {
                    "train": dataset_dir / "train.jsonl",
                    "val": dataset_dir / "val.jsonl",
                },
                extra={
                    "run_id": run.run_id,
                    "smoke": smoke,
                    "device": str(target_device),
                    "profile": profile.name,
                    "method": method,
                    "distributed": distributed,
                    "precision": cfg.precision,
                    "model_revision": cfg.model_revision,
                    **save_meta,
                },
            )
            finish_run(model_def, run.run_id, "completed", metrics=metrics_out)
            reporter.update(stage="completed")
            return PreferenceTrainResult(
                out_dir=out_dir,
                metrics=metrics_out,
                train_runtime=float(metrics_out.get("train_runtime", 0.0)),
            )
    except TrainingInterrupted as exc:
        reporter.update(stage="interrupted")
        finish_run(model_def, run.run_id, "interrupted", error=str(exc))
        raise
    except (Exception, KeyboardInterrupt) as exc:
        reporter.update(stage="failed")
        finish_run(model_def, run.run_id, "aborted", error=str(exc) or type(exc).__name__)
        raise


def train_dpo(model_def: ModelDefinition, **kwargs: Any) -> PreferenceTrainResult:
    return train_preference(model_def, method="dpo", **kwargs)


def train_orpo(model_def: ModelDefinition, **kwargs: Any) -> PreferenceTrainResult:
    return train_preference(model_def, method="orpo", **kwargs)
