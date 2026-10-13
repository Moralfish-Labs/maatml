"""Generic encoder-decoder (seq2seq) trainer.

Full fine-tune of an encoder-decoder LM (e.g. flan-t5). Input is the
request text optionally prefixed with ``dataset.source_prefix``; target
is the serialised ``dataset.target_field`` value.

Public surface:
  - ``Seq2SeqConfig``: typed config built from ``model.yml::training``
  - ``train_seq2seq_model(model_def, ...)``: entry point invoked by the CLI
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..device import (
    effective_dataloader_workers,
    resolve_training_placement,
)
from ..runs import begin_training_run, finish_run, normalize_report_to
from ..utils.io import iter_jsonl
from .guards import ensure_tokenizer_model_contract, make_nan_guard_callback, write_run_metadata
from .load import from_pretrained_kwargs
from .schedule import precision_flags, total_training_steps
from .schedule import warmup_steps as resolve_warmup_steps
from .sft_config import reject_unknown_training_keys, validate_precision


@dataclass
class GenerationCfg:
    num_beams: int = 1
    max_new_tokens: int = 512


# Every `training:` key this architecture reads, here and in the trainer body.
_SEQ2SEQ_KEYS = frozenset(
    {
        "model_id",
        "source_max_len",
        "target_max_len",
        "batch_size",
        "grad_accum",
        "learning_rate",
        "epochs",
        "weight_decay",
        "warmup_ratio",
        "seed",
        "precision",
        "grad_checkpointing",
        "eval_steps",
        "save_steps",
        "logging_steps",
        "max_steps",
        "generation",
        "attn_implementation",
        "dataloader_workers",
        "model_revision",
        "keep_checkpoints",
        "embedding_strategy",
        "report_to",
        "group_by_length",
    }
)


@dataclass
class Seq2SeqConfig:
    model_id: str = "google/flan-t5-base"
    source_max_len: int = 1024
    target_max_len: int = 512
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 3.0e-5
    epochs: float = 6.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 1337
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 250
    logging_steps: int = 25
    max_steps: int = -1
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    attn_implementation: Optional[str] = None
    dataloader_workers: Optional[int] = None
    model_revision: Optional[str] = None
    keep_checkpoints: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "Seq2SeqConfig":
        reject_unknown_training_keys(d, _SEQ2SEQ_KEYS, architecture="seq2seq")
        gen = d.get("generation") or {}
        return cls(
            model_id=d.get("model_id", "google/flan-t5-base"),
            source_max_len=int(d.get("source_max_len", 1024)),
            target_max_len=int(d.get("target_max_len", 512)),
            batch_size=int(d.get("batch_size", 8)),
            grad_accum=int(d.get("grad_accum", 2)),
            learning_rate=float(d.get("learning_rate", 3.0e-5)),
            # Fractional epochs are honoured (parity with the SFT config);
            # int() silently floored `epochs: 0.5` to zero epochs of training.
            epochs=float(d.get("epochs", 6.0)),
            weight_decay=float(d.get("weight_decay", 0.01)),
            warmup_ratio=float(d.get("warmup_ratio", 0.06)),
            seed=int(d.get("seed", 1337)),
            precision=validate_precision(d.get("precision", "bf16")),
            grad_checkpointing=bool(d.get("grad_checkpointing", False)),
            eval_steps=int(d.get("eval_steps", 9999)),
            save_steps=int(d.get("save_steps", 250)),
            logging_steps=int(d.get("logging_steps", 25)),
            max_steps=int(d.get("max_steps", -1)),
            generation=GenerationCfg(
                num_beams=int(gen.get("num_beams", 1)),
                max_new_tokens=int(gen.get("max_new_tokens", 512)),
            ),
            attn_implementation=d.get("attn_implementation"),
            dataloader_workers=(
                int(d["dataloader_workers"]) if d.get("dataloader_workers") is not None else None
            ),
            model_revision=d.get("model_revision"),
            keep_checkpoints=int(d.get("keep_checkpoints", 2)),
        )


@dataclass
class Seq2SeqResult:
    out_dir: Path
    metrics: dict
    train_runtime: float


def has_target(row: dict, target_field: str) -> bool:
    """Does this row carry a usable seq2seq target?

    A missing / empty / falsy target used to serialise to the literal string
    ``"{}"`` (or ``""``), so the model was trained to emit that. Rows that
    cannot supply a target are counted and dropped, never silently trained on.
    """
    value = row.get(target_field)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _drop_targetless(rows: list[dict], target_field: str) -> tuple[list[dict], int]:
    """Split rows into (usable, dropped_count) by :func:`has_target`."""
    kept = [row for row in rows if has_target(row, target_field)]
    return kept, len(rows) - len(kept)


def _serialise_target(target, *, key_order: Optional[list[str]] = None) -> str:
    """Canonical compact-JSON (or passthrough string) serialisation of the target."""
    if target is None or (hasattr(target, "__len__") and len(target) == 0):
        raise ValueError(
            "seq2seq target is empty; rows without a target must be dropped "
            "before tokenization (see has_target)"
        )
    if isinstance(target, str):
        return target
    if not isinstance(target, dict):
        return json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    if key_order:
        ordered = {k: target.get(k) for k in key_order if k in target}
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    # Preserve insertion order when present; otherwise sort for stability.
    return json.dumps(dict(target), ensure_ascii=False, separators=(",", ":"))


def _build_dataset(
    rows: list[dict],
    tokenizer,
    source_max_len: int,
    target_max_len: int,
    *,
    source_prefix: str = "",
    target_key_order: Optional[list[str]] = None,
    request_field: str = "request",
    target_field: str = "target",
):
    """Build a seq2seq dataset from prepared JSONL."""
    import torch
    from torch.utils.data import Dataset

    class _Seq2SeqDataset(Dataset):
        def __init__(self, samples: list[dict]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            row = self.samples[idx]
            request = row.get(request_field, "")
            target_val = row.get(target_field)

            source = source_prefix + request
            target = _serialise_target(target_val, key_order=target_key_order)

            src = tokenizer(
                source,
                max_length=source_max_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            tgt = tokenizer(
                target,
                max_length=target_max_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            pad_id = tokenizer.pad_token_id
            labels = [tid if tid != pad_id else -100 for tid in tgt["input_ids"]]

            return {
                "input_ids": torch.tensor(src["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(src["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    return _Seq2SeqDataset(rows)


def _train_loop(
    cfg: Seq2SeqConfig,
    train_ds,
    val_ds,
    out_dir: Path,
    tokenizer,
    *,
    profile,
    embedding_strategy: Optional[str],
    model_def: ModelDefinition,
    dataset_dir: Path,
    smoke: bool,
    device_name: str,
    run_id: str,
    report_to,
    group_by_length: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    distributed: bool = False,
) -> Seq2SeqResult:
    from transformers import (
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)

    load_kwargs = from_pretrained_kwargs(
        profile,
        precision=cfg.precision,
        attn_implementation=cfg.attn_implementation,
        revision=cfg.model_revision,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id, **load_kwargs)
    ensure_tokenizer_model_contract(model, tokenizer, embedding_strategy=embedding_strategy)
    model.generation_config.max_new_tokens = cfg.generation.max_new_tokens
    model.generation_config.num_beams = cfg.generation.num_beams

    total_steps = total_training_steps(
        len(train_ds),
        batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
    )
    warmup_steps = resolve_warmup_steps(total_steps, cfg.warmup_ratio)

    use_bf16, use_fp16 = precision_flags(cfg.precision, device=device_name, distributed=distributed)
    use_grad_ckpt = bool(cfg.grad_checkpointing) and profile.allow_grad_checkpointing
    run_eval = val_ds is not None and profile.allow_mid_train_eval and cfg.eval_steps < total_steps
    num_workers = effective_dataloader_workers(profile, cfg.dataloader_workers)

    args_kwargs: dict = {
        "output_dir": str(out_dir),
        "run_name": run_id,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": warmup_steps,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": "steps" if run_eval else "no",
        "eval_steps": cfg.eval_steps if run_eval else None,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.keep_checkpoints,
        "seed": cfg.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": use_grad_ckpt,
        "dataloader_num_workers": num_workers,
        "report_to": report_to,
        "optim": "adamw_torch",
        "max_steps": cfg.max_steps,
        "predict_with_generate": False,
        "generation_max_length": cfg.generation.max_new_tokens,
        "generation_num_beams": cfg.generation.num_beams,
        "remove_unused_columns": False,
        "use_cpu": (not distributed) and str(device_name).startswith("cpu"),
    }
    # transformers≥5 dropped group_by_length from TrainingArguments.
    import inspect

    if "group_by_length" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters:
        args_kwargs["group_by_length"] = bool(group_by_length)
    args = Seq2SeqTrainingArguments(**args_kwargs)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if run_eval else None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[make_nan_guard_callback()],
    )

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics: dict = {}
    if val_ds is not None:
        profile.empty_cache()
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    write_run_metadata(
        out_dir,
        model_def,
        {
            "train": dataset_dir / "train.jsonl",
            "val": dataset_dir / "val.jsonl",
        },
        extra={
            "run_id": run_id,
            "smoke": smoke,
            "device": device_name,
            "profile": profile.name,
            "distributed": distributed,
            "model_revision": cfg.model_revision,
        },
    )

    return Seq2SeqResult(
        out_dir=out_dir,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(
            getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
            if hasattr(train_output, "metrics")
            else 0.0
        ),
    )


def train_seq2seq_model(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    trial: Optional[dict] = None,
) -> Seq2SeqResult:
    """Train a seq2seq model from a ``ModelDefinition``."""
    from transformers import AutoTokenizer

    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = Seq2SeqConfig.from_dict(training_dict)
    if seed is not None:
        cfg.seed = seed

    ds_cfg = get_dataset_cfg(model_def)
    source_prefix = ds_cfg.get("source_prefix") or ""
    target_key_order = ds_cfg.get("target_key_order")  # None → as-is / sorted in serialise
    request_field = ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request"
    target_field = ds_cfg.get("target_field") or "target"

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
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

    # Everything after the run record exists runs under the finish handler:
    # a tokenizer download failure or an unreadable split must mark the run
    # `aborted`, not leave it `running` forever.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, use_fast=True, revision=cfg.model_revision
        )

        train_rows = list(iter_jsonl(dataset_dir / "train.jsonl"))
        val_rows = list(iter_jsonl(dataset_dir / "val.jsonl"))
        if limit is not None:
            train_rows = train_rows[:limit]
            val_rows = val_rows[: max(2, limit // 4)]
        if not train_rows:
            raise ValueError(f"No training rows in {dataset_dir / 'train.jsonl'}")

        train_rows, dropped_train = _drop_targetless(train_rows, target_field)
        val_rows, dropped_val = _drop_targetless(val_rows, target_field)
        if dropped_train or dropped_val:
            print(  # noqa: T201  training progress goes to stdout
                f"seq2seq: dropped {dropped_train} train / {dropped_val} val rows "
                f"with no {target_field!r} (they would have trained on an empty target)"
            )
        if not train_rows:
            raise ValueError(
                f"Every training row in {dataset_dir / 'train.jsonl'} is missing "
                f"{target_field!r}; check dataset.target_field"
            )

        print(  # noqa: T201  training progress goes to stdout
            f"seq2seq: run={run.run_id} model={cfg.model_id} train={len(train_rows)} "
            f"val={len(val_rows)} src_len={cfg.source_max_len} tgt_len={cfg.target_max_len} "
            f"epochs={cfg.epochs}"
        )

        key_order = list(target_key_order) if target_key_order else None
        train_ds = _build_dataset(
            train_rows,
            tokenizer,
            cfg.source_max_len,
            cfg.target_max_len,
            source_prefix=source_prefix,
            target_key_order=key_order,
            request_field=request_field,
            target_field=target_field,
        )
        val_ds = (
            _build_dataset(
                val_rows,
                tokenizer,
                cfg.source_max_len,
                cfg.target_max_len,
                source_prefix=source_prefix,
                target_key_order=key_order,
                request_field=request_field,
                target_field=target_field,
            )
            if val_rows
            else None
        )

        report_to = normalize_report_to(training_dict.get("report_to"))
        group_by_length = bool(training_dict.get("group_by_length", False))
        result = _train_loop(
            cfg,
            train_ds,
            val_ds,
            out_dir,
            tokenizer,
            profile=profile,
            embedding_strategy=training_dict.get("embedding_strategy"),
            model_def=model_def,
            dataset_dir=dataset_dir,
            smoke=smoke,
            device_name=str(target_device),
            run_id=run.run_id,
            report_to=report_to,
            group_by_length=group_by_length,
            resume_from_checkpoint=str(resume_path) if resume_path else None,
            distributed=distributed,
        )
        finish_run(model_def, run.run_id, "completed", metrics=result.metrics)
        return result
    except Exception as exc:
        finish_run(model_def, run.run_id, "aborted", error=str(exc))
        raise


# Alias kept for builtins / callers that prefer the short name.
train_seq2seq = train_seq2seq_model
