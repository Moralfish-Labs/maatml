"""Config-driven multi-head classifier trainer.

Full fine-tune of an encoder (e.g. ModernBERT) with N classification /
line_pointer heads declared in ``training.heads``. Loss is a weighted sum
across heads. Optional ``dataset.text_transform`` looks up the TRANSFORMS
registry before tokenization; optional ``dataset.tokenizer`` loads a
custom tokenizer.json.

Public surface:
  - ``MultiHeadConfig``: typed config from ``model.yml::training``
  - ``train_multi_head(model_def, ...)``: CLI entry point
  - ``_resolve_path`` / ``parse_heads``: shared by trainer + predictor
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..device import (
    effective_dataloader_workers,
    resolve_training_placement,
)
from ..registry import TRANSFORMS
from ..runs import begin_training_run, finish_run, normalize_report_to
from ..utils.io import iter_jsonl
from ..utils.tokenizers import resolve_special_tokens as _default_special_tokens
from .guards import ensure_tokenizer_model_contract, make_nan_guard_callback, write_run_metadata
from .load import from_pretrained_kwargs
from .schedule import precision_flags, total_training_steps
from .schedule import warmup_steps as resolve_warmup_steps
from .sft_config import reject_unknown_training_keys, validate_precision

_PATH_TOKEN = re.compile(r"([^[.\]]+)|\[(\d+)\]")


def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path with optional ``[idx]`` segments.

    Examples: ``valid``, ``errors[0].code``, ``a.b[1].c``.
    Missing segments → ``None``.
    """
    if not path:
        return obj
    cur: Any = obj
    for match in _PATH_TOKEN.finditer(path):
        key, idx = match.group(1), match.group(2)
        if cur is None:
            return None
        if key is not None:
            cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
        else:
            i = int(idx)
            if isinstance(cur, (list, tuple)) and 0 <= i < len(cur):
                cur = cur[i]
            else:
                return None
    return cur


@dataclass
class HeadSpec:
    name: str
    kind: str  # classification | line_pointer
    labels: list[str] = field(default_factory=list)
    target_path: str = ""
    loss_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "labels": list(self.labels),
            "target_path": self.target_path,
            "loss_weight": self.loss_weight,
        }


_LEGACY_HEAD_KEYS = ("error_codes", "severities")


def parse_heads(training_dict: dict) -> list[HeadSpec]:
    """Parse ``training.heads`` (list or dict) into ``HeadSpec`` list.

    Also accepts the legacy ``head_loss_weights`` + ``heads.error_codes`` /
    ``heads.severities`` shape used by older jcl model.yml files. That
    fallback fires only when those legacy keys are actually present: an absent
    or malformed ``training.heads`` is a configuration error, not a silent
    switch to JCL's four hardcoded heads.
    """
    raw = training_dict.get("heads")
    if isinstance(raw, list) and raw:
        out: list[HeadSpec] = []
        for item in raw:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError(f"Invalid head entry: {item!r}")
            out.append(
                HeadSpec(
                    name=str(item["name"]),
                    kind=str(item.get("kind", "classification")),
                    labels=list(item.get("labels") or []),
                    target_path=str(item.get("target_path") or ""),
                    loss_weight=float(item.get("loss_weight", 1.0)),
                )
            )
        return out

    # Legacy jcl shape: heads.error_codes / heads.severities + head_loss_weights
    legacy = raw if isinstance(raw, dict) else {}
    weights = training_dict.get("head_loss_weights") or {}
    has_legacy_keys = any(k in legacy for k in _LEGACY_HEAD_KEYS) or bool(weights)
    if not has_legacy_keys:
        raise ValueError(
            "training.heads must be a non-empty list of head specs "
            "(name / kind / labels / target_path / loss_weight). "
            f"Got {raw!r}. The legacy JCL head shape (heads.error_codes / "
            "heads.severities / head_loss_weights) is still accepted when "
            "those keys are present."
        )
    error_codes = list(
        legacy.get("error_codes")
        or [
            "missing_dd",
            "invalid_job_card",
            "unresolved_symbolic_parameter",
            "continuation_error",
            "invalid_exec_statement",
            "invalid_dataset_reference_structure",
            "other",
            "none",
        ]
    )
    severities = list(legacy.get("severities") or ["error", "warning", "info", "none"])
    return [
        HeadSpec(
            name="validity",
            kind="classification",
            labels=["invalid", "valid"],
            target_path="valid",
            loss_weight=float(weights.get("validity", 1.0)),
        ),
        HeadSpec(
            name="error_code",
            kind="classification",
            labels=error_codes,
            target_path="errors[0].code",
            loss_weight=float(weights.get("error_code", 1.0)),
        ),
        HeadSpec(
            name="severity",
            kind="classification",
            labels=severities,
            target_path="errors[0].severity",
            loss_weight=float(weights.get("severity", 0.5)),
        ),
        HeadSpec(
            name="line",
            kind="line_pointer",
            labels=[],
            target_path="errors[0].line",
            loss_weight=float(weights.get("line", 0.3)),
        ),
    ]


class UnknownLabelError(ValueError):
    """A gold value that does not map to any declared head label."""


_TRUE_ALIASES = ("true", "valid", "yes", "pass", "ok")
_FALSE_ALIASES = ("false", "invalid", "no", "fail")


def _label_index(value: Any, labels: list[str]) -> int:
    """Map a gold value to a class index, or raise :class:`UnknownLabelError`.

    Unknown values used to land on ``none`` (or index 0), so a typo'd or
    out-of-vocabulary gold label trained the model to predict the wrong class
    while every count still looked healthy.

    Booleans map through the declared labels, not through position: ``True``
    picks whichever of ``true`` / ``valid`` / ``yes`` / ``pass`` / ``ok`` the
    head declares (``False`` likewise), so ``labels: [valid, invalid]`` and
    ``labels: [invalid, valid]`` both behave. A two-label head with no
    recognisable names falls back to False → index 0, True → index 1.
    """
    if isinstance(value, bool):
        aliases = _TRUE_ALIASES if value else _FALSE_ALIASES
        for alias in aliases:
            if alias in labels:
                return labels.index(alias)
        if len(labels) == 2:
            return 1 if value else 0
        raise UnknownLabelError(
            f"boolean gold {value!r} does not map to any of {labels!r}; "
            "declare a boolean-looking label pair (e.g. [invalid, valid])"
        )
    if value is None or value == "":
        if "none" in labels:
            return labels.index("none")
        raise UnknownLabelError(f"missing gold value and no 'none' label declared in {labels!r}")
    s = str(value)
    if s in labels:
        return labels.index(s)
    raise UnknownLabelError(f"gold label {s!r} is not one of {labels!r}")


def scan_label_coverage(
    rows: list[dict], heads: list[HeadSpec], *, target_field: str
) -> dict[str, dict[str, int]]:
    """Count gold values that no classification head label covers.

    Runs before training so an unmappable corpus fails loudly at startup
    instead of quietly training every unknown row as class 0 / ``none``.
    """
    from collections import Counter

    unknown: dict[str, Counter] = {}
    for row in rows:
        gold = row.get(target_field)
        if not isinstance(gold, dict):
            gold = {}
        for head in heads:
            if head.kind == "line_pointer":
                continue
            raw = _resolve_path(gold, head.target_path)
            try:
                _label_index(raw, head.labels)
            except UnknownLabelError:
                unknown.setdefault(head.name, Counter())[repr(raw)] += 1
    return {name: dict(counter.most_common(5)) for name, counter in unknown.items()}


def _line_start_offset(text: str, line_no: int) -> int:
    """1-based line number → character offset of that line's start."""
    if line_no <= 1:
        return 0
    count = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            count += 1
            if count == line_no - 1:
                return i + 1
    return len(text)


# Every `training:` key this architecture reads, here, in parse_heads, and in
# the trainer body.
_MULTI_HEAD_KEYS = frozenset(
    {
        "model_id",
        "max_input_tokens",
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
        "attn_implementation",
        "dataloader_workers",
        "model_revision",
        "keep_checkpoints",
        "heads",
        "head_loss_weights",
        "embedding_strategy",
        "report_to",
    }
)


@dataclass
class MultiHeadConfig:
    model_id: str = "answerdotai/ModernBERT-base"
    max_input_tokens: int = 2048
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 2.0e-5
    epochs: float = 4.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 1337
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 250
    logging_steps: int = 25
    max_steps: int = -1
    heads: list[HeadSpec] = field(default_factory=list)
    attn_implementation: Optional[str] = None
    dataloader_workers: Optional[int] = None
    model_revision: Optional[str] = None
    keep_checkpoints: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "MultiHeadConfig":
        reject_unknown_training_keys(d, _MULTI_HEAD_KEYS, architecture="multi_head")
        return cls(
            model_id=d.get("model_id", "answerdotai/ModernBERT-base"),
            max_input_tokens=int(d.get("max_input_tokens", 2048)),
            batch_size=int(d.get("batch_size", 16)),
            grad_accum=int(d.get("grad_accum", 1)),
            learning_rate=float(d.get("learning_rate", 2.0e-5)),
            # Fractional epochs are honoured (parity with the SFT config).
            epochs=float(d.get("epochs", 4.0)),
            weight_decay=float(d.get("weight_decay", 0.01)),
            warmup_ratio=float(d.get("warmup_ratio", 0.06)),
            seed=int(d.get("seed", 1337)),
            precision=validate_precision(d.get("precision", "bf16")),
            grad_checkpointing=bool(d.get("grad_checkpointing", False)),
            eval_steps=int(d.get("eval_steps", 9999)),
            save_steps=int(d.get("save_steps", 250)),
            logging_steps=int(d.get("logging_steps", 25)),
            max_steps=int(d.get("max_steps", -1)),
            heads=parse_heads(d),
            attn_implementation=d.get("attn_implementation"),
            dataloader_workers=(
                int(d["dataloader_workers"]) if d.get("dataloader_workers") is not None else None
            ),
            model_revision=d.get("model_revision"),
            keep_checkpoints=int(d.get("keep_checkpoints", 2)),
        )


@dataclass
class MultiHeadResult:
    out_dir: Path
    metrics: dict
    train_runtime: float


def _build_head_module(hidden_size: int, heads: list[HeadSpec]):
    """Construct an nn.ModuleDict of Linear heads."""
    from torch import nn

    class MultiHeadModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            modules: dict[str, nn.Module] = {}
            for h in heads:
                if h.kind == "line_pointer":
                    modules[h.name] = nn.Linear(hidden_size, 2)
                else:
                    n = max(2, len(h.labels))
                    modules[h.name] = nn.Linear(hidden_size, n)
            self.heads = nn.ModuleDict(modules)

        def forward(self, sequence_output, pooled):
            out: dict[str, Any] = {}
            for h in heads:
                x = sequence_output if h.kind == "line_pointer" else pooled
                out[f"{h.name}_logits"] = self.heads[h.name](x)
            return out

    return MultiHeadModule()


def _build_dataset(
    rows: list[dict],
    tokenizer,
    max_length: int,
    heads: list[HeadSpec],
    *,
    request_field: str,
    target_field: str,
    text_transform,
):
    """Flatten prepared JSONL rows into per-head classifier targets."""
    import torch
    from torch.utils.data import Dataset

    class _MultiHeadDataset(Dataset):
        def __init__(self, samples: list[dict]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            row = self.samples[idx]
            text = row.get(request_field, "") or ""
            gold = row.get(target_field) or {}
            if not isinstance(gold, dict):
                gold = {}

            pre = text_transform(text) if text_transform else text
            enc = tokenizer(
                pre,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors=None,
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            offsets = enc.get("offset_mapping") or [(0, 0)] * len(input_ids)

            item: dict[str, Any] = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

            for h in heads:
                raw = _resolve_path(gold, h.target_path)
                if h.kind == "line_pointer":
                    err_line = int(raw or 0) if raw is not None else 0
                    line_labels = [-100] * len(input_ids)
                    if err_line > 0:
                        start_char = _line_start_offset(pre, err_line)
                        end_char = _line_start_offset(pre, err_line + 1)
                        for i, (s, e) in enumerate(offsets):
                            if attention_mask[i] == 0:
                                continue
                            if s == e == 0:
                                line_labels[i] = -100
                                continue
                            line_labels[i] = 1 if (s >= start_char and s < end_char) else 0
                    else:
                        for i, (s, e) in enumerate(offsets):
                            if attention_mask[i] == 0 or s == e == 0:
                                continue
                            line_labels[i] = 0
                    item[f"{h.name}_labels"] = torch.tensor(line_labels, dtype=torch.long)
                else:
                    item[f"{h.name}_label"] = torch.tensor(
                        _label_index(raw, h.labels), dtype=torch.long
                    )

            return item

    return _MultiHeadDataset(rows)


def _train_loop(
    cfg: MultiHeadConfig,
    train_ds,
    val_ds,
    out_dir: Path,
    tokenizer,
    *,
    heads: list[HeadSpec],
    profile,
    embedding_strategy: Optional[str],
    model_def: ModelDefinition,
    dataset_dir: Path,
    smoke: bool,
    device_name: str,
    run_id: str,
    report_to,
    resume_from_checkpoint: Optional[str] = None,
    distributed: bool = False,
) -> MultiHeadResult:
    from torch import nn
    from transformers import AutoModel, Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)

    load_kwargs = from_pretrained_kwargs(
        profile,
        precision=cfg.precision,
        attn_implementation=cfg.attn_implementation,
        revision=cfg.model_revision,
    )
    encoder = AutoModel.from_pretrained(cfg.model_id, **load_kwargs)
    ensure_tokenizer_model_contract(encoder, tokenizer, embedding_strategy=embedding_strategy)
    hidden = encoder.config.hidden_size
    head_module = _build_head_module(hidden, heads)

    class MultiHeadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.heads = head_module.heads
            self._head_specs = heads
            self.ce = nn.CrossEntropyLoss()
            self.ce_line = nn.CrossEntropyLoss(ignore_index=-100)

        def forward(self, input_ids, attention_mask, **label_kwargs):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]
            logits: dict[str, Any] = {}
            for h in self._head_specs:
                x = seq if h.kind == "line_pointer" else pooled
                logits[f"{h.name}_logits"] = self.heads[h.name](x)

            loss = None
            head_losses: dict[str, Any] = {}
            has_labels = any(
                f"{h.name}_label" in label_kwargs or f"{h.name}_labels" in label_kwargs
                for h in self._head_specs
            )
            if has_labels:
                total = 0.0
                for h in self._head_specs:
                    if h.kind == "line_pointer":
                        ll = label_kwargs.get(f"{h.name}_labels")
                        if ll is None:
                            continue
                        head_loss = self.ce_line(
                            logits[f"{h.name}_logits"].view(-1, 2),
                            ll.view(-1),
                        )
                    else:
                        lab = label_kwargs.get(f"{h.name}_label")
                        if lab is None:
                            continue
                        head_loss = self.ce(logits[f"{h.name}_logits"], lab)
                    head_losses[h.name] = head_loss
                    total = total + h.loss_weight * head_loss
                loss = total
            return {"loss": loss, "head_losses": head_losses, **logits}

    model = MultiHeadModel()

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

    args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_id,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if run_eval else "no",
        eval_steps=cfg.eval_steps if run_eval else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.keep_checkpoints,
        seed=cfg.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=use_grad_ckpt,
        dataloader_num_workers=num_workers,
        report_to=report_to,
        optim="adamw_torch",
        max_steps=cfg.max_steps,
        remove_unused_columns=False,
        use_cpu=(not distributed) and str(device_name).startswith("cpu"),
    )

    class _MultiHeadTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._last_head_losses: dict[str, float] = {}

        # num_items_in_batch is named rather than swept into **kwargs so the
        # override stays signature-compatible with transformers' Trainer. It is
        # unused here; the multi-head loss is already reduced by the model.
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
        ):
            outputs = model(**inputs)
            loss = outputs["loss"]
            head_losses = outputs.get("head_losses") or {}
            stashed: dict[str, float] = {}
            for name, hl in head_losses.items():
                try:
                    stashed[name] = float(hl.detach().item())
                except Exception:  # noqa: BLE001
                    continue
            self._last_head_losses = stashed
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], *args, **kwargs):
            merged = dict(logs)
            for name, val in self._last_head_losses.items():
                merged[f"loss_{name}"] = val
            return super().log(merged, *args, **kwargs)

    trainer = _MultiHeadTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if run_eval else None,
        processing_class=tokenizer,
        callbacks=[make_nan_guard_callback()],
    )

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics: dict = {}
    if val_ds is not None:
        profile.empty_cache()
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    model.eval()
    encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    from safetensors.torch import save_file

    head_state = {
        f"heads.{name}.{param_name}": tensor.detach().cpu()
        for name, head in model.heads.items()
        for param_name, tensor in head.state_dict().items()
    }
    save_file(head_state, out_dir / "classifier_heads.safetensors")

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
            "heads": [h.to_dict() for h in heads],
        },
    )

    return MultiHeadResult(
        out_dir=out_dir,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(
            getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
            if hasattr(train_output, "metrics")
            else 0.0
        ),
    )


def train_multi_head(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    tokenizer_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
) -> MultiHeadResult:
    """Train a multi-head classifier from a ``ModelDefinition``."""
    from transformers import AutoTokenizer

    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = MultiHeadConfig.from_dict(training_dict)
    if seed is not None:
        cfg.seed = seed
    heads = cfg.heads
    if not heads:
        raise ValueError("training.heads must declare at least one head")

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

    ds_cfg = get_dataset_cfg(model_def)
    request_field = ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request"
    target_field = ds_cfg.get("target_field") or "target"

    # From here on the run record exists, so every failure (tokenizer load,
    # unreadable split, unmappable labels) must mark it `aborted`.
    try:
        text_transform = None
        transform_name = ds_cfg.get("text_transform")
        if transform_name:
            text_transform = TRANSFORMS.require(str(transform_name))

        resolved_tokenizer = (
            Path(tokenizer_path)
            if tokenizer_path
            else (model_def.resolve(ds_cfg["tokenizer"]) if ds_cfg.get("tokenizer") else None)
        )
        if resolved_tokenizer and resolved_tokenizer.exists():
            from transformers import PreTrainedTokenizerFast

            specials = _default_special_tokens(resolved_tokenizer)
            tokenizer: Any = PreTrainedTokenizerFast(
                tokenizer_file=str(resolved_tokenizer),
                model_max_length=cfg.max_input_tokens,
                **specials,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)

        train_rows = list(iter_jsonl(dataset_dir / "train.jsonl"))
        val_rows = list(iter_jsonl(dataset_dir / "val.jsonl"))
        if limit is not None:
            train_rows = train_rows[:limit]
            val_rows = val_rows[: max(2, limit // 4)]
        if not train_rows:
            raise ValueError(f"No training rows in {dataset_dir / 'train.jsonl'}")

        unknown = scan_label_coverage(train_rows + val_rows, heads, target_field=target_field)
        if unknown:
            detail = "; ".join(f"{head}: {counts}" for head, counts in sorted(unknown.items()))
            raise ValueError(
                "gold values do not map to the declared head labels "
                f"({detail}). Fix training.heads[].labels or the corpus; these "
                "rows would otherwise train as class 0 / 'none'."
            )

        print(  # noqa: T201  training progress goes to stdout
            f"multi_head: run={run.run_id} model={cfg.model_id} heads={[h.name for h in heads]} "
            f"train={len(train_rows)} val={len(val_rows)} "
            f"max_len={cfg.max_input_tokens} epochs={cfg.epochs}"
        )

        train_ds = _build_dataset(
            train_rows,
            tokenizer,
            cfg.max_input_tokens,
            heads,
            request_field=request_field,
            target_field=target_field,
            text_transform=text_transform,
        )
        val_ds = (
            _build_dataset(
                val_rows,
                tokenizer,
                cfg.max_input_tokens,
                heads,
                request_field=request_field,
                target_field=target_field,
                text_transform=text_transform,
            )
            if val_rows
            else None
        )

        report_to = normalize_report_to(training_dict.get("report_to"))
        result = _train_loop(
            cfg,
            train_ds,
            val_ds,
            out_dir,
            tokenizer,
            heads=heads,
            profile=profile,
            embedding_strategy=training_dict.get("embedding_strategy"),
            model_def=model_def,
            dataset_dir=dataset_dir,
            smoke=smoke,
            device_name=str(target_device),
            run_id=run.run_id,
            report_to=report_to,
            resume_from_checkpoint=str(resume_path) if resume_path else None,
            distributed=distributed,
        )
        finish_run(model_def, run.run_id, "completed", metrics=result.metrics)
        return result
    except Exception as exc:
        finish_run(model_def, run.run_id, "aborted", error=str(exc))
        raise
