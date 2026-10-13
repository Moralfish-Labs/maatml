"""CPU-safe SFT config models (no torch / transformers import)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# `training.precision` values the device profiles understand. Anything else
# used to fall through resolve_load_dtype as "fp32", so `precision: bfloat16`
# or a typo trained at a precision nobody asked for.
VALID_PRECISIONS = ("bf16", "fp16", "fp32")
# Read by the lifecycle after training, not by any trainer.
LIFECYCLE_TRAINING_KEYS = frozenset({"select_by"})


def reject_unknown_training_keys(
    training: dict, known: frozenset[str], *, architecture: str
) -> None:
    """Fail on a ``training:`` key this architecture does not read.

    The pydantic configs get this from ``extra="forbid"``; the dataclass ones
    build themselves with ``d.get(...)``, which silently ignores a typo. The
    run then trains at the built-in default and reports success, and because
    ``training_config`` is hashed into the lifecycle fingerprint the typo'd key
    still makes the run look legitimately fresh.
    """
    unknown = sorted(set(training) - known - LIFECYCLE_TRAINING_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown training key(s) for {architecture}: {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}."
        )


def validate_precision(value: Any) -> str:
    """Return ``value`` if it is a supported precision, else raise."""
    text = str(value)
    if text not in VALID_PRECISIONS:
        raise ValueError(
            f"training.precision must be one of {', '.join(VALID_PRECISIONS)}; got {value!r}"
        )
    return text


class LoraSettings(BaseModel):
    # Strict like every sibling config: a typo under `training.lora:` would
    # otherwise be dropped in silence, and because training_config is hashed
    # into the lifecycle fingerprint the run still looks legitimately fresh.
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # merged (default) | adapter | both, see train_sft artifact save path.
    save_mode: str = "merged"


class QuantizationSettings(BaseModel):
    """Optional bitsandbytes / QLoRA settings (CUDA-only)."""

    model_config = ConfigDict(extra="forbid")

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bf16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def enabled(self) -> bool:
        return bool(self.load_in_4bit or self.load_in_8bit)


class SFTTrainConfig(BaseModel):
    """Generic SFT training config, same shape for all three tasks."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-1.7B"
    max_input_tokens: int = Field(default=4096, gt=0)
    batch_size: int = Field(default=2, gt=0)
    grad_accum: int = Field(default=8, gt=0)
    learning_rate: float = 1e-4
    epochs: float = 4.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 7331
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 200
    logging_steps: int = 20
    max_steps: int = -1
    lora: LoraSettings = Field(default_factory=LoraSettings)
    report_to: Any = None
    group_by_length: bool = False
    quantization: Optional[QuantizationSettings] = None
    attn_implementation: Optional[str] = None
    dataloader_workers: Optional[int] = None
    model_revision: Optional[str] = None
    # save_total_limit: how many checkpoint-* dirs survive for select_by to compare.
    keep_checkpoints: int = Field(default=2, ge=1)

    @field_validator("precision")
    @classmethod
    def _check_precision(cls, v: str) -> str:
        return validate_precision(v)


@dataclass
class SFTTrainResult:
    out_dir: Path
    metrics: dict[str, float]
    train_runtime: float


def render_assistant_target(sample: dict, target_field: str) -> str:
    """Render the gold assistant content.

    A structured target is serialised compactly: no indent, to keep the token
    count down, and the runtime parses pretty or compact equally well.

    A target that is already a string **is** the assistant's text, and is used
    as it stands. Passing it through ``json.dumps`` would wrap it in quotes and
    escape every quote inside it, and the model learns to answer that way —
    correct content inside a string literal, which no validator downstream
    accepts. ``distill.target_format: text`` writes string targets, so this is
    the difference between a model that answers and one that answers in quotes.
    """
    payload = sample[target_field]
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
