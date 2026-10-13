"""Shared ``from_pretrained`` / quantization helpers for trainers."""

from __future__ import annotations

from typing import Any, Optional

from ..device import DeviceProfile, ensure_quantization_allowed, resolve_load_dtype


def _dtype_name_to_torch(name: str) -> Any:
    import torch

    key = (name or "bf16").lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unknown dtype name {name!r}")


def build_bitsandbytes_config(quantization: dict[str, Any] | Any) -> Any:
    """Build ``BitsAndBytesConfig`` from a dict or pydantic settings object.

    Raises ``ImportError`` with an install hint when bitsandbytes is missing.
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "bitsandbytes / BitsAndBytesConfig required for quantization; install maatml[cuda]"
        ) from exc

    raw = quantization
    if hasattr(quantization, "model_dump"):
        raw = quantization.model_dump()
    if not isinstance(raw, dict):
        raise TypeError(f"quantization must be a dict-like mapping; got {type(raw)!r}")

    load_in_4bit = bool(raw.get("load_in_4bit", False))
    load_in_8bit = bool(raw.get("load_in_8bit", False))
    if not load_in_4bit and not load_in_8bit:
        raise ValueError("training.quantization requires load_in_4bit or load_in_8bit to be true")
    if load_in_4bit and load_in_8bit:
        raise ValueError("Set only one of load_in_4bit / load_in_8bit")

    kwargs: dict[str, Any] = {
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
    }
    if load_in_4bit:
        kwargs["bnb_4bit_compute_dtype"] = _dtype_name_to_torch(
            str(raw.get("bnb_4bit_compute_dtype", "bf16"))
        )
        kwargs["bnb_4bit_quant_type"] = str(raw.get("bnb_4bit_quant_type", "nf4"))
        kwargs["bnb_4bit_use_double_quant"] = bool(raw.get("bnb_4bit_use_double_quant", True))
    return BitsAndBytesConfig(**kwargs)


def from_pretrained_kwargs(
    profile: DeviceProfile,
    *,
    precision: str = "bf16",
    attn_implementation: Optional[str] = None,
    quantization: Optional[dict[str, Any] | Any] = None,
    revision: Optional[str] = None,
    frozen_base: bool = False,
) -> dict[str, Any]:
    """Keyword args shared by causal / seq2seq / encoder ``from_pretrained``.

    ``frozen_base`` says the loaded weights will not be trained — a LoRA, where
    only the adapter carries gradients. It lets the fp32-master policy be
    skipped on mps/cpu; see :func:`maatml.device.resolve_load_dtype`.
    """
    kwargs: dict[str, Any] = {}
    if quantization is not None:
        ensure_quantization_allowed(profile)
        qcfg = build_bitsandbytes_config(quantization)
        kwargs["quantization_config"] = qcfg
        # Device map lets bitsandbytes place layers; HF Trainer still owns the loop.
        kwargs.setdefault("device_map", "auto")
    else:
        dtype = resolve_load_dtype(profile, precision, frozen_base=frozen_base)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if revision:
        kwargs["revision"] = revision
    return kwargs


def maybe_prepare_kbit(model: Any, quantization: Optional[Any]) -> Any:
    """Call ``prepare_model_for_kbit_training`` when quantization is active."""
    if quantization is None:
        return model
    enabled = False
    if hasattr(quantization, "load_in_4bit"):
        enabled = bool(quantization.load_in_4bit or quantization.load_in_8bit)
    elif isinstance(quantization, dict):
        enabled = bool(quantization.get("load_in_4bit") or quantization.get("load_in_8bit"))
    if not enabled:
        return model
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "peft is required for QLoRA; install maatml[ml] (and maatml[cuda] for bnb)"
        ) from exc
    return prepare_model_for_kbit_training(model)
