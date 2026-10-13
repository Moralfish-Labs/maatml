"""Lazy trainer registration, importable without torch / transformers.

Heavy trainer modules pull ML deps at import time (notably ``sft_base``).
CLI commands like ``scaffold`` / ``plugins`` / ``validate`` only need the
architecture names registered, so discovery loads this shim instead.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_trainer


@register_trainer("causal_sft")
def train_causal_sft(model_def: Any, **kwargs: Any) -> Any:
    from .sft_base import train_sft

    return train_sft(model_def, **kwargs)


@register_trainer("multi_head_classifier")
@register_trainer("classifier")
def train_classifier(model_def: Any, **kwargs: Any) -> Any:
    from .multi_head import train_multi_head

    return train_multi_head(model_def, **kwargs)


@register_trainer("seq2seq")
def train_seq2seq(model_def: Any, **kwargs: Any) -> Any:
    from .seq2seq import train_seq2seq_model

    return train_seq2seq_model(model_def, **kwargs)


@register_trainer("dpo")
def train_dpo(model_def: Any, **kwargs: Any) -> Any:
    from .preference import train_dpo as _train_dpo

    return _train_dpo(model_def, **kwargs)


@register_trainer("orpo")
def train_orpo(model_def: Any, **kwargs: Any) -> Any:
    from .preference import train_orpo as _train_orpo

    return _train_orpo(model_def, **kwargs)


@register_trainer("diffusion_lora")
def train_diffusion_lora(model_def: Any, **kwargs: Any) -> Any:
    from .diffusion import train_diffusion_lora as _train

    return _train(model_def, **kwargs)
