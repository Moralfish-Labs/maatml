"""Shared validation primitives for out-of-model task gates.

Each task validator is an N-layer pipeline that returns a
:class:`ValidationResult`. Shared helpers cover fence-stripping and JSON/schema
loading so task modules stay focused on their layer logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

_FENCE_RX = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)
_THINK_BLOCK_RX = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass
class ValidationError:
    layer: int
    code: str
    message: str
    location: Optional[str] = None
    # Optional "how to fix it" guidance for humans and retry feedback.
    hint: Optional[str] = None


@dataclass
class ValidationResult:
    """Outcome of running an N-layer validator on one model output."""

    raw_output: str
    parsed: Optional[dict[str, Any]] = None
    errors: list[ValidationError] = field(default_factory=list)
    passed_layers: set[int] = field(default_factory=set)
    extras: dict[str, Any] = field(default_factory=dict)
    # When set, ``ok`` requires these layers to have passed.
    required_layers: Optional[set[int]] = None
    # Alternative: require layers 1..n_layers inclusive.
    n_layers: Optional[int] = None

    @property
    def ok(self) -> bool:
        if self.required_layers is not None:
            return self.passed_layers >= self.required_layers
        if self.n_layers is not None:
            return self.passed_layers == set(range(1, self.n_layers + 1))
        return not self.errors and bool(self.passed_layers)


@runtime_checkable
class Validator(Protocol):
    """The call shape every maatml validator must accept.

    Registered validators are called directly (``validate(raw_output, ...)``),
    not through a ``.validate()`` method, and the harness, serve, datagen, and
    ingest all pass the same optional keywords: a validator may be given only
    ``user_prompt`` (no schema or contracts resolved) or all of them. Every
    keyword is optional so one implementation satisfies all four call sites.
    """

    def __call__(
        self,
        raw_output: str,
        *,
        schema_path: str | Path | None = ...,
        contracts_path: str | Path | None = ...,
        user_prompt: Optional[str] = ...,
        strip_fences: bool = ...,
    ) -> ValidationResult: ...


def strip_fences(text: str) -> str:
    """Strip Qwen3 ``<think>...</think>`` blocks and ```json fences."""
    text = text.strip()
    text = _THINK_BLOCK_RX.sub("", text).strip()
    fence = _FENCE_RX.match(text)
    if fence:
        return fence.group(1).strip()
    return text


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_schema(path: str | Path) -> dict[str, Any]:
    """Load a JSON Schema document (alias of ``_load_json`` for clarity)."""
    return _load_json(path)


def _load_contracts(path: str | Path) -> dict[str, Any]:
    """Load a node-contracts / enum catalogue JSON."""
    return _load_json(path)
