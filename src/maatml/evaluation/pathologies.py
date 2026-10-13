"""Named pathology signatures: shapes of output no floor should have to catch.

A model that never fires, answers one class, or says the same thing to every
input can still post respectable-looking aggregates — precision is high when
nothing is predicted, a pooled rate is flat while one class carries it. These
signatures are reported on every evaluate as ``pathologies[]`` and fail the
smoke tier outright, so a rehearsal cannot pass on a model that does not
work; at the production tier the floors decide, with the signature beside
them. A metrics plugin adds its own through a ``__pathologies__`` entry.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

PATHOLOGIES_KEY = "__pathologies__"
MIN_ROWS = 5

_NEVER_FIRES = "never_fires"
_ONE_CLASS = "one_class"
_IDENTICAL_OUTPUT = "identical_output"
_LABEL_FIELDS = ("class", "category", "label", "family", "intent", "kind")
_RECALL = re.compile(r"(^|_)recall($|_)")


def _recall_like(metrics: dict[str, float]) -> list[tuple[str, float, Optional[float]]]:
    """(recall metric, value, its precision counterpart when reported)."""
    found = []
    for name, value in metrics.items():
        if not _RECALL.search(name):
            continue
        precision_name = _RECALL.sub(lambda m: f"{m.group(1)}precision{m.group(2)}", name)
        precision = metrics.get(precision_name)
        found.append((name, float(value), float(precision) if precision is not None else None))
    return found


def detect_pathologies(
    rows: Sequence[Any],
    metrics: dict[str, float],
    per_class: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Signatures over the evaluated rows (``RowEval``: ``gen_text``, ``result.parsed``)."""
    n = len(rows)
    if n < MIN_ROWS:
        return []
    found: list[dict[str, Any]] = []

    texts = [(getattr(item, "gen_text", "") or "").strip() for item in rows]
    nonempty = sum(1 for text in texts if text)
    if nonempty == 0:
        found.append(
            {
                "name": _NEVER_FIRES,
                "evidence": f"no output on any of {n} rows (output_nonempty_rate 0)",
            }
        )
    else:
        for name, recall, precision in _recall_like(metrics):
            if recall <= 0.01 and (precision is None or precision >= 0.9):
                shown = f"{precision:.3f}" if precision is not None else "not reported"
                found.append(
                    {
                        "name": _NEVER_FIRES,
                        "evidence": f"{name} {recall:.3f} with precision {shown}: "
                        "nothing is being predicted",
                    }
                )
                break

    if nonempty == n and len(set(texts)) == 1:
        found.append(
            {
                "name": _IDENTICAL_OUTPUT,
                "evidence": f"the same output on all {n} rows: {texts[0][:80]!r}",
            }
        )

    if len(per_class) >= 2:
        parsed = [getattr(getattr(item, "result", None), "parsed", None) for item in rows]
        if all(isinstance(p, dict) for p in parsed):
            for field_name in _LABEL_FIELDS:
                if not all(field_name in p for p in parsed):  # type: ignore[operator]
                    continue
                values = {str(p[field_name]) for p in parsed}  # type: ignore[index]
                if len(values) == 1:
                    found.append(
                        {
                            "name": _ONE_CLASS,
                            "evidence": f"every output carries {field_name}="
                            f"{next(iter(values))!r} across {len(per_class)} gold classes",
                        }
                    )
                break
    return found


def normalize_plugin_pathologies(raw: Any, *, plugin: str) -> list[dict[str, Any]]:
    """``__pathologies__`` from a metrics plugin: names or {name, evidence} dicts."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"evaluation.metrics: {plugin!r} {PATHOLOGIES_KEY} must be a list")
    out: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append({"name": entry, "evidence": f"reported by {plugin}"})
        elif isinstance(entry, dict) and entry.get("name"):
            out.append(
                {
                    "name": str(entry["name"]),
                    "evidence": str(entry.get("evidence") or f"reported by {plugin}"),
                }
            )
        else:
            raise ValueError(
                f"evaluation.metrics: {plugin!r} {PATHOLOGIES_KEY} entries are names "
                "or {name, evidence}"
            )
    return out
