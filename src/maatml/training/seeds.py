"""``maatml train --seeds N``: one recipe, N seeds, the spread in one record.

A single run reports one point; a floor set from it is set by luck as much
as by the recipe. A seed study trains the same effective config N times and
records mean / sd / min / max per metric, and ``gates derive --seed-study``
takes the per-metric minimum across those runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from ..utils.io import write_json_atomic

SEED_STUDY_KIND = "maatml.seed_study/1"


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def summarize_seed_runs(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """mean / sd / min / max / n per metric reported by every run.

    A metric missing from any run is left out rather than averaged over the
    runs that happened to report it; ``sd`` is the sample standard deviation
    and reads 0.0 for a single run.
    """
    if not runs:
        return {}
    keys: Optional[set[str]] = None
    for run in runs:
        present = {k for k, v in (run.get("metrics") or {}).items() if _numeric(v) is not None}
        keys = present if keys is None else keys & present
    stats: dict[str, dict[str, float]] = {}
    for key in sorted(keys or ()):
        values = [float(run["metrics"][key]) for run in runs]
        n = len(values)
        mean = sum(values) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
        stats[key] = {"mean": mean, "sd": sd, "min": min(values), "max": max(values), "n": float(n)}
    return stats


def seed_study_path(model_def: Any, label: str) -> Path:
    return Path(model_def.output_dir) / "seeds" / f"{label}.json"


def write_seed_study(
    model_def: Any,
    *,
    label: str,
    seeds: Sequence[int],
    runs: Sequence[dict[str, Any]],
    smoke: bool,
) -> Path:
    payload = {
        "kind": SEED_STUDY_KIND,
        "label": label,
        "smoke": bool(smoke),
        "seeds": list(seeds),
        "runs": [
            {"run_id": r["run_id"], "seed": r["seed"], "metrics": r.get("metrics") or {}}
            for r in runs
        ],
        "stats": summarize_seed_runs(runs),
    }
    return write_json_atomic(seed_study_path(model_def, label), payload)


def load_seed_study(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != SEED_STUDY_KIND:
        raise ValueError(f"{path}: not a {SEED_STUDY_KIND} file")
    return payload


def render_seed_study(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"seed study {payload.get('label')}: {len(payload.get('runs') or [])} runs, "
        f"seeds {payload.get('seeds')}" + ("  (smoke)" if payload.get("smoke") else "")
    ]
    for run in payload.get("runs") or []:
        lines.append(f"  {run['run_id']}  seed={run['seed']}")
    for key, stat in (payload.get("stats") or {}).items():
        lines.append(
            f"{key}: mean {stat['mean']:.4f}  sd {stat['sd']:.4f}  "
            f"min {stat['min']:.4f}  max {stat['max']:.4f}"
        )
    return lines
