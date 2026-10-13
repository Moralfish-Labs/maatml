"""``training.select_by``: the checkpoint a run ships is chosen by a val metric.

A trainer keeps its last checkpoints and saves its final weights; "last" is
not a selection, and the trainer's own val loss is not the metric the gates
read. With ``select_by`` set, every saved ``checkpoint-*`` and the final
weights are evaluated on the val split with the same predictor, validator
and metrics as ``evaluate``, the best by the named metric is recorded on the
run (``selection``, ``selected_checkpoint``), and from then on the run id
resolves to that checkpoint for evaluate, export and serve. The test split
is never touched.

```yaml
training:
  select_by: all_layers_pass_rate   # a metric the eval harness reports
  keep_checkpoints: 4               # save_total_limit; 2 keeps only the last two
```
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

SELECT_DIR = "select"
FINAL = "final"
_STEP = re.compile(r"^checkpoint-(\d+)$")


def resolve_select_by(model_def: Any, *, smoke: bool = False) -> Optional[str]:
    training = model_def.merged_smoke() if smoke else model_def.training
    value = (training or {}).get("select_by")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("training.select_by must name a metric the eval harness reports")
    return value.strip()


def candidate_checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    """Saved ``checkpoint-<step>`` dirs by step, then the final weights."""
    run_dir = Path(run_dir)
    steps: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        match = _STEP.match(child.name)
        if match and child.is_dir():
            steps.append((int(match.group(1)), child))
    ordered = [(path.name, path) for _, path in sorted(steps)]
    ordered.append((FINAL, run_dir))
    return ordered


def select_checkpoint(
    model_def: Any,
    run_id: str,
    *,
    device: str = "auto",
    smoke: bool = False,
) -> Optional[dict[str, Any]]:
    """Evaluate every candidate on val and record the best; ``None`` when unset."""
    metric = resolve_select_by(model_def, smoke=smoke)
    if metric is None:
        return None
    from ..evaluation.runner import evaluate_model
    from ..runs import get_run, record_selection

    rec = get_run(model_def, run_id)
    if rec is None:
        raise ValueError(f"{run_id} is not in runs.jsonl; nothing to select for")
    run_dir = Path(rec.out_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_id}: run directory {run_dir} is missing")
    out_dir = Path(model_def.eval_dir) / SELECT_DIR / run_id
    candidates: list[dict[str, Any]] = []
    for name, path in candidate_checkpoints(run_dir):
        report, report_path = evaluate_model(
            model_def,
            checkpoint=str(path),
            split="val",
            device=device,
            gate=False,
            out_dir=out_dir,
            record_gates=False,
            label=name,
        )
        if metric not in report.metrics:
            raise ValueError(
                f"training.select_by {metric!r} is not a metric the harness reports; "
                f"reported: {sorted(report.metrics)}"
            )
        candidates.append(
            {
                "name": name,
                "path": str(path),
                "value": float(report.metrics[metric]),
                "n": report.n,
                "report": str(report_path),
            }
        )
    best = candidates[0]
    for candidate in candidates[1:]:
        # Ties go to the later checkpoint; the final weights are last.
        if candidate["value"] >= best["value"]:
            best = candidate
    selection = {
        "metric": metric,
        "split": "val",
        "candidates": candidates,
        "selected": best["name"],
        "smoke": bool(smoke),
    }
    record_selection(
        model_def,
        run_id,
        selection,
        selected_checkpoint=None if best["name"] == FINAL else best["name"],
    )
    return selection


def render_selection(selection: dict[str, Any]) -> list[str]:
    lines = [f"select_by {selection['metric']} on {selection['split']}:"]
    for candidate in selection.get("candidates", []):
        mark = "*" if candidate["name"] == selection.get("selected") else " "
        lines.append(
            f"  {mark} {candidate['name']:<16} {candidate['value']:.4f}  n={candidate['n']}"
        )
    lines.append(f"selected {selection.get('selected')}")
    return lines
