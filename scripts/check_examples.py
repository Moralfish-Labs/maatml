"""Cheap per-example invariants: splits, gate wiring, and gold self-consistency.

Runs no training and needs no GPU. For each example it prepares the corpus and
then scores the test split using each row's own gold as the prediction, which
gives the full set of metric keys a real evaluation would emit.

Checks:
  1. every split is non-empty (whole-group hashing with too few families
     silently leaves val empty, and training then skips evaluation)
  2. every declared gate names a metric the plugin actually emits
  3. the corpus's own gold passes the validator's contract layers

Full lifecycle verification stays local; see scripts/train_all.py and
scripts/evaluate_all.py.

Usage:
    python scripts/check_examples.py
    python scripts/check_examples.py --only vision support-ticket-triage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maatml.config import get_dataset_cfg, load_model_def  # noqa: E402
from maatml.evaluation.harness import (  # noqa: E402
    RowEval,
    _resolve_metrics,
    effective_gates,
    resolve_validator,
)
from maatml.registry import FORMATS, discover_plugins  # noqa: E402
from maatml.utils.io import iter_jsonl  # noqa: E402

# Reported by the harness rather than by a model's metrics plugin.
_HARNESS_METRICS = frozenset({"output_nonempty_rate"})


def _model_dirs(only: list[str] | None) -> list[Path]:
    root = ROOT / "examples"
    dirs = [d for d in sorted(root.iterdir()) if (d / "model.yml").is_file()]
    if only:
        wanted = set(only)
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def _gold_row_evals(model_def, rows: list[dict]) -> list[RowEval]:
    """Score each row against its own gold, so no model is needed."""
    cfg = get_dataset_cfg(model_def)
    target_field = str(cfg.get("target_field") or "target")
    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")
    validator_name = (model_def.evaluation or {}).get("validator")
    validate = resolve_validator(str(validator_name)) if validator_name else None
    schema = model_def.resolve(cfg["schema"]) if isinstance(cfg.get("schema"), str) else None
    contracts = (
        model_def.resolve(cfg["contracts"]) if isinstance(cfg.get("contracts"), str) else None
    )

    out: list[RowEval] = []
    for row in rows:
        gold = row.get(target_field)
        text = gold if isinstance(gold, str) else json.dumps(gold)
        result = validate(
            text,
            schema_path=schema,
            contracts_path=contracts,
            user_prompt=row.get(request_field),
        )
        out.append(RowEval(row=row, gen_text=text, result=result))
    return out


def check(model_dir: Path) -> list[str]:
    errors: list[str] = []
    model_def = load_model_def(model_dir)
    name = model_dir.name

    fmt = str(get_dataset_cfg(model_def).get("format", "jsonl_seed"))
    summary = FORMATS.require(fmt)(model_def) or {}
    counts = summary.get("split_counts") or {}
    for split in ("train", "val", "test"):
        if not counts.get(split):
            errors.append(f"{name}: {split} split is empty ({counts})")
    if errors:
        return errors

    test_rows = list(iter_jsonl(model_def.prepared_dir / "test.jsonl"))
    gates = effective_gates(model_def)
    if not gates:
        errors.append(f"{name}: no evaluation.gates declared")

    metrics_fns = _resolve_metrics((model_def.evaluation or {}).get("metrics"))
    if not metrics_fns:
        errors.append(f"{name}: evaluation.metrics does not resolve")
        return errors

    row_evals = _gold_row_evals(model_def, test_rows)
    emitted = set(_HARNESS_METRICS)
    for fn in metrics_fns:
        emitted |= set(fn(row_evals))
    missing = sorted(set(gates) - emitted)
    if missing:
        errors.append(f"{name}: gates name metrics never emitted: {missing}")

    failed_gold = sum(1 for r in row_evals if not r.result.ok)
    if failed_gold:
        errors.append(f"{name}: {failed_gold}/{len(row_evals)} gold rows fail the validator")

    errors.extend(_check_readme_gates(model_dir, name, gates))
    return errors


def _readme_gate_table(readme: str) -> dict[str, float]:
    """Parse the `| \\`metric\\` | >= X |` rows of a README quality-gates table.

    Only rows that use the ``>=`` threshold form count as gates. Other tables
    that happen to put a backtick name next to a number (validator error codes
    with a layer column, for example) must not be mistaken for gate claims.
    """
    found: dict[str, float] = {}
    for match in re.finditer(
        r"\|\s*`([a-z0-9_]+)`\s*\|\s*>=\s*([0-9.]+)\s*\|",
        readme,
    ):
        found[match.group(1)] = float(match.group(2))
    return found


def _check_readme_gates(model_dir: Path, name: str, gates: dict) -> list[str]:
    """The README gate table must match model.yml.

    Hand-maintained tables drifted on every example: thresholds disagreed with
    the model, some gates were undocumented, and one README advertised a
    stricter gate than the model enforces. A reader takes the README as the
    contract, so a mismatch is a wrong contract.
    """
    readme = model_dir / "README.md"
    if not readme.is_file():
        return []
    documented = _readme_gate_table(readme.read_text(encoding="utf-8"))
    if not documented:
        return [f"{name}: README has no quality-gates table"]
    problems = []
    for metric in sorted(set(gates) - set(documented)):
        problems.append(f"{name}: gate {metric} is not in the README table")
    for metric in sorted(set(documented) - set(gates)):
        problems.append(f"{name}: README documents {metric}, which is not a gate")
    for metric in sorted(set(gates) & set(documented)):
        if float(gates[metric]) != documented[metric]:
            problems.append(
                f"{name}: README says {metric} >= {documented[metric]}, "
                f"model.yml gates at {float(gates[metric])}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Subset by folder name")
    args = parser.parse_args(argv)

    discover_plugins()
    failures: list[str] = []
    for model_dir in _model_dirs(args.only):
        try:
            errors = check(model_dir)
        except Exception as exc:  # noqa: BLE001  report, do not abort the sweep
            errors = [f"{model_dir.name}: {type(exc).__name__}: {exc}"]
        if errors:
            failures.extend(errors)
            for err in errors:
                print(f"FAIL {err}")
        else:
            print(f"ok   {model_dir.name}")
    if failures:
        print(f"\n{len(failures)} problem(s)")
        return 1
    print("\nall examples pass the cheap invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
