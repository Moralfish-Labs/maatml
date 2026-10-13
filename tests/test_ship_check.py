"""The ship decision: absolute floors, delta with one-row tolerance, controlled replay."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from maatml.cli import app
from maatml.evaluation.harness import Report
from maatml.evaluation.shipcheck import render_verdict, ship_check

runner = CliRunner()

GATES = {"acc": 0.8, "recall": 0.3}


def _report(
    *,
    metrics: dict,
    counts: dict | None = None,
    split: str = "a" * 64,
    gates_passed: bool | None = True,
    smoke: bool = False,
    failed: tuple[str, ...] = (),
) -> Report:
    gates = None
    if gates_passed is not None:
        results = {
            name: {
                "minimum": GATES[name],
                "actual": metrics.get(name),
                "passed": name not in failed,
                "tier": "blocking",
            }
            for name in GATES
        }
        gates = {
            "passed": gates_passed,
            "results": results,
            "advisory_failed": [],
            "smoke": smoke,
            "benchmark_sha256": split,
        }
    return Report(
        model_id="m",
        dataset="d",
        n=100,
        metrics=metrics,
        counts=counts or {},
        extras={"split_sha256": split},
        gates=gates,
        passed=gates_passed,
    )


def test_ships_when_floors_pass_and_nothing_regresses() -> None:
    cand = _report(metrics={"acc": 0.9, "recall": 0.5}, counts={"acc": {"k": 90, "n": 100}})
    base = _report(metrics={"acc": 0.85, "recall": 0.5})
    verdict = ship_check(cand, base, gates=GATES)
    assert verdict.ship is True
    assert verdict.reasons == []
    assert verdict.delta["metrics"]["acc"]["delta"] > 0


def test_one_row_drop_at_n_30_is_exempt_but_two_rows_regress() -> None:
    base = _report(metrics={"acc": 0.90, "recall": 0.5})
    one_row = _report(metrics={"acc": 0.89, "recall": 0.5}, counts={"acc": {"k": 89, "n": 100}})
    verdict = ship_check(one_row, base, gates=GATES)
    assert verdict.ship is True
    assert verdict.delta["exempt"] == ["acc"]
    two_rows = _report(metrics={"acc": 0.88, "recall": 0.5}, counts={"acc": {"k": 88, "n": 100}})
    verdict = ship_check(two_rows, base, gates=GATES)
    assert verdict.ship is False
    assert verdict.delta["regressions"] == ["acc"]
    assert "allowed -0.0100" in verdict.reasons[0]


def test_without_counts_any_drop_is_a_regression_unless_max_regression() -> None:
    base = _report(metrics={"acc": 0.90, "recall": 0.5})
    cand = _report(metrics={"acc": 0.899, "recall": 0.5})
    assert ship_check(cand, base, gates=GATES).ship is False
    assert ship_check(cand, base, gates=GATES, max_regression=0.01).ship is True


def test_advisory_regression_is_recorded_not_fatal() -> None:
    base = _report(metrics={"acc": 0.90, "recall": 0.5})
    cand = _report(metrics={"acc": 0.90, "recall": 0.1})
    verdict = ship_check(cand, base, gates=GATES, tiers={"recall": "advisory"})
    assert verdict.ship is True
    assert verdict.delta["advisory_regressions"] == ["recall"]


def test_absolute_failures_block() -> None:
    base = _report(metrics={"acc": 0.9, "recall": 0.5})
    no_evidence = _report(metrics={"acc": 0.9, "recall": 0.5}, gates_passed=None)
    assert "no gate evidence" in ship_check(no_evidence, base, gates=GATES).reasons[0]
    smoke = _report(metrics={"acc": 0.9, "recall": 0.5}, smoke=True)
    assert "smoke tier" in ship_check(smoke, base, gates=GATES).reasons[0]
    below = _report(metrics={"acc": 0.5, "recall": 0.5}, gates_passed=False, failed=("acc",))
    verdict = ship_check(below, base, gates=GATES)
    assert verdict.absolute["failed"] == ["acc"]
    assert any("below floor: acc" in r for r in verdict.reasons)


def test_different_splits_require_a_replay() -> None:
    base = _report(metrics={"acc": 0.9, "recall": 0.5}, split="b" * 64)
    cand = _report(metrics={"acc": 0.95, "recall": 0.6})
    verdict = ship_check(cand, base, gates=GATES)
    assert verdict.ship is False
    assert verdict.population["same"] is False
    assert "different splits" in verdict.reasons[0]
    assert ship_check(cand, base, gates=GATES, replayed=True).ship is True


def test_incomparable_metric_is_named() -> None:
    base = _report(metrics={"acc": 0.9})
    cand = _report(metrics={"acc": 0.9, "recall": 0.5})
    verdict = ship_check(cand, base, gates=GATES)
    assert verdict.delta["incomparable"] == ["recall"]
    assert verdict.ship is False


def test_render_lists_every_part() -> None:
    base = _report(metrics={"acc": 0.9, "recall": 0.5})
    cand = _report(metrics={"acc": 0.95, "recall": 0.5}, counts={"acc": {"k": 95, "n": 100}})
    lines = render_verdict(ship_check(cand, base, gates=GATES))
    assert lines[0].endswith("SHIP")
    assert any(line.startswith("absolute: passed") for line in lines)
    assert any("delta: acc: 0.9000 -> 0.9500" in line for line in lines)
    assert any(line.startswith("population: same split") for line in lines)


# --- CLI -------------------------------------------------------------------------

_MODEL_YML = """name: ship-test
model_id: ship-test
version: 0.1.0
architecture: causal_sft
dataset:
  format: jsonl_seed
  seed_samples: datasets/samples/seed_samples.jsonl
evaluation:
  predictor: causal_sft
  gates:
    acc: 0.8
    recall: 0.3
training:
  base_model: sshleifer/tiny-gpt2
"""


def _model_dir(tmp_path: Path) -> Path:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    (mdir / "datasets" / "samples" / "seed_samples.jsonl").write_text(
        json.dumps({"request": "x", "expected": "y", "family": "f"}) + "\n"
    )
    (mdir / "model.yml").write_text(_MODEL_YML)
    return mdir


def test_cli_ship_check_exit_codes_and_json(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path)
    eval_dir = mdir / "output" / "eval"
    _report(metrics={"acc": 0.95, "recall": 0.5}, counts={"acc": {"k": 95, "n": 100}}).write(
        eval_dir / "cand.json"
    )
    _report(metrics={"acc": 0.90, "recall": 0.5}).write(eval_dir / "base.json")
    result = runner.invoke(app, ["ship-check", str(mdir), "cand", "base"])
    assert result.exit_code == 0, result.output
    assert "SHIP" in result.output and "DO NOT" not in result.output

    _report(metrics={"acc": 0.80, "recall": 0.5}, counts={"acc": {"k": 80, "n": 100}}).write(
        eval_dir / "worse.json"
    )
    result = runner.invoke(app, ["ship-check", str(mdir), "worse", "base", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ship"] is False
    assert payload["delta"]["regressions"] == ["acc"]

    result = runner.invoke(app, ["ship-check", str(mdir), "missing", "base"])
    assert result.exit_code != 0
    assert "no eval report" in result.output


def test_cli_replay_re_evaluates_both_when_splits_differ(tmp_path: Path, monkeypatch) -> None:
    mdir = _model_dir(tmp_path)
    eval_dir = mdir / "output" / "eval"
    _report(metrics={"acc": 0.95, "recall": 0.5}).write(eval_dir / "cand.json")
    _report(metrics={"acc": 0.90, "recall": 0.5}, split="b" * 64).write(eval_dir / "base.json")

    result = runner.invoke(app, ["ship-check", str(mdir), "cand", "base"])
    assert result.exit_code == 1
    assert "different splits" in result.output

    calls: list[tuple[str, Path, bool]] = []

    def fake_evaluate_model(md, *, checkpoint, out_dir, record_gates, **kwargs):
        calls.append((checkpoint, out_dir, record_gates))
        acc = 0.93 if checkpoint == "cand" else 0.90
        report = _report(metrics={"acc": acc, "recall": 0.5}, split="c" * 64)
        path = Path(out_dir) / f"{checkpoint}.json"
        report.write(path)
        return report, path

    import maatml.evaluation.runner as runner_mod

    monkeypatch.setattr(runner_mod, "evaluate_model", fake_evaluate_model)
    result = runner.invoke(app, ["ship-check", str(mdir), "cand", "base", "--replay"])
    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == ["cand", "base"]
    assert all(c[1] == eval_dir / "replay" and c[2] is False for c in calls)
    assert "controlled replay" in result.output
    # The runs' own evidence is untouched.
    assert json.loads((eval_dir / "base.json").read_text())["extras"]["split_sha256"] == "b" * 64
