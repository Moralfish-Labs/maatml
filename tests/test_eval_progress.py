"""The eval loop says where it is while it runs."""

from __future__ import annotations

import json
from pathlib import Path

from maatml.evaluation.harness import run_evaluation


def _dataset(tmp_path: Path, rows: int) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with (prepared / "test.jsonl").open("w") as handle:
        for index in range(rows):
            handle.write(json.dumps({"request": f"ask {index}", "expected": "ok"}) + "\n")
    return prepared


def _run(tmp_path: Path, rows: int) -> None:
    run_evaluation(
        checkpoint_dir=tmp_path / "ckpt",
        dataset_dir=_dataset(tmp_path, rows),
        out_path=tmp_path / "report.json",
        predictor=lambda row: "ok",
        device="cpu",
    )


def _progress(captured: str) -> list[str]:
    return [line for line in captured.splitlines() if line.startswith("eval ") and "/" in line]


def test_a_long_split_reports_while_it_generates(tmp_path: Path, capsys) -> None:
    _run(tmp_path, 40)
    lines = _progress(capsys.readouterr().out)
    assert lines, "the loop generated 40 rows and said nothing"
    assert lines[-1].startswith("eval 40/40")


def test_progress_is_bounded_so_a_large_split_does_not_flood(tmp_path: Path, capsys) -> None:
    _run(tmp_path, 400)
    assert len(_progress(capsys.readouterr().out)) <= 21


def test_a_short_split_still_reports_every_row(tmp_path: Path, capsys) -> None:
    # Integer division gives a zero interval below twenty rows, and a zero
    # interval is a modulo by zero rather than a quiet loop.
    _run(tmp_path, 3)
    assert len(_progress(capsys.readouterr().out)) == 3


def test_the_last_row_reports_even_when_it_is_not_on_the_interval(tmp_path: Path, capsys) -> None:
    # 41 rows at an interval of 2: without the explicit final line the run ends
    # at 40/41 and looks stalled one row from the end.
    _run(tmp_path, 41)
    assert _progress(capsys.readouterr().out)[-1].startswith("eval 41/41")
