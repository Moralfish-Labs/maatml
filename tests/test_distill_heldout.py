"""distill refuses prompts from held-out populations and records cache provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from maatml.config import load_model_def
from maatml.data import distill as distill_mod
from maatml.data.distill import TEACHER_CACHE_KIND, DistillConfigError, TeacherCache, run_distill
from maatml.registry import register_validator
from maatml.utils.io import iter_jsonl, sha256_file, write_jsonl
from maatml.validation.base import ValidationResult


@register_validator("heldout_test_validator")
def _validator(raw_output, **_kw):
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    result.parsed = json.loads(raw_output)
    result.passed_layers.add(1)
    return result


class _Teacher:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def chat_completions(self, messages, **kwargs):
        return json.dumps({"ok": True})


@pytest.fixture(autouse=True)
def _teacher(monkeypatch):
    monkeypatch.setattr(distill_mod, "TeacherClient", _Teacher)


def _model(
    tmp_path: Path,
    prompts: list[dict],
    *,
    benchmark: list[dict] | None = None,
    blind: list[dict] | None = None,
    pins: dict | None = None,
):
    mdir = tmp_path / "model"
    samples = mdir / "datasets" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    write_jsonl(mdir / "datasets" / "prompts.jsonl", prompts)
    dataset: dict = {
        "seed_samples": "datasets/samples/seed_samples.jsonl",
        "target_field": "expected_output",
    }
    if benchmark is not None:
        write_jsonl(samples / "benchmark_samples.jsonl", benchmark)
        dataset["benchmark_samples"] = "datasets/samples/benchmark_samples.jsonl"
    if blind is not None:
        write_jsonl(samples / "blind_v001.jsonl", blind)
        dataset["blind_samples"] = "datasets/samples/blind_v001.jsonl"
    if pins is not None:
        dataset["isolation"] = {"fields": ["camera"]}
        dataset["pins"] = pins
    body = {
        "name": "heldout-test",
        "model_id": "heldout-test",
        "architecture": "causal_sft",
        "version": "0.1.0",
        "dataset": dataset,
        "evaluation": {"validator": "heldout_test_validator"},
        "distill": {
            "prompt_source": "datasets/prompts.jsonl",
            "teacher_model": "fake",
            "teacher_revision": "v1",
            "cache": "datasets/cache.jsonl",
        },
    }
    (mdir / "model.yml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_model_def(mdir)


def test_a_benchmark_prompt_in_the_pool_is_refused_before_any_label(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        [{"request": "fresh ask"}, {"request": "held ask"}],
        benchmark=[{"request": "held ask", "expected_output": {"ok": True}, "family": "b"}],
    )
    with pytest.raises(DistillConfigError, match="'held ask' is in benchmark_samples"):
        run_distill(md)
    assert not (md.model_dir / "datasets" / "cache.jsonl").exists()


def test_blind_and_prepared_val_prompts_are_held_out_too(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        [{"request": "blind ask"}],
        blind=[{"request": "blind ask", "expected_output": {"ok": True}}],
    )
    with pytest.raises(DistillConfigError, match="is in blind_samples"):
        run_distill(md, replay=True)

    md = _model(tmp_path / "b", [{"request": "val ask"}])
    md.prepared_dir.mkdir(parents=True)
    write_jsonl(md.prepared_dir / "val.jsonl", [{"request": "val ask", "split": "val"}])
    with pytest.raises(DistillConfigError, match="is in prepared val"):
        run_distill(md, replay=True)


def test_a_pool_row_tagged_with_a_pinned_group_is_refused(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        [{"request": "new ask", "camera": "G341"}, {"request": "other", "camera": "G1"}],
        pins={"benchmark": ["camera:G341"]},
    )
    with pytest.raises(DistillConfigError, match="1 prompt row\\(s\\) carry pinned camera:G341"):
        run_distill(md)


def test_live_run_writes_cache_provenance_and_replay_reads_it(tmp_path: Path) -> None:
    md = _model(tmp_path, [{"request": "one"}, {"request": "two"}])
    summary = run_distill(md)
    assert summary["accepted"] == 2
    cache_path = md.model_dir / "datasets" / "cache.jsonl"
    rows = list(iter_jsonl(cache_path))
    assert rows[0]["kind"] == TEACHER_CACHE_KIND
    assert rows[0]["teacher"] == "fake@v1"
    pool = md.model_dir / "datasets" / "prompts.jsonl"
    assert rows[0]["prompt_sources"] == [{"path": str(pool), "sha256": sha256_file(pool)}]
    assert rows[0]["benchmark_version"] is None
    assert len(rows) == 3
    assert summary["cache_provenance"]["teacher"] == "fake@v1"

    cache = TeacherCache(cache_path)
    assert len(cache._store) == 2 and cache.provenance["teacher"] == "fake@v1"
    replayed = run_distill(md, replay=True)
    assert replayed["cache_hits"] == 2
    cards = list((md.model_dir / "datasets" / "samples").glob("*.md"))
    text = cards[0].read_text()
    assert "cache provenance:" in text and "'teacher': 'fake@v1'" in text
