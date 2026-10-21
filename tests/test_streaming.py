"""Streaming prepare: same decisions as ``prepare_rows``, rows never held in memory."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
import yaml

import maatml.data.streaming as streaming
from maatml.config import ModelDefinition, load_model_def
from maatml.data.attribution import AttributionError
from maatml.data.pipeline import prepare_rows
from maatml.data.populations import PopulationError
from maatml.data.streaming import BENCHMARK, BLIND, SEED, TRAIN_ONLY, prepare_stream

_TABLE = """# Corpus attribution

| source | version/checksum | URL | licence (verbatim link, retrieval date) | commercial-use yes/no/unknown | provenance/consent | fields ingested | attribution string | sign-off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | `synth.py` | (generated) | generated; no people | yes | no people | request | SIP synthetic | cleared |
| pending | - | - | tbd | yes | tbd | - | - | *unsigned* |
"""  # noqa: E501


def _model(tmp_path: Path, **dataset: Any) -> ModelDefinition:
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = {"seed_samples": "seeds.jsonl", "request_field": "request", **dataset}
    cfg = {k: v for k, v in cfg.items() if v is not None}
    (mdir / "model.yml").write_text(
        yaml.safe_dump(
            {
                "name": "streaming",
                "model_id": "streaming",
                "architecture": "causal_sft",
                "version": "0.1.0",
                "dataset": cfg,
            }
        ),
        encoding="utf-8",
    )
    return load_model_def(mdir)


def _row(family: str, i: int, **extra: Any) -> dict:
    return {
        "sample_id": f"{family}-{i}",
        "source": "synthetic",
        "family": family,
        "category": "even" if i % 2 == 0 else "odd",
        "request": f"request of {family} number {i}",
        **extra,
    }


def _families(names: str, per: int = 4, **extra: Any) -> list[dict]:
    return [_row(f"{name}", i, **extra) for name in names.split() for i in range(per)]


def _stream(
    seed: list[dict],
    bench: Optional[list[dict]] = None,
    train: Optional[list[dict]] = None,
    blind: Optional[list[dict]] = None,
) -> Iterator[tuple[str, dict]]:
    """Seed rows with the other populations interleaved among them."""
    others = (
        [(BENCHMARK, r) for r in bench or []]
        + [(TRAIN_ONLY, r) for r in train or []]
        + [(BLIND, r) for r in blind or []]
    )
    for n, row in enumerate(seed):
        yield SEED, row
        if n % 3 == 2 and others:
            yield others.pop(0)
    yield from others


def _attempt(call: Any) -> tuple[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return "ok", call()
        except Exception as exc:  # noqa: BLE001  the outcome is what is compared
            return type(exc).__name__, str(exc)


def _lines(path: Path) -> list[str]:
    return sorted(path.read_text(encoding="utf-8").splitlines())


def _both(
    tmp_path: Path,
    md: ModelDefinition,
    seed: list[dict],
    *,
    bench: Optional[list[dict]] = None,
    train: Optional[list[dict]] = None,
    blind: Optional[list[dict]] = None,
) -> tuple[tuple[str, Any], tuple[str, Any]]:
    old = _attempt(
        lambda: prepare_rows(
            md,
            seed,
            out_dir=tmp_path / "old",
            benchmark_rows=bench,
            train_rows=train,
            blind_rows=blind,
        )
    )
    new = _attempt(
        lambda: prepare_stream(md, _stream(seed, bench, train, blind), out_dir=tmp_path / "new")
    )
    return old, new


def _assert_same_outputs(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        assert _lines(tmp_path / "new" / f"{split}.jsonl") == _lines(
            tmp_path / "old" / f"{split}.jsonl"
        ), split
    for name in ("dataset_card.md", "benchmark.json", "corpus.lock.json"):
        new, old = tmp_path / "new" / name, tmp_path / "old" / name
        assert new.read_bytes() == old.read_bytes(), name


def _assert_same_summary(old: dict, new: dict) -> None:
    for key in old:
        if key in ("out_dir", "splits", "card"):
            continue
        assert new[key] == old[key], key


# --- same decisions as prepare_rows ---------------------------------------------------


def test_stream_matches_prepare_rows_with_benchmark_and_rollouts(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family", split_ratios=[0.6, 0.2, 0.2])
    seed = _families("alpha beta gamma delta epsilon zeta eta theta iota kappa", per=5)
    bench = _families("bench1 bench2", per=3)
    rollouts = [_row("roll1", 0, source="synthetic"), _row("roll2", 0)]
    old, new = _both(tmp_path, md, seed, bench=bench, train=rollouts)
    assert old[0] == new[0] == "ok", (old, new)
    _assert_same_outputs(tmp_path)
    _assert_same_summary(old[1], new[1])
    assert new[1]["population_rows"] == {SEED: 50, BENCHMARK: 6, BLIND: 0, TRAIN_ONLY: 2}
    # Seed rows, then benchmark rows in test, then train-only rows in train: the
    # layout prepare_rows writes when groups are contiguous.
    test_lines = (tmp_path / "new" / "test.jsonl").read_text().splitlines()
    test_ids = [json.loads(x)["sample_id"] for x in test_lines]
    assert test_ids[-6:] == [r["sample_id"] for r in bench]
    for split in ("train", "val", "test"):
        assert (tmp_path / "new" / f"{split}.jsonl").read_bytes() == (
            tmp_path / "old" / f"{split}.jsonl"
        ).read_bytes()


def test_no_scratch_files_remain(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family")
    prepare_stream(md, _stream(_families("a b c d e f")), out_dir=tmp_path / "new")
    assert sorted(p.name for p in (tmp_path / "new").iterdir() if p.name.startswith(".")) == []


def test_benchmark_version_survives_an_external_sort_with_many_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(streaming, "_SORT_CHUNK_CHARS", 300)
    md = _model(tmp_path, group_by="family", split_ratios=[0.5, 0.2, 0.3])
    seed = _families("a b c d e f g h i j k l m n o p", per=6)
    bench = _families("zz yy xx", per=4)
    old, new = _both(tmp_path, md, seed, bench=bench)
    assert old[0] == new[0] == "ok"
    assert new[1]["benchmark_version"] == old[1]["benchmark_version"]
    _assert_same_outputs(tmp_path)


def test_degenerate_group_is_split_per_row_like_prepare_rows(tmp_path: Path) -> None:
    md = _model(tmp_path, split_ratios=[0.6, 0.2, 0.2])
    seed = [{"sample_id": f"t-{i}", "source": "teacher", "request": f"req {i}"} for i in range(38)]
    seed += [_row("odd1", 0), _row("odd2", 0)]
    with pytest.warns(RuntimeWarning, match="covers 38/40 rows"):
        new = prepare_stream(md, _stream(seed), out_dir=tmp_path / "new")
    old = prepare_rows(md, seed, out_dir=tmp_path / "old")
    assert new["degenerate_group"] == old["degenerate_group"] == "source:teacher"
    _assert_same_outputs(tmp_path)
    _assert_same_summary(old, new)
    for split in ("train", "val", "test"):
        assert new["split_counts"][split] > 0
        assert (tmp_path / "new" / f"{split}.jsonl").read_bytes() == (
            tmp_path / "old" / f"{split}.jsonl"
        ).read_bytes()


def test_degenerate_group_with_pins_and_benchmark(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        split_ratios=[0.6, 0.2, 0.2],
        isolation={"fields": ["clip", "camera"], "policy": {"val": "camera"}},
        pins={"val": ["camera:cam7"]},
    )
    seed = [
        {"sample_id": f"t-{i}", "source": "teacher", "camera": f"cam{i % 8}", "clip": f"c{i}",
         "request": f"req {i}"}
        for i in range(60)
    ]
    old, new = _both(tmp_path, md, seed)
    assert old[0] == new[0], (old, new)
    if old[0] == "ok":
        _assert_same_outputs(tmp_path)
        _assert_same_summary(old[1], new[1])
    else:
        assert old[1] == new[1]


def test_one_group_per_row_when_sample_id_is_the_only_key(tmp_path: Path) -> None:
    md = _model(tmp_path)
    seed = [{"sample_id": f"s{i}", "request": f"r{i}"} for i in range(60)]
    old, new = _both(tmp_path, md, seed)
    assert old[0] == new[0] == "ok"
    _assert_same_outputs(tmp_path)


# --- pins and isolation ----------------------------------------------------------------


def _camera_rows() -> list[dict]:
    return [
        {
            "sample_id": f"cam{c}-clip{k}-{i}",
            "source": "synthetic",
            "camera": f"cam{c}",
            "clip": f"cam{c}-clip{k}",
            "request": f"cam {c} clip {k} row {i}",
        }
        for c in range(8)
        for k in range(3)
        for i in range(2)
    ]


def test_pins_and_isolation_match_prepare_rows(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        group_by="clip",
        isolation={
            "fields": ["clip", "camera"],
            "policy": {"val": "camera", "benchmark": "camera"},
        },
        pins={"val": ["camera:cam0"], "benchmark": ["camera:cam1"]},
    )
    old, new = _both(tmp_path, md, _camera_rows())
    assert old[0] == new[0] == "ok", (old, new)
    _assert_same_outputs(tmp_path)
    _assert_same_summary(old[1], new[1])
    assert new[1]["pins"] == {"camera:cam0": 6, "camera:cam1": 6}
    cameras = {
        split: {
            json.loads(x)["camera"]
            for x in (tmp_path / "new" / f"{split}.jsonl").read_text().splitlines()
        }
        for split in ("train", "val", "test")
    }
    assert cameras["val"] == {"cam0"} and cameras["test"] == {"cam1"}
    assert not cameras["train"] & (cameras["val"] | cameras["test"])


def test_a_pin_that_matches_nothing_is_refused_with_the_same_message(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="clip", pins={"val": ["camera:nowhere"]})
    old, new = _both(tmp_path, md, _camera_rows())
    assert old[0] == new[0] == PopulationError.__name__
    assert old[1] == new[1] and "matches no row" in new[1]


def test_isolation_violation_is_refused_with_the_same_message(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        group_by="clip",
        split_ratios=[0.4, 0.3, 0.3],
        isolation={
            "fields": ["clip", "camera"],
            "policy": {"val": "camera", "benchmark": "camera"},
        },
    )
    old, new = _both(tmp_path, md, _camera_rows())
    assert old[0] == new[0] == "ValueError", (old, new)
    assert old[1] == new[1] and "dataset.isolation violated" in new[1]


def test_blind_rows_are_isolation_checked_and_never_written(tmp_path: Path) -> None:
    md = _model(
        tmp_path,
        group_by="clip",
        isolation={"fields": ["clip", "camera", "site"], "policy": {"blind": "site"}},
    )
    seed = [dict(r, site=f"site{int(r['camera'][3:]) % 2}") for r in _camera_rows()]
    blind_ok = [
        {"sample_id": "b1", "source": "synthetic", "site": "elsewhere", "request": "blind 1"}
    ]
    blind_bad = [dict(blind_ok[0], site="site0", request="blind 2", sample_id="b2")]
    old, new = _both(tmp_path, md, seed, blind=blind_ok)
    assert old[0] == new[0] == "ok", (old, new)
    assert new[1]["blind_rows"] == 1
    _assert_same_outputs(tmp_path)
    old, new = _both(tmp_path, md, seed, blind=blind_bad)
    assert old[0] == new[0] == "ValueError" and old[1] == new[1]
    assert "blind is site-disjoint" in new[1]


# --- benchmark and train-only checks -----------------------------------------------------


def test_benchmark_duplicating_seed_content_is_refused(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family")
    seed = _families("alpha beta gamma delta epsilon zeta", per=3)
    bench = [_row("own", i) for i in range(3)]
    bench[1] = dict(bench[1], request=seed[4]["request"])
    old, new = _both(tmp_path, md, seed, bench=bench)
    assert old[0] == new[0] == "ValueError", (old, new)
    assert old[1] == new[1]
    assert "1 benchmark row(s) duplicate seed rows by content: own-1 == " in new[1]


def test_benchmark_content_check_ignores_identity_fields(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family", request_field=None)
    seed = [
        {"sample_id": f"s{i}", "family": f"f{i}", "source": "x", "payload": {"n": i}}
        for i in range(8)
    ]
    bench = [{"sample_id": "retagged", "family": "other", "source": "y", "payload": {"n": 3}}]
    old, new = _both(tmp_path, md, seed, bench=bench)
    assert old[0] == new[0] == "ValueError" and old[1] == new[1]
    assert "retagged == s3" in new[1]


def test_benchmark_sharing_a_group_with_training_is_refused(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family", split_ratios=[0.5, 0.25, 0.25])
    seed = _families("alpha beta gamma delta epsilon zeta eta theta", per=3)
    bench = [_row("alpha", 99), _row("beta", 99), _row("gamma", 99), _row("delta", 99)]
    old, new = _both(tmp_path, md, seed, bench=bench)
    assert old[0] == new[0] == "ValueError", (old, new)
    assert old[1] == new[1] and "share group keys with the training splits" in new[1]


def test_blind_rows_duplicating_the_benchmark_are_refused(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family")
    seed = _families("alpha beta gamma delta epsilon zeta", per=3)
    bench = [_row("own", 0)]
    blind = [_row("blind", 0, request=bench[0]["request"])]
    old, new = _both(tmp_path, md, seed, bench=bench, blind=blind)
    assert old[0] == new[0] == "ValueError" and old[1] == new[1]
    assert "own-0" in new[1] and "blind-0" in new[1]


def test_train_only_row_in_a_held_out_group_is_refused(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family", split_ratios=[0.5, 0.25, 0.25])
    seed = _families("alpha beta gamma delta epsilon zeta eta theta", per=3)
    old = prepare_rows(md, seed, out_dir=tmp_path / "probe")
    held = next(
        json.loads(x)["family"] for x in (tmp_path / "probe" / "val.jsonl").read_text().splitlines()
    )
    leak = [_row(held, 77)]
    old, new = _both(tmp_path, md, seed, train=leak)
    assert old[0] == new[0] == "ValueError" and old[1] == new[1]
    assert "held-out group" in new[1]


# --- attribution, chat rows, and what a refusal leaves behind ----------------------------------


def test_attribution_refuses_unsigned_and_unsourced_rows_alike(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "ATTRIBUTION.md").write_text(_TABLE, encoding="utf-8")
    md = _model(tmp_path, group_by="family", attribution="ATTRIBUTION.md")
    seed = _families("alpha beta gamma delta", per=3)
    ok_old, ok_new = _both(tmp_path, md, seed)
    assert ok_old[0] == ok_new[0] == "ok", (ok_old, ok_new)
    _assert_same_outputs(tmp_path)
    assert ok_new[1]["corpus_lock"] == ok_old[1]["corpus_lock"]

    bad = seed + [_row("elsewhere", 0, source="elsewhere"), _row("pend", 0, source="pending")]
    bad.append({k: v for k, v in _row("nosource", 0).items() if k != "source"})
    old, new = _both(tmp_path, md, bad)
    assert old[0] == new[0] == AttributionError.__name__, (old, new)
    assert old[1] == new[1]
    assert "elsewhere: no attribution row" in new[1] and "pending: unsigned" in new[1]
    assert "1 row(s) carry no `source`" in new[1]


def test_refusal_leaves_the_previous_output_untouched(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "ATTRIBUTION.md").write_text(_TABLE, encoding="utf-8")
    md = _model(tmp_path, group_by="family", attribution="ATTRIBUTION.md")
    out = tmp_path / "new"
    prepare_stream(md, _stream(_families("a b c d e f")), out_dir=out)
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    bad = _families("a b c d e f") + [_row("x", 0, source="elsewhere")]
    with pytest.raises(AttributionError, match="elsewhere"):
        prepare_stream(md, _stream(bad), out_dir=out)
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before
    assert [p for p in out.iterdir() if p.is_dir()] == []


def test_malformed_chat_rows_are_refused_with_the_same_message(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family")
    seed = _families("alpha beta gamma delta epsilon zeta", per=4)
    for n in range(0, 24, 2):
        seed[n] = dict(seed[n], tools="not a list")
    bench = [_row("own", 0, tools="nope"), _row("own", 1)]
    old, new = _both(tmp_path, md, seed, bench=bench)
    assert old[0] == new[0] == "ValueError", (old, new)
    assert old[1] == new[1]
    assert "tools must be a list" in new[1] and "... and " in new[1]


def test_empty_split_warning_is_kept(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family")
    seed = _families("a b c d e f", per=3)
    with pytest.warns(RuntimeWarning, match="are empty: 6 group"):
        prepare_stream(md, _stream(seed), out_dir=tmp_path / "new")


def test_unknown_population_is_refused(tmp_path: Path) -> None:
    md = _model(tmp_path)
    with pytest.raises(ValueError, match="unknown population"):
        prepare_stream(md, iter([("mystery", _row("a", 0))]), out_dir=tmp_path / "new")
    assert not list((tmp_path / "new").glob(".prepare-*"))


def test_labels_and_input_files_may_be_resolved_after_the_stream(tmp_path: Path) -> None:
    md = _model(tmp_path, group_by="family")
    seen: list[str] = []

    def rows() -> Iterator[tuple[str, dict]]:
        yield from _stream(_families("a b c d e f"))
        seen.append("done")

    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text('{"sample_id": "b"}\n', encoding="utf-8")

    def label() -> str:
        assert seen == ["done"]
        return str(benchmark)

    summary = prepare_stream(
        md, rows(), out_dir=tmp_path / "new", benchmark_label=label, input_files=lambda: [benchmark]
    )
    assert f"Benchmark source: {benchmark}" in Path(summary["card"]).read_text(encoding="utf-8")
    lock = json.loads((tmp_path / "new" / "corpus.lock.json").read_text(encoding="utf-8"))
    assert [f["path"] for f in lock["files"]] == [str(benchmark)]
