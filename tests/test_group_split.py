"""Family-aware group split: members of one family must not straddle splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.data.pipeline import _assign_group_splits, _group_key, prepare, prepare_rows
from maatml.utils.io import iter_jsonl, write_jsonl


def test_family_group_does_not_straddle(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    # Two families, three samples each, enough that a per-sample hash
    # would often split a family, but group-hash must keep them together.
    rows = []
    for family in ("alpha", "beta", "gamma", "delta"):
        for i in range(4):
            rows.append(
                {
                    "sample_id": f"{family}-{i}",
                    "source": "hand",
                    "family": family,
                    "category": family,
                    "request": f"req {family} {i}",
                    "expected_output": {"answer": f"{family}-{i}"},
                }
            )
    write_jsonl(seeds, rows)
    (mdir / "model.yml").write_text(
        """name: group-split-test
model_id: group-split-test
task: support_ticket_triage
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  target_field: expected_output
""",
        encoding="utf-8",
    )

    md = load_model_def(mdir)
    prepare(md)

    family_to_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for row in iter_jsonl(md.prepared_dir / f"{split}.jsonl"):
            fam = row["family"]
            family_to_splits.setdefault(fam, set()).add(split)

    for fam, splits in family_to_splits.items():
        assert len(splits) == 1, f"family {fam!r} straddled splits: {splits}"


def test_group_by_config_prefers_field() -> None:
    row = {"family": "fam", "source": "src", "sample_id": "sid", "bucket": "b1"}
    assert _group_key(row, group_by="bucket") == "bucket:b1"
    # Missing preferred field → fall back to family chain
    assert _group_key({"family": "fam"}, group_by="bucket") == "family:fam"
    # Empty / null group_by keeps default chain
    assert _group_key(row, group_by=None) == "family:fam"
    assert _group_key(row, group_by="") == "family:fam"


def test_group_by_config_in_prepare(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    rows = []
    for bucket in ("x", "y"):
        for i in range(3):
            rows.append(
                {
                    "sample_id": f"{bucket}-{i}",
                    "bucket": bucket,
                    "family": "shared",  # would merge all if family used
                    "request": f"r {bucket} {i}",
                    "expected_output": {"a": i},
                }
            )
    write_jsonl(seeds, rows)
    (mdir / "model.yml").write_text(
        """name: group-by-cfg
model_id: group-by-cfg
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  group_by: bucket
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    prepare(md)
    bucket_to_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            bucket_to_splits.setdefault(row["bucket"], set()).add(split)
    for b, splits in bucket_to_splits.items():
        assert len(splits) == 1, f"bucket {b!r} straddled: {splits}"


def test_datagen_style_corpus_still_yields_val_and_test(tmp_path: Path) -> None:
    """One source, no families: prepare must not put everything in one split."""
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    rows = [
        {
            "sample_id": f"teacher-0-{i}",
            "source": "teacher",  # every row: one group under the old keying
            "request": f"req {i}",
            "expected_output": {"answer": i},
        }
        for i in range(40)
    ]
    write_jsonl(seeds, rows)
    (mdir / "model.yml").write_text(
        """name: degenerate-group
model_id: degenerate-group
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.6, 0.2, 0.2]
  target_field: expected_output
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    with pytest.warns(RuntimeWarning, match="covers 40/40 rows"):
        summary = prepare(md)

    assert summary["degenerate_group"] == "source:teacher"
    for split in ("train", "val", "test"):
        assert summary["split_counts"][split] > 0, summary["split_counts"]
    assert sum(summary["split_counts"].values()) == 40


def test_benchmark_sharing_a_family_with_train_is_refused(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: bench-leak
model_id: bench-leak
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    # Distinct request texts: rows that are byte-identical would trip the
    # content-level duplicate check, which is not what these tests are about.
    seed_rows = [
        {"sample_id": f"{fam}-{i}", "family": fam, "request": f"request {fam} {i}"}
        for fam in ("alpha", "beta", "gamma", "delta")
        for i in range(4)
    ]
    _splits, assignment, _degenerate = _assign_group_splits(seed_rows, (0.8, 0.1, 0.1))
    trained_family = next(key for key, split in assignment.items() if split.value == "train").split(
        ":", 1
    )[1]

    with pytest.raises(ValueError, match="share group keys with the training splits"):
        prepare_rows(
            md,
            seed_rows,
            out_dir=tmp_path / "out",
            benchmark_rows=[
                {
                    "sample_id": "b-1",
                    "family": trained_family,
                    "request": "unseen benchmark request",
                }
            ],
        )


def test_benchmark_with_its_own_family_is_pinned_to_test(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: bench-ok
model_id: bench-ok
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    # Distinct request texts: rows that are byte-identical would trip the
    # content-level duplicate check, which is not what these tests are about.
    seed_rows = [
        {"sample_id": f"{fam}-{i}", "family": fam, "request": f"request {fam} {i}"}
        for fam in ("alpha", "beta", "gamma", "delta")
        for i in range(4)
    ]
    summary = prepare_rows(
        md,
        seed_rows,
        out_dir=tmp_path / "out",
        benchmark_rows=[{"sample_id": "b-1", "family": "bench_alpha", "request": "unseen request"}],
    )
    test_rows = list(iter_jsonl(tmp_path / "out" / "test.jsonl"))
    assert any(row["sample_id"] == "b-1" for row in test_rows)
    assert summary["split_counts"]["test"] == len(test_rows)


def test_benchmark_duplicating_seed_content_is_refused(tmp_path: Path) -> None:
    """A benchmark drawn from the same generator as the seeds is caught even
    when it carries its own family namespace: re-tagging renames a duplicate,
    it does not prevent one. This is what let 66 of 120 triage benchmark rows
    score memorisation while the family-key guard reported no leakage."""
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: bench-dupe
model_id: bench-dupe
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
  request_field: request
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    seed_rows = [
        {"sample_id": f"{fam}-{i}", "family": fam, "request": f"request {fam} {i}"}
        for fam in ("alpha", "beta", "gamma", "delta")
        for i in range(4)
    ]
    # Same text, different sample_id and a bench: family namespace, exactly the
    # shape the old guard waved through.
    dupe = {
        "sample_id": "bench-1",
        "family": "bench:alpha",
        "request": "request alpha 0",
    }
    with pytest.raises(ValueError, match="duplicate seed rows by content"):
        prepare_rows(md, seed_rows, out_dir=tmp_path / "out", benchmark_rows=[dupe])


def test_benchmark_content_check_ignores_identity_fields(tmp_path: Path) -> None:
    """With no request_field configured the check hashes the row minus its
    identity fields, so a re-tagged duplicate still collides."""
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: bench-dupe2
model_id: bench-dupe2
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    seed_rows = [
        {"sample_id": f"s-{i}", "family": f"fam{i}", "source": "seed", "payload": i}
        for i in range(8)
    ]
    dupe = {"sample_id": "b-1", "family": "bench:fam0", "source": "bench", "payload": 0}
    with pytest.raises(ValueError, match="duplicate seed rows by content"):
        prepare_rows(md, seed_rows, out_dir=tmp_path / "out", benchmark_rows=[dupe])


def test_train_only_rows_join_train_and_never_a_held_out_group(tmp_path: Path) -> None:
    """Rows a policy collected from its own rollouts train the next round but
    must not move val or test, and must not smuggle in a held-out group."""
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        "name: rounds\nmodel_id: rounds\narchitecture: causal_sft\nversion: 0.1.0\n"
        "dataset:\n  seed_samples: seeds.jsonl\n",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    seed_rows = [
        {"sample_id": f"{fam}-{i}", "family": fam, "request": f"request {fam} {i}"}
        for fam in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
        for i in range(4)
    ]
    rollout = [{"sample_id": "r-1", "family": "rollout_fam", "request": "visited state"}]
    summary = prepare_rows(md, seed_rows, out_dir=tmp_path / "out", train_rows=rollout)

    def ids(split: str) -> set[str]:
        return {row["sample_id"] for row in iter_jsonl(tmp_path / "out" / f"{split}.jsonl")}

    assert "r-1" in ids("train") and "r-1" not in ids("val") | ids("test")
    assert summary["split_counts"]["train"] == len(ids("train"))

    held_out = next(iter(ids("val") | ids("test"))).rsplit("-", 1)[0]
    with pytest.raises(ValueError, match="held-out group"):
        prepare_rows(
            md,
            seed_rows,
            out_dir=tmp_path / "out2",
            train_rows=[{"sample_id": "r-2", "family": held_out, "request": "leak"}],
        )
