"""Sources carry their licence: sidecar attribution table, prepare refusal, corpus lock."""

# ruff: noqa: E501  the attribution table literal mirrors a real one, long headers included

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maatml.config import ModelDefinition, PackagingSpec, load_model_def
from maatml.data.attribution import (
    AttributionError,
    check_sources,
    classify_commercial_use,
    read_attribution,
    read_corpus_lock,
    render_attribution_block,
)
from maatml.data.pipeline import prepare
from maatml.export.bundle import export_safetensors_bundle
from maatml.export.manifest import verify_manifest
from maatml.registry import discover_plugins
from maatml.utils.io import read_json, sha256_file

# A table shaped like a real one: long headers, extra columns, prose cells.
_TABLE = """# Corpus attribution

Prose before the table is ignored.

| source | version/checksum | URL | licence (verbatim link, retrieval date) | commercial-use yes/no/unknown | provenance/consent | fields ingested | attribution string | sign-off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | `synth.py` | (generated) | generated; no people | yes | no people | boxes | SIP synthetic | cleared |
| meva | starter | https://mevadata.org | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | yes (with attribution) | consenting actors | boxes | Contains material from MEVA, © Kitware Inc. and IARPA | cleared — A. Name 2026-08-17 |
| crowdhuman | fetch | https://www.crowdhuman.org/ | research / education only | no | crowd stills | boxes | CrowdHuman (research use) | accepted-risk — A. Name 2026-08-17 |
| wildtrack | fetch | — | unverified | unknown until filtered | public square | boxes | WILDTRACK | accepted-risk — A. Name 2026-08-17 |
| virat | — | https://viratdata.org | protection agreement | no — research only | signed agreement | boxes | VIRAT Ground 2.0 | cleared — agreement signed, A. Name 2026-08-22 |
| dukemtmc | — | — | withdrawn 2019 | no | non-consented | — | — | **do not use** |
| pending | — | — | tbd | yes | tbd | — | — | *unsigned* |
"""


def _attribution(tmp_path: Path, text: str = _TABLE) -> Path:
    path = tmp_path / "ATTRIBUTION.md"
    path.write_text(text, encoding="utf-8")
    return path


# --- table ---------------------------------------------------------------------------


def test_read_attribution_parses_a_real_shaped_table(tmp_path: Path) -> None:
    entries = read_attribution(_attribution(tmp_path))
    assert set(entries) == {
        "synthetic",
        "meva",
        "crowdhuman",
        "wildtrack",
        "virat",
        "dukemtmc",
        "pending",
    }
    meva = entries["meva"]
    assert meva.commercial_use == "yes" and meva.signed and not meva.risk_accepted
    assert meva.attribution.startswith("Contains material from MEVA")
    assert meva.consent == "consenting actors"
    assert meva.licence.startswith("[CC BY 4.0]")
    assert meva.raw["fields ingested"] == "boxes"
    assert entries["crowdhuman"].commercial_use == "no"
    assert entries["crowdhuman"].risk_accepted
    assert entries["wildtrack"].commercial_use == "unknown"
    assert entries["wildtrack"].risk_accepted
    assert entries["virat"].commercial_use == "no" and not entries["virat"].risk_accepted
    assert entries["dukemtmc"].blocked and not entries["dukemtmc"].signed
    assert not entries["pending"].signed and not entries["pending"].blocked


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("yes", "yes"),
        ("Yes (with attribution)", "yes"),
        ("no", "no"),
        ("no — research only", "no"),
        ("unknown", "unknown"),
        ("unknown until filtered", "unknown"),
        ("", "unknown"),
        ("not stated", "unknown"),
        ("yesterday", "unknown"),
    ],
)
def test_commercial_use_is_read_from_the_start_of_the_cell(cell: str, expected: str) -> None:
    assert classify_commercial_use(cell) == expected


def test_read_attribution_needs_the_four_columns_and_unique_sources(tmp_path: Path) -> None:
    with pytest.raises(AttributionError, match="sign-off"):
        read_attribution(
            _attribution(
                tmp_path,
                "| source | licence | commercial-use |\n| --- | --- | --- |\n| a | x | yes |\n",
            )
        )
    with pytest.raises(AttributionError, match="two rows"):
        read_attribution(
            _attribution(
                tmp_path,
                "| source | licence | commercial-use | sign-off |\n| --- | --- | --- | --- |\n"
                "| a | x | yes | ok |\n| a | y | yes | ok |\n",
            )
        )
    with pytest.raises(AttributionError, match="no Markdown table"):
        read_attribution(_attribution(tmp_path, "# nothing here\n"))
    with pytest.raises(AttributionError, match="does not exist"):
        read_attribution(tmp_path / "missing.md")


def test_check_sources_applies_every_rule(tmp_path: Path) -> None:
    entries = read_attribution(_attribution(tmp_path))
    rows = [
        {"source": "meva"},
        {"source": "meva"},
        {"source": "crowdhuman"},
        {"source": "wildtrack"},
    ]
    check = check_sources(rows, entries)
    assert check.ok
    assert check.used == {"meva": 2, "crowdhuman": 1, "wildtrack": 1}
    assert set(check.accepted_risk) == {"crowdhuman", "wildtrack"}

    check = check_sources(
        [
            {"source": "virat"},
            {"source": "dukemtmc"},
            {"source": "pending"},
            {"source": "nowhere"},
            {"request": "no source"},
        ],
        entries,
    )
    joined = "\n".join(check.refusals)
    assert "virat: commercial-use is no" in joined and "accepted-risk" in joined
    assert "dukemtmc: sign-off is '**do not use**'" in joined
    assert "pending: unsigned" in joined
    assert "nowhere: no attribution row" in joined
    assert "1 row(s) carry no `source`" in joined
    assert check.accepted_risk == {}


def test_check_sources_can_key_on_another_row_field(tmp_path: Path) -> None:
    """A corpus whose `source` is finer than its licence keys the table on `dataset`."""
    entries = read_attribution(_attribution(tmp_path))
    rows = [
        {"dataset": "meva", "source": "meva:kpf:2018-03-05.G339.geom.yml"},
        {"dataset": "meva", "source": "benchmark:holdout"},
        {"dataset": "crowdhuman", "source": "crowdhuman:odgt"},
    ]
    assert not check_sources(rows, entries).ok
    check = check_sources(rows, entries, field="dataset")
    assert check.ok and check.used == {"meva": 2, "crowdhuman": 1}
    unkeyed = check_sources([{"source": "meva"}], entries, field="dataset")
    assert "1 row(s) carry no `dataset`" in "\n".join(unkeyed.refusals)


# --- prepare -------------------------------------------------------------------------

_MODEL_YML = """name: attr-test
model_id: attr-test
version: 0.1.0
architecture: causal_sft
dataset:
  format: jsonl_seed
  seed: 7
  seed_samples: datasets/samples/seed_samples.jsonl
  attribution: datasets/ATTRIBUTION.md
  group_by: family
  split_ratios: [0.6, 0.2, 0.2]
evaluation:
  predictor: causal_sft
  gates:
    output_nonempty_rate: 0.5
training:
  base_model: sshleifer/tiny-gpt2
"""


def _rows(sources: list[str]) -> list[dict]:
    return [
        {
            "sample_id": f"s{i}",
            "request": f"ask {i}",
            "expected": "ok",
            "family": f"f{i}",
            "source": source,
        }
        for i, source in enumerate(sources)
    ]


def _model_dir(tmp_path: Path, rows: list[dict], *, with_table: bool = True) -> Path:
    mdir = tmp_path / "model"
    samples = mdir / "datasets" / "samples"
    samples.mkdir(parents=True)
    with (samples / "seed_samples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (mdir / "datasets" / "ATTRIBUTION.md").write_text(_TABLE, encoding="utf-8")
    yml = (
        _MODEL_YML
        if with_table
        else _MODEL_YML.replace("  attribution: datasets/ATTRIBUTION.md\n", "")
    )
    (mdir / "model.yml").write_text(yml)
    return mdir


def test_prepare_admits_signed_sources_and_writes_the_corpus_lock(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, _rows(["meva"] * 8 + ["crowdhuman"] * 4))
    md = load_model_def(mdir, load_plugins=False)
    summary = prepare(md)
    lock = read_corpus_lock(md.prepared_dir)
    assert lock is not None
    assert lock["lock_sha256"] == summary["corpus_lock"]
    assert summary["accepted_risk"] == {"crowdhuman": "accepted-risk — A. Name 2026-08-17"}
    seed = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    assert lock["files"] == [{"path": str(seed), "sha256": sha256_file(seed), "rows": 12}]
    assert lock["attribution"]["sha256"] == sha256_file(mdir / "datasets" / "ATTRIBUTION.md")
    assert [(s["source"], s["rows"]) for s in lock["sources"]] == [("crowdhuman", 4), ("meva", 8)]
    assert lock["sources"][1]["attribution"].startswith("Contains material from MEVA")
    card = (md.prepared_dir / "dataset_card.md").read_text()
    assert "Accepted risk: {'crowdhuman'" in card and "Corpus lock:" in card

    again = prepare(load_model_def(mdir, load_plugins=False))
    assert again["corpus_lock"] == summary["corpus_lock"]


def test_prepare_refuses_a_source_without_a_signed_row(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, _rows(["meva"] * 8 + ["virat"] * 4))
    md = load_model_def(mdir, load_plugins=False)
    with pytest.raises(AttributionError, match="virat: commercial-use is no"):
        prepare(md)
    assert not (md.prepared_dir / "train.jsonl").exists()
    assert read_corpus_lock(md.prepared_dir) is None

    mdir = _model_dir(tmp_path / "b", _rows(["meva"] * 8 + ["dukemtmc"] * 4))
    with pytest.raises(AttributionError, match="do not use"):
        prepare(load_model_def(mdir, load_plugins=False))

    mdir = _model_dir(tmp_path / "c", _rows(["meva"] * 8 + ["elsewhere"] * 4))
    with pytest.raises(AttributionError, match="elsewhere: no attribution row"):
        prepare(load_model_def(mdir, load_plugins=False))

    rows = _rows(["meva"] * 12)
    del rows[0]["source"]
    mdir = _model_dir(tmp_path / "d", rows)
    with pytest.raises(AttributionError, match="carry no `source`"):
        prepare(load_model_def(mdir, load_plugins=False))


def test_prepare_keys_the_table_on_dataset_attribution_field(tmp_path: Path) -> None:
    rows = _rows(["meva"] * 8 + ["crowdhuman"] * 4)
    for i, row in enumerate(rows):
        row["dataset"] = row.pop("source")
        row["source"] = f"{row['dataset']}:file{i}"
    mdir = _model_dir(tmp_path, rows)
    md = load_model_def(mdir, load_plugins=False)
    with pytest.raises(AttributionError, match="no attribution row"):
        prepare(md)

    yml = mdir / "model.yml"
    yml.write_text(
        yml.read_text().replace(
            "  attribution: datasets/ATTRIBUTION.md\n",
            "  attribution: datasets/ATTRIBUTION.md\n  attribution_field: dataset\n",
        )
    )
    md = load_model_def(mdir, load_plugins=False)
    summary = prepare(md)
    lock = read_corpus_lock(md.prepared_dir)
    assert lock is not None and lock["attribution"]["field"] == "dataset"
    assert [(s["source"], s["rows"]) for s in lock["sources"]] == [("crowdhuman", 4), ("meva", 8)]
    assert summary["accepted_risk"] == {"crowdhuman": "accepted-risk — A. Name 2026-08-17"}
    assert "keyed on `dataset`" in (md.prepared_dir / "dataset_card.md").read_text()

    yml.write_text(yml.read_text().replace("attribution_field: dataset", "attribution_field: 3"))
    with pytest.raises(AttributionError, match="attribution_field"):
        prepare(load_model_def(mdir, load_plugins=False))


def test_prepare_without_a_table_still_locks_the_input_files(tmp_path: Path) -> None:
    mdir = _model_dir(tmp_path, _rows(["virat"] * 12), with_table=False)
    md = load_model_def(mdir, load_plugins=False)
    summary = prepare(md)
    lock = read_corpus_lock(md.prepared_dir)
    assert lock is not None and lock["attribution"] is None and lock["sources"] == []
    assert len(lock["files"]) == 1 and summary["accepted_risk"] == {}


# --- export --------------------------------------------------------------------------


def test_export_carries_the_lock_and_ships_attribution_md(tmp_path: Path) -> None:
    discover_plugins(force=True)
    mdir = _model_dir(tmp_path, _rows(["meva"] * 8 + ["crowdhuman"] * 4))
    prepare(load_model_def(mdir, load_plugins=False))

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"not-real-weights")
    (ckpt / "config.json").write_text('{"architectures":["Toy"]}', encoding="utf-8")
    md = ModelDefinition(
        name="attr-test",
        model_id="attr-test",
        version="0.1.0",
        architecture="causal_sft",
        base_model="toy/base",
        dataset={},
        packaging=PackagingSpec(weights_dtype="f16"),
    )
    object.__setattr__(md, "model_dir", mdir)

    out = tmp_path / "export"
    export_safetensors_bundle(md, ckpt, out, run_id="run-1")
    manifest = read_json(out / "manifest.json")
    lock = read_corpus_lock(mdir / "output" / "prepared")
    assert lock is not None
    assert manifest["corpus_lock"]["lock_sha256"] == lock["lock_sha256"]
    assert manifest["corpus_lock"]["accepted_risk"] == {
        "crowdhuman": "accepted-risk — A. Name 2026-08-17"
    }
    assert "kind" not in manifest["corpus_lock"]
    paths = {e["path"] for e in manifest["files"]}
    assert "ATTRIBUTION.md" in paths
    assert verify_manifest(out) == []
    text = (out / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "## Sources" in text
    assert "**meva** (8 rows)" in text and "Contains material from MEVA" in text
    assert "**crowdhuman** (4 rows)" in text and "commercial-use no" in text
    assert "Risk accepted at prepare" in text and "Corpus lock `" in text


def test_render_attribution_block_without_sources_says_so() -> None:
    lines = render_attribution_block({"sources": [], "files": [], "lock_sha256": "ab" * 32})
    assert lines[0] == "## Sources"
    assert any("No attribution table" in line for line in lines)
    assert lines[-1].startswith("Corpus lock `abababababababab`: 0 input file(s)")
