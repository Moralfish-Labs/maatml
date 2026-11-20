"""Adapter export: exactly two files, checked against a layout spec (no torch needed)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Optional

import pytest

from maatml.config import ModelDefinition
from maatml.export.adapter import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    check_adapter,
    export_adapter,
    find_adapter,
    load_adapter_spec,
)
from maatml.export.bundle import export_model, resolve_export_format
from maatml.export.manifest import read_safetensors_shapes, verify_manifest
from maatml.registry import discover_plugins
from maatml.utils.io import read_json

RANK = 4
# Two adapted projections, as PEFT names them in a saved adapter.
LAYOUT = {
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": [RANK, 8],
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": [6, RANK],
    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight": [RANK, 16],
    "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight": [8, RANK],
}


@pytest.fixture(autouse=True)
def _boot_exporters():
    discover_plugins(force=True)


def _write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def _checkpoint(
    tmp_path: Path,
    shapes: Optional[dict[str, list[int]]] = None,
    *,
    rank: Any = RANK,
    sub: str = "",
) -> Path:
    """A checkpoint as ``save_mode: adapter`` leaves it: the adapter beside tokenizer files."""
    ckpt = tmp_path / "ckpt"
    folder = ckpt / sub if sub else ckpt
    folder.mkdir(parents=True)
    _write_safetensors(folder / ADAPTER_WEIGHTS, LAYOUT if shapes is None else shapes)
    (folder / ADAPTER_CONFIG).write_text(
        json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": 2 * RANK}), encoding="utf-8"
    )
    for extra in ("tokenizer.json", "chat_template.jinja", "README.md", "run_metadata.json"):
        (ckpt / extra).write_text("{}", encoding="utf-8")
    return ckpt


def _model(tmp_path: Path, **adapter: Any) -> ModelDefinition:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    md = ModelDefinition(
        name="toy-adapter",
        model_id="toy-adapter",
        version="0.1.0",
        architecture="causal_sft",
        base_model="toy/base",
        extensions={"adapter": adapter} if adapter else {},
    )
    object.__setattr__(md, "model_dir", model_dir)
    return md


def _spec(tmp_path: Path, tensors: Optional[dict[str, list[int]]] = None, rank: int = RANK) -> str:
    path = tmp_path / "model" / "adapter-spec.json"
    path.parent.mkdir(exist_ok=True)
    # Extra keys, as a spec written by a serving gate carries, are ignored.
    payload = {"rank": rank, "alpha": 8, "tensors": LAYOUT if tensors is None else tensors}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.name


def test_export_writes_exactly_the_two_adapter_files(tmp_path: Path) -> None:
    md = _model(tmp_path, name="e4b-wb", spec=_spec(tmp_path))
    out = tmp_path / "export"
    assert export_model(md, _checkpoint(tmp_path), out, format="adapter", run_id="run-1") == out

    folder = out / "adapters" / "e4b-wb"
    assert sorted(p.name for p in folder.iterdir()) == [ADAPTER_CONFIG, ADAPTER_WEIGHTS]
    assert sorted(p.name for p in out.iterdir()) == ["adapters", "manifest.json"]
    assert read_safetensors_shapes(folder / ADAPTER_WEIGHTS) == LAYOUT

    manifest = read_json(out / "manifest.json")
    assert {e["path"] for e in manifest["files"]} == {
        f"adapters/e4b-wb/{ADAPTER_CONFIG}",
        f"adapters/e4b-wb/{ADAPTER_WEIGHTS}",
    }
    hints = manifest["runtime_hints"]
    assert hints["formats"] == ["adapter"]
    assert hints["weights_dtype"] == "bf16"
    assert hints["adapter"]["name"] == "e4b-wb"
    assert (hints["adapter"]["rank"], hints["adapter"]["tensors"]) == (RANK, len(LAYOUT))
    assert len(hints["adapter"]["spec"]["sha256"]) == 64
    assert verify_manifest(out) == []


def test_the_name_defaults_to_the_model_and_no_spec_is_recorded_as_none(tmp_path: Path) -> None:
    out = tmp_path / "export"
    export_adapter(_model(tmp_path), _checkpoint(tmp_path), out)
    assert (out / "adapters" / "toy-adapter" / ADAPTER_WEIGHTS).is_file()
    assert read_json(out / "manifest.json")["runtime_hints"]["adapter"]["spec"] is None


def test_a_rerun_replaces_the_folder_instead_of_adding_to_it(tmp_path: Path) -> None:
    md = _model(tmp_path, name="e4b-wb")
    out = tmp_path / "export"
    stale = out / "adapters" / "e4b-wb"
    stale.mkdir(parents=True)
    (stale / "README.md").write_text("left by an earlier tool", encoding="utf-8")
    export_adapter(md, _checkpoint(tmp_path), out)
    assert sorted(p.name for p in stale.iterdir()) == [ADAPTER_CONFIG, ADAPTER_WEIGHTS]


def test_the_adapter_is_found_under_save_mode_both(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, sub="adapter")
    assert find_adapter(ckpt) == ckpt / "adapter"
    export_adapter(_model(tmp_path, name="a"), ckpt, tmp_path / "export")
    assert (tmp_path / "export" / "adapters" / "a" / ADAPTER_CONFIG).is_file()


def test_a_merged_checkpoint_has_no_adapter_to_export(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"merged")
    with pytest.raises(FileNotFoundError, match="save_mode: adapter or both"):
        export_adapter(_model(tmp_path), ckpt, tmp_path / "export")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "1 spec tensor(s) missing: .*q_proj.lora_B.weight"),
        ("extra", "1 tensor(s) not in the spec: .*vision_tower"),
        (
            "reshaped",
            r"1 tensor(s) differ in shape: .*down_proj.lora_A.weight \[4, 32\] \(spec \[4, 16\]\)",
        ),
    ],
)
def test_a_layout_that_differs_from_the_spec_is_refused(
    tmp_path: Path, change: str, message: str
) -> None:
    shapes = dict(LAYOUT)
    if change == "missing":
        del shapes["base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"]
    elif change == "extra":
        shapes["base_model.model.model.vision_tower.layers.0.q_proj.lora_A.weight"] = [RANK, 8]
    else:
        shapes["base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"] = [
            RANK,
            32,
        ]
    md = _model(tmp_path, name="e4b-wb", spec=_spec(tmp_path))
    out = tmp_path / "export"
    with pytest.raises(ValueError, match=message.replace("(s)", r"\(s\)")):
        export_adapter(md, _checkpoint(tmp_path, shapes), out)
    assert not out.exists()


def test_another_rank_is_refused_even_when_names_match(tmp_path: Path) -> None:
    doubled = {
        name: ([2 * RANK, shape[1]] if ".lora_A." in name else [shape[0], 2 * RANK])
        for name, shape in LAYOUT.items()
    }
    md = _model(tmp_path, name="e4b-wb", spec=_spec(tmp_path))
    with pytest.raises(ValueError, match=f"rank {2 * RANK}, the spec has {RANK}"):
        export_adapter(md, _checkpoint(tmp_path, doubled, rank=2 * RANK), tmp_path / "export")


def test_tensors_must_have_the_rank_the_config_states(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="4 tensor.s. are not rank 8"):
        check_adapter(_checkpoint(tmp_path, rank=8))
    assert check_adapter(_checkpoint(tmp_path / "ok")) == {"rank": RANK, "tensors": len(LAYOUT)}


@pytest.mark.parametrize("rank", [None, 0, True, "16"])
def test_a_config_without_a_usable_rank_is_refused(tmp_path: Path, rank: Any) -> None:
    with pytest.raises(ValueError, match="states no rank"):
        check_adapter(_checkpoint(tmp_path, rank=rank))


def test_unreadable_weights_are_refused(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path)
    (ckpt / ADAPTER_WEIGHTS).write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="no readable tensor header"):
        export_adapter(_model(tmp_path), ckpt, tmp_path / "export")


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"name": "../escape"}, "plain folder name"),
        ({"name": "a/b"}, "plain folder name"),
        ({"name": ""}, "plain folder name"),
        ({"name": "ok", "sepc": "typo.json"}, "unknown key"),
        ({"name": "ok", "spec": 3}, "must be a path"),
        ({"name": "ok", "spec": "missing-spec.json"}, "unreadable"),
    ],
)
def test_settings_are_validated(tmp_path: Path, settings: dict, message: str) -> None:
    out = tmp_path / "export"
    with pytest.raises(ValueError, match=message):
        export_adapter(_model(tmp_path, **settings), _checkpoint(tmp_path), out)
    assert not out.exists()


def test_a_spec_needs_a_rank_and_tensor_shapes(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    for payload in (
        {"tensors": LAYOUT},
        {"rank": 4, "tensors": {}},
        {"rank": 4, "tensors": {"a": "4x8"}},
        [1],
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="integer rank and tensors"):
            load_adapter_spec(path)
    path.write_text(json.dumps({"rank": 4, "tensors": LAYOUT}), encoding="utf-8")
    assert load_adapter_spec(path).tensors == LAYOUT


def test_adapter_is_a_registered_format_for_any_architecture() -> None:
    assert resolve_export_format("causal_sft", "adapter") == "adapter"
    assert resolve_export_format("dpo", "adapter") == "adapter"


def test_shape_reader_rejects_a_malformed_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    blob = json.dumps({"w": {"dtype": "F32", "shape": "4x8", "data_offsets": [0, 4]}}).encode(
        "utf-8"
    )
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)
    assert read_safetensors_shapes(path) == {}
    assert read_safetensors_shapes(tmp_path / "missing.safetensors") == {}
