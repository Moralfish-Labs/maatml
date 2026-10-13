"""Vision example tests, pure-python by default; torch/PIL optional."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def plugin():
    from vision_plugin import (  # noqa: F401
        compute_vision_metrics,
        validate_vision_scene,
    )

    return True


def test_validate_model_dir() -> None:
    from maatml.scaffold import validate_model_dir

    errors = validate_model_dir(ROOT)
    assert errors == [], errors


def test_resolve_image_confined_to_model_dir(tmp_path) -> None:
    """S5: request image paths must stay inside model_dir (no LFI at serve)."""
    from vision_plugin.dataset import resolve_image_bytes_or_path

    img = tmp_path / "images" / "ok.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"PNGDATA")
    assert resolve_image_bytes_or_path("images/ok.png", model_dir=tmp_path) == b"PNGDATA"

    with pytest.raises(ValueError):
        resolve_image_bytes_or_path("/etc/passwd", model_dir=tmp_path)
    with pytest.raises(ValueError):
        resolve_image_bytes_or_path("../secret.png", model_dir=tmp_path)
    with pytest.raises(ValueError):
        resolve_image_bytes_or_path("x.png", model_dir=None)
    # A symlink inside model_dir pointing outside must be rejected: this is the
    # case the containment check relies on .resolve() to catch.
    import os

    secret = tmp_path.parent / "secret.txt"
    secret.write_text("s", encoding="utf-8")
    os.symlink(secret, tmp_path / "images" / "evil")
    with pytest.raises(ValueError):
        resolve_image_bytes_or_path("images/evil", model_dir=tmp_path)


def test_validator_accepts_expected_payload(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    seed = ROOT / "datasets" / "samples" / "seed_samples.jsonl"
    row = json.loads(seed.read_text(encoding="utf-8").splitlines()[0])
    result = validate_vision_scene(
        json.dumps(row["expected"]),
        schema_path=ROOT / "datasets" / "schema.json",
    )
    assert result.ok, result.errors


def test_validator_rejects_bad_json(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    result = validate_vision_scene("not-json")
    assert not result.ok
    err = next(e for e in result.errors if e.code == "invalid_json")
    assert err.layer == 1
    assert err.location and "line" in err.location
    assert err.hint and "JSON" in err.hint


def test_validator_rejects_non_object_root(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    result = validate_vision_scene("[1, 2, 3]")
    assert not result.ok
    err = next(e for e in result.errors if e.code == "not_object")
    assert err.location == "$"
    assert "list" in err.message
    assert err.hint


def test_validator_schema_error_names_location(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    # scene.label must be a string per schema; a number fails layer 2.
    payload = {
        "scene": {"label": 1},
        "detections": [],
        "pose": {
            "keypoints": [
                {"name": n, "x": 0.0, "y": 0.0}
                for n in [
                    "head",
                    "neck",
                    "l_shoulder",
                    "r_shoulder",
                    "l_elbow",
                    "r_elbow",
                    "l_wrist",
                    "r_wrist",
                    "hip",
                    "l_knee",
                    "r_knee",
                    "feet",
                ]
            ]
        },
    }
    result = validate_vision_scene(
        json.dumps(payload),
        schema_path=ROOT / "datasets" / "schema.json",
    )
    assert not result.ok
    err = next(e for e in result.errors if e.code == "schema")
    assert err.layer == 2
    assert err.location == "scene/label"
    assert err.hint and "schema.json" in err.hint


def test_validator_reports_every_bad_detection(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    payload = {
        "scene": {"label": "plain"},
        "detections": [
            {"label": "circle"},  # missing box
            "not-an-object",
            {"label": "square", "box": [0.1, 0.2]},  # wrong box length
        ],
        "pose": {"keypoints": [{"name": "head", "x": 0.5, "y": 0.1}]},
    }
    result = validate_vision_scene(json.dumps(payload))
    assert not result.ok
    det_errors = [
        e
        for e in result.errors
        if e.layer == 3 and e.location and e.location.startswith("detections")
    ]
    assert len(det_errors) >= 3
    assert any(e.code == "detection_item" and e.location == "detections[0]" for e in det_errors)
    assert any(e.code == "detection_item" and e.location == "detections[1]" for e in det_errors)
    assert any(e.code == "box_shape" and e.location == "detections[2].box" for e in det_errors)
    assert all(e.hint for e in det_errors)


def test_validator_enum_errors_list_allowed_values(plugin) -> None:
    from vision_plugin.constants import KEYPOINT_NAMES, SCENE_LABELS, SHAPE_LABELS
    from vision_plugin.validator import validate_vision_scene

    payload = {
        "scene": {"label": "nebula"},
        "detections": [{"label": "hexagon", "box": [0.1, 0.2, 0.3, 0.4]}],
        "pose": {"keypoints": [{"name": "head", "x": 0.5, "y": 0.1}]},
    }
    result = validate_vision_scene(json.dumps(payload))
    assert not result.ok

    scene_err = next(e for e in result.errors if e.code == "scene_label")
    assert scene_err.location == "scene.label"
    assert "nebula" in scene_err.message
    assert all(label in (scene_err.hint or "") for label in SCENE_LABELS)

    shape_err = next(e for e in result.errors if e.code == "shape_label")
    assert shape_err.location == "detections[0].label"
    assert "hexagon" in shape_err.message
    assert all(label in (shape_err.hint or "") for label in SHAPE_LABELS)

    kp_err = next(e for e in result.errors if e.code == "keypoints_missing")
    assert kp_err.location == "pose.keypoints"
    assert "neck" in kp_err.message
    assert all(name in (kp_err.hint or "") for name in KEYPOINT_NAMES)


def test_markdown_summary_renders_sample_failures(tmp_path: Path) -> None:
    from maatml.evaluation.runner import Report, write_markdown_summary

    report = Report(
        model_id="vision",
        task="vision_multitask",
        dataset="d",
        n=1,
        metrics={"all_layers_pass_rate": 0.0},
        sample_failures=[
            {
                "sample_id": "s1",
                "errors": [
                    {
                        "layer": 4,
                        "code": "scene_label",
                        "message": "unknown scene label 'nebula'",
                        "location": "scene.label",
                        "hint": "use one of: 'plain', 'gradient'",
                    }
                ],
            }
        ],
    )
    body = write_markdown_summary(report, tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Sample failures" in body
    assert "`s1`" in body
    assert "L4/scene_label at `scene.label`" in body
    assert "fix: use one of: 'plain', 'gradient'" in body


def test_metrics_perfect_match(plugin) -> None:
    from vision_plugin.metrics import compute_vision_metrics

    from maatml.validation.base import ValidationResult

    seed = ROOT / "datasets" / "samples" / "seed_samples.jsonl"
    row = json.loads(seed.read_text(encoding="utf-8").splitlines()[0])
    expected = row["expected"]

    class _Item:
        def __init__(self):
            self.row = row
            self.gen_text = json.dumps(expected)
            self.result = ValidationResult(
                raw_output=self.gen_text, n_layers=4, required_layers={1, 2, 3, 4}
            )
            self.result.passed_layers = {1, 2, 3, 4}

    metrics = compute_vision_metrics([_Item()])
    assert metrics["scene_accuracy"] == 1.0
    assert metrics["pck_0_2"] == 1.0
    assert metrics["map_50"] == 1.0


def test_decode_scene_argmax(plugin) -> None:
    from vision_plugin.constants import SCENE_LABELS
    from vision_plugin.decode import decode_scene

    out = decode_scene([0.1, 5.0, 0.2, 0.0, -1.0], SCENE_LABELS)
    assert out["label"] == SCENE_LABELS[1]
    assert out["confidence"] > 0.5


def test_synth_deterministic(plugin) -> None:
    pytest.importorskip("PIL")
    from vision_plugin.synth import make_scene_spec, render_scene

    a = make_scene_spec(3, base_seed=42, size=64)
    b = make_scene_spec(3, base_seed=42, size=64)
    assert a == b
    img1, exp1 = render_scene(a, size=64)
    img2, exp2 = render_scene(b, size=64)
    assert exp1 == exp2
    assert list(img1.getdata()) == list(img2.getdata())


def test_plugin_registers_trainer_and_onnx(plugin) -> None:
    from maatml.registry import EXPORTERS, GENERATORS, PREDICTORS, TRAINERS

    assert "vision_multitask" in TRAINERS.names()
    assert "vision_multitask" in PREDICTORS.names()
    assert "synthetic_scenes" in GENERATORS.names()
    assert "onnx" in EXPORTERS.names()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required",
)
def test_tiny_train_predict_onnx_roundtrip(tmp_path: Path, plugin) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("safetensors")

    import torch
    from torch.utils.data import DataLoader
    from vision_plugin.dataset import VisionSceneDataset, collate_vision
    from vision_plugin.export_onnx import export_onnx
    from vision_plugin.model import MultitaskConfig, MultitaskNet, save_checkpoint
    from vision_plugin.predictor import VisionMultitaskPredictor
    from vision_plugin.synth import build_sample_row

    from maatml.config import ModelDefinition

    # Tiny synthetic corpus
    images = tmp_path / "images"
    images.mkdir()
    rows = [
        build_sample_row(
            i,
            base_seed=0,
            size=64,
            image_rel="images/{id}.png",
            images_dir=images,
        )
        for i in range(8)
    ]
    for r in rows:
        r["image"] = str(Path("images") / Path(r["image"]).name)

    cfg = MultitaskConfig(
        image_size=64,
        backbone="mobilenet_v3_small",
        pretrained=False,
        output_stride=16,
    )
    model = MultitaskNet.build(cfg)
    ds = VisionSceneDataset.build(rows, model_dir=tmp_path, cfg=cfg)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_vision)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step, batch in enumerate(loader):
        if step >= 2:
            break
        out = model(batch["image"])
        losses = model.compute_loss(
            out,
            {
                "scene_idx": batch["scene_idx"],
                "heatmaps": batch["heatmaps"],
                "sizes": batch["sizes"],
                "offsets": batch["offsets"],
                "center_mask": batch["center_mask"],
                "pose_coords": batch["pose_coords"],
            },
        )
        losses["loss"].backward()
        opt.step()
        opt.zero_grad()

    ckpt = tmp_path / "ckpt"
    save_checkpoint(model, cfg, ckpt)

    pred = VisionMultitaskPredictor()

    class _MD:
        model_dir = tmp_path

    pred.setup(ckpt, model_def=_MD(), device="cpu")
    raw = pred.predict(rows[0])
    parsed = json.loads(raw)
    assert "scene" in parsed and "detections" in parsed and "pose" in parsed
    assert len(parsed["pose"]["keypoints"]) == 12

    md = ModelDefinition(
        name="vision",
        model_id="vision",
        version="0.1.0",
        architecture="vision_multitask",
        dataset={},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    export_dir = tmp_path / "export"
    export_onnx(md, ckpt, export_dir)
    assert (export_dir / "model.onnx").is_file()
    assert (export_dir / "deploy" / "client.py").is_file()

    pred2 = VisionMultitaskPredictor()
    pred2.setup(export_dir, model_def=_MD(), device="cpu")
    assert pred2.backend == "onnx"
    raw2 = pred2.predict(rows[0])
    parsed2 = json.loads(raw2)
    assert parsed2["scene"]["label"] in cfg.scene_labels


def test_scaffold_hook_produces_a_valid_folder(tmp_path: Path, plugin) -> None:
    """`maatml scaffold -a vision_multitask --plugin ...` must validate."""
    from maatml.scaffold import scaffold_model, validate_model_dir

    target = tmp_path / "scaffolded-vision"
    scaffold_model(target, architecture="vision_multitask", plugins=[str(ROOT / "vision_plugin")])

    assert validate_model_dir(target) == []
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "backbone: mobilenet_v3_large" in body
    assert "generator: synthetic_scenes" in body
    assert "CHANGE_ME" not in body
    # The scaffolded schema is the one the validator and generator were written
    # against, not a lookalike: a copy that drifts makes datagen reject every
    # row it generates.
    assert json.loads((target / "datasets" / "schema.json").read_text()) == json.loads(
        (ROOT / "datasets" / "schema.json").read_text()
    )


def test_scaffold_fallback_schema_matches_the_canonical_one(plugin) -> None:
    from vision_plugin.scaffold import _FALLBACK_SCHEMA

    canonical = json.loads((ROOT / "datasets" / "schema.json").read_text())
    assert _FALLBACK_SCHEMA["required"] == canonical["required"]
    assert set(_FALLBACK_SCHEMA["properties"]) == set(canonical["properties"])
