"""Validator for multitask vision JSON outputs.

Layers:
  1. JSON parse (root must be an object)
  2. JSON Schema (when ``schema_path`` is provided)
  3. Required keys / shapes (scene, detections, pose)
  4. Enum membership (scene labels, shape labels, keypoint names)

Every ``ValidationError`` names what failed, where, and how to fix it so eval
reports, ``serve --enforce`` 422s, and datagen reject rows stay actionable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from maatml.validation.base import ValidationError, ValidationResult
from maatml.validation.base import strip_fences as _strip_fences

from .constants import KEYPOINT_NAMES, SCENE_LABELS, SHAPE_LABELS

_MAX_DETECTION_ERRORS = 10


def _allowed(labels: list[str]) -> str:
    return ", ".join(repr(x) for x in labels)


def validate_vision_scene(
    raw_output: str,
    *,
    schema_path: str | Path | None = None,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    """Layered gate: parse → schema (optional) → field shape → label enums."""
    del user_prompt, contracts_path
    text = _strip_fences(raw_output) if strip_fences else raw_output
    result = ValidationResult(raw_output=raw_output, n_layers=4, required_layers={1, 2, 3, 4})

    # Layer 1: JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(
                layer=1,
                code="invalid_json",
                message=str(exc),
                location=f"line {exc.lineno} col {exc.colno}",
                hint="return a single JSON object with no markdown fences or trailing commas",
            )
        )
        return result
    if not isinstance(parsed, dict):
        result.errors.append(
            ValidationError(
                layer=1,
                code="not_object",
                message=f"root must be an object, got {type(parsed).__name__}",
                location="$",
                hint='wrap the payload as {"scene": ..., "detections": ..., "pose": ...}',
            )
        )
        return result
    result.parsed = parsed
    result.passed_layers.add(1)

    # Layer 2: JSON Schema (when provided)
    if schema_path is not None and Path(schema_path).is_file():
        try:
            import jsonschema

            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            jsonschema.validate(parsed, schema)
            result.passed_layers.add(2)
        except Exception as exc:  # noqa: BLE001
            # Prefer jsonschema's short message + absolute path over the
            # multi-paragraph str(ValidationError) dump.
            try:
                import jsonschema as _js

                is_schema_err = isinstance(exc, _js.ValidationError)
            except ImportError:
                is_schema_err = False
            if is_schema_err:
                location = "/".join(str(p) for p in exc.absolute_path) or None
                result.errors.append(
                    ValidationError(
                        layer=2,
                        code="schema",
                        message=exc.message,
                        location=location,
                        hint="match datasets/schema.json field types and required keys",
                    )
                )
            else:
                result.errors.append(
                    ValidationError(
                        layer=2,
                        code="schema",
                        message=str(exc),
                        hint="fix or remove dataset.schema if the schema file is unreadable",
                    )
                )
    else:
        result.passed_layers.add(2)

    # Layer 3: required keys / shapes
    scene = parsed.get("scene")
    dets = parsed.get("detections")
    pose = parsed.get("pose")
    ok3 = True
    if not isinstance(scene, dict) or "label" not in scene:
        found = type(scene).__name__ if scene is not None else "missing"
        result.errors.append(
            ValidationError(
                layer=3,
                code="scene_shape",
                message=f"scene.label required, got {found}",
                location="scene",
                hint='set scene to {"label": "<scene_label>", "confidence": 0.0..1.0}',
            )
        )
        ok3 = False
    if not isinstance(dets, list):
        found = type(dets).__name__ if dets is not None else "missing"
        result.errors.append(
            ValidationError(
                layer=3,
                code="detections_shape",
                message=f"detections must be a list, got {found}",
                location="detections",
                hint="set detections to a JSON array of {label, box, confidence} objects",
            )
        )
        ok3 = False
    else:
        n_det_errors = 0
        for i, d in enumerate(dets):
            if n_det_errors >= _MAX_DETECTION_ERRORS:
                result.errors.append(
                    ValidationError(
                        layer=3,
                        code="detection_item",
                        message=(
                            f"too many detection errors; stopped after "
                            f"{_MAX_DETECTION_ERRORS} (of {len(dets)} items)"
                        ),
                        location="detections",
                        hint="fix the reported detections first, then re-validate",
                    )
                )
                ok3 = False
                break
            if not isinstance(d, dict):
                result.errors.append(
                    ValidationError(
                        layer=3,
                        code="detection_item",
                        message=f"detections[{i}] must be an object, got {type(d).__name__}",
                        location=f"detections[{i}]",
                        hint='each detection needs {"label": ..., "box": [x1,y1,x2,y2]}',
                    )
                )
                ok3 = False
                n_det_errors += 1
                continue
            missing = [k for k in ("label", "box") if k not in d]
            if missing:
                result.errors.append(
                    ValidationError(
                        layer=3,
                        code="detection_item",
                        message=f"detections[{i}] missing {', '.join(missing)}",
                        location=f"detections[{i}]",
                        hint='each detection needs {"label": ..., "box": [x1,y1,x2,y2]}',
                    )
                )
                ok3 = False
                n_det_errors += 1
                continue
            box = d.get("box")
            if not (isinstance(box, list) and len(box) == 4):
                shape = (
                    f"list[{len(box)}]"
                    if isinstance(box, list)
                    else type(box).__name__
                    if box is not None
                    else "missing"
                )
                result.errors.append(
                    ValidationError(
                        layer=3,
                        code="box_shape",
                        message=f"box must be [x1,y1,x2,y2], got {shape}",
                        location=f"detections[{i}].box",
                        hint="use four normalized floats in [0,1]: [x1, y1, x2, y2]",
                    )
                )
                ok3 = False
                n_det_errors += 1
    if not isinstance(pose, dict) or not isinstance(pose.get("keypoints"), list):
        if pose is None:
            found = "missing"
        elif not isinstance(pose, dict):
            found = type(pose).__name__
        else:
            kp = pose.get("keypoints")
            found = f"keypoints={type(kp).__name__ if kp is not None else 'missing'}"
        result.errors.append(
            ValidationError(
                layer=3,
                code="pose_shape",
                message=f"pose.keypoints list required, got {found}",
                location="pose.keypoints",
                hint='set pose to {"keypoints": [{"name": ..., "x": ..., "y": ...}, ...]}',
            )
        )
        ok3 = False
    if ok3:
        result.passed_layers.add(3)

    # Layer 4: enum membership
    ok4 = True
    if isinstance(scene, dict):
        label = scene.get("label")
        if label not in SCENE_LABELS:
            result.errors.append(
                ValidationError(
                    layer=4,
                    code="scene_label",
                    message=f"unknown scene label {label!r}",
                    location="scene.label",
                    hint=f"use one of: {_allowed(SCENE_LABELS)}",
                )
            )
            ok4 = False
    if isinstance(dets, list):
        for i, d in enumerate(dets):
            if not isinstance(d, dict):
                continue
            shape = d.get("label")
            if shape not in SHAPE_LABELS:
                result.errors.append(
                    ValidationError(
                        layer=4,
                        code="shape_label",
                        message=f"unknown shape {shape!r}",
                        location=f"detections[{i}].label",
                        hint=f"use one of: {_allowed(SHAPE_LABELS)}",
                    )
                )
                ok4 = False
    if isinstance(pose, dict) and isinstance(pose.get("keypoints"), list):
        names = {k.get("name") for k in pose["keypoints"] if isinstance(k, dict)}
        missing = [n for n in KEYPOINT_NAMES if n not in names]
        if missing:
            result.errors.append(
                ValidationError(
                    layer=4,
                    code="keypoints_missing",
                    message=f"missing keypoints: {missing}",
                    location="pose.keypoints",
                    hint=f"include all {len(KEYPOINT_NAMES)} names: {_allowed(KEYPOINT_NAMES)}",
                )
            )
            ok4 = False
    if ok4:
        result.passed_layers.add(4)
    return result
