"""Export manifest build / verify checksum roundtrip."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from maatml.config import ModelDefinition, PackagingSpec
from maatml.export.manifest import (
    build_manifest,
    read_safetensors_dtypes,
    read_safetensors_tensor_names,
    verify_manifest,
    write_manifest,
)
from maatml.utils.io import sha256_file, write_json


def _write_safetensors(path: Path, entries: list[tuple[str, str, int]]) -> None:
    """Write a minimal valid .safetensors file (header + zeroed data)."""
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, nbytes in entries:
        header[name] = {
            "dtype": dtype,
            "shape": [nbytes],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\x00" * offset)


def _toy_model_def(tmp_path: Path, weights_dtype: str = "f16") -> ModelDefinition:
    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
        packaging=PackagingSpec(
            max_input_tokens=512, expected_latency_ms=100, weights_dtype=weights_dtype
        ),
    )
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def test_manifest_verify_roundtrip(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    a = export_dir / "weights.bin"
    b = export_dir / "tokenizer.json"
    a.write_bytes(b"fake-weights")
    b.write_text('{"version":1}', encoding="utf-8")

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
        packaging=PackagingSpec(max_input_tokens=512, expected_latency_ms=100),
    )
    object.__setattr__(md, "model_dir", tmp_path)

    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[a, b],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
        run_id="run-1",
    )
    write_manifest(export_dir, manifest)

    assert (export_dir / "manifest.json").is_file()
    assert manifest["identity"] == "toy@0.1.0"
    assert manifest["run_id"] == "run-1"
    assert len(manifest["files"]) == 2
    assert manifest["files"][0]["sha256"] == sha256_file(a)

    assert verify_manifest(export_dir) == []
    assert verify_manifest(export_dir / "manifest.json") == []


def test_verify_detects_tamper(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    f = export_dir / "model.safetensors"
    f.write_bytes(b"v1")
    write_json(
        export_dir / "manifest.json",
        {
            "files": [{"path": "model.safetensors", "sha256": sha256_file(f)}],
        },
    )
    assert verify_manifest(export_dir) == []
    f.write_bytes(b"v2-tampered")
    errors = verify_manifest(export_dir)
    assert len(errors) == 1
    assert "checksum mismatch" in errors[0]


def test_read_safetensors_dtypes(tmp_path: Path) -> None:
    st = tmp_path / "model.safetensors"
    _write_safetensors(st, [("w", "F16", 4), ("b", "F16", 2)])
    assert read_safetensors_dtypes(st) == ["F16", "F16"]
    # Non-safetensors / garbage parses to nothing rather than raising.
    junk = tmp_path / "notes.txt"
    junk.write_text("hello", encoding="utf-8")
    assert read_safetensors_dtypes(junk) == []


def test_read_safetensors_tensor_names(tmp_path: Path) -> None:
    st = tmp_path / "model.safetensors"
    _write_safetensors(st, [("shared.weight", "F32", 4), ("lm_head.weight", "F32", 4)])
    assert read_safetensors_tensor_names(st) == ["shared.weight", "lm_head.weight"]
    junk = tmp_path / "notes.txt"
    junk.write_text("hello", encoding="utf-8")
    assert read_safetensors_tensor_names(junk) == []


def test_weights_dtype_verified_from_tensors(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    st = export_dir / "model.safetensors"
    _write_safetensors(st, [("w", "F16", 8), ("b", "F16", 4)])

    # Declared hint disagrees with the tensors on purpose (declared bf16).
    md = _toy_model_def(tmp_path, weights_dtype="bf16")
    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[st],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
    )
    hints = manifest["runtime_hints"]
    assert hints["weights_dtype"] == "f16"  # verified from tensors, not the hint
    assert hints["weights_dtype_declared"] == "bf16"
    assert hints["weights_dtype_verified"] is True
    assert "weights_dtypes_observed" not in hints  # uniform dtype


def test_weights_dtype_unverified_without_safetensors(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    blob = export_dir / "model.gguf"
    blob.write_bytes(b"not-safetensors")

    md = _toy_model_def(tmp_path, weights_dtype="f16")
    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[blob],
        formats=["gguf"],
        source_checkpoint=tmp_path / "ckpt",
    )
    hints = manifest["runtime_hints"]
    assert hints["weights_dtype"] == "f16"  # falls back to the declared hint
    assert hints["weights_dtype_declared"] == "f16"
    assert hints["weights_dtype_verified"] is False


def test_dummy_safetensors_does_not_crash(tmp_path: Path) -> None:
    # A .safetensors file full of garbage decodes to an absurd header length;
    # the reader must reject it, not attempt a multi-GB read (regression).
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    st = export_dir / "model.safetensors"
    st.write_bytes(b"not-real-weights")
    assert read_safetensors_dtypes(st) == []

    md = _toy_model_def(tmp_path, weights_dtype="f16")
    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[st],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
    )
    hints = manifest["runtime_hints"]
    assert hints["weights_dtype"] == "f16"
    assert hints["weights_dtype_verified"] is False


def test_weights_dtype_mixed_precision_surfaced(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    st = export_dir / "model.safetensors"
    # Two F32 tensors, one F16 → dominant is f32, both surfaced.
    _write_safetensors(st, [("a", "F32", 8), ("b", "F32", 8), ("c", "F16", 2)])

    md = _toy_model_def(tmp_path, weights_dtype="f16")
    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[st],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
    )
    hints = manifest["runtime_hints"]
    assert hints["weights_dtype"] == "f32"  # dominant by tensor count
    assert hints["weights_dtype_verified"] is True
    assert hints["weights_dtypes_observed"] == ["f16", "f32"]


def test_amend_adds_files_and_verify_covers_them(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    st = export_dir / "model.safetensors"
    _write_safetensors(st, [("a", "F16", 4)])
    md = _toy_model_def(tmp_path)
    write_manifest(
        export_dir,
        build_manifest(
            model_def=md,
            export_dir=export_dir,
            files=[st],
            formats=["safetensors"],
            source_checkpoint=tmp_path / "ckpt",
        ),
    )

    quantized = export_dir / "toy-Q4_K_M.gguf"
    quantized.write_bytes(b"gguf-bytes")
    from maatml.export.manifest import amend_manifest

    manifest = amend_manifest(export_dir, [quantized], formats=["gguf"])
    paths = {e["path"] for e in manifest["files"]}
    assert "toy-Q4_K_M.gguf" in paths
    assert manifest["runtime_hints"]["formats"] == ["safetensors", "gguf"]
    assert verify_manifest(export_dir) == []

    # Tampering with the amended file is now caught.
    quantized.write_bytes(b"different")
    assert verify_manifest(export_dir) != []


def test_amend_replaces_an_existing_entry_and_keeps_evidence(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    st = export_dir / "model.safetensors"
    _write_safetensors(st, [("a", "F16", 4)])
    md = _toy_model_def(tmp_path)
    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[st],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
    )
    manifest["gate_evidence"] = {"gated": True, "passed": True}
    write_manifest(export_dir, manifest)

    _write_safetensors(st, [("a", "F16", 8)])
    from maatml.export.manifest import amend_manifest

    amended = amend_manifest(export_dir, [st])
    assert len(amended["files"]) == 1
    assert amended["files"][0]["sha256"] == sha256_file(st)
    assert amended["gate_evidence"] == {"gated": True, "passed": True}
    assert verify_manifest(export_dir) == []


def test_amend_refuses_a_file_outside_the_export(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    st = export_dir / "model.safetensors"
    _write_safetensors(st, [("a", "F16", 4)])
    md = _toy_model_def(tmp_path)
    write_manifest(
        export_dir,
        build_manifest(
            model_def=md,
            export_dir=export_dir,
            files=[st],
            formats=["safetensors"],
            source_checkpoint=tmp_path / "ckpt",
        ),
    )
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"x")
    import pytest

    from maatml.export.manifest import amend_manifest

    with pytest.raises(ValueError, match="outside the export directory"):
        amend_manifest(export_dir, [outside])


def test_verify_detects_unlisted_files(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    listed = export_dir / "model.safetensors"
    listed.write_bytes(b"weights")
    write_json(
        export_dir / "manifest.json",
        {"files": [{"path": "model.safetensors", "sha256": sha256_file(listed)}]},
    )
    assert verify_manifest(export_dir) == []

    (export_dir / "rogue.txt").write_text("surprise", encoding="utf-8")
    nested = export_dir / "extra" / "weights.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"hidden")

    errors = verify_manifest(export_dir)
    assert "unlisted file: rogue.txt" in errors
    assert "unlisted file: extra/weights.bin" in errors
