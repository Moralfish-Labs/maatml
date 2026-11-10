"""Multimodal export requires a newly produced, manifested companion."""

import json
from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.export.gguf import export_gguf
from maatml.export.manifest import verify_manifest


def setup_export(tmp_path: Path, monkeypatch, *, mode: str = "success"):
    monkeypatch.delenv("MAATML_LLAMA_CONVERT", raising=False)
    monkeypatch.delenv("MAATML_LLAMA_QUANTIZE", raising=False)
    script = tmp_path / "convert.py"
    script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        f"mode = {mode!r}\n"
        "if '--mmproj' in sys.argv:\n"
        "    assert sys.argv[sys.argv.index('--outtype') + 1] == 'f16'\n"
        "    if mode == 'fail':\n"
        "        sys.exit('unsupported modality')\n"
        "    if mode != 'missing':\n"
        "        out.write_bytes(b'' if mode == 'empty' else b'GGUF companion')\n"
        "else:\n"
        "    out.write_bytes(b'GGUF language')\n"
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test weights")
    for name in ("processor_config.json", "video_preprocessor_config.json"):
        (checkpoint / name).write_text("{}")
    model = ModelDefinition(
        name="multimodal",
        model_id="multimodal",
        version="0.1.0",
        architecture="causal_sft",
        extensions={"gguf": {"convert_script": "convert.py", "mmproj": True}},
    )
    object.__setattr__(model, "model_dir", tmp_path)
    return model, checkpoint, tmp_path / "export"


def test_companion_and_processor_assets_are_verified(tmp_path, monkeypatch):
    model, checkpoint, out = setup_export(tmp_path, monkeypatch)
    export_gguf(model, checkpoint, out)
    manifest = json.loads((out / "manifest.json").read_text())
    paths = {item["path"] for item in manifest["files"]}
    assert {
        "mmproj-multimodal.gguf",
        "multimodal.gguf",
        "processor_config.json",
        "video_preprocessor_config.json",
    } <= paths
    assert verify_manifest(out) == []
    (out / "mmproj-multimodal.gguf").write_bytes(b"corrupted")
    assert verify_manifest(out)


@pytest.mark.parametrize("mode", ["fail", "missing", "empty"])
def test_failed_conversion_cannot_reuse_stale_companion(tmp_path, monkeypatch, mode):
    model, checkpoint, out = setup_export(tmp_path, monkeypatch, mode=mode)
    out.mkdir()
    (out / "mmproj-multimodal.gguf").write_bytes(b"old companion")
    with pytest.raises(RuntimeError, match="Multimodal GGUF"):
        export_gguf(model, checkpoint, out)


def test_companion_is_opt_in(tmp_path, monkeypatch):
    model, checkpoint, out = setup_export(tmp_path, monkeypatch)
    model.extensions["gguf"]["mmproj"] = False
    export_gguf(model, checkpoint, out)
    assert not (out / "mmproj-multimodal.gguf").exists()


@pytest.mark.parametrize("setting", ["true", 1, None])
def test_invalid_option_rejected_before_export(tmp_path, monkeypatch, setting):
    model, checkpoint, out = setup_export(tmp_path, monkeypatch)
    model.extensions["gguf"]["mmproj"] = setting
    with pytest.raises(ValueError, match="boolean"):
        export_gguf(model, checkpoint, out)
    assert not out.exists()


def test_companion_requires_explicit_converter(tmp_path, monkeypatch):
    model, checkpoint, out = setup_export(tmp_path, monkeypatch)
    model.extensions["gguf"].pop("convert_script")
    with pytest.raises(ValueError, match="explicit convert_script"):
        export_gguf(model, checkpoint, out)
