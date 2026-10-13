"""Format-neutral compiler and server registries (TensorRT / vLLM / GGUF shaped)."""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from maatml.compile import compile_export, parse_options, promotion_status
from maatml.registry import (
    COMPILERS,
    SERVERS,
    discover_plugins,
    register_compiler,
    register_server,
)
from maatml.servers import LifecycleServer, dispatch_server
from maatml.utils.io import write_json


def _write_export(tmp_path: Path, *, name: str = "toy") -> Path:
    export = tmp_path / "export"
    export.mkdir()
    artifact = export / "model.onnx"
    artifact.write_bytes(b"onnx-bytes")
    write_json(
        export / "manifest.json",
        {
            "name": name,
            "version": "1",
            "identity": f"{name}@1",
            "architecture": "sip_person_detector",
            "run_id": "run-1",
            "gate_evidence": {"passed": True},
            "files": [{"path": "model.onnx", "sha256": "abc"}],
        },
    )
    return export


def test_parse_options_roundtrip() -> None:
    assert parse_options(["precision=fp16", "profile=4cam", "empty="]) == {
        "precision": "fp16",
        "profile": "4cam",
        "empty": "",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_options(["nope"])
    with pytest.raises(ValueError, match="empty key"):
        parse_options(["=value"])


def test_discover_registers_http_server() -> None:
    discover_plugins(force=True)
    assert SERVERS.get("http") is not None


def test_compiler_tensorrt_shaped(tmp_path: Path) -> None:
    @register_compiler("fake_tensorrt")
    def _trt(export_dir: Path, out_dir: Path, *, manifest, options):
        assert manifest["identity"] == "toy@1"
        plan = out_dir / "engine.plan"
        plan.write_bytes(b"trt-plan")
        write_json(
            out_dir / "nvinfer.txt",
            {"precision": options.get("precision", "fp16"), "source": str(export_dir)},
        )
        return out_dir

    export = _write_export(tmp_path)
    out = tmp_path / "deploy"
    result = compile_export(
        export, target="fake_tensorrt", out_dir=out, options={"precision": "fp16"}
    )
    assert result == out
    assert (out / "engine.plan").is_file()
    target = json.loads((out / "target_manifest.json").read_text(encoding="utf-8"))
    assert target["kind"] == "maatml.target/1"
    assert target["compiler"] == "fake_tensorrt"
    assert target["source"]["identity"] == "toy@1"
    assert target["options"]["precision"] == "fp16"
    assert target["promotion_eligible"] is True
    assert target["promotion_reason"] == "production gate passed"
    assert "fake_tensorrt" in COMPILERS.names()


def test_compiler_gguf_shaped(tmp_path: Path) -> None:
    @register_compiler("fake_gguf_quantize")
    def _gguf(export_dir: Path, out_dir: Path, *, manifest, options):
        del export_dir, manifest
        q = out_dir / "model.Q4_K_M.gguf"
        q.write_bytes(b"gguf")
        write_json(out_dir / "quantize.json", {"method": options.get("method", "Q4_K_M")})
        return out_dir

    export = _write_export(tmp_path)
    # Swap primary artifact name for the source identity only.
    out = tmp_path / "gguf-out"
    compile_export(
        export,
        target="fake_gguf_quantize",
        out_dir=out,
        options={"method": "Q5_K_M"},
    )
    assert (out / "model.Q4_K_M.gguf").is_file()
    target = json.loads((out / "target_manifest.json").read_text(encoding="utf-8"))
    assert target["compiler"] == "fake_gguf_quantize"
    assert target["options"]["method"] == "Q5_K_M"


def test_compiler_vllm_shaped(tmp_path: Path) -> None:
    @register_compiler("fake_vllm_package")
    def _vllm(export_dir: Path, out_dir: Path, *, manifest, options):
        del export_dir, manifest
        write_json(
            out_dir / "vllm_bundle.json",
            {
                "runtime": "vllm",
                "dtype": options.get("dtype", "auto"),
                "max_model_len": int(options.get("max_model_len", "4096")),
            },
        )
        return out_dir

    export = _write_export(tmp_path)
    out = tmp_path / "vllm-out"
    compile_export(
        export,
        target="fake_vllm_package",
        out_dir=out,
        options={"dtype": "half", "max_model_len": "2048"},
    )
    bundle = json.loads((out / "vllm_bundle.json").read_text(encoding="utf-8"))
    assert bundle == {"runtime": "vllm", "dtype": "half", "max_model_len": 2048}


def test_unknown_compiler_raises(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    with pytest.raises(KeyError):
        compile_export(export, target="missing_compiler", out_dir=tmp_path / "x")


def test_promotion_status_refuses_smoke_and_missing() -> None:
    assert promotion_status({}) == (False, "manifest has no gate_evidence")
    assert promotion_status({"gate_evidence": {}})[0] is False
    ok, reason = promotion_status({"gate_evidence": {"passed": True, "smoke_gated": True}})
    assert ok is False
    assert "smoke-gated" in reason
    ok, reason = promotion_status({"gate_evidence": {"passed": False}})
    assert ok is False
    assert "did not pass" in reason
    assert promotion_status({"gate_evidence": {"passed": True}}) == (
        True,
        "production gate passed",
    )


def test_require_gated_refuses_before_the_plugin_runs(tmp_path: Path) -> None:
    called = {"n": 0}

    @register_compiler("gated_only")
    def _cmp(export_dir: Path, out_dir: Path, *, manifest, options):
        del export_dir, manifest, options
        called["n"] += 1
        return out_dir

    export = _write_export(tmp_path)
    manifest_path = export / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["gate_evidence"] = {"passed": True, "smoke_gated": True}
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="require-gated"):
        compile_export(export, target="gated_only", out_dir=tmp_path / "nope", require_gated=True)
    assert called["n"] == 0


def test_require_gated_allows_a_production_pass(tmp_path: Path) -> None:
    @register_compiler("gated_ok")
    def _cmp(export_dir: Path, out_dir: Path, *, manifest, options):
        del export_dir, manifest, options
        (out_dir / "ok").write_text("1", encoding="utf-8")
        return out_dir

    export = _write_export(tmp_path)
    out = tmp_path / "yes"
    compile_export(export, target="gated_ok", out_dir=out, require_gated=True)
    target = json.loads((out / "target_manifest.json").read_text(encoding="utf-8"))
    assert target["promotion_eligible"] is True


def test_server_dispatch_fake_backends() -> None:
    seen: dict[str, object] = {}

    @register_server("fake_deepstream")
    def _ds(model_def, *, checkpoint=None, options=None, **kwargs):
        seen["deepstream"] = {
            "options": options,
            "socket": (options or {}).get("socket"),
            "model": getattr(model_def, "identity", model_def),
        }
        return "ds-ok"

    @register_server("fake_vllm")
    def _vllm(model_def, *, checkpoint=None, options=None, **kwargs):
        seen["vllm"] = options
        return "vllm-ok"

    @register_server("fake_gguf")
    def _gguf(model_def, *, checkpoint=None, options=None, **kwargs):
        seen["gguf"] = options
        return "gguf-ok"

    class _MD:
        identity = "toy@1"

    assert (
        dispatch_server("fake_deepstream", _MD(), options={"socket": "/run/sip/perception.sock"})
        == "ds-ok"
    )
    assert seen["deepstream"]["socket"] == "/run/sip/perception.sock"
    assert dispatch_server("fake_vllm", _MD(), options={"tp": "1"}) == "vllm-ok"
    assert seen["vllm"] == {"tp": "1"}
    assert dispatch_server("fake_gguf", _MD(), options={"threads": "4"}) == "gguf-ok"
    assert seen["gguf"] == {"threads": "4"}
    assert "http" in SERVERS.names()


def test_lifecycle_server_verify_warmup_close() -> None:
    order: list[str] = []

    handle = LifecycleServer(
        name="toy",
        verify_fn=lambda: order.append("verify"),
        warmup_fn=lambda: order.append("warmup"),
        serve_fn=lambda: order.append("serve"),
        close_fn=lambda: order.append("close"),
        capabilities_fn=lambda: {
            "modality": "vision",
            "task": "detect",
            "runtime": "fake",
            "execution_class": "continuous",
        },
    )
    caps = handle.run()
    assert order == ["verify", "warmup", "serve", "close"]
    assert caps["modality"] == "vision"
    assert caps["execution_class"] == "continuous"


def test_lifecycle_record_capture_uses_attached_writer() -> None:
    seen: list[tuple[object, object, str]] = []

    class _Writer:
        def record(self, row, output, raw):
            seen.append((row, output, raw))
            return True

    handle = LifecycleServer(
        name="toy",
        verify_fn=lambda: None,
        warmup_fn=lambda: None,
        serve_fn=lambda: None,
        close_fn=lambda: None,
        capture=_Writer(),
    )
    assert handle.record_capture({"image": "a.png"}, {"ok": True}, '{"ok":true}')
    assert seen[0][0]["image"] == "a.png"
    bare = LifecycleServer(
        name="bare",
        verify_fn=lambda: None,
        warmup_fn=lambda: None,
        serve_fn=lambda: None,
        close_fn=lambda: None,
    )
    assert bare.record_capture({}, {}, "") is False


def test_lifecycle_sigterm_closes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    handlers = {}
    order: list[str] = []

    monkeypatch.setattr(signal, "getsignal", lambda signum: f"previous-{signum}")
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    def serve() -> None:
        order.append("serve")
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        order.append("drained")

    handle = LifecycleServer(
        name="toy",
        verify_fn=lambda: None,
        warmup_fn=lambda: None,
        serve_fn=serve,
        close_fn=lambda: order.append("close"),
    )
    handle.serve_forever()
    assert order == ["serve", "close", "drained"]
