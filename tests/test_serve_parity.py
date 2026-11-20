"""Core served-model parity predictor, artifact binding, and parity comparison."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from maatml.evaluation.served import (
    ServedModelPredictor,
    compare_parity,
    probe_server_identity,
    verify_served_artifact,
)
from maatml.registry import PREDICTORS, discover_plugins


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _install_fake_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[tuple[str, dict[str, Any] | None, dict[str, str]]]:
    calls: list[tuple[str, dict[str, Any] | None, dict[str, str]]] = []

    def _fake_urlopen(req: Any, timeout: float = 10.0) -> _FakeResponse:
        del timeout
        url = req.full_url
        raw = req.data
        payload = json.loads(raw.decode("utf-8")) if raw else None
        headers = {k.lower(): v for k, v in req.header_items()}
        calls.append((url, payload, headers))
        body = handler(url, payload, headers)
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    return calls


def test_served_predictors_are_registered_by_builtin_discovery() -> None:
    discover_plugins(force=True)
    assert PREDICTORS.require("served_http").__name__ == "ServedModelPredictor"
    assert PREDICTORS.require("serve_parity").__name__ == "ServedModelPredictor"


def test_verify_served_artifact_checks_hash_gate_evidence_and_run_id(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    gguf_bytes = b"GGUF-mock-weights-v1"
    gguf_path = export_dir / "model-q4.gguf"
    gguf_path.write_bytes(gguf_bytes)

    manifest = {
        "run_id": "run-001",
        "identity": "demo@0.1.0",
        "files": [{"path": "model-q4.gguf", "sha256": _sha256(gguf_bytes)}],
        "gate_evidence": {"gated": True, "passed": True, "smoke_gated": False},
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    receipt = verify_served_artifact(
        export_dir, "model-q4.gguf", checkpoint_dir=tmp_path / "run-001"
    )
    assert receipt["artifact_sha256"] == _sha256(gguf_bytes)
    assert receipt["run_id"] == "run-001"

    # Run id mismatch fails.
    with pytest.raises(ValueError, match="run_id"):
        verify_served_artifact(export_dir, "model-q4.gguf", checkpoint_dir=tmp_path / "run-999")

    # Smoke-only gate evidence fails when production gate is required.
    manifest["gate_evidence"]["smoke_gated"] = True
    (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="production gate evidence"):
        verify_served_artifact(export_dir, "model-q4.gguf")
    # Allowed when require_production_gate=False.
    assert (
        verify_served_artifact(export_dir, "model-q4.gguf", require_production_gate=False)[
            "artifact"
        ]
        == "model-q4.gguf"
    )

    # Tampered weights fail hash verification.
    gguf_path.write_bytes(b"GGUF-tampered")
    with pytest.raises(ValueError, match="does not match export manifest"):
        verify_served_artifact(export_dir, "model-q4.gguf", require_production_gate=False)


def test_probe_server_identity_verifies_props_and_lists_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf")

    def handler(url: str, payload: Any, headers: Any) -> dict[str, Any]:
        del payload, headers
        if url.endswith("/props"):
            return {"build_info": "b4200", "model_path": str(gguf)}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "served-m"}]}
        raise AssertionError(url)

    _install_fake_http(monkeypatch, handler)
    info = probe_server_identity(
        "http://127.0.0.1:8080",
        expected_build_info="b4200",
        expected_model_path=gguf,
    )
    assert info["build_info"] == "b4200"
    assert info["served_models"] == ["served-m"]

    with pytest.raises(ValueError, match="build_info differs"):
        probe_server_identity("http://127.0.0.1:8080", expected_build_info="b9999")

    other = tmp_path / "other.gguf"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="server serves"):
        probe_server_identity("http://127.0.0.1:8080", expected_model_path=other)


def test_served_model_predictor_handles_prompts_tools_batches_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt = tmp_path / "run-001"
    ckpt.mkdir()
    (ckpt / "prompt_spec.json").write_text(
        json.dumps({"system": "Be concise.", "user_template": "Q: <<USER_REQUEST>>"}),
        encoding="utf-8",
    )

    def handler(url: str, payload: Any, headers: Any) -> dict[str, Any]:
        if url.endswith("/props"):
            return {"build_info": "b4200"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "demo-gguf"}]}
        if url.endswith("/v1/chat/completions"):
            assert headers.get("authorization") == "Bearer tok-1"
            if payload.get("tools"):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"id": 7}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": '{"answer": "ok"}'}}]}
        raise AssertionError(url)

    calls = _install_fake_http(monkeypatch, handler)
    pred = ServedModelPredictor()
    pred.setup(ckpt, base_url="http://127.0.0.1:8080", api_key="tok-1", seed=42)

    plain = pred.predict({"request": "status?"})
    assert plain == '{"answer": "ok"}'

    tool_row = {
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "find 7"},
            {"role": "assistant", "content": "gold reply that must be stripped"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    structured = pred.predict_structured(tool_row)
    assert structured["errors"] == []
    assert structured["tool_calls"][0]["function"]["name"] == "lookup"
    assert structured["tool_calls"][0]["function"]["arguments"] == {"id": 7}

    # Verify the trailing assistant message was stripped before calling the server.
    chat_calls = [p for u, p, _ in calls if u.endswith("/v1/chat/completions")]
    assert len(chat_calls) == 2
    assert all(m["role"] != "assistant" for m in chat_calls[1]["messages"])
    assert chat_calls[0]["temperature"] == 0.0
    assert chat_calls[0]["seed"] == 42

    batch = pred.predict_batch([{"request": "a"}, {"request": "b"}])
    assert batch == ['{"answer": "ok"}', '{"answer": "ok"}']

    extras = pred.report_extras()["serve_parity"]
    assert extras["model"] == "demo-gguf"
    assert extras["errors"] == 0
    assert len(extras["wall_ms"]) == 4


def test_compare_parity_scores_exact_json_equivalence_and_validator_rates() -> None:
    ref = ['{"a":1,"b":2}', '{"a": 3}', "hello"]
    srv = ['{"b": 2, "a": 1}', '{"a": 4}', "hello"]

    def validator(text: str) -> bool:
        return text.startswith("{")

    report = compare_parity(ref, srv, validator=validator)
    assert report.total == 3
    assert report.exact_matches == 1
    assert report.equivalent_matches == 2
    assert report.exact_match_rate == pytest.approx(1 / 3)
    assert report.equivalent_match_rate == pytest.approx(2 / 3)
    assert report.reference_valid_rate == pytest.approx(2 / 3)
    assert report.served_valid_rate == pytest.approx(2 / 3)
    assert len(report.mismatches) == 1
    assert report.mismatches[0].index == 1
    assert report.to_dict()["total"] == 3
