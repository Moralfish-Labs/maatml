"""Tests for ``maatml serve``: fake predictor, no torch required."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from maatml.config import ModelDefinition, PackagingSpec
from maatml.registry import PREDICTORS, VALIDATORS, discover_plugins
from maatml.serve import build_serve_context, serve_model
from maatml.validation.base import ValidationError, ValidationResult


class _FakePredictor:
    def __init__(self) -> None:
        self.setup_called = False
        self.last_row: dict | None = None

    def setup(self, checkpoint_dir, **kwargs) -> None:  # noqa: ANN001
        del kwargs
        self.setup_called = True
        self.checkpoint_dir = Path(checkpoint_dir)

    def predict(self, row: dict) -> str:
        self.last_row = row
        text = row.get("request") or row.get("image") or ""
        return json.dumps({"echo": text, "ok": True})


def _fake_validator(raw_output: str, **kwargs) -> ValidationResult:  # noqa: ANN003
    del kwargs
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        parsed = json.loads(raw_output)
        result.parsed = parsed
        if parsed.get("ok") is True:
            result.passed_layers.add(1)
        else:
            result.errors.append(ValidationError(layer=1, code="not_ok", message="ok flag missing"))
    except json.JSONDecodeError as exc:
        result.errors.append(ValidationError(layer=1, code="invalid_json", message=str(exc)))
    return result


@pytest.fixture()
def serve_model_dir(tmp_path: Path) -> tuple[ModelDefinition, Path]:
    discover_plugins(force=True)
    PREDICTORS.register("fake_echo", _FakePredictor, source="test")
    VALIDATORS.register("fake_echo", _fake_validator, source="test")

    model_dir = tmp_path / "toy"
    model_dir.mkdir()
    ckpt = model_dir / "output" / "checkpoints" / "smoke"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"x")

    md = ModelDefinition(
        name="toy-serve",
        model_id="toy-serve",
        version="0.1.0",
        architecture="causal_sft",
        evaluation={"predictor": "fake_echo", "validator": "fake_echo"},
        packaging=PackagingSpec(expected_latency_ms=100),
    )
    object.__setattr__(md, "model_dir", model_dir)
    return md, ckpt


def _http_json(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def test_build_serve_context_loads_predictor(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    assert ctx.predictor.setup_called
    assert ctx.checkpoint_dir == ckpt.resolve()
    assert ctx.validator is not None


def test_serve_health_info_predict_validate(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    server = serve_model(md, host="127.0.0.1", port=0, context=ctx)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{port}"
    try:
        status, health = _http_json("GET", f"{base}/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["identity"] == "toy-serve@0.1.0"
        assert health["architecture"] == "causal_sft"

        status, info = _http_json("GET", f"{base}/info")
        assert status == 200
        assert info["packaging"]["expected_latency_ms"] == 100
        assert info["predictor"] == "fake_echo"

        status, pred = _http_json("POST", f"{base}/predict", {"request": "hello serve"})
        assert status == 200
        assert pred["output"]["echo"] == "hello serve"
        assert pred["output"]["ok"] is True
        assert "latency_ms" in pred
        assert "valid" not in pred

        status, pred_v = _http_json(
            "POST",
            f"{base}/predict?validate=1",
            {"request": "hello serve"},
        )
        assert status == 200
        assert pred_v["valid"] is True
        assert pred_v["errors"] == []

        status, missing = _http_json("GET", f"{base}/nope")
        assert status == 404
        assert "error" in missing
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_predict_bad_body(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    server = serve_model(md, host="127.0.0.1", port=0, context=ctx)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{port}"
    try:
        req = urllib.request.Request(
            f"{base}/predict",
            data=b"not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _serve(md: ModelDefinition, ctx) -> tuple[object, str, threading.Thread]:
    server = serve_model(md, host="127.0.0.1", port=0, context=ctx)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://{host}:{port}", thread


def test_cors_off_by_default(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    assert ctx.cors_origin is None
    server, base, thread = _serve(md, ctx)
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cors_enabled_when_configured(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", cors_origin="*")
    server, base, thread = _serve(md, ctx)
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_body_size_cap_returns_413(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", max_body_bytes=64)
    server, base, thread = _serve(md, ctx)
    try:
        big = json.dumps({"request": "x" * 500}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/predict",
            data=big,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# --- G2 / G5 / S1 -----------------------------------------------------------


class _FailPredictor(_FakePredictor):
    def predict(self, row: dict) -> str:
        self.last_row = row
        return json.dumps({"echo": row.get("request", ""), "ok": False})


class _BoomPredictor(_FakePredictor):
    def predict(self, row: dict) -> str:
        raise RuntimeError("kaboom")


def _md_with_predictor(tmp_path, pred_cls, *, pred_name, with_validator=True):
    discover_plugins(force=True)
    PREDICTORS.register(pred_name, pred_cls, source="test")
    if with_validator:
        VALIDATORS.register("fake_echo", _fake_validator, source="test")
    model_dir = tmp_path / "toy"
    model_dir.mkdir()
    ckpt = model_dir / "output" / "checkpoints" / "smoke"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"x")
    evaluation = {"predictor": pred_name}
    if with_validator:
        evaluation["validator"] = "fake_echo"
    md = ModelDefinition(
        name="toy-serve",
        model_id="toy-serve",
        version="0.1.0",
        architecture="causal_sft",
        evaluation=evaluation,
    )
    object.__setattr__(md, "model_dir", model_dir)
    return md, ckpt


def test_serve_enforce_blocks_invalid(tmp_path: Path) -> None:
    md, ckpt = _md_with_predictor(tmp_path, _FailPredictor, pred_name="fail_echo")
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", enforce=True)
    server, base, thread = _serve(md, ctx)
    try:
        status, body = _http_json("POST", f"{base}/predict", {"request": "x"})
        assert status == 422
        assert body["valid"] is False
        assert body["errors"]
        assert "output" in body and "latency_ms" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_enforce_allows_valid(tmp_path: Path) -> None:
    md, ckpt = _md_with_predictor(tmp_path, _FakePredictor, pred_name="ok_echo")
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", enforce=True)
    server, base, thread = _serve(md, ctx)
    try:
        status, body = _http_json("POST", f"{base}/predict", {"request": "x"})
        assert status == 200
        assert body["valid"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_enforce_requires_validator(tmp_path: Path) -> None:
    md, ckpt = _md_with_predictor(
        tmp_path, _FakePredictor, pred_name="ok_novalidator", with_validator=False
    )
    with pytest.raises(ValueError, match="requires evaluation.validator"):
        build_serve_context(md, checkpoint=ckpt, device="cpu", enforce=True)


@pytest.mark.parametrize("token", ["", "   "])
def test_serve_refuses_empty_auth_token(tmp_path: Path, token: str) -> None:
    """An empty token must not yield a server that advertises auth_required
    while accepting a bare 'Bearer '. Fail closed instead."""
    md, ckpt = _md_with_predictor(tmp_path, _FakePredictor, pred_name="empty_tok")
    with pytest.raises(ValueError, match="auth token is empty"):
        build_serve_context(md, checkpoint=ckpt, device="cpu", auth_token=token)


def test_serve_non_loopback_requires_auth_token(tmp_path: Path) -> None:
    """The --auth-token help promises it is required for a non-loopback bind,
    so a bare 0.0.0.0 bind is refused before the socket is opened."""
    from maatml.serve import run_server

    md, ckpt = _md_with_predictor(tmp_path, _FakePredictor, pred_name="nonloop")
    with pytest.raises(ValueError, match="requires --auth-token"):
        run_server(md, checkpoint=ckpt, host="0.0.0.0", port=0, device="cpu")
    # An empty host binds every interface, so it is not loopback either.
    with pytest.raises(ValueError, match="requires --auth-token"):
        run_server(md, checkpoint=ckpt, host="", port=0, device="cpu")


def test_serve_500_hides_traceback_by_default(tmp_path: Path) -> None:
    md, ckpt = _md_with_predictor(tmp_path, _BoomPredictor, pred_name="boom")
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    server, base, thread = _serve(md, ctx)
    try:
        status, body = _http_json("POST", f"{base}/predict", {"request": "x"})
        assert status == 500
        assert body["error"] == "internal server error"
        assert "traceback" not in body
        assert "kaboom" not in json.dumps(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_500_debug_includes_traceback(tmp_path: Path) -> None:
    md, ckpt = _md_with_predictor(tmp_path, _BoomPredictor, pred_name="boom2")
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", debug=True)
    server, base, thread = _serve(md, ctx)
    try:
        status, body = _http_json("POST", f"{base}/predict", {"request": "x"})
        assert status == 500
        assert body["error"] == "kaboom"
        assert "traceback" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


_strict_seen: dict = {}


def _strict_validator(raw_output, user_prompt=None):  # no **kwargs, no schema_path
    _strict_seen["user_prompt"] = user_prompt
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    result.passed_layers.add(1)
    return result


def test_serve_strict_validator_signature_through_server(tmp_path: Path) -> None:
    """G5 end-to-end: a validator whose signature omits schema_path is called
    with only the kwargs it accepts (no TypeError fallback dropping user_prompt),
    even when a schema is resolved."""
    _strict_seen.clear()
    md, ckpt = _md_with_predictor(
        tmp_path, _FakePredictor, pred_name="strict_pred", with_validator=False
    )
    (ckpt / "schema.json").write_text("{}", encoding="utf-8")  # so schema_path resolves
    from maatml.registry import VALIDATORS as _V

    _V.register("strict_val", _strict_validator, source="test")
    md.evaluation["validator"] = "strict_val"

    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    assert ctx.schema_path is not None
    assert ctx.validator_params == frozenset({"user_prompt"})
    server, base, thread = _serve(md, ctx)
    try:
        status, body = _http_json("POST", f"{base}/predict?validate=1", {"request": "hi"})
        assert status == 200
        assert body["valid"] is True
        assert _strict_seen["user_prompt"] == "hi"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_resolve_validator_params_subset_and_kwargs() -> None:
    from maatml.serve import _resolve_validator_params

    def strict(raw, user_prompt=None, schema_path=None):  # noqa: ANN001
        return None

    assert _resolve_validator_params(strict) == frozenset({"user_prompt", "schema_path"})

    def loose(raw, **kw):  # noqa: ANN001
        return None

    assert _resolve_validator_params(loose) is None


def _http_with_headers(method, url, body=None, headers=None):
    data = None
    hdrs = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# --- auth token ------------------------------------------------------------


def test_auth_token_required_when_set(serve_model_dir) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", auth_token="s3cret")
    server, base, _thread = _serve(md, ctx)
    try:
        # No token: 401.
        status, _ = _http_with_headers("POST", f"{base}/predict", {"request": "hi"})
        assert status == 401
        # Wrong token: 401.
        status, _ = _http_with_headers(
            "POST",
            f"{base}/predict",
            {"request": "hi"},
            {"Authorization": "Bearer nope"},
        )
        assert status == 401
        # Right token: 200.
        status, pred = _http_with_headers(
            "POST",
            f"{base}/predict",
            {"request": "hi"},
            {"Authorization": "Bearer s3cret"},
        )
        assert status == 200
        assert pred["output"]["echo"] == "hi"
    finally:
        server.shutdown()
        server.server_close()


def test_info_advertises_auth_and_capture(serve_model_dir) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", auth_token="t")
    server, base, _thread = _serve(md, ctx)
    try:
        _status, info = _http_with_headers("GET", f"{base}/info")
        assert info["auth_required"] is True
        assert info["capture_enabled"] is False
    finally:
        server.shutdown()
        server.server_close()


# --- capture ---------------------------------------------------------------


def test_capture_requires_auth_token(serve_model_dir, tmp_path) -> None:
    md, ckpt = serve_model_dir
    with pytest.raises(ValueError, match="requires --auth-token"):
        build_serve_context(md, checkpoint=ckpt, device="cpu", capture_path=tmp_path / "cap.jsonl")


def test_capture_writes_unapproved_rows(serve_model_dir, tmp_path) -> None:
    md, ckpt = serve_model_dir
    cap = tmp_path / "capture.jsonl"
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", auth_token="tok", capture_path=cap)
    server, base, _thread = _serve(md, ctx)
    try:
        status, pred = _http_with_headers(
            "POST",
            f"{base}/predict?capture=1",
            {"request": "capture me"},
            {"Authorization": "Bearer tok"},
        )
        assert status == 200
        assert pred["captured"] is True
    finally:
        server.shutdown()
        server.server_close()

    rows = [json.loads(line) for line in cap.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["source"] == "serve_capture"
    assert rows[0]["approved"] is False
    assert rows[0]["needs_review"] is True
    assert rows[0]["request"] == "capture me"


def test_capture_without_token_is_401(serve_model_dir, tmp_path) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(
        md,
        checkpoint=ckpt,
        device="cpu",
        auth_token="tok",
        capture_path=tmp_path / "c.jsonl",
    )
    server, base, _thread = _serve(md, ctx)
    try:
        status, _ = _http_with_headers("POST", f"{base}/predict?capture=1", {"request": "x"})
        assert status == 401
    finally:
        server.shutdown()
        server.server_close()


# --- retry-with-feedback ---------------------------------------------------


class _FlakyPredictor:
    """Invalid until it sees validation feedback, then valid (once)."""

    def setup(self, checkpoint_dir, **kwargs) -> None:  # noqa: ANN001
        del checkpoint_dir, kwargs

    def predict(self, row: dict) -> str:
        if row.get("_validation_feedback"):
            return json.dumps({"echo": row.get("request"), "ok": True})
        return json.dumps({"echo": row.get("request"), "ok": False})


def test_enforce_retry_recovers_and_counts(serve_model_dir) -> None:
    md, ckpt = serve_model_dir
    PREDICTORS.register("flaky", _FlakyPredictor, source="test")
    md.evaluation["predictor"] = "flaky"
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", enforce=True, max_retries=1)
    server, base, _thread = _serve(md, ctx)
    try:
        status, pred = _http_with_headers("POST", f"{base}/predict", {"request": "hi"})
        assert status == 200
        assert pred["valid"] is True
        assert pred["retries"] == 1
        assert pred["attempts"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_enforce_without_retry_returns_422_with_retry_count(serve_model_dir) -> None:
    """Exit criterion: a validation failure under --enforce is 422 + retry count."""
    md, ckpt = serve_model_dir
    PREDICTORS.register("flaky", _FlakyPredictor, source="test")
    md.evaluation["predictor"] = "flaky"
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu", enforce=True)
    server, base, _thread = _serve(md, ctx)
    try:
        status, pred = _http_with_headers("POST", f"{base}/predict", {"request": "hi"})
        assert status == 422
        assert pred["valid"] is False
        assert pred["retries"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_max_retries_needs_a_validator(serve_model_dir) -> None:
    md, ckpt = serve_model_dir
    md.evaluation.pop("validator")
    with pytest.raises(ValueError, match="max-retries needs evaluation.validator"):
        build_serve_context(md, checkpoint=ckpt, device="cpu", max_retries=2)


def test_capture_sample_id_distinguishes_requests_with_equal_output(
    tmp_path: Path,
) -> None:
    """sample_id was a hash of the model output alone, so two different
    requests that produced the same output collided and ingest dropped the
    later one as a duplicate."""
    from maatml.serve import CaptureWriter

    writer = CaptureWriter(tmp_path / "cap.jsonl", request_field="request")
    assert writer.record({"request": "first ticket"}, {"ok": True}, '{"ok": true}')
    assert writer.record({"request": "second ticket"}, {"ok": True}, '{"ok": true}')

    rows = [json.loads(line) for line in (tmp_path / "cap.jsonl").read_text().splitlines()]
    assert len({r["sample_id"] for r in rows}) == 2, "distinct requests collided"
    # A genuine repeat of the same (request, output) still dedupes.
    writer.record({"request": "first ticket"}, {"ok": True}, '{"ok": true}')
    rows = [json.loads(line) for line in (tmp_path / "cap.jsonl").read_text().splitlines()]
    assert rows[0]["sample_id"] == rows[2]["sample_id"]


def test_capture_applies_declared_sanitizers(tmp_path: Path) -> None:
    """The docstring and docs promised the captured request is sanitized. It
    was written verbatim; now the model's dataset.sanitize tags are applied."""
    from maatml.serve import CaptureWriter

    def _redact(text: str) -> str:
        return text.replace("USER=ALICE", "USER=REDACT")

    writer = CaptureWriter(tmp_path / "cap.jsonl", request_field="request", sanitizers=[_redact])
    writer.record({"request": "job for USER=ALICE"}, {"ok": True}, '{"ok": true}')
    row = json.loads((tmp_path / "cap.jsonl").read_text().splitlines()[0])
    assert row["request"] == "job for USER=REDACT"
    assert "ALICE" not in json.dumps(row)


def test_open_capture_requires_auth_and_writes_serve_capture_rows(tmp_path: Path) -> None:
    from maatml.config import ModelDefinition
    from maatml.serve import open_capture

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        architecture="causal_sft",
        dataset={"request_field": "image"},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    with pytest.raises(ValueError, match="auth-token"):
        open_capture(md, tmp_path / "cap.jsonl", auth_token=None)
    writer = open_capture(md, tmp_path / "cap.jsonl", auth_token="secret")
    assert writer is not None
    writer.record({"image": "cam.png"}, {"detections": []}, '{"detections":[]}')
    row = json.loads((tmp_path / "cap.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["source"] == "serve_capture"
    assert row["approved"] is False
    assert row["image"] == "cam.png"
