"""HTTP inference server for any maatml model folder.

Uses stdlib ``ThreadingHTTPServer``: no FastAPI/uvicorn, so the base install
stays light (important on JetPack). The predictor is loaded once at startup via
the same registry path the eval harness uses.

Endpoints:

* ``GET /health``: liveness + identity
* ``GET /info``: model summary + packaging hints
* ``POST /predict``: dataset-shaped JSON row → prediction
  (``?validate=1`` runs the registered validator when configured)
"""

from __future__ import annotations

import hmac
import inspect
import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .config import ModelDefinition, get_dataset_cfg
from .device import resolve_device
from .registry import (
    PREDICTORS,
    SANITIZERS,
    VALIDATORS,
    discover_plugins,
    load_model_plugins,
)
from .runs import resolve_checkpoint
from .scaffold import normalize_architecture
from .utils.io import sha256_bytes
from .validation.base import ValidationResult, strip_fences

# Reject request bodies larger than this by default (1 MiB). Predict rows are
# small JSON; anything larger is almost certainly abuse or a mistake.
DEFAULT_MAX_BODY_BYTES = 1_048_576

logger = logging.getLogger("maatml.serve")

# "" is deliberately absent: an empty host binds every interface, so treating it
# as loopback would suppress the unauthenticated-bind check on the widest bind
# there is.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class RequestTooLarge(Exception):
    """Request body exceeded the configured size cap (maps to HTTP 413)."""


class CaptureWriter:
    """Appends served predictions to a JSONL file for later human review.

    Captured rows are explicitly **not** gold: each carries ``approved: false``
    and ``needs_review: true``, and ``maatml ingest`` refuses to accept a
    captured row until a human (or teacher) has corrected and approved it. The
    file is size-capped so an unattended server cannot fill the disk, and the
    request is passed through the model's declared ``dataset.sanitize`` tags
    before it is written.
    """

    def __init__(
        self,
        path: Path,
        *,
        request_field: str,
        max_rows: int = 10_000,
        max_bytes: int = 32 * 1024 * 1024,
        sanitizers: Optional[list[Any]] = None,
    ) -> None:
        self.path = Path(path)
        self.request_field = request_field
        self.sanitizers = list(sanitizers or ())
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self._rows = 0
        self._lock = threading.Lock()
        if self.path.is_file():
            with open(self.path, "r", encoding="utf-8") as handle:
                self._rows = sum(1 for line in handle if line.strip())

    def capped(self) -> bool:
        if self._rows >= self.max_rows:
            return True
        return self.path.is_file() and self.path.stat().st_size >= self.max_bytes

    def record(self, row: dict[str, Any], output: Any, raw: str) -> bool:
        """Append one capture row; returns False when the cap is reached."""
        with self._lock:
            if self.capped():
                return False
            request = row.get(self.request_field)
            for sanitize in self.sanitizers:
                if isinstance(request, str):
                    request = sanitize(request)
            # Identity is the (request, output) pair. Hashing the output alone
            # collided across distinct requests, and ingest then dropped the
            # later ones as "duplicate": exactly the reviewed rows the flywheel
            # exists to keep.
            identity = f"{request}\x00{raw}".encode("utf-8")
            entry = {
                "sample_id": f"capture-{sha256_bytes(identity)[:16]}",
                "source": "serve_capture",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                self.request_field: request,
                "model_output": output,
                # A served prediction is a proposal, not a label. ingest gates
                # on approved: only a reviewed, approved row becomes a seed.
                "approved": False,
                "needs_review": True,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._rows += 1
            return True


def open_capture(
    model_def: Any,
    capture_path: str | Path | None,
    *,
    auth_token: Optional[str] = None,
) -> CaptureWriter | None:
    """Build a ``CaptureWriter`` for any server backend (HTTP, UDS, …).

    Same rules as ``http``: capture requires a non-empty auth token. Returns
    ``None`` when ``capture_path`` is omitted. Custom servers should call this
    and then ``writer.record(row, output, raw)`` (or
    ``LifecycleServer.record_capture``) so ingest sees the same
    ``source: serve_capture`` / ``approved: false`` rows.
    """
    if capture_path is None:
        return None
    _reject_empty_token(auth_token)
    if not auth_token:
        raise ValueError(
            "serve --capture requires --auth-token (or MAATML_SERVE_TOKEN): "
            "captured rows are unreviewed writes and must not be open to "
            "anonymous clients."
        )
    cfg = get_dataset_cfg(model_def)
    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")
    sanitize_tags = [str(tag).strip() for tag in (cfg.get("sanitize") or []) if str(tag).strip()]
    sanitizers = [SANITIZERS.require(tag) for tag in sanitize_tags]
    return CaptureWriter(Path(capture_path), request_field=request_field, sanitizers=sanitizers)


def _reject_empty_token(auth_token: Optional[str]) -> None:
    """Refuse an empty auth token rather than serving formality-only auth.

    An empty token would advertise ``auth_required`` on /info while
    ``_token_matches`` accepted a bare ``Bearer ``. The usual cause is an unset
    variable expanding to "" in a unit file or compose env.
    """
    if auth_token is not None and not auth_token.strip():
        raise ValueError(
            "serve auth token is empty; unset MAATML_SERVE_TOKEN to serve "
            "without auth, or provide a non-empty token."
        )


def _token_matches(provided: Optional[str], expected: str) -> bool:
    """Constant-time bearer-token comparison (avoids a timing side channel)."""
    if not provided:
        return False
    if provided.startswith("Bearer "):
        provided = provided[len("Bearer ") :]
    return hmac.compare_digest(provided, expected)


@dataclass
class ServeContext:
    """Shared state for the request handler (one per server)."""

    model_def: ModelDefinition
    checkpoint_dir: Path
    device: str
    predictor: Any
    validator: Any | None = None
    schema_path: Path | None = None
    contracts_path: Path | None = None
    prompt_spec_path: Path | None = None
    # CORS is opt-in: when None, no Access-Control-Allow-Origin header is sent,
    # so a browser on another origin cannot read responses. Set to "*" or a
    # specific origin to enable cross-origin access deliberately.
    cors_origin: str | None = None
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    # When True, a /predict whose validator rejects the output returns HTTP 422
    # (the validator gates live inference) instead of 200 with valid:false.
    enforce: bool = False
    # When True, 500 responses include the exception message and traceback.
    # Off by default so internals never leak to unauthenticated clients.
    debug: bool = False
    # Subset of {user_prompt, schema_path, contracts_path} the validator accepts,
    # resolved once at startup; None means it accepts anything (**kwargs).
    validator_params: frozenset[str] | None = None
    # Bearer token required on protected endpoints (always for /predict?capture,
    # and for every request when set). None means no auth (loopback only).
    auth_token: Optional[str] = None
    # On a validation failure under --enforce, feed the error back and re-ask
    # the model once more before giving up. 0 disables (reject on first failure).
    max_retries: int = 0
    # Capture writer: when set, accepted /predict rows are appended for review.
    capture: Optional["CaptureWriter"] = None
    started_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _resolve_eval_asset(
    key: str,
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    filenames: tuple[str, ...],
) -> Path | None:
    cfg = get_dataset_cfg(model_def)
    rel = cfg.get(key)
    if isinstance(rel, str):
        path = model_def.resolve(rel)
        if path.is_file():
            return path
        # Declared and missing is a config error. Falling through to None would
        # start the server with the validator silently short of an asset it was
        # told to use, which under --enforce means gating on less than declared.
        raise ValueError(f"model.yml declares {key}={rel!r} but the file is missing: {path}")
    for name in filenames:
        cand = checkpoint_dir / name
        if cand.is_file():
            return cand
    return None


def _resolve_predictor_name(model_def: ModelDefinition) -> str:
    ev = model_def.evaluation or {}
    predictor = ev.get("predictor")
    if isinstance(predictor, str) and predictor:
        return predictor
    arch = normalize_architecture(model_def.architecture)
    if PREDICTORS.get(model_def.architecture):
        return model_def.architecture
    if PREDICTORS.get(arch):
        return arch
    raise KeyError(
        f"No evaluation.predictor in model.yml and no predictor registered for "
        f"architecture={model_def.architecture!r}. Known: {PREDICTORS.names()}"
    )


def _resolve_validator_params(validator: Any) -> frozenset[str] | None:
    """Return the accepted subset of the known validator kwargs, resolved once.

    Returns None when the validator takes ``**kwargs`` (accepts anything) or its
    signature cannot be introspected. This replaces a per-request try/except
    TypeError fallback that could silently drop kwargs and weaken validation.
    """
    known = ("user_prompt", "schema_path", "contracts_path")
    try:
        sig = inspect.signature(validator)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    names = {p.name for p in params}
    return frozenset(k for k in known if k in names)


def build_serve_context(
    model_def: ModelDefinition,
    *,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    enforce: bool = False,
    debug: bool = False,
    auth_token: Optional[str] = None,
    max_retries: int = 0,
    capture_path: Optional[str | Path] = None,
) -> ServeContext:
    """Load plugins, resolve checkpoint, instantiate and setup the predictor."""
    discover_plugins()
    if model_def.plugins:
        load_model_plugins(model_def.model_dir, model_def.plugins)

    checkpoint_dir = resolve_checkpoint(model_def, checkpoint)
    try:
        target_device = resolve_device(device)
    except ImportError:
        # torch absent: real serving needs it, but keep a plain device string so
        # torch-free predictors (and tests) can still build a serve context.
        target_device = "cpu" if device in (None, "auto") else device

    pred_name = _resolve_predictor_name(model_def)
    pred_obj = PREDICTORS.require(pred_name)
    if isinstance(pred_obj, type):
        pred_obj = pred_obj()

    schema_path = _resolve_eval_asset(
        "schema",
        model_def,
        checkpoint_dir,
        ("schema.json",),
    )
    contracts_path = _resolve_eval_asset(
        "contracts",
        model_def,
        checkpoint_dir,
        ("node_contracts.json",),
    )
    prompt_spec_path = _resolve_eval_asset(
        "prompt_spec",
        model_def,
        checkpoint_dir,
        ("prompt_spec.json",),
    )

    setup = getattr(pred_obj, "setup", None)
    if callable(setup):
        setup(
            checkpoint_dir,
            model_def=model_def,
            device=target_device,
            max_input_tokens=model_def.packaging.max_input_tokens,
            schema_path=schema_path,
            contracts_path=contracts_path,
            prompt_spec_path=prompt_spec_path,
        )

    validator = None
    ev = model_def.evaluation or {}
    val_name = ev.get("validator")
    if isinstance(val_name, str) and val_name:
        validator = VALIDATORS.require(val_name)

    _reject_empty_token(auth_token)

    if enforce and validator is None:
        raise ValueError(
            "serve --enforce requires evaluation.validator in model.yml so live "
            "inference can be gated; none is configured."
        )
    if max_retries and validator is None:
        raise ValueError(
            "serve --max-retries needs evaluation.validator: there is no way to "
            "tell a retry succeeded without a validator to re-check the output."
        )

    capture = open_capture(model_def, capture_path, auth_token=auth_token)

    return ServeContext(
        model_def=model_def,
        checkpoint_dir=checkpoint_dir,
        device=str(target_device),
        predictor=pred_obj,
        validator=validator,
        schema_path=schema_path,
        contracts_path=contracts_path,
        prompt_spec_path=prompt_spec_path,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
        enforce=enforce,
        debug=debug,
        validator_params=_resolve_validator_params(validator) if validator else None,
        auth_token=auth_token,
        max_retries=max_retries,
        capture=capture,
    )


def _try_parse_json(raw: str) -> Any | None:
    text = strip_fences(raw)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _validation_payload(result: ValidationResult) -> dict[str, Any]:
    return {
        "valid": bool(result.ok),
        "errors": [
            {
                "layer": e.layer,
                "code": e.code,
                "message": e.message,
                "location": e.location,
                "hint": e.hint,
            }
            for e in result.errors
        ],
    }


def _validate_output(ctx: ServeContext, row: dict[str, Any], raw: Any) -> ValidationResult:
    assert ctx.validator is not None  # callers guard on this
    cfg = get_dataset_cfg(ctx.model_def)
    request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
    user_prompt = row.get(request_field)
    if not isinstance(user_prompt, str):
        user_prompt = None
    raw_str = raw if isinstance(raw, str) else json.dumps(raw)
    available: dict[str, Any] = {"user_prompt": user_prompt}
    if ctx.schema_path is not None:
        available["schema_path"] = ctx.schema_path
    if ctx.contracts_path is not None:
        available["contracts_path"] = ctx.contracts_path
    if ctx.validator_params is None:
        kwargs = available
    else:
        kwargs = {k: v for k, v in available.items() if k in ctx.validator_params}
    return ctx.validator(raw_str, **kwargs)


def _feedback_row(
    ctx: ServeContext, row: dict[str, Any], result: ValidationResult
) -> dict[str, Any]:
    """Row for a retry: the original plus the validation error to correct.

    The predictor decides how to use ``_validation_feedback`` (a chat predictor
    appends it as a user turn); a predictor that ignores the key simply re-runs
    unchanged, which is a harmless no-op rather than an error.
    """
    parts: list[str] = []
    for e in result.errors:
        part = f"{e.code}: {e.message}"
        if e.hint:
            part = f"{part} ({e.hint})"
        parts.append(part)
    errors = "; ".join(parts) or "invalid output"
    retry = dict(row)
    retry["_validation_feedback"] = (
        f"The previous response failed validation ({errors}). "
        "Return a corrected response that satisfies the contract."
    )
    return retry


def _run_predict(
    ctx: ServeContext,
    row: dict[str, Any],
    *,
    do_validate: bool,
) -> dict[str, Any]:
    predict = ctx.predictor.predict if hasattr(ctx.predictor, "predict") else ctx.predictor
    t0 = time.perf_counter()
    attempts = 0
    result: Optional[ValidationResult] = None
    with ctx.lock:
        raw = predict(row)
        attempts += 1
        if do_validate and ctx.validator is not None:
            result = _validate_output(ctx, row, raw)
            # Bounded retry-with-feedback: feed the error back and re-ask, up to
            # max_retries times. Each retry is counted and reported, never silent.
            while not result.ok and attempts <= ctx.max_retries:
                raw = predict(_feedback_row(ctx, row, result))
                attempts += 1
                result = _validate_output(ctx, row, raw)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    payload: dict[str, Any] = {
        "output": _try_parse_json(raw) if isinstance(raw, str) else raw,
        "raw": raw if isinstance(raw, str) else json.dumps(raw),
        "latency_ms": round(latency_ms, 3),
        "attempts": attempts,
        "retries": attempts - 1,
    }

    if do_validate:
        if ctx.validator is None:
            payload["valid"] = None
            payload["errors"] = [
                {
                    "layer": 0,
                    "code": "no_validator",
                    "message": "No evaluation.validator configured in model.yml",
                    "location": None,
                }
            ]
        elif result is not None:
            payload.update(_validation_payload(result))
    return payload


def _make_handler(ctx: ServeContext) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "maatml-serve/0.4"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Quiet by default; rich console prints startup banner.
            return

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if ctx.cors_origin:
                self.send_header("Access-Control-Allow-Origin", ctx.cors_origin)
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("POST body required (JSON object)")
            if length > ctx.max_body_bytes:
                raise RequestTooLarge(
                    f"request body {length} bytes exceeds cap {ctx.max_body_bytes} bytes"
                )
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON body: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            if ctx.cors_origin:
                self.send_header("Access-Control-Allow-Origin", ctx.cors_origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            md = ctx.model_def
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "identity": md.identity,
                        "architecture": md.architecture,
                        "checkpoint": str(ctx.checkpoint_dir),
                        "device": ctx.device,
                        "uptime_s": round(time.time() - ctx.started_at, 1),
                    },
                )
                return
            if path == "/info":
                pkg = md.packaging
                self._send_json(
                    200,
                    {
                        "name": md.name,
                        "model_id": md.model_id,
                        "version": md.version,
                        "identity": md.identity,
                        "architecture": md.architecture,
                        "task": md.task,
                        "base_model": md.base_model,
                        "description": md.description,
                        "checkpoint": str(ctx.checkpoint_dir),
                        "device": ctx.device,
                        "packaging": {
                            "max_input_tokens": pkg.max_input_tokens,
                            "expected_latency_ms": pkg.expected_latency_ms,
                            "weights_dtype": pkg.weights_dtype,
                            "confidence_thresholds": pkg.confidence_thresholds,
                        },
                        "sidecars": {
                            "schema": str(ctx.schema_path) if ctx.schema_path else None,
                            "contracts": (str(ctx.contracts_path) if ctx.contracts_path else None),
                            "prompt_spec": (
                                str(ctx.prompt_spec_path) if ctx.prompt_spec_path else None
                            ),
                        },
                        "predictor": _resolve_predictor_name(md),
                        "validator": (md.evaluation or {}).get("validator"),
                        "enforce": ctx.enforce,
                        "max_retries": ctx.max_retries,
                        "auth_required": ctx.auth_token is not None,
                        "capture_enabled": ctx.capture is not None,
                    },
                )
                return
            self._send_json(404, {"error": f"Unknown path {path!r}"})

        def _authorized(self) -> bool:
            if ctx.auth_token is None:
                return True
            return _token_matches(self.headers.get("Authorization"), ctx.auth_token)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path != "/predict":
                self._send_json(404, {"error": f"Unknown path {path!r}"})
                return
            qs = parse_qs(parsed.query)
            want_capture = any(v.lower() in ("1", "true", "yes") for v in qs.get("capture", []))
            # Capture always needs the token; when a token is configured it is
            # required for every /predict.
            if (want_capture or ctx.auth_token is not None) and not self._authorized():
                self._send_json(401, {"error": "missing or invalid auth token"})
                return
            if want_capture and ctx.capture is None:
                self._send_json(
                    400,
                    {"error": "capture not enabled; start serve with --capture PATH"},
                )
                return
            # Under --enforce the validator gates every /predict regardless of
            # whether the client asked for validation.
            do_validate = ctx.enforce or any(
                v.lower() in ("1", "true", "yes") for v in qs.get("validate", [])
            )
            try:
                row = self._read_json_body()
                result = _run_predict(ctx, row, do_validate=do_validate)
                if want_capture and ctx.capture is not None:
                    stored = ctx.capture.record(row, result["output"], result["raw"])
                    result["captured"] = stored
                    if not stored:
                        result["capture_note"] = "capture cap reached; row not stored"
                status = 200
                if ctx.enforce and result.get("valid") is False:
                    status = 422
                self._send_json(status, result)
            except RequestTooLarge as exc:
                self._send_json(413, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                # Never leak internals to the client. Log the full traceback
                # server-side; include it in the response only under --debug.
                logger.exception("predict failed")
                body: dict[str, Any] = {"error": "internal server error"}
                if ctx.debug:
                    body["error"] = str(exc)
                    body["traceback"] = traceback.format_exc()
                self._send_json(500, body)

    return Handler


def serve_model(
    model_def: ModelDefinition,
    *,
    checkpoint: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    enforce: bool = False,
    debug: bool = False,
    context: ServeContext | None = None,
) -> ThreadingHTTPServer:
    """Build context (unless provided) and return a ready ``ThreadingHTTPServer``.

    Caller is responsible for ``serve_forever()`` / shutdown. Useful for tests
    that want an ephemeral port without blocking.
    """
    ctx = context or build_serve_context(
        model_def,
        checkpoint=checkpoint,
        device=device,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
        enforce=enforce,
        debug=debug,
    )
    handler = _make_handler(ctx)
    server = ThreadingHTTPServer((host, port), handler)
    # Attach context for callers / tests.
    server.maatml_context = ctx  # type: ignore[attr-defined]
    return server


def run_server(
    model_def: ModelDefinition,
    *,
    checkpoint: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    enforce: bool = False,
    debug: bool = False,
    auth_token: Optional[str] = None,
    max_retries: int = 0,
    capture_path: Optional[str | Path] = None,
    allow_unauthenticated: bool = False,
) -> None:
    """Block serving until KeyboardInterrupt."""
    from rich.console import Console

    console = Console()
    # Both checked before the socket is bound: refusing after `serve_model`
    # would leave the port open on the interface we are refusing to serve on.
    _reject_empty_token(auth_token)
    if host not in _LOOPBACK_HOSTS and not auth_token and not allow_unauthenticated:
        raise ValueError(
            f"serve on non-loopback host {host!r} requires --auth-token (or "
            "MAATML_SERVE_TOKEN): anyone who can reach the port could query the "
            "model. Pass --allow-unauthenticated to bind anyway on a trusted "
            "network."
        )
    ctx = build_serve_context(
        model_def,
        checkpoint=checkpoint,
        device=device,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
        enforce=enforce,
        debug=debug,
        auth_token=auth_token,
        max_retries=max_retries,
        capture_path=capture_path,
    )
    server = serve_model(
        model_def,
        checkpoint=checkpoint,
        host=host,
        port=port,
        device=device,
        context=ctx,
    )
    if host not in _LOOPBACK_HOSTS and not auth_token:
        console.print(
            f"[yellow]warning[/] binding non-loopback host {host!r} with no "
            "auth token because --allow-unauthenticated was passed; anyone who "
            "can reach the port can query the model."
        )
    console.print(
        f"[green]serving[/] {model_def.identity} ({model_def.architecture}) "
        f"ckpt={ctx.checkpoint_dir} device={ctx.device}"
        + (" [bold]enforce[/]" if enforce else "")
        + (f" [bold]retries={max_retries}[/]" if max_retries else "")
        + (" [bold]auth[/]" if auth_token else "")
        + (" [bold]capture[/]" if capture_path else "")
    )
    console.print(f"  GET  http://{host}:{port}/health")
    console.print(f"  GET  http://{host}:{port}/info")
    console.print(f"  POST http://{host}:{port}/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]shutting down[/]")
    finally:
        server.shutdown()
        server.server_close()
