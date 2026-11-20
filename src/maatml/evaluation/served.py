"""Core served-model parity predictor and provenance checks.

Evaluates a model served over an OpenAI-compatible HTTP endpoint (such as
``llama-server``, ``vLLM``, or ``maatml serve``) using the exact same rows,
prompt spec, validator, and metrics as an in-process checkpoint predictor.
Optionally verifies the served artifact against its export ``manifest.json``
and probes the server's ``/props`` or ``/v1/models`` identity so parity
reports carry full artifact and server provenance.

Torch-free: uses only the Python standard library so parity checks can run on
any host without PyTorch installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import register_predictor
from ..utils.chat import normalize_messages, parse_tool_calls
from .harness import resolve_eval_asset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_json(
    url: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    data: Optional[bytes] = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object from {url}, got {type(parsed).__name__}")
    return parsed


def verify_served_artifact(
    export_dir: str | Path,
    artifact_relpath: str,
    *,
    checkpoint_dir: Optional[str | Path] = None,
    require_production_gate: bool = True,
) -> dict[str, Any]:
    """Verify a served export artifact against ``<export_dir>/manifest.json``.

    Checks that the file exists, matches the SHA-256 recorded in the export
    manifest, optionally carries non-smoke passing gate evidence, and matches
    ``checkpoint_dir.name`` when ``checkpoint_dir`` is supplied.
    """
    export_root = Path(export_dir)
    manifest_path = export_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"export manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid export manifest: {manifest_path}")

    artifact_path = export_root / artifact_relpath
    if not artifact_path.is_file():
        raise FileNotFoundError(f"served artifact not found: {artifact_path}")
    listed = {
        str(entry["path"]): str(entry["sha256"])
        for entry in manifest.get("files") or []
        if isinstance(entry, dict) and "path" in entry and "sha256" in entry
    }
    artifact_sha = _sha256_file(artifact_path)
    if listed.get(artifact_relpath) != artifact_sha:
        raise ValueError(
            f"served artifact {artifact_relpath!r} hash {artifact_sha} does not match "
            f"export manifest entry {listed.get(artifact_relpath)!r}"
        )

    if require_production_gate:
        evidence = manifest.get("gate_evidence") or {}
        if not (
            isinstance(evidence, dict)
            and evidence.get("gated")
            and evidence.get("passed")
            and not evidence.get("smoke_gated")
        ):
            raise ValueError("export manifest carries no production gate evidence")

    if checkpoint_dir is not None and manifest.get("run_id"):
        ckpt_name = Path(checkpoint_dir).name
        if str(manifest["run_id"]) != ckpt_name:
            raise ValueError(
                f"export manifest run_id {manifest['run_id']!r} does not match "
                f"checkpoint {ckpt_name!r}"
            )

    return {
        "kind": "served_export_artifact",
        "run_id": manifest.get("run_id"),
        "identity": manifest.get("identity"),
        "artifact": artifact_relpath,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact_path.stat().st_size,
        "export_manifest_sha256": _sha256_file(manifest_path),
    }


def probe_server_identity(
    base_url: str,
    *,
    expected_build_info: Optional[str] = None,
    expected_model_path: Optional[str | Path] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Read server identity from ``/props`` (when exposed) and ``/v1/models``.

    Raises ``ValueError`` when ``expected_build_info`` or ``expected_model_path``
    is supplied and does not match the running server.
    """
    base = base_url.rstrip("/")
    props: dict[str, Any] = {}
    models: list[str] = []

    try:
        props = _http_json(f"{base}/props", headers=headers, timeout=timeout)
    except (urllib.error.URLError, ValueError, OSError):
        if expected_build_info is not None or expected_model_path is not None:
            raise

    if expected_build_info is not None and props.get("build_info") != expected_build_info:
        raise ValueError(
            f"server build_info differs: expected {expected_build_info!r}, "
            f"got {props.get('build_info')!r}"
        )
    if expected_model_path is not None:
        served_path = props.get("model_path")
        if not served_path:
            raise ValueError("server /props did not report model_path")
        if Path(str(served_path)).resolve() != Path(expected_model_path).resolve():
            raise ValueError(
                f"server serves {served_path!r}, expected {str(expected_model_path)!r}"
            )

    try:
        listed = _http_json(f"{base}/v1/models", headers=headers, timeout=timeout)
        for item in listed.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
    except (urllib.error.URLError, ValueError, OSError):
        pass

    return {
        "base_url": base,
        "build_info": props.get("build_info"),
        "model_path": props.get("model_path"),
        "served_models": models,
    }


def _normalize_for_parity(text: str) -> Any:
    stripped = (text or "").strip()
    try:
        return json.loads(stripped)
    except ValueError:
        return stripped


@dataclass(frozen=True)
class ParityMismatch:
    """One row where the served model diverged from the reference output."""

    index: int
    reference: str
    served: str
    reference_valid: Optional[bool] = None
    served_valid: Optional[bool] = None


@dataclass(frozen=True)
class ParityReport:
    """Comparison summary between reference checkpoint outputs and served outputs."""

    total: int
    exact_matches: int
    equivalent_matches: int
    exact_match_rate: float
    equivalent_match_rate: float
    reference_valid_rate: Optional[float] = None
    served_valid_rate: Optional[float] = None
    mismatches: tuple[ParityMismatch, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "exact_matches": self.exact_matches,
            "equivalent_matches": self.equivalent_matches,
            "exact_match_rate": self.exact_match_rate,
            "equivalent_match_rate": self.equivalent_match_rate,
            "reference_valid_rate": self.reference_valid_rate,
            "served_valid_rate": self.served_valid_rate,
            "mismatches": [
                {
                    "index": m.index,
                    "reference": m.reference,
                    "served": m.served,
                    "reference_valid": m.reference_valid,
                    "served_valid": m.served_valid,
                }
                for m in self.mismatches
            ],
        }


def compare_parity(
    reference_outputs: Sequence[str],
    served_outputs: Sequence[str],
    *,
    validator: Optional[Callable[[str], Any]] = None,
    max_mismatches: int = 25,
) -> ParityReport:
    """Compare reference predictions against served-model predictions."""
    if len(reference_outputs) != len(served_outputs):
        raise ValueError(
            f"output length mismatch: {len(reference_outputs)} reference vs "
            f"{len(served_outputs)} served"
        )
    total = len(reference_outputs)
    if total == 0:
        return ParityReport(
            total=0,
            exact_matches=0,
            equivalent_matches=0,
            exact_match_rate=1.0,
            equivalent_match_rate=1.0,
            reference_valid_rate=1.0 if validator is not None else None,
            served_valid_rate=1.0 if validator is not None else None,
        )

    exact = 0
    equiv = 0
    ref_ok = 0
    srv_ok = 0
    mismatches: list[ParityMismatch] = []

    for idx, (ref_raw, srv_raw) in enumerate(zip(reference_outputs, served_outputs, strict=True)):
        ref_s = (ref_raw or "").strip()
        srv_s = (srv_raw or "").strip()
        is_exact = ref_s == srv_s
        is_equiv = is_exact or (_normalize_for_parity(ref_s) == _normalize_for_parity(srv_s))
        if is_exact:
            exact += 1
        if is_equiv:
            equiv += 1

        ref_v: Optional[bool] = None
        srv_v: Optional[bool] = None
        if validator is not None:
            ref_res = validator(ref_s)
            srv_res = validator(srv_s)
            ref_v = bool(getattr(ref_res, "ok", ref_res))
            srv_v = bool(getattr(srv_res, "ok", srv_res))
            ref_ok += int(ref_v)
            srv_ok += int(srv_v)

        if not is_equiv and len(mismatches) < max_mismatches:
            mismatches.append(
                ParityMismatch(
                    index=idx,
                    reference=ref_s,
                    served=srv_s,
                    reference_valid=ref_v,
                    served_valid=srv_v,
                )
            )

    return ParityReport(
        total=total,
        exact_matches=exact,
        equivalent_matches=equiv,
        exact_match_rate=exact / total,
        equivalent_match_rate=equiv / total,
        reference_valid_rate=(ref_ok / total) if validator is not None else None,
        served_valid_rate=(srv_ok / total) if validator is not None else None,
        mismatches=tuple(mismatches),
    )


def _format_openai_message(message: dict[str, Any]) -> str:
    """Turn an OpenAI chat completion message into a single string prediction.

    Preserves tool calls in Hermes ``<tool_call>`` JSON blocks so downstream
    parsers and validators read the same representation as local generation.
    """
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        raw_fn = call.get("function")
        fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args: Any = json.loads(raw_args)
            except ValueError:
                args = raw_args
        else:
            args = raw_args
        block = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        parts.append(f"<tool_call>{block}</tool_call>")
    return "\n".join(parts).strip()


class ServedModelPredictor:
    """Generic HTTP OpenAI-compatible predictor for served-model parity checks.

    Configuration sources (highest priority first):
      1. Keyword arguments to ``setup(...)`` (``base_url``, ``model``, ...)
      2. ``model_def.extensions["serve_parity"]`` (inline dict or ``config`` JSON path)
      3. ``MAATML_SERVE_PARITY_URL`` / ``MAATML_SERVE_PARITY_MODEL`` env vars
    """

    def __init__(self) -> None:
        self._base_url = ""
        self._model = ""
        self._headers: dict[str, str] = {}
        self._timeout = 120.0
        self._max_new_tokens = 512
        self._temperature = 0.0
        self._seed: Optional[int] = 0
        self._extra_body: dict[str, Any] = {}
        self._prompt_spec: dict[str, Any] = {}
        self._request_field = "request"
        self._user_placeholder = "<<USER_REQUEST>>"
        self._binding: dict[str, Any] = {}
        self._server_identity: dict[str, Any] = {}
        self._wall_ms: list[float] = []
        self._errors = 0

    def _load_config(
        self, model_def: Optional[ModelDefinition], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if model_def is not None:
            ext = (model_def.extensions or {}).get("serve_parity")
            if isinstance(ext, dict):
                cfg_path = ext.get("config")
                if isinstance(cfg_path, str) and cfg_path.strip():
                    loaded = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        cfg.update(loaded)
                for key, value in ext.items():
                    if key != "config":
                        cfg[key] = value
        for key in (
            "base_url",
            "model",
            "api_key",
            "timeout",
            "temperature",
            "seed",
            "extra_body",
            "export_dir",
            "artifact",
            "artifact_path",
            "baseline",
            "require_production_gate",
            "expected_build_info",
            "verify_server_props",
        ):
            if key in kwargs and kwargs[key] is not None:
                cfg[key] = kwargs[key]
        return cfg

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Optional[ModelDefinition] = None,
        max_new_tokens: int = 512,
        prompt_spec_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        cfg = self._load_config(model_def, kwargs)
        base_url = str(cfg.get("base_url") or os.environ.get("MAATML_SERVE_PARITY_URL") or "")
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            raise ValueError(
                "served predictor requires base_url (set extensions.serve_parity.base_url "
                "or MAATML_SERVE_PARITY_URL)"
            )
        self._base_url = base_url
        self._max_new_tokens = int(cfg.get("max_new_tokens") or max_new_tokens)
        self._temperature = float(cfg.get("temperature", 0.0))
        self._seed = None if cfg.get("seed") is None else int(cfg["seed"])
        self._timeout = float(cfg.get("timeout", 120.0))
        self._extra_body = (
            dict(cfg["extra_body"]) if isinstance(cfg.get("extra_body"), dict) else {}
        )
        self._headers = {}
        api_key = cfg.get("api_key") or os.environ.get("MAATML_SERVE_PARITY_API_KEY")
        if isinstance(api_key, str) and api_key.strip():
            self._headers["Authorization"] = f"Bearer {api_key.strip()}"

        if model_def is not None:
            ds_cfg = get_dataset_cfg(model_def)
            self._request_field = (
                ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request"
            )
            self._user_placeholder = ds_cfg.get("user_placeholder") or "<<USER_REQUEST>>"

        try:
            spec_path = resolve_eval_asset(
                "prompt_spec",
                model_def=model_def,
                checkpoint_dir=Path(checkpoint_dir),
                explicit=prompt_spec_path,
                filenames=("prompt_spec.json",),
            )
            self._prompt_spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._prompt_spec = {}

        # Optional export-artifact verification (baseline or gate-evidenced export).
        expected_model_path: Optional[Path] = None
        if cfg.get("baseline") and cfg.get("artifact_path"):
            artifact_path = Path(str(cfg["artifact_path"]))
            self._binding = {
                "kind": "baseline_artifact",
                "artifact": artifact_path.name,
                "artifact_path": str(artifact_path.resolve()),
                "artifact_sha256": _sha256_file(artifact_path),
                "artifact_bytes": artifact_path.stat().st_size,
            }
            expected_model_path = artifact_path
        elif cfg.get("export_dir") and cfg.get("artifact"):
            self._binding = verify_served_artifact(
                cfg["export_dir"],
                str(cfg["artifact"]),
                checkpoint_dir=checkpoint_dir if cfg.get("check_run_id", True) else None,
                require_production_gate=bool(cfg.get("require_production_gate", True)),
            )
            expected_model_path = Path(self._binding["artifact_path"])
        else:
            self._binding = {"kind": "remote_endpoint", "base_url": self._base_url}

        verify_props = bool(
            cfg.get("verify_server_props")
            or cfg.get("expected_build_info") is not None
            or (expected_model_path is not None and cfg.get("verify_model_path", False))
        )
        self._server_identity = probe_server_identity(
            self._base_url,
            expected_build_info=cfg.get("expected_build_info"),
            expected_model_path=expected_model_path if verify_props else None,
            headers=self._headers,
            timeout=min(self._timeout, 15.0),
        )

        model = str(cfg.get("model") or os.environ.get("MAATML_SERVE_PARITY_MODEL") or "").strip()
        if not model and self._server_identity.get("served_models"):
            model = str(self._server_identity["served_models"][0])
        if not model:
            model = Path(checkpoint_dir).name or "default"
        self._model = model
        self._wall_ms = []
        self._errors = 0

    def _build_messages(
        self, row: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], Optional[list[dict]]]:
        raw_messages = row.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            messages = normalize_messages(raw_messages)
            while messages and messages[-1].get("role") == "assistant":
                messages.pop()
            tools = row.get("tools")
            return messages, (list(tools) if isinstance(tools, list) and tools else None)

        request_text = str(row.get(self._request_field, ""))
        user_tpl = str(self._prompt_spec.get("user_template") or self._user_placeholder)
        user_text = user_tpl.replace(self._user_placeholder, request_text)
        messages = []
        system = self._prompt_spec.get("system")
        if isinstance(system, str) and system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return messages, None

    def _chat_completion(self, row: dict[str, Any]) -> dict[str, Any]:
        messages, tools = self._build_messages(row)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_new_tokens,
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        if tools:
            payload["tools"] = tools
        row_kwargs = row.get("chat_template_kwargs")
        if isinstance(row_kwargs, dict) and row_kwargs:
            payload["chat_template_kwargs"] = dict(row_kwargs)
        payload.update(self._extra_body)

        started = time.perf_counter()
        try:
            response = _http_json(
                f"{self._base_url}/v1/chat/completions",
                payload=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
        except Exception:
            self._errors += 1
            raise
        finally:
            self._wall_ms.append(round((time.perf_counter() - started) * 1000.0, 2))
        return response

    def predict(self, row: dict[str, Any]) -> str:
        response = self._chat_completion(row)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ValueError(f"served endpoint returned no choices: {response!r}")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise ValueError(f"served endpoint choice missing message: {choices[0]!r}")
        return _format_openai_message(message)

    def predict_structured(self, row: dict[str, Any]) -> dict[str, Any]:
        raw = self.predict(row)
        parsed = parse_tool_calls(raw)
        return {
            "raw": raw,
            "content": parsed.content,
            "tool_calls": parsed.tool_calls,
            "errors": parsed.errors,
        }

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[str]:
        return [self.predict(row) for row in rows]

    def report_extras(self) -> dict[str, Any]:
        return {
            "serve_parity": {
                "base_url": self._base_url,
                "model": self._model,
                "binding": self._binding,
                "server_identity": self._server_identity,
                "wall_ms": list(self._wall_ms),
                "errors": self._errors,
            }
        }


register_predictor("served_http")(ServedModelPredictor)
register_predictor("serve_parity")(ServedModelPredictor)
