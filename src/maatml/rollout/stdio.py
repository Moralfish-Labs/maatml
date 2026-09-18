"""Carry the rollout environment protocol over a subprocess's stdio."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Optional, Sequence

from .protocol import (
    PROTOCOL,
    Hello,
    Labels,
    RolloutEnvError,
    StepResult,
    Verification,
)


class StdioEnvironment:
    """An environment served by ``command`` over JSON lines.

    stderr is inherited, so the server's own logging stays visible.
    """

    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise RolloutEnvError("environment command is empty")
        self._command = list(command)
        self._process: Optional[subprocess.Popen[str]] = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def __enter__(self) -> "StdioEnvironment":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Close stdin so the server exits on EOF, then reap it."""
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()

    def _call(self, op: str, **fields: Any) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RolloutEnvError("environment is shut down")
        request = json.dumps({"op": op, **fields}, separators=(",", ":"))
        try:
            process.stdin.write(request + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RolloutEnvError(f"{op}: environment pipe failed: {exc}") from exc
        if not line:
            raise RolloutEnvError(f"{op}: environment closed its output (exit {process.poll()})")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutEnvError(f"{op}: reply is not JSON: {line.strip()[:200]}") from exc
        if not isinstance(reply, dict) or reply.get("ok") is not True:
            error = reply.get("error") if isinstance(reply, dict) else reply
            raise RolloutEnvError(f"{op}: {error}")
        return reply

    def hello(self) -> Hello:
        reply = self._call("hello")
        hello = Hello(
            name=str(reply.get("name") or ""),
            version=str(reply.get("version") or ""),
            protocol=str(reply.get("protocol") or ""),
            capabilities=tuple(str(item) for item in reply.get("capabilities") or ()),
        )
        if hello.protocol != PROTOCOL:
            raise RolloutEnvError(
                f"environment speaks {hello.protocol!r}, this driver speaks {PROTOCOL!r}"
            )
        return hello

    def tasks(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self._call("tasks", query=query).get("tasks") or [])

    def reset(self, episode: str, task: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        reply = self._call("reset", episode=episode, task=task, options=options)
        return dict(reply.get("observation") or {})

    def candidates(self, episode: str) -> list[dict[str, Any]]:
        return list(self._call("candidates", episode=episode).get("candidates") or [])

    def label(self, episode: str) -> Labels:
        reply = self._call("label", episode=episode)
        valid = reply.get("valid_ids")
        return Labels(
            valid_ids=None if valid is None else tuple(str(item) for item in valid),
            reason=reply.get("reason"),
        )

    def step(self, episode: str, candidate_id: str) -> StepResult:
        reply = self._call("step", episode=episode, id=candidate_id)
        return StepResult(
            observation=dict(reply.get("observation") or {}),
            applied=bool(reply.get("applied")),
            terminal=reply.get("terminal"),
            reward=float(reply.get("reward") or 0.0),
        )

    def verify(self, episode: str) -> Verification:
        reply = self._call("verify", episode=episode)
        return Verification(
            success=bool(reply.get("success")),
            status=str(reply.get("status") or ""),
            reward=float(reply.get("reward") or 0.0),
            detail=dict(reply.get("detail") or {}),
        )

    def close(self, episode: str) -> None:
        self._call("close", episode=episode)


def serve(environment: Any, *, input_stream: Any = None, output_stream: Any = None) -> int:
    """Serve an in-process environment over JSON lines until EOF.

    The server half of the carrier, so a Python environment can run as a
    separate process and so the wire format has one reference implementation.
    Returns the number of requests answered.
    """
    import sys

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    served = 0
    for raw in input_stream:
        line = raw.strip()
        if not line:
            continue
        try:
            reply = _dispatch(environment, json.loads(line))
        except Exception as exc:  # noqa: BLE001 - every failure is a protocol reply
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        output_stream.write(json.dumps(reply, separators=(",", ":")) + "\n")
        output_stream.flush()
        served += 1
    return served


def _dispatch(environment: Any, request: dict[str, Any]) -> dict[str, Any]:
    op = request.get("op")
    episode = str(request.get("episode") or "")
    if op == "hello":
        hello = environment.hello()
        return {
            "ok": True,
            "protocol": hello.protocol,
            "name": hello.name,
            "version": hello.version,
            "capabilities": list(hello.capabilities),
        }
    if op == "tasks":
        return {"ok": True, "tasks": environment.tasks(dict(request.get("query") or {}))}
    if op == "reset":
        observation = environment.reset(
            episode, dict(request.get("task") or {}), dict(request.get("options") or {})
        )
        return {"ok": True, "observation": observation}
    if op == "candidates":
        return {"ok": True, "candidates": environment.candidates(episode)}
    if op == "label":
        labels = environment.label(episode)
        return {
            "ok": True,
            "valid_ids": None if labels.valid_ids is None else list(labels.valid_ids),
            "reason": labels.reason,
        }
    if op == "step":
        result = environment.step(episode, str(request.get("id") or ""))
        return {
            "ok": True,
            "observation": result.observation,
            "applied": result.applied,
            "terminal": result.terminal,
            "reward": result.reward,
        }
    if op == "verify":
        verdict = environment.verify(episode)
        return {
            "ok": True,
            "success": verdict.success,
            "status": verdict.status,
            "reward": verdict.reward,
            "detail": verdict.detail,
        }
    if op == "close":
        environment.close(episode)
        return {"ok": True}
    return {"ok": False, "error": f"unknown op {op!r}"}
