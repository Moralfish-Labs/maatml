"""``line_world``: the smallest environment that exercises the whole protocol.

An agent stands on an integer line and must reach a target, then finish. It
has a scripted teacher, a step budget, an action that never applies, and a
verifier that ignores what the agent claims. maatml's own tests and docs use
it, so the rollout driver is testable without any real environment.
"""

from __future__ import annotations

from typing import Any

from maatml.registry import register_environment

from .protocol import (
    EXHAUSTED,
    FREE_TEXT_CAPABILITY,
    SUCCESS,
    Hello,
    Labels,
    RolloutEnvError,
    StepResult,
    Verification,
)

MOVES = {"left": -1, "right": 1}


class LineWorld:
    """In-process implementation of :class:`maatml.rollout.Environment`."""

    def __init__(self) -> None:
        self._episodes: dict[str, dict[str, Any]] = {}

    def hello(self) -> Hello:
        return Hello(name="line_world", version="1", capabilities=("label", "tasks"))

    def tasks(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """``count`` seeded tasks; ``slack`` extra steps over the shortest path."""
        import random

        rng = random.Random(int(query.get("seed") or 0))
        slack = int(query.get("slack") or 0)
        span = max(1, int(query.get("span") or 5))
        tasks = []
        for index in range(int(query.get("count") or 1)):
            start, target = rng.randint(-span, span), rng.randint(-span, span)
            tasks.append(
                {
                    "id": f"line-{query.get('seed') or 0}-{index}",
                    "family": "far" if abs(target - start) > span else "near",
                    "start": start,
                    "target": target,
                    "max_steps": abs(target - start) + 1 + slack,
                }
            )
        return tasks

    def _state(self, episode: str) -> dict[str, Any]:
        try:
            return self._episodes[episode]
        except KeyError as exc:
            raise RolloutEnvError(f"unknown episode {episode!r}") from exc

    @staticmethod
    def _observe(state: dict[str, Any]) -> dict[str, Any]:
        return {key: state[key] for key in ("position", "target", "steps", "max_steps")}

    def reset(self, episode: str, task: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        state = {
            "position": int(task["start"]),
            "target": int(task["target"]),
            "steps": 0,
            "max_steps": int(task["max_steps"]) + int(options.get("extra_steps") or 0),
            "terminal": None,
        }
        self._episodes[episode] = state
        return self._observe(state)

    def candidates(self, episode: str) -> list[dict[str, Any]]:
        self._state(episode)
        return [
            {"id": "finish", "valid": True},
            {"id": "jump", "valid": False},
            {"id": "left", "valid": True},
            {"id": "right", "valid": True},
        ]

    def label(self, episode: str) -> Labels:
        state = self._state(episode)
        gap = state["target"] - state["position"]
        if gap == 0:
            return Labels(valid_ids=("finish",))
        return Labels(valid_ids=("right" if gap > 0 else "left",))

    def step(self, episode: str, candidate_id: str) -> StepResult:
        state = self._state(episode)
        if state["terminal"] is not None:
            raise RolloutEnvError(f"episode {episode!r} already ended")
        if candidate_id not in MOVES and candidate_id != "finish":
            return StepResult(observation=self._observe(state), applied=False)
        state["steps"] += 1
        if candidate_id == "finish":
            state["terminal"] = SUCCESS if state["position"] == state["target"] else "wrong_finish"
        else:
            state["position"] += MOVES[candidate_id]
            if state["steps"] >= state["max_steps"]:
                state["terminal"] = EXHAUSTED
        return StepResult(
            observation=self._observe(state),
            applied=True,
            terminal=state["terminal"],
            reward=1.0 if state["terminal"] == SUCCESS else 0.0,
        )

    def verify(self, episode: str) -> Verification:
        state = self._state(episode)
        success = state["terminal"] == SUCCESS and state["position"] == state["target"]
        return Verification(
            success=success,
            status=state["terminal"] or "running",
            reward=1.0 if success else 0.0,
            detail={"steps": state["steps"]},
        )

    def close(self, episode: str) -> None:
        self._episodes.pop(episode, None)


register_environment("line_world")(LineWorld)


class ArithmeticWorld:
    """Minimal free-text environment for testing ``step_text`` and GRPO.

    Each task asks for the exact integer result of ``a + b`` (or ``a * b``),
    submitted either directly or as JSON ``{"answer": <int>}``. Only verified
    correct answers receive reward 1.0; well-formed but wrong answers receive
    0.0 so format compliance alone is never rewarded.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, dict[str, Any]] = {}

    def hello(self) -> Hello:
        return Hello(
            name="arithmetic_world",
            version="1",
            capabilities=(FREE_TEXT_CAPABILITY, "tasks"),
        )

    def tasks(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        import random

        rng = random.Random(int(query.get("seed") or 0))
        split = str(query.get("split") or "train")
        out: list[dict[str, Any]] = []
        for index in range(int(query.get("count") or 4)):
            op = "add" if index % 2 == 0 else "mul"
            a = rng.randint(1, 20)
            b = rng.randint(1, 20)
            answer = a + b if op == "add" else a * b
            sym = "+" if op == "add" else "*"
            out.append(
                {
                    "id": f"arith-{query.get('seed') or 0}-{index}",
                    "family": op,
                    "split": split,
                    "prompt": f"Compute {a} {sym} {b}. Reply with only the integer.",
                    "expected": answer,
                    "max_steps": 1,
                }
            )
        return out

    def _state(self, episode: str) -> dict[str, Any]:
        try:
            return self._episodes[episode]
        except KeyError as exc:
            raise RolloutEnvError(f"unknown episode {episode!r}") from exc

    def reset(self, episode: str, task: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        max_steps = int(task.get("max_steps") or 1) + int(options.get("extra_steps") or 0)
        state = {
            "prompt": str(task["prompt"]),
            "expected": int(task["expected"]),
            "steps": 0,
            "max_steps": max(1, max_steps),
            "terminal": None,
            "verified": False,
            "last_answer": None,
        }
        self._episodes[episode] = state
        return {
            "prompt": state["prompt"],
            "steps": 0,
            "max_steps": state["max_steps"],
        }

    @staticmethod
    def _parse_answer(text: str) -> int | None:
        import json as _json
        import re as _re

        cleaned = text.strip()
        # Strip optional reasoning blocks before reading the final answer.
        cleaned = _re.sub(
            r"<(?:think|thought)>.*?</(?:think|thought)>",
            "",
            cleaned,
            flags=_re.S,
        ).strip()
        if cleaned.startswith("{"):
            try:
                obj = _json.loads(cleaned)
                if isinstance(obj, dict) and isinstance(obj.get("answer"), int):
                    return int(obj["answer"])
            except ValueError:
                return None
        match = _re.fullmatch(r"-?\d+", cleaned)
        return int(match.group(0)) if match else None

    def step_text(
        self,
        episode: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        del metadata
        state = self._state(episode)
        if state["terminal"] is not None:
            raise RolloutEnvError(f"episode {episode!r} already ended")
        state["steps"] += 1
        parsed = self._parse_answer(text)
        state["last_answer"] = parsed
        correct = parsed is not None and parsed == state["expected"]
        if correct:
            state["verified"] = True
            state["terminal"] = SUCCESS
        elif state["steps"] >= state["max_steps"]:
            state["terminal"] = EXHAUSTED
        return StepResult(
            observation={
                "prompt": state["prompt"],
                "steps": state["steps"],
                "max_steps": state["max_steps"],
                "last_answer": parsed,
            },
            applied=True,
            terminal=state["terminal"],
            reward=1.0 if correct else 0.0,
            info={"parsed": parsed, "format_valid": parsed is not None, "correct": correct},
        )

    def verify(self, episode: str) -> Verification:
        state = self._state(episode)
        success = bool(state["verified"] and state["terminal"] == SUCCESS)
        return Verification(
            success=success,
            status=str(state["terminal"] or "running"),
            reward=1.0 if success else 0.0,
            detail={"steps": state["steps"], "last_answer": state["last_answer"]},
        )

    def close(self, episode: str) -> None:
        self._episodes.pop(episode, None)


register_environment("arithmetic_world")(ArithmeticWorld)
