"""``line_world``: the smallest environment that exercises the whole protocol.

An agent stands on an integer line and must reach a target, then finish. It
has a scripted teacher, a step budget, an action that never applies, and a
verifier that ignores what the agent claims. maatml's own tests and docs use
it, so the rollout driver is testable without any real environment.
"""

from __future__ import annotations

from typing import Any

from maatml.registry import register_environment

from .protocol import EXHAUSTED, SUCCESS, Hello, Labels, RolloutEnvError, StepResult, Verification

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
