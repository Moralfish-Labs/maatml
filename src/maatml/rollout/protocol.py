"""The rollout environment protocol, ``maatml.rollout_env/1``.

An environment hosts episodes of a decision task. At every state it lists the
candidate actions, applies the one the driver picks, and verifies the outcome
itself: the driver and the learner never compute a reward. An environment may
also *label* a state with the actions a teacher accepts there, which is what
DAgger-style relabeling needs.

Observations, candidates and tasks are JSON objects the driver passes through
without reading, except for a candidate's ``id`` and optional ``valid``. One
environment hosts many episodes at once, each named by the driver.

The same contract has two carriers: an in-process Python object (this module's
:class:`Environment`) and an external process speaking JSON lines over stdio
(:mod:`maatml.rollout.stdio`), one request object per line and one reply per
request::

    {"op": "hello"}
    {"op": "tasks", "query": {...}}
    {"op": "reset", "episode": "e1", "task": {...}, "options": {...}}
    {"op": "candidates", "episode": "e1"}
    {"op": "label", "episode": "e1"}
    {"op": "step", "episode": "e1", "id": "<candidate id>"}
    {"op": "verify", "episode": "e1"}
    {"op": "close", "episode": "e1"}

A reply is ``{"ok": true, ...}`` or ``{"ok": false, "error": "..."}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

PROTOCOL = "maatml.rollout_env/1"

# Terminal statuses every environment shares. An environment may add its own.
SUCCESS = "success"
EXHAUSTED = "exhausted"


class RolloutEnvError(RuntimeError):
    """The environment refused a request or broke the protocol."""


@dataclass(frozen=True)
class Hello:
    """What an environment says about itself."""

    name: str
    version: str
    protocol: str = PROTOCOL
    capabilities: tuple[str, ...] = ()

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True)
class Labels:
    """The teacher's valid set at a state, or why it has none."""

    valid_ids: Optional[tuple[str, ...]]
    reason: Optional[str] = None

    @property
    def labelled(self) -> bool:
        return self.valid_ids is not None


@dataclass(frozen=True)
class StepResult:
    """The outcome of applying one candidate."""

    observation: dict[str, Any]
    applied: bool
    terminal: Optional[str] = None
    reward: float = 0.0


@dataclass(frozen=True)
class Verification:
    """The environment's own verdict on an episode."""

    success: bool
    status: str
    reward: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    """In-process carrier of the protocol."""

    def hello(self) -> Hello: ...

    def tasks(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Tasks for ``query``; each has an ``id`` and is accepted by ``reset``."""
        ...

    def reset(self, episode: str, task: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """Start ``episode`` on ``task`` and return the first observation."""
        ...

    def candidates(self, episode: str) -> list[dict[str, Any]]:
        """Candidates at the current state; each has an ``id``."""
        ...

    def label(self, episode: str) -> Labels: ...

    def step(self, episode: str, candidate_id: str) -> StepResult: ...

    def verify(self, episode: str) -> Verification: ...

    def close(self, episode: str) -> None: ...
