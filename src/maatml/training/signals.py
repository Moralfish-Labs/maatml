"""Process signal handling for clean trainer interruption and checkpoint flush."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class TrainingInterrupted(RuntimeError):
    """Raised when a training run is stopped by a process signal (e.g. SIGTERM)."""

    def __init__(self, signum: int, signal_name: Optional[str] = None) -> None:
        self.signum = int(signum)
        self.signal_name = signal_name or _signal_name(signum)
        super().__init__(f"Interrupted by {self.signal_name} ({self.signum})")


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


class TrainingStopController:
    """Coordinate SIGTERM/SIGINT handling across a training run.

    Outside ``training_loop()``, ``SIGTERM`` raises :class:`TrainingInterrupted`
    immediately so pre-train setup aborts instead of hanging until a timeout.
    Inside ``training_loop()`` (around ``trainer.train()``), the first signal
    records a stop request so a ``TrainerCallback`` can trigger one checkpoint
    flush and stop cleanly at the step boundary; a second signal interrupts
    immediately.
    """

    def __init__(self) -> None:
        self.stop_requested: bool = False
        self.signum: Optional[int] = None
        self.signal_name: Optional[str] = None
        self._in_train_loop: bool = False

    def _record_signal(self, signum: int) -> None:
        self.stop_requested = True
        self.signum = int(signum)
        self.signal_name = _signal_name(self.signum)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        graceful = self._in_train_loop and not self.stop_requested
        self._record_signal(signum)
        if not graceful:
            raise TrainingInterrupted(self.signum or signum, self.signal_name)

    def raise_if_interrupted(self) -> None:
        if self.stop_requested:
            sig = self.signum if self.signum is not None else getattr(signal, "SIGTERM", 15)
            raise TrainingInterrupted(sig, self.signal_name)

    @contextmanager
    def installed(self) -> Iterator[TrainingStopController]:
        """Install ``SIGTERM`` handler for the duration of a training run."""
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        sigterm = getattr(signal, "SIGTERM", None)
        if sigterm is None:
            yield self
            return
        previous = signal.getsignal(sigterm)
        signal.signal(sigterm, self._handle_signal)
        try:
            yield self
        finally:
            signal.signal(sigterm, previous)

    @contextmanager
    def training_loop(self) -> Iterator[TrainingStopController]:
        """Enable deferred checkpoint-flushing stop for ``SIGTERM`` and ``SIGINT``."""
        was_in_loop = self._in_train_loop
        self._in_train_loop = True
        previous_int: Any = None
        installed_int = False
        sigint = getattr(signal, "SIGINT", None)
        if sigint is not None and threading.current_thread() is threading.main_thread():
            previous_int = signal.getsignal(sigint)
            signal.signal(sigint, self._handle_signal)
            installed_int = True
        try:
            yield self
        finally:
            if installed_int and sigint is not None:
                signal.signal(sigint, previous_int)
            self._in_train_loop = was_in_loop


def make_signal_stop_callback(controller: TrainingStopController) -> Any:
    """Return a transformers ``TrainerCallback`` that flushes and stops on signal."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class _SignalStopCallback(TrainerCallback):
        def _maybe_stop(self, control: TrainerControl) -> TrainerControl:
            if controller.stop_requested:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_substep_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> TrainerControl:
            del args, state, kwargs
            return self._maybe_stop(control)

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> TrainerControl:
            del args, state, kwargs
            return self._maybe_stop(control)

    return _SignalStopCallback()
