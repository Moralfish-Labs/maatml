"""Decorator-based plugin registries for trainers, validators, metrics, etc.

Plugins can come from:
  - Decorators in in-tree modules (``@register_trainer(...)``)
  - Setuptools entry points group ``maatml.plugins``
  - Per-model ``model.yml`` ``plugins:`` list (module paths or folder-local packages)
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class PluginEntry:
    """A registered callable plus where it came from (for ``maatml plugins``)."""

    name: str
    obj: Any
    source: str  # e.g. "decorator:maatml.training.sft_base" or "entry_point:..."


_MODEL_PLUGIN_NS_RX = re.compile(r"maatml\._model_plugins\.[^.]+\.")


def _comparable_source(source: str) -> str:
    """Source string with the per-folder plugin namespace removed.

    The namespace encodes the model folder's resolved path, so the same plugin
    loaded from two locations (a copy under tmp_path, or a folder loaded both
    directly and by path) has two source strings for identical code. Comparing
    raw sources reported those as collisions. What matters is the plugin module
    itself: ``vision_plugin.export_onnx`` and ``vendor_plugin.export_onnx`` are two
    implementations claiming one name, and stay distinguishable here.
    """
    return _MODEL_PLUGIN_NS_RX.sub("", source)


class _Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, PluginEntry] = {}

    def register(self, name: str, obj: Any, *, source: str = "unknown") -> Any:
        # Allow overwrite so discover_plugins(force=True) / module reload can
        # re-bind the same plugin name after a registry wipe. Re-binding from
        # the same source is that case and stays silent; a *different* source
        # claiming the name is two plugins colliding, where the later one wins
        # and the earlier one becomes unreachable under that name.
        existing = self._entries.get(name)
        if existing is not None and _comparable_source(existing.source) != _comparable_source(
            source
        ):
            warnings.warn(
                f"{self.kind} plugin {name!r} re-registered by {source}, "
                f"replacing {existing.source}. The later registration wins; "
                "give them distinct names if both are meant to coexist.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._entries[name] = PluginEntry(name=name, obj=obj, source=source)
        return obj

    def get(self, name: str, default: Any = None) -> Any:
        entry = self._entries.get(name)
        return entry.obj if entry is not None else default

    def require(self, name: str) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            known = ", ".join(sorted(self._entries)) or "(none)"
            message = f"Unknown {self.kind} plugin {name!r}. Known: {known}"
            if _load_errors:
                # A plugin that failed to import is the usual reason a name is
                # missing; saying so beats "Known: (none)".
                failed = "; ".join(f"{src}: {err}" for src, err in _load_errors)
                message += f". Plugin sources that failed to load: {failed}"
            raise KeyError(message)
        return entry.obj

    def names(self) -> list[str]:
        return sorted(self._entries)

    def items(self) -> list[PluginEntry]:
        return [self._entries[k] for k in sorted(self._entries)]

    def unregister(self, name: str) -> bool:
        """Drop one registration. Returns True when something was removed."""
        return self._entries.pop(name, None) is not None

    def snapshot(self) -> dict[str, PluginEntry]:
        """Copy of the current registrations (restore with :meth:`restore`)."""
        return dict(self._entries)

    def restore(self, entries: dict[str, PluginEntry]) -> None:
        """Replace registrations with a previous :meth:`snapshot`."""
        self._entries.clear()
        self._entries.update(entries)

    def clear(self) -> None:
        """Wipe registrations (tests, and ``discover_plugins(force=True)``)."""
        self._entries.clear()


TRAINERS = _Registry("trainer")
VALIDATORS = _Registry("validator")
METRICS = _Registry("metrics")
PREDICTORS = _Registry("predictor")
FORMATS = _Registry("format")
SCAFFOLD_HOOKS = _Registry("scaffold_hook")
SANITIZERS = _Registry("sanitizer")
TRANSFORMS = _Registry("transform")
EXPORTERS = _Registry("exporter")
GENERATORS = _Registry("generator")
# Target compilers turn a portable export (ONNX, GGUF, HF) into a device-
# specific runtime artifact. Servers launch a long-lived inference process for
# that artifact. Both are format-agnostic: plugins decide TensorRT / vLLM /
# llama.cpp / HTTP, core only dispatches.
COMPILERS = _Registry("compiler")
SERVERS = _Registry("server")
# Rollout environments: a factory returning a `maatml.rollout.Environment`.
ENVIRONMENTS = _Registry("environment")

_ALL_REGISTRIES: dict[str, _Registry] = {
    "trainer": TRAINERS,
    "validator": VALIDATORS,
    "metrics": METRICS,
    "predictor": PREDICTORS,
    "format": FORMATS,
    "scaffold_hook": SCAFFOLD_HOOKS,
    "sanitizer": SANITIZERS,
    "transform": TRANSFORMS,
    "exporter": EXPORTERS,
    "generator": GENERATORS,
    "compiler": COMPILERS,
    "server": SERVERS,
    "environment": ENVIRONMENTS,
}

_discovered = False
# (source, error) for every plugin module / entry point that failed to load.
# Surfaced by `maatml plugins` and by "Unknown … plugin" errors instead of
# being swallowed, so a broken plugin looks broken rather than absent.
_load_errors: list[tuple[str, str]] = []
# Model-folder plugins already executed, keyed by generated module name: a
# model folder is executable code and must run once per process, not once per
# command that happens to load the model definition.
_loaded_model_plugins: dict[str, str] = {}


def load_errors() -> list[tuple[str, str]]:
    """Plugin sources that failed to import, as ``(source, error)`` pairs."""
    return list(_load_errors)


def restore_load_errors(entries: Iterable[tuple[str, str]] = ()) -> None:
    """Replace the recorded load errors (``()`` clears them).

    Public because the list is process-global: a test that provokes a plugin
    failure would otherwise leave it visible to every later `maatml audit` /
    `Unknown … plugin` message in the same process.
    """
    _load_errors.clear()
    _load_errors.extend(entries)


def _record_load_error(source: str, exc: BaseException) -> None:
    entry = (source, f"{type(exc).__name__}: {exc}")
    if entry not in _load_errors:
        _load_errors.append(entry)


def _caller_source(extra: str = "") -> str:
    """Best-effort module name of the registering caller."""
    import inspect

    frame = inspect.currentframe()
    try:
        # decorator helper -> register_* -> caller module
        caller = frame.f_back.f_back if frame and frame.f_back else None
        mod = caller.f_globals.get("__name__", "unknown") if caller else "unknown"
    finally:
        del frame
    return f"decorator:{mod}" + (f":{extra}" if extra else "")


def _make_decorator(registry: _Registry) -> Callable[[str], Callable[[F], F]]:
    def decorator(name: str) -> Callable[[F], F]:
        def wrapper(fn: F) -> F:
            registry.register(name, fn, source=_caller_source(name))
            return fn

        return wrapper

    return decorator


register_trainer = _make_decorator(TRAINERS)
register_validator = _make_decorator(VALIDATORS)
register_metrics = _make_decorator(METRICS)
register_predictor = _make_decorator(PREDICTORS)
register_format = _make_decorator(FORMATS)
register_scaffold_hook = _make_decorator(SCAFFOLD_HOOKS)
register_sanitizer = _make_decorator(SANITIZERS)
register_transform = _make_decorator(TRANSFORMS)
register_exporter = _make_decorator(EXPORTERS)
register_generator = _make_decorator(GENERATORS)
register_compiler = _make_decorator(COMPILERS)
register_server = _make_decorator(SERVERS)
register_environment = _make_decorator(ENVIRONMENTS)


def get_registry(kind: str) -> _Registry:
    try:
        return _ALL_REGISTRIES[kind]
    except KeyError as exc:
        raise KeyError(
            f"Unknown registry kind {kind!r}. Known: {', '.join(_ALL_REGISTRIES)}"
        ) from exc


def list_all_plugins() -> dict[str, list[PluginEntry]]:
    """Return every registered plugin grouped by registry kind."""
    return {kind: reg.items() for kind, reg in _ALL_REGISTRIES.items()}


def snapshot_registries() -> dict[str, dict[str, PluginEntry]]:
    """Snapshot every registry (public API for tests and embedding hosts)."""
    return {kind: reg.snapshot() for kind, reg in _ALL_REGISTRIES.items()}


def restore_registries(snapshot: dict[str, dict[str, PluginEntry]]) -> None:
    """Restore a :func:`snapshot_registries` result."""
    for kind, entries in snapshot.items():
        _ALL_REGISTRIES[kind].restore(entries)


def reset_registries(*, rediscover: bool = False) -> None:
    """Wipe every registry: the blank slate for tests and embedding hosts.

    Also forgets which model-folder plugins have run, so a later
    :func:`load_model_plugins` re-executes them instead of assuming their
    registrations are still there. With ``rediscover``, re-imports the
    built-ins afterwards.
    """
    global _discovered
    for reg in _ALL_REGISTRIES.values():
        reg.clear()
    _loaded_model_plugins.clear()
    _load_errors.clear()
    _discovered = False
    if rediscover:
        discover_plugins(force=True)


def _load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_package_from_dir(pkg_dir: Path, module_name: str) -> Any:
    """Load a directory containing ``__init__.py`` as a package.

    Registers the package in ``sys.modules`` with
    ``submodule_search_locations`` so intra-package imports
    (``from .validator import ...``) resolve correctly.
    """
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise ImportError(f"Plugin package missing __init__.py: {pkg_dir}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin package from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def looks_like_plugin_path(entry: str, base_dir: Optional[Path] = None) -> bool:
    """Is this ``plugins:`` entry a filesystem path rather than a module name?

    Testing only for ``/`` sent every Windows path (``C:\\models\\my_plugin``,
    ``plugins\\hook.py``) down the ``import_module`` branch, where it failed as
    an unknown module. Separators are checked for both platforms, and a bare
    name that exists next to the model folder is treated as a path too.
    """
    if entry.endswith(".py") or entry.startswith("."):
        return True
    if "/" in entry or "\\" in entry or (os.altsep and os.altsep in entry):
        return True
    if os.path.splitdrive(entry)[0]:
        return True
    return bool(base_dir is not None and (Path(base_dir) / entry).exists())


def load_model_plugins(
    model_dir: str | Path,
    plugin_list: Optional[Iterable[str]] = None,
    *,
    force: bool = False,
) -> list[str]:
    """Load plugins declared in a model's ``plugins:`` list. Idempotent.

    Each entry is either:
      - a dotted module path (``my_pkg.plugins.foo``),
      - a model-folder-relative ``.py`` file (``plugins/local_hook.py``), or
      - a model-folder-relative package directory (``./triage_plugin``) containing
        ``__init__.py``.

    Both ``load_model_def`` and the CLI ask for a model's plugins, so this is
    the single owner of that work: an entry already executed in this process
    is not run again (``force=True`` re-executes it). Returns the list of
    loaded module names, whether or not this call was the one that ran them.
    """
    model_dir = Path(model_dir).resolve()
    loaded: list[str] = []
    namespace = _model_plugin_namespace(model_dir)
    for entry in plugin_list or []:
        entry = entry.strip()
        if not entry:
            continue
        if looks_like_plugin_path(entry, model_dir):
            path = Path(entry)
            if not path.is_absolute():
                path = (model_dir / entry).resolve()
            if path.is_dir():
                mod_name = f"maatml._model_plugins.{namespace}.{path.name}"
                if force or not _already_loaded(mod_name, path):
                    _load_package_from_dir(path, mod_name)
                    _loaded_model_plugins[mod_name] = str(path)
                loaded.append(mod_name)
            elif path.is_file():
                mod_name = f"maatml._model_plugins.{namespace}.{path.stem}"
                if force or not _already_loaded(mod_name, path):
                    _load_module_from_path(path, mod_name)
                    _loaded_model_plugins[mod_name] = str(path)
                loaded.append(mod_name)
            else:
                raise FileNotFoundError(f"Model plugin not found: {path} (from plugins: {entry!r})")
        else:
            if force or entry not in sys.modules:
                importlib.import_module(entry)
            loaded.append(entry)
    return loaded


def _model_plugin_namespace(model_dir: Path) -> str:
    """Module namespace for a model folder, unique per resolved path.

    Keyed on the basename alone, two folders of the same name (``~/models/triage``
    and ``./examples/triage`` is a normal layout) shared one module name, so
    loading the second replaced the first in ``sys.modules`` and the registry
    kept whichever ran last. Folding the resolved path into the name keeps them
    distinct while staying stable across runs.
    """
    digest = hashlib.sha256(str(model_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r"\W+", "_", model_dir.name).strip("_") or "model"
    return f"{safe}_{digest}"


def _already_loaded(mod_name: str, path: Path) -> bool:
    """Has this exact plugin path already been executed as ``mod_name``?"""
    return _loaded_model_plugins.get(mod_name) == str(path) and mod_name in sys.modules


_BUILTIN_PLUGIN_MODULES = (
    "maatml.training.builtins",
    "maatml.data.pipeline",
    "maatml.data.formats",
    "maatml.data.image_folder",
    "maatml.data.preference",
    "maatml.evaluation.predictors",
    "maatml.export.bundle",
    "maatml.export.gguf",
    "maatml.export.mlx_export",
    "maatml.servers",
    "maatml.rollout.example",
)


def discover_plugins(*, force: bool = False) -> None:
    """Import built-in modules + entry-point plugins.

    Idempotent unless ``force=True``, which re-imports the built-in modules
    and entry points. Discovery never wipes registrations: model-folder
    plugins register through :func:`load_model_plugins`, and clearing here
    threw away whatever a model folder had just registered. The old
    "rediscover when the trainer registry looks empty" heuristic could not
    tell "wiped by a test" from "nothing registered yet"; tests that want a
    blank slate call :func:`reset_registries`.
    """
    global _discovered
    if _discovered and not force:
        return

    if force:
        _load_errors.clear()

    for mod in _BUILTIN_PLUGIN_MODULES:
        try:
            # Reaching here means the registries still need populating (first
            # call, after reset_registries, or force): a cached module import
            # would be a no-op, so reload to re-run its @register_* decorators.
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
            else:
                importlib.import_module(mod)
        except ImportError as exc:
            # Optional extras (gguf / mlx / torch) legitimately miss; record so
            # `maatml plugins` can say why a format is unavailable.
            _record_load_error(f"module:{mod}", exc)

    # Setuptools / importlib entry points, one failure at a time so a single
    # broken third-party plugin cannot hide every other one.
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        selected: Any
        if hasattr(eps, "select"):
            selected = eps.select(group="maatml.plugins")
        else:
            selected = eps["maatml.plugins"]
    except Exception as exc:  # noqa: BLE001  discovery must not abort CLI startup
        _record_load_error("entry_points:maatml.plugins", exc)
        selected = []

    for ep in selected:
        try:
            loaded = ep.load()
            if callable(loaded) and not isinstance(loaded, type):
                with contextlib.suppress(TypeError):
                    loaded()
        except Exception as exc:  # noqa: BLE001
            _record_load_error(f"entry_point:{getattr(ep, 'name', ep)}", exc)

    _discovered = True
