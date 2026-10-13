"""Populations beyond one group key: isolation hierarchy, pins, benchmark version, blind.

``dataset.group_by`` keeps correlated rows on one side of a split. It cannot
say *which* side a camera lands on, nor that a site is absent from training
altogether. ``isolation`` names the hierarchy a row carries (fine to coarse)
and the level each population must be disjoint from training at; ``pins``
assign groups; a ``blind`` manifest never enters train, val or the benchmark
and is evaluated once per frozen candidate.

```yaml
dataset:
  isolation:
    fields: [clip, camera, site]            # fine -> coarse, all row fields
    policy: {val: camera, benchmark: camera, blind: site}
  pins:
    val: ["camera:G339"]
    benchmark: ["camera:G341", "camera:G421"]
  blind_samples: datasets/samples/blind_v001.jsonl
```
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..utils.io import sha256_file, write_json_atomic

POPULATIONS = ("train", "val", "benchmark", "blind")
BENCHMARK_STATE = "benchmark.json"
BENCHMARK_KIND = "maatml.benchmark/1"


class PopulationError(ValueError):
    """A population violates the declared isolation, or a pin names nothing."""


@dataclass(frozen=True)
class Isolation:
    fields: tuple[str, ...]
    policy: dict[str, str] = field(default_factory=dict)

    def level_for(self, population: str) -> Optional[str]:
        """The field a population must be disjoint from training at, if any."""
        return self.policy.get(population)


@dataclass(frozen=True)
class Pin:
    population: str
    field_name: str
    value: str

    @property
    def label(self) -> str:
        return f"{self.field_name}:{self.value}"

    def matches(self, row: dict[str, Any]) -> bool:
        raw = row.get(self.field_name)
        return raw is not None and str(raw) == self.value


def resolve_isolation(cfg: dict[str, Any]) -> Optional[Isolation]:
    spec = cfg.get("isolation")
    if spec is None:
        return None
    fields: Any
    policy: Any
    if isinstance(spec, list):
        fields, policy = spec, {}
    elif isinstance(spec, dict):
        fields, policy = spec.get("fields"), spec.get("policy") or {}
    else:
        raise PopulationError("dataset.isolation must be a list of fields or {fields, policy}")
    if not isinstance(fields, list) or not fields or not all(isinstance(f, str) for f in fields):
        raise PopulationError("dataset.isolation.fields must be a non-empty list of row fields")
    if not isinstance(policy, dict):
        raise PopulationError("dataset.isolation.policy must map populations to a field")
    for population, level in policy.items():
        if population not in POPULATIONS or population == "train":
            raise PopulationError(
                f"dataset.isolation.policy names {population!r}; "
                f"populations are {[p for p in POPULATIONS if p != 'train']}"
            )
        if level not in fields:
            raise PopulationError(
                f"dataset.isolation.policy.{population}={level!r} is not one of fields {fields}"
            )
    return Isolation(fields=tuple(fields), policy={str(k): str(v) for k, v in policy.items()})


def resolve_pins(cfg: dict[str, Any], isolation: Optional[Isolation]) -> list[Pin]:
    spec = cfg.get("pins")
    if spec is None:
        return []
    if not isinstance(spec, dict):
        raise PopulationError("dataset.pins must map val / benchmark to a list of field:value")
    pins: list[Pin] = []
    for population, entries in spec.items():
        if population not in ("val", "benchmark"):
            raise PopulationError(
                f"dataset.pins.{population}: only val and benchmark can be pinned"
            )
        if not isinstance(entries, list):
            raise PopulationError(f"dataset.pins.{population} must be a list of field:value")
        for entry in entries:
            if not isinstance(entry, str) or ":" not in entry:
                raise PopulationError(
                    f"dataset.pins.{population} entry {entry!r} must be field:value"
                )
            field_name, value = entry.split(":", 1)
            if isolation is not None and field_name not in isolation.fields:
                raise PopulationError(
                    f"dataset.pins.{population} {entry!r}: {field_name!r} is not an "
                    f"isolation field {list(isolation.fields)}"
                )
            pins.append(Pin(population=population, field_name=field_name, value=value))
    seen: dict[str, str] = {}
    for pin in pins:
        if pin.label in seen and seen[pin.label] != pin.population:
            raise PopulationError(
                f"pin {pin.label} is assigned to both {seen[pin.label]} and {pin.population}"
            )
        seen[pin.label] = pin.population
    return pins


def apply_pins(rows_by_split: dict[str, list[dict]], pins: Iterable[Pin]) -> dict[str, int]:
    """Move every row matching a pin into its population; returns moved counts.

    ``rows_by_split`` keys are ``train`` / ``val`` / ``test`` (the benchmark
    population is the test split). A pin that matches nothing is an error: a
    typo must not silently pin nothing.
    """
    moved: dict[str, int] = {}
    for pin in pins:
        target = "test" if pin.population == "benchmark" else pin.population
        for split in list(rows_by_split):
            if split == target:
                continue
            keep: list[dict] = []
            for row in rows_by_split[split]:
                if pin.matches(row):
                    tagged = dict(row)
                    tagged["split"] = target
                    rows_by_split.setdefault(target, []).append(tagged)
                else:
                    keep.append(row)
            rows_by_split[split] = keep
        pinned = 0
        for row in rows_by_split.get(target, []):
            if pin.matches(row):
                row["split"] = target
                pinned += 1
        if not pinned:
            raise PopulationError(f"dataset.pins.{pin.population} {pin.label} matches no row")
        moved[pin.label] = pinned
    return moved


def _values(rows: Iterable[dict[str, Any]], field_name: str) -> set[str]:
    return {str(row[field_name]) for row in rows if row.get(field_name) is not None}


def check_isolation(
    populations: dict[str, list[dict[str, Any]]], isolation: Isolation
) -> list[str]:
    """Violations of the declared policy: a level value shared with training,
    or between two held-out populations."""
    train = populations.get("train", [])
    problems: list[str] = []
    held = [p for p in ("val", "benchmark", "blind") if populations.get(p)]
    for population in held:
        level = isolation.level_for(population)
        if level is None:
            continue
        rows = populations[population]
        missing = sum(1 for row in rows if row.get(level) is None)
        if missing:
            problems.append(
                f"{population}: {missing} row(s) lack isolation field {level!r}; "
                "every row must name the level it is isolated at"
            )
        shared = _values(rows, level) & _values(train, level)
        if shared:
            problems.append(
                f"{population} is {level}-disjoint from train by policy but shares "
                f"{level} {sorted(shared)[:5]}"
            )
        for other in held:
            if other <= population:
                continue
            other_level = isolation.level_for(other)
            if other_level is None:
                continue
            coarse = (
                level
                if isolation.fields.index(level) >= isolation.fields.index(other_level)
                else other_level
            )
            overlap = _values(rows, coarse) & _values(populations[other], coarse)
            if overlap:
                problems.append(f"{population} and {other} share {coarse} {sorted(overlap)[:5]}")
    return problems


def _row_identity(row: dict[str, Any]) -> str:
    payload = {k: v for k, v in row.items() if k != "split"}
    return json.dumps(payload, sort_keys=True, default=str)


def benchmark_version(rows: Iterable[dict[str, Any]], pins: Iterable[Pin]) -> str:
    """Order-insensitive content hash of the benchmark rows plus the pins that built it."""
    h = hashlib.sha256()
    for identity in sorted(_row_identity(row) for row in rows):
        h.update(identity.encode("utf-8"))
        h.update(b"\n")
    for pin in sorted((p.population, p.label) for p in pins):
        h.update(f"pin {pin[0]} {pin[1]}".encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def benchmark_state_path(prepared_dir: Path) -> Path:
    return Path(prepared_dir) / BENCHMARK_STATE


def read_benchmark_state(prepared_dir: Path) -> Optional[dict[str, Any]]:
    path = benchmark_state_path(prepared_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) and payload.get("kind") == BENCHMARK_KIND else None


def refuse_in_place_benchmark_change(prepared_dir: Path, benchmark_path: Optional[Path]) -> None:
    """A benchmark file edited under the same name is a silent population change.

    Benchmarks grow by version: write the new rows to a new file and point
    ``dataset.benchmark_samples`` at it. The previous state stays readable so
    floors derived on the old version can still be traced.
    """
    if benchmark_path is None:
        return
    previous = read_benchmark_state(prepared_dir)
    if previous is None:
        return
    same_path = str(previous.get("benchmark_samples")) == str(benchmark_path)
    if same_path and previous.get("file_sha256") != sha256_file(benchmark_path):
        raise PopulationError(
            f"{benchmark_path.name} changed in place since it was last prepared "
            f"(version {str(previous.get('version'))[:16]}). Benchmarks are versioned: "
            "write the new rows to a new file (for example a _v002 suffix) and point "
            "dataset.benchmark_samples at it, so floors keep naming the population "
            "they were derived on."
        )


def write_benchmark_state(
    prepared_dir: Path,
    *,
    version: str,
    n: int,
    benchmark_path: Optional[Path],
    pins: Iterable[Pin],
    isolation: Optional[Isolation],
) -> Path:
    payload = {
        "kind": BENCHMARK_KIND,
        "version": version,
        "n": n,
        "benchmark_samples": str(benchmark_path) if benchmark_path else None,
        "file_sha256": sha256_file(benchmark_path) if benchmark_path else None,
        "pins": sorted(f"{p.population}={p.label}" for p in pins),
        "isolation": (
            {"fields": list(isolation.fields), "policy": dict(isolation.policy)}
            if isolation
            else None
        ),
    }
    return write_json_atomic(benchmark_state_path(prepared_dir), payload)


def check_prepared_isolation(model_def: Any) -> list[str]:
    """Re-run the isolation check over the prepared splits on disk (for ``audit``)."""
    from ..config import get_dataset_cfg
    from ..utils.io import iter_jsonl

    cfg = get_dataset_cfg(model_def)
    isolation = resolve_isolation(cfg)
    if isolation is None:
        return []
    prepared = Path(model_def.prepared_dir)
    populations: dict[str, list[dict[str, Any]]] = {}
    for split, population in (("train", "train"), ("val", "val"), ("test", "benchmark")):
        path = prepared / f"{split}.jsonl"
        if path.is_file():
            populations[population] = list(iter_jsonl(path))
    blind = cfg.get("blind_samples")
    if blind:
        blind_path = model_def.resolve(blind)
        if blind_path.is_file():
            populations["blind"] = list(iter_jsonl(blind_path))
    if not populations:
        return []
    return check_isolation(populations, isolation)
