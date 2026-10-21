"""Data preparation: load seed corpora, split into train/val/test JSONLs.

The generic ``prepare`` entry point reads knobs from ``dataset:`` (falling
back to ``data:``), optionally sanitizes by tag (via the ``SANITIZERS``
registry), splits by group key (``dataset.group_by`` when set, else
``family`` → ``source`` → ``sample_id``), pins benchmarks to test, and
writes splits + a dataset card.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from rich.console import Console

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import SANITIZERS, register_format
from ..utils.chat import validate_chat_row
from ..utils.io import iter_jsonl, stable_hash, write_jsonl
from .attribution import (
    AttributionError,
    check_sources,
    read_attribution,
    resolve_attribution,
    resolve_attribution_field,
    write_corpus_lock,
)
from .populations import (
    apply_pins,
    benchmark_version,
    check_isolation,
    refuse_in_place_benchmark_change,
    resolve_isolation,
    resolve_pins,
    write_benchmark_state,
)
from .schemas import Split

console = Console()


def _split_from_hash(key: str, ratios: tuple[float, float, float]) -> Split:
    digest = stable_hash(key)
    bucket = int(digest[:12], 16) / float(1 << 48)
    train_r, val_r, _ = ratios
    if bucket < train_r:
        return Split.train
    if bucket < train_r + val_r:
        return Split.val
    return Split.test


_warned_group_fallback: set[str] = set()


def _group_key(row: dict, group_by: Optional[str] = None) -> str:
    """Stable group key for leakage-safe splitting.

    When ``dataset.group_by`` is set (e.g. ``family``), that field is tried
    first. Otherwise, or if the preferred field is missing, fall back to
    ``family`` → ``source`` → ``sample_id``. Falling back past ``family``
    emits a one-shot warning so operators notice missing family tags.
    """
    preferred = (group_by or "").strip() or None
    if preferred:
        val = row.get(preferred)
        if isinstance(val, str) and val.strip():
            return f"{preferred}:{val.strip()}"
        if val is not None and not isinstance(val, str):
            return f"{preferred}:{val}"

    family = row.get("family")
    if isinstance(family, str) and family.strip():
        return f"family:{family.strip()}"
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        key = f"source:{source.strip()}"
        if key not in _warned_group_fallback:
            _warned_group_fallback.add(key)
            warnings.warn(
                f"samples with source={source!r} have no family; splitting by source "
                f"(re-run seed builders to stamp family)",
                stacklevel=3,
            )
        return key
    sid = str(row.get("sample_id") or "")
    key = f"sample_id:{sid}"
    if "sample_id" not in _warned_group_fallback:
        _warned_group_fallback.add("sample_id")
        warnings.warn(
            "samples missing family/source; splitting by sample_id",
            stacklevel=3,
        )
    return key


def _write_split(rows: list[dict], out_dir: Path, split: Split) -> Path:
    return write_jsonl(out_dir / f"{split.value}.jsonl", rows)


def _write_dataset_card(
    out_dir: Path,
    *,
    title: str,
    counts: dict[str, int],
    split_counts: dict[str, int],
    extra_lines: Iterable[str] = (),
) -> Path:
    lines = [f"# {title}", "", "## Counts by split", ""]
    for split in (Split.train, Split.val, Split.test):
        lines.append(f"- {split.value}: {split_counts.get(split.value, 0)}")
    lines.extend(["", "## Counts by label", ""])
    for label, n in sorted(counts.items()):
        lines.append(f"- {label}: {n}")
    lines.extend(["", *extra_lines])
    path = out_dir / "dataset_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _resolve_ratios(cfg: dict) -> tuple[float, float, float]:
    ratios = tuple(cfg.get("split_ratios", [0.8, 0.1, 0.1]))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0; got {ratios}")
    return ratios


def _apply_sanitize(row: dict, sanitize_tags: list[str], request_field: str) -> dict:
    if not sanitize_tags:
        return row
    out = dict(row)
    text = out.get(request_field)
    if not isinstance(text, str):
        return out
    for tag in sanitize_tags:
        fn = SANITIZERS.get(tag)
        if fn is None:
            raise ValueError(
                f"Unknown sanitize tag {tag!r}; known: {SANITIZERS.names()} "
                f"(register via @register_sanitizer or model.yml plugins:)"
            )
        text = fn(text)
    out[request_field] = text
    return out


# A group holding at least this share of the corpus cannot be split: hashing
# it once sends (nearly) every row to the same split, leaving val/test empty.
_DEGENERATE_GROUP_SHARE = 0.9


def _row_identity(row: dict) -> str:
    """Per-row group key: ``sample_id`` when present, else a content hash."""
    sid = row.get("sample_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    if sid is not None and not isinstance(sid, str):
        return str(sid)
    return stable_hash(json.dumps(row, sort_keys=True, default=str))[:16]


def _warn_degenerate_group(dominant_key: str, members: int, total: int) -> None:
    message = (
        f"group key {dominant_key!r} covers {members}/{total} rows: "
        "splitting those rows individually so val/test are not empty. "
        "Group-level leakage protection does NOT apply to them; stamp a "
        "real family/group_by field (seed builders do) to restore it."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=4)
    console.print(f"[yellow]warning[/] {message}")


def _group_rows(
    rows: list[dict], *, group_by: Optional[str] = None
) -> tuple[dict[str, list[dict]], Optional[str]]:
    """Group rows by key, falling back to per-row keys for a degenerate group.

    Corpora produced by ``maatml datagen`` / ``maatml ingest`` share one
    ``source`` and carry no ``family``, so every row hashed to the same group
    and the whole corpus landed in a single split (empty val/test, a green
    prepare, and nothing to evaluate on). When one group covers ~the whole
    corpus its members are split per row instead, loudly.

    Returns ``(groups, degenerate_key)``.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_by=group_by)].append(row)
    if len(rows) < 2:
        return groups, None

    dominant_key, members = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(members) < 2 or len(members) / len(rows) < _DEGENERATE_GROUP_SHARE:
        return groups, None

    _warn_degenerate_group(dominant_key, len(members), len(rows))
    del groups[dominant_key]
    for row in members:
        groups[f"{dominant_key}#row:{_row_identity(row)}"].append(row)
    return groups, dominant_key


def _assign_group_splits(
    rows: list[dict],
    ratios: tuple[float, float, float],
    *,
    group_by: Optional[str] = None,
) -> tuple[dict[Split, list[dict]], dict[str, Split], Optional[str]]:
    """Hash each group once; every member inherits that split.

    Returns ``(by_split, group_assignment, degenerate_key)`` so the caller can
    check benchmark rows against the split each group landed in.
    """
    groups, degenerate_key = _group_rows(rows, group_by=group_by)

    by_split: dict[Split, list[dict]] = {
        Split.train: [],
        Split.val: [],
        Split.test: [],
    }
    assignment: dict[str, Split] = {}
    for gkey, members in groups.items():
        split = _split_from_hash(gkey, ratios)
        assignment[gkey] = split
        for row in members:
            tagged = dict(row)
            tagged["split"] = split.value
            by_split[split].append(tagged)
    return by_split, assignment, degenerate_key


def _warn_on_empty_splits(by_split: dict[Split, list[dict]], *, n_groups: int) -> None:
    """Say so when a split came out empty instead of reporting a clean prepare.

    Whole-group hashing with only a handful of groups can leave val (or test)
    with nothing in it. Training then skips evaluation entirely and still
    reports success, so the operator needs to hear about it here.
    """
    _warn_on_empty_counts({split: len(rows) for split, rows in by_split.items()}, n_groups=n_groups)


def _warn_on_empty_counts(counts: dict[Split, int], *, n_groups: int) -> None:
    """``_warn_on_empty_splits`` over per-split row counts."""
    empty = [split.value for split, n in counts.items() if not n]
    if not empty or not any(counts.values()):
        return
    message = (
        f"split(s) {', '.join(sorted(empty))} are empty: {n_groups} group(s) "
        "cannot fill the configured split_ratios. Add more groups (family / "
        "dataset.group_by values) or widen split_ratios; training will skip "
        "evaluation on an empty val split."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    console.print(f"[yellow]warning[/] {message}")


_IDENTITY_FIELDS = frozenset({"sample_id", "family", "source"})


def _content_key(row: dict, request_field: Optional[str]) -> Optional[str]:
    """Stable key for exact-duplicate detection across corpora.

    Prefers the configured request field, else the row with its identity fields
    stripped, so a row that was merely re-tagged still collides with its twin.
    """
    if request_field:
        value = row.get(request_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = {k: v for k, v in row.items() if k not in _IDENTITY_FIELDS}
    if not payload:
        return None
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def _check_benchmark_leakage(
    benchmark_rows: list[dict],
    group_assignment: dict[str, Split],
    *,
    group_by: Optional[str] = None,
    seed_rows: Optional[list[dict]] = None,
    request_field: Optional[str] = None,
) -> None:
    """Refuse to pin a benchmark row that the seed corpus already contains.

    Two checks, because they catch different mistakes. The content check
    compares the rendered request (or the row minus its identity fields), so a
    benchmark drawn from the same generator as the seeds is caught even when it
    carries its own family namespace: re-tagging renames a duplicate, it does
    not prevent one. The group check then refuses a benchmark row whose group
    key already trains, since benchmark rows are pinned to test unconditionally
    and the same family would sit on both sides of the split.
    """
    if seed_rows:
        seed_content: dict[str, Any] = {}
        for row in seed_rows:
            key = _content_key(row, request_field)
            if key is not None:
                seed_content.setdefault(key, row.get("sample_id"))
        duplicates: list[tuple[Any, Any]] = []
        for row in benchmark_rows:
            key = _content_key(row, request_field)
            if key is not None and key in seed_content:
                duplicates.append((row.get("sample_id"), seed_content[key]))
        if duplicates:
            preview = ", ".join(f"{b} == {s}" for b, s in duplicates[:5])
            more = " ..." if len(duplicates) > 5 else ""
            raise ValueError(
                f"{len(duplicates)} benchmark row(s) duplicate seed rows by "
                f"content: {preview}{more}. The benchmark would score "
                "memorisation on those rows. Regenerate it disjoint from the "
                "seed corpus (a separate family namespace is not enough)."
            )

    leaked: dict[str, str] = {}
    for row in benchmark_rows:
        key = _group_key(row, group_by=group_by)
        split = group_assignment.get(key)
        if split in (Split.train, Split.val):
            leaked.setdefault(key, split.value)
    if leaked:
        detail = ", ".join(f"{key} (in {split})" for key, split in sorted(leaked.items()))
        raise ValueError(
            "benchmark_samples share group keys with the training splits: "
            f"{detail}. A benchmark is pinned to test, so those groups would sit "
            "on both sides of the split. Re-tag the benchmark rows (family / "
            "dataset.group_by) or remove them from the seed corpus."
        )


def _check_chat_rows(populations: dict[str, Optional[list[dict]]]) -> None:
    """Refuse rows whose tool-call fields are malformed.

    ``tools``, assistant ``tool_calls``, ``tool`` turns, ``chat_template_kwargs``
    and the ``family`` / ``surface`` / ``category`` / ``scenario`` tags are
    accepted; only their structure is checked, so plain rows are untouched.
    """
    problems: list[str] = []
    for label, rows in populations.items():
        for n, row in enumerate(rows or []):
            for problem in validate_chat_row(row):
                problems.append(f"{label} row {row.get('sample_id') or n}: {problem}")
    if problems:
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise ValueError(
            "malformed chat rows; refusing to write splits:\n  " + "\n  ".join(problems[:10]) + more
        )


def prepare_rows(
    model_def: ModelDefinition,
    seed_rows: list[dict],
    *,
    out_dir: Optional[Path] = None,
    seed_label: str = "",
    benchmark_rows: Optional[list[dict]] = None,
    benchmark_label: Optional[str] = None,
    sanitize_applied: Optional[list[str]] = None,
    blind_rows: Optional[list[dict]] = None,
    input_files: Optional[Sequence[Path]] = None,
    train_rows: Optional[list[dict]] = None,
) -> dict:
    """Split already-loaded rows and write train/val/test + dataset card.

    ``train_rows`` join train after the hash split and never reach val or the
    benchmark: rows collected from a policy's own rollouts, whose distribution
    must not move the numbers earlier runs were measured on. A train-only row
    whose group the split sent to val or test is refused.

    ``dataset.pins`` move whole groups into val or the benchmark after the
    hash split; ``dataset.isolation`` is then asserted over every population,
    including ``blind_rows``, which are checked but never written.

    Shared by ``jsonl_seed`` and format adapters (alpaca / sharegpt).

    ``sanitize_applied`` is the list of sanitizer tags actually run on the rows
    (the card reports these, not the declared config, so it never claims a
    sanitizer ran when it did not).

    With ``dataset.attribution`` declared every row's ``source`` must hold a
    signed entry; the corpus lock records ``input_files`` (the seed,
    benchmark and blind files as they were read) by sha256.
    """
    cfg = get_dataset_cfg(model_def)
    _check_chat_rows(
        {"seed": seed_rows, "benchmark": benchmark_rows, "blind": blind_rows, "train": train_rows}
    )
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    ratios = _resolve_ratios(cfg)
    group_by = cfg.get("group_by")
    if group_by is not None:
        group_by = str(group_by).strip() or None

    by_split, group_assignment, degenerate_key = _assign_group_splits(
        seed_rows, ratios, group_by=group_by
    )
    isolation = resolve_isolation(cfg)
    pins = resolve_pins(cfg, isolation)
    pinned: dict[str, int] = {}
    if pins:
        as_str = {split.value: rows for split, rows in by_split.items()}
        pinned = apply_pins(as_str, pins)
        # A pinned population is exactly its pins: whatever the hash split
        # dealt it from unpinned groups would share a camera or site with
        # training, so those rows return to train.
        pinned_populations = {"test" if p.population == "benchmark" else p.population for p in pins}
        for name in pinned_populations:
            keep: list[dict] = []
            for row in as_str.get(name, []):
                if any(
                    p.matches(row)
                    for p in pins
                    if (p.population if p.population != "benchmark" else "test") == name
                ):
                    keep.append(row)
                else:
                    back = dict(row)
                    back["split"] = Split.train.value
                    as_str[Split.train.value].append(back)
            as_str[name] = keep
        by_split = {split: as_str.get(split.value, []) for split in by_split}
        # A pinned group has a declared home; the hash assignment no longer
        # describes it, and the benchmark guard must see where it went.
        for split, rows in by_split.items():
            for row in rows:
                group_assignment[_group_key(row, group_by=group_by)] = split
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for rows in by_split.values():
        for row in rows:
            category_counts[str(row.get("category") or "unknown")] += 1
            source_counts[str(row.get("source") or "unknown")] += 1
            if row.get("family"):
                family_counts[str(row["family"])] += 1

    if benchmark_rows:
        _check_benchmark_leakage(
            benchmark_rows,
            group_assignment,
            group_by=group_by,
            seed_rows=seed_rows,
            request_field=str(cfg.get("request_field") or cfg.get("raw_field") or "") or None,
        )
        for row in benchmark_rows:
            tagged = dict(row)
            tagged["split"] = Split.test.value
            by_split[Split.test].append(tagged)
            category_counts[str(tagged.get("category") or "unknown")] += 1
            source_counts[str(tagged.get("source") or "unknown")] += 1
            if tagged.get("family"):
                family_counts[str(tagged["family"])] += 1

    if train_rows:
        leaked = sorted(
            {
                key
                for key in (_group_key(row, group_by=group_by) for row in train_rows)
                if group_assignment.get(key) in (Split.val, Split.test)
            }
        )
        if leaked:
            raise ValueError(
                f"{len(leaked)} train-only group(s) are already in val or test, "
                f"e.g. {leaked[:3]}; refusing to train on a held-out group"
            )
        for row in train_rows:
            tagged = dict(row)
            tagged["split"] = Split.train.value
            by_split[Split.train].append(tagged)
            category_counts[str(tagged.get("category") or "unknown")] += 1
            source_counts[str(tagged.get("source") or "unknown")] += 1
            if tagged.get("family"):
                family_counts[str(tagged["family"])] += 1

    if isolation is not None:
        populations = {
            "train": by_split[Split.train],
            "val": by_split[Split.val],
            "benchmark": by_split[Split.test],
            "blind": list(blind_rows or []),
        }
        problems = check_isolation(populations, isolation)
        if problems:
            raise ValueError(
                "dataset.isolation violated; refusing to write splits:\n  "
                + "\n  ".join(problems)
                + "\nPin whole groups at the policy level (dataset.pins) or set "
                "dataset.group_by to that level so the hash split cannot straddle it."
            )
    if blind_rows:
        _check_benchmark_leakage(
            blind_rows,
            group_assignment,
            group_by=group_by,
            seed_rows=seed_rows + (benchmark_rows or []),
            request_field=str(cfg.get("request_field") or cfg.get("raw_field") or "") or None,
        )

    attribution_path = resolve_attribution(model_def)
    attribution_field = resolve_attribution_field(model_def)
    source_check = None
    if attribution_path is not None:
        entries = read_attribution(attribution_path)
        every_row = [row for rows in by_split.values() for row in rows] + list(blind_rows or [])
        source_check = check_sources(every_row, entries, field=attribution_field)
        if not source_check.ok:
            raise AttributionError(
                f"{attribution_path.name} refuses to prepare:\n  "
                + "\n  ".join(source_check.refusals)
            )

    # After benchmark pinning: test may be non-empty only because of it.
    _warn_on_empty_splits(by_split, n_groups=len(group_assignment))

    version = benchmark_version(by_split[Split.test], pins)
    write_benchmark_state(
        out,
        version=version,
        n=len(by_split[Split.test]),
        benchmark_path=Path(benchmark_label) if benchmark_label else None,
        pins=pins,
        isolation=isolation,
    )

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    lock_files = (
        list(input_files)
        if input_files is not None
        else [
            Path(label)
            for label in (seed_label, benchmark_label)
            if label and Path(label).is_file()
        ]
    )
    lock = write_corpus_lock(
        out,
        files=lock_files,
        attribution_path=attribution_path,
        check=source_check,
        field=attribution_field,
    )
    lock_sha = json.loads(lock.read_text(encoding="utf-8"))["lock_sha256"]
    accepted_risk = dict(source_check.accepted_risk) if source_check else {}
    card = _write_dataset_card(
        out,
        title=f"{model_def.identity} dataset",
        counts=dict(category_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Seed source: {seed_label or 'in-memory'}",
            f"Benchmark source: {benchmark_label or 'none'}",
            f"group_by: {group_by or '(default family→source→sample_id)'}",
            f"Sources: {dict(source_counts)}",
            f"Families: {dict(family_counts) if family_counts else '{}'}",
            f"Sanitize: {list(sanitize_applied) if sanitize_applied else 'none'}",
            f"Degenerate group split per row: {degenerate_key or 'none'}",
            f"Benchmark version: {version}",
            f"Pins: {pinned if pinned else 'none'}",
            f"Isolation: {isolation.policy if isolation else 'none'}",
            f"Blind rows (checked, not written): {len(blind_rows or [])}",
            f"Train-only rows: {len(train_rows or [])}",
            f"Attribution: {attribution_path or 'none'}"
            + (f" (keyed on `{attribution_field}`)" if attribution_path else ""),
            f"Accepted risk: {accepted_risk if accepted_risk else 'none'}",
            f"Corpus lock: {lock_sha}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "family_counts": dict(family_counts),
        "split_counts": split_counts,
        "degenerate_group": degenerate_key,
        "benchmark_version": version,
        "pins": pinned,
        "blind_rows": len(blind_rows or []),
        "corpus_lock": lock_sha,
        "accepted_risk": accepted_risk,
    }
    console.print(
        f"[green]prepare complete[/] ({model_def.identity}): {summary['split_counts']} "
        f"(categories: {len(category_counts)})"
    )
    return summary


@register_format("jsonl_seed")
def prepare(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Generic JSONL-seed prepare: sanitize → group-split → write card.

    Reads ``seed_samples`` / ``benchmark_samples`` / ``split_ratios`` /
    ``sanitize`` / ``request_field`` from ``get_dataset_cfg(model_def)``.
    """
    cfg = get_dataset_cfg(model_def)

    if "seed_samples" not in cfg:
        raise ValueError("model.yml `dataset:`/`data:` must declare `seed_samples`")

    request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
    sanitize_tags = list(cfg.get("sanitize") or [])

    seed_path = model_def.resolve(cfg["seed_samples"])
    seed_rows: list[dict] = []
    for raw in iter_jsonl(seed_path):
        seed_rows.append(_apply_sanitize(raw, sanitize_tags, request_field))

    bench_rows: list[dict] = []
    benchmark_path = cfg.get("benchmark_samples")
    bench_label = None
    if benchmark_path:
        bench = model_def.resolve(benchmark_path)
        bench_label = str(bench)
        refuse_in_place_benchmark_change(
            Path(out_dir) if out_dir else model_def.prepared_dir, bench
        )
        for raw in iter_jsonl(bench):
            bench_rows.append(_apply_sanitize(raw, sanitize_tags, request_field))

    blind_rows: list[dict] = []
    blind_path = cfg.get("blind_samples")
    input_files = [seed_path]
    if benchmark_path:
        input_files.append(model_def.resolve(benchmark_path))
    if blind_path:
        input_files.append(model_def.resolve(blind_path))
        for raw in iter_jsonl(model_def.resolve(blind_path)):
            blind_rows.append(_apply_sanitize(raw, sanitize_tags, request_field))

    return prepare_rows(
        model_def,
        seed_rows,
        out_dir=out_dir,
        seed_label=str(seed_path),
        benchmark_rows=bench_rows or None,
        benchmark_label=bench_label,
        sanitize_applied=sanitize_tags,
        blind_rows=blind_rows or None,
        input_files=input_files,
    )
