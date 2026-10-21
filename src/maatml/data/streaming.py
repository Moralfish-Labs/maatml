"""Streaming prepare: rows are split and written as they arrive.

``prepare_rows`` holds every row of every population in memory (the rows, a
copy of each, the groups, the splits) before it writes anything, so a corpus
of tens of gigabytes cannot be prepared on one machine. ``prepare_stream``
makes the same decisions with state that does not grow with the rows:

* a row's split is a hash of its group key, so each row is written to its
  split file the moment it is read;
* group sizes (one integer per group), the exact-duplicate check (one short
  digest per seed row) and the isolation, attribution and card tallies are
  kept as counters and sets of distinct values, never as rows;
* the benchmark version, an order-insensitive hash over every test row, is
  computed by an external merge sort of the row identities on disk.

Two rules need more than one look at the rows. ``dataset.group_by`` groups
that turn out to cover nearly the whole corpus are split per row; that is only
known after the last row, so the rows are written provisionally and the
split files are rewritten once, only in that case. Everything else (pins,
benchmark placement, train-only rows) is decided from the row and the config.

Row order inside a split file is the order rows arrived in (seed rows, then
benchmark rows for ``test``, then train-only rows for ``train``). For input
whose groups are contiguous (the rows of one episode follow each other) that
is also the order ``prepare_rows`` writes.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Iterator, Optional, Sequence, Union

from ..config import ModelDefinition, get_dataset_cfg
from ..utils.chat import validate_chat_row
from .attribution import (
    AttributionError,
    check_source_counts,
    read_attribution,
    resolve_attribution,
    resolve_attribution_field,
    write_corpus_lock,
)
from .pipeline import (
    _DEGENERATE_GROUP_SHARE,
    _content_key,
    _group_key,
    _resolve_ratios,
    _row_identity,
    _split_from_hash,
    _warn_degenerate_group,
    _warn_on_empty_counts,
    _write_dataset_card,
    console,
)
from .populations import (
    IsolationTally,
    PopulationError,
    Pin,
    benchmark_version_from_identities,
    check_isolation_tallies,
    resolve_isolation,
    resolve_pins,
    row_identity,
    write_benchmark_state,
)
from .schemas import Split

SEED = "seed"
BENCHMARK = "benchmark"
BLIND = "blind"
TRAIN_ONLY = "train"
POPULATIONS = (SEED, BENCHMARK, BLIND, TRAIN_ONLY)

# Row identities are sorted in chunks of about this many characters and merged.
_SORT_CHUNK_CHARS = 64 * 1024 * 1024
_SPLITS = (Split.train, Split.val, Split.test)
_BIT = {Split.train: 1, Split.val: 2, Split.test: 4}

Labelled = Union[str, Callable[[], Optional[str]], None]


def _digest(key: Optional[str]) -> Optional[bytes]:
    if key is None:
        return None
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).digest()[:16]


def _resolve(value: Any) -> Any:
    return value() if callable(value) else value


class _Segment:
    """Counters over the rows one split file (or tail) received."""

    def __init__(self, isolation: Any) -> None:
        self.rows = 0
        self.categories: Counter[str] = Counter()
        self.sources: Counter[str] = Counter()
        self.families: Counter[str] = Counter()
        self.tally = IsolationTally(isolation) if isolation is not None else None

    def observe(self, row: dict) -> None:
        self.rows += 1
        self.categories[str(row.get("category") or "unknown")] += 1
        self.sources[str(row.get("source") or "unknown")] += 1
        if row.get("family"):
            self.families[str(row["family"])] += 1
        if self.tally is not None:
            self.tally.add(row)

    def then(self, later: "_Segment") -> "_Segment":
        """This segment's rows followed by ``later``'s, in counter insertion order too."""
        joined = _Segment(None)
        joined.rows = self.rows + later.rows
        for name in ("categories", "sources", "families"):
            counter = getattr(joined, name)
            counter.update(getattr(self, name))
            counter.update(getattr(later, name))
        if self.tally is not None and later.tally is not None:
            joined.tally = self.tally.merge(later.tally)
        return joined


class _IdentitySorter:
    """Sorted stream of row identities without holding them all.

    Identities are ASCII (``json.dumps`` escapes), so their sort order is the
    order ``sorted()`` gives ``benchmark_version`` over the same strings.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._chunk: list[str] = []
        self._chars = 0
        self._runs: list[Path] = []

    def add(self, identity: str) -> None:
        self._chunk.append(identity)
        self._chars += len(identity)
        if self._chars >= _SORT_CHUNK_CHARS:
            self._spill()

    def _spill(self) -> None:
        self._chunk.sort()
        fd, name = tempfile.mkstemp(dir=self._dir, prefix="ids-", suffix=".run")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for identity in self._chunk:
                handle.write(identity)
                handle.write("\n")
        self._runs.append(Path(name))
        self._chunk = []
        self._chars = 0

    def lines(self) -> Iterator[str]:
        with ExitStack() as stack:
            streams: list[Iterable[str]] = []
            for run in self._runs:
                handle = stack.enter_context(open(run, "r", encoding="utf-8"))
                streams.append(line.rstrip("\n") for line in handle)
            self._chunk.sort()
            streams.append(iter(self._chunk))
            yield from heapq.merge(*streams)

    def discard(self) -> None:
        for run in self._runs:
            run.unlink(missing_ok=True)
        self._runs = []
        self._chunk = []
        self._chars = 0


class _Router:
    """Where a seed row goes: pins first, else the hash of its group."""

    def __init__(self, ratios: tuple[float, float, float], pins: Sequence[Pin]) -> None:
        self.ratios = ratios
        self.pins = [
            (pin, Split.test if pin.population == "benchmark" else Split(pin.population))
            for pin in pins
        ]
        # A pinned population is exactly its pins; what the hash dealt it from
        # unpinned groups returns to train.
        self.pinned_populations = {target.value for _, target in self.pins}

    def split_for(self, row: dict, gkey: str, dominant: Optional[str]) -> Split:
        landed: Optional[Split] = None
        for pin, target in self.pins:
            if pin.matches(row):
                landed = target  # later pins move the row again: the last one wins
        if landed is not None:
            return landed
        hashed = gkey
        if dominant is not None and gkey == dominant:
            hashed = f"{gkey}#row:{_row_identity(row)}"
        split = _split_from_hash(hashed, self.ratios)
        if split.value in self.pinned_populations:
            return Split.train
        return split


class _Stream:
    def __init__(
        self,
        model_def: ModelDefinition,
        out: Path,
        work: Path,
    ) -> None:
        cfg = get_dataset_cfg(model_def)
        self.model_def = model_def
        self.out = out
        self.work = work
        self.ratios = _resolve_ratios(cfg)
        group_by = cfg.get("group_by")
        self.group_by: Optional[str] = None
        if group_by is not None:
            self.group_by = str(group_by).strip() or None
        self.request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "") or None
        self.isolation = resolve_isolation(cfg)
        self.pins = resolve_pins(cfg, self.isolation)
        self.router = _Router(self.ratios, self.pins)

        self.attribution_path = resolve_attribution(model_def)
        self.attribution_field = resolve_attribution_field(model_def)
        self.used_sources: dict[str, int] = {}
        self.unsourced = 0

        self.rows = {name: 0 for name in POPULATIONS}
        self.chat_first: dict[str, list[str]] = {name: [] for name in POPULATIONS}
        self.chat_total = 0

        self.group_rows: dict[str, int] = {}
        self.masks: dict[str, int] = {}
        self.pin_hits = [0] * len(self.pins)

        self.seed_content: dict[bytes, Any] = {}
        self.bench_content: dict[bytes, Any] = {}
        self.bench_checks: list[tuple[Optional[bytes], Any]] = []
        self.blind_checks: list[tuple[Optional[bytes], Any]] = []
        self.bench_keys: set[str] = set()
        self.blind_keys: set[str] = set()
        self.rollout_keys: set[str] = set()

        self.segments = self._fresh_segments()
        self.bench = _Segment(self.isolation)
        self.rollout = _Segment(self.isolation)
        self.blind = IsolationTally(self.isolation) if self.isolation is not None else None
        self.test_ids = _IdentitySorter(work)
        self.bench_ids = _IdentitySorter(work)

        self.main = {split: work / f"main.{split.value}.jsonl" for split in _SPLITS}
        self.tails = {Split.test: work / "tail.test.jsonl", Split.train: work / "tail.train.jsonl"}
        self._stack = ExitStack()
        self.main_out = {s: self._open(p) for s, p in self.main.items()}
        self.tail_out = {s: self._open(p) for s, p in self.tails.items()}

        self.dominant: Optional[str] = None
        self.n_identities = 0

    # -- plumbing ---------------------------------------------------------

    def _open(self, path: Path) -> IO[str]:
        return self._stack.enter_context(open(path, "w", encoding="utf-8"))

    def _opened(self, stack: ExitStack, name: str) -> IO[str]:
        return stack.enter_context(open(self.work / name, "w", encoding="utf-8"))

    def _fresh_segments(self) -> dict[Split, _Segment]:
        return {split: _Segment(self.isolation) for split in _SPLITS}

    # -- ingestion --------------------------------------------------------

    def ingest(self, population: str, row: dict) -> None:
        if population not in POPULATIONS:
            raise ValueError(f"unknown population {population!r}; expected one of {POPULATIONS}")
        index = self.rows[population]
        self.rows[population] += 1
        for problem in validate_chat_row(row):
            self.chat_total += 1
            if len(self.chat_first[population]) < 10:
                label = row.get("sample_id") or index
                self.chat_first[population].append(f"{population} row {label}: {problem}")
        if self.attribution_path is not None:
            source = row.get(self.attribution_field)
            if source is None or not str(source).strip():
                self.unsourced += 1
            else:
                name = str(source).strip()
                self.used_sources[name] = self.used_sources.get(name, 0) + 1

        key = _group_key(row, group_by=self.group_by)
        if population == SEED:
            self._ingest_seed(row, key)
        elif population == BENCHMARK:
            self.bench_keys.add(key)
            content = _digest(_content_key(row, self.request_field))
            self.bench_checks.append((content, row.get("sample_id")))
            if content is not None:
                self.bench_content.setdefault(content, row.get("sample_id"))
            tagged = dict(row)
            tagged["split"] = Split.test.value
            self.tail_out[Split.test].write(json.dumps(tagged, sort_keys=True) + "\n")
            self.bench.observe(tagged)
            self.bench_ids.add(row_identity(tagged))
        elif population == TRAIN_ONLY:
            self.rollout_keys.add(key)
            tagged = dict(row)
            tagged["split"] = Split.train.value
            self.tail_out[Split.train].write(json.dumps(tagged, sort_keys=True) + "\n")
            self.rollout.observe(tagged)
        else:
            self.blind_keys.add(key)
            self.blind_checks.append(
                (_digest(_content_key(row, self.request_field)), row.get("sample_id"))
            )
            if self.blind is not None:
                self.blind.add(row)

    def _ingest_seed(self, row: dict, key: str) -> None:
        self.group_rows[key] = self.group_rows.get(key, 0) + 1
        content = _digest(_content_key(row, self.request_field))
        if content is not None:
            self.seed_content.setdefault(content, row.get("sample_id"))
        for i, (pin, _) in enumerate(self.router.pins):
            if pin.matches(row):
                self.pin_hits[i] += 1
        split = self.router.split_for(row, key, None)
        if self.pins:
            self.masks[key] = self.masks.get(key, 0) | _BIT[split]
        tagged = dict(row)
        tagged["split"] = split.value
        self.main_out[split].write(json.dumps(tagged, sort_keys=True) + "\n")
        self.segments[split].observe(tagged)
        if split is Split.test:
            self.test_ids.add(row_identity(tagged))

    # -- decisions made after the last row --------------------------------

    def raise_on_chat_problems(self) -> None:
        if not self.chat_total:
            return
        problems = [p for name in POPULATIONS for p in self.chat_first[name]]
        more = f"\n  ... and {self.chat_total - 10} more" if self.chat_total > 10 else ""
        raise ValueError(
            "malformed chat rows; refusing to write splits:\n  " + "\n  ".join(problems[:10]) + more
        )

    def find_degenerate(self) -> Optional[str]:
        total = self.rows[SEED]
        if total < 2:
            return None
        key, members = max(self.group_rows.items(), key=lambda kv: kv[1])
        if members < 2 or members / total < _DEGENERATE_GROUP_SHARE:
            return None
        _warn_degenerate_group(key, members, total)
        return key

    def check_pins_matched(self) -> None:
        for (pin, _), hits in zip(self.router.pins, self.pin_hits):
            if not hits:
                raise PopulationError(f"dataset.pins.{pin.population} {pin.label} matches no row")

    def reroute_dominant(self, dominant: str) -> None:
        """Split the dominant group's rows one by one; the rest keep their place.

        The split files are read back once. Rows outside the group are copied
        verbatim in order, the group's rows follow them (the order
        ``prepare_rows`` gives them), and every seed-side tally is rebuilt from
        the rows as they now stand.
        """
        self.dominant = dominant
        for handle in self.main_out.values():
            handle.close()
        self.test_ids.discard()
        self.test_ids = _IdentitySorter(self.work)
        # The group's rows trail the others inside each split file, and their
        # tallies are joined in that order so the card lists keys as it would.
        kept_segments = self._fresh_segments()
        moved_segments = self._fresh_segments()
        self.masks = {}
        identities: set[str] = set()
        with ExitStack() as stack:
            kept = {s: self._opened(stack, f"fix.{s.value}") for s in _SPLITS}
            moved = {s: self._opened(stack, f"dom.{s.value}") for s in _SPLITS}
            for old in _SPLITS:
                with open(self.main[old], "r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        row.pop("split", None)
                        key = _group_key(row, group_by=self.group_by)
                        split = self.router.split_for(row, key, dominant)
                        if self.pins:
                            self.masks[key] = self.masks.get(key, 0) | _BIT[split]
                        tagged = dict(row)
                        tagged["split"] = split.value
                        if key == dominant:
                            identities.add(_row_identity(row))
                            moved[split].write(json.dumps(tagged, sort_keys=True) + "\n")
                        else:
                            kept[split].write(line if line.endswith("\n") else line + "\n")
                        segments = moved_segments if key == dominant else kept_segments
                        segments[split].observe(tagged)
                        if split is Split.test:
                            self.test_ids.add(row_identity(tagged))
        for split in _SPLITS:
            fixed = self.work / f"fix.{split.value}"
            with open(fixed, "ab") as dst, open(self.work / f"dom.{split.value}", "rb") as src:
                shutil.copyfileobj(src, dst, 1 << 20)
            self.main[split].unlink()
            os.replace(fixed, self.main[split])
            (self.work / f"dom.{split.value}").unlink()
        self.segments = {s: kept_segments[s].then(moved_segments[s]) for s in _SPLITS}
        self.n_identities = len(identities)

    def assignment_of(self, key: str) -> Optional[Split]:
        """Where the seed rows of group ``key`` landed (``None`` if it has none)."""
        if self.pins:
            mask = self.masks.get(key)
            if not mask:
                return None
            for split in (Split.test, Split.val, Split.train):
                if mask & _BIT[split]:
                    return split
        if key not in self.group_rows or key == self.dominant:
            return None
        return _split_from_hash(key, self.ratios)

    def n_groups(self) -> int:
        n = len(self.group_rows)
        if self.dominant is not None:
            n += self.n_identities - 1 + (1 if self.pins else 0)
        return n

    def check_content(
        self,
        checks: list[tuple[Optional[bytes], Any]],
        keys: Iterable[str],
        lookups: Sequence[dict[bytes, Any]],
        *,
        have_reference_rows: bool,
    ) -> None:
        """``_check_benchmark_leakage`` for rows that were held back as digests."""
        if have_reference_rows:
            duplicates: list[tuple[Any, Any]] = []
            for content, sid in checks:
                if content is None:
                    continue
                for lookup in lookups:
                    if content in lookup:
                        duplicates.append((sid, lookup[content]))
                        break
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
        for key in keys:
            split = self.assignment_of(key)
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

    def check_rollouts(self) -> None:
        leaked = sorted(
            key for key in self.rollout_keys if self.assignment_of(key) in (Split.val, Split.test)
        )
        if leaked:
            raise ValueError(
                f"{len(leaked)} train-only group(s) are already in val or test, "
                f"e.g. {leaked[:3]}; refusing to train on a held-out group"
            )

    def populations(self) -> dict[str, IsolationTally]:
        seg = self.segments
        assert self.isolation is not None
        tallies = {
            "train": _tally(seg[Split.train]).merge(_tally(self.rollout)),
            "val": _tally(seg[Split.val]),
            "benchmark": _tally(seg[Split.test]).merge(_tally(self.bench)),
        }
        tallies["blind"] = self.blind if self.blind is not None else IsolationTally(self.isolation)
        return tallies

    def merged(self, attr: str) -> Counter[str]:
        total: Counter[str] = Counter()
        for segment in (*self.segments.values(), self.bench, self.rollout):
            total.update(getattr(segment, attr))
        return total

    def split_counts(self) -> dict[Split, int]:
        counts = {split: self.segments[split].rows for split in _SPLITS}
        counts[Split.test] += self.bench.rows
        counts[Split.train] += self.rollout.rows
        return counts


def _tally(segment: _Segment) -> IsolationTally:
    assert segment.tally is not None
    return segment.tally


def prepare_stream(
    model_def: ModelDefinition,
    rows: Iterable[tuple[str, dict]],
    *,
    out_dir: Optional[Path] = None,
    seed_label: str = "",
    benchmark_label: Labelled = None,
    sanitize_applied: Optional[list[str]] = None,
    input_files: Union[Sequence[Path], Callable[[], Sequence[Path]], None] = None,
) -> dict:
    """``prepare_rows`` for rows that arrive one at a time.

    ``rows`` yields ``(population, row)`` pairs where population is
    ``"seed"`` (hash-split), ``"benchmark"`` (written to test), ``"train"``
    (train-only rollout rows) or ``"blind"`` (checked, never written). The
    populations may be interleaved. Each row is read once and may be discarded
    by the caller afterwards; nothing but counters and digests is kept.

    The result, the card, the benchmark state, the corpus lock and every
    refusal are those of ``prepare_rows`` given the same rows; the one
    difference is the order of rows inside a split file (see the module
    docstring). Nothing is written to the final locations until every check
    has passed, so a refused prepare leaves the previous output untouched.

    ``benchmark_label`` and ``input_files`` may be callables, evaluated once
    ``rows`` is exhausted, for producers that only learn them while streaming.
    """
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".prepare-", dir=out))
    try:
        return _run(
            model_def, rows, out, work, seed_label, benchmark_label, sanitize_applied, input_files
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run(
    model_def: ModelDefinition,
    rows: Iterable[tuple[str, dict]],
    out: Path,
    work: Path,
    seed_label: str,
    benchmark_label_in: Labelled,
    sanitize_applied: Optional[list[str]],
    input_files_in: Union[Sequence[Path], Callable[[], Sequence[Path]], None],
) -> dict:
    stream = _Stream(model_def, out, work)
    try:
        for population, row in rows:
            stream.ingest(population, row)
    finally:
        stream._stack.close()

    stream.raise_on_chat_problems()
    dominant = stream.find_degenerate()
    stream.check_pins_matched()
    if dominant is not None:
        stream.reroute_dominant(dominant)

    n_seed = stream.rows[SEED]
    n_bench = stream.rows[BENCHMARK]
    n_blind = stream.rows[BLIND]
    n_rollout = stream.rows[TRAIN_ONLY]

    if n_bench:
        stream.check_content(
            stream.bench_checks,
            stream.bench_keys,
            [stream.seed_content],
            have_reference_rows=bool(n_seed),
        )
    if n_rollout:
        stream.check_rollouts()
    isolation = stream.isolation
    if isolation is not None:
        problems = check_isolation_tallies(stream.populations(), isolation)
        if problems:
            raise ValueError(
                "dataset.isolation violated; refusing to write splits:\n  "
                + "\n  ".join(problems)
                + "\nPin whole groups at the policy level (dataset.pins) or set "
                "dataset.group_by to that level so the hash split cannot straddle it."
            )
    if n_blind:
        stream.check_content(
            stream.blind_checks,
            stream.blind_keys,
            [stream.seed_content, stream.bench_content],
            have_reference_rows=bool(n_seed + n_bench),
        )

    attribution_path = stream.attribution_path
    source_check = None
    if attribution_path is not None:
        entries = read_attribution(attribution_path)
        source_check = check_source_counts(
            stream.used_sources, stream.unsourced, entries, field=stream.attribution_field
        )
        if not source_check.ok:
            raise AttributionError(
                f"{attribution_path.name} refuses to prepare:\n  "
                + "\n  ".join(source_check.refusals)
            )

    split_counts = stream.split_counts()
    _warn_on_empty_counts(split_counts, n_groups=stream.n_groups())

    pins = stream.pins
    version = benchmark_version_from_identities(
        heapq.merge(stream.test_ids.lines(), stream.bench_ids.lines()), pins
    )
    benchmark_label = _resolve(benchmark_label_in)
    write_benchmark_state(
        out,
        version=version,
        n=split_counts[Split.test],
        benchmark_path=Path(benchmark_label) if benchmark_label else None,
        pins=pins,
        isolation=isolation,
    )

    paths: dict[str, str] = {}
    for split in _SPLITS:
        final = out / f"{split.value}.jsonl"
        os.replace(stream.main[split], final)
        tail = stream.tails.get(split)
        if tail is not None:
            with open(final, "ab") as dst, open(tail, "rb") as src:
                shutil.copyfileobj(src, dst, 1 << 20)
        paths[split.value] = str(final)

    declared_files = _resolve(input_files_in)
    lock_files = (
        list(declared_files)
        if declared_files is not None
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
        field=stream.attribution_field,
    )
    lock_sha = json.loads(lock.read_text(encoding="utf-8"))["lock_sha256"]
    accepted_risk = dict(source_check.accepted_risk) if source_check else {}
    category_counts = stream.merged("categories")
    source_counts = stream.merged("sources")
    family_counts = stream.merged("families")
    group_by = stream.group_by
    degenerate_key = dominant
    pinned = {pin.label: hits for (pin, _), hits in zip(stream.router.pins, stream.pin_hits)}
    counts_by_name = {split.value: n for split, n in split_counts.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.identity} dataset",
        counts=dict(category_counts),
        split_counts=counts_by_name,
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
            f"Blind rows (checked, not written): {n_blind}",
            f"Train-only rows: {n_rollout}",
            f"Attribution: {attribution_path or 'none'}"
            + (f" (keyed on `{stream.attribution_field}`)" if attribution_path else ""),
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
        "split_counts": counts_by_name,
        "degenerate_group": degenerate_key,
        "benchmark_version": version,
        "pins": pinned,
        "blind_rows": n_blind,
        "corpus_lock": lock_sha,
        "accepted_risk": accepted_risk,
        "population_rows": dict(stream.rows),
    }
    console.print(
        f"[green]prepare complete[/] ({model_def.identity}): {summary['split_counts']} "
        f"(categories: {len(category_counts)})"
    )
    return summary
