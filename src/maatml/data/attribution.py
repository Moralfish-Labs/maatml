"""Sources carry their licence: a sidecar attribution table gates ``prepare``.

``dataset.attribution`` names a Markdown file with one table row per value
the corpus rows carry in ``source`` (or the field ``dataset.attribution_field``
names, for a corpus whose ``source`` is finer than its licence). ``prepare``
refuses a row whose source has no entry, an entry whose sign-off is blocked or
unsigned, and a source whose commercial-use is ``no`` / ``unknown`` unless its
sign-off records an
accepted risk (a name and a date, not a flag). What entered prepare — every
input file with its sha256, the table rows it used, the risk accepted — is
written to ``output/prepared/corpus.lock.json`` and copied into the export
manifest, where the same rows become the bundle's ``ATTRIBUTION.md``.

```markdown
| source     | licence   | commercial-use | consent           | sign-off                           |
| ---        | ---       | ---            | ---               | ---                                |
| meva       | CC BY 4.0 | yes            | consented actors  | cleared — A. Name 2026-08-17       |
| crowdhuman | research  | no             | third-party photos| accepted-risk — A. Name 2026-08-17 |
| dukemtmc   | withdrawn | no             | non-consented     | do not use                         |
```

Columns are matched by the casefolded prefix of their header (``licence`` or
``license``, ``commercial-use …``, ``provenance`` or ``consent``,
``attribution …``, ``sign-off``); every other column is kept verbatim in the
lock. Nothing here is looked up or inferred: the table is declared metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from ..utils.io import iter_jsonl, sha256_file, write_json_atomic

ATTRIBUTION_KEY = "attribution"
ATTRIBUTION_FIELD_KEY = "attribution_field"
DEFAULT_ATTRIBUTION_FIELD = "source"
ATTRIBUTION_FILE = "ATTRIBUTION.md"
CORPUS_LOCK = "corpus.lock.json"
CORPUS_LOCK_KIND = "maatml.corpus_lock/1"

_BLOCKED_SIGN_OFF = ("do not use", "blocked", "fixtures only")
_ACCEPTED_RISK = ("accepted-risk", "accepted risk", "risk accepted", "accept-risk")
_COLUMNS: dict[str, tuple[str, ...]] = {
    "source": ("source",),
    "licence": ("licence", "license"),
    "commercial_use": ("commercial-use", "commercial use", "commercial"),
    "consent": ("provenance", "consent"),
    "attribution": ("attribution",),
    "sign_off": ("sign-off", "sign off", "signoff"),
}
_REQUIRED = ("source", "licence", "commercial_use", "sign_off")


class AttributionError(ValueError):
    """The attribution table is unreadable, or a source may not enter prepare."""


@dataclass(frozen=True)
class SourceEntry:
    name: str
    licence: str
    commercial_use: str  # yes | no | unknown
    consent: str
    attribution: str
    sign_off: str
    raw: dict[str, str]

    @property
    def blocked(self) -> bool:
        sign = self.sign_off.casefold()
        return any(token in sign for token in _BLOCKED_SIGN_OFF)

    @property
    def signed(self) -> bool:
        sign = self.sign_off.strip().strip("*").strip().casefold()
        return bool(sign) and not self.blocked and not sign.startswith("unsigned")

    @property
    def risk_accepted(self) -> bool:
        sign = self.sign_off.casefold()
        return self.signed and any(token in sign for token in _ACCEPTED_RISK)

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "licence": self.licence,
            "commercial_use": self.commercial_use,
            "consent": self.consent,
            "attribution": self.attribution,
            "sign_off": self.sign_off,
            "columns": dict(self.raw),
        }


def classify_commercial_use(cell: str) -> str:
    text = cell.strip().strip("*").strip().casefold()
    if re.match(r"^yes\b", text):
        return "yes"
    if re.match(r"^no\b", text):
        return "no"
    # Anything the table cannot state plainly is unknown and needs a signed acceptance.
    return "unknown"


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _column_index(headers: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for field_name, prefixes in _COLUMNS.items():
        for position, header in enumerate(headers):
            if any(header.startswith(prefix) for prefix in prefixes) and field_name not in index:
                index[field_name] = position
    return index


def _column_reader(index: dict[str, int], cells: list[str]) -> Callable[[str], str]:
    def col(field_name: str) -> str:
        position = index.get(field_name)
        return cells[position] if position is not None and position < len(cells) else ""

    return col


def read_attribution(path: Path) -> dict[str, SourceEntry]:
    """Parse the first ``source`` table in a Markdown file into entries by source name."""
    path = Path(path)
    if not path.is_file():
        raise AttributionError(f"dataset.attribution {path} does not exist")
    headers: list[str] = []
    index: dict[str, int] = {}
    entries: dict[str, SourceEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if not headers:
            if cells[0].casefold() != "source":
                continue
            headers = [cell.casefold() for cell in cells]
            index = _column_index(headers)
            missing = [name for name in _REQUIRED if name not in index]
            if missing:
                raise AttributionError(
                    f"{path.name}: the source table lacks column(s) "
                    f"{[m.replace('_', '-') for m in missing]}; "
                    "it needs source, licence, commercial-use and sign-off"
                )
            continue
        raw = {headers[i]: cells[i] if i < len(cells) else "" for i in range(len(headers))}
        col = _column_reader(index, cells)
        name = col("source").strip("`* ")
        if not name:
            continue
        if name in entries:
            raise AttributionError(f"{path.name}: source {name!r} has two rows")
        entries[name] = SourceEntry(
            name=name,
            licence=col("licence"),
            commercial_use=classify_commercial_use(col("commercial_use")),
            consent=col("consent"),
            attribution=col("attribution"),
            sign_off=col("sign_off"),
            raw=raw,
        )
    if not headers:
        raise AttributionError(f"{path.name}: no Markdown table whose first column is `source`")
    return entries


def resolve_attribution(model_def: Any) -> Optional[Path]:
    from ..config import get_dataset_cfg

    value = get_dataset_cfg(model_def).get(ATTRIBUTION_KEY)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AttributionError("dataset.attribution must be a path to a Markdown table")
    return model_def.resolve(value)


def resolve_attribution_field(model_def: Any) -> str:
    """The row field the table is keyed on (``dataset.attribution_field``, default ``source``)."""
    from ..config import get_dataset_cfg

    value = get_dataset_cfg(model_def).get(ATTRIBUTION_FIELD_KEY)
    if value is None:
        return DEFAULT_ATTRIBUTION_FIELD
    if not isinstance(value, str) or not value.strip():
        raise AttributionError("dataset.attribution_field must be a row field name")
    return value.strip()


@dataclass
class SourceCheck:
    used: dict[str, int]
    refusals: list[str]
    accepted_risk: dict[str, str]
    entries: dict[str, SourceEntry]

    @property
    def ok(self) -> bool:
        return not self.refusals


def check_sources(
    rows: Iterable[dict[str, Any]],
    entries: dict[str, SourceEntry],
    *,
    field: str = DEFAULT_ATTRIBUTION_FIELD,
) -> SourceCheck:
    """Every distinct ``field`` value among ``rows`` against the table; refusals are strings."""
    used: dict[str, int] = {}
    unsourced = 0
    for row in rows:
        source = row.get(field)
        if source is None or not str(source).strip():
            unsourced += 1
            continue
        used[str(source).strip()] = used.get(str(source).strip(), 0) + 1
    refusals: list[str] = []
    accepted: dict[str, str] = {}
    if unsourced:
        refusals.append(
            f"{unsourced} row(s) carry no `{field}`; with dataset.attribution declared "
            "every row must name the entry it enters under"
        )
    for name in sorted(used):
        entry = entries.get(name)
        if entry is None:
            refusals.append(f"{name}: no attribution row; add and sign it before prepare")
            continue
        if entry.blocked:
            refusals.append(f"{name}: sign-off is {entry.sign_off!r}")
            continue
        if not entry.signed:
            refusals.append(f"{name}: unsigned (sign-off {entry.sign_off!r})")
            continue
        if entry.commercial_use in ("no", "unknown"):
            if entry.risk_accepted:
                accepted[name] = entry.sign_off
            else:
                refusals.append(
                    f"{name}: commercial-use is {entry.commercial_use}; a source enters "
                    "only under an `accepted-risk — <name> <date>` sign-off"
                )
    return SourceCheck(used=used, refusals=refusals, accepted_risk=accepted, entries=entries)


def corpus_lock_path(prepared_dir: Path) -> Path:
    return Path(prepared_dir) / CORPUS_LOCK


def _lock_digest(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_corpus_lock(
    prepared_dir: Path,
    *,
    files: Sequence[Path],
    attribution_path: Optional[Path],
    check: Optional[SourceCheck],
    field: str = DEFAULT_ATTRIBUTION_FIELD,
) -> Path:
    """Record what entered prepare: input files by hash, the table rows used, the risk accepted."""
    file_entries = []
    for path in files:
        path = Path(path)
        if not path.is_file():
            continue
        file_entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": sum(1 for _ in iter_jsonl(path)),
            }
        )
    sources = []
    if check is not None:
        for name in sorted(check.used):
            entry = check.entries.get(name)
            if entry is not None:
                record = entry.to_record()
                record["rows"] = check.used[name]
                sources.append(record)
    payload: dict[str, Any] = {
        "kind": CORPUS_LOCK_KIND,
        "attribution": (
            {
                "path": str(attribution_path),
                "sha256": sha256_file(attribution_path),
                "field": field,
            }
            if attribution_path is not None
            else None
        ),
        "files": file_entries,
        "sources": sources,
        "accepted_risk": dict(check.accepted_risk) if check is not None else {},
    }
    payload["lock_sha256"] = _lock_digest(payload)
    return write_json_atomic(corpus_lock_path(prepared_dir), payload)


def read_corpus_lock(prepared_dir: Path) -> Optional[dict[str, Any]]:
    path = corpus_lock_path(prepared_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != CORPUS_LOCK_KIND:
        return None
    return payload


def render_attribution_block(lock: dict[str, Any]) -> list[str]:
    """Per-source attribution from the lock; declared metadata only."""
    lines = ["## Sources", ""]
    sources = lock.get("sources") or []
    if not sources:
        lines.append("No attribution table was declared for this corpus.")
    for entry in sources:
        lines.append(
            f"- **{entry.get('source')}** ({entry.get('rows', 0)} rows) — "
            f"{entry.get('licence') or 'licence not stated'}; "
            f"commercial-use {entry.get('commercial_use')}; "
            f"{entry.get('consent') or 'consent basis not stated'}; "
            f"sign-off: {entry.get('sign_off')}"
        )
        if entry.get("attribution"):
            lines.append(f"  - attribution: {entry['attribution']}")
    accepted = lock.get("accepted_risk") or {}
    if accepted:
        lines.append("")
        lines.append("Risk accepted at prepare, as signed in the attribution table:")
        for name, sign in accepted.items():
            lines.append(f"- {name}: {sign}")
    lines.append("")
    lines.append(
        f"Corpus lock `{str(lock.get('lock_sha256'))[:16]}`: "
        f"{len(lock.get('files') or [])} input file(s) by sha256."
    )
    return lines


def write_attribution_file(export_dir: Path, lock: dict[str, Any]) -> Path:
    path = Path(export_dir) / ATTRIBUTION_FILE
    path.write_text("\n".join(render_attribution_block(lock)) + "\n", encoding="utf-8")
    return path
