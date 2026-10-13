"""``maatml distill``: validator-gated teacher labels over a prompt pool.

Where ``datagen`` invents whole rows, ``distill`` starts from prompts you
already have and asks the teacher only for the label. Every response is gated
by the same validator eval and serve use, so a rejected label never reaches the
seed corpus. Accepted rows carry their provenance (teacher model / revision,
prompt hash, source, family), the rejections are kept, and teacher responses
are recorded so a run replays offline and reproduces the same corpus.

The stage config is typed (:class:`DistillConfig`); there is no new untyped
``dict[str, Any]`` surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import VALIDATORS
from ..utils.io import iter_jsonl, sha256_bytes, write_jsonl_atomic
from ..validation.base import strip_fences
from .ingest import _row_id
from .teacher import TeacherClient


class DistillConfigError(ValueError):
    """distill was asked to run without something it needs (validator, prompts)."""


class DistillConfig(BaseModel):
    """The ``distill:`` section of ``model.yml`` (and ``maatml distill`` flags).

    Prompts come from ``prompt_source`` (a JSONL of rows with the request field,
    or a plain-text file, one prompt per line). ``teacher_model`` /
    ``teacher_revision`` identify the teacher for provenance and for the replay
    cache key, so a different teacher does not silently reuse another's labels.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_source: str
    teacher_model: str = "gpt-4o-mini"
    teacher_revision: str = "unpinned"
    system_prompt: Optional[str] = None
    # Read from a file instead, resolved against the model folder. A briefing
    # that carries a real system prompt and a tool catalogue runs to thousands
    # of tokens; inlined in model.yml it would dwarf the config and drift from
    # whatever generated it. Wins over ``system_prompt`` when both are set.
    system_prompt_file: Optional[str] = None
    # Merged into the chat-completions payload. Reasoning teachers need a
    # max_tokens above the 1024 default (hidden reasoning spends it before any
    # content) and template switches like chat_template_kwargs.
    request_params: dict[str, Any] = Field(default_factory=dict)
    max_prompts: Optional[int] = Field(default=None, gt=0)
    family: str = "distill"
    # What the teacher's reply is. ``json`` parses it and rejects what will not
    # parse; ``text`` hands the reply to the validator as it stands.
    #
    # Not every task's target is JSON. A tool-calling model trained on an inline
    # protocol answers with a sentence and then a call object, which is not a
    # JSON document and never will be — under ``json`` every such label is
    # refused as unparseable before the validator is asked, so the gate that is
    # supposed to decide never sees them. The validator still decides; this only
    # says what to hand it.
    target_format: str = "json"
    cache: str = "output/distill/cache.jsonl"
    output: Optional[str] = None

    @classmethod
    def from_model(cls, model_def: ModelDefinition) -> "DistillConfig":
        section = getattr(model_def, "distill", None) or {}
        if not isinstance(section, dict) or not section:
            raise DistillConfigError(
                "no distill: section in model.yml. Add one (prompt_source is "
                "required) or pass --prompts on the command line."
            )
        try:
            return cls(**section)
        except Exception as exc:  # noqa: BLE001  pydantic error → config error
            raise DistillConfigError(f"invalid distill: section: {exc}") from exc


TEACHER_CACHE_KIND = "maatml.teacher_cache/1"


class TeacherCache:
    """Record/replay store for teacher responses, keyed by prompt + teacher.

    The key binds the prompt hash to the teacher model and revision, so a run
    against a different teacher does not reuse cached labels, and a replay is a
    faithful reproduction of what that teacher said. Stored as JSONL so it is
    diffable and can ship with an example for offline CI. The first line is a
    provenance header: which prompt pools (by sha256) and which benchmark
    version the labels were recorded against.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._store: dict[str, str] = {}
        self.provenance: dict[str, Any] = {"kind": TEACHER_CACHE_KIND, "prompt_sources": []}
        if self.path.is_file():
            for row in iter_jsonl(self.path):
                if row.get("kind") == TEACHER_CACHE_KIND:
                    self.provenance = dict(row)
                    continue
                key = row.get("key")
                response = row.get("response")
                if isinstance(key, str) and isinstance(response, str):
                    self._store[key] = response
        self._dirty = False

    def note_provenance(
        self,
        *,
        teacher: str,
        prompt_source: Path,
        benchmark_version: Optional[str],
    ) -> None:
        """Record the pool and benchmark version this session labels against."""
        from ..utils.io import sha256_file

        entry = {"path": str(prompt_source), "sha256": sha256_file(prompt_source)}
        sources = [
            s
            for s in self.provenance.get("prompt_sources", [])
            if s.get("sha256") != entry["sha256"]
        ]
        sources.append(entry)
        self.provenance = {
            "kind": TEACHER_CACHE_KIND,
            "teacher": teacher,
            "prompt_sources": sources,
            "benchmark_version": benchmark_version,
        }

    @staticmethod
    def key(prompt_hash: str, model: str, revision: str) -> str:
        return f"{model}@{revision}:{prompt_hash}"

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    # Flush cadence: a long local-teacher run is hours of paid-for responses,
    # and an end-only flush loses all of them to one crash.
    FLUSH_EVERY = 25

    def put(self, key: str, response: str) -> None:
        if self._store.get(key) != response:
            self._store[key] = response
            self._dirty = True
            self._unflushed = getattr(self, "_unflushed", 0) + 1
            if self._unflushed >= self.FLUSH_EVERY:
                self.flush()

    def flush(self) -> None:
        if not self._dirty:
            return
        # The header rides with the labels: a cache that recorded nothing
        # leaves no file, so a re-run reaches the teacher again.
        rows: list[dict[str, Any]] = (
            [dict(self.provenance)] if self.provenance.get("teacher") else []
        )
        rows.extend({"key": k, "response": v} for k, v in sorted(self._store.items()))
        write_jsonl_atomic(self.path, rows)
        self._dirty = False
        self._unflushed = 0


def _prompt_hash(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))[:16]


# Distill defines these itself, so a pool declaring one does not win. `family`
# above all: distill's is one per prompt, and a pool-wide family would put every
# accepted row into a single split group.
_RESERVED_ROW_FIELDS = frozenset({"sample_id", "source", "family", "provenance"})


def load_prompt_rows(path: Path, request_field: str) -> list[tuple[str, dict[str, Any]]]:
    """Read a prompt pool as each prompt paired with the rest of its row.

    A JSONL pool usually declares more than the ask — a scenario key, a project,
    a subject. Those travel onto the accepted row, because a pool that groups
    its prompts is saying which of them are the same situation, and a split that
    cannot see the grouping puts paraphrases of one ask on both sides of it.
    """
    path = Path(path)
    if not path.is_file():
        raise DistillConfigError(f"prompt_source not found: {path}")
    rows: list[tuple[str, dict[str, Any]]] = []
    if path.suffix.lower() == ".jsonl":
        asked = (request_field, "prompt", "request")
        for row in iter_jsonl(path):
            value = row.get(request_field) or row.get("prompt") or row.get("request")
            if isinstance(value, str) and value.strip():
                rows.append((value, {k: v for k, v in row.items() if k not in asked}))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((line.rstrip("\n"), {}))
    if not rows:
        raise DistillConfigError(f"no usable prompts in {path}")
    return rows


def heldout_overlaps(
    model_def: ModelDefinition, prompt_rows: list[tuple[str, dict[str, Any]]], request_field: str
) -> list[str]:
    """Prompts that already belong to a held-out population.

    Checked by content against ``benchmark_samples``, ``blind_samples`` and the
    prepared val / test splits, and by ``dataset.pins`` against any isolation
    field the pool row declares (a pool row tagged with a pinned camera is
    that camera's row, whatever the prompt says).
    """
    from .populations import resolve_isolation, resolve_pins

    ds_cfg = get_dataset_cfg(model_def)
    held: dict[str, str] = {}
    sources: list[tuple[str, Path]] = []
    for key in ("benchmark_samples", "blind_samples"):
        rel = ds_cfg.get(key)
        if isinstance(rel, str) and rel:
            sources.append((key, model_def.resolve(rel)))
    for split in ("val", "test"):
        sources.append((f"prepared {split}", Path(model_def.prepared_dir) / f"{split}.jsonl"))
    for label, path in sources:
        if not path.is_file():
            continue
        for row in iter_jsonl(path):
            value = row.get(request_field) or row.get("prompt") or row.get("request")
            if isinstance(value, str) and value.strip():
                held.setdefault(value.strip(), label)
    try:
        pins = resolve_pins(ds_cfg, resolve_isolation(ds_cfg))
    except ValueError:
        pins = []
    problems: list[str] = []
    by_content = 0
    by_pin: dict[str, int] = {}
    for prompt, declared in prompt_rows:
        where = held.get(prompt.strip())
        if where is not None:
            by_content += 1
            if by_content <= 3:
                problems.append(f"{prompt[:60]!r} is in {where}")
            continue
        for pin in pins:
            if pin.matches(declared):
                by_pin[pin.label] = by_pin.get(pin.label, 0) + 1
                break
    if by_content > 3:
        problems.append(f"... {by_content} prompt(s) by content in total")
    for label, count in sorted(by_pin.items()):
        problems.append(f"{count} prompt row(s) carry pinned {label}")
    return problems


def load_prompts(path: Path, request_field: str) -> list[str]:
    """Read a prompt pool from JSONL (request field) or plain text lines."""
    return [prompt for prompt, _ in load_prompt_rows(path, request_field)]


def _build_validate_fn(model_def: ModelDefinition):
    """Same validator gate datagen uses, returning (ok, result) per row."""
    cfg = get_dataset_cfg(model_def)
    validator_name = (model_def.evaluation or {}).get("validator")
    if not validator_name:
        raise DistillConfigError(
            "distill requires evaluation.validator so every teacher label is "
            "gated by the same validator used in eval and serve. Add "
            "evaluation.validator to model.yml."
        )
    try:
        validate = VALIDATORS.require(str(validator_name))
    except KeyError as exc:
        raise DistillConfigError(
            f"evaluation.validator={validator_name!r} is not registered "
            f"(known: {', '.join(VALIDATORS.names()) or '(none)'})."
        ) from exc
    schema_path = model_def.resolve(cfg["schema"]) if isinstance(cfg.get("schema"), str) else None
    contracts_path = (
        model_def.resolve(cfg["contracts"]) if isinstance(cfg.get("contracts"), str) else None
    )

    def _fn(prompt: str, target: Any) -> bool:
        raw = target if isinstance(target, str) else json.dumps(target)
        result = validate(
            raw,
            schema_path=schema_path,
            contracts_path=contracts_path,
            user_prompt=prompt,
        )
        return bool(result.ok)

    return _fn


def run_distill(
    model_def: ModelDefinition,
    *,
    config: Optional[DistillConfig] = None,
    prompt_source: Optional[str] = None,
    replay: bool = False,
    offline: bool = False,
    limit: Optional[int] = None,
    append: bool = True,
    out_path: Optional[str] = None,
) -> dict[str, Any]:
    """Distill validator-gated teacher labels over the prompt pool.

    ``replay`` uses only cached teacher responses (a cache miss is skipped, not
    fetched); ``offline`` forbids any network call and is implied by ``replay``.
    Returns a summary dict; accepted rows are appended to the seed corpus.
    """
    if config is not None:
        cfg = config
    elif getattr(model_def, "distill", None):
        cfg = DistillConfig.from_model(model_def)
    elif prompt_source:
        # A prompt pool on the command line is enough; the rest defaults.
        cfg = DistillConfig(prompt_source=prompt_source)
    else:
        raise DistillConfigError(
            "no distill: section in model.yml and no --prompts given. "
            "Point distill at a prompt pool one way or the other."
        )
    if prompt_source:
        cfg = cfg.model_copy(update={"prompt_source": prompt_source})
    ds_cfg = get_dataset_cfg(model_def)
    request_field = str(ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request")
    target_field = str(ds_cfg.get("target_field") or "target")

    validate_fn = _build_validate_fn(model_def)
    prompt_rows = load_prompt_rows(model_def.resolve(cfg.prompt_source), request_field)
    cap = limit if limit is not None else cfg.max_prompts
    if cap is not None:
        prompt_rows = prompt_rows[:cap]

    overlaps = heldout_overlaps(model_def, prompt_rows, request_field)
    if overlaps:
        raise DistillConfigError(
            "the prompt pool overlaps held-out populations; a teacher label for one of "
            "those prompts would enter training:\n  "
            + "\n  ".join(overlaps)
            + "\nRemove them from the pool (the benchmark and blind manifests are "
            "not prompt sources)."
        )
    cache = TeacherCache(model_def.resolve(cfg.cache))
    if not (replay or offline):
        from .populations import read_benchmark_state

        state = read_benchmark_state(model_def.prepared_dir)
        cache.note_provenance(
            teacher=f"{cfg.teacher_model}@{cfg.teacher_revision}",
            prompt_source=model_def.resolve(cfg.prompt_source),
            benchmark_version=str(state.get("version")) if state else None,
        )
    offline = offline or replay
    teacher: Optional[TeacherClient] = None
    briefing = cfg.system_prompt
    if cfg.system_prompt_file:
        briefing = model_def.resolve(cfg.system_prompt_file).read_text(encoding="utf-8")
    system = briefing or (
        f"You label training samples for the MaatML model {model_def.identity}. "
        f"Given a user request, return only a JSON object for the "
        f"'{target_field}' field. No markdown fences, no commentary."
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats = {
        "cache_hits": 0,
        "teacher_calls": 0,
        "teacher_failures": 0,
        # Answered with nothing at all: not cached, so raising the token budget
        # and re-running is enough to recover the prompt.
        "teacher_blank": 0,
        "cache_misses": 0,
        "validator_errors": 0,
    }
    # An unreachable endpoint, a wrong base URL or a revoked key fails every
    # prompt identically. Stop instead of walking the whole pool to produce a
    # run that accepted nothing.
    max_consecutive_failures = 5
    consecutive_failures = 0
    aborted_reason: Optional[str] = None

    for prompt, declared in prompt_rows:
        phash = _prompt_hash(prompt)
        key = TeacherCache.key(phash, cfg.teacher_model, cfg.teacher_revision)
        response = cache.get(key)
        if response is not None:
            stats["cache_hits"] += 1
        elif offline:
            # Replay / offline never reaches the network: an uncached prompt is
            # a skip, so a replay reproduces exactly what was recorded.
            stats["cache_misses"] += 1
            rejected.append({"prompt_sha256": phash, "_reject_reason": "cache_miss"})
            continue
        else:
            if teacher is None:
                # `timeout` rides in request_params but belongs to the client:
                # a local reasoning teacher on long documents routinely needs
                # more than the 60 s default.
                params = dict(cfg.request_params)
                timeout = params.pop("timeout", None)
                if timeout is not None:
                    teacher = TeacherClient(model=cfg.teacher_model, timeout=float(timeout))
                else:
                    teacher = TeacherClient(model=cfg.teacher_model)
            try:
                response = teacher.chat_completions(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    **params,
                )
            except Exception as exc:  # noqa: BLE001  a teacher failure is a rejected row
                stats["teacher_failures"] += 1
                consecutive_failures += 1
                rejected.append({"prompt_sha256": phash, "_reject_reason": f"teacher_error: {exc}"})
                if consecutive_failures >= max_consecutive_failures:
                    aborted_reason = (
                        f"{consecutive_failures} consecutive teacher failures; last error: {exc}"
                    )
                    break
                continue
            stats["teacher_calls"] += 1
            consecutive_failures = 0
            # An empty reply is not a label, and caching one is permanent: the
            # prompt would return blank on every later run, including
            # `--replay`, with no way back short of editing the cache by hand.
            # A reasoning teacher that spends its whole token budget thinking
            # returns exactly this, so it is a budget away from being a good
            # label rather than an answer worth recording.
            if response.strip():
                cache.put(key, response)
            else:
                stats["teacher_blank"] += 1

        target = _parse_target(response, cfg.target_format)
        ok = False
        reason = "unparseable"
        if target is not None:
            reason = "invalid_target"
            try:
                ok = bool(validate_fn(prompt, target))
            except Exception as exc:  # noqa: BLE001  a raising validator is a rejected row
                # Without this, one malformed target aborts the whole run and
                # discards every accepted row plus any teacher response since
                # the last cache flush.
                reason = f"validator_error: {exc}"
                stats["validator_errors"] += 1
        if not ok:
            rejected.append(
                {
                    "prompt_sha256": phash,
                    request_field: prompt[:500],
                    "teacher_response": response[:1500],
                    "_reject_reason": reason,
                }
            )
            continue

        row = {
            # What the pool declared and distill does not define itself: the
            # group key `dataset.group_by` names, and anything else the pool
            # carries. Written first so a reserved field cannot be shadowed.
            **{
                k: v
                for k, v in declared.items()
                if k not in _RESERVED_ROW_FIELDS and k != target_field
            },
            "sample_id": f"distill-{cfg.teacher_model}-{phash}",
            "source": "distill",
            "family": f"{cfg.family}:{phash}",
            request_field: prompt,
            target_field: target,
            "provenance": {
                "teacher_model": cfg.teacher_model,
                "teacher_revision": cfg.teacher_revision,
                "prompt_sha256": phash,
            },
        }
        accepted.append(row)

    if not (replay or offline):
        cache.flush()

    seed_target = out_path or cfg.output
    dest = (
        model_def.resolve(seed_target)
        if seed_target
        else model_def.resolve(
            str(ds_cfg.get("seed_samples") or "datasets/samples/seed_samples.jsonl")
        )
    )
    duplicates = 0
    seed_written = False
    if accepted:
        existing = list(iter_jsonl(dest)) if (append and dest.is_file()) else []
        seen = {_row_id(r) for r in existing}
        fresh = [r for r in accepted if _row_id(r) not in seen]
        duplicates = len(accepted) - len(fresh)
        if fresh:
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl_atomic(dest, existing + fresh)
            seed_written = True
        accepted = fresh

    reject_path = dest.with_name(dest.stem + ".distill_rejected.jsonl")
    if rejected:
        write_jsonl_atomic(reject_path, rejected)
    else:
        reject_path.unlink(missing_ok=True)

    summary = {
        "prompts": len(prompt_rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duplicates": duplicates,
        "seed_written": seed_written,
        "aborted": aborted_reason,
        "replay": bool(replay or offline),
        "teacher_model": cfg.teacher_model,
        "teacher_revision": cfg.teacher_revision,
        "out_path": str(dest),
        "cache_path": str(cache.path),
        "cache_provenance": {k: v for k, v in cache.provenance.items() if k != "kind"},
        **stats,
    }
    _write_distill_card(dest, summary)
    return summary


def _parse_target(response: str, target_format: str = "json") -> Optional[Any]:
    """The teacher's target, ready for the validator.

    Uses the canonical ``strip_fences`` so a reasoning teacher that emits a
    ``<think>`` block in the message content parses instead of being rejected
    as unparseable; a local copy here previously handled fences only.

    Under ``text`` the reply is the target. An empty reply is still nothing,
    so it is refused here rather than passed on as a label.
    """
    stripped = strip_fences(response)
    if target_format == "text":
        return stripped if stripped.strip() else None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


def _write_distill_card(dest: Path, summary: dict[str, Any]) -> Path:
    lines = [
        f"# distill card: {dest.name}",
        "",
        f"- teacher: {summary['teacher_model']}@{summary['teacher_revision']}",
        f"- prompts: {summary['prompts']}",
        f"- accepted: {summary['accepted']}",
        f"- rejected: {summary['rejected']}",
        f"- duplicates_skipped: {summary['duplicates']}",
        f"- replay: {summary['replay']}",
        f"- cache: hits={summary['cache_hits']} calls={summary['teacher_calls']} "
        f"misses={summary['cache_misses']} failures={summary['teacher_failures']}",
        f"- validator_errors: {summary['validator_errors']}",
        *([f"- aborted: {summary['aborted']}"] if summary.get("aborted") else []),
        f"- cache provenance: {summary.get('cache_provenance')}",
        "",
        "Every accepted row was gated by evaluation.validator before entering "
        "the corpus, and carries teacher provenance. Teacher responses are "
        "recorded in the cache above; re-run with --replay to reproduce this "
        "corpus offline.",
    ]
    card = dest.with_name(dest.stem + ".distill_card.md")
    card.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return card
