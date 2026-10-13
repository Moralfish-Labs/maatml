"""maatml distill: validator gating, provenance, and offline replay.

A fake in-process teacher stands in for the OpenAI-compatible client, so the
gating and record/replay behaviour is tested without a network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.data import distill as distill_mod
from maatml.data.distill import (
    DistillConfig,
    DistillConfigError,
    TeacherCache,
    _prompt_hash,
    run_distill,
)
from maatml.registry import register_validator
from maatml.utils.io import iter_jsonl, write_jsonl
from maatml.validation.base import ValidationError, ValidationResult

PROMPTS = ["good prompt one", "good prompt two", "bad prompt"]


@register_validator("distill_test_validator")
def _validator(raw_output, *, schema_path=None, contracts_path=None, user_prompt=None, **_kw):
    """Accepts any JSON object whose 'ok' is True."""
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        result.errors.append(ValidationError(layer=1, code="bad_json", message="x"))
        return result
    result.parsed = parsed
    if isinstance(parsed, dict) and parsed.get("ok") is True:
        result.passed_layers.add(1)
    else:
        result.errors.append(ValidationError(layer=1, code="not_ok", message="ok!=True"))
    return result


class _FakeTeacher:
    """Returns a canned label per prompt; the 'bad' prompt yields a failing one."""

    def __init__(self, *args, **kwargs) -> None:
        _FakeTeacher.calls = getattr(_FakeTeacher, "calls", 0)

    def chat_completions(self, messages, **kwargs):
        type(self).calls = getattr(type(self), "calls", 0) + 1
        prompt = messages[-1]["content"]
        if "bad" in prompt:
            return json.dumps({"ok": False, "why": "wrong"})
        return json.dumps({"ok": True, "label": prompt.upper()})


def _model(
    tmp_path: Path,
    *,
    distill_section: dict | None = None,
    validator: str = "distill_test_validator",
    prompts: list[str] | None = None,
):
    mdir = tmp_path / "model"
    (mdir / "datasets").mkdir(parents=True, exist_ok=True)
    write_jsonl(
        mdir / "datasets" / "prompts.jsonl",
        [{"request": p} for p in (prompts if prompts is not None else PROMPTS)],
    )
    section = distill_section or {
        "prompt_source": "datasets/prompts.jsonl",
        "teacher_model": "fake",
        "teacher_revision": "v1",
        "cache": "datasets/cache.jsonl",
    }
    body = {
        "name": "distill-test",
        "model_id": "distill-test",
        "architecture": "causal_sft",
        "version": "0.1.0",
        "dataset": {
            "seed_samples": "datasets/samples/seed_samples.jsonl",
            "target_field": "expected_output",
        },
        "evaluation": {"validator": validator},
        "distill": section,
    }
    import yaml

    (mdir / "model.yml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_model_def(mdir)


@pytest.fixture
def teacher(monkeypatch):
    _FakeTeacher.calls = 0
    monkeypatch.setattr(distill_mod, "TeacherClient", _FakeTeacher)
    return _FakeTeacher


# --- gating ----------------------------------------------------------------


def test_rejected_teacher_row_is_absent_from_the_corpus(tmp_path, teacher) -> None:
    """Exit criterion: a validator-rejected teacher label never enters seeds."""
    md = _model(tmp_path)
    summary = run_distill(md)

    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    rows = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    assert len(rows) == 2
    # The bad prompt's label (ok:false) is nowhere in the corpus.
    assert all(row["expected_output"]["ok"] is True for row in rows)
    assert all("bad" not in row["request"] for row in rows)

    reject_path = md.resolve("datasets/samples/seed_samples.distill_rejected.jsonl")
    rejects = list(iter_jsonl(reject_path))
    assert rejects[0]["_reject_reason"] == "invalid_target"


def test_accepted_rows_carry_provenance(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    run_distill(md)
    row = next(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    prov = row["provenance"]
    assert prov["teacher_model"] == "fake"
    assert prov["teacher_revision"] == "v1"
    assert prov["prompt_sha256"] == row["family"].split(":")[1]
    assert row["source"] == "distill"


def test_distill_requires_a_validator(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    md.evaluation.pop("validator")
    with pytest.raises(DistillConfigError, match="requires evaluation.validator"):
        run_distill(md)


# --- record / replay -------------------------------------------------------


def test_live_run_records_a_cache_that_replays_offline(tmp_path, teacher) -> None:
    """Exit criterion: replay with no network reproduces the accepted corpus."""
    md = _model(tmp_path)
    live = run_distill(md)
    assert teacher.calls == len(PROMPTS)
    cache_path = md.resolve("datasets/cache.jsonl")
    assert cache_path.is_file()

    # Wipe the produced corpus, then replay from the cache with the teacher
    # replaced by one that raises if it is ever called.
    md.resolve("datasets/samples/seed_samples.jsonl").unlink()

    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("replay must not construct a teacher")

    import maatml.data.distill as mod

    mod.TeacherClient = _NoNetwork
    try:
        replayed = run_distill(md, replay=True)
    finally:
        mod.TeacherClient = _FakeTeacher

    assert replayed["accepted"] == live["accepted"] == 2
    assert replayed["cache_hits"] == len(PROMPTS)
    replay_rows = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    assert len(replay_rows) == 2


def test_offline_run_skips_uncached_prompts(tmp_path) -> None:
    md = _model(tmp_path)

    # No cache exists and no teacher is available: every prompt is a cache miss.
    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("offline must not construct a teacher")

    import maatml.data.distill as mod

    mod.TeacherClient = _NoNetwork
    try:
        summary = run_distill(md, offline=True)
    finally:
        mod.TeacherClient = _FakeTeacher

    assert summary["accepted"] == 0
    assert summary["cache_misses"] == len(PROMPTS)
    assert not md.resolve("datasets/samples/seed_samples.jsonl").is_file()


def test_a_different_teacher_does_not_reuse_cached_labels(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    run_distill(md)  # records under fake@v1

    # A different revision is a different cache key, so replay finds nothing.
    md.distill["teacher_revision"] = "v2"
    md.resolve("datasets/samples/seed_samples.jsonl").unlink()
    summary = run_distill(md, replay=True)
    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == len(PROMPTS)


def test_second_live_run_dedups(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    run_distill(md)
    again = run_distill(md)
    assert again["duplicates"] == 2
    assert again["accepted"] == 0
    assert len(list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))) == 2


# --- prompt loading + config ----------------------------------------------


def test_load_prompts_from_jsonl_and_text(tmp_path) -> None:
    from maatml.data.distill import load_prompts

    (tmp_path / "p.jsonl").write_text(
        json.dumps({"request": "a"}) + "\n" + json.dumps({"prompt": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_prompts(tmp_path / "p.jsonl", "request") == ["a", "b"]
    (tmp_path / "p.txt").write_text("line one\nline two\n\n", encoding="utf-8")
    assert load_prompts(tmp_path / "p.txt", "request") == ["line one", "line two"]


def test_a_pool_field_travels_onto_the_accepted_row(tmp_path, teacher) -> None:
    """A pool that groups its prompts keeps the grouping through distillation.

    ``dataset.group_by`` names a field on the row. A pool declaring one whose
    accepted rows do not carry it splits silently by something else, and
    paraphrases of a single situation land on both sides of the split.
    """
    md = _model(tmp_path)
    write_jsonl(
        md.resolve("datasets/prompts.jsonl"),
        [
            {"request": "good prompt one", "scenario": "read:a", "family": "pool_wide"},
            {"request": "good prompt two", "scenario": "read:a"},
        ],
    )
    run_distill(md)
    rows = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))

    assert [row["scenario"] for row in rows] == ["read:a", "read:a"]
    # `family` stays distill's own, one per prompt. A pool-wide family taken at
    # face value would collapse every accepted row into a single split group.
    assert rows[0]["family"] != rows[1]["family"]
    assert not any(row["family"].startswith("pool_wide") for row in rows)


def test_cache_key_binds_prompt_and_teacher() -> None:
    a = TeacherCache.key(_prompt_hash("x"), "m", "r1")
    b = TeacherCache.key(_prompt_hash("x"), "m", "r2")
    assert a != b


def test_config_rejects_unknown_keys() -> None:
    # pydantic's ValidationError subclasses ValueError; matching the key
    # keeps this specific instead of a blind `Exception`.
    with pytest.raises(ValueError, match="teecher_model"):
        DistillConfig(prompt_source="p.jsonl", teecher_model="typo")


# --- the shipped triage example --------------------------------------------

REPO = Path(__file__).resolve().parents[1]
TRIAGE = REPO / "examples" / "support-ticket-triage"


def test_triage_distill_replays_offline_from_the_shipped_cache(tmp_path, monkeypatch) -> None:
    """The example replays with no network and rejects the wrong-routing label.

    Mirrors the CI exit criterion without touching the committed corpus: the
    prompt pool and cache ship in the repo; output goes to a temp seed file.
    """
    import sys

    sys.path.insert(0, str(TRIAGE))
    from maatml.registry import discover_plugins, load_model_plugins

    # force=True: the autouse registry-restore fixture may have dropped the
    # triage validator that an earlier test loaded, while load_model_plugins
    # still records the module as imported.
    discover_plugins()
    load_model_plugins(TRIAGE, ["./triage_plugin"], force=True)

    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("the shipped replay must not hit the network")

    monkeypatch.setattr(distill_mod, "TeacherClient", _NoNetwork)

    md = load_model_def(TRIAGE)
    out = tmp_path / "seeds.jsonl"
    summary = run_distill(md, replay=True, append=False, out_path=str(out))

    # Four prompts, three valid labels, one rejected by the routing contract.
    assert summary["prompts"] == 4
    assert summary["accepted"] == 3
    assert summary["rejected"] == 1
    assert summary["cache_hits"] == 4

    rows = list(iter_jsonl(out))
    teams = {r["expected_output"]["team"] for r in rows}
    assert teams == {"payments", "identity", "docs"}
    # The billing->platform row violates the routing contract and is absent.
    assert "platform" not in teams
    for row in rows:
        assert row["provenance"]["teacher_model"] == "recorded-teacher"


# --- request params and cache durability -----------------------------------


def test_request_params_reach_the_teacher_call(tmp_path, teacher, monkeypatch) -> None:
    """distill.request_params is merged into the chat payload; timeout goes to
    the client constructor instead."""
    seen: dict = {}

    class _Recording(_FakeTeacher):
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs
            super().__init__(*args, **kwargs)

        def chat_completions(self, messages, **kwargs):
            seen["request_kwargs"] = kwargs
            return super().chat_completions(messages, **kwargs)

    monkeypatch.setattr(distill_mod, "TeacherClient", _Recording)
    md = _model(
        tmp_path,
        distill_section={
            "prompt_source": "datasets/prompts.jsonl",
            "teacher_model": "fake",
            "teacher_revision": "v1",
            "cache": "datasets/cache.jsonl",
            "request_params": {
                "timeout": 300,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    )
    run_distill(md, append=False, out_path=str(tmp_path / "out.jsonl"))

    assert seen["client_kwargs"] == {"model": "fake", "timeout": 300.0}
    assert seen["request_kwargs"] == {
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_cache_flushes_incrementally(tmp_path) -> None:
    """A crash mid-run must not lose every teacher response: the cache hits
    disk every FLUSH_EVERY puts, not only at the final flush."""
    cache = TeacherCache(tmp_path / "cache.jsonl")
    for index in range(TeacherCache.FLUSH_EVERY):
        cache.put(f"k{index}", "v")
    on_disk = list(iter_jsonl(tmp_path / "cache.jsonl"))
    assert len(on_disk) == TeacherCache.FLUSH_EVERY
    # A rewritten identical value stays clean: no dirty flag, no rewrite.
    cache.put("k0", "v")
    assert cache._dirty is False


# --- robustness ------------------------------------------------------------


def test_parse_target_accepts_reasoning_teacher_output() -> None:
    """A reasoning teacher emits its trace in the message content. The local
    fence-only parser rejected all of it as unparseable; _parse_target now uses
    the canonical strip_fences, which also drops <think> blocks."""
    from maatml.data.distill import _parse_target

    assert _parse_target('<think>weighing it up</think>{"ok": true}') == {"ok": True}
    assert _parse_target('```json\n{"ok": true}\n```') == {"ok": True}
    assert _parse_target('<think>hmm</think>\n```json\n{"ok": true}\n```') == {"ok": True}
    assert _parse_target("not json at all") is None


def test_validator_exception_rejects_the_row_without_aborting(tmp_path, monkeypatch):
    """A validator that raises on one row must not discard the whole run: the
    row is rejected and counted, and the good rows still reach the corpus."""
    from maatml.registry import register_validator

    @register_validator("distill_raises_on_bad")
    def _raiser(raw_output, **_kw):
        parsed = json.loads(raw_output)
        # Only the 'bad prompt' response carries this key.
        if parsed.get("why") == "wrong":
            raise RuntimeError("validator blew up")
        result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
        result.parsed = parsed
        result.passed_layers.add(1)
        return result

    md = _model(tmp_path, validator="distill_raises_on_bad")
    monkeypatch.setattr(distill_mod, "TeacherClient", _FakeTeacher)

    summary = run_distill(md)
    assert summary["validator_errors"] == 1
    assert summary["accepted"] == 2
    assert summary["aborted"] is None
    reasons = [
        r["_reject_reason"]
        for r in iter_jsonl(
            Path(summary["out_path"]).with_name(
                Path(summary["out_path"]).stem + ".distill_rejected.jsonl"
            )
        )
    ]
    assert any(r.startswith("validator_error:") for r in reasons)


def test_consecutive_teacher_failures_abort_the_run(tmp_path, monkeypatch) -> None:
    """An unreachable endpoint fails every prompt identically; stop rather than
    walking the whole pool to produce a run that accepted nothing."""

    class _DeadTeacher:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def chat_completions(self, messages, **kwargs):
            raise ConnectionError("connection refused")

    md = _model(tmp_path, prompts=[f"prompt {i}" for i in range(20)])
    monkeypatch.setattr(distill_mod, "TeacherClient", _DeadTeacher)

    summary = run_distill(md)
    assert summary["accepted"] == 0
    assert summary["aborted"] is not None
    assert "consecutive teacher failures" in summary["aborted"]
    # Stopped early rather than calling the dead endpoint for every prompt.
    assert summary["teacher_failures"] == 5


def test_a_blank_teacher_reply_is_not_cached(tmp_path, monkeypatch) -> None:
    """A reasoning teacher that spends its budget thinking answers with nothing.

    Caching that would be permanent: the prompt returns blank on every later
    run, `--replay` included, and no token budget can recover it.
    """

    class _Mute:
        def __init__(self, *a, **k) -> None:
            pass

        def chat_completions(self, messages, **kwargs):
            return "   "

    monkeypatch.setattr(distill_mod, "TeacherClient", _Mute)
    md = _model(tmp_path)
    summary = run_distill(md)

    assert summary["accepted"] == 0
    assert summary["teacher_blank"] == len(PROMPTS)
    # Nothing recorded at all, so raising the budget and re-running reaches the
    # teacher again rather than replaying the silence.
    cache_path = md.resolve("datasets/cache.jsonl")
    assert not cache_path.is_file() or not list(iter_jsonl(cache_path))


def test_a_string_target_is_not_wrapped_in_quotes() -> None:
    """`target_format: text` writes a string target, which is the text itself.

    Serialising it would train the model to answer inside a string literal:
    correct content, every quote escaped, and nothing downstream accepts it.
    """
    from maatml.training.sft_config import render_assistant_target

    said = 'Reading the file.\n{"calls":[{"name":"read_file","input":{"path":"a"}}]}'
    assert render_assistant_target({"expected": said}, "expected") == said
    # A structured target still serialises, compactly.
    assert render_assistant_target({"expected": {"ok": True}}, "expected") == '{"ok":true}'


def test_a_relocated_run_resolves_to_the_canonical_checkpoint_dir(tmp_path) -> None:
    """A run trained elsewhere records that machine's absolute path.

    The weights come home to `output/checkpoints/<run_id>`; the path in the
    record still points at the runtime that is gone. Resolving by run id has to
    find them, or the bundle exports from an anonymous directory and its
    manifest records no gate evidence.
    """
    from maatml.runs import RunRecord, _append_record, resolve_checkpoint

    md = _model(tmp_path)
    run_id = "20260813-024700-13417d"
    landed = md.checkpoints_dir / run_id
    landed.mkdir(parents=True, exist_ok=True)
    (landed / "model.safetensors").write_bytes(b"weights")

    _append_record(
        md,
        RunRecord(
            run_id=run_id,
            identity="distill-test@0.1.0",
            architecture="causal_sft",
            status="completed",
            started_at="2026-08-13T02:47:00Z",
            out_dir="/content/gone/output/checkpoints/" + run_id,
        ),
    )
    assert resolve_checkpoint(md, run_id) == landed.resolve()
