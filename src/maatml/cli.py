"""maatml CLI.

Each command takes a model folder (containing ``model.yml``) and dispatches via
the plugin registry (``architecture`` / ``dataset.format`` / ``evaluation``).
Outputs land under ``<model-dir>/output/`` (gitignored).

  maatml prepare   <model-dir>
  maatml train     <model-dir> [--smoke] [--resume auto|PATH] [--set K=V]
  maatml sweep     <model-dir> --param K=a,b [--metric NAME] [--smoke]
  maatml evaluate  <model-dir> [--checkpoint X] [--split test] [--gate] [--set K=V]
  maatml export    <model-dir> [--checkpoint X] [--format gguf|mlx|safetensors|onnx]
  maatml compile   <export-dir> --target NAME --out DIR [--option K=V]
  maatml verify    <export-dir-or-manifest>
  maatml manifest  amend <export-dir> <files...> [--format NAME]
  maatml serve     <model-dir> [--server NAME] [--server-option K=V] …
  maatml datagen   <model-dir> [--target N] [--teacher]
  maatml ingest    <model-dir> --input PATH [--map field=col] [--sanitize tag]
  maatml runs      <model-dir>
  maatml scaffold  <dir> --architecture causal_sft [--name my-model]
  maatml validate  <model-dir>
  maatml plan      <model-dir>
  maatml plugins
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from . import __version__
from .config import config_key_warnings, get_dataset_cfg, load_model_def
from .overrides import (
    apply_overrides,
    expand_param_grid,
    minimizes,
    overrides_from_mapping,
    parse_override,
    pick_metric,
)
from .registry import (
    FORMATS,
    TRAINERS,
    discover_plugins,
    list_all_plugins,
    load_errors,
    load_model_plugins,
)
from .runs import list_runs, resolve_checkpoint
from .scaffold import normalize_architecture, scaffold_model, validate_model_dir

# MPS unsupported-op fallback to CPU; harmless on non-Mac.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


app = typer.Typer(no_args_is_help=True, add_completion=False, help="maatml CLI")
console = Console()

_STATE: dict[str, bool] = {"debug": False}

# Failures that mean "the input or the model folder is wrong", not "maatml
# broke". They print one actionable line; --debug restores the traceback.
_USER_ERRORS = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
    ValueError,
    KeyError,
    ImportError,
)


def _version_callback(value: bool) -> None:
    """Prints `maatml <version>` and exits, so tool probes can read the version."""
    if value:
        console.print(f"maatml {__version__}")
        raise typer.Exit


@app.callback()
def _main_callback(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print full tracebacks for user errors (missing files, invalid "
        "model.yml, unknown plugins) instead of a single line.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the maatml version and exit.",
    ),
) -> None:
    """maatml CLI."""
    _STATE["debug"] = debug


def _user_message(exc: BaseException) -> str:
    """One-line message for a user error (KeyError's str() adds quotes)."""
    if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc) or type(exc).__name__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boot_plugins(md) -> None:
    discover_plugins()
    if md.plugins:
        load_model_plugins(md.model_dir, md.plugins)


def _clone_model_def(md):
    """Deep-copy nested dict sections so sweep trials don't share state."""
    clone = md.model_copy(deep=True)
    object.__setattr__(clone, "model_dir", md.model_dir)
    return clone


def _print_run_comparison(
    records, metrics: Optional[list[str]], *, include_telemetry: bool = False
) -> None:
    """Render `runs --compare` as a run-by-metric table."""
    from rich.table import Table

    from .runs import compare_runs

    keys, rows, hidden = compare_runs(records, metrics=metrics, include_telemetry=include_telemetry)
    # Metrics run down the left and runs across the top: a run usually reports
    # more metrics than there are runs to compare, and this way the metric
    # names stay readable instead of being truncated into column headers.
    table = Table(title=f"{len(rows)} run(s)")
    table.add_column("metric", no_wrap=True)
    for row in rows:
        table.add_column(row["run_id"], justify="right", no_wrap=True)

    table.add_row("status", *[row["status"] for row in rows])
    table.add_row("smoke", *["yes" if row["smoke"] else "no" for row in rows])
    table.add_row(
        "gates",
        *[
            "-" if row["gates_passed"] is None else ("pass" if row["gates_passed"] else "fail")
            for row in rows
        ],
    )
    for key in keys:
        # A metric a run never reported stays "-" instead of reading as 0.
        table.add_row(
            key,
            *["-" if row["metrics"][key] is None else f"{row['metrics'][key]:.4g}" for row in rows],
        )
    console.print(table)
    if not keys:
        console.print("[dim]no metrics recorded on these runs[/]")
    if hidden:
        console.print(
            f"[dim]{len(hidden)} timing metric(s) hidden "
            f"({', '.join(hidden)}); --all-metrics to include[/]"
        )


def _parse_field_maps(maps: Optional[list[str]]) -> dict[str, str]:
    """Parse ``dest=src`` field maps for ingest."""
    out: dict[str, str] = {}
    for item in maps or []:
        if "=" not in item:
            raise typer.BadParameter(f"Expected field=col mapping, got {item!r}")
        dest, src = item.split("=", 1)
        dest, src = dest.strip(), src.strip()
        if not dest or not src:
            raise typer.BadParameter(f"Empty field map entry: {item!r}")
        out[dest] = src
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("prepare")
def cmd_prepare(
    model_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Path to a model folder containing model.yml",
    ),
) -> None:
    """Build train/val/test splits under <model-dir>/output/prepared/."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    fmt = get_dataset_cfg(md).get("format", "jsonl_seed")
    prepare_fn = FORMATS.require(str(fmt))
    prepare_fn(md)


@app.command("train")
def cmd_train(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    smoke: bool = typer.Option(False, "--smoke", help="Use the `smoke:` overrides in model.yml"),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Cap number of train rows for ad-hoc smoke"
    ),
    device: str = typer.Option("auto", "--device", help="auto|mps|cpu|cuda"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    seeds: Optional[int] = typer.Option(
        None,
        "--seeds",
        help="Train the same recipe N times (seed, seed+1, ...) and record the spread "
        "under output/seeds/; gates derive --seed-study takes the per-metric minimum",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume training: 'auto' (latest incomplete run) or checkpoint path/run_id",
    ),
    set_overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override model.yml after load (repeatable), e.g. --set training.learning_rate=1e-4",
    ),
) -> None:
    """Fine-tune the model declared by <model-dir>/model.yml."""
    md = load_model_def(model_dir)
    if set_overrides:
        try:
            apply_overrides(md, set_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--set") from exc
    _boot_plugins(md)
    arch = normalize_architecture(md.architecture)
    trainer = TRAINERS.get(md.architecture) or TRAINERS.require(arch)
    from .training.selection import render_selection, select_checkpoint

    if seeds is None:
        result = trainer(md, smoke=smoke, limit=limit, device=device, seed=seed, resume=resume)
        console.print(f"[green]done[/] out_dir={result.out_dir} metrics={result.metrics}")
        selection = select_checkpoint(md, Path(result.out_dir).name, device=device, smoke=smoke)
        if selection:
            for line in render_selection(selection):
                console.print(line)
        return
    if seeds < 2:
        raise typer.BadParameter("--seeds needs at least 2", param_hint="--seeds")
    if resume:
        raise typer.BadParameter(
            "--seeds cannot resume; each seed is its own run", param_hint="--resume"
        )
    from .training.seeds import load_seed_study, render_seed_study, write_seed_study

    base = seed if seed is not None else 0
    seed_values = [base + i for i in range(seeds)]
    runs: list[dict[str, Any]] = []
    for value in seed_values:
        console.print(f"[cyan]seeds[/] {len(runs) + 1}/{seeds} seed={value}")
        result = trainer(md, smoke=smoke, limit=limit, device=device, seed=value)
        metrics = dict(result.metrics or {})
        selection = select_checkpoint(md, Path(result.out_dir).name, device=device, smoke=smoke)
        if selection:
            chosen = next(c for c in selection["candidates"] if c["name"] == selection["selected"])
            metrics[f"select:{selection['metric']}"] = chosen["value"]
            console.print(f"  selected {selection['selected']} ({chosen['value']:.4f})")
        runs.append({"run_id": Path(result.out_dir).name, "seed": value, "metrics": metrics})
    label = f"{runs[0]['run_id']}-x{seeds}"
    path = write_seed_study(md, label=label, seeds=seed_values, runs=runs, smoke=smoke)
    for line in render_seed_study(load_seed_study(path)):
        console.print(line)
    console.print(f"[green]seed study[/] {path}")


@app.command("sweep")
def cmd_sweep(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    param: Optional[list[str]] = typer.Option(
        None,
        "--param",
        help="Grid axis KEY=v1,v2 (repeatable); cartesian product of values",
    ),
    set_overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Base overrides applied to every trial (repeatable)",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Metric key to rank trials (default: first numeric metric)",
    ),
    smoke: bool = typer.Option(False, "--smoke"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    device: str = typer.Option("auto", "--device"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    max_trials: Optional[int] = typer.Option(
        None, "--max-trials", help="Cap number of grid combinations"
    ),
) -> None:
    """Offline cartesian HPO over ``--param`` axes (no Optuna required)."""
    base = load_model_def(model_dir)
    _boot_plugins(base)
    grid = expand_param_grid(param, max_trials=max_trials)
    if not grid:
        console.print("[yellow]empty grid — nothing to run[/]")
        raise typer.Exit(code=0)

    arch = normalize_architecture(base.architecture)
    trainer = TRAINERS.get(base.architecture) or TRAINERS.require(arch)
    scored: list[tuple[float, dict, Any, str]] = []
    failures: list[tuple[dict, str]] = []
    skipped: list[tuple[dict, Optional[str]]] = []

    for i, trial_map in enumerate(grid):
        md = _clone_model_def(base)
        try:
            if set_overrides:
                apply_overrides(md, set_overrides)
            apply_overrides(md, overrides_from_mapping(trial_map))
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--set/--param") from exc
        trial_meta = {"index": i, "params": trial_map}
        console.print(f"[cyan]sweep[/] trial {i + 1}/{len(grid)} params={trial_map}")
        # One bad combination used to abort the whole sweep after the earlier
        # trials had already trained. Record it and keep going; the exit code
        # still reports the failure at the end.
        try:
            result = trainer(
                md,
                smoke=smoke,
                limit=limit,
                device=device,
                seed=seed,
                trial=trial_meta,
            )
        except Exception as exc:  # noqa: BLE001  a trial failure is data, not a crash
            failures.append((trial_map, f"{type(exc).__name__}: {exc}"))
            console.print(f"  [red]-> trial failed[/] {type(exc).__name__}: {exc}")
            continue
        key, val = pick_metric(result.metrics, metric)
        console.print(
            f"  -> run={Path(result.out_dir).name} "
            f"{key or 'metric'}={val if val is not None else 'n/a'}"
        )
        if val is None or key is None:
            skipped.append((trial_map, key))
            continue
        scored.append((val, trial_map, result, key))

    # Rank only trials that reported the same metric: comparing eval_loss
    # against, say, accuracy would order them by an accident of naming.
    ranked_key = metric or (scored[0][3] if scored else None)
    comparable = [entry for entry in scored if entry[3] == ranked_key]
    incomparable = [entry for entry in scored if entry[3] != ranked_key]

    if comparable:
        minimize = minimizes(ranked_key)
        comparable.sort(key=lambda t: t[0], reverse=not minimize)
        direction = "lower is better" if minimize else "higher is better"
        console.print(f"[bold]sweep ranking[/] ({ranked_key}, {direction})")
        for rank, (val, trial_map, result, key) in enumerate(comparable, start=1):
            console.print(
                f"  {rank}. {key}={val:.6g}  params={trial_map}  out={Path(result.out_dir).name}"
            )
    else:
        console.print("[yellow]sweep finished with no numeric metrics to rank[/]")

    for trial_map, _key in skipped:
        console.print(f"[yellow]unranked[/] params={trial_map} (no numeric metric)")
    for _val, trial_map, _result, key in incomparable:
        console.print(
            f"[yellow]unranked[/] params={trial_map} reported {key!r}, not {ranked_key!r}"
        )
    if failures:
        console.print(f"[red]{len(failures)} trial(s) failed[/]")
        for trial_map, err in failures:
            console.print(f"  params={trial_map}: {err}")
        raise typer.Exit(code=1)


@app.command("evaluate")
def cmd_evaluate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="run_id or checkpoint dir; defaults to latest completed run",
    ),
    split: str = typer.Option("test", "--split"),
    device: str = typer.Option("auto", "--device"),
    baseline: Optional[Path] = typer.Option(None, "--baseline"),
    max_input_tokens: Optional[int] = typer.Option(
        None,
        "--max-input-tokens",
        help="Input token budget; defaults to packaging.max_input_tokens so "
        "eval measures the same budget serve and export --parity enforce",
    ),
    limit: Optional[int] = typer.Option(None, "--limit"),
    gate: bool = typer.Option(
        False,
        "--gate",
        help="Enforce evaluation.gates minima; exit non-zero on failure",
    ),
    max_regression: list[str] = typer.Option(
        [],
        "--max-regression",
        help="With --baseline: fail when a gated metric drops more than this "
        "(e.g. 0.03). Repeatable; metric=0.05 overrides one metric, and an "
        "override may also name an ungated metric.",
    ),
    cache: Optional[bool] = typer.Option(
        None,
        "--cache/--no-cache",
        help="Keep every row's prediction beside the report "
        "(<run>.predictions.jsonl) so floors and threshold sweeps derive from "
        "it without re-running inference; defaults to evaluation.cache_predictions",
    ),
    strict_population: bool = typer.Option(
        False,
        "--strict-population",
        help="With --gate: refuse when evaluation.gates_benchmark is not this "
        "split's content hash, instead of warning",
    ),
    blind: bool = typer.Option(
        False,
        "--blind",
        help="Spend dataset.blind_samples once on a candidate whose production gate pass "
        "is current (gates enforced; writes <run>.blind.json; recorded on the run)",
    ),
    force: bool = typer.Option(
        False, "--force", help="With --blind: repeat a spend on the same frozen candidate"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Rows per predict_batch call (defaults to evaluation.batch_size, 1); the "
        "outputs are identical, so this is a device knob, not an override",
    ),
    set_overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override model.yml for this evaluate only (repeatable), e.g. "
        "--set evaluation.score_thresh=0.05 to cache val predictions at a low cut "
        "for operating-point derive; recorded as extras.overrides, refused with "
        "--gate / --blind",
    ),
) -> None:
    """Evaluate a checkpoint and write report.{json,md} under output/eval/."""
    md = load_model_def(model_dir)
    applied: dict[str, Any] = {}
    if set_overrides:
        try:
            apply_overrides(md, set_overrides)
            applied = dict(parse_override(spec) for spec in set_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--set") from exc
    _boot_plugins(md)
    from .evaluation.harness import GateConfigError
    from .evaluation.runner import evaluate_model

    if not (md.evaluation or {}).get("validator"):
        console.print("[yellow]no validator configured, scoring JSON parse only[/]")

    # evaluate_model performs every configuration check before it resolves or
    # loads a checkpoint, so a misconfigured model.yml says so instead of
    # reporting "no checkpoints found" (or spending minutes loading weights).
    try:
        report, out_path = evaluate_model(
            md,
            checkpoint=checkpoint,
            split=split,
            device=device,
            baseline=baseline,
            max_input_tokens=max_input_tokens,
            limit=limit,
            gate=gate,
            cache_predictions=cache,
            strict_population=strict_population,
            blind=blind,
            force=force,
            overrides=applied or None,
            batch_size=batch_size,
        )
    except GateConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="evaluation") from exc
    except KeyError as exc:
        raise typer.BadParameter(_user_message(exc), param_hint="evaluation") from exc

    console.print(f"[green]done[/] report={out_path}")
    if max_regression:
        from .evaluation.harness import regression_failures

        if baseline is None:
            raise typer.BadParameter(
                "--max-regression needs --baseline to diff against",
                param_hint="--max-regression",
            )
        default_max: Optional[float] = None
        overrides: dict[str, float] = {}
        for entry in max_regression:
            key, sep, value = entry.partition("=")
            try:
                if sep:
                    overrides[key.strip()] = float(value)
                else:
                    default_max = float(entry)
            except ValueError as exc:
                raise typer.BadParameter(
                    f"expected a number or metric=number, got {entry!r}",
                    param_hint="--max-regression",
                ) from exc
        if not report.baseline_delta:
            console.print(
                "[red]regression check failed[/] the baseline report shares no "
                "metrics with this run"
            )
            raise typer.Exit(code=1)
        gate_keys = set(((md.evaluation or {}).get("gates") or {}).keys())
        failures = regression_failures(report.baseline_delta, gate_keys, default_max, overrides)
        if failures:
            console.print("[red]regressions beyond the allowed drop[/]")
            for failure in failures:
                console.print(f"  - {failure}")
            raise typer.Exit(code=1)
        console.print("[green]no regressions beyond the allowed drop[/]")
    if gate and report.passed is False:
        console.print("[red]eval gates failed[/]")
        raise typer.Exit(code=1)


@app.command("export")
def cmd_export(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="run_id or checkpoint dir; defaults to latest completed run",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        help="safetensors (default) | gguf | mlx | onnx (plugin-registered)",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Export directory (default: <model-dir>/output/export/<run_id>)",
    ),
    parity: bool = typer.Option(
        False,
        "--parity",
        help="Re-run evaluation.gates on dataset.benchmark_samples after export",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Export a checkpoint as a deployable bundle with manifest.json."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .export.bundle import export_model, run_parity_check
    from .runs import run_id_for_checkpoint

    ckpt = resolve_checkpoint(md, checkpoint)
    run_id = run_id_for_checkpoint(md, ckpt)
    out_dir = out if out is not None else (md.output_dir / "export" / run_id)
    try:
        export_model(md, ckpt, out_dir, format=format, run_id=run_id)
    except (ImportError, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
        console.print(f"[red]export failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]exported[/] {out_dir}")

    if parity:
        result = run_parity_check(md, out_dir, device=device)
        if result.get("skipped"):
            console.print(f"[yellow]parity skipped[/] {result.get('reason')}")
        elif not result.get("passed"):
            console.print(f"[red]parity gates failed[/] {result.get('gates')}")
            raise typer.Exit(code=1)
        else:
            console.print("[green]parity ok[/]")


@app.command("verify")
def cmd_verify(
    path: Path = typer.Argument(
        ...,
        help="Export directory or path to manifest.json",
    ),
) -> None:
    """Recompute sha256 of files listed in an export manifest; exit 1 on mismatch."""
    from .export.manifest import verify_manifest

    errors = verify_manifest(path)
    if errors:
        console.print("[red]verify failed[/]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/] {path}")


manifest_app = typer.Typer(no_args_is_help=True, help="Inspect or amend export manifests")
app.add_typer(manifest_app, name="manifest")


gates_app = typer.Typer(no_args_is_help=True, help="Derive evaluation gates from measured reports")
app.add_typer(gates_app, name="gates")


@gates_app.command("derive")
def cmd_gates_derive(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    run: list[str] = typer.Option(
        [],
        "--run",
        help="Run id or report path; repeat to take the per-metric minimum across seeds",
    ),
    seed_study: Optional[Path] = typer.Option(
        None,
        "--seed-study",
        help="A train --seeds record under output/seeds/; its runs join --run",
    ),
    metric: list[str] = typer.Option(
        [],
        "--metric",
        help="Metric (or slice:<field>=<value>) to floor; defaults to the configured "
        "gates, then to every rate the reports carry counts for",
    ),
    section: str = typer.Option("evaluation", "--section", help="evaluation or smoke"),
    min_n: int = typer.Option(30, "--min-n", help="Refuse a rate with a thinner denominator"),
    min_groups: int = typer.Option(
        3, "--min-groups", help="Refuse a clustered rate spanning fewer groups"
    ),
    cluster_by: str = typer.Option(
        "family",
        "--cluster-by",
        help="Row field whose values are resampled by the cluster bootstrap "
        "(needs the run's predictions cache)",
    ),
    write: bool = typer.Option(
        False, "--write", help="Replace <section>.gates in model.yml and stamp gates_benchmark"
    ),
) -> None:
    """Wilson / cluster-bootstrap floors from eval reports, with the measurement beside each."""
    md = load_model_def(model_dir)
    from .evaluation.gates import DeriveError, derive_gates, write_gates
    from .evaluation.harness import GateConfigError, gate_tiers

    runs = list(run)
    if seed_study is not None:
        from .training.seeds import load_seed_study

        try:
            study = load_seed_study(seed_study)
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--seed-study") from exc
        runs.extend(r["run_id"] for r in study.get("runs") or [])
    if not runs:
        raise typer.BadParameter("give --run (repeatable) or --seed-study", param_hint="--run")
    try:
        result = derive_gates(
            md,
            runs=runs,
            metrics=metric or None,
            section=section,
            min_n=min_n,
            min_groups=min_groups,
            cluster_by=cluster_by,
        )
        tiers = gate_tiers(md, smoke=section == "smoke")
    except (DeriveError, GateConfigError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--run") from exc

    console.print(f"benchmark {result.benchmark_sha256[:16]}  runs {', '.join(result.runs)}")
    for line in result.lines():
        console.print(line)
    for refusal in result.refusals:
        console.print(f"[yellow]refused[/] {refusal}")
    if write:
        try:
            path = write_gates(md.resolve("model.yml"), result, tiers)
        except DeriveError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]wrote[/] {len(result.floors)} floors -> {path} ({section}.gates)")


operating_point_app = typer.Typer(
    no_args_is_help=True, help="Choose a decision threshold on val and spend test once"
)
app.add_typer(operating_point_app, name="operating-point")


@operating_point_app.command("derive")
def cmd_operating_point_derive(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    run: str = typer.Option(
        ..., "--run", help="Run id whose --split report carries a predictions cache"
    ),
    split: str = typer.Option(
        "val", "--split", help="Split the cache was evaluated on (never test)"
    ),
    grid: list[float] = typer.Option([], "--grid", help="Override evaluation.operating_point.grid"),
    write: bool = typer.Option(
        False, "--write", help="Write evaluation.<threshold_key> into model.yml with provenance"
    ),
    confirm_on_test: bool = typer.Option(
        False,
        "--confirm-on-test",
        help="Evaluate the run once on test at the chosen threshold and record the spend "
        "on the run (implies --write)",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Sweep the predictor's rescore() over cached val predictions and pick under the budget."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .evaluation.harness import GateConfigError
    from .evaluation.operating_point import (
        OperatingPointError,
        derive_operating_point,
        render,
        write_artifact,
        write_threshold,
    )

    try:
        result = derive_operating_point(md, run=run, split=split, grid=grid or None)
    except (OperatingPointError, GateConfigError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--run") from exc
    artifact = write_artifact(md, result)
    for line in render(result):
        console.print(line)
    console.print(f"[dim]sweep -> {artifact}[/]")
    if not (write or confirm_on_test):
        return
    try:
        path = write_threshold(md.resolve("model.yml"), result, artifact)
    except OperatingPointError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    assert result.chosen is not None
    chosen = float(result.chosen["threshold"])
    console.print(f"[green]wrote[/] evaluation.{result.spec.threshold_key}: {chosen:g} -> {path}")
    if not confirm_on_test:
        return

    from .evaluation.runner import evaluate_model
    from .runs import record_test_spend

    # The written cut is what test is spent at; the in-memory model_def is
    # updated so the predictor decodes at it without re-reading the file.
    md.evaluation[result.spec.threshold_key] = chosen
    try:
        report, report_path = evaluate_model(
            md, checkpoint=run, split="test", device=device, gate=True, cache_predictions=True
        )
    except (GateConfigError, KeyError) as exc:
        raise typer.BadParameter(_user_message(exc), param_hint="evaluation") from exc
    spend = {
        "benchmark_sha256": report.extras.get("split_sha256"),
        "split": "test",
        "threshold_key": result.spec.threshold_key,
        "threshold": chosen,
        "report": report_path.name,
        "val_split_sha256": result.split_sha256,
    }
    rec, prior = record_test_spend(md, run, spend)
    if rec is None:
        console.print(
            f"[yellow]warning[/] {run} is not in runs.jsonl; the test spend was not recorded"
        )
    elif prior:
        console.print(
            f"[yellow]warning[/] test split {str(spend['benchmark_sha256'])[:16]} was already "
            f"spent {prior} time(s) for {run}; this confirmation is the {prior + 1}th"
        )
    else:
        console.print(f"[green]confirmed on test[/] {report_path.name}; spend recorded on {run}")


@app.command("ship-check")
def cmd_ship_check(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    candidate: str = typer.Argument(
        ..., help="Candidate run id (its eval report under output/eval/)"
    ),
    baseline: str = typer.Argument(..., help="Accepted release run id"),
    max_regression: Optional[float] = typer.Option(
        None,
        "--max-regression",
        help="Fixed allowed drop per gated metric instead of the one-row tolerance",
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="When the two reports were measured on different splits, re-evaluate both "
        "checkpoints over the current test split (under output/eval/replay/) and judge those",
    ),
    device: str = typer.Option("auto", "--device"),
    as_json: bool = typer.Option(False, "--json", help="Print the verdict as JSON"),
) -> None:
    """Absolute floors, delta vs the baseline, and a controlled replay when the split changed."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .evaluation.harness import (
        GateConfigError,
        Report,
        ReportSchemaError,
        effective_gates,
        gate_tiers,
    )
    from .evaluation.shipcheck import render_verdict, ship_check

    def _load(run: str, directory: Path) -> Report:
        path = directory / f"{run}.json"
        if not path.is_file():
            raise typer.BadParameter(f"no eval report for {run!r} at {path}", param_hint="run")
        try:
            return Report.read(path, strict=True)
        except ReportSchemaError as exc:
            raise typer.BadParameter(f"{run}: {exc}", param_hint="run") from exc

    try:
        gates = effective_gates(md)
        tiers = gate_tiers(md)
    except GateConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="evaluation") from exc
    if not gates:
        raise typer.BadParameter(
            "no evaluation.gates configured; nothing to check against", param_hint="evaluation"
        )

    cand = _load(candidate, md.eval_dir)
    base = _load(baseline, md.eval_dir)
    replayed = False
    cand_split = (cand.gates or {}).get("benchmark_sha256") or cand.extras.get("split_sha256")
    base_split = (base.gates or {}).get("benchmark_sha256") or base.extras.get("split_sha256")
    if replay and cand_split != base_split:
        from .evaluation.runner import evaluate_model

        replay_dir = md.eval_dir / "replay"
        console.print(
            f"[cyan]replay[/] splits differ ({str(cand_split)[:16]} vs {str(base_split)[:16]}); "
            f"re-evaluating both over the current test split -> {replay_dir}"
        )
        try:
            for run in (candidate, baseline):
                evaluate_model(
                    md,
                    checkpoint=run,
                    split="test",
                    device=device,
                    gate=True,
                    out_dir=replay_dir,
                    record_gates=False,
                )
        except (GateConfigError, KeyError) as exc:
            raise typer.BadParameter(_user_message(exc), param_hint="evaluation") from exc
        cand = _load(candidate, replay_dir)
        base = _load(baseline, replay_dir)
        replayed = True

    verdict = ship_check(
        cand,
        base,
        gates=gates,
        tiers=tiers,
        max_regression=max_regression,
        replayed=replayed,
        candidate_label=candidate,
        baseline_label=baseline,
    )
    if as_json:
        import json as _json

        console.print(_json.dumps(verdict.as_dict(), indent=2, sort_keys=True))
    else:
        for line in render_verdict(verdict):
            console.print(line)
    if not verdict.ship:
        raise typer.Exit(code=1)


@manifest_app.command("amend")
def cmd_manifest_amend(
    export_dir: Path = typer.Argument(
        ..., exists=True, help="Export directory or path to manifest.json"
    ),
    files: list[Path] = typer.Argument(..., help="Files inside the export dir to add"),
    format: list[str] = typer.Option(
        [],
        "--format",
        help="Format name(s) to append to runtime_hints.formats (repeatable)",
    ),
) -> None:
    """Add out-of-band artifacts (a quantized GGUF, say) to an existing manifest.

    The files gain sha256 entries so `maatml verify` covers them and a serving
    catalog's checksum can be traced to a gate-evidenced export. Everything
    else in the manifest, gate evidence included, is left untouched.
    """
    from .export.manifest import amend_manifest

    try:
        manifest = amend_manifest(export_dir, files, formats=format or None)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]amend failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]amended[/] {len(files)} file(s); manifest now lists "
        f"{len(manifest.get('files', []))} file(s)"
    )


@app.command("compile")
def cmd_compile(
    export_dir: Path = typer.Argument(
        ...,
        exists=True,
        help="Portable export directory (must contain manifest.json)",
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Registered compiler (e.g. sip_tensorrt, fake_vllm). See `maatml plugins`.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Directory for the target-specific deployment bundle",
    ),
    option: Optional[list[str]] = typer.Option(
        None,
        "--option",
        help="Compiler option KEY=VALUE (repeatable). Format-agnostic.",
    ),
    require_gated: bool = typer.Option(
        False,
        "--require-gated",
        help="Refuse unless manifest.gate_evidence is a non-smoke production "
        "pass (passed=true, smoke_gated=false). Promotion compiles should "
        "set this so a rehearsal cannot become a device artifact.",
    ),
) -> None:
    """Compile a portable export into a device-specific runtime artifact.

    Compilers are plugins: TensorRT plans, vLLM packages, GGUF quantize, etc.
    Core writes ``target_manifest.json`` recording the source identity and
    ``promotion_eligible``.
    """
    from .compile import compile_export, parse_options

    discover_plugins()
    try:
        opts = parse_options(option)
        result = compile_export(
            export_dir,
            target=target,
            out_dir=out,
            options=opts,
            require_gated=require_gated,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        console.print(f"[red]compile failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]compiled[/] target={target} → {result}")


@app.command("serve")
def cmd_serve(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="run_id, checkpoint dir, or export dir; defaults to latest completed run",
    ),
    server: str = typer.Option(
        "http",
        "--server",
        help="Registered server backend (default: http). See `maatml plugins`.",
    ),
    server_option: Optional[list[str]] = typer.Option(
        None,
        "--server-option",
        help="Server option KEY=VALUE (repeatable). Format-agnostic.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8080, "--port", help="Bind port"),
    device: str = typer.Option("auto", "--device"),
    cors: Optional[str] = typer.Option(
        None,
        "--cors",
        help=(
            "Enable CORS for this origin ('*' or e.g. https://app.example.com). "
            "Off by default; also reads MAATML_SERVE_CORS."
        ),
    ),
    max_body_bytes: int = typer.Option(
        1_048_576,
        "--max-body-bytes",
        help="Reject POST bodies larger than this many bytes (default 1 MiB).",
    ),
    enforce: bool = typer.Option(
        False,
        "--enforce",
        help="Return HTTP 422 when the configured validator rejects a prediction "
        "(gate live inference). Off by default (annotate only).",
    ),
    max_retries: int = typer.Option(
        0,
        "--max-retries",
        help="On a validation failure under --enforce, feed the error back and "
        "re-ask the model up to this many times. Retries are counted in the "
        "response. Needs evaluation.validator.",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help="Require this bearer token on /predict (also reads "
        "MAATML_SERVE_TOKEN). Required for --capture, and for a non-loopback "
        "bind unless --allow-unauthenticated is passed.",
    ),
    allow_unauthenticated: bool = typer.Option(
        False,
        "--allow-unauthenticated",
        help="Permit a non-loopback bind with no auth token. Only for a "
        "trusted network: anyone who can reach the port can query the model.",
    ),
    capture: Optional[Path] = typer.Option(
        None,
        "--capture",
        help="Append served predictions to this JSONL for review. Captured rows "
        "are NOT gold: ingest refuses them until approved. Requires --auth-token.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Include the exception message and traceback in 500 responses. "
        "Off by default so internals never leak to clients.",
    ),
) -> None:
    """Serve a model via a registered backend (default: JSON HTTP API).

    ``--server http`` (default) exposes GET /health, GET /info, POST /predict.
    Other backends (DeepStream, vLLM, llama.cpp, …) are plugins that ignore the
    HTTP-specific flags and read ``--server-option KEY=VALUE`` instead.
    """
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .compile import parse_options
    from .servers import dispatch_server

    cors_origin = cors if cors is not None else os.environ.get("MAATML_SERVE_CORS")
    token = auth_token if auth_token is not None else os.environ.get("MAATML_SERVE_TOKEN")
    try:
        opts = parse_options(server_option)
        dispatch_server(
            server,
            md,
            checkpoint=checkpoint,
            options=opts,
            host=host,
            port=port,
            device=device,
            cors_origin=cors_origin,
            max_body_bytes=max_body_bytes,
            enforce=enforce,
            debug=debug,
            auth_token=token,
            max_retries=max_retries,
            capture_path=capture,
            allow_unauthenticated=allow_unauthenticated,
        )
    except (FileNotFoundError, KeyError, ImportError, RuntimeError, ValueError) as exc:
        console.print(f"[red]serve failed[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("datagen")
def cmd_datagen(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    target: int = typer.Option(100, "--target", help="Number of accepted samples"),
    seed: int = typer.Option(0, "--seed"),
    out: Optional[Path] = typer.Option(None, "--out", help="Override seed JSONL path"),
    teacher: bool = typer.Option(
        False,
        "--teacher",
        help="Use OpenAI-compatible teacher (MAATML_TEACHER_* env); requires [teacher]",
    ),
    append: bool = typer.Option(True, "--append/--no-append"),
    allow_ungated: bool = typer.Option(
        False,
        "--allow-ungated",
        help="Permit datagen with no evaluation.validator; the run and dataset "
        "card are marked UNGATED (rows are NOT validator-gated).",
    ),
) -> None:
    """Generate validator-gated seed rows via a registered generator or teacher."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .data.datagen import DatagenConfigError, run_datagen
    from .data.gated import GenerationAbort

    try:
        result = run_datagen(
            md,
            target=target,
            seed=seed,
            out_path=out,
            use_teacher=teacher,
            append=append,
            allow_ungated=allow_ungated,
        )
    except DatagenConfigError as exc:
        console.print(f"[red]datagen refused[/] {exc}")
        raise typer.Exit(code=1) from exc
    except GenerationAbort as exc:
        console.print(f"[red]datagen aborted[/] {exc}")
        raise typer.Exit(code=1) from exc
    except (KeyError, ImportError) as exc:
        console.print(f"[red]datagen failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    status = "GATED" if result["gated"] else "UNGATED"
    console.print(
        f"[green]datagen[/] status={status} generator={result['generator']} "
        f"accepted={result['accepted']} rejected={result['rejected']} "
        f"duplicates={result['duplicates']} out={result['out_path']}"
    )
    if result["teacher_failures"]:
        console.print(
            f"[yellow]note[/] teacher request failures: {result['teacher_failures']} "
            f"(see {result['card_path']})"
        )
    if result["protected_existing"]:
        console.print("[yellow]note[/] nothing new accepted; existing seed file left unchanged")


@app.command("distill")
def cmd_distill(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    prompts: Optional[str] = typer.Option(
        None, "--prompts", help="Prompt pool path (JSONL or text); overrides distill.prompt_source"
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use only cached teacher responses (no network); reproduces a "
        "prior run's corpus offline",
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Never call the teacher; an uncached prompt is skipped"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap prompts processed"),
    out: Optional[str] = typer.Option(None, "--out", help="Override seed JSONL path"),
    append: bool = typer.Option(True, "--append/--no-append"),
) -> None:
    """Label a prompt pool with a validator-gated teacher (data flywheel).

    Every teacher label is gated by evaluation.validator before it enters the
    seed corpus; accepted rows carry teacher provenance. Teacher responses are
    cached, so `--replay` reproduces the same corpus with no network.
    """
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .data.distill import DistillConfigError, run_distill

    try:
        result = run_distill(
            md,
            prompt_source=prompts,
            replay=replay,
            offline=offline,
            limit=limit,
            append=append,
            out_path=out,
        )
    except DistillConfigError as exc:
        console.print(f"[red]distill refused[/] {exc}")
        raise typer.Exit(code=1) from exc
    mode = "replay" if result["replay"] else "live"
    console.print(
        f"[green]distill[/] mode={mode} teacher={result['teacher_model']} "
        f"prompts={result['prompts']} accepted={result['accepted']} "
        f"rejected={result['rejected']} duplicates={result['duplicates']} "
        f"out={result['out_path']}"
    )
    if result["teacher_failures"]:
        console.print(f"[yellow]note[/] teacher request failures: {result['teacher_failures']}")
    if result["replay"] and result["cache_misses"]:
        console.print(
            f"[yellow]note[/] {result['cache_misses']} prompt(s) not in the cache "
            "were skipped (record a live run first)"
        )
    if result.get("validator_errors"):
        console.print(
            f"[yellow]note[/] validator raised on {result['validator_errors']} "
            "row(s); they were rejected, see the rejection report"
        )
    if result.get("aborted"):
        console.print(f"[red]distill aborted[/] {result['aborted']}")
        raise typer.Exit(code=1)
    # A run over a non-empty pool that accepted nothing gained no data. Exiting
    # zero here would let a scheduled or CI-driven flywheel report success while
    # the teacher was unreachable or misconfigured.
    if result["prompts"] and not result["accepted"] and not result["duplicates"]:
        console.print(
            "[red]distill failed[/] no rows accepted from "
            f"{result['prompts']} prompt(s); nothing was added to the corpus"
        )
        raise typer.Exit(code=1)


@app.command("ingest")
def cmd_ingest(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        help="JSON or JSONL file to ingest (annotation sidecar when --video is set)",
    ),
    video: Optional[Path] = typer.Option(
        None,
        "--video",
        exists=True,
        help="Extract frames named by the sidecar (frame / timestamp_ms / t) "
        "into datasets/samples/images and set the request field to each PNG. "
        "Needs ffmpeg. Annotation dialects stay in the sidecar, not in core.",
    ),
    fps: Optional[float] = typer.Option(
        None,
        "--fps",
        help="Frames per second used when the sidecar has timestamp_ms or t "
        "instead of frame (default 30).",
    ),
    field_map: Optional[list[str]] = typer.Option(
        None,
        "--map",
        help="Map dest=src fields (repeatable), e.g. --map request=text",
    ),
    sanitize: Optional[str] = typer.Option(
        None, "--sanitize", help="Sanitizer registry tag (e.g. pii)"
    ),
    append: bool = typer.Option(True, "--append/--no-append"),
    out: Optional[Path] = typer.Option(None, "--out", help="Override seed JSONL path"),
) -> None:
    """Ingest external samples into seed_samples with optional validation."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .data.ingest import ingest_samples

    try:
        result = ingest_samples(
            md,
            input_path,
            field_map=_parse_field_maps(field_map),
            sanitize_tag=sanitize,
            append=append,
            out_path=out,
            video=video,
            fps=fps,
        )
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]ingest failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]ingest[/] accepted={result['accepted']} "
        f"rejected={result['rejected']} "
        f"skipped_unvalidated={result['skipped_unvalidated']} "
        f"seeds={result['seeds_path']}"
    )
    if result.get("unapproved_capture"):
        console.print(
            f"[yellow]note[/] refused {result['unapproved_capture']} unapproved "
            "serve-capture row(s); set approved: true after review to ingest them"
        )
    if result.get("protected_existing"):
        console.print(
            "[yellow]note[/] nothing accepted; left the existing seed corpus "
            "untouched rather than rewriting it"
        )


@app.command("mint")
def cmd_mint(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    input_path: Path = typer.Option(
        ..., "--input", exists=True, help="JSONL of {prompt, candidates: [...]}"
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Override seed JSONL path"),
    append: bool = typer.Option(True, "--append/--no-append"),
) -> None:
    """Mint dpo/orpo preference pairs from candidate completions.

    For each prompt the model's registered validator splits its candidates into
    pass / fail; a prompt with both yields one {prompt, chosen, rejected} pair.
    An explicit source op, never a default `run` step.
    """
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .data.preference import MintConfigError, run_mint

    try:
        result = run_mint(md, input_path, out_path=out, append=append)
    except MintConfigError as exc:
        console.print(f"[red]mint refused[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]mint[/] prompts={result['prompts']} pairs={result['pairs']} "
        f"duplicates={result['duplicates']} malformed={result['malformed']} "
        f"out={result['out_path']}"
    )
    if result["pairs"] == 0:
        console.print(
            "[yellow]note[/] no pairs minted; each prompt needs at least one "
            "passing and one failing candidate"
        )


@app.command("runs")
def cmd_runs(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Print a run-by-metric table instead of one line per run",
    ),
    metric: Optional[list[str]] = typer.Option(
        None,
        "--metric",
        help="Restrict --compare to these metric keys (repeatable)",
    ),
    all_metrics: bool = typer.Option(
        False,
        "--all-metrics",
        help="Include trainer timing metrics (runtime, samples/s) in --compare",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the most recent N runs"),
    pack: Optional[str] = typer.Option(
        None,
        "--pack",
        help="Write RUN as a portable bundle (run dir without checkpoint-*, its eval "
        "reports and caches, the record, the environment) under output/bundles/",
    ),
    with_checkpoints: bool = typer.Option(
        False, "--with-checkpoints", help="Include checkpoint-* dirs in --pack"
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Bundle path or directory for --pack"),
    adopt: Optional[Path] = typer.Option(
        None, "--adopt", help="Unpack a bundle into this model folder and append its record"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="--adopt: accept a bundle from another model.yml / identity, or overwrite a run",
    ),
) -> None:
    """List training runs recorded in <model-dir>/output/runs.jsonl."""
    md = load_model_def(model_dir)
    if pack or adopt:
        from .portable import BundleError, adopt_bundle, pack_run

        try:
            if pack and adopt:
                raise BundleError("--pack and --adopt are separate operations")
            if pack:
                path = pack_run(md, pack, out=out, with_checkpoints=with_checkpoints)
                console.print(f"[green]packed[/] {pack} -> {path}")
            else:
                assert adopt is not None
                rec = adopt_bundle(md, adopt, force=force)
                console.print(f"[green]adopted[/] {rec.run_id} -> {rec.out_dir}")
        except BundleError as exc:
            console.print(f"[red]runs refused[/] {exc}")
            raise typer.Exit(code=1) from exc
        return
    records = list_runs(md)
    if limit is not None and limit > 0:
        records = records[-limit:]
    if not records:
        console.print(f"[dim]no runs yet under {md.output_dir}[/]")
        return

    if compare:
        _print_run_comparison(records, metric, include_telemetry=all_metrics)
        return

    for rec in records:
        metrics = ""
        if rec.metrics:
            top = list(rec.metrics.items())[:3]
            metrics = " " + " ".join(f"{k}={v:.4g}" for k, v in top)
        spends = ""
        if rec.test_spends:
            benches = sorted({str(s.get("benchmark_sha256"))[:8] for s in rec.test_spends})
            spends = f"  test-spends={len(rec.test_spends)} ({', '.join(benches)})"
        if rec.blind_spends:
            spends += f"  blind-spends={len(rec.blind_spends)}"
        console.print(
            f"{rec.run_id}  [{rec.status}]  smoke={rec.smoke}  "
            f"device={rec.device or '-'}  {rec.out_dir}{metrics}{spends}"
        )


@app.command("report")
def cmd_report(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    format: str = typer.Option("md", "--format", help="md | csv"),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write here instead of stdout (default output/REPORT.md / .csv)"
    ),
    stdout: bool = typer.Option(False, "--stdout", help="Print instead of writing a file"),
) -> None:
    """Runs, floors with their derivation, slices, seed statistics and spends —
    regenerated from output/runs.jsonl, output/eval/*.json and output/seeds/*.json alone."""
    from .report import build_report, render_csv, render_markdown

    md = load_model_def(model_dir)
    data = build_report(md.output_dir)
    if format not in ("md", "csv"):
        raise typer.BadParameter("--format must be md or csv", param_hint="--format")
    text = render_markdown(data) if format == "md" else render_csv(data)
    if stdout:
        console.print(text, markup=False, highlight=False, end="")
        return
    target = out if out is not None else md.output_dir / f"REPORT.{format}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    console.print(f"[green]report[/] {target}")


@app.command("scaffold")
def cmd_scaffold(
    target_dir: Path = typer.Argument(..., help="Directory to create (model folder)"),
    architecture: str = typer.Option(
        ...,
        "--architecture",
        "-a",
        help="Registered trainer architecture (e.g. causal_sft, seq2seq, classifier, dpo)",
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Model name (default: folder name)"),
    plugin: Optional[list[str]] = typer.Option(
        None,
        "--plugin",
        help="Plugin to load before resolving --architecture, and record in the "
        "new model.yml: a folder/.py path or an installed module (repeatable). "
        "Required for plugin-owned architectures such as vision_multitask.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing model.yml and seed_samples.jsonl",
    ),
) -> None:
    """Create a new model folder with model.yml, datasets, and README."""
    discover_plugins()
    try:
        path = scaffold_model(
            target_dir,
            architecture=architecture,
            name=name,
            force=force,
            plugins=list(plugin or []),
        )
    except FileExistsError as exc:
        console.print(f"[red]scaffold refused[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]scaffolded[/] {path}")


@app.command("validate")
def cmd_validate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    no_plugins: bool = typer.Option(
        False,
        "--no-plugins",
        help="Validate schema and paths without importing trainer or model "
        "plugin code (a model folder is executable code).",
    ),
) -> None:
    """Validate model.yml paths, architecture registration, and dataset.format."""
    errors = validate_model_dir(model_dir, load_plugins=not no_plugins)
    if errors:
        console.print("[red]validate failed[/]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)
    md = load_model_def(model_dir, load_plugins=not no_plugins)
    for warn in config_key_warnings(md):
        console.print(f"[yellow]warning[/] {warn}")
    if no_plugins:
        console.print("[dim]architecture and dataset.format not verified (--no-plugins)[/]")
    console.print(f"[green]OK[/] {md.identity} ({md.architecture})")


def _print_pipeline_plan(md, plans) -> None:
    """Per-step fresh/stale table with the reason a step is stale."""
    from rich.table import Table

    console.print(f"[bold]{md.identity}[/] ({md.architecture})")
    table = Table()
    table.add_column("step", no_wrap=True)
    table.add_column("action")
    table.add_column("reason")
    marks = {"run": "[yellow]run[/]", "skip": "[green]skip[/]", "not selected": "[dim]-[/]"}
    for plan in plans:
        table.add_row(plan.name, marks.get(plan.action, plan.action), plan.reason)
    console.print(table)


def _lifecycle_options(
    smoke: bool,
    device: str,
    force: bool,
    from_step: Optional[str],
    until_step: Optional[str],
    seed: Optional[int],
    limit: Optional[int],
    export_format: Optional[str],
):
    from .lifecycle import RunOptions

    return RunOptions(
        smoke=smoke,
        device=device,
        force=force,
        from_step=from_step,
        until_step=until_step,
        seed=seed,
        limit=limit,
        export_format=export_format,
    )


@app.command("run")
def cmd_run(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    smoke: bool = typer.Option(
        False, "--smoke", help="Use the smoke: overlay, including smoke.gates"
    ),
    force: bool = typer.Option(False, "--force", help="Re-run every selected step, fresh or not"),
    from_step: Optional[str] = typer.Option(
        None, "--from", help="First step to run (prepare|train|evaluate|export|verify)"
    ),
    until_step: Optional[str] = typer.Option(None, "--until", help="Last step to run"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show each step's fresh/stale status and stop"
    ),
    device: str = typer.Option("auto", "--device"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    export_format: Optional[str] = typer.Option(
        None, "--format", help="Export format (default safetensors)"
    ),
    set_overrides: Optional[list[str]] = typer.Option(
        None, "--set", help="Override model.yml (repeatable); feeds the fingerprint"
    ),
) -> None:
    """Run the fixed lifecycle: prepare, train, evaluate (gated), export, verify.

    Steps that are already fresh are skipped; anything else re-runs. The first
    failure stops the run with a non-zero exit, so a green line means every
    stage passed. `datagen` / `ingest` stay outside: they change the seed
    corpus, which is what makes prepare stale.
    """
    md = load_model_def(model_dir)
    # Overrides are validated here, before they reach a fingerprint or any
    # step: an invalid --set must not leave state behind.
    if set_overrides:
        try:
            apply_overrides(md, set_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--set") from exc
    _boot_plugins(md)

    from .evaluation.harness import GateConfigError
    from .lifecycle import (
        _resolve_paths,
        _selected_steps,
        plan_pipeline,
        run_pipeline,
        validate_run_config,
    )

    options = _lifecycle_options(
        smoke, device, force, from_step, until_step, seed, limit, export_format
    )
    # A bad step name is a usage error; a config the run cannot use is a
    # refusal. Both are reported before any step runs, and before a dry run
    # prints a plan that could not be executed.
    try:
        selected = _selected_steps(from_step, until_step)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--from/--until") from exc

    try:
        validate_run_config(md, smoke=smoke, steps=selected)
    except (GateConfigError, ValueError, KeyError) as exc:
        console.print(f"[red]run refused[/] {_user_message(exc)}")
        raise typer.Exit(code=1) from exc

    checkpoint, export_dir, report = _resolve_paths(md)
    plans = plan_pipeline(
        md,
        smoke=smoke,
        device=device,
        force=force,
        from_step=from_step,
        until_step=until_step,
        checkpoint=checkpoint,
        export_dir=export_dir,
        export_format=export_format,
        eval_report=report,
        limit=limit,
        seed=seed,
    )

    if dry_run:
        _print_pipeline_plan(md, plans)
        return

    def _on_step(name: str, status: str, detail: str) -> None:
        colour = {"running": "cyan", "ran": "green", "skipped": "dim", "failed": "red"}
        console.print(f"[{colour.get(status, 'white')}]{status:>8}[/] {name}  {detail}")

    result = run_pipeline(md, options, on_step=_on_step)
    if not result.ok:
        console.print(f"[red]lifecycle failed at {result.failed}[/]")
        raise typer.Exit(code=1)
    ran = [o.name for o in result.outcomes if o.status == "ran"]
    console.print(
        f"[green]lifecycle complete[/] ran={ran or 'nothing (all fresh)'}"
        + (" [yellow](smoke tier)[/]" if smoke else "")
    )


@app.command("plan")
def cmd_plan(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    smoke: bool = typer.Option(False, "--smoke"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Show which lifecycle steps are stale (alias for `run --dry-run`)."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .lifecycle import _resolve_paths, plan_pipeline

    checkpoint, export_dir, report = _resolve_paths(md)
    _print_pipeline_plan(
        md,
        plan_pipeline(
            md,
            smoke=smoke,
            device=device,
            checkpoint=checkpoint,
            export_dir=export_dir,
            eval_report=report,
        ),
    )
    console.print("[dim]run it with: maatml run " + str(md.model_dir) + "[/]")


@app.command("audit")
def cmd_audit(
    model_dir: Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        help="Optional model folder to check as well as the environment",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the diagnostics as JSON instead of a table"
    ),
) -> None:
    """Report environment, plugin, and model-folder health. Exits 1 on errors."""
    import json as _json

    from .audit import ERROR, OK, WARN, collect_diagnostics

    diag = collect_diagnostics(model_dir)

    if json_out:
        console.print_json(_json.dumps(diag.as_dict()))
    else:
        from rich.markup import escape

        marks = {OK: "[green]ok[/]", WARN: "[yellow]warn[/]", ERROR: "[red]error[/]"}
        for section, checks in diag.sections.items():
            console.print(f"[bold]{section}[/]")
            for check in checks:
                # Details mention extras like [ml] / [vision]; escape them so
                # rich prints the name instead of eating it as markup.
                console.print(
                    f"  {marks[check.status]} {escape(check.name)}: {escape(check.detail)}"
                )

    errors = diag.errors
    if errors:
        # --json output stays parseable: the exit code carries the verdict.
        if not json_out:
            console.print(f"[red]{len(errors)} problem(s) found[/]")
        raise typer.Exit(code=1)


@app.command("rollout")
def cmd_rollout(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    env: Optional[str] = typer.Option(
        None, "--env", help="A registered environment plugin, e.g. line_world"
    ),
    env_command: Optional[str] = typer.Option(
        None,
        "--env-command",
        help="Command serving the maatml.rollout_env/1 protocol over stdio",
    ),
    tasks: str = typer.Option(
        "{}", "--tasks", help="JSON task query passed to the environment's `tasks`"
    ),
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", help="run_id or checkpoint dir; defaults to latest completed run"
    ),
    device: str = typer.Option("auto", "--device"),
    label: bool = typer.Option(
        False, "--label", help="Record the environment teacher's valid set at every state"
    ),
    teacher_mix: float = typer.Option(
        0.0, "--teacher-mix", help="Probability of acting with the teacher (needs --label)"
    ),
    option: list[str] = typer.Option(
        [], "--option", help="KEY=VALUE passed to `reset` (repeatable), e.g. extra_steps=8"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Episode provenance; defaults to dagger with --label, else rollout"
    ),
    round_index: int = typer.Option(0, "--round"),
    seed: int = typer.Option(0, "--seed"),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Defaults to <model-dir>/output/episodes/<source>-r<round>"
    ),
) -> None:
    """Act with a checkpoint in an environment and store the episodes it verifies."""
    import json
    import shlex

    from .device import resolve_device
    from .registry import ENVIRONMENTS, PREDICTORS
    from .rollout import RolloutConfig, StdioEnvironment, rollout, write_episodes
    from .runs import resolve_checkpoint, run_id_for_checkpoint

    if bool(env) == bool(env_command):
        raise ValueError("pass exactly one of --env and --env-command")
    md = load_model_def(model_dir)
    _boot_plugins(md)

    name = (md.evaluation or {}).get("predictor") or md.architecture
    predictor = PREDICTORS.require(str(name))
    if isinstance(predictor, type):
        predictor = predictor()
    if not callable(getattr(predictor, "score", None)):
        raise ValueError(
            f"predictor {name!r} has no score(observation, candidates); "
            "a rollout policy must rank an environment's candidates"
        )
    checkpoint_dir = resolve_checkpoint(md, checkpoint)
    predictor.setup(checkpoint_dir, model_def=md, device=resolve_device(device))

    options: dict[str, object] = {}
    for item in option:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--option wants KEY=VALUE, not {item!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    source = source or ("dagger" if label else "rollout")
    config = RolloutConfig(
        source=source,
        policy_version=run_id_for_checkpoint(md, checkpoint_dir),
        round=round_index,
        label=label,
        teacher_mix=teacher_mix,
        options=options,
        seed=seed,
    )
    out_dir = out or md.model_dir / "output" / "episodes" / f"{source}-r{round_index}"
    environment = (
        StdioEnvironment(shlex.split(env_command))
        if env_command
        else ENVIRONMENTS.require(str(env))()
    )
    try:
        task_list = environment.tasks(json.loads(tasks))
        if not task_list:
            raise ValueError(f"the environment returned no tasks for {tasks}")
        summary = write_episodes(rollout(environment, predictor, task_list, config=config), out_dir)
    finally:
        if isinstance(environment, StdioEnvironment):
            environment.shutdown()
    console.print(
        f"[green]rollout complete[/] {summary['success']}/{summary['episodes']} verified "
        f"({summary['success_rate']:.1%}), {summary['decisions']} decisions, "
        f"{summary['truncated']} truncated -> {out_dir}"
    )
    for family, bucket in summary["families"].items():
        console.print(f"  {family}: {bucket['success']}/{bucket['episodes']}")


@app.command("plugins")
def cmd_plugins() -> None:
    """List registered trainers, validators, metrics, formats, servers, compilers, environments."""
    discover_plugins()
    # Import submodule directly so listing plugins does not require torch.
    from . import servers as _servers  # noqa: F401
    from .evaluation import predictors as _predictors  # noqa: F401
    from .export import bundle as _bundle  # noqa: F401
    from .export import gguf as _gguf  # noqa: F401
    from .export import mlx_export as _mlx  # noqa: F401

    for kind, entries in list_all_plugins().items():
        console.print(f"[bold]{kind}[/] ({len(entries)})")
        for entry in entries:
            console.print(f"  {entry.name}  [dim]{entry.source}[/]")

    errors = load_errors()
    if errors:
        console.print(f"[yellow]unavailable[/] ({len(errors)} source(s) failed to load)")
        for source, err in errors:
            console.print(f"  {source}  [dim]{err}[/]")


def main() -> None:
    try:
        app()
    except _USER_ERRORS as exc:
        if _STATE["debug"]:
            raise
        console.print(f"[red]error[/] {_user_message(exc)}")
        console.print("[dim]re-run with --debug for the full traceback[/]")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
