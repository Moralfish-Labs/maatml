"""``maatml ship-check``: whether a candidate may replace the accepted release.

Three parts, in order, and skipping the third makes release decisions wrong:

1. **Absolute** — every blocking gate at or above its floor, at production
   tier (a smoke-gated pass is not evidence).
2. **Delta** — no gated metric regresses against the baseline. A move
   smaller than one row at n >= 30 is exempt: one row is not evidence of
   decay.
3. **Controlled replay** — when the two reports were measured on different
   splits, both checkpoints are replayed over identical rows first. A raw
   delta across a changed benchmark reads benchmark hardening as decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .harness import Report, gate_actual

DELTA_MIN_N = 30


@dataclass
class ShipVerdict:
    candidate: str
    baseline: str
    ship: bool = False
    reasons: list[str] = field(default_factory=list)
    absolute: dict[str, Any] = field(default_factory=dict)
    delta: dict[str, Any] = field(default_factory=dict)
    population: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "ship": self.ship,
            "reasons": list(self.reasons),
            "absolute": self.absolute,
            "delta": self.delta,
            "population": self.population,
        }


def _split_of(report: Report) -> Optional[str]:
    gates = report.gates or {}
    return gates.get("benchmark_sha256") or report.extras.get("split_sha256")


def ship_check(
    candidate: Report,
    baseline: Report,
    *,
    gates: dict[str, float],
    tiers: Optional[dict[str, str]] = None,
    max_regression: Optional[float] = None,
    replayed: bool = False,
    gates_benchmark: Optional[str] = None,
    candidate_label: str = "candidate",
    baseline_label: str = "baseline",
) -> ShipVerdict:
    """Absolute, delta and population checks over two reports.

    ``max_regression`` replaces the one-row tolerance with a fixed drop for
    every gated metric. ``replayed`` records a controlled replay, but both
    resulting reports must still identify the same population. When supplied,
    ``gates_benchmark`` binds absolute floors to their measured population.
    """
    tiers = tiers or {}
    verdict = ShipVerdict(candidate=candidate_label, baseline=baseline_label)

    # 1. absolute
    evidence = candidate.gates
    if evidence is None:
        verdict.absolute = {"passed": None, "smoke": None, "failed": []}
        verdict.reasons.append(
            f"{candidate_label} carries no gate evidence; run evaluate --gate first"
        )
    else:
        failed = sorted(
            name
            for name, info in (evidence.get("results") or {}).items()
            if not info.get("passed") and info.get("tier", "blocking") != "advisory"
        )
        smoke = bool(evidence.get("smoke"))
        verdict.absolute = {
            "passed": bool(evidence.get("passed")),
            "smoke": smoke,
            "failed": failed,
            "advisory_failed": list(evidence.get("advisory_failed") or []),
        }
        if smoke:
            verdict.reasons.append(
                f"{candidate_label} was gated at the smoke tier; a rehearsal is not a release pass"
            )
        if failed:
            verdict.reasons.append(f"{candidate_label} below floor: {', '.join(failed)}")
        if evidence.get("passed") is not True:
            verdict.reasons.append(f"{candidate_label} has no successful aggregate gate verdict")

    # 2. delta
    regressions: list[str] = []
    advisory_regressions: list[str] = []
    exempt: list[str] = []
    incomparable: list[str] = []
    per_metric: dict[str, dict[str, Any]] = {}
    for name in gates:
        cand = gate_actual(name, candidate.metrics, candidate.slices)
        base = gate_actual(name, baseline.metrics, baseline.slices)
        if cand is None or base is None:
            incomparable.append(name)
            per_metric[name] = {"candidate": cand, "baseline": base, "comparable": False}
            continue
        drop = base - cand
        count = candidate.counts.get(name)
        n = int(count["n"]) if count else None
        if max_regression is not None:
            tolerance = float(max_regression)
            basis = "max_regression"
        elif n is not None and n >= DELTA_MIN_N:
            tolerance = 1.0 / n
            basis = f"one row at n={n}"
        else:
            tolerance = 0.0
            basis = "no count at n>=30; any drop counts" if n is None else f"n={n} < {DELTA_MIN_N}"
        regressed = drop > tolerance + 1e-12
        per_metric[name] = {
            "candidate": cand,
            "baseline": base,
            "delta": cand - base,
            "tolerance": tolerance,
            "tolerance_basis": basis,
            "regressed": regressed,
            "tier": tiers.get(name, "blocking"),
            "comparable": True,
        }
        if regressed:
            if tiers.get(name) == "advisory":
                advisory_regressions.append(name)
            else:
                regressions.append(name)
        elif drop > 0:
            exempt.append(name)
    verdict.delta = {
        "regressions": regressions,
        "advisory_regressions": advisory_regressions,
        "exempt": exempt,
        "incomparable": incomparable,
        "metrics": per_metric,
    }
    if regressions:
        verdict.reasons.append(
            "regressed against "
            f"{baseline_label}: "
            + ", ".join(
                f"{m} {per_metric[m]['delta']:+.4f} (allowed -{per_metric[m]['tolerance']:.4f})"
                for m in regressions
            )
        )
    if incomparable:
        verdict.reasons.append(
            f"not comparable on {', '.join(incomparable)}: one side never reported it"
        )

    # 3. population
    cand_split = _split_of(candidate)
    base_split = _split_of(baseline)
    same = cand_split is not None and cand_split == base_split
    verdict.population = {
        "candidate_split": cand_split,
        "baseline_split": base_split,
        "same": same,
        "replayed": bool(replayed),
        "gates_benchmark": gates_benchmark,
    }
    if gates_benchmark is not None and cand_split != gates_benchmark:
        verdict.reasons.append(
            "candidate population differs from evaluation.gates_benchmark; "
            "reconcile measured baseline evidence before applying these floors"
        )
    if not same:
        shown = f"{str(cand_split)[:16]} vs {str(base_split)[:16]}"
        verdict.reasons.append(
            f"reports were measured on different splits ({shown}); a delta across a changed "
            "benchmark reads hardening as decay. Replay both over identical rows (--replay)"
        )

    verdict.ship = not verdict.reasons
    return verdict


def render_verdict(verdict: ShipVerdict) -> list[str]:
    lines = [
        f"ship-check {verdict.candidate} vs {verdict.baseline}: "
        + ("SHIP" if verdict.ship else "DO NOT SHIP")
    ]
    absolute = verdict.absolute
    if absolute.get("passed") is None:
        lines.append("absolute: no gate evidence")
    else:
        status = "passed" if absolute["passed"] else "failed"
        tier = " (smoke tier)" if absolute.get("smoke") else ""
        lines.append(f"absolute: {status}{tier}")
        if absolute.get("advisory_failed"):
            lines.append("  advisory below floor: " + ", ".join(absolute["advisory_failed"]))
    for name, info in verdict.delta.get("metrics", {}).items():
        if not info.get("comparable"):
            lines.append(f"delta: {name}: not comparable")
            continue
        mark = "REGRESSED" if info["regressed"] else ("exempt" if info["delta"] < 0 else "ok")
        lines.append(
            f"delta: {name}: {info['baseline']:.4f} -> {info['candidate']:.4f} "
            f"({info['delta']:+.4f}, tolerance {info['tolerance']:.4f}, {info['tolerance_basis']}) "
            f"{mark}"
        )
    population = verdict.population
    if population.get("replayed"):
        lines.append("population: controlled replay over identical rows")
    elif population.get("same"):
        lines.append(f"population: same split {str(population['candidate_split'])[:16]}")
    else:
        lines.append(
            "population: DIFFERENT splits "
            f"{str(population.get('candidate_split'))[:16]} vs "
            f"{str(population.get('baseline_split'))[:16]}"
        )
    for reason in verdict.reasons:
        lines.append(f"reason: {reason}")
    return lines
