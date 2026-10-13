# Evidence: floors, populations, portable runs

A finished run and a passed gate are not yet a claim. Between them sit the
questions a reviewer asks: where did the floor come from, on which rows, at
what threshold, against which baseline, on which machine, and can the same
answer be produced again from the records alone. maatml answers each of them
as a lifecycle step with a record, so a model folder does not have to carry
its own scripts for any of it.

Every number below is read from or written to three places: the eval report
(`output/eval/<run>.json`, versioned), the run record (`output/runs.jsonl`)
and `model.yml`. Nothing is inferred at read time.

## Floors are derived, never typed

```bash
maatml gates derive <model-dir> --run RUN            # print the floors a run's report supports
maatml gates derive <model-dir> --run RUN --write    # rewrite evaluation.gates in model.yml
maatml gates derive <model-dir> --seed-study FILE    # the per-metric minimum across seeds
```

A floor is the **Wilson 95 % lower bound of the observed rate at that
metric's own denominator**, floored to two places, written with its
derivation beside it:

```yaml
evaluation:
  gates:
    routing_refusal_recall_rate: 0.84   # 66/71 = 0.930, w95 0.846
    all_layers_pass_rate: 0.94          # 401/414 = 0.969, w95 0.947
  gates_benchmark: 3f9c…                # the split these floors were derived on
```

The denominators come from the report's `counts` (`{metric: {k, n}}`), which
the harness records for every rate it computes and a metrics plugin adds for
its own through `__counts__`. A metric below `--min-n` rows is refused rather
than floored on thin evidence. When the run carries a prediction cache
(`evaluate --cache`), supported per-row rates use a cluster bootstrap over the
group key (`--cluster-by`, default `family`). Plugin rates need a per-row
verdict to use that path. Without counts, an existing metric gate falls back
to its observed value, labelled `observed`; that is not a confidence bound.

`--write` stamps `evaluation.gates_benchmark` with the split's content hash.
A later `evaluate --gate` on a different split warns (`population_mismatch`
in the gates payload), and `--strict-population` refuses: floors describe the
population they came from.

Gates carry a **tier**: `{min: 0.80, tier: advisory}` is reported and listed
under `advisory_failed` but never fails the step. A `"slice:<field>=<value>"`
gate floors one slice of the split (declared under `evaluation.slices`),
which is how a pooled rate is kept from hiding the family it averages away.

## Whether a run ships

```bash
maatml ship-check <model-dir> CANDIDATE BASELINE
maatml ship-check <model-dir> CANDIDATE BASELINE --replay   # benchmark changed since BASELINE
```

One verdict in three parts, in order: **absolute** (every blocking floor met
at production tier), **delta** (no gated metric drops more than one row at
n ≥ 30, or `--max-regression`), **population** (both reports on the same
split; else `--replay` re-evaluates both checkpoints over the current test
split under `output/eval/replay/`, touching neither run's own evidence). Exit
1 on DO NOT SHIP; `--json` for machines.

## The operating point comes from val

```bash
maatml evaluate <model-dir> --checkpoint RUN --split val --cache --set evaluation.score_thresh=0.05
maatml operating-point derive <model-dir> --run RUN --split val
maatml operating-point derive <model-dir> --run RUN --write --confirm-on-test
```

```yaml
evaluation:
  operating_point:
    threshold_key: score_thresh    # the evaluation key the sweep writes
    objective: recall
    budget: {metric: false_alarms_per_hour, max: 2.0}
```

The sweep re-scores a val prediction cache through the predictor's
`rescore(rows, threshold)`, with no inference re-run, skips cuts below the one
the cache was decoded at, and picks the best objective under the budget. The
val evaluate that writes the cache runs under `--set`, which overrides
`model.yml` for that one evaluate, is recorded on the report as
`extras.overrides`, and is refused together with `--gate` or `--blind`.
`--write` sets the threshold in `model.yml` with the sweep as provenance.
`--confirm-on-test` evaluates once on test at that cut and records a **test
spend** on the run; a second spend on the same benchmark version warns and
lists both. Deriving on the test split is refused.

## Populations are named, not implied

```yaml
dataset:
  isolation:
    fields: [clip, camera, site]                  # row fields, fine -> coarse
    policy: {val: camera, benchmark: camera, blind: site}
  pins:
    val: ["camera:G339"]
    benchmark: ["camera:G341", "camera:G421"]
  blind_samples: datasets/samples/blind_v001.jsonl
```

`group_by` keeps correlated rows on one side of a split; it cannot say which
side a camera lands on or that a site is absent from training. `isolation`
declares the hierarchy and the level each held-out population is disjoint at;
`pins` move whole groups after the hash split, and a pinned population is
exactly its pins. `prepare` refuses to write splits that violate the policy,
and `audit` re-checks the ones on disk.

Every prepare records a **benchmark version** (`output/prepared/benchmark.json`,
an order-insensitive hash of the test rows plus the pins) that reports carry
as `extras.benchmark_version`. Editing `benchmark_samples` in place is
refused; a new file is a new version, so floors keep naming the population
they were derived on.

`maatml evaluate --blind` spends the blind manifest **once per frozen
candidate**: the run must hold a production gate pass whose fingerprint
(evaluation config + weights) is unchanged, the gates are enforced on the
blind rows, and the spend is recorded on the run. A repeat needs `--force`
and is recorded as forced.

`maatml train --seeds N` trains the same recipe at N seeds and records the
spread (`output/seeds/<first-run>-xN.json`), which `gates derive
--seed-study` floors on.

## Sources carry their licence

```yaml
dataset:
  attribution: datasets/ATTRIBUTION.md
```

A sidecar Markdown table, one row per `source` (`licence`, `commercial-use`,
`sign-off` required; consent and attribution read when present), keyed on
another row field when `dataset.attribution_field` names one. `prepare`
refuses a row without a source or without a table row, a blocked or unsigned
sign-off, and a `no` / `unknown` commercial-use unless the sign-off itself
reads `accepted-risk — <name> <date>`. Acceptance is signed in the table, not
passed as a flag, so it carries a name and a date.

Every prepare writes `output/prepared/corpus.lock.json` (input files by
sha256 and row count, the attribution rows used, the risk accepted), and
`export` copies it into `manifest.json` as `corpus_lock`, with the rows
rendered as the bundle's `ATTRIBUTION.md`. Declared metadata only; nothing is
looked up.

## Training tells the truth earlier

```yaml
training:
  select_by: all_layers_pass_rate   # choose the checkpoint on val, by the gated metric
  keep_checkpoints: 4
```

After training, every saved `checkpoint-<step>` and the final weights are
evaluated on **val** with the evaluate harness; the best is recorded on the
run and the run id resolves to it from then on, with evidence still recorded
against the run. Reports under `output/eval/select/<run>/`.

Every evaluate reports **pathologies** (`never_fires`, `identical_output`,
`one_class`, plus a plugin's `__pathologies__`), and at the smoke tier each
one is a failing gate, so a rehearsal cannot pass on a model that does not
work. `distill` refuses a prompt pool that overlaps a held-out population
before any teacher call.

## Runs travel

```bash
maatml runs <model-dir> --pack RUN              # output/bundles/<run>.maatml-run.tar.gz
maatml runs <model-dir> --adopt BUNDLE          # on the receiving machine
maatml report <model-dir>                       # output/REPORT.md (--format csv)
```

A bundle holds the run directory (without `checkpoint-*` unless
`--with-checkpoints`), the run's eval reports and caches, the registry record,
the environment and the `model.yml` fingerprint, every file hashed. `--adopt`
verifies each file, refuses another identity or recipe and an existing run
without `--force`, and appends the record with its paths rewritten. Records
move; jobs do not.

Every train record and every evaluate carries an **environment manifest**:
git SHA, Python, OS, package versions, CUDA / cuDNN, GPUs and driver,
determinism settings.

`maatml report` renders runs, every report's metrics with k / n and the
Wilson bound beside the floor that judged them, slice and pathology gates,
seed statistics and spends, from `runs.jsonl`, `output/eval/*.json` and
`output/seeds/*.json` alone, so it regenerates byte-identically and never
reads `model.yml`.

The configuration surface for all of this is in the skill reference,
[model.yml reference](skill/references/model-yml.md).
