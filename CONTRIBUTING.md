# Contributing to maatml

Thanks for contributing. This guide is for human contributors. AI coding agents
should also read [AGENTS.md](AGENTS.md).

## Code of conduct

Please read and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Do not open public issues for security reports. See [SECURITY.md](SECURITY.md).

## Using MaatML (as a package)

End users should install from PyPI, not from a source checkout:

```bash
pip install maatml
pip install "maatml[ml]"          # training stack
```

Docs and site: [maatml.pages.dev](https://maatml.pages.dev) ·
[PyPI](https://pypi.org/project/maatml/).

## Development setup

Clone only when changing the framework itself:

```bash
git clone https://github.com/moralfish/maatml.git
cd maatml
python -m venv .venv
source .venv/bin/activate

# CPU-free contributions (tests run without torch)
pip install -e ".[dev]"

# Training / evaluation work
pip install -e ".[dev,ml]"
```

`pyproject.toml` is the only dependency manifest: extras (`dev`, `ml`, `cuda`,
`pref`, `teacher`, `docs`, `vision`, `vllm`) replace the `requirements*.txt`
files that used to sit alongside it and drift out of sync.

## Lint and tests

```bash
ruff check src tests scripts examples
ruff format --check src tests scripts examples
mypy src/maatml --ignore-missing-imports
pytest tests/ examples/ -q
```

## Where to start

New here? These are good first contributions: small, self-contained, and
genuinely useful:

- **Extend real trainer coverage.** CI already runs a tiny causal SFT CPU
  lifecycle through export and verification, then checks an idempotent rerun.
  Add comparable coverage for other architectures and recovery paths; one
  trainer's smoke pass does not establish every backend.
- **Add a broadly relatable example model.** Scaffold a new `examples/<task>/`
  for an everyday task (log triage, PII redaction, a small captioner). See the
  [plugin author guide](docs/plugins.md) and the
  [validator-gated lifecycle](docs/lifecycle.md).
- **Tighten a validator or its gates.** Validators are where MaatML earns its
  keep. Clearer contracts and error messages help every model.
- **Docs.** Improve a walkthrough, a plugin hook, or an example README (the VLM
  path especially).

Browse [issues labelled `good first issue`](https://github.com/moralfish/maatml/labels/good%20first%20issue),
or open a [feature](https://github.com/moralfish/maatml/issues/new?template=feature_request.md)
or [plugin/task](https://github.com/moralfish/maatml/issues/new?template=plugin_request.md)
issue to discuss anything larger first.

## Pull requests

- Keep PRs focused: one concern per PR when practical.
- Add or update tests for behavior changes.
- Sign off every commit with the [Developer Certificate of Origin](https://developercertificate.org/)
  (`git commit -s` adds a `Signed-off-by:` trailer).

## Releases / PyPI

Publishing uses **Trusted Publishing** (OIDC) via
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). Creating a
GitHub Release (`vX.Y.Z`) builds the wheel/sdist and uploads to PyPI, with no API
token in CI secrets.

One-time PyPI setup (maintainer): project → Publishing → add a trusted
publisher with:

| Field | Value |
| ----- | ----- |
| Owner | `moralfish` |
| Repository name | `maatml` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

## Versioning

### Python package

The package stays **0.x** while the plugin API is still churning. Breaking
changes to public surfaces are expected before 1.0.

### Per-model versions (`examples/<name>/model.yml`)

| Bump | When |
| ---- | ---- |
| **major** | Breaking change to the model's output schema or public contract |
| **minor** | Retrain, data change, or training/config change that affects quality |
| **patch** | Metadata-only (docs, packaging notes, non-behavioral edits) |

## Models

Each task model lives under `examples/` as a standalone folder with its own
`model.yml`. Any folder with a valid `model.yml` can be prepared, trained, and
evaluated via the `maatml` CLI. Scaffold new models with
`maatml scaffold <dir> --architecture causal_sft`.

## AI agent guidance

Operational architecture notes for coding agents live in [AGENTS.md](AGENTS.md).
