# Contributing

Use a source checkout and an editable installation for development:

```bash
git clone https://github.com/Moralfish-Labs/maatml.git
cd maatml
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Add `[ml]` only when the work needs the training stack. From the repository root:

```bash
ruff check src tests scripts examples
ruff format --check src tests scripts examples
mypy src/maatml --ignore-missing-imports
python -m pytest tests/ examples/ -q
mkdocs build --strict
```

Preview documentation with `mkdocs serve`. The site includes the example
READMEs directly, so edit those files for example content. Add each new page to
`mkdocs.yml`; missing pages, local links, anchors and snippet files fail the
strict build. Generated `site/` output is not source documentation.

See the repository [contributor guide](https://github.com/Moralfish-Labs/maatml/blob/main/CONTRIBUTING.md)
for DCO sign-off, release and versioning policy, the
[security policy](https://github.com/Moralfish-Labs/maatml/blob/main/SECURITY.md)
for private reporting, and the
[code of conduct](https://github.com/Moralfish-Labs/maatml/blob/main/CODE_OF_CONDUCT.md).
