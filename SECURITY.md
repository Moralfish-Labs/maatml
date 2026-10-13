# Security Policy

## Supported Versions

Security fixes are applied to the latest **0.x** release of maatml.

| Version | Supported |
| ------- | --------- |
| Latest 0.x release | Yes |
| Earlier releases | No backport commitment |

## Trust model

maatml treats a model folder as trusted, executable code. `plugins:` entries in
`model.yml` are imported when the folder is loaded, so every subcommand that
reads `model.yml`, including `maatml validate`, `maatml plan`, `maatml prepare`,
`maatml train`, `maatml evaluate`, `maatml export`, and `maatml serve`, can
execute code shipped in that folder. Do not run maatml against model folders
from untrusted sources. There is no sandbox; a model folder has the same power
as any local Python package you choose to import. `maatml validate --no-plugins`
checks the schema and paths without importing plugin code.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security reports.

Prefer one of:

1. **GitHub private vulnerability reporting** (preferred): use the
   [Security Advisories](https://github.com/Moralfish-Labs/maatml/security/advisories/new) flow on this repository
   when available.
2. **Email**: nedal.elghamry@gmail.com

Include enough detail to reproduce the issue (affected version or commit,
environment, and steps). You can expect an acknowledgement when the report is
received; remediation timelines depend on severity and scope.
