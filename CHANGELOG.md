# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org). Hardware revisions have their own tags (`hw-revA`, ...).

## [Unreleased]

### Added

- Monorepo skeleton: uv workspace for the Python packages, pnpm workspace for TypeScript,
  `justfile` with every recipe, pre-commit hooks, import contracts and example configuration.
- Forbidden-file guard that blocks planning files, agent files, secrets, personal config,
  media, large files, coordinates outside public test sites and private search patterns.
- `tools/bootstrap.sh` and `tools/setup/wsl-setup.sh` for a fresh WSL2 Ubuntu 24.04.
- Project documents: package READMEs, security policy, contributing guide, code of conduct,
  issue templates and this changelog.
- Shared message schemas (`sd_schemas`): ids, limits and their outer cap, mission charter,
  gate input and output, mission states, SD-link messages (including manual stick control),
  panel and commander API models, reason codes with Swedish texts, and the decision
  interface. Generated JSON Schema and TypeScript types (`@searchdrone/schemas`) via
  `just gen`, checked by `just gen-check`.
- Skill registry with all 18 skills, every one a stub until its executor is built.
- Local Postgres 18 + PostGIS 3.6 in `compose.yaml` (loopback port 5433, image pinned by
  digest), `just db-up` / `db-down` / `db-check`, and the first Alembic migration with the
  ground station tables, append-only guards and the restricted `sd_app` role.
