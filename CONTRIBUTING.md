# Contributing to Searchdrone

Thank you for helping. Searchdrone is a safety-related project, so the rules below are strict on purpose.

## Setup

```bash
bash tools/setup/wsl-setup.sh   # once, on WSL2 Ubuntu 24.04
bash tools/bootstrap.sh
mkdir -p config/local && touch config/local/private-patterns.txt   # your private search patterns
just check
```

## Rules

- **Run `just check` before every pull request.** It runs lint, type checks, tests, import contracts and `reuse lint`.
- **Never skip the git hooks** (`--no-verify`, `SKIP=`) and never `git add -f` an ignored file. If a hook fails, fix the cause.
- **No personal data.** No home addresses, real home coordinates, network details, keys or passwords in files, commits, issues or pull requests. Example coordinates must stay near the public ArduPilot SITL site (CMAC); the forbidden-file check enforces this. Use your GitHub `noreply` e-mail address for commits.
- **Safety first.** STOP and RETURN HOME always bypass the AI. The AI never changes flight-controller parameters, never disables the geofence, never raises its own autonomy level and never confirms a find. Changes that affect what the drone can do need a simulator (SITL) test.
- **Settings are `key=value`** after the recipe name, never `--flags` (`just test-sitl sc=SC04`).
- **Dependencies:** only licences compatible with Apache-2.0 (MIT, BSD, ISC, Apache, MPL, unmodified LGPL). No GPL or AGPL code is imported. New tool versions must be at least two weeks old.

## Pull requests

- One topic per pull request, on a branch, never directly on `main`.
- Title in [Conventional Commits](https://www.conventionalcommits.org) form, for example `feat(ground): add mission endpoint` or `fix(safety): reject altitude above frame`.
- Describe what changed, why, how you tested it, and the flight-safety impact ("None" is a valid answer).
- Every pull request is reviewed by three AI reviewers (architecture, IT security, flight safety) and the automated checks. The reviewers follow the instructions in [`.github/review/`](.github/review/) and post one comment each, starting with `review/<role>: approve` or `review/<role>: request_changes` and naming the reviewed commit; the check `gate/review` is green only when all three approved the latest commit. Changes under `hardware/` also need the maintainer's approval (`gate/owner`).
- Squash merge only.

## Reporting bugs

Use the issue templates. Security problems go through private vulnerability reporting, see [SECURITY.md](SECURITY.md).

By contributing you agree that your contribution is licensed under Apache-2.0 (software) or CERN-OHL-S-2.0 (`hardware/`), as described in the README.
