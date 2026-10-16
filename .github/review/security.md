# Review instructions: IT security

Role name: `security`. Read `common.md` first; its rules apply here too.

## What you review

- **Secrets.** No keys, tokens, passwords or private certificates in files, tests,
  fixtures, logs or commit metadata. Secrets come from `.env` or `secrets/` (both
  ignored by git) and are never printed or logged.
- **Injection.** SQL (only parameterised queries), shell commands (no `shell=True`
  with outside data, no string-built commands), file paths (no path traversal from
  request data), and prompt injection into the AI commander: operator chat, tool
  results and image text reach the model as data and can never widen what it may do.
- **Access control.** Every HTTP and WebSocket endpoint checks the session and the
  role (`pilot`, `observer`, `commander`, `system`). STOP is allowed for every
  logged-in user; RETURN HOME, manual control and approvals are for `pilot` only. The
  AI process (`commander`) cannot call panel endpoints.
- **Exposure.** The ground server listens on localhost, and on the LAN only when the
  owner starts it that way. No new port, tunnel or `0.0.0.0` listener without an owner
  decision. Compose services publish to `127.0.0.1` only.
- **Secret separation.** The AI process never receives database or ground-server
  secrets (`SD_GROUND__*`, database passwords). No `set -a; source .env` that exports
  everything to every process. The cloud model key goes only to the AI process.
- **Dependencies.** New packages: licence (no GPL or AGPL in production), maintenance,
  pinned versions, typosquatting, at least two weeks old.
- **GitHub Actions.** `pull_request_target` and `issue_comment` workflows never check
  out and run pull request code. No untrusted input (`github.event.*` text, branch
  names, titles) directly inside `run:`; pass it through `env:`. Minimal
  `permissions:`, actions pinned to a full commit SHA, `persist-credentials: false`.
- **The gates' commit statuses.** Only the gate workflows (`review-gate.yml`,
  `owner-gate.yml`, `pr-title.yml`) write `gate/*` statuses, each its own context, and
  only from `pull_request_target` or `issue_comment`. A same-repository pull request
  runs its own `pull_request` workflows, so a new or changed workflow or action that
  requests `statuses: write` or `checks: write`, posts any `gate/*` or `review/*`
  status, or has a job named `gate/*` or `review/*` (which creates a check run of that
  name), could approve itself. The same goes for a workflow that posts `review/*`
  comments, sets the `owner-ok` label or uses a personal access token from secrets:
  both gates trust only the owner's own account. Every workflow sets a top-level
  `permissions:` block (`{}` or `contents: read`) and grants per job only what that
  job needs; a missing block or `write-all` is a finding.
- **Repository hygiene.** Personal data in committed files and in commit metadata
  (author e-mail must be a GitHub `noreply` address), symbolic links, agent files
  (`CLAUDE.md`, `CLAUDE.local.md`, `.claude/`, `.mcp.json`), references to internal
  documents or private repositories, quotes from a specification.
- **Pull request text.** Title and body contain only the ticket ID, a short summary,
  how it was tested and technical spec deviations. No quotes from internal documents,
  no personal data, no real coordinates, no links to private repositories.

## Blocking examples (`blocker` or `major`)

- A hard-coded key, token or password (even a "test" one that looks real).
- A new endpoint without a session and role check.
- `shell=True`, `os.system` or `eval` with data from a request, the AI or a file.
- A workflow that checks out pull request code and runs it with secrets or write
  permissions.
- A new dependency under GPL or AGPL.
- A coordinate that looks like a real address rather than a SITL example.
- A symbolic link, or any agent file, anywhere in the change (always `blocker`).
- `set -a; source .env`, or `SD_GROUND__*` reaching the AI process.
- Signs that a commit skipped the git hooks.
- A workflow or action, other than the gate workflows, that can write commit
  statuses or check runs (`statuses: write`, `checks: write`, a `gate/*` or
  `review/*` context or job name).
- A workflow that posts `review/*` comments, sets `owner-ok`, or uses a personal
  access token secret.
- Pull request text that quotes internal documents or links a private repository.
