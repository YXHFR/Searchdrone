"""Which GitHub Actions workflows may write the gate statuses.

The gate statuses (``gate/review``, ``gate/owner``, ``gate/pr-title``) are what branch
protection and ``tools/check_merged_pr_gates.py`` trust, and every workflow writes as
``github-actions[bot]``. So only the three gate workflows may write statuses or check
runs or name a job like a gate, and they may only run from the base branch
(``pull_request_target``, ``issue_comment``), never from a pull request's own code.
Nor may any other workflow approve on the owner's behalf (``.github/review/security.md``):
no ``pull-requests: write`` or ``issues: write`` (labels and comments are the gates'
evidence) and no secret other than ``ALLOWED_SECRETS``, whatever it is passed as, nor
``secrets: inherit`` (a personal access token acts as its owner). Adding a secret to
that list needs the owner, since this file is protected. A workflow that only lives on a
branch is never merged and never read here: keep owner tokens out of repository secrets.
``tools/tests/test_workflows.py`` holds main to these rules; ``check_merged_pr_gates.py``
applies them to the workflow files each merged pull request changed (A18).
"""

from __future__ import annotations

import re
from typing import Any

GATE_WORKFLOWS = frozenset({"review-gate.yml", "owner-gate.yml", "pr-title.yml"})
GATE_NAME_PREFIXES = ("gate/", "review/")
# Both run the workflow file from the base branch, so a pull request cannot change it.
GATE_TRIGGERS = frozenset({"pull_request_target", "issue_comment"})
# Scopes that can comment, label or review: what the gates read as the owner's word.
APPROVAL_SCOPES = ("pull-requests", "issues")
# Secrets a workflow outside the gates may read: none that can act on GitHub.
ALLOWED_SECRETS = frozenset({"GITHUB_TOKEN", "OPENROUTER_API_KEY"})
SECRET_RE = re.compile(r"\bsecrets\b", re.IGNORECASE)
SECRET_NAME_RE = re.compile(
    r"\bsecrets\s*(?:\.\s*(?P<dot>[A-Za-z_][\w-]*)|\[\s*['\"](?P<key>[^'\"]+)['\"]\s*\])",
    re.IGNORECASE,
)


def triggers(doc: dict[Any, Any]) -> dict[str, Any]:
    """The ``on:`` block as a dict (PyYAML reads the bare key ``on`` as ``True``)."""
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {str(t): None for t in on}
    return on if isinstance(on, dict) else {}


def _gate_like(name: object) -> bool:
    """A job name that is, or could evaluate to, ``gate/...`` or ``review/...``."""
    text = str(name).lower()
    if text.startswith(GATE_NAME_PREFIXES):
        return True
    # An expression decides the rest: flag it unless the fixed start rules a gate out.
    static = text.split("${{", 1)[0]
    return "${{" in text and any(p.startswith(static) for p in GATE_NAME_PREFIXES)


def status_writers(doc: dict[Any, Any]) -> list[str]:
    """Where a workflow could write a gate: ``statuses: write``, ``checks: write`` or
    ``write-all`` (``top`` or ``job <id>``), a job with no ``permissions`` at either level,
    which gets the repository's default token (``default <id>``), or a job named like a
    gate (``name <id>``)."""

    def grants(perms: Any) -> bool:
        if perms == "write-all":
            return True
        return isinstance(perms, dict) and "write" in (perms.get("statuses"), perms.get("checks"))

    out = ["top"] if grants(doc.get("permissions")) else []
    for name, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        if grants(job.get("permissions")):
            out.append(f"job {name}")
        elif "permissions" not in doc and "permissions" not in job:
            out.append(f"default {name}")
        if any(_gate_like(n) for n in (name, job.get("name", ""))):
            out.append(f"name {name}")
    return out


def gate_trigger_problems(doc: dict[Any, Any]) -> list[str]:
    """Triggers of a gate workflow other than ``pull_request_target``/``issue_comment``."""
    on = triggers(doc)
    if not on:
        return ["inga triggers"]
    return [f"trigger {t}" for t in sorted(map(str, on)) if t not in GATE_TRIGGERS]


def _strings(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return [node] if isinstance(node, str) else []


def _expressions(text: str) -> list[str]:
    """The ``${{ ... }}`` bodies in ``text``. A ``}}`` inside a quoted string does not end
    one (GitHub's strings are single-quoted, ``''`` escapes a quote); an expression left
    open runs to the end of the text."""
    out = []
    start = text.find("${{")
    while start >= 0:
        i, quoted = start + 3, False
        while i < len(text) and (quoted or not text.startswith("}}", i)):
            if text[i] == "'":
                quoted = not quoted
            i += 1
        out.append(text[start + 3 : i])
        start = text.find("${{", i + 2)
    return out


def _other_secrets(node: Any) -> bool:
    """``node`` names a secret outside ``ALLOWED_SECRETS``, reads ``secrets`` some other
    way (``toJSON(secrets)``), or passes ``secrets: inherit``."""
    if isinstance(node, dict) and str(node.get("secrets", "")).strip() == "inherit":
        return True
    # Secrets exist only inside ${{ }}: "No secrets in ..." and a job named secrets do not.
    for text in (e for s in _strings(node) for e in _expressions(s)):
        if not SECRET_RE.search(text):
            continue
        names = [m["dot"] or m["key"] for m in SECRET_NAME_RE.finditer(text)]
        if len(names) != len(SECRET_RE.findall(text)):
            return True
        if any(n.upper() not in ALLOWED_SECRETS for n in names):
            return True
    return any(_other_secrets(v) for v in (node.values() if isinstance(node, dict) else []))


def approval_writers(doc: dict[Any, Any]) -> list[str]:
    """Where a workflow could speak for the owner: ``pull-requests: write`` or
    ``issues: write`` (``write-all`` is already a status writer) at ``top`` or
    ``job <id>``, or a secret outside ``ALLOWED_SECRETS`` or ``secrets: inherit``
    (``secret top``, ``secret <id>``)."""

    def grants(perms: Any) -> bool:
        return isinstance(perms, dict) and any(perms.get(s) == "write" for s in APPROVAL_SCOPES)

    out = ["top"] if grants(doc.get("permissions")) else []
    if _other_secrets({k: v for k, v in doc.items() if k != "jobs"}):
        out.append("secret top")
    for name, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        if grants(job.get("permissions")):
            out.append(f"job {name}")
        if _other_secrets(job):
            out.append(f"secret {name}")
    return out
