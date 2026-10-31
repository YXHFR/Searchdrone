"""Which GitHub Actions workflows may write the gate statuses.

The gate statuses (``gate/review``, ``gate/owner``, ``gate/pr-title``) are what branch
protection and ``tools/check_merged_pr_gates.py`` trust, and every workflow writes as
``github-actions[bot]``. So only the three gate workflows may write statuses or check
runs or name a job like a gate, and they may only run from the base branch
(``pull_request_target``, ``issue_comment``), never from a pull request's own code.
``tools/tests/test_workflows.py`` holds main to these rules; ``check_merged_pr_gates.py``
applies them to the workflow files each merged pull request changed (A18).
"""

from __future__ import annotations

from typing import Any

GATE_WORKFLOWS = frozenset({"review-gate.yml", "owner-gate.yml", "pr-title.yml"})
GATE_NAME_PREFIXES = ("gate/", "review/")
# Both run the workflow file from the base branch, so a pull request cannot change it.
GATE_TRIGGERS = frozenset({"pull_request_target", "issue_comment"})


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
    ``write-all`` (``top`` or ``job <id>``), or a job named like a gate (``name <id>``)."""

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
        if any(_gate_like(n) for n in (name, job.get("name", ""))):
            out.append(f"name {name}")
    return out


def gate_trigger_problems(doc: dict[Any, Any]) -> list[str]:
    """Triggers of a gate workflow other than ``pull_request_target``/``issue_comment``."""
    on = triggers(doc)
    if not on:
        return ["inga triggers"]
    return [f"trigger {t}" for t in sorted(map(str, on)) if t not in GATE_TRIGGERS]
