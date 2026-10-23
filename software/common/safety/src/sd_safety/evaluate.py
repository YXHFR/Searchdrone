"""The safety gate: ``evaluate(req, ctx) -> GateDecision``.

A pure function: no I/O, no clock, no randomness. The same input always gives the same
answer. Every rule runs and every finding is collected; any rejection code rejects, else
any approval code asks for a click, else the order is allowed.
"""

from __future__ import annotations

from typing import Any

from sd_safety.policy import FORBIDDEN_SKILL_NAMES
from sd_safety.priority import message_sv, sort_codes
from sd_safety.rules import CLEAR_AROUND_DRONE, RULES, Facts, Finding
from sd_safety.targets import Plane
from sd_schemas.base import LatLonAlt
from sd_schemas.gate import GateContext, GateDecision, GateRequest
from sd_schemas.reason_codes import approval_codes
from sd_schemas.skills import GateTargets, SkillParams, SkillSpec, TargetContext

__all__ = ["evaluate"]

_APPROVAL_CODES = approval_codes()


def evaluate(req: GateRequest, ctx: GateContext) -> GateDecision:
    """Decide ALLOW, REJECT or PENDING_APPROVAL for one order."""
    facts = _facts(req, ctx)
    codes: list[str] = []
    rule_ids: list[str] = []
    details: dict[str, Any] = {}
    for rule in RULES:
        if req.issuer_kind not in rule.issuers or (rule.ground_only and ctx.side == "onboard"):
            continue
        try:
            findings = rule.check(facts)
        except Exception:  # a bug in a rule must reject, never crash the gate
            findings = [Finding("SCHEMA_INVALID", {"internal_error": rule.rule_id})]
        for finding in findings:
            codes.append(finding.code)
            if rule.rule_id not in rule_ids:
                rule_ids.append(rule.rule_id)
            details.update(finding.details)
    ordered = sort_codes(codes)
    if any(code not in _APPROVAL_CODES for code in ordered):
        decision = "REJECT"
    elif ordered:
        decision = "PENDING_APPROVAL"
    else:
        decision = "ALLOW"
    confirmations = (
        [CLEAR_AROUND_DRONE]
        if decision == "PENDING_APPROVAL" and "NEEDS_CLEARANCE" in ordered
        else []
    )
    return GateDecision(
        decision=decision,
        reason_codes=ordered,
        rule_ids=rule_ids,
        message_sv=message_sv(ordered, details, req.skill),
        details=details,
        required_confirmations=confirmations,
        side=ctx.side,
    )


def _facts(req: GateRequest, ctx: GateContext) -> Facts:
    spec = _spec(req.skill, ctx)
    params = _params(spec, req.params) if spec is not None else None
    targets = _targets(params, ctx) if params is not None else None
    charter = ctx.charter
    plane = Plane(charter.home.lat, charter.home.lon) if charter is not None else None
    return Facts(req=req, ctx=ctx, spec=spec, params=params, targets=targets, plane=plane)


def _spec(skill: str, ctx: GateContext) -> SkillSpec | None:
    """The registry entry, unless the name is forbidden (the registry can be wrong)."""
    if skill in FORBIDDEN_SKILL_NAMES:
        return None
    spec = ctx.registry.get(skill)
    return spec if spec is not None and spec.name == skill else None


def _params(spec: SkillSpec, raw: dict[str, Any]) -> SkillParams | None:
    try:
        return spec.params_model.model_validate(raw)
    except Exception:  # a validator that raises anything at all means invalid parameters
        return None


def _targets(params: SkillParams, ctx: GateContext) -> GateTargets | None:
    """Where the order goes; None if a parameter model fails to say (fail closed)."""
    v = ctx.vehicle
    position: LatLonAlt | None = None
    try:
        if v is not None:
            position = LatLonAlt(lat=v.lat, lon=v.lon, alt_amsl_m=v.alt_amsl_m)
        home = ctx.charter.home if ctx.charter is not None else (v.home if v is not None else None)
        return params.gate_targets(
            TargetContext(
                charter=ctx.charter, position=position, home=home, keepouts=list(ctx.keepouts)
            )
        )
    except Exception:  # any failure here must reject, never crash the gate
        return None
