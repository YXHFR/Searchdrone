"""``FakeGround``: the ground station's commander API in memory, for evaluations and tests.

It answers orders and charter proposals according to scripted ``ground_responses``
(first match), keeps the chat, memory and decision log, and updates the situation after
each event the way the real server would (see ``apply_event``). A chat line never
changes a stop latch here, whatever it says.
"""

from __future__ import annotations

import copy
import datetime as dt
from typing import Any

from sd_brain.charter_draft import CharterError, CharterProposalOutcome
from sd_schemas.api import ChatMessage
from sd_schemas.base import new_id
from sd_schemas.charter import CharterProposal
from sd_schemas.commander import (
    BrainMessage,
    DecisionLog,
    EstimateResult,
    MissionMemory,
    MissionMemoryUpdate,
    Situation,
    SkillRequest,
    SkillSubmitResult,
)
from sd_schemas.decide import ChatIntentReport
from sd_schemas.gate import GateDecision
from sd_schemas.skills import SKILL_REGISTRY, SkillRegistry, SkillSpecPublic

__all__ = ["FakeGround"]

_LATCH_CODES = {
    "stop": "RESUME_NEEDS_OPERATOR",
    "return": "OPERATOR_OVERRIDE_ACTIVE",
    "battery": "OPERATOR_OVERRIDE_ACTIVE",
    "failsafe": "FAILSAFE_ACTIVE",
    "held": "CONTROL_HELD",
    "manual": "CONTROL_HELD",
}


def _iso(t: dt.datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


class FakeGround:
    def __init__(
        self,
        situation: dict[str, Any],
        *,
        next_charter: dict[str, Any] | None = None,
        reg: SkillRegistry = SKILL_REGISTRY,
        memory: dict[str, Any] | None = None,
        now: Any = None,
        chat_intents: bool = True,
    ) -> None:
        self.sit = copy.deepcopy(situation)
        Situation.model_validate(self.sit)
        self.next_charter = next_charter
        self.reg = reg
        self.mission_id = str(self.sit["mission_id"])
        self._now = now or (lambda: dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
        self.chat_rows: list[ChatMessage] = []
        self.seq = 0
        self.posted: list[BrainMessage] = []
        self.commands: list[SkillRequest] = []
        self.results: list[SkillSubmitResult] = []
        self.proposals: list[CharterProposal] = []
        self.decisions: list[DecisionLog] = []
        self.intents: list[ChatIntentReport] = []
        self.chat_intents = chat_intents
        self.responses: list[dict[str, Any]] = []
        self.step_responses: list[dict[str, Any]] = []
        mem = {
            "mission_id": self.mission_id,
            "summary_sv": "",
            "plan_note": "",
            "summarized_up_to_seq": 0,
            "pending_operator_request": None,
            "updated_at": None,
            **(memory or {}),
        }
        self.mem = MissionMemory.model_validate(mem)

    # --- helpers for the runner -------------------------------------------------------------

    def situation_model(self) -> Situation:
        return Situation.model_validate(self.sit)

    def operator_message(
        self,
        text: str,
        *,
        delivery: str = "delivered",
        handled_by: str | None = None,
        parsed_intent: str | None = None,
        author_role: str = "pilot",
        created_at: dt.datetime | None = None,
    ) -> dict[str, Any]:
        """Add an operator chat line and return the event the server would send."""
        self.seq += 1
        msg = ChatMessage.model_validate(
            {
                "id": str(new_id()),
                "seq": self.seq,
                "mission_id": self.mission_id,
                "role": "operator",
                "author_ref": "operator:demo",
                "kind": "text",
                "text": text,
                "in_reply_to": None,
                "refs": {},
                "author_role": author_role,
                "parsed_intent": parsed_intent,
                "handled_by": handled_by,
                "delivery": delivery,
                "created_at": _iso(created_at or self._now()),
            }
        )
        self.chat_rows.append(msg)
        return {
            "type": "event.operator_message",
            "mission_id": self.mission_id,
            "seq": self.seq,
            "body": {
                "message_id": str(msg.id),
                "seq": msg.seq,
                "text": text,
                "author_role": author_role,
                "parsed_intent": parsed_intent,
                "handled_by": handled_by,
                "delivery": delivery,
                "created_at": _iso(msg.created_at),
            },
        }

    def apply_event(self, evt: dict[str, Any]) -> None:
        """Update the situation after an event, as the server would."""
        s = self.sit
        kind = evt.get("type")
        body = evt.get("body") or {}
        v = s["vehicle"]
        if kind == "event.takeoff_clearance" and body.get("status") in ("active", "confirmed"):
            s["takeoff_clearance_valid_until"] = body.get("valid_until")
            s["brain_may_takeoff"] = bool(s["charter_installed"] and s["fence_verified"])
        elif kind == "event.charter" and body.get("status") == "approved" and self.next_charter:
            s["charter"] = copy.deepcopy(self.next_charter)
            s["autonomy_mode"] = self.next_charter["autonomy_mode"]
            s["pending_proposal"] = None
            s["charter_installed"] = s["fence_verified"] = True
            if v["landed"]:
                s["mission_status"] = "READY"
            s["sectors"] = [
                {"sector_id": x["sector_id"], "name": x["name"], "status": "pending", "progress": 0}
                for x in self.next_charter["sectors"]
            ]
        elif kind == "event.skill":
            skill, status = body.get("skill"), body.get("status")
            if status == "done":
                if skill == "takeoff":
                    v.update(airborne=True, landed=False, armed=True, executor_state="HOLD")
                    alt = (self._last_params("takeoff") or {}).get("alt_m", 40)
                    v["alt_rel_home_m"] = alt
                    s.update(
                        takeoff_clearance_valid_until=None,
                        brain_may_takeoff=False,
                        active_command=None,
                        mission_status="ACTIVE",
                    )
                else:
                    s["active_command"] = None
                    v["executor_state"] = "HOLD"
                    if skill == "survey_sector":
                        sector = (self._last_params("survey_sector") or {}).get("sector_id")
                        for x in s["sectors"]:
                            if x.get("sector_id") == sector:
                                x.update(status="done", progress=1)
        elif kind == "event.operator_override":
            if body.get("active", True):
                action = body.get("action")
                latch = {
                    "vehicle_id": s["vehicle_id"],
                    "action": "return_home" if action == "return_home" else "stop",
                    "underlying": None,
                    "action_seq": self.seq + 1,
                    "set_by": "operator:demo",
                    "set_at": _iso(self._now()),
                    "set_mono_ms": 1,
                }
                s["override"] = latch
                s["active_command"] = None
                if action == "return_home":
                    s.update(mission_status="RETURNING", returning_cause="operator", paused_by=None)
                else:
                    s.update(mission_status="PAUSED", paused_by="operator", returning_cause=None)
                    v["executor_state"] = "HELD"
            else:
                s.update(override=None, mission_status="ACTIVE", paused_by=None)
                v["executor_state"] = "HOLD"
        elif kind == "event.mission":
            for key, field in (("status", "mission_status"), ("paused_by", "paused_by")):
                if key in body:
                    s[field] = body[key]
            if "returning_cause" in body:
                s["returning_cause"] = body["returning_cause"]
        s["state_seq"] += 1
        if s.get("snapshot_state_seq") is not None:
            s["snapshot_state_seq"] += 1
        Situation.model_validate(s)

    def patch(self, patch: dict[str, Any]) -> None:
        def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
            for k, val in src.items():
                if isinstance(val, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], val)
                else:
                    dst[k] = val

        merge(self.sit, patch)
        Situation.model_validate(self.sit)

    def _last_params(self, skill: str) -> dict[str, Any] | None:
        for c in reversed(self.commands):
            if c.skill == skill:
                return c.params
        return None

    # --- GroundApi ------------------------------------------------------------------------

    async def skills(self) -> list[SkillSpecPublic]:
        return self.reg.public()

    async def active_mission(self, vehicle_id: str) -> str | None:
        return self.mission_id

    async def situation(self, mission_id: str) -> Situation:
        return self.situation_model()

    async def chat(self, mission_id: str, after_seq: int, limit: int) -> list[ChatMessage]:
        return [m for m in self.chat_rows if m.seq > after_seq][-limit:]

    async def post_message(self, mission_id: str, msg: BrainMessage) -> str:
        self.posted.append(msg)
        self.seq += 1
        self.chat_rows.append(
            ChatMessage.model_validate(
                {
                    "id": str(msg.client_msg_id),
                    "seq": self.seq,
                    "mission_id": self.mission_id,
                    "role": "brain",
                    "author_ref": "brain:ground",
                    "kind": msg.kind,
                    "text": msg.text,
                    "in_reply_to": None if msg.in_reply_to is None else str(msg.in_reply_to),
                    "refs": msg.refs,
                    "author_role": None,
                    "parsed_intent": None,
                    "handled_by": None,
                    "delivery": "n/a",
                    "created_at": _iso(self._now()),
                }
            )
        )
        return str(msg.client_msg_id)

    def _match(self, skill: str, params: dict[str, Any]) -> dict[str, Any] | None:
        for rule in [*self.step_responses, *self.responses]:
            m = rule.get("match") or {}
            if m.get("skill") not in (None, skill):
                continue
            want = m.get("params") or {}
            if any(params.get(k) != val for k, val in want.items()):
                continue
            if rule.get("times") is not None:
                if rule["times"] <= 0:
                    continue
                rule["times"] -= 1
            return dict(rule["result"])
        return None

    async def submit(self, mission_id: str, req: SkillRequest) -> SkillSubmitResult:
        self.commands.append(req)
        from sd_brain.state_view import motion_block

        spec = self.reg.get(req.skill) if req.skill in self.reg else None
        res = self._match(req.skill, req.params)
        block = motion_block(self.situation_model())
        if res is None and block is not None and not (spec and spec.allowed_under_latch):
            res = {
                "decision": "REJECT",
                "reason_codes": [_LATCH_CODES[block]],
                "message_sv": "Spärren gäller.",
                "status": "rejected_by_gate",
            }
        if res is None:
            done = spec is not None and spec.runs_on == "ground"
            res = {"decision": "ALLOW", "status": "done" if done else "dispatched"}
        result = SkillSubmitResult.model_validate(
            {
                "command_id": str(req.command_id),
                "decision": res["decision"],
                "reason_codes": res.get("reason_codes", []),
                "message_sv": res.get("message_sv", ""),
                "details": res.get("details", {}),
                "approval_id": str(new_id()) if res["decision"] == "PENDING_APPROVAL" else None,
                "status": res.get(
                    "status",
                    {"ALLOW": "dispatched", "PENDING_APPROVAL": "pending_approval"}.get(
                        res["decision"], "rejected_by_gate"
                    ),
                ),
            }
        )
        self.results.append(result)
        if result.decision == "ALLOW" and spec is not None and spec.runs_on == "onboard":
            self.sit["active_command"] = None
        return result

    async def estimate(self, mission_id: str, req: SkillRequest) -> EstimateResult:
        gate = GateDecision(
            decision="ALLOW", reason_codes=[], rule_ids=[], message_sv="", details={}, side="ground"
        )
        return EstimateResult(gate=gate, duration_s=60.0, battery_after_pct=60.0)

    async def propose_charter(
        self, mission_id: str, proposal: CharterProposal
    ) -> CharterProposalOutcome:
        self.proposals.append(proposal)
        res = self._match("propose_charter_change", proposal.model_dump(mode="json")) or {
            "ok": True
        }
        if res.get("ok", True):
            return CharterProposalOutcome(ok=True, proposal_id=new_id(), approval_id=new_id())
        return CharterProposalOutcome(
            ok=False, errors=[CharterError.model_validate(e) for e in res.get("errors", [])]
        )

    async def memory(self, mission_id: str) -> MissionMemory:
        return self.mem

    async def put_memory(self, mission_id: str, mem: MissionMemoryUpdate) -> None:
        self.mem = MissionMemory.model_validate(
            {**mem.model_dump(mode="json"), "updated_at": _iso(self._now())}
        )

    async def post_decision(self, mission_id: str, entry: DecisionLog) -> None:
        self.decisions.append(entry)

    async def report_chat_intent(self, mission_id: str, report: ChatIntentReport) -> bool:
        if not self.chat_intents:
            return False
        self.intents.append(report)
        return True
