"""The mission chat and its two ways to STOP or FLY HOME.

* **Exact words** (owner decision): a pilot's line that is exactly "stopp", "stop" or
  "flyg hem" (case, punctuation and repeats aside) is STOP or FLY HOME at once, the same
  path as the buttons (``VehicleActions.chat_intent``, ``via = "chat"``), also when the AI
  or the network is down. It is decided before the line is stored, and the line is stored
  with ``parsed_intent`` and ``handled_by = "ground"``.
* **The AI's stop check** (``POST /commander/v1/missions/{id}/chat-intents``): the AI
  reports that a pilot's line means STOP or FLY HOME. The ground only accepts it for a
  pilot's own line in that mission at most ``INTENT_MAX_AGE_S`` old, and the pilot named
  is the line's author (never the AI). An exact-word report acts like the button. A
  report from a language model only ever stops the vehicle where it is (owner decision:
  it stands still until the pilot decides; it never flies home on a model's reading),
  and the chat says why.

Either way the same line acts once (``client_action_id = chat:<message id>``), and the
chat never releases a stop: words like "fortsätt" are ordinary lines for the AI.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any, Final, Literal

from sd_ground.auth import Principal
from sd_ground.logging import get_logger
from sd_ground.services.missions import FlowError
from sd_ground.sources import EmergencyUnavailableError
from sd_schemas.api import ActionDispatched, ChatMessage, ChatPost, Mission
from sd_schemas.base import new_id
from sd_schemas.sv_commands import OverrideIntent, classify_override

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["INTENT_MAX_AGE_S", "add_line", "exact_stop_word", "post_operator", "report_intent"]

INTENT_MAX_AGE_S: Final = 10.0
log = get_logger("sd_ground.chat")
_WORDS: Final = {"stopp": "stop", "stop": "stop", "flyg hem": "return_home"}
_PUNCT: Final = re.compile(r"[^\w\s]+")


def exact_stop_word(text: str) -> OverrideIntent | None:
    """``stop`` or ``return_home`` when the line is only the exact words (repeated words
    and punctuation allowed: "Stopp, stopp!"), else ``None``."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    norm = " ".join(_PUNCT.sub(" ", norm).split())
    if not norm:
        return None
    if norm in _WORDS:
        return _WORDS[norm]  # type: ignore[return-value]
    tokens = norm.split()
    if all(t in ("stopp", "stop") for t in tokens):
        return "stop"
    if len(tokens) % 2 == 0 and all(
        tokens[i] == "flyg" and tokens[i + 1] == "hem" for i in range(0, len(tokens), 2)
    ):
        return "return_home"
    return None


def _delivery(flow: MissionFlow) -> str:
    return "delivered" if flow.g.brain_state != "down" else "queued"


async def add_line(
    flow: MissionFlow,
    mission: Mission,
    *,
    msg_id: str,
    role: Literal["operator", "brain", "system"],
    author_ref: str,
    kind: str,
    text: str,
    in_reply_to: str | None,
    refs: dict[str, Any],
    parsed_intent: str | None = None,
) -> ChatMessage:
    if flow.g.repo is None:
        raise FlowError(503, "DB_DOWN", "Databasen svarar inte.")
    try:
        msg, created = await flow.g.repo.add_chat(
            mission_id=str(mission.id),
            vehicle_id=mission.vehicle_id,
            msg_id=msg_id,
            role=role,
            author_ref=author_ref,
            kind=kind,
            text_=text,
            in_reply_to=in_reply_to,
            refs=refs,
            parsed_intent=parsed_intent,
        )
    except LookupError:
        raise FlowError(409, "ID_CONFLICT", "Meddelandets id används redan.") from None
    if role == "operator":
        msg = msg.model_copy(update={"delivery": _delivery(flow)})
    if created:
        data = msg.model_dump(mode="json")
        data["annotation_ids"] = refs.get("annotation_ids", [])
        flow.publish(
            "chat", "chat.message", data, vehicle_id=mission.vehicle_id, mission_id=str(mission.id)
        )
    return msg


async def post_operator(
    flow: MissionFlow, mission_id: str, body: ChatPost, who: Principal
) -> ChatMessage:
    """``POST /api/v1/missions/{id}/chat``."""
    mission = await flow.open_mission_of(mission_id)
    repo = flow.g.repo
    if repo is None:
        raise FlowError(503, "DB_DOWN", "Databasen svarar inte.")
    if body.in_reply_to is not None and not await repo.chat_message_exists(
        mission_id, str(body.in_reply_to)
    ):
        raise FlowError(422, "REPLY_TARGET_UNKNOWN", "Meddelandet som svaret gäller finns inte.")
    msg_id = str(body.client_msg_id)
    intent: OverrideIntent | None = exact_stop_word(body.text) or classify_override(body.text)
    acted = None
    if intent is not None and who.role == "pilot":
        try:
            acted = _act(flow, mission, intent, msg_id, who)
        except EmergencyUnavailableError as exc:  # a refused action must not break the chat
            log.warning("chat.stop_word_refused", code=exc.code)
    return await add_line(
        flow,
        mission,
        msg_id=msg_id,
        role="operator",
        author_ref=who.issuer_ref,
        kind="text",
        text=body.text,
        in_reply_to=None if body.in_reply_to is None else str(body.in_reply_to),
        refs={"annotation_ids": [str(a) for a in body.annotation_ids]},
        parsed_intent=None if acted is None else intent,
    )


def _act(
    flow: MissionFlow, mission: Mission, intent: OverrideIntent, msg_id: str, by: Principal
) -> ActionDispatched | None:
    actions = flow.actions
    if actions is None:
        return None
    return actions.chat_intent(intent, vehicle_id=mission.vehicle_id, message_id=msg_id, by=by)


async def report_intent(
    flow: MissionFlow,
    mission_id: str,
    *,
    message_id: str,
    intent: OverrideIntent,
    engine: str,
    probability: float,
) -> ActionDispatched:
    """The AI's stop check said yes for a pilot's line (see the module text)."""
    mission = await flow.mission(mission_id)
    repo = flow.g.repo
    if repo is None:
        raise FlowError(503, "DB_DOWN", "Databasen svarar inte.")
    line = await repo.chat_line(mission_id, message_id)
    if line is None:
        raise FlowError(404, "MESSAGE_NOT_FOUND", "Chattraden finns inte i uppdraget.")
    if line.role != "operator" or line.author_role != "pilot":
        raise FlowError(403, "NOT_PILOT_LINE", "Bara pilotens egna rader kan stoppa drönaren.")
    age = (flow.now() - line.created_at).total_seconds()
    if mission.status.value == "CLOSED" or age > INTENT_MAX_AGE_S:
        raise FlowError(409, "MESSAGE_TOO_OLD", "Chattraden är för gammal för att stoppa drönaren.")
    by = Principal(role="pilot", issuer_ref=line.author_ref, auth_mode="local")
    model_reading = engine != "rule"
    action: OverrideIntent = "stop" if model_reading else intent
    try:
        acted = _act(flow, mission, action, str(line.id), by)
    except EmergencyUnavailableError as exc:
        raise FlowError(exc.status, exc.code, exc.detail_sv) from None
    if acted is None:
        raise FlowError(503, "NO_PATH", "Nödvägen är inte igång.")
    flow.g.add_event(
        "chat.intent",
        mission_id=mission_id,
        vehicle_id=mission.vehicle_id,
        payload={
            "message_id": str(line.id),
            "intent": intent,
            "acted": action,
            "engine": engine,
            "probability": probability,
        },
    )
    if model_reading:
        word = "flyga hem" if intent == "return_home" else "stoppa"
        await _explain(flow, mission, line.id, word, intent, engine)
    return acted


async def _explain(
    flow: MissionFlow, mission: Mission, line_id: Any, word: str, intent: str, engine: str
) -> ChatMessage:
    return await add_line(
        flow,
        mission,
        msg_id=str(new_id()),
        role="system",
        author_ref="system:ground",
        kind="status",
        text=(
            f"AI:n tolkade ditt meddelande som att du ville {word}. Drönaren står still "
            "tills du väljer: Fortsätt eller FLYG HEM."
        ),
        in_reply_to=str(line_id),
        refs={"chat_intent": intent, "engine": engine},
    )
