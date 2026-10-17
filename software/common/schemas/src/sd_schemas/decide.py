"""The decision interface (``sd_brain.decide``): a state plus typed questions -> probabilities.

The same questions go to whichever engine is available (cloud first, local as the
fallback), and every answer is logged so the engines can be compared. In v0.1.0 it
backs the quick stop check on chat messages; images and candidates come later.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import Id, SdModel, json_bytes_at_most
from sd_schemas.sv_commands import OverrideIntent

__all__ = [
    "STOP_CHECK_TARGET_S",
    "BoxAnswer",
    "BoxQuestion",
    "ChatIntentReport",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecideAnswer",
    "DecideQuestion",
    "DecideRequest",
    "DecideResult",
    "DecisionEngine",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
]

DecisionEngine = Literal["local", "jev", "openai_decisions"]
_NAME = r"^[a-z][a-z0-9_]{0,63}$"


class NoulQuestion(SdModel):
    """Yes/no. The answer is the probability that the statement is true."""

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1, max_length=2000)


class ChoiceQuestion(SdModel):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: dict[Annotated[str, Field(pattern=_NAME)], str] = Field(min_length=2, max_length=255)


class ScoreQuestion(SdModel):
    """Place the state on an ordered scale ``0 .. max_score``."""

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1, max_length=2000)
    max_score: int = Field(ge=1, le=10)


class BoxQuestion(SdModel):
    """Where in an image something is (stage 2; schema only)."""

    type: Literal["box"] = "box"
    instructions: str = Field(min_length=1, max_length=2000)
    image_id: Id  # a MissionFile id; never a path and never inline bytes


DecideQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion | BoxQuestion, Field(discriminator="type")
]


class DecideRequest(SdModel):
    request_id: Id
    purpose: str = Field(pattern=_NAME)  # e.g. "stop_check"
    state: Annotated[dict[str, Any], json_bytes_at_most(32 * 1024)]
    questions: dict[Annotated[str, Field(pattern=_NAME)], DecideQuestion] = Field(
        min_length=1, max_length=64
    )


class NoulAnswer(SdModel):
    type: Literal["noul"] = "noul"
    p_true: float = Field(ge=0, le=1)


class ChoiceAnswer(SdModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Annotated[float, Field(ge=0, le=1)]]

    @model_validator(mode="after")
    def _choice_listed(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError("choice must be one of the probabilities' keys")
        return self


class ScoreAnswer(SdModel):
    type: Literal["score"] = "score"
    score: int = Field(ge=0)
    probabilities: list[Annotated[float, Field(ge=0, le=1)]]  # index = score


class BoxAnswer(SdModel):
    type: Literal["box"] = "box"
    found: bool
    bbox: tuple[float, float, float, float] | None  # normalised x, y, w, h
    confidence: float = Field(ge=0, le=1)


DecideAnswer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | BoxAnswer, Field(discriminator="type")
]


class DecideResult(SdModel):
    request_id: Id
    engine: DecisionEngine
    location: Literal["cloud", "local"]
    model_id: str | None
    latency_ms: int = Field(ge=0)
    fallback_from: DecisionEngine | None = None  # set when a better engine failed first
    answers: dict[str, DecideAnswer]
    created_at: AwareDatetime


# The quick stop check on every chat line aims to answer within this time.
STOP_CHECK_TARGET_S = 0.5


class ChatIntentReport(SdModel):
    """The stop check found that an operator's chat line means STOP or FLY HOME.

    Sent by ``sd-brain`` to ``POST /commander/v1/missions/{id}/chat-intents``. The ground
    station checks that ``message_id`` is a pilot's own, recent chat line in that mission
    and names that pilot, never the AI. A line that is exactly the stop words acts like
    the button; a model's reading only ever stops the vehicle where it is. It can never
    release a stop.
    """

    request_id: Id
    message_id: Id
    intent: OverrideIntent
    probability: float = Field(ge=0, le=1)
    engine: DecisionEngine | Literal["rule"]
    location: Literal["cloud", "local"]
    latency_ms: int = Field(ge=0)
