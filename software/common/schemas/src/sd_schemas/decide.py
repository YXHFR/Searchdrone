"""The decision interface (``sd_brain.decide``): a state plus typed questions -> probabilities.

The same questions go to whichever engine is available (cloud first, local as the
fallback), and every answer is logged so the engines can be compared. In v0.1.0 it
backs the quick stop check on chat messages; images and candidates come later.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import Id, SdModel

__all__ = [
    "BoxAnswer",
    "BoxQuestion",
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
    image_ref: str  # mission file id or path, never inline bytes in the log


DecideQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion | BoxQuestion, Field(discriminator="type")
]


class DecideRequest(SdModel):
    request_id: Id
    purpose: str = Field(pattern=_NAME)  # e.g. "stop_check"
    state: dict[str, Any]
    questions: dict[Annotated[str, Field(pattern=_NAME)], DecideQuestion] = Field(min_length=1)


class NoulAnswer(SdModel):
    type: Literal["noul"] = "noul"
    p_true: float = Field(ge=0, le=1)


class ChoiceAnswer(SdModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def _choice_listed(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError("choice must be one of the probabilities' keys")
        return self


class ScoreAnswer(SdModel):
    type: Literal["score"] = "score"
    score: int = Field(ge=0)
    probabilities: list[float]  # index = score


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
