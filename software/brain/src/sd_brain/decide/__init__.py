"""The decision interface (``sd_brain.decide``): state + typed questions → probabilities.

Three engines behind the same questions: the local model, Jev (TypeSafe AI) and the
OpenAI Decisions API. The best available engine decides and the local one is the
reserve. In v0.1.0 it backs the quick stop check on chat messages.
"""

from sd_brain.decide.engines import (
    DecideError,
    EngineAnswer,
    JevEngine,
    LocalDecisionEngine,
    OpenAiDecisionsEngine,
    QuestionEngine,
)
from sd_brain.decide.service import DecisionService, EngineOutcome
from sd_brain.decide.stop_check import StopCheckResult, stop_check, stop_check_request

__all__ = [
    "DecideError",
    "DecisionService",
    "EngineAnswer",
    "EngineOutcome",
    "JevEngine",
    "LocalDecisionEngine",
    "OpenAiDecisionsEngine",
    "QuestionEngine",
    "StopCheckResult",
    "stop_check",
    "stop_check_request",
]
