"""The decision interface (``sd_brain.decide``): state + typed questions → probabilities.

Two hosted engines behind the same questions: Jev (TypeSafe AI) and the OpenAI Decisions
API. The best available engine decides. There is no local engine: without a hosted one,
the stop check rests on the fixed stop words. In v0.1.0 it backs the quick stop check on
chat messages.
"""

from sd_brain.decide.engines import (
    DecideError,
    EngineAnswer,
    JevEngine,
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
    "OpenAiDecisionsEngine",
    "QuestionEngine",
    "StopCheckResult",
    "stop_check",
    "stop_check_request",
]
