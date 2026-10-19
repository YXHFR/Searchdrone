"""Simulator exams for the AI commander: random searches with an answer key, graded.

The simulator farm uses three things from here: ``make_decision_maker`` (the cloud model
under the daily cost cap, the script otherwise; a ``CommanderBackend``), ``score_flight``
(a ``FlightLog`` in, a ``FlightScore`` out) and ``worth_keeping``/``bank_from_env`` (the
example bank). ``sd-brain eval --backend sim`` runs the same pieces against a fast 2D world.
"""

from sd_brain.simeval.bank import bank_from_env, example_record, worth_keeping
from sd_brain.simeval.budget import DAILY_CAP_SEK, CostLedger, MeteredRuntime
from sd_brain.simeval.decider import DecisionMaker, SearchScript, make_decision_maker
from sd_brain.simeval.score import FlightLog, FlightScore, score_flight

__all__ = [
    "DAILY_CAP_SEK",
    "CostLedger",
    "DecisionMaker",
    "FlightLog",
    "FlightScore",
    "MeteredRuntime",
    "SearchScript",
    "bank_from_env",
    "example_record",
    "make_decision_maker",
    "score_flight",
    "worth_keeping",
]
