"""``sd-brain eval-profiles``: does the AI fill in the search profile from a description?

Each case (``cases.yaml``) is one fictional Swedish description of a missing person, fed
to the commander as an operator line, with an answer key: the garments and items that
were said, their colours, and which of them were only a guess. The commander's
``propose_search_profile`` call is scored:

* **field accuracy**: every garment, item and hair in the key is two fields (the part or
  kind, and the main colour); at least ``FIELD_ACCURACY_MIN`` must be right;
* **invented and certain**: a garment, item or hair nobody mentioned, or a colour nobody
  said, marked ``certain``: there must be none (``INVENTED_CERTAIN_MAX``).

``--backend replay`` answers from ``golden.jsonl`` (the cloud model's answers from one
recorded run, no network: what CI runs); ``--backend cloud`` asks the cloud model and
``--record`` writes its answers as the new golden file. The descriptions are fictional
and the report holds scores only, never the profile text.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from sd_brain.backends import GrammarBackend, LlmCommander
from sd_brain.config import BrainConfig
from sd_brain.context import ContextBuilder
from sd_brain.decide.service import DecisionService
from sd_brain.evals.fake_ground import FakeGround
from sd_brain.evals.runner import EvalClock
from sd_brain.health import HealthState
from sd_brain.loop import CommanderLoop
from sd_brain.runtime.base import ModelReply, ModelRuntime, ModelTimeoutError, Sampling
from sd_brain.runtime.replay import ReplayExhaustedError, ReplayRuntime
from sd_schemas.search import SearchProfileDraft
from sd_schemas.skills import SKILL_REGISTRY

__all__ = [
    "FIELD_ACCURACY_MIN",
    "INVENTED_CERTAIN_MAX",
    "CaseScore",
    "RecordingRuntime",
    "load_cases",
    "load_profile_golden",
    "run_profiles",
    "score_profile",
]

FIELD_ACCURACY_MIN: Final = 0.9
INVENTED_CERTAIN_MAX: Final = 0
MAX_RETRIES: Final = 2  # per case, after the model never answered in time (a stall)
FIXTURE: Final = Path(__file__).parent / "fixtures" / "draft_p1.json"


@dataclass
class CaseScore:
    id: str
    proposed: bool
    fields: int = 0
    correct: int = 0
    invented_certain: int = 0
    overconfident: int = 0  # a guess in the key that came back as certain
    misses: list[str] = field(default_factory=list[str])  # field keys, never values


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    return [dict(c) for c in cases]


def _match(
    expected: list[dict[str, Any]], got: list[Any], key: str
) -> list[tuple[dict[str, Any], Any | None]]:
    """Pair each expected entry with an unused proposed one of an accepted part or kind:
    first those whose main colour is right too, then on the part or kind alone."""
    used: set[int] = set()
    hits: dict[int, Any] = {}

    def accepted(exp: dict[str, Any]) -> list[Any]:
        return exp[key] if isinstance(exp[key], list) else [exp[key]]

    for strict in (True, False):
        for n, exp in enumerate(expected):
            if n in hits:
                continue
            for i, g in enumerate(got):
                if i in used or getattr(g, key) not in accepted(exp):
                    continue
                if strict and exp.get("colors") and _main_color(g) not in exp["colors"]:
                    continue
                used.add(i)
                hits[n] = g
                break
    return [(exp, hits.get(n)) for n, exp in enumerate(expected)]


def _main_color(entry: Any) -> str | None:
    colors = getattr(entry, "colors", None)
    if colors is None:
        color = getattr(entry, "color", None)
        return None if color is None else str(color)
    return str(colors[0]) if colors else None


def score_profile(case: dict[str, Any], draft: SearchProfileDraft | None) -> CaseScore:
    score = CaseScore(id=str(case["id"]), proposed=draft is not None)
    key = case["expect"]
    groups: list[tuple[str, str, list[dict[str, Any]], list[Any]]] = [
        ("garment", "part", list(key.get("garments") or []), []),
        ("item", "kind", list(key.get("items") or []), []),
    ]
    hair = key.get("hair")
    if draft is not None:
        groups[0][3].extend(draft.garments)
        groups[1][3].extend(draft.items)
    for name, attr, expected, got in groups:
        pairs = _match(expected, got, attr)
        for i, (exp, hit) in enumerate(pairs):
            score.fields += 1 + (1 if exp.get("colors") else 0)
            if hit is None:
                score.misses.append(f"{name}:{i}")
                continue
            score.correct += 1
            if exp.get("colors"):
                if _main_color(hit) in exp["colors"]:
                    score.correct += 1
                else:
                    score.misses.append(f"{name}:{i}:color")
                    if hit.certainty == "certain":
                        score.invented_certain += 1
            if exp.get("unsure") and hit.certainty == "certain":
                score.overconfident += 1
        matched = {id(h) for _, h in pairs if h is not None}
        optional = {str(o) for o in key.get(f"optional_{name}s") or []}
        for extra in got:
            if id(extra) in matched or str(getattr(extra, attr)) in optional:
                continue
            if extra.certainty == "certain":
                score.invented_certain += 1
    if hair is not None:
        score.fields += 1
        got_hair = None if draft is None else draft.hair
        if got_hair is not None and str(got_hair.color) in hair["colors"]:
            score.correct += 1
        else:
            score.misses.append("hair")
            if got_hair is not None and got_hair.certainty == "certain":
                score.invented_certain += 1
    elif draft is not None and draft.hair is not None and draft.hair.certainty == "certain":
        score.invented_certain += 1
    return score


class RecordingRuntime:
    """Passes the calls on and keeps every answer, for a new golden file."""

    name = "recording"
    supports_images = False

    def __init__(self, inner: ModelRuntime) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.case = ""
        self.rows: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        reply = await self.inner.chat(messages, tools, timeout_s, sampling)
        calls = []
        for c in reply.tool_calls:
            try:
                args: Any = json.loads(c.arguments_json)
            except ValueError:
                args = c.arguments_json
            calls.append({"name": c.name, "arguments": args})
        n = sum(1 for r in self.rows if r["case"] == self.case)
        self.rows.append(
            {"case": self.case, "call": n, "reply": {"content": reply.content, "tool_calls": calls}}
        )
        return reply


def load_profile_golden(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.setdefault(str(row["case"]), []).append(row)
    return out


async def _run_case(
    case: dict[str, Any], runtime: ModelRuntime
) -> tuple[SearchProfileDraft | None, bool]:
    """The proposed profile, and whether the model never answered in time (a transport
    stall, not an extraction result)."""
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    clock = EvalClock()
    fg = FakeGround(copy.deepcopy(fx["situation"]), reg=SKILL_REGISTRY, now=clock.utc_now)
    cfg = BrainConfig()
    ctx = ContextBuilder(
        SKILL_REGISTRY, takeoff_policy=cfg.behaviour.takeoff_policy, prompt_budget_tokens=6000
    )
    smp = Sampling(temperature=0.2, top_p=0.8, max_tokens=1024, seed=42)
    commander = LlmCommander(runtime, ctx, SKILL_REGISTRY, cfg.behaviour, smp)
    loop = CommanderLoop(
        fg,
        commander,
        GrammarBackend(),
        SKILL_REGISTRY,
        cfg,
        "eval-profiles",
        health=HealthState(clock.monotonic),
        decide=DecisionService([]),
        clock=clock,  # type: ignore[arg-type]
    )
    await loop.set_mission(fg.mission_id)
    loop.ready_since = clock.utc_now()
    clock.t += 1
    loop.handle_event(fg.operator_message(str(case["text"]), created_at=clock.utc_now()))
    await loop.settle()
    diag = [r.output.diagnostics for r in loop.records if r.output is not None]
    timed_out = any(d.error == "timeout" and d.rounds == 0 for d in diag)
    return (fg.profiles[-1].profile if fg.profiles else None), timed_out


async def run_profiles(
    cases: list[dict[str, Any]], runtime_for: Callable[[str], ModelRuntime]
) -> dict[str, Any]:
    """Every case against the runtime ``runtime_for(case_id)`` gives; scores only."""
    scores: list[CaseScore] = []
    retries = 0
    for case in cases:
        draft, timed_out = await _run_case(case, runtime_for(str(case["id"])))
        for _ in range(MAX_RETRIES):
            if draft is not None or not timed_out:
                break
            # The model never answered in time: ask again (the report counts it).
            retries += 1
            draft, timed_out = await _run_case(case, runtime_for(str(case["id"])))
        scores.append(score_profile(case, draft))
    fields = sum(s.fields for s in scores)
    correct = sum(s.correct for s in scores)
    accuracy = correct / fields if fields else 0.0
    invented = sum(s.invented_certain for s in scores)
    return {
        "cases": len(scores),
        "proposed": sum(1 for s in scores if s.proposed),
        "fields": fields,
        "correct": correct,
        "field_accuracy": round(accuracy, 4),
        "invented_certain": invented,
        "overconfident": sum(s.overconfident for s in scores),
        "timeout_retries": retries,
        "passed": accuracy >= FIELD_ACCURACY_MIN and invented <= INVENTED_CERTAIN_MAX,
        "per_case": [
            {
                "id": s.id,
                "proposed": s.proposed,
                "correct": f"{s.correct}/{s.fields}",
                "invented_certain": s.invented_certain,
                "misses": s.misses,
            }
            for s in scores
        ],
    }


class _GoldenReplay(ReplayRuntime):
    """The recorded answers; past the last one, the model did not answer in the cloud run
    (a stalled call after the proposal), so the replay stalls there too."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        try:
            return await super().chat(messages, tools, timeout_s, sampling)
        except ReplayExhaustedError:
            raise ModelTimeoutError("the recorded model did not answer in time") from None


def replay_for(golden: dict[str, list[dict[str, Any]]]) -> Callable[[str], ModelRuntime]:
    from sd_brain.runtime.replay import load_golden_rows

    def make(case_id: str) -> ModelRuntime:
        rows = [
            {"step": 0, "call": r["call"], "reply": r["reply"]} for r in golden.get(case_id, [])
        ]
        return _GoldenReplay(load_golden_rows(rows))

    return make
