"""Replay fixtures from good exam flights: ``sd-brain bank export-golden`` and their replay.

A fixture is the scenario seed plus the cloud model's answers in order, nothing else: no
prompt, no flight id, no model name, no time stamps. The seed rebuilds the same synthetic
scenario around the public SITL site (``make_scenario``) and the world is seeded too, so
flying the answers again through the real loop and the gate gives the same flight. That is
what ``sd-brain eval --backend replay`` (``just brain-eval``) does for every fixture in
``tests/golden/sim/``, without a network: a change to the loop, the tools or the gate that
breaks a flight the model once flew well shows up in CI.

Export refuses a flight whose answers hold anything that looks like a key, a token, an
e-mail address or a phone number, or the value of a secret set in the environment.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sd_brain.config import BehaviourConfig
from sd_brain.runtime.base import ModelError, ModelReply, Sampling
from sd_brain.runtime.replay import load_golden_rows
from sd_brain.secrets import BRAIN_KEYS
from sd_brain.simeval.bank import R2_ENV
from sd_brain.simeval.decider import DecisionMaker, cloud_commander
from sd_brain.simeval.runner import fly
from sd_brain.simeval.scenario import make_scenario

__all__ = [
    "GOLDEN_SCHEMA",
    "SECRET_ENV",
    "GoldenError",
    "export_golden",
    "load_fixture",
    "refuse_secrets",
    "replay_golden",
]

GOLDEN_SCHEMA = "sd-brain-golden-sim/1"
# Points below the recorded grade a replay may land before it counts as a regression.
TOTAL_SLACK = 10.0
# Built from the lists the keys are read by, so a new key the AI reads is refused here too.
SECRET_ENV = (*BRAIN_KEYS, *R2_ENV, "OPENROUTER_API_KEY", "SD_BRAIN_TOKEN")
_LOOKS_SECRET = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"[A-Za-z0-9+/=_-]{40,}"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\+?\d[\d ()-]{8,}\d"),
)


class GoldenError(ValueError):
    pass


def _clean_reply(raw: Mapping[str, Any]) -> dict[str, Any]:
    calls = []
    for c in raw.get("tool_calls") or []:
        args = c.get("arguments", {})
        if isinstance(args, str):
            # A broken JSON stays text: the replay must see the same broken answer.
            with contextlib.suppress(ValueError):
                args = json.loads(args)
        calls.append({"name": str(c["name"]), "arguments": args})
    return {"content": raw.get("content"), "tool_calls": calls}


def refuse_secrets(replies: list[dict[str, Any]], env: Mapping[str, str]) -> None:
    """Raise :class:`GoldenError` when recorded answers hold a key or personal data."""
    text = json.dumps(replies, ensure_ascii=False)
    for name in SECRET_ENV:
        value = env.get(name, "").strip()
        if len(value) >= 8 and value in text:
            raise GoldenError(f"svaren innehåller värdet av {name}; exporteras inte")
    for pat in _LOOKS_SECRET:
        if pat.search(text):
            raise GoldenError("svaren innehåller något som liknar en nyckel eller personuppgift")


def export_golden(record: Mapping[str, Any], name: str, env: Mapping[str, str]) -> dict[str, Any]:
    """A replay fixture from a bank record of a good exam flight."""
    if not re.fullmatch(r"SIM-[0-9]{2,3}", name):
        raise GoldenError("fixturens namn ska vara SIM-NN")
    score = record.get("score") or {}
    if not (score.get("exam") and score.get("passed")):
        raise GoldenError("bara godkända prov blir replay-fixturer")
    raw = record.get("replies") or []
    if not raw:
        raise GoldenError("flygningen saknar modellens svar")
    replies = [_clean_reply(r) for r in raw]
    refuse_secrets(replies, env)
    return {
        "schema": GOLDEN_SCHEMA,
        "id": name,
        "seed": int(record["scenario"]["seed"]),
        "max_turns": int(score.get("turns") or 40) + 5,
        "expect": {
            "safety_ok": True,
            "found": bool(score.get("found")),
            "min_total": max(0.0, round(float(score["total"]) - TOTAL_SLACK, 1)),
        },
        "replies": replies,
    }


class _StrictReplay:
    """Replays the answers; a call past the last one ends the cloud part of the flight."""

    name = "replay"
    model_id = "replay"
    supports_images = False

    def __init__(self, replies: list[ModelReply]) -> None:
        self._replies = list(replies)
        self.overrun = False

    @property
    def left(self) -> int:
        return len(self._replies)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        if not self._replies:
            self.overrun = True
            raise ModelError("replay: the flight asked for more answers than recorded")
        return self._replies.pop(0)


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def replay_golden(fx: Mapping[str, Any]) -> list[str]:
    """Fly one fixture again; the list of what differs from the recording (empty: ok)."""
    if fx.get("schema") != GOLDEN_SCHEMA:
        return [f"okänt schema {fx.get('schema')!r}"]
    rows = [{"step": 0, "call": i, "reply": r} for i, r in enumerate(fx["replies"])]
    runtime = _StrictReplay(load_golden_rows(rows))
    behaviour = BehaviourConfig()
    decider = DecisionMaker(cloud_commander(runtime, behaviour), None)
    res = await fly(
        make_scenario(int(fx["seed"])),
        decider,
        model="replay",
        ai_config_hash="replay",
        max_turns=int(fx.get("max_turns", 40)),
    )
    exp, sc = fx["expect"], res.score
    failures = []
    if runtime.overrun:
        failures.append("flygningen bad om fler svar än inspelningen har")
    elif runtime.left:
        failures.append(f"{runtime.left} inspelade svar användes inte")
    if sc.safety.ok != exp["safety_ok"]:
        failures.append(f"säkerhet {sc.safety.model_dump()}")
    if sc.found != exp["found"]:
        failures.append(f"hittad {sc.found}, väntat {exp['found']}")
    if sc.total < exp["min_total"]:
        failures.append(f"betyg {sc.total:.1f} under {exp['min_total']:.1f}")
    return failures
