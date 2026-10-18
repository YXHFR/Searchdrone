"""The local language model is gone: no code, config or recipe brings it back.

The cloud model is the AI commander and the fixed command mode its only reserve. This
scans the source, the configs, the compose file and the justfile for the removed local
runtime, its server, its key and its recipes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Where the local model used to live. Tests and the changelog may still name it (to check
# that it is gone, or to record that it went).
SCANNED = (
    "software/brain/src",
    "software/ground/src",
    "software/common",
    "software/panel/src",
    "simulation/sd_sim/src",
    "config",
    "compose.yaml",
    "justfile",
    ".env.example",
)
FORBIDDEN = re.compile(
    r"sd_brain\.runtime\.local|from \.local import|LocalModelRuntime|LocalDecisionEngine"
    r"|checkmodel|models_fetch|llama|SD_LLM_API_KEY|llm_local|LLM__ROOT_URL|runtime\.lock"
    r"|models-fetch|check-model|brain-eval-model|llm-up|llm_http",
    re.IGNORECASE,
)
# Old configs are migrated on load; those two loaders name the old kind on purpose.
ALLOWED = {
    ("software/brain/src/sd_brain/config.py", "llm_local"),
    ("simulation/sd_sim/src/sd_sim/config.py", "llm_local"),
}


def _tracked(paths: tuple[str, ...]) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


def test_no_local_model_left() -> None:
    hits: list[str] = []
    for rel in _tracked(SCANNED):
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in FORBIDDEN.finditer(line):
                if (rel, m.group(0).lower()) not in ALLOWED:
                    hits.append(f"{rel}:{n}: {m.group(0)}")
    assert not hits, "the local model is back:\n" + "\n".join(hits)


def test_removed_modules_are_gone() -> None:
    for rel in (
        "software/brain/src/sd_brain/runtime/local.py",
        "software/brain/src/sd_brain/checkmodel.py",
        "software/brain/runtime.lock.toml",
        "tools/models_fetch.py",
    ):
        assert not (ROOT / rel).exists(), rel


def test_compose_has_no_model_server() -> None:
    services = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))["services"]
    assert "llm" not in services
    for name, svc in services.items():
        image = str(svc.get("image", ""))
        assert "llama" not in image.lower(), name
