"""The training scripts import ``bench`` (tools/) and their neighbours (tools/train)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
