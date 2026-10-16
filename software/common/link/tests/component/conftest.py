import sys
from pathlib import Path

# Lets the tests import link_helpers (pytest runs with --import-mode=importlib).
sys.path.insert(0, str(Path(__file__).parents[1]))
