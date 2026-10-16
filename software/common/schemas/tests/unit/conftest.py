import sys
from pathlib import Path

# Lets the tests here import sd_schemas_examples (pytest runs with --import-mode=importlib).
sys.path.insert(0, str(Path(__file__).parent))
