"""Pure safety-gate logic: every order to the vehicle is checked here.

``evaluate(req, ctx)`` is the gate. It does no I/O and never reads a clock; the caller
passes the time in ``GateContext.now_mono_ms``. The same function runs on the ground and
onboard (``ctx.side``).
"""

from importlib.metadata import version

from sd_safety.evaluate import evaluate

__all__ = ["__version__", "evaluate"]

__version__ = version("sd-safety")
