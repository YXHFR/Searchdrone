"""Test-only helpers shared by every package. Never imported by production packages."""

from importlib.metadata import version

from sd_testing.clock import FakeClock

__version__ = version("sd-testing")

__all__ = ["FakeClock", "__version__"]
