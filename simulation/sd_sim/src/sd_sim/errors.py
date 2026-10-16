"""Errors from the simulator tooling."""

from __future__ import annotations


class InfraError(RuntimeError):
    """The test rig failed (a process did not start, a port was busy), not the code under test."""


class VehicleError(RuntimeError):
    """The simulated vehicle did not do what a test asked in time (mode, arming, altitude,
    a rejected command). A product failure: unlike ``InfraError`` it is never retried."""


class StopRequestedError(Exception):
    """Ctrl-C or ``just stop`` arrived; stop everything that was started."""
