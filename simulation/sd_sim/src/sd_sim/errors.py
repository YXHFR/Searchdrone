"""Errors from the simulator tooling."""

from __future__ import annotations


class InfraError(RuntimeError):
    """The test rig failed (a process did not start, a port was busy), not the code under test."""
