"""Tests cannot leave loopback (B1-05 point 8, 04 T20): a call to the internet fails the test."""

from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def test_external_address_is_blocked() -> None:
    with (
        pytest.raises(SocketConnectBlockedError),
        socket.create_connection(("203.0.113.10", 443), timeout=1),
    ):
        pass


def test_loopback_is_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            pass
