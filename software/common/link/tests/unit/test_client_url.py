"""The link token never travels in clear text to another machine."""

from __future__ import annotations

import pytest

from sd_link.client import InsecureLinkUrlError, LinkClient, is_safe_url
from sd_schemas.link import Hello
from sd_testing import FakeClock


@pytest.mark.parametrize(
    "url",
    [
        "ws://127.0.0.1:7600/sdlink/v1/control",
        "ws://127.5.6.7:7600/sdlink/v1/control",
        "ws://localhost:7600/sdlink/v1/control",
        "ws://[::1]:7600/sdlink/v1/control",
        "wss://ground.example.org/sdlink/v1/control",
        "wss://192.0.2.10:8443/sdlink/v1/control",
    ],
)
def test_safe_urls(url: str) -> None:
    assert is_safe_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "ws://192.0.2.10:7600/sdlink/v1/control",
        "ws://ground.example.org/sdlink/v1/control",
        "ws://localhost.example.org/sdlink/v1/control",
        "ws://127.0.0.1.example.org/sdlink/v1/control",
        "ws:///sdlink/v1/control",
        "http://127.0.0.1:7600/",
        "not a url",
    ],
)
def test_token_refused_in_clear_to_another_host(url: str) -> None:
    assert not is_safe_url(url)

    def make_hello() -> Hello:
        raise AssertionError("never dialled")

    async def on_session(*_: object) -> None:
        raise AssertionError("never connected")

    with pytest.raises(InsecureLinkUrlError):
        LinkClient(
            url=url,
            token="x" * 43,
            vehicle_id="sim-01",
            make_hello=make_hello,
            on_session=on_session,
            clock=FakeClock(),
        )
