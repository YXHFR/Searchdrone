"""E4: only small, clean crops go to a cloud model; everything else is refused unsent."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from brain_helpers import chat_reply, mock_client
from PIL import Image

from sd_brain.config import CloudConfig
from sd_brain.runtime.base import ModelError, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.images import (
    CloudImagePolicy,
    CloudImageRejected,
    ImagePart,
    check_messages,
)

KEY = "k" * 40


def _jpeg(w: int, h: int, *, exif: bool = False, fmt: str = "JPEG", mode: str = "RGB") -> bytes:
    img = Image.new(mode, (w, h), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128))
    out = io.BytesIO()
    if exif:
        meta = Image.Exif()
        meta[0x010F] = "kamera"  # Make
        img.save(out, format=fmt, exif=meta.tobytes())
    else:
        img.save(out, format=fmt)
    return out.getvalue()


def _crop(side: int = 384) -> ImagePart:
    return ImagePart(origin="crop", mime="image/jpeg", data=_jpeg(side, side))


def _runtime(calls: list[httpx.Request]) -> CloudModelRuntime:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json=chat_reply("ok"))

    return CloudModelRuntime(CloudConfig(fallback_model=""), KEY, mock_client(handler))


def _msgs(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"role": "user", "content": [{"type": "text", "text": "titta"}, *parts]}]


@pytest.mark.parametrize(
    "parts",
    [
        [ImagePart(origin="still", mime="image/jpeg", data=_jpeg(384, 384))],
        [ImagePart(origin="view", mime="image/jpeg", data=_jpeg(384, 384))],
        [ImagePart(origin="frame", mime="image/jpeg", data=_jpeg(64, 64))],
        [ImagePart(origin="crop", mime="image/jpeg", data=_jpeg(640, 384))],
        [ImagePart(origin="crop", mime="image/jpeg", data=_jpeg(384, 513))],
        [_crop(), _crop(), _crop(), _crop()],
        [ImagePart(origin="crop", mime="image/png", data=_jpeg(64, 64, fmt="PNG", mode="RGBA"))],
        [ImagePart(origin="crop", mime="image/jpeg", data=_jpeg(64, 64, fmt="PNG"))],
        [ImagePart(origin="crop", mime="image/jpeg", data=b"\xff\xd8 not a jpeg")],
    ],
    ids=["still", "view", "frame", "640px", "513px", "four", "png", "png-as-jpeg", "broken"],
)
def test_cloud_image_policy(parts: list[ImagePart]) -> None:
    with pytest.raises(CloudImageRejected):
        CloudImagePolicy().check(parts)


async def test_three_crops_pass_cleaned_and_reach_the_cloud() -> None:
    calls: list[httpx.Request] = []
    parts = CloudImagePolicy().check(
        [ImagePart(origin="crop", mime="image/jpeg", data=_jpeg(384, 384, exif=True))] * 3
    )
    assert len(parts) == 3
    for p in parts:
        url = p["image_url"]["url"]  # type: ignore[index]
        data = base64.b64decode(url.removeprefix("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(data)) as img:
            assert img.format == "JPEG" and img.size == (384, 384)
            assert not img.getexif()  # the camera's metadata never leaves
    reply = await _runtime(calls).chat(_msgs(parts), [], 5, Sampling())
    assert reply.content == "ok" and len(calls) == 1


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "https://example.org/still.jpg"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,!!"},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(_jpeg(640, 640)).decode()
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(_jpeg(64, 64, exif=True)).decode()
            },
        },
    ],
    ids=["url", "png", "bad-base64", "audio", "640px", "exif"],
)
async def test_runtime_refuses_any_other_image_unsent(part: dict[str, object]) -> None:
    calls: list[httpx.Request] = []
    with pytest.raises(CloudImageRejected) as exc:
        await _runtime(calls).chat(_msgs([part]), [], 5, Sampling())
    assert isinstance(exc.value, ModelError)  # a failed cloud call to every caller
    assert calls == []


async def test_runtime_refuses_a_fourth_crop_unsent() -> None:
    calls: list[httpx.Request] = []
    three = CloudImagePolicy().check([_crop(64)] * 3)
    msgs = _msgs(three) + _msgs(CloudImagePolicy().check([_crop(64)]))
    with pytest.raises(CloudImageRejected):
        await _runtime(calls).chat(msgs, [], 5, Sampling())
    assert calls == []


def test_text_only_messages_pass() -> None:
    check_messages([{"role": "system", "content": "regler"}, {"role": "user", "content": []}])


def test_a_text_part_goes_on_with_its_text_only() -> None:
    part = {"type": "text", "text": "titta", "image_url": {"url": "https://x"}, "cache": 1}
    (msg,) = check_messages([{"role": "user", "content": [part]}])
    assert msg["content"] == [{"type": "text", "text": "titta"}]


@pytest.mark.parametrize("part", [{"type": "text"}, {"type": "text", "text": ["a"]}])
def test_a_text_part_without_text_is_refused(part: dict[str, object]) -> None:
    with pytest.raises(CloudImageRejected):
        check_messages([{"role": "user", "content": [part]}])


def _app0(payload: bytes) -> bytes:
    """A JPEG whose APP0 segment is followed by a second, free-form APP0."""
    data = _jpeg(64, 64)
    seg = b"\xff\xe0" + (len(payload) + 2).to_bytes(2, "big") + payload
    return data[:2] + seg + data[2:]


def _progressive_with_comment() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(out, format="JPEG", progressive=True)
    data = out.getvalue()
    second_scan = data.index(b"\xff\xda", data.index(b"\xff\xda") + 2)
    note = b"GPSLat=59.1"
    com = b"\xff\xfe" + (len(note) + 2).to_bytes(2, "big") + note
    return data[:second_scan] + com + data[second_scan:]


@pytest.mark.parametrize(
    "data",
    [
        _jpeg(64, 64) + b"GPSLat=59.1;" + _jpeg(500, 400),  # after the end marker
        _app0(b"GPSLat=59.1;camera=SN123"),
        _progressive_with_comment(),
    ],
    ids=["trailing", "app0-text", "comment-between-scans"],
)
async def test_runtime_sends_only_its_own_copy(data: bytes) -> None:
    """Whatever hides in the caller's bytes, what leaves is the copy encoded here."""
    calls: list[httpx.Request] = []
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    part = {"type": "image_url", "image_url": {"url": url}}
    try:
        await _runtime(calls).chat(_msgs([part]), [], 5, Sampling())
    except CloudImageRejected:
        assert calls == []
        return
    (req,) = calls
    sent = json.loads(req.content)["messages"][0]["content"][1]["image_url"]["url"]
    image = base64.b64decode(sent.removeprefix("data:image/jpeg;base64,"))
    assert image != data and b"GPSLat" not in image
    assert b"GPSLat" not in req.content
