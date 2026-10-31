"""``CloudImagePolicy``: the only images that may go to a cloud model are small crops.

Every image in a request to a cloud URL is a crop around one detection: at most three
per call, each a JPEG of at most 512 x 512 decoded pixels, re-encoded here so that no
EXIF or other metadata (camera, time, position) leaves the ground station. A whole
still, a view, a video frame, a larger picture, a PNG or a fourth image is refused with
:class:`CloudImageRejected` before anything is sent. The size is read from the decoded
image, never from metadata.

:meth:`CloudImagePolicy.check` takes the parts a caller wants to send and returns them
as chat content parts; :func:`check_messages` runs in ``CloudModelRuntime`` on every
request and refuses any image part that is not such a crop (a URL to fetch, another
format, a larger image, too many), so the commander's calls are held to the same rule.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from PIL import Image

from sd_brain.runtime.base import ModelError

__all__ = [
    "MAX_IMAGES",
    "MAX_SIDE_PX",
    "CloudImagePolicy",
    "CloudImageRejected",
    "ImageOrigin",
    "ImagePart",
    "check_messages",
]

log = logging.getLogger("sd_brain.runtime.images")

MAX_IMAGES: Final = 3
MAX_SIDE_PX: Final = 512
JPEG_QUALITY: Final = 90
_DATA_URL: Final = "data:image/jpeg;base64,"
# Read at most this much before decoding: a 512 x 512 JPEG is far smaller.
MAX_BYTES: Final = 2_000_000

ImageOrigin = Literal["crop", "still", "view", "video", "frame"]


class CloudImageRejected(ModelError):  # noqa: N818 - the name the design uses
    """An image that may not leave the ground station. Nothing was sent; to a caller it
    is a failed cloud call like any other."""


@dataclass(frozen=True)
class ImagePart:
    origin: ImageOrigin
    mime: str
    data: bytes


def _reject(why: str) -> CloudImageRejected:
    log.warning("cloud_image_rejected: %s", why)
    return CloudImageRejected(why)


def _clean_jpeg(data: bytes) -> bytes:
    """Decode, check the decoded size and encode again without any metadata."""
    if len(data) > MAX_BYTES:
        raise _reject("bilden är för stor")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "JPEG":
                raise _reject(f"bara JPEG, inte {img.format}")
            w, h = img.size  # from the frame header, checked before the pixels are decoded
            if w > MAX_SIDE_PX or h > MAX_SIDE_PX or w < 1 or h < 1:
                raise _reject(f"utsnittet är {w} x {h} px, högst {MAX_SIDE_PX}")
            img.load()
            if img.size != (w, h):
                raise _reject("bildens storlek ändrades vid avkodning")
            rgb = img.convert("RGB")
    except CloudImageRejected:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise _reject(f"bilden går inte att avkoda ({type(exc).__name__})") from None
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=JPEG_QUALITY)  # no exif=, no icc, no comment
    return out.getvalue()


# Segments a clean JPEG may hold before its scan: JFIF (APP0), quantisation and Huffman
# tables, restart interval and the frame headers. APP1-APP15 (EXIF, XMP, ICC, maker
# notes) and comments are metadata.
_ALLOWED_SEGMENTS: Final = frozenset({0xE0, 0xDB, 0xC4, 0xDD, *range(0xC0, 0xC4)})


def _metadata_free(data: bytes) -> bool:
    if data[:2] != b"\xff\xd8":
        return False
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return False
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the headers are over
            return True
        if marker not in _ALLOWED_SEGMENTS:
            return False
        i += 2 + int.from_bytes(data[i + 2 : i + 4], "big")
    return False


class CloudImagePolicy:
    """The rule for images to a cloud model (crops only, small, at most three, clean)."""

    max_images = MAX_IMAGES
    max_side_px = MAX_SIDE_PX

    def check(self, parts: Sequence[ImagePart]) -> list[dict[str, Any]]:
        """The parts as chat content parts, or :class:`CloudImageRejected` for any one
        that may not go. All or none."""
        if len(parts) > MAX_IMAGES:
            raise _reject(f"{len(parts)} bilder, högst {MAX_IMAGES} per anrop")
        out: list[dict[str, Any]] = []
        for part in parts:
            if part.origin != "crop":
                raise _reject(f"bara utsnitt, inte {part.origin}")
            if part.mime != "image/jpeg":
                raise _reject(f"bara image/jpeg, inte {part.mime}")
            clean = _clean_jpeg(part.data)
            url = _DATA_URL + base64.b64encode(clean).decode("ascii")
            out.append({"type": "image_url", "image_url": {"url": url}})
        return out


def _image_url(part: dict[str, Any]) -> str:
    raw = part.get("image_url")
    if isinstance(raw, dict):
        url = cast("dict[str, Any]", raw).get("url")
        return url if isinstance(url, str) else ""
    return raw if isinstance(raw, str) else ""


def check_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The request as it may leave: refused unless its images are all crops (at most
    three in all), and every image replaced by its re-encoded copy.

    Only ``text`` and ``image_url`` content parts are allowed; an image must be an inline
    JPEG (never a URL the provider would fetch), at most 512 x 512 decoded, with no
    metadata segment (as :meth:`CloudImagePolicy.check` produces it). What is sent is
    always the copy encoded here, never the caller's bytes, so nothing hidden in them
    (after the image's end, in a segment the scan does not read) can leave.
    """
    images = 0
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts: list[Any] = []
        for raw in cast("list[Any]", content):
            if not isinstance(raw, dict):
                raise _reject("okänd del i meddelandet")
            part = cast("dict[str, Any]", raw)
            kind = part.get("type")
            if kind == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise _reject("textdelen saknar text")
                parts.append({"type": "text", "text": text})  # no other key goes along
                continue
            if kind != "image_url":
                raise _reject(f"bara text och utsnitt, inte {kind}")
            images += 1
            if images > MAX_IMAGES:
                raise _reject(f"fler än {MAX_IMAGES} bilder i anropet")
            url = _image_url(part)
            if not url.startswith(_DATA_URL):
                raise _reject("bara inbäddade JPEG-utsnitt, ingen adress")
            try:
                data = base64.b64decode(url.removeprefix(_DATA_URL), validate=True)
            except (binascii.Error, ValueError):
                raise _reject("utsnittet är inte giltig base64") from None
            if not _metadata_free(data):
                raise _reject("utsnittet har metadata (EXIF, kommentar)")
            clean = _clean_jpeg(data)  # decodes, checks the size, encodes again
            url = _DATA_URL + base64.b64encode(clean).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        out.append({**msg, "content": parts})
    return out
