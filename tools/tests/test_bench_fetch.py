"""``just bench-fetch``: nothing without yes=1, every archive checked, safe unpacking."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import http.client
import io
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest
from bench import common, fetch
from bench.common import BenchError, RangeFile, download, safe_relpath
from bench.datasets import Archive, load_register, read_index
from bench.splits import Split, ids_digest
from PIL import Image as PILImage

ROOT_DIR = "heridal_keras_retinanet_voc"


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, headers: dict[str, str]) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers


class FakeServer:
    """Serves one file, with or without byte ranges, and counts requests."""

    def __init__(self, body: bytes, ranges: bool = True, ctype: str = "application/zip") -> None:
        self.body = body
        self.ranges = ranges
        self.ctype = ctype
        self.requests: list[str | None] = []
        self.cut_after: int | None = None

    def __call__(self, req: urllib.request.Request) -> FakeResponse:
        rng = req.get_header("Range")
        self.requests.append(rng)
        headers = {"Content-Type": self.ctype}
        if rng and self.ranges:
            spec = rng.removeprefix("bytes=")
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(self.body) - 1
            chunk = self.body[start : end + 1]
            headers["Content-Range"] = f"bytes {start}-{end}/{len(self.body)}"
            if self.cut_after is not None:
                chunk = chunk[: self.cut_after]
                self.cut_after = None
            return FakeResponse(chunk, 206, headers)
        return FakeResponse(self.body, 200, headers)


def _jpeg(width: int = 64, height: int = 48) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), (40, 90, 40)).save(out, format="JPEG")
    return out.getvalue()


def _voc(boxes: list[tuple[int, int, int, int]]) -> bytes:
    objs = "".join(
        f"<object><name>person</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for a, b, c, d in boxes
    )
    return (
        f"<annotation><size><width>64</width><height>48</height></size>{objs}</annotation>".encode()
    )


def heridal_zip(extra: dict[str, bytes] | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(f"{ROOT_DIR}/JPEGImages/test_TRS_0001.jpg", _jpeg())
        zf.writestr(f"{ROOT_DIR}/Annotations/test_TRS_0001.xml", _voc([(10, 10, 30, 40)]))
        zf.writestr(f"{ROOT_DIR}/JPEGImages/train_VRD_0002.jpg", _jpeg())
        zf.writestr(f"{ROOT_DIR}/Annotations/train_VRD_0002.xml", _voc([]))
        zf.writestr(f"{ROOT_DIR}/ImageSets/Main/test.txt", b"test_TRS_0001\n")
        for name, body in (extra or {}).items():
            zf.writestr(name, body)
    return out.getvalue()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


# --- yes=1 -------------------------------------------------------------------------------


def test_nothing_is_fetched_without_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("downloaded without yes=1")

    monkeypatch.setattr(fetch, "download", boom)
    monkeypatch.setattr(fetch, "RangeFile", boom)
    assert fetch.main([]) == 2
    assert fetch.main(["sets=heridal"]) == 2
    assert fetch.main(["mini=1"]) == 2
    assert fetch.main(["yes=0", "sets=heridal"]) == 2
    out = capsys.readouterr().out
    assert "yes=1" in out
    assert "CC-BY-3.0" in out


def test_by_default_only_pinned_measured_archives_are_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetched: list[str] = []
    monkeypatch.setattr(fetch, "fetch_archive", lambda ds, data, keep: fetched.append(ds.name))
    assert fetch.main(["yes=1", f"data={tmp_path}"]) == 0
    register = load_register()
    assert fetched == [
        n for n, ds in register.items() if ds.url and fetch.pinned(ds) and "measure" in ds.use
    ]
    assert "wisard" not in fetched  # its SHA-256 is not pinned yet
    out = capsys.readouterr().out
    assert "wisard hoppas över" in out
    # the training-only sets (tens of GB) come only when named
    assert "kios_carperson" not in fetched and "seadronessee" not in fetched
    assert "hämtas med sets=: kios_carperson, seadronessee" in out
    fetched.clear()
    assert fetch.main(["yes=1", f"data={tmp_path}", "sets=kios_carperson,seadronessee"]) == 0
    assert fetched == ["kios_carperson", "seadronessee"]


def test_unknown_key_or_set_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch.main(["yes=1", "sets=visdrone"]) == 2
    assert fetch.main(["force=1"]) == 2
    err = capsys.readouterr().err
    assert "Okänd datamängd" in err
    assert "okänd nyckel" in err


# --- the download ------------------------------------------------------------------------


def test_download_checks_sha256_and_keeps_no_bad_file(tmp_path: Path) -> None:
    body = b"x" * 1000
    dest = tmp_path / "a.zip"
    with pytest.raises(BenchError, match="Kontrollsumman"):
        download("https://example.test/a.zip", dest, len(body), "0" * 64, opener=FakeServer(body))
    assert not dest.exists()
    assert (tmp_path / "a.zip.bad").exists()
    assert not (tmp_path / "a.zip.partial").exists()


def test_download_resumes_a_stopped_run(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40
    dest = tmp_path / "a.zip"
    (tmp_path / "a.zip.partial").write_bytes(body[:3000])
    server = FakeServer(body)
    download("https://example.test/a.zip", dest, len(body), _sha(body), opener=server)
    assert dest.read_bytes() == body
    assert server.requests[0] == "bytes=3000-10239"


def test_download_continues_after_a_cut_connection(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40
    server = FakeServer(body)
    server.cut_after = 100
    dest = tmp_path / "a.zip"
    download("https://example.test/a.zip", dest, len(body), _sha(body), opener=server)
    assert dest.read_bytes() == body
    assert server.requests == ["bytes=0-10239", "bytes=100-10239"]


def test_download_asks_for_bounded_ranges(tmp_path: Path) -> None:
    """Google Drive answers an open-ended request for a big file with a quota page."""
    body = bytes(range(256)) * 40
    server = FakeServer(body)
    dest = tmp_path / "a.zip"
    download("https://example.test/a.zip", dest, len(body), _sha(body), opener=server, span=4096)
    assert dest.read_bytes() == body
    assert server.requests == ["bytes=0-4095", "bytes=4096-8191", "bytes=8192-10239"]


def test_download_refuses_a_range_it_did_not_ask_for(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40
    (tmp_path / "a.zip.partial").write_bytes(body[:100])

    def wrong_start(req: urllib.request.Request) -> FakeResponse:
        resp = FakeServer(body)(req)
        resp.headers["Content-Range"] = f"bytes 0-{len(body) - 1}/{len(body)}"
        return resp

    with pytest.raises(BenchError, match="begärda delen"):
        download(
            "https://example.test/a.zip",
            tmp_path / "a.zip",
            len(body),
            _sha(body),
            opener=wrong_start,
        )
    (tmp_path / "a.zip.partial").unlink()

    def no_header(req: urllib.request.Request) -> FakeResponse:
        resp = FakeServer(body)(req)
        del resp.headers["Content-Range"]
        return resp

    with pytest.raises(BenchError, match="begärda delen"):
        download(
            "https://example.test/a.zip",
            tmp_path / "a.zip",
            len(body),
            _sha(body),
            opener=no_header,
        )


def test_download_starts_over_when_ranges_are_ignored(tmp_path: Path) -> None:
    body = b"abcdefgh" * 100
    (tmp_path / "a.zip.partial").write_bytes(b"zzzz")
    dest = tmp_path / "a.zip"
    download(
        "https://example.test/a.zip",
        dest,
        len(body),
        _sha(body),
        opener=FakeServer(body, ranges=False),
    )
    assert dest.read_bytes() == body


def test_download_refuses_a_web_page(tmp_path: Path) -> None:
    page = b"<html>Quota exceeded</html>"
    with pytest.raises(BenchError, match="webbsida"):
        download(
            "https://example.test/a.zip",
            tmp_path / "a.zip",
            1000,
            "0" * 64,
            opener=FakeServer(page, ctype="text/html; charset=utf-8"),
        )


def test_download_refuses_plain_http(tmp_path: Path) -> None:
    with pytest.raises(BenchError, match="https"):
        download(
            "http://example.test/a.zip", tmp_path / "a.zip", 1, "0" * 64, opener=FakeServer(b"x")
        )


def test_range_file_reads_a_member_without_the_whole_archive() -> None:
    body = heridal_zip()
    server = FakeServer(body)
    with zipfile.ZipFile(
        io.BufferedReader(RangeFile("https://example.test/h.zip", server), 1 << 20)
    ) as zf:
        assert zf.read(f"{ROOT_DIR}/Annotations/test_TRS_0001.xml").startswith(b"<annotation>")
    assert all(r and r.startswith("bytes=") for r in server.requests)


class Flaky:
    """Fails the first ``fails`` requests with ``exc`` (raised by a factory), then hands
    them to ``server``."""

    def __init__(self, server: FakeServer, fails: int, exc: Any) -> None:
        self.server = server
        self.fails = fails
        self.exc = exc
        self.calls = 0

    def __call__(self, req: urllib.request.Request) -> FakeResponse:
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc()
        return self.server(req)


def _handshake_timeout() -> Exception:
    return urllib.error.URLError(TimeoutError("_ssl.c:983: The handshake operation timed out"))


def _not_found() -> Exception:
    return urllib.error.HTTPError("https://example.test/h.zip", 404, "Not Found", {}, None)  # type: ignore[arg-type]


def _unavailable() -> Exception:
    return urllib.error.HTTPError("https://example.test/h.zip", 503, "Unavailable", {}, None)  # type: ignore[arg-type]


def _bad_certificate() -> Exception:
    return urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))


@pytest.fixture
def no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "RETRY_WAITS_S", (0, 0, 0))


@pytest.mark.usefixtures("no_waits")
def test_range_file_retries_a_passing_network_fault() -> None:
    body = bytes(range(256)) * 40
    server = FakeServer(body)
    rf = RangeFile("https://example.test/h.zip", server)
    flaky = Flaky(server, 1, _handshake_timeout)
    rf._opener = flaky
    rf.seek(100)
    assert rf.read(50) == body[100:150]
    assert flaky.calls == 2


@pytest.mark.usefixtures("no_waits")
@pytest.mark.parametrize("exc", [_handshake_timeout, ConnectionResetError, TimeoutError])
def test_range_file_gives_up_after_four_tries(exc: Any) -> None:
    flaky = Flaky(FakeServer(b"x" * 10), 99, exc)
    with pytest.raises((urllib.error.URLError, OSError)):
        RangeFile("https://example.test/h.zip", flaky)
    assert flaky.calls == 4


@pytest.mark.usefixtures("no_waits")
def test_range_file_does_not_retry_a_404_or_a_web_page() -> None:
    flaky = Flaky(FakeServer(b"x" * 10), 99, _not_found)
    with pytest.raises(urllib.error.HTTPError):
        RangeFile("https://example.test/h.zip", flaky)
    assert flaky.calls == 1
    page = FakeServer(b"<html></html>", ctype="text/html")
    with pytest.raises(BenchError, match="webbsida"):
        RangeFile("https://example.test/h.zip", page)
    assert len(page.requests) == 1


@pytest.mark.usefixtures("no_waits")
def test_download_retries_a_passing_network_fault(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40
    flaky = Flaky(FakeServer(body), 2, ConnectionResetError)
    dest = tmp_path / "a.zip"
    download("https://example.test/a.zip", dest, len(body), _sha(body), opener=flaky)
    assert dest.read_bytes() == body
    assert flaky.calls == 3


@pytest.mark.usefixtures("no_waits")
def test_download_resumes_a_read_cut_mid_range(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40

    class CutResponse(FakeResponse):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() >= 100:
                raise http.client.IncompleteRead(b"")
            return super().read(min(100 - self.tell(), size if size and size > 0 else 100))

    server = FakeServer(body)
    first = True

    def opener(req: urllib.request.Request) -> FakeResponse:
        nonlocal first
        resp = server(req)
        if first:
            first = False
            return CutResponse(resp.getvalue(), resp.status, resp.headers)
        return resp

    dest = tmp_path / "a.zip"
    download("https://example.test/a.zip", dest, len(body), _sha(body), opener=opener)
    assert dest.read_bytes() == body
    assert server.requests == ["bytes=0-10239", "bytes=100-10239"]


@pytest.mark.usefixtures("no_waits")
def test_download_gives_up_after_four_tries(tmp_path: Path) -> None:
    flaky = Flaky(FakeServer(b"x" * 10), 99, _handshake_timeout)
    with pytest.raises(BenchError, match="Kör samma kommando igen"):
        download("https://example.test/a.zip", tmp_path / "a.zip", 10, "0" * 64, opener=flaky)
    assert flaky.calls == 4


@pytest.mark.usefixtures("no_waits")
def test_a_5xx_is_tried_again_and_a_4xx_is_not(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40
    flaky = Flaky(FakeServer(body), 2, _unavailable)
    dest = tmp_path / "a.zip"
    download("https://example.test/a.zip", dest, len(body), _sha(body), opener=flaky)
    assert dest.read_bytes() == body
    assert flaky.calls == 3
    for code, again in ((500, True), (503, True), (599, True), (400, False), (403, False)):
        err = urllib.error.HTTPError("https://example.test/a.zip", code, "x", {}, None)  # type: ignore[arg-type]
        assert common._transient(err) is again, code


@pytest.mark.usefixtures("no_waits")
def test_a_certificate_that_does_not_verify_is_not_tried_again(tmp_path: Path) -> None:
    flaky = Flaky(FakeServer(b"x" * 10), 99, _bad_certificate)
    with pytest.raises(urllib.error.URLError):
        RangeFile("https://example.test/h.zip", flaky)
    assert flaky.calls == 1
    flaky = Flaky(FakeServer(b"x" * 10), 99, _bad_certificate)
    with pytest.raises(BenchError):
        download("https://example.test/a.zip", tmp_path / "a.zip", 10, "0" * 64, opener=flaky)
    assert flaky.calls == 1


@pytest.mark.usefixtures("no_waits")
def test_download_does_not_retry_a_404(tmp_path: Path) -> None:
    flaky = Flaky(FakeServer(b"x" * 10), 99, _not_found)
    with pytest.raises(BenchError, match="404"):
        download("https://example.test/a.zip", tmp_path / "a.zip", 10, "0" * 64, opener=flaky)
    assert flaky.calls == 1


# --- names -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["../x.jpg", "/etc/passwd", "a/../../x", "a\\..\\b", "", "a/" + "b/" * 9 + "c", ".hidden"],
)
def test_unsafe_member_names_are_refused(name: str) -> None:
    with pytest.raises(BenchError):
        safe_relpath(name)


def test_safe_member_names_pass() -> None:
    assert str(safe_relpath("heridal_keras_retinanet_voc/JPEGImages/test_TRS_0001.jpg")).endswith(
        ".jpg"
    )
    assert (
        safe_relpath("200402_Carnation_Inspire_VIS/DJI_0001_00000029.jpg").name
        == "DJI_0001_00000029.jpg"
    )


# --- a whole fetch -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _synthetic_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthetic archives hold their own pictures, not the real split's."""
    seqs = {("heridal", "TRS"): ["test_TRS_0001"], ("heridal", "VRD"): ["train_VRD_0002"]}
    split = Split(
        "v1",
        dict.fromkeys(seqs, "test"),
        {k: len(v) for k, v in seqs.items()},
        {k: ids_digest(v) for k, v in seqs.items()},
    )
    monkeypatch.setattr(fetch, "load_split", lambda: split)


def _heridal_row(body: bytes) -> Any:
    ds = load_register()["heridal"]
    return dataclasses.replace(
        ds, url="https://example.test/h.zip", size_bytes=len(body), sha256=_sha(body)
    )


def test_fetch_unpacks_only_pictures_and_labels_and_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = heridal_zip()
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=FakeServer(body)))
    ds = _heridal_row(body)
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0
    images = {i.image_id: i for i in read_index(tmp_path, "heridal")}
    assert set(images) == {"heridal:test_TRS_0001", "heridal:train_VRD_0002"}
    one = images["heridal:test_TRS_0001"]
    assert (one.seq, one.width, one.height) == ("TRS", 64, 48)
    assert one.boxes == ((10.0, 10.0, 30.0, 40.0),)
    assert images["heridal:train_VRD_0002"].boxes == ()
    raw = tmp_path / "datasets" / "heridal" / "raw" / ROOT_DIR
    assert not (raw / "ImageSets").exists()
    assert not (tmp_path / "datasets" / "heridal" / "heridal.zip").exists()
    # a second run finds it done and downloads nothing
    monkeypatch.setattr(fetch, "download", lambda *a, **k: pytest.fail("downloaded twice"))
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0


def test_fetch_refuses_an_archive_without_a_pinned_sha(tmp_path: Path) -> None:
    ds = dataclasses.replace(load_register()["heridal"], sha256="PENDING")
    with pytest.raises(BenchError, match="SHA-256"):
        fetch.fetch_archive(ds, tmp_path, keep=False)


def test_fetch_refuses_a_wrong_archive_and_unpacks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = heridal_zip()
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=FakeServer(body)))
    ds = dataclasses.replace(_heridal_row(body), sha256="1" * 64)
    with pytest.raises(BenchError, match="Kontrollsumman"):
        fetch.fetch_archive(ds, tmp_path, keep=False)
    assert not (tmp_path / "datasets" / "heridal" / "raw").exists()
    assert not (tmp_path / "datasets" / "heridal" / "index.json").exists()


def test_label_with_a_dtd_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bomb = b'<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><annotation>&x;</annotation>'
    body = heridal_zip()
    out = io.BytesIO(body)
    with zipfile.ZipFile(out, "a") as zf:
        zf.writestr(f"{ROOT_DIR}/JPEGImages/test_TRS_0009.jpg", _jpeg())
        zf.writestr(f"{ROOT_DIR}/Annotations/test_TRS_0009.xml", bomb)
    body = out.getvalue()
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=FakeServer(body)))
    with pytest.raises(BenchError, match="etikett"):
        fetch.fetch_archive(_heridal_row(body), tmp_path, keep=True)


def test_mini_fetch_checks_each_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = heridal_zip()
    jpg = f"{ROOT_DIR}/JPEGImages/test_TRS_0001.jpg"
    xml = f"{ROOT_DIR}/Annotations/test_TRS_0001.xml"
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        members = {n: zf.read(n) for n in (jpg, xml)}
    mini = tmp_path / "mini.json"

    def write_mini(sha: str) -> None:
        row = {
            "id": "heridal:test_TRS_0001",
            "image": {"member": jpg, "sha256": sha, "bytes": len(members[jpg])},
            "label": {"member": xml, "sha256": _sha(members[xml]), "bytes": len(members[xml])},
        }
        mini.write_text(
            json.dumps({"schema": "bench.mini/v1", "split": "v1", "images": [row]}),
            encoding="utf-8",
        )

    from bench import splits

    monkeypatch.setattr(fetch, "load_mini", functools.partial(splits.load_mini, path=mini))
    server = FakeServer(body)
    monkeypatch.setattr(fetch, "RangeFile", lambda url: RangeFile(url, server))
    write_mini("2" * 64)
    with pytest.raises(BenchError, match="Kontrollsumman"):
        fetch.fetch_mini(load_register(), tmp_path)
    assert not list((tmp_path / "datasets-mini").rglob("*.jpg"))
    write_mini(_sha(members[jpg]))
    infolist = zipfile.ZipFile.infolist

    def inflated(self: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        infos = infolist(self)
        for info in infos:
            info.compress_size = fetch.MAX_MEMBER_BYTES + 1
        return infos

    with monkeypatch.context() as m:
        m.setattr(zipfile.ZipFile, "infolist", inflated)
        with pytest.raises(BenchError, match="fel storlek"):
            fetch.fetch_mini(load_register(), tmp_path)
    assert not list((tmp_path / "datasets-mini").rglob("*.jpg"))
    assert fetch.fetch_mini(load_register(), tmp_path) == 0
    (image,) = read_index(tmp_path, "heridal", mini=True)
    assert image.boxes == ((10.0, 10.0, 30.0, 40.0),)


# --- limits and answers that must be refused ----------------------------------------------


def test_a_huge_member_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "MAX_MEMBER_BYTES", 100)
    body = heridal_zip()
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=FakeServer(body)))
    with pytest.raises(BenchError, match="orimligt stor"):
        fetch.fetch_archive(_heridal_row(body), tmp_path, keep=True)


def test_a_target_through_a_symlink_out_of_raw_is_refused(tmp_path: Path) -> None:
    from bench.common import inside

    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (raw / "JPEGImages").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BenchError, match="utanför"):
        inside(raw, safe_relpath("JPEGImages/a.jpg"))


def test_a_redirect_to_plain_http_is_refused() -> None:
    from bench.common import _HttpsOnlyRedirects, request

    handler = _HttpsOnlyRedirects()
    req = request("https://example.test/a.zip")
    with pytest.raises(BenchError, match="utan https"):
        handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "http://example.test/a")
    again = handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "https://cdn.test/a")
    assert again is not None and again.full_url == "https://cdn.test/a"


def test_range_file_refuses_a_server_that_ignores_ranges() -> None:
    body = bytes(range(256)) * 4
    server = FakeServer(body)
    rf = RangeFile("https://example.test/a.zip", server)
    server.ranges = False  # from now on the whole file, from the start
    rf.seek(100)
    with pytest.raises(BenchError, match="begärda delen"):
        rf.read(10)


def test_range_file_asks_for_at_most_one_span(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import common

    monkeypatch.setattr(common, "RANGE_SPAN", 64)
    body = bytes(range(256)) * 4
    server = FakeServer(body)
    rf = RangeFile("https://example.test/a.zip", server)
    buf = bytearray(500)
    assert rf.readinto(buf) == 64
    assert server.requests[-1] == "bytes=0-63"


def test_the_sim_farm_is_indexed_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import common, datasets

    repo = tmp_path / "repo"
    farm = repo / "build" / "simfarm"
    (farm / "r1" / "images").mkdir(parents=True)
    (farm / "r1" / "labels").mkdir(parents=True)
    PILImage.new("RGB", (40, 20)).save(farm / "r1" / "images" / "a.png")
    (farm / "r1" / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    for mod in (fetch, datasets, common):
        monkeypatch.setattr(mod, "ROOT", repo, raising=False)
    ds = dataclasses.replace(load_register()["simfarm"], local_path="build/simfarm")
    data = tmp_path / "data"
    assert fetch.index_local(ds, data) == 0
    (image,) = read_index(data, "simfarm")
    assert (image.image_id, image.seq, image.boxes) == (
        "simfarm:r1/a",
        "r1",
        ((10.0, 5.0, 30.0, 15.0),),
    )
    assert datasets.image_root(data, "simfarm") == farm.resolve()
    # an index that points outside the repository is not followed
    index = datasets.index_path(data, "simfarm")
    body = json.loads(index.read_text(encoding="utf-8"))
    body["source"] = "local:../../elsewhere"
    index.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(BenchError, match="utanför repot"):
        datasets.image_root(data, "simfarm")


# --- a set published as pictures and labels apart (KIOS) -------------------------------


def _kios_zips(extra_labels: dict[str, bytes] | None = None) -> tuple[bytes, bytes]:
    pictures, labels = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(pictures, "w") as zf:
        zf.writestr("Images/Train/1_people_0007_0.jpg", _jpeg(100, 50))
        zf.writestr("Images/Test/2_ppl_4_00001.jpg", _jpeg(100, 50))
    with zipfile.ZipFile(labels, "w") as zf:
        zf.writestr("Annotations/Yolo/Train/1_people_0007_0.txt", b"0 0.5 0.5 0.1 0.2\n")
        zf.writestr("Annotations/Yolo/Test/2_ppl_4_00001.txt", b"")
        zf.writestr("Annotations/VOC/Train/1_people_0007_0.xml", b"<annotation/>")
        zf.writestr("Annotations/COCO/Train/Train.json", b"{}")
        for name, body in (extra_labels or {}).items():
            zf.writestr(name, body)
    return pictures.getvalue(), labels.getvalue()


class TwoFiles:
    """Serves each archive at its own address."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.servers = {url: FakeServer(body) for url, body in files.items()}

    def __call__(self, req: urllib.request.Request) -> FakeResponse:
        return self.servers[req.full_url](req)


def _kios_row(pictures: bytes, labels: bytes) -> Any:
    return dataclasses.replace(
        load_register()["kios"],
        url="https://example.test/Images.zip",
        size_bytes=len(pictures),
        sha256=_sha(pictures),
        labels_url="https://example.test/Annotations.zip",
        labels_size_bytes=len(labels),
        labels_sha256=_sha(labels),
    )


@pytest.fixture
def kios_split(monkeypatch: pytest.MonkeyPatch) -> None:
    seqs = {("kios", "campus_exercise"): ["1_people_0007_0"], ("kios", "meadow"): ["2_ppl_4_00001"]}
    split = Split(
        "v2",
        {("kios", "campus_exercise"): "train", ("kios", "meadow"): "test"},
        {k: len(v) for k, v in seqs.items()},
        {k: ids_digest(v) for k, v in seqs.items()},
    )
    monkeypatch.setattr(fetch, "load_split", lambda: split)


@pytest.mark.usefixtures("kios_split")
def test_fetch_takes_pictures_and_labels_from_their_two_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pictures, labels = _kios_zips()
    ds = _kios_row(pictures, labels)
    server = TwoFiles({ds.url: pictures, ds.labels_url: labels})
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=server))
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0
    images = {i.image_id: i for i in read_index(tmp_path, "kios")}
    assert set(images) == {"kios:1_people_0007_0", "kios:2_ppl_4_00001"}
    one = images["kios:1_people_0007_0"]
    assert (one.seq, one.width, one.height) == ("campus_exercise", 100, 50)
    assert one.boxes == ((45.0, 20.0, 55.0, 30.0),)
    assert images["kios:2_ppl_4_00001"].boxes == ()
    raw = tmp_path / "datasets" / "kios" / "raw"
    assert not (raw / "Annotations" / "VOC").exists()
    assert not (raw / "Annotations" / "COCO").exists()
    assert not list((tmp_path / "datasets" / "kios").glob("*.zip"))
    # a second run finds it done and downloads nothing
    monkeypatch.setattr(fetch, "download", lambda *a, **k: pytest.fail("downloaded twice"))
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0


@pytest.mark.usefixtures("kios_split")
def test_a_wrong_labels_archive_unpacks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pictures, labels = _kios_zips()
    ds = dataclasses.replace(_kios_row(pictures, labels), labels_sha256="2" * 64)
    server = TwoFiles({ds.url: pictures, ds.labels_url: labels})
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=server))
    with pytest.raises(BenchError, match="Kontrollsumman"):
        fetch.fetch_archive(ds, tmp_path, keep=False)
    assert not (tmp_path / "datasets" / "kios" / "raw").exists()


@pytest.mark.usefixtures("kios_split")
@pytest.mark.parametrize(
    # the same file, and the same file spelt another way: each lands on one path (the
    # last only on a case-insensitive disk, which a data folder under /mnt/c is)
    "name",
    [
        "Images/Train/1_people_0007_0.jpg",
        "./Images/Train/1_people_0007_0.jpg",
        "images/train/1_people_0007_0.JPG",
    ],
)
def test_a_member_in_both_archives_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    pictures, labels = _kios_zips({name: _jpeg(100, 50)})
    ds = _kios_row(pictures, labels)
    server = TwoFiles({ds.url: pictures, ds.labels_url: labels})
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=server))
    with pytest.raises(BenchError, match="två arkiv"):
        fetch.fetch_archive(ds, tmp_path, keep=False)


def test_a_set_is_pinned_only_with_its_labels_archive() -> None:
    pictures, labels = _kios_zips()
    ds = _kios_row(pictures, labels)
    assert fetch.pinned(ds)
    assert not fetch.pinned(dataclasses.replace(ds, labels_sha256="PENDING"))
    assert not fetch.pinned(dataclasses.replace(ds, sha256="PENDING"))


# --- a set in several archives of zips (KIOS car/person) ---------------------------------

_KCP_VIDEOS = {
    "40m-random-walking": "DJI_20250326112630_0006_D",
    "60m-random-walking": "DJI_20250326112218_0001_D",
}


def _zip(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return out.getvalue()


def _kcp_zips() -> tuple[list[bytes], bytes]:
    """One picture archive per flight, each a zip per video; the labels apart."""
    archives = []
    for flight, video in _KCP_VIDEOS.items():
        inner = _zip({f"{video}/frame_{n:06d}.jpg": _jpeg(200, 100) for n in (0, 5, 10)})
        archives.append(_zip({f"{flight}/": b"", f"{flight}/{video}.zip": inner}))
    labels = {"YOLO/obj.names": b"Human\nCar\n"}
    for video in _KCP_VIDEOS.values():
        for n in (0, 5, 10):
            labels[f"YOLO/{video[-4:-2]}D/frame_{n:06d}.txt"] = (
                b"0 0.5 0.5 0.1 0.2\n1 0.1 0.1 0.2 0.2\n"
            )
    return archives, _zip(labels)


def _kcp_row(archives: list[bytes], labels: bytes) -> Any:
    first, *rest = archives
    return dataclasses.replace(
        load_register()["kios_carperson"],
        url="https://example.test/a0.zip",
        size_bytes=len(first),
        sha256=_sha(first),
        more_archives=tuple(
            Archive(f"https://example.test/a{n}.zip", len(body), _sha(body))
            for n, body in enumerate(rest, 1)
        ),
        labels_url="https://example.test/YOLO.zip",
        labels_size_bytes=len(labels),
        labels_sha256=_sha(labels),
    )


def test_fetch_opens_the_zips_inside_every_archive_and_keeps_every_tenth_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archives, labels = _kcp_zips()
    ds = _kcp_row(archives, labels)
    files = {f"https://example.test/a{n}.zip": body for n, body in enumerate(archives)}
    server = TwoFiles({**files, "https://example.test/YOLO.zip": labels})
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=server))
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0
    images = {i.image_id: i for i in read_index(tmp_path, "kios_carperson")}
    assert set(images) == {
        f"kios_carperson:{v}/frame_{n:06d}" for v in _KCP_VIDEOS.values() for n in (0, 10)
    }
    one = images["kios_carperson:DJI_20250326112630_0006_D/frame_000010"]
    assert (one.seq, one.width, one.height) == ("DJI_20250326112630_0006_D", 200, 100)
    assert one.boxes == ((90.0, 40.0, 110.0, 60.0),)  # the person, not the car
    folder = tmp_path / "datasets" / "kios_carperson"
    assert not list(folder.glob("*.zip")) and not (folder / "nested").exists()
    raw = folder / "raw" / "40m-random-walking" / _KCP_VIDEOS["40m-random-walking"]
    assert sorted(p.name for p in raw.rglob("*.jpg")) == ["frame_000000.jpg", "frame_000010.jpg"]


def test_a_set_is_pinned_only_with_all_its_archives() -> None:
    ds = _kcp_row(*_kcp_zips())
    assert fetch.pinned(ds)
    wrong = dataclasses.replace(ds.more_archives[0], sha256="PENDING")
    assert not fetch.pinned(dataclasses.replace(ds, more_archives=(wrong,)))


def test_a_wrong_second_archive_unpacks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archives, labels = _kcp_zips()
    ds = _kcp_row(archives, labels)
    bad = dataclasses.replace(ds.more_archives[0], sha256="3" * 64)
    ds = dataclasses.replace(ds, more_archives=(bad,))
    files = {f"https://example.test/a{n}.zip": body for n, body in enumerate(archives)}
    server = TwoFiles({**files, "https://example.test/YOLO.zip": labels})
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=server))
    with pytest.raises(BenchError, match="Kontrollsumman"):
        fetch.fetch_archive(ds, tmp_path, keep=False)
    assert not (tmp_path / "datasets" / "kios_carperson" / "raw").exists()


# --- a set of loose files on a WebDAV share (SeaDronesSee) -------------------------------

SHARE = "https://example.test/s/TOKEN/Compressed%20Version/"
SHARE_PATH = "/s/TOKEN/Compressed Version/"


def _sds_share() -> dict[str, bytes]:
    def coco(images: list[tuple[int, str, str]], boxes: list[tuple[int, list[int]]]) -> bytes:
        return json.dumps(
            {
                "categories": [{"id": 1, "name": "swimmer"}, {"id": 2, "name": "boat"}],
                "images": [
                    {"id": i, "file_name": f, "source": {"folder_name": v}} for i, f, v in images
                ],
                "annotations": [{"image_id": i, "category_id": 1, "bbox": b} for i, b in boxes],
            }
        ).encode()

    return {
        "annotations/instances_train.json": coco([(1, "1.jpg", "DJI_0032")], [(1, [8, 4, 10, 20])]),
        "annotations/instances_val.json": coco([(2, "2.jpg", "DJI_0063")], []),
        "images/train/1.jpg": _jpeg(64, 48),
        "images/val/2.jpg": _jpeg(64, 48),
    }


class Share:
    """A WebDAV share: PROPFIND lists a folder one level down, GET serves a file."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.servers = {SHARE + urllib.parse.quote(k): FakeServer(v) for k, v in files.items()}

    def propfind(self, url: str) -> list[tuple[str, int | None]]:
        folder = urllib.parse.unquote(url).removeprefix(urllib.parse.unquote(SHARE))
        out: list[tuple[str, int | None]] = [(SHARE_PATH + folder, None)]
        subs = set()
        for rel, body in self.files.items():
            if not rel.startswith(folder):
                continue
            rest = rel[len(folder) :]
            if "/" in rest:
                subs.add(folder + rest.split("/")[0] + "/")
            else:
                out.append((SHARE_PATH + rel, len(body)))
        return out + [(SHARE_PATH + s, None) for s in sorted(subs)]

    def __call__(self, req: urllib.request.Request) -> FakeResponse:
        return self.servers[req.full_url](req)


def _share_digest(files: dict[str, bytes]) -> str:
    lines = "".join(f"{_sha(files[k])}  {k}\n" for k in sorted(files))
    return _sha(lines.encode())


def _sds_row(files: dict[str, bytes], **kw: Any) -> Any:
    row = dataclasses.replace(
        load_register()["seadronessee"],
        url=SHARE,
        share_folders=("annotations", "images/train", "images/val"),
        share_files=len(files),
        size_bytes=sum(len(b) for b in files.values()),
        sha256=_share_digest(files),
    )
    return dataclasses.replace(row, **kw)


def _serve_share(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> Share:
    share = Share(files)
    monkeypatch.setattr(fetch, "propfind", share.propfind)
    monkeypatch.setattr(fetch, "download", functools.partial(download, opener=share))
    return share


def test_fetch_downloads_a_share_file_by_file_and_indexes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _sds_share()
    _serve_share(monkeypatch, {**files, "images/test/3.jpg": _jpeg()})  # test: not fetched
    ds = _sds_row(files)
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0
    images = {i.image_id: i for i in read_index(tmp_path, "seadronessee")}
    assert set(images) == {"seadronessee:DJI_0032/1", "seadronessee:DJI_0063/2"}
    one = images["seadronessee:DJI_0032/1"]
    assert (one.seq, one.file, one.boxes) == ("DJI_0032", "images/train/1.jpg", ((8, 4, 18, 24),))
    assert images["seadronessee:DJI_0063/2"].boxes == ()
    assert not (tmp_path / "datasets" / "seadronessee" / "raw" / "images" / "test").exists()
    # a second run finds it done and downloads nothing
    monkeypatch.setattr(fetch, "propfind", lambda *a, **k: pytest.fail("listed twice"))
    assert fetch.fetch_archive(ds, tmp_path, keep=False) == 0


def test_a_share_with_a_wrong_checksum_is_not_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _sds_share()
    _serve_share(monkeypatch, files)
    with pytest.raises(BenchError, match="Kontrollsumman"):
        fetch.fetch_archive(_sds_row(files, sha256="4" * 64), tmp_path, keep=False)
    assert not (tmp_path / "datasets" / "seadronessee" / "index.json").exists()


def test_a_share_that_changed_is_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _sds_share()
    _serve_share(monkeypatch, {**files, "images/train/9.jpg": _jpeg()})
    with pytest.raises(BenchError, match="har ändrats"):
        fetch.fetch_archive(_sds_row(files), tmp_path, keep=False)
    assert not (tmp_path / "datasets" / "seadronessee" / "raw").exists()


def test_a_share_listing_outside_the_share_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _sds_share()
    monkeypatch.setattr(fetch, "propfind", lambda url: [("/elsewhere/a.jpg", 1)])
    with pytest.raises(BenchError, match="utanför"):
        fetch.share_listing(_sds_row(files))


def test_propfind_reads_names_and_sizes_and_refuses_a_dtd() -> None:
    body = (
        b'<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">'
        b"<d:response><d:href>/s/a%20b/</d:href><d:propstat><d:prop>"
        b"<d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>"
        b"<d:response><d:href>/s/a%20b/1.jpg</d:href><d:propstat><d:prop>"
        b"<d:getcontentlength>12</d:getcontentlength><d:resourcetype/></d:prop></d:propstat>"
        b"</d:response></d:multistatus>"
    )
    server = FakeServer(body, ranges=False, ctype="application/xml")
    assert common.propfind("https://example.test/s/a%20b/", server) == [
        ("/s/a b/", None),
        ("/s/a b/1.jpg", 12),
    ]
    bomb = FakeServer(b'<!DOCTYPE a [<!ENTITY x "y">]><a/>', ranges=False)
    with pytest.raises(BenchError, match="DTD"):
        common.propfind("https://example.test/s/", bomb)
