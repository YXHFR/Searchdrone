"""Where the farm's rounds go: the local folder, and with a key Cloudflare R2 in the EU.

Locally every round stays in ``build/simfarm/rounds/<round_id>/`` (gitignored). With an
R2 key in ``config/simfarm.toml`` (gitignored; the repository has only
``config/simfarm.example.toml``) each passed round is also uploaded to the bucket
through R2's EU-jurisdiction endpoint (``<account>.eu.r2.cloudflarestorage.com``), which
only serves buckets created with EU jurisdiction: the data stays in the EU.

The upload survives interruptions and never sends the same thing twice:

* every file goes to ``<prefix>/rounds/<round_id>/<path>`` and carries its SHA-256 as
  ``x-amz-meta-sha256``;
* a file already in the ledger (``build/simfarm/uploaded.jsonl``) is skipped, and one
  that the bucket already has with the same hash (``HEAD``) is recorded and skipped;
* ``manifest.json`` (every file with its hash) goes last: a round with its manifest in
  the bucket is whole.

The requests are signed with AWS Signature Version 4 (standard library only).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
SKIP_DIRS = {"work"}
_ACCOUNT = re.compile(r"^[0-9a-f]{32}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_PREFIX = re.compile(r"^[A-Za-z0-9._/-]{0,200}$")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
TIMEOUT_S = 120
CONTENT_TYPES = {".png": "image/png", ".json": "application/json", ".txt": "text/plain"}


class UploadError(RuntimeError):
    """The bucket refused or could not be reached; the round stays local and is retried."""


@dataclass(frozen=True)
class R2Config:
    account_id: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    prefix: str = "simfarm/v1"

    def __post_init__(self) -> None:
        if not _ACCOUNT.match(self.account_id):
            raise ValueError("r2.account_id ska vara Cloudflare-kontots id (32 hextecken)")
        if not _BUCKET.match(self.bucket):
            raise ValueError("r2.bucket är inget giltigt bucketnamn")
        if not self.access_key_id or not self.secret_access_key:
            raise ValueError("r2.access_key_id och r2.secret_access_key behövs")
        if not _PREFIX.match(self.prefix) or ".." in self.prefix.split("/"):
            raise ValueError("r2.prefix får bara ha bokstäver, siffror och ._/-")

    @property
    def endpoint(self) -> str:
        """The EU-jurisdiction endpoint: the only one the farm ever uses."""
        return f"https://{self.account_id}.eu.r2.cloudflarestorage.com"

    def __repr__(self) -> str:  # never print the secret
        return (
            f"R2Config(account_id={self.account_id!r}, bucket={self.bucket!r}, "
            f"prefix={self.prefix!r})"
        )

    @classmethod
    def from_table(cls, table: Mapping[str, Any]) -> R2Config | None:
        """``[r2]`` from the farm's config; ``None`` when no key is set (local only)."""
        keys = ("account_id", "bucket", "access_key_id", "secret_access_key")
        values = {k: str(table.get(k, "")).strip() for k in keys}
        if not any(values.values()):
            return None
        prefix = str(table.get("prefix", "simfarm/v1")).strip().strip("/")
        return cls(**values, prefix=prefix)


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def sigv4_headers(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload_sha256: str,
    access_key: str,
    secret_key: str,
    now: dt.datetime,
    region: str = "auto",
    service: str = "s3",
    sign_content_sha: bool = True,
) -> dict[str, str]:
    """The request's headers plus ``x-amz-date``, ``x-amz-content-sha256`` and
    ``Authorization`` (AWS Signature Version 4, all headers signed).
    ``sign_content_sha=False`` leaves the content header out (only the AWS
    test suite wants that; S3 and R2 require it)."""
    parts = urllib.parse.urlsplit(url)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    day = amz_date[:8]
    out = {k.lower(): v.strip() for k, v in headers.items()}
    out["host"] = parts.netloc
    out["x-amz-date"] = amz_date
    if sign_content_sha:
        out["x-amz-content-sha256"] = payload_sha256
    signed = sorted(out)
    canonical_headers = "".join(f"{k}:{out[k]}\n" for k in signed)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
        for k, v in sorted(query)
    )
    canonical = "\n".join(
        [
            method,
            parts.path or "/",  # the URL's path is already encoded (``R2Store._url``)
            canonical_query,
            canonical_headers,
            ";".join(signed),
            payload_sha256,
        ]
    )
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = _hmac(
        _hmac(_hmac(_hmac(f"AWS4{secret_key}".encode(), day), region), service), "aws4_request"
    )
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    out["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={';'.join(signed)}, Signature={signature}"
    )
    return out


Send = Callable[[urllib.request.Request], tuple[int, Mapping[str, str]]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """R2 never redirects; following one would send the signed request to another host."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _send(req: urllib.request.Request) -> tuple[int, Mapping[str, str]]:
    try:
        with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise UploadError(f"R2 svarade med omdirigering ({exc.code})") from exc
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}
    except (urllib.error.URLError, OSError) as exc:
        raise UploadError(f"R2 svarar inte: {exc}") from exc


def files_of(round_dir: Path) -> Iterator[Path]:
    """The round's files to upload (not the manifest, the raw work folder or temp files)."""
    for path in sorted(round_dir.rglob("*")):
        rel = path.relative_to(round_dir)
        # Never through a symlink: the worktree links local secrets, which must not upload.
        linked = any(
            round_dir.joinpath(*rel.parts[: i + 1]).is_symlink() for i in range(len(rel.parts))
        )
        if linked or not path.is_file() or rel.parts[0] in SKIP_DIRS or rel.name == MANIFEST:
            continue
        if path.stat().st_nlink > 1:  # a hard link could point at anything on the disk
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path


def read_nofollow(path: Path) -> bytes:
    """The file's bytes; a link swapped in after the listing is refused, never followed."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        return f.read()


class Ledger:
    """What has been uploaded: one JSON line per object, plus one per finished round."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.objects: dict[str, str] = {}
        self.rounds: set[str] = set()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a line cut by a crash: that object is checked again
                if "round" in row:
                    self.rounds.add(str(row["round"]))
                elif "key" in row:
                    self.objects[str(row["key"])] = str(row["sha256"])

    def _append(self, row: Mapping[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def add_object(self, key: str, sha: str) -> None:
        self.objects[key] = sha
        self._append({"key": key, "sha256": sha})

    def add_round(self, round_id: str) -> None:
        self.rounds.add(round_id)
        self._append({"round": round_id})


class R2Store:
    def __init__(
        self,
        cfg: R2Config,
        ledger: Ledger,
        send: Send = _send,
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.send = send
        self.clock = clock
        self.puts = 0

    def key(self, round_id: str, rel: str) -> str:
        prefix = f"{self.cfg.prefix}/" if self.cfg.prefix else ""
        return f"{prefix}rounds/{round_id}/{rel}"

    def _url(self, key: str) -> str:
        return f"{self.cfg.endpoint}/{self.cfg.bucket}/{urllib.parse.quote(key, safe='/-_.~')}"

    def _request(
        self, method: str, key: str, body: bytes | None, headers: Mapping[str, str], sha: str
    ) -> tuple[int, Mapping[str, str]]:
        url = self._url(key)
        signed = sigv4_headers(
            method,
            url,
            headers,
            sha,
            self.cfg.access_key_id,
            self.cfg.secret_access_key,
            self.clock(),
        )
        req = urllib.request.Request(url, data=body, method=method, headers=signed)  # noqa: S310
        return self.send(req)

    def has(self, key: str, sha: str) -> bool:
        status, headers = self._request("HEAD", key, None, {}, EMPTY_SHA256)
        if status == 200:
            return headers.get("x-amz-meta-sha256") == sha
        if status == 404:
            return False
        raise UploadError(f"HEAD {key}: HTTP {status}")

    def put(self, key: str, body: bytes, sha: str, content_type: str) -> None:
        headers = {
            "content-type": content_type,
            "content-length": str(len(body)),
            "x-amz-meta-sha256": sha,
        }
        status, _ = self._request("PUT", key, body, headers, sha)
        if status != 200:
            raise UploadError(f"PUT {key}: HTTP {status}")
        self.puts += 1

    def upload_file(self, key: str, path: Path) -> str:
        data = read_nofollow(path)  # hash and upload the same bytes
        sha = hashlib.sha256(data).hexdigest()
        if self.ledger.objects.get(key) == sha:
            return sha
        if not self.has(key, sha):
            ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            self.put(key, data, sha, ctype)
        self.ledger.add_object(key, sha)
        return sha

    def upload_round(self, round_dir: Path) -> bool:
        """Upload a finished round; ``False`` if it was already whole in the bucket."""
        round_id = round_dir.name
        if round_id in self.ledger.rounds:
            return False
        manifest: list[dict[str, Any]] = []
        for path in files_of(round_dir):
            rel = path.relative_to(round_dir).as_posix()
            sha = self.upload_file(self.key(round_id, rel), path)
            manifest.append({"path": rel, "sha256": sha, "bytes": path.stat().st_size})
        body = (json.dumps({"round_id": round_id, "files": manifest}, indent=1) + "\n").encode()
        sha = hashlib.sha256(body).hexdigest()
        key = self.key(round_id, MANIFEST)
        if not self.has(key, sha):
            self.put(key, body, sha, CONTENT_TYPES[".json"])
        self.ledger.add_round(round_id)
        return True
