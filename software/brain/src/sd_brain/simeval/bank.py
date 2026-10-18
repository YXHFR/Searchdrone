"""The example bank: graded exam flights, kept to improve the commander's instructions.

Each kept flight is one gzipped JSON file: the grade, the scenario seed, the cloud model's
answers in order and, per turn, the messages the model saw, the tool calls it made, what it
said and did, and what the gate answered. Only exams are kept (the cloud model decided the
whole flight):

- a flight that passed with at least ``BANK_MIN_TOTAL`` points (``good-<id>``) can become a
  replay fixture that ``just brain-eval`` flies again in CI without a network
  (``sd-brain bank export-golden``, see ``simeval.golden``);
- a flight that failed (``fail-<id>``) shows where the instructions went wrong; the list
  of them is ``sd-brain bank failures``.

Where it goes: a Cloudflare R2 bucket in the EU jurisdiction when ``SD_BANK_R2_ENDPOINT``,
``SD_BANK_R2_BUCKET``, ``SD_BANK_R2_ACCESS_KEY_ID`` and ``SD_BANK_R2_SECRET_ACCESS_KEY`` are
set (the endpoint must be ``https://<account>.eu.r2.cloudflarestorage.com``), otherwise a
local folder. It never goes into the repository: inside a git work tree only a folder under
its gitignored ``data/`` is accepted.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from sd_brain.loop import TurnRecord
from sd_brain.simeval.score import FlightScore

__all__ = [
    "BANK_MIN_TOTAL",
    "BankError",
    "ExampleBank",
    "LocalBank",
    "R2Bank",
    "bank_from_env",
    "bank_name",
    "example_record",
    "worth_keeping",
]

BANK_MIN_TOTAL = 75.0
R2_ENV = (
    "SD_BANK_R2_ENDPOINT",
    "SD_BANK_R2_BUCKET",
    "SD_BANK_R2_ACCESS_KEY_ID",
    "SD_BANK_R2_SECRET_ACCESS_KEY",
)
_EU_ENDPOINT = re.compile(r"^[0-9a-f]{32}\.eu\.r2\.cloudflarestorage\.com$")


class BankError(RuntimeError):
    pass


def worth_keeping(score: FlightScore) -> bool:
    """A good exam (a replay fixture candidate) or a failed one (for the failure list)."""
    return score.exam and (not score.passed or score.total >= BANK_MIN_TOTAL)


def bank_name(score: FlightScore) -> str:
    return f"{'good' if score.passed else 'fail'}-{score.flight_id}"


def example_record(
    score: FlightScore,
    records: list[TurnRecord],
    scenario: Mapping[str, Any],
    replies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One flight: the decisions, what the model saw, the model's answers and the outcome."""
    turns = []
    for r in records:
        out = r.output
        diag = out.diagnostics if out is not None else None
        turns.append(
            {
                "trigger": r.trigger,
                "backend": r.backend,
                "messages": (diag.prompt_json or {}).get("messages") if diag else None,
                "tool_calls": diag.tool_calls if diag else [],
                "texts": r.texts,
                "action": r.ai_action.model_dump(mode="json") if r.ai_action else None,
                "gate": r.result.decision if r.result is not None else None,
                "reason_codes": list(r.result.reason_codes) if r.result is not None else [],
            }
        )
    return {
        "schema": "sd-brain-example/1",
        "score": score.model_dump(mode="json"),
        "scenario": dict(scenario),
        "replies": list(replies or []),
        "turns": turns,
    }


class ExampleBank(Protocol):
    where: str

    def put(self, name: str, record: dict[str, Any]) -> str: ...


def _gz(record: dict[str, Any]) -> bytes:
    raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, mtime=0)


class LocalBank:
    def __init__(self, folder: Path) -> None:
        self.folder = folder.resolve()
        _refuse_tracked(self.folder)
        self.where = str(self.folder)

    def put(self, name: str, record: dict[str, Any]) -> str:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{name}.json.gz"
        path.write_bytes(_gz(record))
        return str(path)


def _refuse_tracked(folder: Path) -> None:
    """Refuse a folder inside a git work tree unless it lies under its ``data/``.

    The repository's ``.gitignore`` ignores ``/data/`` (a test proves it), so a bank there
    never reaches git. Anything else inside a work tree is refused without asking git.
    """
    for top in (folder, *folder.parents):
        if (top / ".git").exists():
            if not folder.is_relative_to(top / "data"):
                raise BankError(
                    f"exempelbanken får inte ligga i repot: {folder} (använd {top / 'data'})"
                )
            return


class R2Bank:
    """A Cloudflare R2 bucket in the EU, written with S3 ``PUT`` (signature version 4)."""

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        *,
        client: httpx.Client | None = None,
        prefix: str = "example-bank",
        now: Any = None,
    ) -> None:
        url = urlsplit(endpoint)
        if url.scheme != "https" or not _EU_ENDPOINT.match(url.hostname or ""):
            raise BankError(
                "R2-banken måste ligga i EU: https://<konto>.eu.r2.cloudflarestorage.com"
            )
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
            raise BankError("ogiltigt namn på R2-hinken")
        self.host = url.hostname or ""
        self.bucket = bucket
        self._ak = access_key_id
        self._sk = secret_access_key
        self._client = client
        self.prefix = prefix
        self._now = now or (lambda: dt.datetime.now(dt.UTC))
        self.where = f"r2://{bucket}/{prefix}/"

    def _sign(self, path: str, payload_sha: str, stamp: dt.datetime) -> dict[str, str]:
        amz_date = stamp.strftime("%Y%m%dT%H%M%SZ")
        day = amz_date[:8]
        scope = f"{day}/auto/s3/aws4_request"
        signed = "host;x-amz-content-sha256;x-amz-date"
        canonical = "\n".join(
            [
                "PUT",
                path,
                "",
                f"host:{self.host}",
                f"x-amz-content-sha256:{payload_sha}",
                f"x-amz-date:{amz_date}",
                "",
                signed,
                payload_sha,
            ]
        )
        to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                hashlib.sha256(canonical.encode()).hexdigest(),
            ]
        )
        key = f"AWS4{self._sk}".encode()
        for part in (day, "auto", "s3", "aws4_request"):
            key = hmac.new(key, part.encode(), hashlib.sha256).digest()
        sig = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
        return {
            "x-amz-content-sha256": payload_sha,
            "x-amz-date": amz_date,
            "authorization": (
                f"AWS4-HMAC-SHA256 Credential={self._ak}/{scope}, "
                f"SignedHeaders={signed}, Signature={sig}"
            ),
        }

    def put(self, name: str, record: dict[str, Any]) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise BankError("ogiltigt namn i exempelbanken")
        body = _gz(record)
        path = f"/{self.bucket}/{self.prefix}/{name}.json.gz"
        headers = self._sign(path, hashlib.sha256(body).hexdigest(), self._now())
        headers["content-type"] = "application/gzip"
        client = self._client or httpx.Client(timeout=30)
        try:
            resp = client.put(f"https://{self.host}{path}", content=body, headers=headers)
        except httpx.HTTPError as exc:
            raise BankError(f"R2 svarar inte: {type(exc).__name__}") from None
        finally:
            if self._client is None:
                client.close()
        if resp.status_code not in (200, 201):
            raise BankError(f"R2 svarade HTTP {resp.status_code}")
        return f"r2://{self.bucket}{path[len(self.bucket) + 1 :]}"


def bank_from_env(env: Mapping[str, str], local: Path) -> ExampleBank:
    """R2 in the EU when all four ``SD_BANK_R2_*`` are set, else the local folder."""
    values = [env.get(k, "").strip() for k in R2_ENV]
    if all(values):
        return R2Bank(*values)
    if any(values):
        raise BankError("sätt alla fyra SD_BANK_R2_* eller ingen")
    return LocalBank(local)
