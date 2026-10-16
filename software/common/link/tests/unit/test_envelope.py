"""Envelopes: building, the per-kind rules, encoding and decoding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from link_helpers import T0, VEHICLE, clearance, envelope_from, goto, sticks, takeover

from sd_link.codes import MSG_TOO_LARGE, SCHEMA_INVALID
from sd_link.constants import MAX_MESSAGE_BYTES
from sd_link.envelope import (
    GROUND_SENDER,
    GROUND_TO_VEHICLE,
    VEHICLE_TO_GROUND,
    DecodeError,
    MessageTooLargeError,
    build_envelope,
    check_ttl,
    decode,
    encode,
    kind_of,
    may_send,
    peer_sender,
)
from sd_schemas.base import new_id
from sd_schemas.link import KIND_MODELS, MANUAL_CONTROL_TIMEOUT_MS, Ack, Heartbeat, SkillCommand

GOLDEN = Path(__file__).parents[3] / "schemas" / "tests" / "golden" / "sdlink"


def test_direction_tables_cover_the_catalogue() -> None:
    assert set(KIND_MODELS) == VEHICLE_TO_GROUND | GROUND_TO_VEHICLE
    assert {"Ack", "Heartbeat"} == VEHICLE_TO_GROUND & GROUND_TO_VEHICLE
    # Only the ground may give orders, stop, release or steer.
    for kind in (
        "SkillCommand",
        "OperatorOverrideNotice",
        "OperatorRelease",
        "ClearanceConfirmation",
        "ManualTakeover",
        "ManualControl",
        "CharterInstall",
    ):
        assert may_send("ground", kind)
        assert not may_send("vehicle", kind)


def test_senders() -> None:
    assert peer_sender("ground", VEHICLE) == "vehicle:sim-01"
    assert peer_sender("vehicle", VEHICLE) == GROUND_SENDER == "ground:main"


def test_build_sets_kind_and_reliable() -> None:
    cmd = goto()
    env = envelope_from(cmd, side="ground", ttl_ms=15_000)
    assert env.kind == "SkillCommand" and env.reliable and env.v == 1
    assert env.body["command_id"] == str(cmd.command_id)
    assert not envelope_from(Heartbeat()).reliable
    assert not envelope_from(Ack(ref_id=new_id(), stage="received")).reliable
    assert envelope_from(Ack(ref_id=new_id(), stage="done")).reliable
    assert envelope_from(takeover(), side="ground").reliable
    assert not envelope_from(sticks(), side="ground", ttl_ms=1000).reliable


@pytest.mark.parametrize(
    ("kind", "ttl", "ok"),
    [
        ("SkillCommand", None, False),
        ("SkillCommand", 15_000, True),
        ("ClearanceConfirmation", None, False),
        ("ClearanceConfirmation", 5000, True),
        ("ClearanceConfirmation", 5001, False),
        ("ManualControl", None, False),
        ("ManualControl", MANUAL_CONTROL_TIMEOUT_MS, True),
        ("ManualControl", MANUAL_CONTROL_TIMEOUT_MS + 1, False),
        ("Heartbeat", None, True),
        ("Heartbeat", 0, False),
        ("PilotPosition", -5, False),
    ],
)
def test_ttl_rules(kind: str, ttl: int | None, ok: bool) -> None:
    if ok:
        check_ttl(kind, ttl)
    else:
        with pytest.raises(ValueError, match="ttl"):
            check_ttl(kind, ttl)


def test_build_refuses_bad_ttl_and_non_messages() -> None:
    with pytest.raises(ValueError, match="ttl"):
        envelope_from(clearance(), side="ground", ttl_ms=None)
    with pytest.raises(ValueError, match="not an SD-link message"):
        kind_of(T0)  # type: ignore[arg-type]


def test_roundtrip_and_golden_files() -> None:
    env = envelope_from(goto(), side="ground", seq=42, ttl_ms=15_000)
    back, body = decode(encode(env))
    assert back == env and isinstance(body, SkillCommand)
    files = sorted(GOLDEN.glob("*.json"))
    assert files
    for path in files:
        decoded, _ = decode(path.read_text(encoding="utf-8"))
        assert decoded.kind == json.loads(path.read_text(encoding="utf-8"))["kind"]


def test_encode_refuses_too_large() -> None:
    big = goto().model_copy(update={"rationale": "x" * (MAX_MESSAGE_BYTES + 1)})
    env = build_envelope(
        big, sender=GROUND_SENDER, vehicle_id=VEHICLE, seq=1, sent_utc=T0, ttl_ms=1000
    )
    with pytest.raises(MessageTooLargeError):
        encode(env)


def test_decode_too_large_keeps_msg_id() -> None:
    env = envelope_from(goto(), side="ground", ttl_ms=1000)
    raw = json.loads(encode(env))
    raw["body"]["rationale"] = "x" * MAX_MESSAGE_BYTES
    with pytest.raises(DecodeError) as err:
        decode(json.dumps(raw))
    assert err.value.reason == MSG_TOO_LARGE
    assert err.value.msg_id == env.msg_id and err.value.reliable


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(kind="Nope"),
        lambda r: r.update(v=2),
        lambda r: r["body"].update(extra=1),
        lambda r: r["body"].pop("skill"),
        lambda r: r.update(sender="ground:evil"),
        lambda r: r.update(reliable=False),
    ],
)
def test_decode_invalid_is_schema_invalid(mutate: object) -> None:
    env = envelope_from(goto(), side="ground", ttl_ms=1000)
    raw = json.loads(encode(env))
    mutate(raw)  # type: ignore[operator]
    with pytest.raises(DecodeError) as err:
        decode(json.dumps(raw))
    assert err.value.reason == SCHEMA_INVALID
    assert err.value.msg_id == env.msg_id
    assert str(err.value).startswith(SCHEMA_INVALID)


@pytest.mark.parametrize(
    "frame",
    [
        b"\x00",
        "not json",
        "[1]",
        '{"msg_id": "x"}',
        '{"msg_id": "0192f4a1-7c2e-4d10-9a3b-5e1f0c2d8a11"}',
    ],
)
def test_decode_garbage_has_no_msg_id(frame: str | bytes) -> None:
    with pytest.raises(DecodeError) as err:
        decode(frame)
    assert err.value.msg_id is None and not err.value.reliable
