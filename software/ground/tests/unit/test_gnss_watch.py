"""The ground's GPS alarms from its own MAVLink (``mav/gnss_watch.py``)."""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from sd_ground.mav.gnss_watch import GnssWatch, text_alarm
from sd_ground.mav.thread import vehicle_events
from sd_mav.vehicle import FcTelemetry

# Public CMAC test field (ArduPilot's default SITL home), never a private place.
LAT, LON, ALT = -35.363261, 149.165230, 584.0


def tel(ms: int, **kw: Any) -> FcTelemetry:
    base = FcTelemetry(
        armed=True,
        lat=LAT,
        lon=LON,
        alt_amsl_m=ALT,
        gps_fix_type=3,
        gps_satellites=12,
    )
    gps: dict[str, Any] = {
        "gps_mono_ms": ms,
        "gps_lat": LAT,
        "gps_lon": LON,
        "gps_alt_msl_m": ALT,
        "gps_hdop": 0.8,
    }
    return dataclasses.replace(base, **(gps | kw))


def run(watch: GnssWatch, seconds: float, start_ms: int = 0, **kw: Any) -> int:
    ms = start_ms
    while ms < start_ms + seconds * 1000:
        watch.tick(tel(ms, **kw), ms, in_flight=True)
        ms += 200
    return ms


def test_weak_lost_and_back() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []
    watch = GnssWatch(lambda *e: seen.append(e))
    ms = run(watch, 2)
    ms = run(watch, 4, ms, gps_satellites=4)
    ms = run(watch, 3, ms, gps_fix_type=1)
    run(watch, 6, ms)
    assert [e[0] for e in seen] == ["alarm.GPS_WEAK", "alarm.GPS_LOST", "alarm.GPS_OK"]
    weak, lost, ok = (e[2] for e in seen)
    assert weak["satellites"] == 4 and "4 satelliter" in weak["message_sv"]
    assert lost["severity"] == "critical" and lost["dr_timeout_s"] == 30
    assert ok["after_spoof"] is False and ok["severity"] == "info"


def test_spoofed_height_once() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []
    watch = GnssWatch(lambda *e: seen.append(e))
    ms = run(watch, 12)  # the takeoff height difference
    run(watch, 10, ms, gps_alt_msl_m=ALT + 40)
    spoof = [e for e in seen if e[0] == "alarm.GPS_SPOOF"]
    assert len(spoof) == 1
    _, severity, payload = spoof[0]
    assert severity == "critical" and payload["kind"] == "altitude"
    assert payload["value_m"] >= 25 and "höjden skiljer" in payload["message_sv"]


def test_flight_controller_texts() -> None:
    glitch = text_alarm("GPS Glitch or Compass error", "2026-01-01T12:00:00+00:00")
    assert glitch is not None and glitch[0] == "alarm.GPS_GLITCH"
    dr = text_alarm("Dead Reckoning started", "2026-01-01T12:00:00+00:00")
    assert dr is not None and dr[0] == "alarm.DEAD_RECKONING" and dr[1] == "critical"
    assert dr[2]["dr_timeout_s"] == 30 and dr[2]["started_utc"].startswith("2026-01-01")
    assert text_alarm("Dead Reckoning timeout", "x") is None
    assert text_alarm("Glitch cleared", "x") is None


def test_dead_reckoning_start_comes_from_the_injected_clock() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []
    when = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
    on = vehicle_events(lambda k, s, p: seen.append((k, s, p)), now=lambda: 0.0, utc=lambda: when)
    on("mav.statustext", {"text": "Dead Reckoning started", "severity": 2})
    dr = [p for k, _s, p in seen if k == "alarm.DEAD_RECKONING"]
    assert dr == [dr[0]] and dr[0]["started_utc"] == when.isoformat()
