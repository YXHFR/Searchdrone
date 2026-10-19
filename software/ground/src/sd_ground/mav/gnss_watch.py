"""The GPS alarms from the ground station's own MAVLink channel.

They work while SD-link is down, so the pilot hears about a weak, lost or spoofed GPS
even when the onboard computer cannot tell. The same ``sd_safety.gnss`` classes as the
onboard spoofing guard (L-SPOOF), but the ground only warns: it never changes the EKF's
sources. Each alarm is an ``alarm.<code>`` event that carries ``code``, ``severity`` and
``message_sv``, which the main loop also puts on the panel's ``alerts`` topic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sd_mav.failsafe_signals import notice_text, text_signal
from sd_mav.vehicle import FcTelemetry
from sd_safety.constants import FS_DR_TIMEOUT_S
from sd_safety.gnss import GnssSample, GpsHealth, SpoofDetector, SpoofFinding

__all__ = ["GnssWatch", "alarm", "text_alarm"]

EventSink = Callable[[str, str, dict[str, Any]], None]


def alarm(
    code: str, severity: str, message_sv: str, **details: Any
) -> tuple[str, str, dict[str, Any]]:
    """An ``alarm.<code>`` event the panel shows as it is."""
    payload = {"code": code, "severity": severity, "message_sv": message_sv, **details}
    return f"alarm.{code}", "critical" if severity == "critical" else "warning", payload


def _spoof_alarm(found: SpoofFinding) -> tuple[str, str, dict[str, Any]]:
    what = (
        f"positionen hoppade {found.value_m:.0f} m"
        if found.kind == "jump"
        else f"höjden skiljer {found.value_m:.0f} m"
    )
    return alarm(
        "GPS_SPOOF",
        "critical",
        f"GPS verkar förfalskad ({what}). Drönaren slutar lita på GPS, flyger mot hem en "
        "kort stund och landar sedan. Ta över med RC.",
        kind=found.kind,
        value_m=found.value_m,
    )


def text_alarm(text: str, utc_iso: str) -> tuple[str, str, dict[str, Any]] | None:
    """The flight controller's GPS texts: "GPS Glitch or Compass error" and
    "Dead Reckoning started" (its timeout is the EKF failsafe's business)."""
    if notice_text(text) == "gps_glitch":
        return alarm(
            "GPS_GLITCH", "warning", "GPS hoppade. Flygkontrollen litar inte på GPS just nu."
        )
    if text_signal(text) == "dead_reckoning" and "started" in text.lower():
        return alarm(
            "DEAD_RECKONING",
            "critical",
            "Drönaren flyger utan GPS sedan 0 s (dödräkning: den gissar sin position utifrån "
            f"fart och vind). Den landar senast om {FS_DR_TIMEOUT_S} s.",
            dr_timeout_s=FS_DR_TIMEOUT_S,
            started_utc=utc_iso,
        )
    return None


class GnssWatch:
    """GPS health and spoofing from ``FcTelemetry``, once per MAVLink tick."""

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink
        self.health = GpsHealth()
        self.spoof = SpoofDetector()

    def reset(self) -> None:
        """After landing: a new flight, a new takeoff height difference."""
        self.health = GpsHealth()
        self.spoof.reset()

    def tick(self, t: FcTelemetry, now_ms: int, *, in_flight: bool) -> None:
        # GPS health only in the air (a weak fix on the ground is the preflight's job); the
        # spoofing guard from arming, for its takeoff height difference.
        status = None
        if in_flight:
            status = self.health.update(
                fix_type=t.gps_fix_type,
                satellites=t.gps_satellites,
                hdop=t.gps_hdop,
                now_ms=now_ms,
            )
        if status == "weak":
            self._sink(
                *alarm(
                    "GPS_WEAK",
                    "warning",
                    f"GPS är svag ({t.gps_satellites} satelliter). Drönaren flyger vidare "
                    "men kan bli osäker på var den är.",
                    satellites=t.gps_satellites,
                    hdop=t.gps_hdop,
                )
            )
        elif status == "lost":
            self._sink(
                *alarm(
                    "GPS_LOST",
                    "critical",
                    "GPS är borta. Drönaren gissar sin position och flyger mot hem i högst "
                    f"{FS_DR_TIMEOUT_S} s, sedan landar den. Ta över med RC om du kan.",
                    dr_timeout_s=FS_DR_TIMEOUT_S,
                )
            )
        elif status == "ok":
            spoofed = self.spoof.finding is not None
            text = (
                "GPS fungerar igen, men drönaren använder den inte förrän du byter tillbaka "
                "med RC-brytaren."
                if spoofed
                else "GPS fungerar igen."
            )
            self._sink(*alarm("GPS_OK", "info", text, after_spoof=spoofed))
        if (
            t.gps_mono_ms is None
            or t.gps_lat is None
            or t.gps_lon is None
            or t.gps_alt_msl_m is None
        ):
            return
        found = self.spoof.update(
            GnssSample(
                mono_ms=t.gps_mono_ms,
                fix_type=t.gps_fix_type,
                satellites=t.gps_satellites,
                hdop=t.gps_hdop,
                gps_lat=t.gps_lat,
                gps_lon=t.gps_lon,
                gps_alt_msl_m=t.gps_alt_msl_m,
                ekf_lat=t.lat,
                ekf_lon=t.lon,
                ekf_alt_msl_m=t.alt_amsl_m,
            ),
            armed=t.armed,
        )
        if found is not None:
            self._sink(*_spoof_alarm(found))
