"""Pure arithmetic for the perf run: percentiles, window means and clock offsets.

Percentiles use the nearest rank, ``sorted[ceil(q * n) - 1]``, the same rule as the
sd-brain eval runner, so the numbers in both reports agree.

Telemetry latency compares two clocks: sd-ground stamps ``Telemetry.time_boot_ms``
with its monotonic clock, the page stamps arrivals with its own clock. Pings measure
the offset between them, ``server_mono_ns - (t_send_ns + t_recv_ns) // 2``, which is
exact when the request and the reply take equally long. A ping series at the start and
one at the end show how far the clocks drifted apart during the measurement.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

# A telemetry measurement only counts when the clocks stayed this close and there is
# enough of it.
MAX_DRIFT_MS = 20.0
MIN_PINGS = 10
MIN_ARRIVALS = 20


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..1). None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered)) - 1
    return float(ordered[min(len(ordered) - 1, max(0, rank))])


def window_means(samples: Sequence[tuple[float, float]], window_s: float = 10.0) -> list[float]:
    """Means of consecutive, non-overlapping windows of ``(mono_s, value)`` samples.

    The first window starts at the first sample. A window is complete when the samples
    reach its end; the last, partial window counts only when the samples cover at least
    half of it. Windows without samples (a gap in the data) are skipped.
    """
    if not samples or window_s <= 0:
        return []
    ordered = sorted(samples)
    start = ordered[0][0]
    last = ordered[-1][0]
    buckets: dict[int, list[float]] = {}
    for t, value in ordered:
        buckets.setdefault(int((t - start) // window_s), []).append(value)
    out: list[float] = []
    for k in sorted(buckets):
        window_end = start + (k + 1) * window_s
        covered = last - (start + k * window_s)
        if last >= window_end or covered >= window_s / 2:
            out.append(statistics.fmean(buckets[k]))
    return out


@dataclass(frozen=True)
class Ping:
    """One ping as the page saw it: send and receive on the page clock, the server's
    wall clock and its monotonic clock (the one ``time_boot_ms`` uses), all in ns."""

    t_send_ns: int
    t_recv_ns: int
    server_ns: int
    server_mono_ns: int


def clock_offset_ns(pings: Sequence[Ping]) -> int | None:
    """Median of ``server_mono_ns - midpoint(page send, page receive)`` (the lower middle
    value for an even count, so it stays an integer). None if there are no pings."""
    if not pings:
        return None
    offsets = [p.server_mono_ns - (p.t_send_ns + p.t_recv_ns) // 2 for p in pings]
    return int(statistics.median_low(offsets))


@dataclass(frozen=True)
class TelemetryResult:
    latencies_ms: list[float]
    p95_ms: float | None
    update_hz: float | None
    drift_ms: float | None
    valid: bool
    samples: int


def telemetry_result(
    arrivals: Sequence[tuple[int, int]],
    pings_start: Sequence[Ping],
    pings_end: Sequence[Ping],
) -> TelemetryResult:
    """Latency and update rate of telemetry from ``(time_boot_ms, t_page_ns)`` arrivals.

    Only the first arrival of each ``time_boot_ms`` counts (a repeated sample is not a
    new update). The server time converts to the page clock with the mean of the start
    and end offsets: ``page_ns = time_boot_ms * 1e6 - offset``. The update rate is the
    number of distinct samples divided by the time between the first and last arrival.
    The result is valid when the clocks drifted at most 20 ms and there are at least 10
    pings on each side and 20 distinct samples.
    """
    first: dict[int, int] = {}
    for boot_ms, t_page in arrivals:
        if boot_ms not in first or t_page < first[boot_ms]:
            first[boot_ms] = t_page
    off_start = clock_offset_ns(pings_start)
    off_end = clock_offset_ns(pings_end)
    known = [o for o in (off_start, off_end) if o is not None]
    offset = sum(known) / len(known) if known else None
    drift_ms = (
        abs(off_end - off_start) / 1e6 if off_start is not None and off_end is not None else None
    )

    latencies: list[float] = []
    if offset is not None:
        for boot_ms, t_page in sorted(first.items()):
            latencies.append((t_page - (boot_ms * 1_000_000 - offset)) / 1e6)

    update_hz: float | None = None
    if len(first) >= 2:
        span_s = (max(first.values()) - min(first.values())) / 1e9
        if span_s > 0:
            update_hz = len(first) / span_s

    valid = (
        drift_ms is not None
        and drift_ms <= MAX_DRIFT_MS
        and len(pings_start) >= MIN_PINGS
        and len(pings_end) >= MIN_PINGS
        and len(first) >= MIN_ARRIVALS
    )
    return TelemetryResult(
        latencies_ms=latencies,
        p95_ms=percentile(latencies, 0.95),
        update_hz=update_hz,
        drift_ms=drift_ms,
        valid=valid,
        samples=len(first),
    )
