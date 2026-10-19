import pytest

from sd_sim.perf.stats import (
    Ping,
    clock_offset_ns,
    percentile,
    telemetry_result,
    window_means,
)

MS = 1_000_000


def test_percentile_nearest_rank() -> None:
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 5.0  # ceil(4.75) - 1 = 4
    assert percentile(values, 0.2) == 1.0  # ceil(1.0) - 1 = 0
    assert percentile(values, 0.21) == 2.0
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0
    assert percentile(list(range(1, 21)), 0.95) == 19.0  # ceil(19) - 1 = 18
    assert percentile([7.0], 0.95) == 7.0


def test_percentile_empty_is_none() -> None:
    assert percentile([], 0.5) is None


def test_window_means_full_windows() -> None:
    samples = [(100.0 + t, 1.0 if t < 10 else 0.5) for t in range(20)]
    samples.append((120.0, 0.9))  # reaches the end of window 2 but covers none of it
    means = window_means(samples, 10.0)
    assert means == [1.0, 0.5]


def test_window_means_partial_last_window() -> None:
    base = [(float(t), 1.0) for t in range(10)]  # window 0: 0..9
    half = [(10.0 + t, 0.2) for t in range(6)]  # window 1 covers 10..15 = 5 s, half full
    assert window_means(base + half, 10.0) == [1.0, pytest.approx(0.2)]
    short = [(10.0 + t, 0.2) for t in range(4)]  # 10..13 covers 3 s, less than half
    assert window_means(base + short, 10.0) == [1.0]


def test_window_means_unsorted_and_gaps() -> None:
    samples = [(25.0, 3.0), (0.0, 1.0), (5.0, 3.0), (30.0, 5.0)]
    # window 0 = [0,10): mean 2; window 1 empty (skipped); window 2 = [20,30): 3; window 3
    # = [30,40) covers 0 s: dropped.
    assert window_means(samples, 10.0) == [2.0, 3.0]


def test_window_means_empty() -> None:
    assert window_means([], 10.0) == []
    assert window_means([(1.0, 1.0)], 10.0) == []


def pings(page_start_ns: int, n: int, offset_ns: int, rtt_ns: int) -> list[Ping]:
    """Pings with a server clock ``offset_ns`` ahead of the page and symmetric delay."""
    out: list[Ping] = []
    for k in range(n):
        t_send = page_start_ns + k * 50 * MS
        t_recv = t_send + rtt_ns
        server_mono = t_send + rtt_ns // 2 + offset_ns
        out.append(Ping(t_send, t_recv, server_mono + 999, server_mono))
    return out


def test_clock_offset_recovers_known_offset() -> None:
    assert clock_offset_ns(pings(10**12, 10, offset_ns=-3_000 * MS, rtt_ns=4 * MS)) == -3_000 * MS
    assert clock_offset_ns([]) is None


def test_clock_offset_is_median() -> None:
    good = pings(0, 5, offset_ns=7 * MS, rtt_ns=2 * MS)
    # One slow, lopsided reply shifts that ping's offset but not the median.
    bad = Ping(0, 400 * MS, 0, 7 * MS + 10 * MS)
    assert clock_offset_ns([*good, bad]) == 7 * MS


def telemetry_fixture(
    *, offset_ns: int, latency_ms: float, drift_ns: int = 0, n: int = 100, hz: float = 10.0
) -> tuple[list[tuple[int, int]], list[Ping], list[Ping]]:
    """Synthetic run: the server clock is ``offset_ns`` ahead of the page clock (plus
    ``drift_ns`` by the end); each sample arrives ``latency_ms`` after it was stamped."""
    page0 = 5_000_000 * MS
    start = pings(page0, 10, offset_ns, 3 * MS)
    arrivals: list[tuple[int, int]] = []
    period_ns = int(1e9 / hz)
    for k in range(n):
        t_page = page0 + 1_000 * MS + k * period_ns
        server_mono = t_page + offset_ns  # what the server clock read at stamping time
        boot_ms = server_mono // MS
        stamped_page = boot_ms * MS - offset_ns
        arrivals.append((boot_ms, stamped_page + int(latency_ms * MS)))
    end = pings(page0 + 1_000 * MS + n * period_ns + 500 * MS, 10, offset_ns + drift_ns, 3 * MS)
    return arrivals, start, end


def test_telemetry_recovers_latency_and_rate() -> None:
    arrivals, start, end = telemetry_fixture(offset_ns=-2_500_000 * MS, latency_ms=42.0)
    res = telemetry_result(arrivals, start, end)
    assert res.valid
    assert res.samples == 100
    assert res.drift_ms == 0.0
    assert res.p95_ms == pytest.approx(42.0, abs=0.01)
    assert all(lat == pytest.approx(42.0, abs=0.01) for lat in res.latencies_ms)
    # 100 distinct samples over 9.9 s
    assert res.update_hz == pytest.approx(100 / 9.9, rel=1e-6)


def test_telemetry_positive_offset() -> None:
    arrivals, start, end = telemetry_fixture(offset_ns=7_123_456_789, latency_ms=150.0, hz=5.0)
    res = telemetry_result(arrivals, start, end)
    assert res.valid and res.p95_ms == pytest.approx(150.0, abs=0.01)


def test_telemetry_drift_uses_mean_offset_and_invalidates() -> None:
    arrivals, start, end = telemetry_fixture(offset_ns=0, latency_ms=100.0, drift_ns=30 * MS)
    res = telemetry_result(arrivals, start, end)
    assert res.drift_ms == pytest.approx(30.0)
    assert not res.valid
    # offset = mean(0, 30 ms) = 15 ms, so every latency reads 15 ms high
    assert res.p95_ms == pytest.approx(115.0, abs=0.01)


def test_telemetry_keeps_first_arrival_per_sample() -> None:
    arrivals, start, end = telemetry_fixture(offset_ns=0, latency_ms=50.0)
    repeated = arrivals + [(boot, t + 900 * MS) for boot, t in arrivals]
    res = telemetry_result(repeated, start, end)
    assert res.samples == 100
    assert res.p95_ms == pytest.approx(50.0, abs=0.01)


def test_telemetry_needs_enough_data() -> None:
    arrivals, start, end = telemetry_fixture(offset_ns=0, latency_ms=50.0, n=19)
    assert not telemetry_result(arrivals, start, end).valid
    arrivals, start, end = telemetry_fixture(offset_ns=0, latency_ms=50.0)
    assert not telemetry_result(arrivals, start[:9], end).valid
    assert not telemetry_result(arrivals, start, []).valid


def test_telemetry_no_data() -> None:
    res = telemetry_result([], [], [])
    assert not res.valid
    assert res.p95_ms is None and res.update_hz is None and res.drift_ms is None
    assert res.latencies_ms == [] and res.samples == 0
