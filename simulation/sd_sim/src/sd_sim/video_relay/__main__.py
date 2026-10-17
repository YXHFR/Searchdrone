"""``python -m sd_sim.video_relay {source|relay} ...`` (see the package docstring)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
import sys
from pathlib import Path

from sd_sim.video_relay.rtsp import RtspPublisher, stream_url
from sd_sim.video_relay.source import run_source

VEHICLE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
PLACEHOLDERS = {"", "change_me", "change-me", "changeme"}


def read_password(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if value.lower() in PLACEHOLDERS:
        raise SystemExit("Publiceringslösenordet saknas eller är en platshållare.")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sd_sim.video_relay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    src = sub.add_parser("source", help="synthetic camera with the time strip")
    src.add_argument("--port", type=int, required=True, help="UDP port on 127.0.0.1")
    src.add_argument("--seconds", type=float, default=None)
    rel = sub.add_parser("relay", help="RTP from UDP to MediaMTX over RTSP")
    rel.add_argument("--udp-port", type=int, required=True)
    rel.add_argument("--rtsp-port", type=int, default=8554)
    rel.add_argument("--vehicle", default="sim-01")
    rel.add_argument("--password-file", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "source":
        return run_source(args.port, seconds=args.seconds)
    if not VEHICLE_RE.match(args.vehicle):
        raise SystemExit(f"ogiltigt fordons-id: {args.vehicle!r}")
    publisher = RtspPublisher(
        stream_url(args.rtsp_port, args.vehicle),
        "publisher",
        read_password(args.password_file),
        args.udp_port,
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(publisher.run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
