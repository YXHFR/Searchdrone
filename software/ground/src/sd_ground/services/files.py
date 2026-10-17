"""The mission's folder when it closes: what happened, as files with their SHA-256.

At ``LANDED -> CLOSED`` and at ``/end`` this writes into ``<data_dir>/missions/<id>/``:

* ``mission.json``: the mission, its limits, every charter version, the orders and the
  approvals;
* ``events.jsonl``: every event of the mission;
* ``exports/track.geojson``: the flown track;
* ``dataflash/<name>.BIN``: the simulator's flight log, copied from ``[sim]
  dataflash_dir`` when that is set;
* ``video/``: the mission's recording from MediaMTX (``services.video``);

and records each file (size, SHA-256) in ``mission_files``. Files are only ever opened
through ``mission_files.rel_path`` under the mission's folder (``resolve()`` and
``is_relative_to``), so ``../`` can never reach anything else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sd_ground.logging import get_logger
from sd_ground.services import video as video_service
from sd_schemas.api import Mission

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["close_files", "resolve_file", "sha256_of"]

log = get_logger("sd_ground.files")
_CHUNK = 1 << 20


def sha256_of(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def resolve_file(root: Path, rel_path: str) -> Path | None:
    """The file ``rel_path`` inside ``root``, or ``None`` when it would leave it."""
    base = root.resolve()
    target = (base / rel_path).resolve()
    return target if target.is_relative_to(base) and target.is_file() else None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _track_geojson(track: list[dict[str, Any]]) -> dict[str, Any]:
    coords = [[float(p["lon"]), float(p["lat"]), float(p["alt_amsl_m"] or 0.0)] for p in track]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "start": None if not track else track[0]["t"].isoformat(),
                    "end": None if not track else track[-1]["t"].isoformat(),
                    "points": len(track),
                },
            }
        ],
    }


async def close_files(flow: MissionFlow, mission: Mission) -> None:
    mid = str(mission.id)
    folder = flow.mission_dir(mid)
    await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    try:
        export = await flow.repo.mission_export(mid)
        text = json.dumps(export, ensure_ascii=False, indent=2, default=str)
        await asyncio.to_thread(_write, folder / "mission.json", text)
        written.append(("export", "mission.json"))
        if flow.g.repo is not None:
            await flow.g.flush_events()
            events = await flow.g.repo.events(mid, None, None, 100_000)
            lines = [e.model_dump_json() for e in events]
            body = "\n".join(lines) + ("\n" if lines else "")
            await asyncio.to_thread(_write, folder / "events.jsonl", body)
            written.append(("log", "events.jsonl"))
        track = await flow.repo.track(mid)
        geo = json.dumps(_track_geojson(track))
        await asyncio.to_thread(_write, folder / "exports" / "track.geojson", geo)
        written.append(("export", "exports/track.geojson"))
        written += await asyncio.to_thread(_copy_dataflash, flow, folder)
    except OSError as exc:
        log.error("files.write_failed", mission_id=mid, error=type(exc).__name__)
        flow.g.add_event(
            "files.write_failed",
            severity="critical",
            mission_id=mid,
            payload={"error": str(exc)[:200]},
        )
    for kind, rel in written:
        path = resolve_file(folder, rel)
        if path is None:
            continue
        size, digest = await asyncio.to_thread(sha256_of, path)
        await flow.repo.add_file(
            mission_id=mid,
            kind=kind,
            rel_path=rel,
            size=size,
            sha256=digest,
            started_at=mission.activated_at,
            ended_at=flow.now(),
            notes=None,
        )
    await video_service.close_mission_video(flow, mission)


def _copy_dataflash(flow: MissionFlow, folder: Path) -> list[tuple[str, str]]:
    source = flow.g.config.sim.dataflash_dir
    if not source:
        return []
    src = flow.g.loaded.path(source)
    logs = sorted(src.glob("*.BIN"), key=lambda p: p.stat().st_mtime) if src.is_dir() else []
    if not logs:
        return []
    latest = logs[-1]
    target = folder / "dataflash" / latest.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(latest, target)
    return [("dataflash", f"dataflash/{latest.name}")]
