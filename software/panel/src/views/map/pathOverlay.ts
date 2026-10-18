// Keeps the path layers in step with the stores: drawn paths, the path the drone follows
// (planned dashed, flown solid) and how far along it is.
import { useEffect, useState } from "react";
import type { Command } from "../../api/types";
import { useLiveStore } from "../../state/liveStore";
import { useMissionStore } from "../../state/missionStore";
import {
  type ActivePath,
  activePath,
  finishedPath,
  pathFlownData,
  pathPlanData,
  pathsData,
  resumedPath,
} from "./pathData";

type MapLike = import("maplibre-gl").Map;
type MapMod = typeof import("./createMap");

function current(): ActivePath | null {
  const m = useMissionStore.getState();
  const live = useLiveStore.getState();
  const pending = Object.values(m.approvals)
    .filter((a) => a.kind === "command" && a.status === "pending")
    .map((a) => m.commands[a.subject_id])
    .filter((c): c is Command => c !== undefined);
  // A finished path stays on the map while the drone hovers at its end (HOLD after the
  // step), until the vehicle does something else.
  const snap = live.vehicle?.snapshot;
  const hovering = snap?.executor_state === "HOLD" && snap.hold_reason === "skill_done";
  const latest = hovering ? finishedPath(Object.values(m.commands)) : null;
  const tel = live.telemetry;
  const drone = tel && Number.isFinite(tel.lat) ? { lat: tel.lat, lon: tel.lon } : null;
  const active = m.activeCommandId ? (m.commands[m.activeCommandId] ?? null) : null;
  // After STOP and Resume the resume order flies the rest of the path: keep showing it.
  const resumed = resumedPath(active, Object.values(m.commands));
  return activePath(active, pending, latest, drone, live.vehicle?.snapshot, resumed);
}

/** Pushes the path layers while the map is ready; returns the path being shown. */
export function usePathOverlay(
  ready: boolean,
  handle: { current: { map: MapLike; mod: MapMod } | null },
  visible: boolean,
): ActivePath | null {
  const [shown, setShown] = useState<ActivePath | null>(null);
  useEffect(() => {
    const h = handle.current;
    if (!ready || !h) return;
    const push = (): void => {
      const p = current();
      const tel = useLiveStore.getState().telemetry;
      const drone = tel ? { lat: tel.lat, lon: tel.lon } : null;
      h.mod.setPathData(
        h.map,
        "paths",
        pathsData(Object.values(useMissionStore.getState().annotations)),
      );
      h.mod.setPathData(h.map, "path-plan", pathPlanData(p));
      h.mod.setPathData(h.map, "path-flown", pathFlownData(p, drone));
      setShown(p);
    };
    push();
    const unsubs = [
      useMissionStore.subscribe((s, prev) => {
        if (
          s.annotations !== prev.annotations ||
          s.commands !== prev.commands ||
          s.approvals !== prev.approvals ||
          s.activeCommandId !== prev.activeCommandId
        )
          push();
      }),
      useLiveStore.subscribe((s, prev) => {
        if (s.telemetry !== prev.telemetry || s.vehicle !== prev.vehicle) push();
      }),
    ];
    return () => {
      for (const u of unsubs) u();
    };
  }, [ready, handle]);
  useEffect(() => {
    const h = handle.current;
    if (ready && h) h.mod.setPathsVisible(h.map, visible);
  }, [ready, handle, visible]);
  return shown;
}
