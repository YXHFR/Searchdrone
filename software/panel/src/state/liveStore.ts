import { create } from "zustand";
import type { Mission, SystemStatus, Telemetry, VehicleLive } from "../api/types";

export type WsState = "connecting" | "up" | "down";

interface LiveState {
  ws: WsState;
  wsDownSince: number | null;
  lastWsRxAt: number | null;
  telemetry: Telemetry | null;
  lastTelemetryAt: number | null;
  vehicle: VehicleLive | null;
  system: SystemStatus | null;
  mission: Mission | null;
  brainBusy: boolean;
  setWs(ws: WsState): void;
  patch(p: Partial<Omit<LiveState, "setWs" | "patch" | "reset">>): void;
  reset(): void;
}

const empty = {
  telemetry: null,
  lastTelemetryAt: null,
  vehicle: null,
  system: null,
  mission: null,
  brainBusy: false,
};

export const useLiveStore = create<LiveState>((set, get) => ({
  ws: "connecting",
  wsDownSince: null,
  lastWsRxAt: null,
  ...empty,
  setWs: (ws) => {
    const prev = get();
    set({
      ws,
      wsDownSince: ws === "up" ? null : (prev.wsDownSince ?? Date.now()),
    });
  },
  patch: (p) => set(p),
  reset: () => set({ ...empty }),
}));
