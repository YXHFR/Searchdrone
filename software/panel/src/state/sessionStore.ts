import { create } from "zustand";
import type { PanelConfig, Session } from "../api/types";

export type AuthStatus = "loading" | "login" | "ready" | "emergency";

interface SessionState {
  status: AuthStatus;
  session: Session | null;
  config: PanelConfig | null;
  /** 401 while the shell is running: show the re-login sheet over it. */
  needsReauth: boolean;
  /** Always false in v1 (no remote access). Kept as a stable interface. */
  is_remote: boolean;
  set(patch: Partial<Omit<SessionState, "set">>): void;
}

export const useSessionStore = create<SessionState>((set) => ({
  status: "loading",
  session: null,
  config: null,
  needsReauth: false,
  is_remote: false,
  set: (patch) => set(patch),
}));

export const VEHICLE_KEY = "sd.panel.vehicle";

/** Last known default_vehicle_id (S-12), so STOP can target it even when config is down. */
export function cachedVehicleId(): string | null {
  try {
    return sessionStorage.getItem(VEHICLE_KEY);
  } catch {
    return null;
  }
}

export function rememberVehicleId(id: string): void {
  try {
    sessionStorage.setItem(VEHICLE_KEY, id);
  } catch {
    // storage blocked: STOP falls back to stop_all
  }
}

export function useIsPilot(): boolean {
  return useSessionStore((s) => s.session?.role === "pilot");
}
