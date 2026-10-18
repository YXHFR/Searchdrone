import { create } from "zustand";
import type { ManualControlState } from "../api/types";

/** After our take-over is accepted, the ground's 2 Hz state may still say "nobody". */
export const TAKEOVER_GRACE_MS = 2_000;

/**
 * Our own take-over, kept outside the manual view so that leaving the view (a phone tab)
 * does not forget it while the ground still holds it.
 */
interface ManualStore {
  takeoverId: string | null;
  acceptedAt: number;
  tookOver(id: string, at: number): void;
  ended(): void;
}

export const useManualStore = create<ManualStore>((set) => ({
  takeoverId: null,
  acceptedAt: 0,
  tookOver: (id, at) => set({ takeoverId: id, acceptedAt: at }),
  ended: () => set({ takeoverId: null, acceptedAt: 0 }),
}));

export interface ManualView {
  /** We steer: the ground holds our take-over (or has not had time to report it yet). */
  steering: boolean;
  /** The ground holds a take-over that is not ours (another panel, or from before a reload). */
  otherActive: boolean;
  /** The ground no longer holds our take-over (hand-back, STOP, FLY HOME, landing). */
  oursEnded: boolean;
}

export function manualView(
  server: ManualControlState | null,
  takeoverId: string | null,
  acceptedAt: number,
  now: number,
): ManualView {
  const active = server?.active === true;
  const oursOnServer = active && takeoverId !== null && server?.takeover_id === takeoverId;
  const inGrace = takeoverId !== null && now - acceptedAt < TAKEOVER_GRACE_MS;
  const steering = oursOnServer || (inGrace && !active);
  return {
    steering,
    otherActive: active && !oursOnServer,
    oursEnded: takeoverId !== null && !steering && !inGrace,
  };
}
