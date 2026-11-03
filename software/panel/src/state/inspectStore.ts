// The closer looks ("Titta närmare") of the current mission: the ground's record of each look
// (GET /missions/{id}/inspections, then inspection.updated) and the drone's progress during
// the open one (candidate.inspect_progress, at most 2 Hz). At most one look is open.
import { create } from "zustand";
import type { Inspection, InspectProgress } from "../api/types";

interface InspectState {
  missionId: string | null;
  looks: Record<string, Inspection>;
  /** The drone's latest progress per look (command id). */
  progress: Record<string, InspectProgress>;
  reset(missionId: string | null): void;
  setLooks(list: Inspection[]): void;
  /** Returns the look as it was before (null: new), so a caller can see a phase change. */
  upsert(look: Inspection): Inspection | null;
  setProgress(p: InspectProgress): void;
}

const ORDER: Record<Inspection["phase"], number> = {
  flying: 0,
  lower_offer: 1,
  await: 2,
  ended: 3,
};

/** An ended look never opens again; a late event about it is dropped. */
function newer(have: Inspection | undefined, next: Inspection): boolean {
  if (!have) return true;
  if (have.phase === "ended") return false;
  // "Gå lägre?" may come and go while the drone flies; the wait and the end never go back.
  return !(ORDER[have.phase] >= ORDER.await && ORDER[next.phase] < ORDER[have.phase]);
}

export const useInspectStore = create<InspectState>((set, get) => ({
  missionId: null,
  looks: {},
  progress: {},
  reset: (missionId) => set({ missionId, looks: {}, progress: {} }),
  setLooks: (list) =>
    set((s) => {
      const looks = { ...s.looks };
      for (const l of list) if (newer(looks[l.command_id], l)) looks[l.command_id] = l;
      return { looks };
    }),
  upsert: (look) => {
    const have = get().looks[look.command_id] ?? null;
    if (newer(have ?? undefined, look))
      set((s) => ({ looks: { ...s.looks, [look.command_id]: look } }));
    return have;
  },
  setProgress: (p) => set((s) => ({ progress: { ...s.progress, [p.command_id]: p } })),
}));

/** The look that is open now (flying, asking to go lower, or waiting for the decision). */
export function openLook(looks: Record<string, Inspection>): Inspection | null {
  return Object.values(looks).find((l) => !l.ended_at && l.phase !== "ended") ?? null;
}

/** The open look on one candidate, if there is one. */
export function useOpenLookOn(candidateId: string): Inspection | null {
  return useInspectStore((s) => {
    const l = openLook(s.looks);
    return l && l.candidate_id === candidateId ? l : null;
  });
}

export function useOpenLook(): Inspection | null {
  return useInspectStore((s) => openLook(s.looks));
}
