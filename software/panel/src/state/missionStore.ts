// Server data that belongs to the current mission: charters, approval cards, orders, the
// clearance, sectors, map annotations and the flown track. Filled by the resync (REST) and
// kept current by the live socket; emptied when the mission or the server epoch changes.
import { create } from "zustand";
import type { Annotation, Approval, Charter, Clearance, Command, Find } from "../api/types";

export interface SectorFeature {
  type: "Feature";
  id?: string;
  geometry: { type: string; coordinates: unknown };
  properties: {
    sector_id: string;
    name: string;
    status: string;
    progress: number; // 0..1
    covered?: unknown;
    [k: string]: unknown;
  };
}

/** `clearance.updated` carries a short form; the full Clearance comes from the REST detail. */
export interface ClearanceView {
  status: Clearance["status"] | "none";
  valid_until: string | null;
}

/** [lon, lat] pairs, oldest first. */
export type Track = [number, number][];

const MAX_TRACK = 20_000;

interface MissionState {
  missionId: string | null;
  charters: Record<string, Charter>;
  approvals: Record<string, Approval>;
  commands: Record<string, Command>;
  activeCommandId: string | null;
  clearance: ClearanceView;
  sectors: SectorFeature[];
  annotations: Record<string, Annotation>;
  track: Track;
  /** Found objects (detections joined per object), by find id. Never confirmed finds. */
  finds: Record<string, Find>;
  /** Approval ids whose subject was already requested once (fetch-once rule). */
  subjectFetched: Record<string, true>;
  reset(missionId: string | null): void;
  upsertCharter(c: Charter): void;
  setCharters(list: Charter[]): void;
  upsertApproval(a: Approval): void;
  setPendingApprovals(list: Approval[]): void;
  upsertCommand(c: Command): void;
  setCommands(list: Command[]): void;
  setActiveCommand(id: string | null): void;
  setClearance(c: ClearanceView): void;
  setSectors(list: SectorFeature[]): void;
  upsertAnnotation(a: Annotation): void;
  removeAnnotation(id: string): void;
  setAnnotations(list: Annotation[]): void;
  setTrack(t: Track): void;
  upsertFind(f: Find): void;
  setFinds(list: Find[]): void;
  appendTrack(lon: number, lat: number): void;
  markSubjectFetched(approvalId: string): void;
}

const NO_CLEARANCE: ClearanceView = { status: "none", valid_until: null };

const byId = <T extends { id: string }>(list: T[]): Record<string, T> =>
  Object.fromEntries(list.map((x) => [x.id, x]));

export const useMissionStore = create<MissionState>((set) => ({
  missionId: null,
  charters: {},
  approvals: {},
  commands: {},
  activeCommandId: null,
  clearance: NO_CLEARANCE,
  sectors: [],
  annotations: {},
  track: [],
  finds: {},
  subjectFetched: {},
  reset: (missionId) =>
    set({
      missionId,
      charters: {},
      approvals: {},
      commands: {},
      activeCommandId: null,
      clearance: NO_CLEARANCE,
      sectors: [],
      annotations: {},
      track: [],
      finds: {},
      subjectFetched: {},
    }),
  upsertCharter: (c) => set((s) => ({ charters: { ...s.charters, [c.id]: c } })),
  setCharters: (list) => set({ charters: byId(list) }),
  upsertApproval: (a) => set((s) => ({ approvals: { ...s.approvals, [a.id]: a } })),
  // The REST list has only the pending ones; decided cards already shown stay in the chat.
  setPendingApprovals: (list) =>
    set((s) => {
      const kept = Object.values(s.approvals).filter((a) => a.status !== "pending");
      return { approvals: { ...byId(kept), ...byId(list) } };
    }),
  upsertCommand: (c) =>
    set((s) => ({
      commands: { ...s.commands, [c.id]: c },
      activeCommandId:
        c.status === "running" || c.status === "dispatched" || c.status === "accepted"
          ? c.id
          : s.activeCommandId === c.id
            ? null
            : s.activeCommandId,
    })),
  setCommands: (list) => set((s) => ({ commands: { ...s.commands, ...byId(list) } })),
  setActiveCommand: (id) => set({ activeCommandId: id }),
  setClearance: (clearance) => set({ clearance }),
  setSectors: (sectors) => set({ sectors }),
  upsertAnnotation: (a) => set((s) => ({ annotations: { ...s.annotations, [a.id]: a } })),
  removeAnnotation: (id) =>
    set((s) => {
      const { [id]: _gone, ...rest } = s.annotations;
      return { annotations: rest };
    }),
  setAnnotations: (list) => set({ annotations: byId(list) }),
  setTrack: (track) => set({ track: track.slice(-MAX_TRACK) }),
  upsertFind: (f) => set((s) => ({ finds: { ...s.finds, [f.find_id]: f } })),
  setFinds: (list) => set({ finds: Object.fromEntries(list.map((f) => [f.find_id, f])) }),
  appendTrack: (lon, lat) =>
    set((s) => {
      const last = s.track.at(-1);
      if (last && last[0] === lon && last[1] === lat) return s;
      const track = [...s.track, [lon, lat] as [number, number]];
      return { track: track.length > MAX_TRACK ? track.slice(-MAX_TRACK) : track };
    }),
  markSubjectFetched: (id) => set((s) => ({ subjectFetched: { ...s.subjectFetched, [id]: true } })),
}));

/** The approved charter in force (the mission's active one), if any. */
export function activeCharter(s: Pick<MissionState, "charters">, activeId: string | null) {
  if (activeId && s.charters[activeId]) return s.charters[activeId];
  const approved = Object.values(s.charters).filter((c) => c.status === "approved");
  return approved.sort((a, b) => b.version - a.version)[0] ?? null;
}
