// The candidate queue of the current mission: one card per candidate, filled by the resync
// (REST) and kept current by the live socket (candidate.new / updated / assessed / decided).
// Which cards are new since the queue was opened, and which candidate is open for review.
import { create } from "zustand";
import type { CandidateCard } from "../api/types";
import { useShellStore } from "./shellStore";

interface CandidateState {
  missionId: string | null;
  cards: Record<string, CandidateCard>;
  /** Cards that arrived live and have not blinked yet. */
  fresh: Record<string, true>;
  /** The candidate open in the review view. */
  openId: string | null;
  /** Candidates whose picture this session has opened in large size (the server keeps the
   * same record and is the one that checks it). */
  viewed: Record<string, true>;
  reset(missionId: string | null): void;
  setCards(list: CandidateCard[]): void;
  /** A newer revision replaces an older one; an older one never overwrites. */
  upsert(card: CandidateCard, live?: boolean): void;
  seen(id: string): void;
  open(id: string | null): void;
  markViewed(id: string): void;
}

export const useCandidateStore = create<CandidateState>((set) => ({
  missionId: null,
  cards: {},
  fresh: {},
  openId: null,
  viewed: {},
  reset: (missionId) => set({ missionId, cards: {}, fresh: {}, openId: null, viewed: {} }),
  setCards: (list) =>
    set((s) => {
      const cards = { ...s.cards };
      for (const c of list) {
        const have = cards[c.candidate_id];
        if (!have || have.rev <= c.rev) cards[c.candidate_id] = c;
      }
      return { cards };
    }),
  upsert: (card, live = false) =>
    set((s) => {
      const have = s.cards[card.candidate_id];
      if (have && have.rev > card.rev) return s;
      return {
        cards: { ...s.cards, [card.candidate_id]: card },
        fresh: live && !have ? { ...s.fresh, [card.candidate_id]: true } : s.fresh,
      };
    }),
  seen: (id) =>
    set((s) => {
      if (!s.fresh[id]) return s;
      const fresh = { ...s.fresh };
      delete fresh[id];
      return { fresh };
    }),
  open: (openId) => set({ openId }),
  markViewed: (id) => set((s) => ({ viewed: { ...s.viewed, [id]: true } })),
}));

/** Granska fills the stage; on a phone the sheet goes down so the review is not under it. */
export function openReview(): void {
  const shell = useShellStore.getState();
  shell.setMode("review");
  if (shell.vp === "phone") shell.setSheet("peek");
}
