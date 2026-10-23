// Pure functions over candidate cards: the tabs, the order, the find numbers and the words
// the queue shows. No React, so they are unit-tested on their own.
import type { CandidateCard, CandidateObservationRow } from "../../api/types";

export type Tab = "review" | "confirmed" | "rejected";

const OPEN = new Set<CandidateCard["state"]>(["new", "assessing", "ready", "inspecting"]);

/** Still waiting for a person's decision. */
export function isOpen(c: CandidateCard): boolean {
  return OPEN.has(c.state);
}

/** More cards than this waiting means the team should get help reviewing. */
export const BACKLOG_AT = 15;

export interface Queue {
  review: CandidateCard[];
  low: CandidateCard[];
  confirmed: CandidateCard[];
  rejected: CandidateCard[];
}

const byPriority = (a: CandidateCard, b: CandidateCard): number =>
  b.priority - a.priority || a.seq - b.seq;

/** Open cards by priority (the low class apart), decided ones newest first. Merged cards
 * are left out: they live on in the candidate they were merged into. */
export function groupQueue(cards: CandidateCard[]): Queue {
  const q: Queue = { review: [], low: [], confirmed: [], rejected: [] };
  for (const c of cards) {
    if (isOpen(c)) (c.priority_class === "low" ? q.low : q.review).push(c);
    else if (c.state === "confirmed") q.confirmed.push(c);
    else if (c.state === "rejected") q.rejected.push(c);
  }
  q.review.sort(byPriority);
  q.low.sort(byPriority);
  q.confirmed.sort((a, b) => a.seq - b.seq);
  q.rejected.sort((a, b) => b.seq - a.seq);
  return q;
}

/** F1, F2, …: confirmed finds in candidate order, so a find keeps its number. */
export function findNumbers(cards: CandidateCard[]): Map<string, number> {
  const confirmed = cards.filter((c) => c.state === "confirmed").sort((a, b) => a.seq - b.seq);
  return new Map(confirmed.map((c, i) => [c.candidate_id, i + 1]));
}

/** The i18n key of a field (garment:0 → garment, with its number). */
export function fieldLabel(key: string): { key: string; n?: number } {
  const m = /^(garment|item):(\d)$/.exec(key);
  if (m) return { key: m[1] as string, n: Number(m[2]) + 1 };
  return { key };
}

/** Pictures of the candidate that can be shown, oldest first. */
export function crops(obs: CandidateObservationRow[]): CandidateObservationRow[] {
  return obs
    .filter((o) => o.has_crop)
    .sort((a, b) => Date.parse(a.captured_utc) - Date.parse(b.captured_utc));
}

/** Minutes since the candidate was first seen, never below zero. */
export function ageMinutes(card: CandidateCard, now: number): number {
  return Math.max(0, Math.floor((now - Date.parse(card.first_seen_utc)) / 60_000));
}
