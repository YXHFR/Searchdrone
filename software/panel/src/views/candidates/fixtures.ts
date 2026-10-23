// Test data only (imported by *.test.ts[x]): a card at the public simulator site.
import type { CandidateCard } from "../../api/types";

export function card(over: Partial<CandidateCard> = {}): CandidateCard {
  return {
    candidate_id: `c-${over.seq ?? 1}`,
    seq: 1,
    rev: 1,
    state: "ready",
    priority: 0.8,
    priority_class: "high",
    best_detector_score: 0.7,
    first_seen_utc: "2026-01-01T10:00:00Z",
    last_seen_utc: "2026-01-01T10:01:00Z",
    lat: 59.28842,
    lon: 18.15953,
    n_seen: 2,
    n_in_view: 3,
    sigma95_m: 8,
    thumb_detection_id: "d-1",
    ...over,
  };
}
