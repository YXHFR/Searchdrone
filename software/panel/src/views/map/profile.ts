// The search profile card's data: the server's profile as an editable draft, and back.
// The card sends the whole draft when the pilot approves; the ground station numbers it.
import type {
  ColorName,
  Garment,
  Hair,
  Item,
  SearchProfile,
  SearchProfileApproval,
  SearchProfileDraft,
} from "@searchdrone/schemas";

export const CATEGORIES = [
  "dementia",
  "hiker",
  "despondent",
  "gatherer",
  "hunter",
  "child_1_6",
  "child_7_12",
  "child_13_15",
  "autism",
  "other",
] as const satisfies readonly SearchProfileDraft["subject_category"][];

export const PARTS = [
  "outer",
  "upper",
  "lower",
  "headwear",
  "footwear",
  "full_body",
] as const satisfies readonly Garment["part"][];

export const ITEM_KINDS = [
  "backpack",
  "bag",
  "cane",
  "walker",
  "umbrella",
  "bicycle",
  "dog",
  "other",
] as const satisfies readonly Item["kind"][];

export const CERTAINTIES = [
  "certain",
  "likely",
  "unsure",
] as const satisfies readonly Garment["certainty"][];

export const STATES = ["walking", "sitting", "lying", "unknown"] as const;

export const COLORS = [
  "black",
  "white",
  "gray",
  "red",
  "orange",
  "yellow",
  "green",
  "blue",
  "purple",
  "pink",
  "brown",
  "beige",
  "unknown",
] as const satisfies readonly ColorName[];

export const MAX_GARMENTS = 8;
export const MAX_ITEMS = 6;
export const FREE_TEXT_MAX = 2000;
export const VISUAL_NOTES_MAX = 300;

/** A draft as the card edits it: plain arrays instead of the schema's fixed-length tuples. */
export interface EditableDraft {
  subject_category: SearchProfileDraft["subject_category"];
  mobility: NonNullable<SearchProfileDraft["mobility"]>;
  age_band: NonNullable<SearchProfileDraft["age_band"]> | null;
  height_band: NonNullable<SearchProfileDraft["height_band"]> | null;
  garments: Garment[];
  items: Item[];
  hair: Hair | null;
  likely_state: (typeof STATES)[number][];
  last_seen: SearchProfileDraft["last_seen"];
  visual_notes: string;
  free_text: string;
  excluded: string[];
}

export interface ProfileState {
  approved: SearchProfile | null;
  proposal: SearchProfile | null;
  proposal_id: string | null;
  proposed_by: string | null;
}

/** The fields the card edits, from a stored profile (the ids and approval left out). */
export function toEditable(p: SearchProfile | SearchProfileDraft): EditableDraft {
  return {
    subject_category: p.subject_category,
    mobility: p.mobility ?? "unknown",
    age_band: p.age_band ?? null,
    height_band: p.height_band ?? null,
    garments: (p.garments ?? []).map((g) => ({ ...g, colors: [...g.colors] as Garment["colors"] })),
    items: (p.items ?? []).map((i) => ({
      ...i,
      colors: [...(i.colors ?? [])] as NonNullable<Item["colors"]>,
    })),
    hair: p.hair ? { ...p.hair } : null,
    likely_state: [...(p.likely_state ?? ["unknown"])],
    last_seen: { ...p.last_seen },
    visual_notes: p.visual_notes ?? "",
    free_text: p.free_text ?? "",
    excluded: [...(p.excluded ?? [])],
  };
}

/** A new, empty profile: last seen at a map pin, now. */
export function emptyDraft(pin: { lat: number; lon: number }, now: Date): EditableDraft {
  return {
    subject_category: "other",
    mobility: "unknown",
    age_band: null,
    height_band: null,
    garments: [],
    items: [],
    hair: null,
    likely_state: ["unknown"],
    last_seen: { lat: pin.lat, lon: pin.lon, at_utc: now.toISOString(), kind: "point_last_seen" },
    visual_notes: "",
    free_text: "",
    excluded: [],
  };
}

/** What the card showed when the pilot began: the approval names it, and the ground
 * refuses it (409) when a newer version or proposal has arrived since. */
export interface ProfileBase {
  base_version: number;
  proposal_id: string | null;
}

export function baseOf(s: ProfileState): ProfileBase {
  return { base_version: s.approved?.profile_version ?? 0, proposal_id: s.proposal_id };
}

/** The body the pilot's approval sends (``SearchProfileApproval``). */
export function toApproval(d: EditableDraft, base: ProfileBase): SearchProfileApproval {
  return { profile: toBody(d), ...base };
}

/** The profile part of the approval (``SearchProfileDraft``). */
export function toBody(d: EditableDraft): SearchProfileDraft {
  return {
    ...d,
    garments: d.garments.slice(0, MAX_GARMENTS) as NonNullable<SearchProfileDraft["garments"]>,
    items: d.items.slice(0, MAX_ITEMS) as NonNullable<SearchProfileDraft["items"]>,
    likely_state: (d.likely_state.length ? d.likely_state : ["unknown"]).slice(0, 4) as NonNullable<
      SearchProfileDraft["likely_state"]
    >,
    excluded: d.excluded.slice(0, 5) as NonNullable<SearchProfileDraft["excluded"]>,
  };
}

/** What the card shows: a proposal newer than the version in force, else that version. */
export function shown(s: ProfileState | null): {
  profile: SearchProfile | null;
  pending: boolean;
} {
  if (!s) return { profile: null, pending: false };
  if (s.proposal) return { profile: s.proposal, pending: true };
  return { profile: s.approved, pending: false };
}

/** A pin's position from a GeoJSON Point geometry, or null. */
export function pinPoint(geometry: { [k: string]: unknown }): { lat: number; lon: number } | null {
  const c = geometry.coordinates;
  if (geometry.type !== "Point" || !Array.isArray(c)) return null;
  const [lon, lat] = c as unknown[];
  return typeof lat === "number" && typeof lon === "number" ? { lat, lon } : null;
}
