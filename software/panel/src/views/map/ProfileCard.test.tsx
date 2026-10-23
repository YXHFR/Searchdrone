import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SearchProfile, Session } from "../../api/types";
import { useMissionStore } from "../../state/missionStore";
import { useSessionStore } from "../../state/sessionStore";
import {
  baseOf,
  emptyDraft,
  type ProfileState,
  shown,
  toApproval,
  toBody,
  toEditable,
} from "./profile";

const PROPOSAL: SearchProfile = {
  profile_id: "p1",
  profile_version: 1,
  mission_id: "mission-1",
  approved_by: null,
  approved_at: null,
  subject_category: "dementia",
  garments: [{ part: "outer", colors: ["red"], certainty: "certain" }],
  items: [],
  last_seen: { lat: -35.36, lon: 149.16, at_utc: "2026-01-01T10:00:00Z" },
  free_text: "",
};

let state: ProfileState = {
  approved: null,
  proposal: PROPOSAL,
  proposal_id: "x",
  proposed_by: "ai",
};
const posted: { path: string; body: unknown }[] = [];
vi.mock("../../api/http", () => ({
  HttpError: class extends Error {},
  api: async (path: string, opts?: { method?: string; body?: unknown }) => {
    if (opts?.method === "POST") {
      posted.push({ path, body: opts.body });
      return { ...PROPOSAL, approved_by: "operator:demo" };
    }
    return state;
  },
}));

const { ProfileCard } = await import("./ProfileCard");

function session(role: "pilot" | "observer"): Session {
  return {
    role,
    auth_mode: "local",
    display_name: "demo",
    expires_at: null,
    issuer_ref: "operator:demo",
  };
}

afterEach(() => {
  posted.length = 0;
  state = { approved: null, proposal: PROPOSAL, proposal_id: "x", proposed_by: "ai" };
  act(() => useSessionStore.getState().set({ status: "loading", session: null }));
});

describe("profile data", () => {
  it("shows a newer proposal before the version in force", () => {
    const approved = {
      ...PROPOSAL,
      approved_by: "operator:a",
      approved_at: "2026-01-01T10:00:00Z",
    };
    expect(shown({ approved, proposal: null, proposal_id: null, proposed_by: null })).toEqual({
      profile: approved,
      pending: false,
    });
    expect(
      shown({ approved, proposal: PROPOSAL, proposal_id: "x", proposed_by: "ai" }).pending,
    ).toBe(true);
  });

  it("sends only the draft's fields, never the ids or the approval", () => {
    const body = toBody(toEditable(PROPOSAL)) as unknown as Record<string, unknown>;
    expect(body).not.toHaveProperty("profile_id");
    expect(body).not.toHaveProperty("approved_by");
    expect(body).not.toHaveProperty("mission_id");
    expect(body.garments).toEqual(PROPOSAL.garments);
  });

  it("the approval names what the card showed", () => {
    const s: ProfileState = {
      approved: null,
      proposal: PROPOSAL,
      proposal_id: "x",
      proposed_by: "ai",
    };
    expect(toApproval(toEditable(PROPOSAL), baseOf(s))).toMatchObject({
      base_version: 0,
      proposal_id: "x",
    });
  });

  it("a new profile starts empty at the pin", () => {
    const d = emptyDraft({ lat: 1, lon: 2 }, new Date("2026-01-01T00:00:00Z"));
    expect(d.garments).toEqual([]);
    expect(d.last_seen).toMatchObject({ lat: 1, lon: 2, at_utc: "2026-01-01T00:00:00.000Z" });
  });
});

describe("ProfileCard", () => {
  it("lets the pilot correct colour and certainty, then approve", async () => {
    act(() => useSessionStore.getState().set({ status: "ready", session: session("pilot") }));
    render(<ProfileCard missionId="mission-1" />);
    fireEvent.click(await screen.findByTestId("profile-garment-0-color"));
    fireEvent.click(screen.getByTestId("profile-garment-0-color-blue"));
    fireEvent.click(screen.getByTestId("profile-garment-0-certainty"));
    fireEvent.click(screen.getByTestId("profile-garment-0-certainty-unsure"));
    expect(screen.getByTestId("profile-free-warning").textContent).toContain("molnmodellen");
    fireEvent.click(screen.getByTestId("profile-approve"));
    await waitFor(() => expect(posted).toHaveLength(1));
    const body = posted[0]?.body as {
      profile: { garments: { colors: string[]; certainty: string }[] };
      base_version: number;
      proposal_id: string | null;
    };
    expect(body.profile.garments[0]).toMatchObject({ colors: ["blue"], certainty: "unsure" });
    expect(body).toMatchObject({ base_version: 0, proposal_id: "x" });
    expect(posted[0]?.path).toBe("/api/v1/missions/mission-1/search-profile");
  });

  it("an observer sees the card but cannot edit or approve", async () => {
    act(() => useSessionStore.getState().set({ status: "ready", session: session("observer") }));
    render(<ProfileCard missionId="mission-1" />);
    const color = await screen.findByTestId("profile-garment-0-color");
    expect((color as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByTestId("profile-approve")).toBeNull();
    expect(screen.queryByTestId("profile-add-garment")).toBeNull();
  });

  it("keeps the pilot's edits when a new proposal arrives", async () => {
    act(() => useSessionStore.getState().set({ status: "ready", session: session("pilot") }));
    render(<ProfileCard missionId="mission-1" />);
    fireEvent.click(await screen.findByTestId("profile-garment-0-color"));
    fireEvent.click(screen.getByTestId("profile-garment-0-color-blue"));
    state = {
      approved: null,
      proposal: { ...PROPOSAL, garments: [] },
      proposal_id: "y",
      proposed_by: "ai",
    };
    act(() => useMissionStore.getState().bumpProfile());
    await screen.findByTestId("profile-newer");
    expect(screen.getByTestId("profile-garment-0-color").dataset.value).toBe("blue");
    fireEvent.click(screen.getByTestId("profile-show-newer"));
    await waitFor(() => expect(screen.queryByTestId("profile-garment-0")).toBeNull());
  });

  it("fetches again on a search_profile event", async () => {
    act(() => useSessionStore.getState().set({ status: "ready", session: session("observer") }));
    state = { approved: null, proposal: null, proposal_id: null, proposed_by: null };
    render(<ProfileCard missionId="mission-1" />);
    expect((await screen.findByTestId("profile-status")).textContent).toBe("Ingen profil");
    state = { approved: null, proposal: PROPOSAL, proposal_id: "x", proposed_by: "ai" };
    act(() => useMissionStore.getState().bumpProfile());
    await screen.findByTestId("profile-garment-0");
    expect(screen.getByTestId("profile-status").textContent).toBe("Förslag, inte godkänt");
  });
});
