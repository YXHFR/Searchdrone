import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CandidateDetail, Mission, Session, StoredAssessment } from "../../api/types";
import { useCandidateStore } from "../../state/candidateStore";
import { useLiveStore } from "../../state/liveStore";
import { useSessionStore } from "../../state/sessionStore";
import { HOLD_MS } from "../../ui/HoldButton";
import { card } from "./fixtures";

const posted: { path: string; body: unknown }[] = [];
let assessment: StoredAssessment | null = null;

function detail(c = useCandidateStore.getState().cards["c-1"] ?? card()): CandidateDetail {
  return {
    mission_id: "m-1",
    card: c,
    assessments: assessment ? [assessment] : [],
    decisions:
      c.state === "confirmed"
        ? [
            {
              id: "x",
              decision: "confirm",
              note: "",
              operator_ref: "operator:demo",
              subject_rev: 1,
              created_at: "2026-01-01T10:05:00Z",
            },
          ]
        : [],
    observations: [
      {
        detection_id: "d-1",
        captured_utc: "2026-01-01T10:00:00Z",
        distance_m: 1,
        has_crop: true,
        label: "person",
        lat: 59.28842,
        lon: 18.15953,
        px_size: 40,
        score: 0.7,
        sigma95_m: 8,
      },
    ],
  };
}

vi.mock("../../api/http", () => ({
  HttpError: class extends Error {},
  api: async (path: string, opts?: { method?: string; body?: unknown }) => {
    if (opts?.method === "POST") {
      posted.push({ path, body: opts.body });
      if (path.endsWith("/view")) return { view_id: "v-1", detection_id: "d-1" };
      const confirmed = card({ state: "confirmed", rev: 2 });
      return detail(confirmed);
    }
    return detail();
  },
}));

const { default: CandidatesView } = await import("./CandidatesView");

const session: Session = {
  role: "pilot",
  auth_mode: "local",
  display_name: "demo",
  expires_at: null,
  issuer_ref: "operator:demo",
};

beforeEach(() => {
  useLiveStore.getState().patch({ mission: { id: "m-1" } as Mission });
  useSessionStore.setState({ session });
  useCandidateStore.getState().reset("m-1");
  assessment = null;
});
afterEach(() => {
  posted.length = 0;
  vi.useRealTimers();
});

const view = () => render(<CandidatesView vehicleId="sim-01" isCompact={false} />);

describe("CandidatesView", () => {
  it("lists cards by priority, folds the low ones under a count and has no confirm in the list", () => {
    useCandidateStore
      .getState()
      .setCards([
        card({ seq: 1, priority: 0.5, priority_class: "medium" }),
        card({ seq: 2, priority: 0.9, engine: "fast", person: "yes" }),
        card({ seq: 3, priority: 0.1, priority_class: "low" }),
      ]);
    view();
    const seqs = () =>
      screen.getAllByTestId("candidate-card").map((li) => li.getAttribute("data-seq"));
    expect(seqs()).toEqual(["2", "1"]);
    expect(screen.getByText("Moln snabb")).toBeTruthy();
    expect(screen.getByText("AI: troligen en person")).toBeTruthy();
    expect(screen.queryByTestId("cand-confirm")).toBeNull();
    const low = screen.getByTestId("cand-low-toggle");
    expect(low.textContent).toBe("Låg prioritet (1)");
    fireEvent.click(low);
    expect(seqs()).toEqual(["2", "1", "3"]);
  });

  it("moves with J and K and opens with Enter", () => {
    useCandidateStore.getState().setCards([card({ seq: 1 }), card({ seq: 2, priority: 0.5 })]);
    view();
    const buttons = () => document.querySelectorAll<HTMLButtonElement>(".cand-open");
    buttons()[0]?.focus();
    fireEvent.keyDown(buttons()[0] as HTMLElement, { key: "j" });
    expect(document.activeElement).toBe(buttons()[1]);
    fireEvent.keyDown(buttons()[1] as HTMLElement, { key: "k" });
    expect(document.activeElement).toBe(buttons()[0]);
    fireEvent.click(buttons()[0] as HTMLElement);
    expect(useCandidateStore.getState().openId).toBe("c-1");
  });

  it("keeps the AI folded and Bekräfta grey until a picture is opened large", async () => {
    assessment = {
      id: "a",
      candidate_id: "c-1",
      created_at: "2026-01-01T10:01:00Z",
      engine: "strong",
      explanation_sv: "Röd jacka syns.",
      flags: ["motion_claim"],
      latency_ms: 1,
      model_id: "m",
      overall: 0.7,
      person: "yes",
      person_p: 0.9,
      profile_version: 1,
    };
    useCandidateStore.getState().setCards([card()]);
    useCandidateStore.getState().open("c-1");
    view();
    await screen.findByTestId("cand-larger");
    expect((screen.getByTestId("cand-confirm") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId("cand-ai-toggle") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId("cand-inspect") as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByText("Röd jacka syns.")).toBeNull();

    fireEvent.click(screen.getByTestId("cand-larger"));
    await screen.findByTestId("cand-large");
    expect(posted[0]?.path).toBe("/api/v1/detections/d-1/view");
    fireEvent.click(screen.getByTestId("cand-ai-toggle"));
    expect(screen.getByText("Röd jacka syns.")).toBeTruthy();
    expect(screen.getByTestId("cand-ai-flag").textContent).toContain("AI-påstående, ej verifierat");
    expect((screen.getByTestId("cand-confirm") as HTMLButtonElement).disabled).toBe(false);
  });

  it("offers only Osäker on a candidate the search service has not made ready", async () => {
    useCandidateStore.getState().setCards([card({ state: "new" })]);
    useCandidateStore.getState().open("c-1");
    view();
    fireEvent.click(await screen.findByTestId("cand-larger"));
    await screen.findByTestId("cand-large");
    expect((screen.getByTestId("cand-confirm") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId("cand-reject") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByTestId("cand-unsure") as HTMLButtonElement).disabled).toBe(false);
  });

  it("confirms after the hold and the question, then shows the position three ways", async () => {
    useCandidateStore.getState().setCards([card()]);
    useCandidateStore.getState().markViewed("c-1");
    useCandidateStore.getState().open("c-1");
    view();
    await screen.findByTestId("cand-larger");
    vi.useFakeTimers();
    fireEvent.pointerDown(screen.getByTestId("cand-confirm"), { button: 0 });
    act(() => vi.advanceTimersByTime(HOLD_MS));
    vi.useRealTimers();
    const dialog = screen.getByTestId("cand-confirm-dialog");
    expect(dialog.textContent).toContain("Bekräfta K1 som personen ni söker?");
    expect(dialog.textContent).toContain("Positionen visas för markteamet.");
    fireEvent.click(screen.getByTestId("cand-confirm-yes"));
    await waitFor(() => expect(screen.getByTestId("cand-found")).toBeTruthy());
    expect(posted.at(-1)).toEqual({
      path: "/api/v1/candidates/c-1/decision",
      body: { decision: "confirm", note: "", subject_rev: 1 },
    });
    expect(screen.getByTestId("cand-coord-decimal").textContent).toBe("59.288420, 18.159530");
    expect(screen.getByTestId("cand-coord-dms").textContent).toBe("59°17′18.3″N 18°09′34.3″E");
    expect(screen.getByTestId("cand-coord-sweref").textContent).toBe("N 6576436 E 679958");
    expect(screen.getByTestId("cand-confirmed-by").textContent).toMatch(
      /^Bekräftad av dig \d\d:\d\d$/,
    );
  });
});
