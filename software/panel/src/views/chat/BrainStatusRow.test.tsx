import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Mission, Session, SystemStatus } from "../../api/types";
import { useLiveStore } from "../../state/liveStore";
import { useSessionStore } from "../../state/sessionStore";

const posted: { path: string; text: string }[] = [];
vi.mock("../../api/http", () => ({
  api: async (path: string, opts: { body?: { text?: string; client_msg_id?: string } }) => {
    posted.push({ path, text: opts.body?.text ?? "" });
    return {
      id: "m1",
      seq: 1,
      role: "operator",
      kind: "text",
      text: opts.body?.text ?? "",
      created_at: new Date().toISOString(),
      refs: {},
    };
  },
}));

const { BrainStatusRow } = await import("./ChatView");

const pilot: Session = {
  role: "pilot",
  auth_mode: "local",
  display_name: "demo",
  expires_at: null,
  issuer_ref: "operator:demo",
};

function setUp(live: Partial<ReturnType<typeof useLiveStore.getState>>): void {
  act(() => {
    useSessionStore.getState().set({ status: "ready", session: pilot });
    useLiveStore.getState().patch({ mission: { id: "mission-1" } as Mission, ...live });
  });
}

afterEach(() => {
  posted.length = 0;
  act(() => {
    useLiveStore.getState().reset();
    useSessionStore.getState().set({ status: "loading", session: null });
  });
});

describe("BrainStatusRow", () => {
  it("says the AI does not answer and offers the fixed commands in the command mode", async () => {
    setUp({ brainState: "ready", brainEngine: "command" });
    render(<BrainStatusRow />);
    expect(screen.getByTestId("brain-row").textContent).toContain("AI:n svarar inte");
    const buttons = screen.getByTestId("command-buttons").querySelectorAll("button");
    expect([...buttons].map((b) => b.textContent)).toEqual([
      "Status",
      "Håll position",
      "Fortsätt sökningen",
      "Hjälp",
    ]);
    await act(async () => {
      fireEvent.click(screen.getByText("Håll position"));
    });
    expect(posted).toEqual([{ path: "/api/v1/missions/mission-1/chat", text: "håll position" }]);
  });

  it("shows the buttons when the brain reports degraded", () => {
    setUp({ brainState: "degraded", brainEngine: null });
    render(<BrainStatusRow />);
    expect(screen.getByTestId("command-buttons")).toBeTruthy();
  });

  it("offers no buttons while the cloud model answers", () => {
    setUp({ brainState: "ready", brainEngine: "cloud" });
    render(<BrainStatusRow />);
    expect(screen.getByTestId("brain-row").textContent).toBe("");
    expect(screen.queryByTestId("command-buttons")).toBeNull();
  });

  it("offers no buttons when the AI is down (the command panel says it, once)", () => {
    setUp({ brainState: "down", brainEngine: null });
    render(<BrainStatusRow />);
    expect(screen.getByTestId("brain-row").textContent).toBe("");
    expect(screen.queryByTestId("command-buttons")).toBeNull();
  });

  it("takes the engine from the system status before the first live event", () => {
    setUp({
      brainState: null,
      system: {
        parts: { brain: { state: "ok", since: null } },
        internet: "unknown",
        operator_present: true,
        brain_engine: "command",
        version: "0",
        profile: "sim",
      } satisfies SystemStatus,
    });
    render(<BrainStatusRow />);
    expect(screen.getByTestId("command-buttons")).toBeTruthy();
  });
});
