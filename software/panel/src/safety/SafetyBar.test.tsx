import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { PanelConfig, Session } from "../api/types";
import { useSessionStore } from "../state/sessionStore";
import { SafetyBar } from "./SafetyBar";

const session = (role: Session["role"]): Session => ({
  role,
  auth_mode: "local",
  display_name: "demo",
  expires_at: null,
  issuer_ref: "operator:demo",
});

afterEach(() =>
  useSessionStore
    .getState()
    .set({ status: "loading", session: null, config: null, needsReauth: false }),
);

describe("SafetyBar", () => {
  it("has exactly two buttons: STOPP and FLYG HEM", () => {
    render(<SafetyBar />);
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(2);
    expect(buttons[0]?.textContent).toContain("STOPP");
    expect(buttons[1]?.textContent).toContain("FLYG HEM");
  });

  it("observer may press STOPP but not FLYG HEM", () => {
    act(() =>
      useSessionStore.getState().set({
        status: "ready",
        session: session("observer"),
        config: { default_vehicle_id: "sim-01" } as PanelConfig,
      }),
    );
    render(<SafetyBar />);
    expect((screen.getByTestId("safety-stop") as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByTestId("safety-home") as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByTestId("safety-home").textContent).toContain("Bara piloten");
  });

  it("shows the not-logged-in warning on the login page and keeps both buttons usable", () => {
    act(() => useSessionStore.getState().set({ status: "login" }));
    render(<SafetyBar />);
    expect(screen.getByText(/Inte inloggad/)).toBeTruthy();
    expect((screen.getByTestId("safety-stop") as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByTestId("safety-home") as HTMLButtonElement).disabled).toBe(false);
  });
});
