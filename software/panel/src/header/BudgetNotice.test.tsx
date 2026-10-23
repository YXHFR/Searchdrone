import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { SystemStatus } from "../api/types";
import { useLiveStore } from "../state/liveStore";
import { AiStatus, BudgetNotice } from "./TopBar";

afterEach(() => {
  act(() => useLiveStore.getState().reset());
});

describe("BudgetNotice", () => {
  it("shows nothing while the budget is fine", () => {
    act(() => useLiveStore.getState().patch({ cloudBudget: "ok" }));
    render(<BudgetNotice />);
    expect(screen.queryByTestId("budget-notice")).toBeNull();
  });

  it("warns at 80 % from the system status loaded at start", () => {
    act(() =>
      useLiveStore.getState().patch({ system: { cloud_budget: "warning" } as SystemStatus }),
    );
    render(<BudgetNotice />);
    expect(screen.getByTestId("budget-notice").textContent).toBe(
      "Molnbudgeten är till 80 % använd",
    );
  });

  it("says matching is paused and the AI shows command mode when it is spent", () => {
    act(() =>
      useLiveStore.getState().patch({
        brainState: "ready",
        brainEngine: "cloud",
        cloudBudget: "exhausted",
      }),
    );
    render(
      <>
        <BudgetNotice />
        <AiStatus />
      </>,
    );
    expect(screen.getByTestId("budget-notice").textContent).toBe(
      "Molnbudget slut – matchning pausad, ingen AI-chatt",
    );
    expect(screen.getByTestId("pill-ai").textContent).toBe("Kommandoläge");
  });
});
