import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLiveStore } from "../state/liveStore";
import { StageBanner } from "./Shell";

const T0 = Date.UTC(2026, 10, 1, 12, 0, 0);

beforeEach(() => {
  act(() => {
    useLiveStore.getState().reset();
    useLiveStore.getState().setWs("up");
  });
  vi.useFakeTimers();
  vi.setSystemTime(T0);
});
afterEach(() => {
  act(() => {
    useLiveStore.getState().setWs("up");
    useLiveStore.getState().reset();
  });
  vi.useRealTimers();
});

/** The ground station has been unreachable `ms` when the banner is first drawn. */
function downFor(ms: number): void {
  vi.setSystemTime(T0 - ms);
  act(() => useLiveStore.getState().setWs("down"));
  vi.setSystemTime(T0);
  render(<StageBanner />);
}

describe("StageBanner, no ground station", () => {
  it("stays hidden at 1999 ms", () => {
    downFor(1_999);
    expect(screen.queryByTestId("connection-banner")).toBeNull();
  });

  it("shows at 2000 ms", () => {
    downFor(2_000);
    expect(screen.getByTestId("connection-banner")).toBeTruthy();
  });

  it("appears by itself on the next tick once 2 s have passed", () => {
    downFor(1_999);
    act(() => vi.advanceTimersByTime(250));
    expect(screen.getByTestId("connection-banner")).toBeTruthy();
  });
});
