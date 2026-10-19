import { beforeEach, describe, expect, it } from "vitest";
import { useShellStore, viewportFor } from "./shellStore";

describe("shell store", () => {
  beforeEach(() =>
    useShellStore.setState({
      vp: "desktop",
      main: "map",
      overlay: null,
      mode: "fly",
      tab: "chat",
      sheet: "peek",
    }),
  );

  it("splits the width into three layouts", () => {
    expect([1920, 1200, 1199, 900, 899, 390].map(viewportFor)).toEqual([
      "desktop",
      "desktop",
      "medium",
      "medium",
      "phone",
      "phone",
    ]);
  });

  it("raises the phone sheet to show a tab, but never lowers it", () => {
    const s = useShellStore.getState();
    s.setViewport("phone");
    s.showTab("events");
    expect(useShellStore.getState()).toMatchObject({ tab: "events", sheet: "half" });
    s.setSheet("full");
    s.showTab("chat");
    expect(useShellStore.getState().sheet).toBe("full");
  });

  it("leaves the sheet alone on a desktop", () => {
    useShellStore.getState().showTab("events");
    expect(useShellStore.getState().sheet).toBe("peek");
  });

  it("goes back to Flyg when a full view opens", () => {
    const s = useShellStore.getState();
    s.setMode("review");
    s.setOverlay("manual");
    expect(useShellStore.getState()).toMatchObject({ mode: "fly", overlay: "manual" });
  });

  it("drops the mission tab when the left column comes back", () => {
    const s = useShellStore.getState();
    s.setViewport("medium");
    s.showTab("mission");
    s.setViewport("desktop");
    expect(useShellStore.getState().tab).toBe("chat");
  });
});
