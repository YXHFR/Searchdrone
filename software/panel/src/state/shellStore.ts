// Where things are on screen. The layout itself is fixed (see app/Shell.tsx); this store only
// holds the operator's choices inside it.
import { create } from "zustand";

/** Desktop: left column, stage, chat. Medium: stage and chat. Phone: stage and bottom sheet. */
export type Viewport = "desktop" | "medium" | "phone";
/** What fills the stage; the other one is the picture-in-picture. */
export type Main = "map" | "video";
/** A full view laid over the stage until it is closed. */
export type Overlay = "manual" | "status";
/** Flyg is the live view; Granska (finds review) comes with the finds flow. */
export type Mode = "fly" | "review";
/** The side column's (or sheet's) tabs. "mission" is the left column when it has no room. */
export type SideTab = "chat" | "events" | "mission";
export type SheetSize = "peek" | "half" | "full";

export function viewportFor(width: number): Viewport {
  if (width >= 1200) return "desktop";
  if (width >= 900) return "medium";
  return "phone";
}

interface ShellState {
  vp: Viewport;
  main: Main;
  overlay: Overlay | null;
  mode: Mode;
  tab: SideTab;
  sheet: SheetSize;
  setViewport(vp: Viewport): void;
  setMain(main: Main): void;
  /** Opening an overlay goes back to Flyg; null closes it. */
  setOverlay(overlay: Overlay | null): void;
  setMode(mode: Mode): void;
  /** Show a side tab, and on a phone raise the sheet so it can be read. */
  showTab(tab: SideTab): void;
  setSheet(size: SheetSize): void;
}

export const useShellStore = create<ShellState>((set, get) => ({
  vp: typeof window === "undefined" ? "desktop" : viewportFor(window.innerWidth),
  main: "map",
  overlay: null,
  mode: "fly",
  tab: "chat",
  sheet: "peek",
  setViewport: (vp) =>
    set((s) => ({ vp, tab: vp !== "desktop" || s.tab !== "mission" ? s.tab : "chat" })),
  setMain: (main) => set({ main }),
  setOverlay: (overlay) => set(overlay ? { overlay, mode: "fly" } : { overlay }),
  setMode: (mode) => set({ mode }),
  showTab: (tab) =>
    set((s) => ({
      tab,
      sheet: get().vp === "phone" && s.sheet === "peek" ? "half" : s.sheet,
    })),
  setSheet: (sheet) => set({ sheet }),
}));
