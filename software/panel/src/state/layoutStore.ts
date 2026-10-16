import { create } from "zustand";
import {
  closeView,
  fromPreset,
  type LayoutState,
  maximize,
  type Path,
  type PresetId,
  parseLayout,
  replaceView,
  restore,
  splitView,
  type ViewId,
  type Viewport,
} from "../layout/layoutModel";

export const LAYOUT_KEYS = {
  mobile: "sd.panel.layout.v1.mobile",
  desktop: "sd.panel.layout.v1.desktop",
} as const;

export type PipView = "map" | "video";
export type PipCorner = "tl" | "tr" | "bl" | "br";

function keyFor(vp: Viewport): string {
  return vp === "mobile" ? LAYOUT_KEYS.mobile : LAYOUT_KEYS.desktop;
}

export function loadLayout(vp: Viewport): LayoutState {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(keyFor(vp));
  } catch {
    raw = null;
  }
  return parseLayout(raw, vp) ?? fromPreset(vp === "mobile" ? "chat_only" : "standard", vp);
}

function save(vp: Viewport, s: LayoutState): void {
  try {
    localStorage.setItem(keyFor(vp), JSON.stringify(s));
  } catch {
    // storage blocked: the layout just is not remembered
  }
}

interface LayoutStore {
  vp: Viewport;
  layout: LayoutState;
  /** Mobile only: small floating map or video over the main view. */
  pip: PipView | null;
  pipCorner: PipCorner;
  setViewport(vp: Viewport): void;
  applyPreset(id: PresetId): void;
  maximize(view: ViewId): void;
  restore(): void;
  replace(path: Path, view: ViewId): void;
  split(path: Path, dir: "row" | "col", view: ViewId): void;
  close(path: Path): void;
  /** Mobile tab bar: show one view. */
  showOnly(view: ViewId): void;
  setPip(pip: PipView | null): void;
  movePip(corner: PipCorner): void;
  /** Tap on the PiP: it becomes the main view and the main view goes into the PiP when possible. */
  swapPip(): void;
}

export const useLayoutStore = create<LayoutStore>((set, get) => {
  const commit = (layout: LayoutState): void => {
    save(get().vp, layout);
    set({ layout });
  };
  const initialVp: Viewport =
    typeof window === "undefined"
      ? "desktop"
      : window.innerWidth < 768
        ? "mobile"
        : window.innerWidth < 1200
          ? "tablet"
          : "desktop";
  return {
    vp: initialVp,
    layout: loadLayout(initialVp),
    pip: null,
    pipCorner: "br",
    setViewport: (vp) => {
      if (vp === get().vp) return;
      set({ vp, layout: loadLayout(vp), pip: vp === "mobile" ? get().pip : null });
    },
    applyPreset: (id) => commit(fromPreset(id, get().vp)),
    maximize: (view) => commit(maximize(get().layout, view)),
    restore: () => commit(restore(get().layout)),
    replace: (path, view) => commit(replaceView(get().layout, path, view, get().vp)),
    split: (path, dir, view) => commit(splitView(get().layout, path, dir, view, get().vp)),
    close: (path) => commit(closeView(get().layout, path, get().vp)),
    showOnly: (view) => {
      const pip = get().pip === view ? null : get().pip;
      set({ pip });
      commit({ version: 1, root: { kind: "view", view }, maximized: null, presetId: "custom" });
    },
    setPip: (pip) => set({ pip }),
    movePip: (pipCorner) => set({ pipCorner }),
    swapPip: () => {
      const { pip, layout } = get();
      if (!pip) return;
      const main = layout.root.kind === "view" ? layout.root.view : null;
      const nextPip: PipView | null = main === "map" || main === "video" ? main : null;
      commit({
        version: 1,
        root: { kind: "view", view: pip },
        maximized: null,
        presetId: "custom",
      });
      set({ pip: nextPip });
    },
  };
});
