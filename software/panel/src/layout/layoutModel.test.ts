import { describe, expect, it } from "vitest";
import {
  closeView,
  fixSizes,
  fromPreset,
  type LayoutNode,
  type LayoutState,
  leaves,
  MAX_VIEWS,
  maximize,
  normalize,
  PRESET_IDS,
  parseLayout,
  replaceView,
  restore,
  setSizes,
  splitView,
  VIEW_IDS,
  type Viewport,
  viewportFor,
  visibleViews,
} from "./layoutModel";

function depth(n: LayoutNode): number {
  return n.kind === "view" ? 0 : 1 + Math.max(...n.children.map(depth));
}

function checkRules(s: LayoutState, vp: Viewport): void {
  const views = leaves(s.root);
  expect(views.length).toBeGreaterThan(0);
  expect(views.length).toBeLessThanOrEqual(MAX_VIEWS[vp]);
  expect(new Set(views).size).toBe(views.length);
  expect(depth(s.root)).toBeLessThanOrEqual(2);
  const walk = (n: LayoutNode): void => {
    if (n.kind === "view") return;
    expect(n.children.length).toBeGreaterThanOrEqual(2);
    expect(n.sizes.length).toBe(n.children.length);
    expect(n.sizes.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 6);
    for (const z of n.sizes) expect(z).toBeGreaterThanOrEqual(15 - 1e-9);
    n.children.forEach(walk);
  };
  walk(s.root);
}

// Small deterministic PRNG (mulberry32) so the random test is repeatable.
function rng(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("layoutModel", () => {
  it("maps widths to viewports", () => {
    expect(viewportFor(375)).toBe("mobile");
    expect(viewportFor(767)).toBe("mobile");
    expect(viewportFor(768)).toBe("tablet");
    expect(viewportFor(1199)).toBe("tablet");
    expect(viewportFor(1200)).toBe("desktop");
  });

  it("presets give the documented views", () => {
    expect(visibleViews(fromPreset("standard"))).toEqual(["chat", "map", "video"]);
    expect(visibleViews(fromPreset("map_video"))).toEqual(["map", "video"]);
    expect(visibleViews(fromPreset("chat_only"))).toEqual(["chat"]);
    expect(visibleViews(fromPreset("quad"))).toEqual(["chat", "map", "video", "status"]);
    expect(visibleViews(fromPreset("flight"))).toEqual(["map", "video", "status"]);
    for (const id of PRESET_IDS) checkRules(fromPreset(id), "desktop");
  });

  it("tablet keeps the first two views", () => {
    expect(visibleViews(fromPreset("quad", "tablet"))).toEqual(["chat", "map"]);
  });

  it("maximize and restore", () => {
    const s = maximize(fromPreset("standard"), "video");
    expect(visibleViews(s)).toEqual(["video"]);
    expect(visibleViews(restore(s))).toEqual(["chat", "map", "video"]);
    // A view that is not in the layout cannot be maximized.
    expect(maximize(fromPreset("standard"), "status").maximized).toBeNull();
  });

  it("replaceView swaps when the view is already shown", () => {
    const s = replaceView(fromPreset("standard"), [0], "map");
    expect(visibleViews(s)).toEqual(["map", "chat", "video"]);
    const t = replaceView(fromPreset("standard"), [0], "status");
    expect(visibleViews(t)).toEqual(["status", "map", "video"]);
    expect(t.presetId).toBe("custom");
  });

  it("splitView adds a view and respects the limits", () => {
    const s = splitView(fromPreset("chat_only"), [], "row", "map");
    expect(visibleViews(s)).toEqual(["chat", "map"]);
    expect(splitView(s, [0], "col", "map")).toBe(s); // already visible
    const full = splitView(fromPreset("standard"), [0], "col", "status");
    expect(visibleViews(full)).toEqual(["chat", "status", "map", "video"]);
    checkRules(splitView(full, [0, 0], "row", "candidates"), "desktop");
  });

  it("closeView lets the sibling take the space and never empties the layout", () => {
    const s = closeView(fromPreset("map_video"), [1]);
    expect(s.root).toEqual({ kind: "view", view: "map" });
    expect(closeView(s, [])).toBe(s);
  });

  it("setSizes keeps the sum at 100 and each part ≥ 15", () => {
    const s = setSizes(fromPreset("standard"), [], [5, 95]);
    if (s.root.kind !== "split") throw new Error("expected split");
    expect(s.root.sizes[0]).toBeCloseTo(15);
    expect(s.root.sizes[1]).toBeCloseTo(85);
    expect(fixSizes([1, 1, 1, 97], 4).every((x) => x >= 15)).toBe(true);
  });

  it("normalize removes duplicates, too many views and too deep splits", () => {
    const deep: LayoutNode = {
      kind: "split",
      dir: "row",
      sizes: [50, 50],
      children: [
        { kind: "view", view: "chat" },
        {
          kind: "split",
          dir: "col",
          sizes: [50, 50],
          children: [
            { kind: "view", view: "chat" },
            {
              kind: "split",
              dir: "row",
              sizes: [50, 50],
              children: [
                { kind: "view", view: "map" },
                { kind: "view", view: "video" },
              ],
            },
          ],
        },
      ],
    };
    const s = normalize({ version: 1, root: deep, maximized: null, presetId: "custom" }, "desktop");
    checkRules(s, "desktop");
    expect(visibleViews(s)).toEqual(["chat", "map"]);
  });

  it("broken or unknown stored layouts give null", () => {
    expect(parseLayout("{not json", "desktop")).toBeNull();
    expect(parseLayout('{"version":2}', "desktop")).toBeNull();
    expect(
      parseLayout(JSON.stringify({ version: 1, root: { kind: "view", view: "radar" } }), "desktop"),
    ).toBeNull();
    const ok = parseLayout(JSON.stringify(fromPreset("flight")), "desktop");
    expect(ok && visibleViews(ok)).toEqual(["map", "video", "status"]);
  });

  it("keeps rules 1–5 for 1000 random operation sequences", () => {
    const r = rng(42);
    const pick = <T>(xs: readonly T[]): T => xs[Math.floor(r() * xs.length)] as T;
    const vps: Viewport[] = ["mobile", "tablet", "desktop"];
    for (let run = 0; run < 1000; run++) {
      const vp = pick(vps);
      let s = fromPreset(pick(PRESET_IDS), vp);
      for (let step = 0; step < 8; step++) {
        const views = leaves(s.root);
        const paths: number[][] = [];
        const collect = (n: LayoutNode, p: number[]): void => {
          if (n.kind === "view") paths.push(p);
          else for (const [i, c] of n.children.entries()) collect(c, [...p, i]);
        };
        collect(s.root, []);
        const path = pick(paths);
        switch (Math.floor(r() * 6)) {
          case 0:
            s = maximize(s, pick(views));
            break;
          case 1:
            s = restore(s);
            break;
          case 2:
            s = replaceView(s, path, pick(VIEW_IDS), vp);
            break;
          case 3:
            s = splitView(s, path, r() < 0.5 ? "row" : "col", pick(VIEW_IDS), vp);
            break;
          case 4:
            s = closeView(s, path, vp);
            break;
          default:
            s = setSizes(s, path.slice(0, -1), [r() * 100, r() * 100, r() * 100].slice(0, 2), vp);
        }
        checkRules(s, vp);
      }
    }
  });
});
