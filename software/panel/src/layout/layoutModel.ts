// Pure layout functions (no React). Every change goes through normalize().
export const VIEW_IDS = ["chat", "map", "video", "candidates", "status", "manual"] as const;
export type ViewId = (typeof VIEW_IDS)[number];

export type LayoutNode =
  | { kind: "view"; view: ViewId }
  | { kind: "split"; dir: "row" | "col"; sizes: number[]; children: LayoutNode[] };

export const PRESET_IDS = ["standard", "map_video", "chat_only", "quad", "flight"] as const;
export type PresetId = (typeof PRESET_IDS)[number];

export interface LayoutState {
  version: 1;
  root: LayoutNode;
  maximized: ViewId | null;
  presetId: PresetId | "custom";
}

export type Viewport = "mobile" | "tablet" | "desktop";
/** Path from the root: index of the child at each split. [] is the root. */
export type Path = number[];

export const MAX_VIEWS: Record<Viewport, number> = { mobile: 1, tablet: 2, desktop: 4 };
export const MAX_SPLIT_DEPTH = 2;
export const MIN_SIZE = 15;

export function viewportFor(width: number): Viewport {
  if (width < 768) return "mobile";
  if (width < 1200) return "tablet";
  return "desktop";
}

const v = (view: ViewId): LayoutNode => ({ kind: "view", view });
const split = (dir: "row" | "col", sizes: number[], children: LayoutNode[]): LayoutNode => ({
  kind: "split",
  dir,
  sizes,
  children,
});

export const PRESETS: Record<PresetId, LayoutNode> = {
  standard: split("row", [35, 65], [v("chat"), split("col", [60, 40], [v("map"), v("video")])]),
  map_video: split("row", [60, 40], [v("map"), v("video")]),
  chat_only: v("chat"),
  quad: split(
    "col",
    [50, 50],
    [
      split("row", [50, 50], [v("chat"), v("map")]),
      split("row", [50, 50], [v("video"), v("status")]),
    ],
  ),
  flight: split("row", [70, 30], [v("map"), split("col", [50, 50], [v("video"), v("status")])]),
};

export function fromPreset(id: PresetId, vp: Viewport = "desktop"): LayoutState {
  return normalize(
    { version: 1, root: structuredClone(PRESETS[id]), maximized: null, presetId: id },
    vp,
  );
}

export function isViewId(x: unknown): x is ViewId {
  return typeof x === "string" && (VIEW_IDS as readonly string[]).includes(x);
}

// --- queries -----------------------------------------------------------------------

export function visibleViews(s: LayoutState): ViewId[] {
  if (s.maximized) return [s.maximized];
  return leaves(s.root);
}

export function leaves(n: LayoutNode): ViewId[] {
  return n.kind === "view" ? [n.view] : n.children.flatMap(leaves);
}

export function nodeAt(root: LayoutNode, path: Path): LayoutNode | null {
  let n: LayoutNode = root;
  for (const i of path) {
    if (n.kind !== "split") return null;
    const c = n.children[i];
    if (!c) return null;
    n = c;
  }
  return n;
}

function replaceAt(
  root: LayoutNode,
  path: Path,
  fn: (n: LayoutNode) => LayoutNode | null,
): LayoutNode | null {
  if (path.length === 0) return fn(root);
  if (root.kind !== "split") return root;
  const [i, ...rest] = path as [number, ...number[]];
  const child = root.children[i];
  if (!child) return root;
  const next = replaceAt(child, rest, fn);
  if (next === null) {
    const children = root.children.filter((_, j) => j !== i);
    const sizes = root.sizes.filter((_, j) => j !== i);
    return { ...root, children, sizes };
  }
  const children = root.children.map((c, j) => (j === i ? next : c));
  return { ...root, children };
}

// --- operations --------------------------------------------------------------------

export function maximize(s: LayoutState, view: ViewId): LayoutState {
  return visibleViews({ ...s, maximized: null }).includes(view) ? { ...s, maximized: view } : s;
}

export function restore(s: LayoutState): LayoutState {
  return { ...s, maximized: null };
}

/** Show `view` in the pane at `path`. If it is already shown elsewhere, the two swap. */
export function replaceView(
  s: LayoutState,
  path: Path,
  view: ViewId,
  vp: Viewport = "desktop",
): LayoutState {
  const target = nodeAt(s.root, path);
  if (target?.kind !== "view" || target.view === view) return s;
  const old = target.view;
  const swap = (n: LayoutNode): LayoutNode =>
    n.kind === "view"
      ? n.view === view
        ? { kind: "view", view: old }
        : n
      : { ...n, children: n.children.map(swap) };
  const swapped = swap(s.root);
  const root = replaceAt(swapped, path, () => ({ kind: "view", view })) ?? swapped;
  return normalize({ ...s, root, maximized: null, presetId: "custom" }, vp);
}

export function splitView(
  s: LayoutState,
  path: Path,
  dir: "row" | "col",
  newView: ViewId,
  vp: Viewport = "desktop",
): LayoutState {
  if (visibleViews(restore(s)).includes(newView)) return s;
  const root =
    replaceAt(s.root, path, (n) => ({
      kind: "split",
      dir,
      sizes: [50, 50],
      children: [n, v(newView)],
    })) ?? s.root;
  return normalize({ ...s, root, maximized: null, presetId: "custom" }, vp);
}

export function closeView(s: LayoutState, path: Path, vp: Viewport = "desktop"): LayoutState {
  if (leaves(s.root).length <= 1) return s;
  const root = replaceAt(s.root, path, () => null);
  if (!root) return s;
  return normalize({ ...s, root, maximized: null, presetId: "custom" }, vp);
}

export function setSizes(
  s: LayoutState,
  path: Path,
  sizes: number[],
  vp: Viewport = "desktop",
): LayoutState {
  const root =
    replaceAt(s.root, path, (n) => (n.kind === "split" ? { ...n, sizes: [...sizes] } : n)) ??
    s.root;
  return normalize({ ...s, root }, vp);
}

// --- normalize ---------------------------------------------------------------------

/** Sizes sum to 100 and each part is at least MIN_SIZE. */
export function fixSizes(sizes: number[], count: number): number[] {
  if (count <= 0) return [];
  let s =
    sizes.length === count && sizes.every((x) => Number.isFinite(x) && x > 0)
      ? [...sizes]
      : Array.from({ length: count }, () => 1);
  const total = s.reduce((a, b) => a + b, 0);
  s = s.map((x) => (x / total) * 100);
  for (let round = 0; round < count; round++) {
    const small = s.map((x) => x < MIN_SIZE);
    if (!small.some(Boolean)) break;
    const fixed = small.filter(Boolean).length * MIN_SIZE;
    const freeTotal = s.reduce((a, x, i) => (small[i] ? a : a + x), 0);
    s = s.map((x, i) => (small[i] ? MIN_SIZE : (x / freeTotal) * (100 - fixed)));
  }
  return s;
}

function prune(n: LayoutNode, depth: number, seen: Set<ViewId>, budget: number): LayoutNode | null {
  if (n.kind === "view") {
    if (!isViewId(n.view) || seen.has(n.view) || seen.size >= budget) return null;
    seen.add(n.view);
    return { kind: "view", view: n.view };
  }
  if (depth >= MAX_SPLIT_DEPTH) {
    // Too deep: keep the first usable view of this subtree.
    for (const view of leaves(n)) {
      const leaf = prune({ kind: "view", view }, depth, seen, budget);
      if (leaf) return leaf;
    }
    return null;
  }
  const kept: { node: LayoutNode; size: number }[] = [];
  n.children.forEach((c, i) => {
    const p = prune(c, depth + 1, seen, budget);
    if (p) kept.push({ node: p, size: n.sizes[i] ?? 0 });
  });
  if (kept.length === 0) return null;
  if (kept.length === 1) return kept[0]?.node ?? null;
  return {
    kind: "split",
    dir: n.dir === "col" ? "col" : "row",
    children: kept.map((k) => k.node),
    sizes: fixSizes(
      kept.map((k) => k.size),
      kept.length,
    ),
  };
}

export function normalize(s: LayoutState, vp: Viewport): LayoutState {
  const seen = new Set<ViewId>();
  const root = prune(s.root, 0, seen, MAX_VIEWS[vp]);
  if (!root) return fromPreset(vp === "mobile" ? "chat_only" : "standard", vp);
  const maximized = s.maximized && seen.has(s.maximized) ? s.maximized : null;
  return { version: 1, root, maximized, presetId: s.presetId };
}

function validNode(x: unknown, depth: number): x is LayoutNode {
  if (typeof x !== "object" || x === null || depth > 8) return false;
  const n = x as Record<string, unknown>;
  if (n.kind === "view") return isViewId(n.view);
  return (
    n.kind === "split" &&
    (n.dir === "row" || n.dir === "col") &&
    Array.isArray(n.sizes) &&
    n.sizes.every((z) => typeof z === "number") &&
    Array.isArray(n.children) &&
    n.children.every((c) => validNode(c, depth + 1))
  );
}

/** Parse a stored layout. Anything broken or unknown gives null (caller uses "standard"). */
export function parseLayout(raw: string | null, vp: Viewport): LayoutState | null {
  if (!raw) return null;
  try {
    const x = JSON.parse(raw) as Record<string, unknown>;
    if (x.version !== 1 || !validNode(x.root, 0)) return null;
    const presetId = (PRESET_IDS as readonly string[]).includes(x.presetId as string)
      ? (x.presetId as PresetId)
      : "custom";
    const maximized = isViewId(x.maximized) ? x.maximized : null;
    return normalize({ version: 1, root: x.root, maximized, presetId }, vp);
  } catch {
    return null;
  }
}
