import { useEffect } from "react";
import { useLayoutStore } from "../state/layoutStore";
import type { LayoutNode, Path } from "./layoutModel";
import { ViewFrame } from "./ViewFrame";

function Node({ node, path, vehicleId }: { node: LayoutNode; path: Path; vehicleId: string }) {
  if (node.kind === "view") return <ViewFrame view={node.view} path={path} vehicleId={vehicleId} />;
  return (
    <div className={`split split-${node.dir}`}>
      {node.children.map((c, i) => (
        <div
          // Children of a split never reorder except through normalize; the path is the identity.
          key={[...path, i].join(".")}
          className="split-cell"
          style={{ flexGrow: node.sizes[i] ?? 1 }}
        >
          <Node node={c} path={[...path, i]} vehicleId={vehicleId} />
        </div>
      ))}
    </div>
  );
}

/** Desktop and tablet: the layout tree, or one view in full screen. Esc restores. */
export function LayoutRoot({ vehicleId }: { vehicleId: string }) {
  const layout = useLayoutStore((s) => s.layout);
  useEffect(() => {
    if (!layout.maximized) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") useLayoutStore.getState().restore();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [layout.maximized]);
  return (
    <main className="layout-root" id="main">
      {layout.maximized ? (
        <ViewFrame view={layout.maximized} path={[]} vehicleId={vehicleId} />
      ) : (
        <Node node={layout.root} path={[]} vehicleId={vehicleId} />
      )}
    </main>
  );
}
