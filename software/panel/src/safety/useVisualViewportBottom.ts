import { useEffect, useState } from "react";

/** Distance from the layout bottom to the visual viewport bottom (virtual keyboard), touch only. */
export function computeBottom(): number {
  if (typeof window === "undefined") return 0;
  const coarse =
    typeof window.matchMedia === "function" && window.matchMedia("(pointer: coarse)").matches;
  const vv = window.visualViewport;
  if (!coarse || !vv) return 0;
  return Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
}

export function useVisualViewportBottom(): number {
  const [bottom, setBottom] = useState(computeBottom);
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const update = (): void => setBottom(computeBottom());
    vv.addEventListener("resize", update);
    vv.addEventListener("scroll", update);
    return () => {
      vv.removeEventListener("resize", update);
      vv.removeEventListener("scroll", update);
    };
  }, []);
  return bottom;
}
