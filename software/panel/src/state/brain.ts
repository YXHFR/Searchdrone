import { useLiveStore } from "./liveStore";

export type BrainState = "ok" | "degraded" | "down" | null;

/**
 * The AI's state for the panel: the live chat.brain_status first (it arrives on every
 * change), else the system status part loaded at start.
 */
export function useBrainState(): BrainState {
  return useLiveStore((s) => {
    if (s.brainState === "degraded" || s.brainState === "down") return s.brainState;
    if (s.brainState === "ready" || s.brainState === "busy") return "ok";
    const part = s.system?.parts.brain?.state;
    return part === "ok" || part === "degraded" || part === "down" ? part : null;
  });
}
