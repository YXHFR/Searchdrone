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

export type BrainEngine = "cloud" | "command" | null;

/**
 * Who answers the chat: the cloud model or the command mode (fixed commands). The live
 * chat.brain_status first, else the system status loaded at start; null while the AI is
 * down or has not said.
 */
export function useBrainEngine(): BrainEngine {
  return useLiveStore((s) => {
    if (s.brainState === "down") return null;
    if (s.brainState !== null) return s.brainEngine;
    return s.system?.brain_engine ?? null;
  });
}

export type CloudBudget = "ok" | "warning" | "exhausted" | null;

/** The cloud budget: the live cloud_budget.state first, else the system status. */
export function useCloudBudget(): CloudBudget {
  return useLiveStore((s) => s.cloudBudget ?? s.system?.cloud_budget ?? null);
}

/**
 * The command mode answers the chat (no AI chat): the brain says so, reports degraded,
 * or the cloud budget is spent (the next cloud call is refused).
 */
export function useCommandMode(): boolean {
  const brain = useBrainState();
  const engine = useBrainEngine();
  const budget = useCloudBudget();
  return (
    brain !== "down" && (brain === "degraded" || engine === "command" || budget === "exhausted")
  );
}
