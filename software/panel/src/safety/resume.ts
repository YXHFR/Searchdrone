import type { VehicleLive } from "../api/types";

export type ResumeView = "button" | "home_active" | "onboard_silent" | "none";

/**
 * When Resume may be offered. Only a click on Resume releases a hold; chat text never does.
 * Shown whenever the vehicle is HELD with resume_available, whatever stopped it, except a
 * return-home the operator started (a click would undo it).
 */
export function resumeView(v: VehicleLive | null): ResumeView {
  if (!v) return "none";
  const snap = v.snapshot;
  if (!snap) return v.latch?.action === "stop" ? "onboard_silent" : "none";
  if (snap.held_by === "operator_home") return "home_active";
  if (snap.executor_state === "HELD" && snap.resume_available === true) return "button";
  return "none";
}
