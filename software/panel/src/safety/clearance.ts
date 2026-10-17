import type { Charter, Telemetry, VehicleLive } from "../api/types";

/** Executor states that mean "on the ground" (the onboard list). */
export const ON_GROUND = new Set(["IDLE", "PREFLIGHT", "LANDED"]);

/**
 * When "Fritt runt drönaren" may be offered: a pilot, an approved and installed charter, and
 * the vehicle on the ground by every signal (executor state, not airborne, not armed).
 */
export function clearanceAvailable(
  isPilot: boolean,
  charter: Charter | null,
  vehicle: VehicleLive | null,
  telemetry: Telemetry | null,
): boolean {
  if (
    !isPilot ||
    !charter ||
    charter.status !== "approved" ||
    charter.install_status !== "installed"
  )
    return false;
  const state = vehicle?.snapshot?.executor_state;
  if (!state || !ON_GROUND.has(state)) return false;
  if (vehicle?.airborne !== false) return false;
  return telemetry !== null && telemetry.armed === false;
}
