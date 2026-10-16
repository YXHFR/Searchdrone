// Every path the panel calls. Nothing else in src/ builds API URLs by hand.
const enc = encodeURIComponent;
const API = "/api/v1";

export type VehicleAction = "stop" | "return_home" | "resume";

export const paths = {
  health: () => `${API}/health`,
  systemStatus: () => `${API}/system/status`,
  panelConfig: () => `${API}/panel/config`,
  session: () => `${API}/session`,
  logout: () => `${API}/session/logout`,
  vehicleState: (vehicleId: string) => `${API}/vehicles/${enc(vehicleId)}/state`,
  vehicleAction: (vehicleId: string, action: VehicleAction) =>
    `${API}/vehicles/${enc(vehicleId)}/actions/${action}`,
  vehicleActionOutcome: (vehicleId: string, actionId: string) =>
    `${API}/vehicles/${enc(vehicleId)}/actions/${enc(actionId)}`,
  allVehiclesAction: (action: "stop" | "return_home") => `${API}/actions/${action}_all`,
  missionOpen: (vehicleId: string) => `${API}/vehicles/${enc(vehicleId)}/missions/open`,
  missionActive: (vehicleId: string) => `${API}/missions/active?vehicle_id=${enc(vehicleId)}`,
  missionChat: (missionId: string) => `${API}/missions/${enc(missionId)}/chat`,
  liveSocket: () => "/ws/v1/live",
} as const;
