// Every path the panel calls. Nothing else in src/ builds API URLs by hand.
const enc = encodeURIComponent;
const API = "/api/v1";

export type VehicleAction = "stop" | "return_home" | "resume" | "takeover" | "handback";

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
  mission: (missionId: string) => `${API}/missions/${enc(missionId)}`,
  missionLimits: (missionId: string) => `${API}/missions/${enc(missionId)}/limits`,
  missionAutonomy: (missionId: string) => `${API}/missions/${enc(missionId)}/autonomy`,
  missionClearance: (missionId: string) => `${API}/missions/${enc(missionId)}/clearance`,
  missionEnd: (missionId: string) => `${API}/missions/${enc(missionId)}/end`,
  missionCharters: (missionId: string) => `${API}/missions/${enc(missionId)}/charters`,
  missionSectors: (missionId: string) => `${API}/missions/${enc(missionId)}/sectors`,
  missionApprovals: (missionId: string, status?: "pending") =>
    `${API}/missions/${enc(missionId)}/approvals${status ? `?status=${status}` : ""}`,
  missionCommands: (missionId: string, status?: string) =>
    `${API}/missions/${enc(missionId)}/commands${status ? `?status=${enc(status)}` : ""}`,
  approvalDecision: (approvalId: string) => `${API}/approvals/${enc(approvalId)}/decision`,
  missionChat: (missionId: string) => `${API}/missions/${enc(missionId)}/chat`,
  missionChatPage: (
    missionId: string,
    q: { after_seq?: number; before_seq?: number; limit?: number },
  ) => {
    const p = new URLSearchParams();
    if (q.after_seq !== undefined) p.set("after_seq", String(q.after_seq));
    if (q.before_seq !== undefined) p.set("before_seq", String(q.before_seq));
    p.set("limit", String(q.limit ?? 50));
    return `${API}/missions/${enc(missionId)}/chat?${p.toString()}`;
  },
  missionAnnotations: (missionId: string) => `${API}/missions/${enc(missionId)}/annotations`,
  missionFinds: (missionId: string) => `${API}/missions/${enc(missionId)}/finds`,
  detectionCrop: (missionId: string, detectionId: string) =>
    `${API}/missions/${enc(missionId)}/detections/${enc(detectionId)}/crop`,
  missionCandidates: (missionId: string) =>
    `${API}/missions/${enc(missionId)}/candidates?limit=200`,
  candidate: (candidateId: string) => `${API}/candidates/${enc(candidateId)}`,
  candidateDecision: (candidateId: string) => `${API}/candidates/${enc(candidateId)}/decision`,
  candidateGpx: (candidateId: string) => `${API}/candidates/${enc(candidateId)}/export.gpx`,
  detectionView: (detectionId: string) => `${API}/detections/${enc(detectionId)}/view`,
  view: (viewId: string) => `${API}/views/${enc(viewId)}`,
  annotation: (annotationId: string) => `${API}/annotations/${enc(annotationId)}`,
  missionTrack: (missionId: string) => `${API}/missions/${enc(missionId)}/track?max_points=5000`,
  searchProfile: (missionId: string) => `${API}/missions/${enc(missionId)}/search-profile`,
  searchProfileVersions: (missionId: string) =>
    `${API}/missions/${enc(missionId)}/search-profile/versions`,
  liveSocket: () => "/ws/v1/live",
} as const;
