// Typed fixtures for the fake ground station. Positions are around the public ArduPilot
// SITL site (CMAC), never a real home.
import type {
  ActionDispatched,
  ActionOutcome,
  Approval,
  Charter,
  ChatMessage,
  Command,
  Mission,
  MissionCharter,
  PanelConfig,
  Session,
  StateSnapshot,
  SystemStatus,
  Telemetry,
  VehicleLive,
} from "@searchdrone/schemas";

export const VEHICLE = "sim-01";
export const MISSION = "M1";
export const HOME = { lat: -35.363261, lon: 149.16523 };

export const pilotSession: Session = {
  role: "pilot",
  auth_mode: "local",
  display_name: "Demo",
  issuer_ref: "operator:demo",
  expires_at: null,
};

export const observerSession: Session = {
  ...pilotSession,
  role: "observer",
  issuer_ref: "operator:observer",
};

export const panelConfig: PanelConfig = {
  default_vehicle_id: VEHICLE,
  vehicle_kind: "sim",
  map: null,
  video: null,
  limits_default: {
    max_alt_agl_m: 120,
    max_speed_ms: 15,
    max_range_from_pilot_m: 500,
    battery_rtl_pct: 15,
  },
  limits_absolute: {
    max_alt_agl_m: 120,
    max_speed_ms: 15,
    max_range_from_pilot_m: 500,
    min_battery_rtl_pct: 15,
    min_alt_agl_m: 5,
    max_area_km2: 1,
  },
  features: {
    candidates: false,
    follow_me: false,
    voice: false,
    manual_control: false,
    order_buttons: false,
  },
  remote: false,
};

export const mission: Mission = {
  id: MISSION,
  vehicle_id: VEHICLE,
  title: "Uppdrag 2026-10-16 10:00",
  status: "DRAFT",
  state_seq: 1,
  autonomy_mode: null,
  active_charter_id: null,
  ai_enabled: true,
  limits: {
    max_alt_agl_m: 120,
    max_speed_ms: 15,
    max_range_from_pilot_m: 500,
    battery_rtl_pct: 15,
  },
  created_at: "2026-10-16T10:00:00Z",
  created_by: "operator:demo",
  activated_at: null,
  closed_at: null,
  closed_reason: null,
  landed_at: null,
  notes: null,
  paused_by: null,
  returning_cause: null,
};

export function telemetry(p: Partial<Telemetry> = {}): Telemetry {
  return {
    t: new Date().toISOString(),
    time_boot_ms: 1000,
    lat: HOME.lat + 0.0005,
    lon: HOME.lon,
    alt_amsl_m: 624,
    alt_rel_home_m: 40,
    armed: true,
    battery_pct: 64,
    battery_v: 15.1,
    climb_ms: 0,
    dist_to_pilot_m: 55,
    flight_mode: "GUIDED",
    gps_fix: 3,
    gps_sats: 14,
    groundspeed_ms: 5,
    heading_deg: 10,
    home: HOME,
    landed_state: "in_air",
    ...p,
  };
}

export const vehicleLive: VehicleLive = {
  vehicle_id: VEHICLE,
  airborne: false,
  in_flight: false,
  latch: null,
  link: { mavlink: "up", sdlink: "up", onboard_heartbeat_age_ms: 200, rtt_ms: 30 },
  snapshot: null,
  telemetry: null,
  manual: null,
};

export const systemStatus: SystemStatus = {
  version: "0.0.0",
  profile: "sim",
  internet: "unknown",
  operator_present: true,
  parts: { brain: { state: "ok", since: "2026-10-16T10:00:00Z" } },
};

export function dispatched(actionId: string): ActionDispatched {
  return {
    action_id: actionId,
    action_seq: 7,
    result: "dispatched",
    paths: { mavlink: "queued", sdlink: "queued" },
  };
}

export function outcome(actionId: string, p: Partial<ActionOutcome> = {}): ActionOutcome {
  return {
    action: "stop",
    action_id: actionId,
    action_seq: 7,
    by: "operator:demo",
    detail_sv: null,
    mode_confirmed: null,
    mode_requested: "BRAKE",
    paths: { mavlink: "queued", sdlink: "queued" },
    result: "dispatched",
    t: new Date().toISOString(),
    vehicle_id: VEHICLE,
    via: "button",
    ...p,
  };
}

// --- charter, approvals, orders --------------------------------------------------------

/** A square of `half` metres around a point east/north of home (metres). */
export function square(
  half: number,
  east = 0,
  north = 0,
): { type: "Polygon"; coordinates: number[][][] } {
  const dLat = 1 / 111_320;
  const dLon = 1 / (111_320 * Math.cos((HOME.lat * Math.PI) / 180));
  const cx = HOME.lon + east * dLon;
  const cy = HOME.lat + north * dLat;
  const ring = [
    [cx - half * dLon, cy - half * dLat],
    [cx + half * dLon, cy - half * dLat],
    [cx + half * dLon, cy + half * dLat],
    [cx - half * dLon, cy + half * dLat],
    [cx - half * dLon, cy - half * dLat],
  ];
  return { type: "Polygon", coordinates: [ring] };
}

export function charterBody(p: Partial<MissionCharter> = {}): MissionCharter {
  return {
    charter_id: "C1",
    mission_id: MISSION,
    vehicle_id: VEHICLE,
    version: 1,
    autonomy_mode: "WITHIN_CHARTER",
    approved_at: null,
    approved_by: null,
    fence_hash: null,
    area: square(50, 0, 180),
    sectors: [{ sector_id: "S1", name: "S1", order_idx: 0, area: square(50, 0, 180) }],
    home: { lat: HOME.lat, lon: HOME.lon, alt_amsl_m: 584 },
    pilot_position: { lat: HOME.lat, lon: HOME.lon },
    alt_min_agl_m: 15,
    alt_max_agl_m: 60,
    default_alt_agl_m: 40,
    max_speed_ms: 8,
    max_range_from_pilot_m: 500,
    battery_rtl_pct: 15,
    geofence_buffer_m: 20,
    on_ai_silent: "continue_plan",
    ai_silent_timeout_s: 20,
    allowed_skills: ["takeoff", "goto", "survey_sector", "hold", "return_home"],
    notes: "Sök av rutan norr om hem.",
    ...p,
  };
}

export function charter(p: Partial<Charter> = {}, body: Partial<MissionCharter> = {}): Charter {
  return {
    id: "C1",
    mission_id: MISSION,
    version: 1,
    status: "proposed",
    install_status: "none",
    proposed_by: "brain:commander",
    created_at: new Date().toISOString(),
    approval_id: "AP1",
    approved_at: null,
    approved_by: null,
    fence_hash: null,
    fence_verified_at: null,
    body: charterBody({ version: p.version ?? 1, ...body }),
    ...p,
  };
}

export function approval(p: Partial<Approval> = {}): Approval {
  return {
    id: "AP1",
    kind: "charter",
    mission_id: MISSION,
    vehicle_id: VEHICLE,
    subject_id: "C1",
    subject_version: 1,
    status: "pending",
    summary_sv: "Sök av en ruta 100 × 100 m norr om hem.",
    requested_at: new Date().toISOString(),
    requested_by: "brain:commander",
    required_confirmations: [],
    default_autonomy_mode: "WITHIN_CHARTER",
    decided_at: null,
    decided_by: null,
    expires_at: null,
    gate: null,
    reason: null,
    ...p,
  };
}

export function command(p: Partial<Command> = {}): Command {
  return {
    id: "K1",
    mission_id: MISSION,
    vehicle_id: VEHICLE,
    skill: "takeoff",
    params: { alt_m: 40 },
    status: "pending_approval",
    status_reason: null,
    rationale_sv: "Vi lyfter till sökhöjd.",
    issuer_kind: "brain",
    issuer_ref: "brain:commander",
    approval_id: "AP2",
    backend: "scripted",
    based_on: null,
    charter_version: 1,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    expires_at: null,
    in_reply_to: null,
    ...p,
  };
}

export function chatMessage(p: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: "m1",
    mission_id: MISSION,
    seq: 1,
    role: "brain",
    kind: "text",
    text: "Hej",
    author_ref: "brain:commander",
    author_role: null,
    created_at: new Date().toISOString(),
    delivery: "n/a",
    handled_by: null,
    in_reply_to: null,
    parsed_intent: null,
    refs: {},
    ...p,
  };
}

export function snapshot(p: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    executor_state: "PREFLIGHT",
    held_by: null,
    resume_available: false,
    ...p,
  } as StateSnapshot;
}
