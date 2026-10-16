// Typed fixtures for the fake ground station. Positions are around the public ArduPilot
// SITL site (CMAC), never a real home.
import type {
  ActionDispatched,
  ActionOutcome,
  Mission,
  PanelConfig,
  Session,
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
    landed_state: "IN_AIR",
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
